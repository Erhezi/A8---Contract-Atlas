# File processing routes
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from flask import current_app, send_from_directory
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import pandas as pd
import os
import time
import re
from ..common.db import get_db_connection, create_temp_table
from ..common.utils import validate_file, save_error_file

# Create the blueprint
file_bp = Blueprint('file_processing', __name__, 
                   url_prefix='/file-processing',
                   template_folder='templates')

@file_bp.route('/upload-file', methods=['POST'])
@login_required
def upload_file():
    """Upload a file and return column headers"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file:
        # Ensure upload directory exists
        upload_dir = os.path.join(current_app.static_folder, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        saved_name = f"{int(time.time())}_{filename}"
        file_path = os.path.join(upload_dir, saved_name)
        file.save(file_path)
        
        # Determine file type
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        # Read headers
        try:
            if file_ext == 'csv':
                df = pd.read_csv(file_path, nrows=0)
            else:
                df = pd.read_excel(file_path, nrows=0)
                
            headers = df.columns.tolist()
            
            # Store file info in session
            session['file_info'] = {
                'name': filename,
                'saved_name': saved_name,
                'path': file_path,
                'type': file_ext
            }
            
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully',
                'filename': filename,
                'headers': headers
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error processing file: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'File upload failed'})

@file_bp.route('/map-columns', methods=['POST'])
@login_required
def map_columns():
    """Save column mapping to session"""
    if 'file_info' not in session:
        return jsonify({'success': False, 'message': 'No file uploaded. Please upload a file first.'})
    
    # Get column mapping from POST data
    column_mapping = {}
    for field, column in request.form.items():
        print(f"Field: {field}, Column: {column}")  # Debugging line
        if column:  # Only include fields that are mapped to a column
            column_mapping[field] = column
    
    # Validate required fields
    required_fields = ['Mfg Part Num', 'Vendor Part Num', 'Description', 
                       'Contract Price', 'UOM', 'QOE', 
                       'Effective Date', 'Expiration Date', 'Contract Number', 'ERP Vendor ID',
                       'Source Contract Type']
    
    missing_fields = [field for field in required_fields if field not in column_mapping and field != 'Buyer Part Num']
    
    if missing_fields:
        return jsonify({
            'success': False, 
            'message': f'Missing required field mapping: {", ".join(missing_fields)}'
        })
    
    # Store mapping in session
    session['column_mapping'] = column_mapping
    
    return jsonify({
        'success': True,
        'message': 'Column mapping saved successfully'
    })

@file_bp.route('/validate-file', methods=['POST'])
@login_required
def validate_file_route():
    """Validate the uploaded file columns against required fields"""
    if 'file_info' not in session:
        return jsonify({'success': False, 'message': 'No file uploaded. Please upload a file first.'})
    
    file_info = session['file_info']
    column_mapping = {}
    
    # Get column mapping from POST data
    for field, column in request.form.items():
        if column:  # Only include fields that are mapped to a column
            column_mapping[field] = column
    
    # Load the file
    file_path = os.path.join(current_app.static_folder, 'uploads', file_info['saved_name'])
    
    try:
        # Read file
        if file_info['type'] == 'csv':
            df = pd.read_csv(file_path, dtype=str)
        else:
            df = pd.read_excel(file_path, dtype=str)

        # Get duplicate checking mode from session or form
        duplicate_mode = request.form.get('duplicate_mode', session.get('duplicate_check_mode', 'default'))
        session['duplicate_check_mode'] = duplicate_mode
        
        # Validate the file
        valid_df, error_df, has_errors = validate_file(df, column_mapping, duplicate_mode)
        
        if isinstance(has_errors, str):
            # This means there was an error in the validation process
            return jsonify({'success': False, 'message': has_errors})
        
        # Save the full result dataframe for user download
        if has_errors:
           # Calculate statistics
            total_rows = int(len(error_df))  # Convert to Python int
            error_rows = int(error_df['Has Error'].sum())  # Convert to Python int
            duplicate_rows = int(len(error_df[(error_df['Warning-Potential Duplicates'] != '')]))  # Convert to Python int
            
            # Save the full result dataframe for user download
            error_file = save_error_file(error_df, current_user.id, file_info['name'])
            session['error_file'] = error_file
            
            # Convert error dataframe to HTML table for display
            # First, get only the rows with errors
            error_rows_df = error_df[error_df['Has Error']].copy()

            # Move the File Row column to the front for better visibility
            cols = ['File Row'] + [col for col in error_rows_df.columns if col != 'File Row']
            error_rows_df = error_rows_df[cols]

            # Generate HTML table properly by stripping out pandas default attributes
            html_content = error_rows_df.to_html(classes="", index=False, header=True)
            # Replace the entire table tag, not just part of it
            html_content = re.sub(r'<table[^>]*>', '', html_content)
            html_content = html_content.replace('</table>', '')
            error_table = f'<div class="table-responsive-container"><table id="error_table" class="table table-striped table-bordered">{html_content}</table></div>'
            
            return jsonify({
                'success': False, 
                'message': 'Validation failed. See the errors below. Review and make changes accordingly, then try again.',
                'error_table': error_table,
                'stats': {
                    'total_rows': total_rows,
                    'error_rows': error_rows,
                    'duplicate_rows': duplicate_rows
                }
            })
                
        # No errors, store validated dataframe in session
        # Since we can't store the dataframe directly in session, convert to dict
        session.pop('error_file', None)  # Clear any previous error file
        total_rows = len(valid_df)
        session['validated_data'] = valid_df.to_dict()
        
        return jsonify({
            'success': True,
            'message': 'File validation passed successfully!',
            'stats': {
                'total_rows': total_rows,
                'error_rows': 0,
                'duplicate_rows': 0
            },
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error validating file: {str(e)}'})

@file_bp.route('/download-error-file')
@login_required
def download_error_file():
    """Download the error file"""
    try:
        # Get the filename from the session
        if 'error_file' not in session:
            # Try alternative session key
            if 'error_file_path' in session:
                file_path = session['error_file_path']
            else:
                flash('No error file available for download', 'danger')
                return redirect(url_for('common.dashboard'))
        else:
            file_path = session['error_file']
        
        print(f"Attempting to download file: {file_path}")
        
        # Skip strict user verification if we're in development
        if not current_app.config.get('DEVELOPMENT', False):
            # Verify the file belongs to the current user - make this check more lenient
            user_id = str(current_user.id) if not isinstance(current_user.id, str) else current_user.id
            expected_user_dir = f'user_{user_id}'
            
            if expected_user_dir not in file_path and 'user_admin' not in file_path:
                flash('Access denied: You do not have permission to download this file', 'danger')
                return redirect(url_for('common.dashboard'))
        
        # Ensure the file exists
        if not os.path.exists(file_path):
            # Try looking in the hardcoded path you mentioned
            alt_path = os.path.join(current_app.root_path, 'temp_files', 'user_admin', os.path.basename(file_path))
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                flash('Error file not found. It may have been deleted.', 'warning')
                return redirect(url_for('common.dashboard'))
        
        print(f"File exists, sending: {file_path}")
        
        # Generate dynamic filename with original filename and timestamp
        timestamp = int(time.time())
        original_filename = "unknown_file"
        
        # Try to get the original filename from session
        if 'file_info' in session and 'name' in session['file_info']:
            # Get original filename without extension
            original_filename = session['file_info']['name']
            if '.' in original_filename:
                original_filename = original_filename.rsplit('.', 1)[0]
        
        # Create the dynamic download name
        download_filename = f"error_report_{original_filename}_{timestamp}.xlsx"
        
        # Return the file as a download attachment
        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        import traceback
        print(f"Error downloading error file: {str(e)}")
        print(traceback.format_exc())
        flash(f"Error downloading error report: {str(e)}", "error")
        return redirect(url_for('common.dashboard'))