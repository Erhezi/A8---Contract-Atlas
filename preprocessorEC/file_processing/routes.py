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
# Import user-specific session helpers
from ..common.session import (
    store_file_info, get_file_info, # Assuming these exist or will be added
    store_column_mapping, get_column_mapping,
    store_validated_data, get_validated_data,
    store_error_file_path, get_error_file_path, clear_error_file_path, # Added clear
    store_session_data, get_session_data # Keep generic ones if needed elsewhere
)

# Create the blueprint
file_bp = Blueprint('file_processing', __name__, 
                   url_prefix='/file-processing',
                   template_folder='templates')

@file_bp.route('/upload-file', methods=['POST'])
@login_required
def upload_file():
    """Upload a file and return column headers"""
    user_id = current_user.id # Get user_id
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
            
            # Store file info in session using helper
            file_info_dict = {
                'name': filename,
                'saved_name': saved_name,
                'path': file_path,
                'type': file_ext
            }
            store_file_info(user_id, file_info_dict) # Use helper
            
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
    user_id = current_user.id # Get user_id
    file_info = get_file_info(user_id) # Use helper
    if not file_info:
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
    
    # Store mapping in session using helper
    store_column_mapping(user_id, column_mapping) # Use helper
    
    return jsonify({
        'success': True,
        'message': 'Column mapping saved successfully'
    })

@file_bp.route('/validate-file', methods=['POST'])
@login_required
def validate_file_route():
    """Validate the uploaded file columns against required fields"""
    user_id = current_user.id # Get user_id
    file_info = get_file_info(user_id) # Use helper
    if not file_info:
        return jsonify({'success': False, 'message': 'No file uploaded. Please upload a file first.'})
    
    column_mapping = get_column_mapping(user_id) # Use helper
    if not column_mapping:
         # If mapping not in session (e.g., direct navigation), try getting from form
        form_mapping = {}
        for field, column in request.form.items():
            if column and field not in ['duplicate_mode']: # Exclude non-mapping fields
                form_mapping[field] = column
        if form_mapping:
            column_mapping = form_mapping
            store_column_mapping(user_id, column_mapping) # Store it now
        else:
            return jsonify({'success': False, 'message': 'Column mapping not found. Please map columns first.'})

    # Load the file
    # Use file_info obtained from session helper
    file_path = os.path.join(current_app.static_folder, 'uploads', file_info['saved_name'])
    
    try:
        # Read file
        if file_info['type'] == 'csv':
            df = pd.read_csv(file_path, dtype=str)
        else:
            df = pd.read_excel(file_path, dtype=str)

        # Get duplicate checking mode from form or session (using generic helper for this example)
        duplicate_mode = request.form.get('duplicate_mode', get_session_data(f'duplicate_check_mode_{user_id}', 'default'))
        store_session_data(f'duplicate_check_mode_{user_id}', duplicate_mode) # Store user-specifically
        
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
            error_file_path = save_error_file(error_df, user_id, file_info['name'])
            store_error_file_path(user_id, error_file_path) # Use helper
            
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
        clear_error_file_path(user_id) # Use helper to clear any previous error file path
        total_rows = len(valid_df)
        store_validated_data(user_id, valid_df.to_dict('records')) # Use helper, store as records (list of dicts)
        
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
    user_id = current_user.id # Get user_id
    try:
        # Get the filename from the session using helper
        file_path = get_error_file_path(user_id) # Use helper
        if not file_path:
            flash('No error file available for download', 'danger')
            return redirect(url_for('common.dashboard')) # Assuming common.dashboard exists

        print(f"Attempting to download file for user {user_id}: {file_path}")

        # Skip strict user verification if we're in development
        if not current_app.config.get('DEVELOPMENT', False):
            # Verify the file belongs to the current user - make this check more lenient
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
        
        # Try to get the original filename from session using helper
        file_info = get_file_info(user_id) # Use helper
        if file_info and 'name' in file_info:
            # Get original filename without extension
            original_filename = file_info['name']
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