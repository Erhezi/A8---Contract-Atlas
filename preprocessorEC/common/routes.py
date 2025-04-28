# common/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from flask import current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import pandas as pd

# Create the blueprint
common_bp = Blueprint('common', __name__, 
                     url_prefix='/common',
                     template_folder='templates')

@common_bp.route('/download-template')
def download_template():
    """Download the upload template file"""
    try:
        # Use the same path structure as your UOM file in utils.py
        template_path = os.path.join(current_app.root_path, 'data', 'upload_template.xlsx')
        # Check if file exists
        if not os.path.exists(template_path):
            print(f"Template file not found at: {template_path}")
            flash("Template file not found. Please contact the administrator.", "error")
            return redirect(url_for('common.dashboard'))
        
        # Log that we found the file
        print(f"Serving template from: {template_path}")
        
        # Return the file as a download attachment
        return send_file(
            template_path,
            as_attachment=True,
            download_name='contract_price_template.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        # Log the exception
        print(f"Error downloading template: {str(e)}")
        flash(f"Error downloading template: {str(e)}", "error")
        return redirect(url_for('common.dashboard'))

@common_bp.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('common.index'))
    return redirect(url_for('auth.landing'))

@common_bp.route('/index')
@login_required
def index():
    return render_template('index.html')

@common_bp.route('/dashboard')
@login_required
def dashboard():
    # Check if we need to restart the process
    if request.args.get('restart'):
        session['current_step_id'] = 1
        session['completed_steps'] = []
        flash('Starting a new process', 'info')
    
    # Get current step and all steps
    current_step = current_app.get_current_step()
    all_steps = current_app.get_all_steps()
    
    return render_template('dashboard.html', current_step=current_step, steps=all_steps)

@common_bp.route('/download-error-file')
@login_required
def download_error_file():
    """Download the error report file"""
    try:
        # Get the error file path from session or use a default location
        error_file_path = session.get('error_file_path')
        
        if not error_file_path or not os.path.exists(error_file_path):
            # If no specific error file exists, use a default
            error_file_path = os.path.join(current_app.root_path, 'data', 'error_report.xlsx')
            
            if not os.path.exists(error_file_path):
                flash("Error report file not found. No errors may have been generated.", "warning")
                return redirect(url_for('common.dashboard'))
        
        # Return the file as a download attachment
        return send_file(
            error_file_path,
            as_attachment=True,
            download_name='error_report.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        # Log the exception
        print(f"Error downloading error file: {str(e)}")
        flash(f"Error downloading error report: {str(e)}", "error")
        return redirect(url_for('common.dashboard'))
    
@common_bp.route('/upload-file', methods=['POST'])
@login_required
def upload_file():
    """Handle file uploads for validation"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part in the request'}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
            
        if file and allowed_file(file.filename):
            # Create a secure filename
            filename = secure_filename(file.filename)
            
            # Save the file to a temporary location
            upload_folder = os.path.join(current_app.root_path, 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)
            
            # Store the file path in the session
            session['uploaded_file_path'] = file_path
            
            # Return success response
            return jsonify({
                'success': True, 
                'message': 'File uploaded successfully', 
                'filename': filename
            })
        else:
            return jsonify({'success': False, 'error': 'File type not allowed'}), 400
            
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return jsonify({'success': False, 'error': f'Error uploading file: {str(e)}'}), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'xlsx', 'xls', 'csv'}

@common_bp.route('/goto-step/<int:step_id>', methods=['GET'])
@login_required
def goto_step(step_id):
    """Navigate to a specific step"""
    # Check if the requested step is accessible
    is_allowed, current_step_id, message = current_app.validate_step_progress(step_id)
    
    if is_allowed:
        # Update the current step in the session without affecting completion status
        session['current_step_id'] = step_id
        return redirect(url_for('common.dashboard'))
    else:
        flash(message, 'warning')
        return redirect(url_for('common.dashboard'))

@common_bp.route('/step/<int:step_id>')
@login_required
def step_view(step_id):
    # Validate step access
    is_allowed, current_id, message = current_app.validate_step_progress(step_id)
    
    if not is_allowed:
        flash(message, 'danger')
        return redirect(url_for('common.dashboard'))
    
    step_id = max(1, min(step_id, 8))  # Clamp between 1 and 8
    session['viewed_step_id'] = step_id  # Track which step is being viewed
    
    return render_template('dashboard.html')

@common_bp.route('/process-step/<int:step_id>', methods=['POST'])
@login_required
def process_step(step_id):
    # Validate if user can process this step
    is_allowed, current_id, message = current_app.validate_step_progress(step_id)
    
    if not is_allowed:
        flash(message, 'danger')
        return redirect(url_for('common.dashboard'))
    
    # Process step based on its ID
    success = False
    error_msg = None
    
    try:
        from ..common.session import update_true_duplicates_count, get_contracts_with_true_duplicates, get_deduped_results
        
        if step_id == 1:
            # Step 1: File validation
            # Check if we have validated data in the session
            if 'validated_data' not in session or 'column_mapping' not in session:
                raise ValueError("File validation incomplete. Please upload and validate a file first.")
            
            # Get validated data from session
            validated_data = session.get('validated_data')
            
            # If we have validated data, then the file has been processed successfully
            if validated_data:
                success = True
                flash("File validated successfully!", "success")
            else:
                raise ValueError("File validation failed or no validated data available.")
            
        elif step_id == 2:
            # Step 2: Duplication Overview - Real business logic
            if 'validated_data' not in session:
                raise ValueError("No validated data available. Please complete Step 1 first.")
            
            # Convert validated data from session back to dataframe
            validated_df = pd.DataFrame.from_dict(session['validated_data'])
            
            # Check if we have contract data in session
            if 'contract_duplicates' not in session:
                raise ValueError("No contract data available. Please process duplicates first.")
            
            # Get duplication results from session
            if 'item_comparison_results' not in session:
                raise ValueError("Item comparison not completed. Please complete the comparison process first.")
            
            # Calculate true duplicates (total duplicates minus false positives)
            true_duplicates = update_true_duplicates_count()
            
            # Find contracts that actually have true duplicates
            contracts_with_true_duplicates = get_contracts_with_true_duplicates()
            print(f"Contracts with true duplicates: {contracts_with_true_duplicates}")
            
            # Count unique contracts with true duplicates
            total_contracts_with_duplicates = len(contracts_with_true_duplicates)
            
            # Set success flag for this step
            success = True
            
            # Handle flow based on true duplicates count
            if true_duplicates > 0:
                flash(f"Found {total_contracts_with_duplicates} contracts with {true_duplicates} true duplicate items.", "success")
            else:
                # No true duplicates found - auto-complete step 3
                flash("No true duplicates found after false positive review.", "info")
                
                # Mark step 3 as complete automatically
                current_app.mark_step_complete(3)
                
                # Update next step to be step 4
                session['current_step_id'] = 4
                session.modified = True
                
                flash("Step 3 (Duplication Resolution) automatically completed as no true duplicates were found.", "info")
        
        elif step_id == 3:
            # Step 3: Duplication Resolution
            
            # Check if item comparison was completed
            if 'item_comparison_results' not in session:
                raise ValueError("Item comparison not completed. Please complete Step 2 first.")

            comparison_results = session['item_comparison_results']
            if not comparison_results or 'summary' not in comparison_results:
                raise ValueError("Invalid comparison results. Please rerun the item comparison.")

            deduplication_results = get_deduped_results()
            if not deduplication_results:
                flash("No deduplication results available. Please apply a deduplication policy before completing this step.", "warning")
                return redirect(url_for('common.step_view', step_id=step_id))

            if deduplication_results:
                resolution_strategy = deduplication_results.get('policy', {}).get('type', 'unkonown')
                print(deduplication_results.keys()) #debugging
                success = True
                flash(f"Deduplication results processed successfully using [{resolution_strategy}] policy!", "success")
            else:
                flash("No deduplication results found. Please check the deduplication process.", "warning")

        elif step_id == 4:
            # Item master matching logic
            matching_method = request.form.get('matching_method')
            if not matching_method:
                raise ValueError("No matching method selected")
                
            # Process matching (dummy implementation)
            import time
            time.sleep(1)  # Simulate processing time
            success = True
            
        elif step_id == 5:
            # Change simulation logic
            # Process simulation approval
            import time
            time.sleep(1)  # Simulate processing time
            success = True
            
        elif step_id == 6:
            # Export changes logic
            export_format = request.form.get('export_format')
            if not export_format:
                raise ValueError("No export format selected")
                
            # Process export (dummy implementation)
            import time
            time.sleep(1)  # Simulate processing time
            success = True
            
        elif step_id == 7:
            # Synchronization inspection logic
            # Process synchronization verification
            import time
            time.sleep(1)  # Simulate processing time
            success = True
            
        elif step_id == 8:
            # Completion verification - no processing needed
            success = True
            
        # If successful, mark step as complete and advance
        if success:
            next_step = current_app.mark_step_complete(step_id)
            flash(f"Step {step_id} completed successfully!", "success")
            return redirect(url_for('common.dashboard'))
            
    except ValueError as e:
        error_msg = str(e)
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        
    if error_msg:
        flash(error_msg, "danger")
        
    return redirect(url_for('common.dashboard'))