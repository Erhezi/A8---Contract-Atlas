# common/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from flask import current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import pandas as pd

# Import specific session helpers
from .session import (
    store_current_step, get_current_step_from_session,
    store_completed_steps, get_completed_steps,
    get_error_file_path, store_error_file_path, # Assuming you store it somewhere
    store_uploaded_file_path, get_uploaded_file_path,
    get_validated_data, get_column_mapping,
    get_contract_duplicates, get_comparison_results,
    update_true_duplicates_count, get_contracts_with_true_duplicates,
    get_deduped_results, store_deduplication_results,
    get_infor_cl_matches,
    get_infor_im_matches,
    get_uom_qoe_validation
)

# Create the blueprint
common_bp = Blueprint('common', __name__,
                     url_prefix='/common',
                     template_folder='templates')

@common_bp.route('/download-template')
def download_template():
    """Download the upload template file"""
    try:
        template_path = os.path.join(current_app.root_path, 'data', 'upload_template.xlsx')
        if not os.path.exists(template_path):
            current_app.logger.error(f"Template file not found at: {template_path}")
            flash("Template file not found. Please contact the administrator.", "error")
            return redirect(url_for('common.dashboard'))

        current_app.logger.info(f"Serving template from: {template_path}")
        return send_file(
            template_path,
            as_attachment=True,
            download_name='contract_price_template.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        current_app.logger.exception(f"Error downloading template: {e}")
        flash(f"Error downloading template: {str(e)}", "error")
        return redirect(url_for('common.dashboard'))

@common_bp.route('/')
def home():
    if current_user.is_authenticated:
        # Redirect authenticated users to the dashboard
        return redirect(url_for('common.dashboard')) # Changed from index to dashboard
    return redirect(url_for('auth.landing'))

@common_bp.route('/index')
@login_required
def index():
    # This route might be redundant if dashboard is the main view
    return render_template('index.html')

@common_bp.route('/dashboard')
@login_required
def dashboard():
    user_id = current_user.id
    # Check if we need to restart the process
    if request.args.get('restart'):
        store_current_step(user_id, 1) # Use helper
        store_completed_steps(user_id, []) # Use helper
        # Optionally clear other user-specific data here
        flash('Starting a new process', 'info')
        # Redirect to remove 'restart' from URL args
        return redirect(url_for('common.dashboard'))

    # Get current step and completed steps for the user from session
    current_step_id = get_current_step_from_session(user_id) # Use helper
    completed_steps = get_completed_steps(user_id) # Use helper

    # Get step definitions using the method attached to the app
    all_steps = current_app.get_all_steps() # Use the method from StepManager

    # Find the current step object based on ID
    current_step_obj = next((step for step in all_steps if step['id'] == current_step_id), None)

    # Handle case where current_step_id is invalid or all_steps is empty
    if not current_step_obj:
        if all_steps: # Check if all_steps is not empty
            current_step_obj = all_steps[0] # Default to first step
            store_current_step(user_id, current_step_obj['id']) # Correct session
            flash("Invalid current step detected, resetting to Step 1.", "warning")
        else:
            # Handle the critical error where no steps are defined
            current_app.logger.error("No workflow steps found in the application.")
            flash("Critical error: Workflow steps are not configured. Please contact the administrator.", "danger")
            # Render a minimal error page or redirect to a safe location
            return render_template('error.html', error_message="Workflow steps not configured."), 500

    return render_template('dashboard.html',
                           current_step=current_step_obj,
                           steps=all_steps,
                           completed_steps=completed_steps) # Pass completed steps to template

@common_bp.route('/download-error-file')
@login_required
def download_error_file():
    """Download the error report file for the current user"""
    user_id = current_user.id
    try:
        error_file_path = get_error_file_path(user_id) # Use helper

        if not error_file_path or not os.path.exists(error_file_path):
            # Fallback or error message if no specific error file exists for the user
            flash("Error report file not found for this process.", "warning")
            return redirect(url_for('common.dashboard'))

        # Return the file as a download attachment
        return send_file(
            error_file_path,
            as_attachment=True,
            download_name=f'error_report_{user_id}.xlsx', # User-specific download name
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        current_app.logger.exception(f"Error downloading error file for user {user_id}: {e}")
        flash(f"Error downloading error report: {str(e)}", "error")
        return redirect(url_for('common.dashboard'))

@common_bp.route('/upload-file', methods=['POST'])
@login_required
def upload_file():
    """Handle file uploads for validation for the current user"""
    user_id = current_user.id
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            # Create a secure filename, potentially adding user_id for uniqueness
            filename_base, file_ext = os.path.splitext(file.filename)
            secure_base = secure_filename(f"{filename_base}_{user_id}")
            filename = f"{secure_base}{file_ext}"

            # Save the file to a temporary location (consider user-specific subfolders)
            upload_folder = os.path.join(current_app.config['UPLOAD_FOLDER']) # Use config
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            # Store the file path in the session using helper
            store_uploaded_file_path(user_id, file_path)

            return jsonify({
                'success': True,
                'message': 'File uploaded successfully',
                'filename': file.filename # Return original filename for display
            })
        else:
            return jsonify({'success': False, 'error': 'File type not allowed'}), 400

    except Exception as e:
        current_app.logger.exception(f"Error uploading file for user {user_id}: {e}")
        return jsonify({'success': False, 'error': f'Error uploading file: {str(e)}'}), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', {'xlsx', 'xls', 'csv'})
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@common_bp.route('/goto-step/<int:step_id>', methods=['GET'])
@login_required
def goto_step(step_id):
    """Navigate to a specific step if allowed, resetting later steps if going backward"""
    user_id = current_user.id
    completed_steps = get_completed_steps(user_id)
    current_step_id = get_current_step_from_session(user_id)
    
    # Check if the step is accessible (must be a step user has completed or current step)
    if step_id in completed_steps or step_id == current_step_id:
        # Going back to an earlier step - reset all steps after this one
        if step_id < current_step_id:
            # Remove all completed steps that are greater than the step we're going back to
            completed_steps = [step for step in completed_steps if step <= step_id]
            store_completed_steps(user_id, completed_steps)
            flash(f"Navigating back to Step {step_id}. You'll need to re-complete subsequent steps.", 'info')
        
        # Update current step to the requested step
        store_current_step(user_id, step_id)
        return redirect(url_for('common.dashboard'))
    else:
        flash(f"Step {step_id} is not yet accessible.", 'warning')
        return redirect(url_for('common.dashboard'))
    

@common_bp.route('/step/<int:step_id>')
@login_required
def step_view(step_id):
    """Render the dashboard focused on a specific step (validation needed)"""
    # This route seems similar to /dashboard, maybe consolidate?
    # Or ensure validation is robust here too.
    user_id = current_user.id
    completed_steps = get_completed_steps(user_id)
    current_step_id = get_current_step_from_session(user_id)

    if step_id <= current_step_id or step_id in completed_steps:
        # Render dashboard, potentially highlighting the requested step_id
        all_steps = current_app.config.get('APP_STEPS', [])
        step_obj = next((step for step in all_steps if step['id'] == step_id), None)
        if not step_obj:
             flash(f"Invalid step ID: {step_id}", 'danger')
             return redirect(url_for('common.dashboard'))

        # Render dashboard, passing step_id to potentially highlight it
        return render_template('dashboard.html',
                               current_step=step_obj, # Show the requested step
                               steps=all_steps,
                               completed_steps=completed_steps)
    else:
        flash(f"Step {step_id} is not yet accessible.", 'warning')
        return redirect(url_for('common.dashboard'))


@common_bp.route('/process-step/<int:step_id>', methods=['POST'])
@login_required
def process_step(step_id):
    user_id = current_user.id
    # Validate if user can process this step (should be the current step)
    current_step_id = get_current_step_from_session(user_id)
    if step_id != current_step_id:
        flash(f"Cannot process Step {step_id}. Current step is {current_step_id}.", 'warning')
        return redirect(url_for('common.dashboard'))

    success = False
    error_msg = None
    next_step_id = step_id + 1 # Default next step

    try:
        # --- Step Processing Logic ---
        if step_id == 1:
            # Step 1: File validation completion check
            validated_data = get_validated_data(user_id) # Use helper
            column_mapping = get_column_mapping(user_id) # Use helper
            if not validated_data or not column_mapping:
                raise ValueError("File validation incomplete. Please upload and validate a file first.")
            success = True
            flash("File validated successfully!", "success")

        elif step_id == 2:
            # Step 2: Duplication Overview completion check
            validated_data = get_validated_data(user_id)
            if not validated_data:
                raise ValueError("No validated data available. Please complete Step 1 first.")

            contract_duplicates = get_contract_duplicates(user_id) # Use helper
            if not contract_duplicates:
                raise ValueError("No contract data available. Please process duplicates first.")

            comparison_results = get_comparison_results(user_id) # Use helper
            if not comparison_results:
                raise ValueError("Item comparison not completed. Please complete the comparison process first.")

            true_duplicates = update_true_duplicates_count(user_id) # Use helper
            contracts_with_true_duplicates = get_contracts_with_true_duplicates(user_id) # Use helper
            total_contracts_with_duplicates = len(contracts_with_true_duplicates)

            success = True
            if true_duplicates > 0:
                flash(f"Found {total_contracts_with_duplicates} contracts with {true_duplicates} true duplicate items.", "success")
            else:
                # No true duplicates found - auto-complete step 3
                flash("No true duplicates found after false positive review.", "info")
                # Mark step 3 as complete automatically
                completed_steps = get_completed_steps(user_id)
                if 3 not in completed_steps:
                    completed_steps.append(3)
                    store_completed_steps(user_id, completed_steps) # Save updated list
                # need to store default info to deduplications_results_{user_id}
                to_upload_count = len(validated_data)
                step3_results = {
                    'policy': {"custom_directions": [],
                               "custom_fields": [],
                               "type": "no_duplicates"},
                    'stacked_data': [],
                    'summary': {
                        "duplicates_removed": 0,
                        "kept_ccx": 0,
                        "kept_uploaded": to_upload_count,
                        "total_items": to_upload_count,
                        "unique_duplicates": to_upload_count}
                }
                store_deduplication_results(user_id, step3_results) # Use helper
                next_step_id = 4 # Skip to step 4
                flash("Step 3 (Duplication Resolution) automatically completed.", "info")

        elif step_id == 3:
            # Step 3: Duplication Resolution completion check
            comparison_results = get_comparison_results(user_id)
            if not comparison_results:
                raise ValueError("Item comparison not completed. Please complete Step 2 first.")

            deduplication_results = get_deduped_results(user_id) # Use helper
            if not deduplication_results:
                flash("No deduplication results available. Please apply a deduplication policy before completing this step.", "warning")
                # Don't redirect here, let the user stay on step 3 to apply policy
                return redirect(url_for('common.dashboard')) # Or step_view

            resolution_strategy = deduplication_results.get('policy', {}).get('type', 'unknown')
            success = True
            flash(f"Deduplication results processed successfully using [{resolution_strategy}] policy!", "success")

        elif step_id == 4:
            # Step 4: Item Master Matching completion check
            # check if validated data exists
            validated_data = get_validated_data(user_id) # Use helper
            if not validated_data:
                raise ValueError("No validated data available. Please complete Step 1 first.")
            
            # Check if matching results exist in session
            infor_cl_matches = get_infor_cl_matches(user_id) # Use helper
            if not infor_cl_matches:
                 raise ValueError("Infor Contract Line matching not completed. Please run the matching process first.")
            
            infor_im_matches = get_infor_im_matches(user_id) # Use helper
            if not infor_im_matches:
                 raise ValueError("Infor Item Master matching not completed. Please run the matching process first.")
            
            uom_qoe_validation = get_uom_qoe_validation(user_id)
            if not uom_qoe_validation:  
                 raise ValueError("UOM and QOE validation not completed. Please run the validation process first.")
            
            success = True
            flash("Step 4 (Item Master Matching) completed successfully.", "success")

        elif step_id == 5:
            # Step 5: Change Simulation completion check
            # Add checks relevant to step 5
            import time; time.sleep(0.1) # Simulate check
            success = True
            flash("Step 5 completed (Placeholder).", "success")

        elif step_id == 6:
            # Step 6: Export Changes completion check
            # Add checks relevant to step 6 (e.g., check if export file path exists)
            export_format = request.form.get('export_format') # Check if form was submitted if needed
            if not export_format: # Example check
                 raise ValueError("No export format selected for Step 6 completion.")
            import time; time.sleep(0.1) # Simulate check
            success = True
            flash("Step 6 completed (Placeholder).", "success")

        elif step_id == 7:
            # Step 7: Synchronization Inspection completion check
            # Add checks relevant to step 7
            import time; time.sleep(0.1) # Simulate check
            success = True
            flash("Step 7 completed (Placeholder).", "success")

        elif step_id == 8:
            # Step 8: Completion - Always successful if reached
            success = True
            flash("Process Completed!", "success")
            next_step_id = 8 # Stay on step 8 or redirect elsewhere

        # --- End Step Processing Logic ---

        # If successful, mark step as complete and advance
        if success:
            completed_steps = get_completed_steps(user_id)
            if step_id not in completed_steps:
                completed_steps.append(step_id)
                store_completed_steps(user_id, completed_steps) # Save updated list

            # Update current step only if moving forward
            if next_step_id > step_id:
                 store_current_step(user_id, next_step_id)

            return redirect(url_for('common.dashboard'))

    except ValueError as e:
        error_msg = str(e)
    except Exception as e:
        current_app.logger.exception(f"Error processing step {step_id} for user {user_id}: {e}")
        error_msg = f"An unexpected error occurred: {str(e)}"

    if error_msg:
        flash(error_msg, "danger")

    # Redirect back to dashboard (which shows the current step) on error
    return redirect(url_for('common.dashboard'))


# --- DEBUG ROUTE ---
# (Keep as is, it already shows all session content)
@common_bp.route('/debug/show-session', methods=['GET'])
@login_required
def show_session_data():
    # ... (implementation remains the same) ...
    if not current_app.debug:
         return jsonify({"error": "This endpoint is only available in debug mode."}), 403
    try:
        session_dict = dict(session)
        session_keys = list(session_dict.keys())
        return jsonify({
            "session_keys": session_keys,
            "session_content": session_dict,
        })
    except Exception as e:
        current_app.logger.error(f"Error displaying session data: {e}")
        return jsonify({"error": f"Could not display session: {str(e)}"}), 500
# --- END DEBUG ROUTE ---