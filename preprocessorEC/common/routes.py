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
    get_uom_qoe_validation,
    get_change_simulation_results
)

# Import speific db helpers
from .db import (
    drop_temp_table,
    get_db_connection)

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
        # Get step definitions
        all_steps = current_app.get_all_steps()
        # Pass None explicitly for current_step to prevent any highlighting
        return render_template('index.html',
                              current_step=None,
                              steps=all_steps,
                              completed_steps=[])
    return redirect(url_for('auth.landing'))

@common_bp.route('/dashboard')
@login_required
def dashboard():
    user_id = current_user.id
    # Check if we need to restart the process
    if request.args.get('restart') or request.args.get('new_process'):
        # Store authentication-related session keys that should be preserved
        auth_keys = ['_user_id', '_fresh', '_id']
        auth_values = {key: session.get(key) for key in auth_keys if key in session}
        
        # Clear all user-specific session data
        for key in list(session.keys()):
            if not key.startswith('_') or key in ['_task_id', '_commit_task_id']:
                session.pop(key, None)
                
        # Restore authentication values
        for key, value in auth_values.items():
            session[key] = value
            
        # Reset step tracking
        store_current_step(user_id, 1)
        store_completed_steps(user_id, [])
        session.modified = True
        
        # Also clear any temp tables if they exist
        try:
            conn = get_db_connection()
            if conn:
                table_to_drop = f'temp_contract_{user_id}'
                drop_temp_table(table_to_drop, conn)
        except Exception as e:
            current_app.logger.warning(f"Failed to drop temp table: {str(e)}")
        
        flash('Starting a new process with clean data', 'info')
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

    # get task_id from query parameters if available for step 7
    if current_step_obj['id'] == 7:
        task_id = request.args.get('task_id', None)
    else:
        task_id = None


    return render_template('dashboard.html',
                           current_step=current_step_obj,
                           steps=all_steps,
                           completed_steps=completed_steps,
                           task_id = task_id) # Pass parameters to front end


@common_bp.route('/goto-step/<int:step_id>', methods=['GET'])
@login_required
def goto_step(step_id):
    """Navigate to a specific step if allowed, resetting later steps if going backward"""
    user_id = current_user.id
    completed_steps = get_completed_steps(user_id)
    current_step_id = get_current_step_from_session(user_id)
    task_id = request.args.get('task_id', None)

    if step_id == 7:
        completed_steps = [6]
        store_completed_steps(user_id, completed_steps)
        store_current_step(user_id, 7)  # Set current step to 7
        session.modified = True
        flash("Navigated to Sync Inspection.", 'info')
        # If task_id is provided, store it in the session
        if task_id:
            return redirect(url_for('common.dashboard', task_id = task_id))
        return redirect(url_for('common.dashboard'))
                
    if step_id == 6:
        completed_steps = []
        store_completed_steps(user_id, completed_steps)
        store_current_step(user_id, 6)  # Set current step to 6
        session.modified = True
        flash("Navigated to Export Changes.", 'info')
        return redirect(url_for('common.dashboard'))
    
    # Treat going back to Step 1 as a restart action
    if step_id == 1 and (current_step_id > 1 or completed_steps):
        # Store authentication-related session keys that should be preserved
        auth_keys = ['_user_id', '_fresh', '_id']
        auth_values = {key: session.get(key) for key in auth_keys if key in session}
        
        # Clear all user-specific session data
        for key in list(session.keys()):
            if not key.startswith('_') or key in ['_task_id', '_commit_task_id']:
                session.pop(key, None)
                
        # Restore authentication values
        for key, value in auth_values.items():
            session[key] = value
            
        # Reset step tracking
        store_current_step(user_id, 1)
        store_completed_steps(user_id, [])
        session.modified = True
        
        # Also clear any temp tables if they exist
        try:
            conn = get_db_connection()
            if conn:
                table_to_drop = f'temp_contract_{user_id}'
                drop_temp_table(table_to_drop, conn)
        except Exception as e:
            current_app.logger.warning(f"Failed to drop temp table: {str(e)}")
        
        flash('Starting a new process with clean data', 'info')
        return redirect(url_for('common.dashboard'))
    
    # Original functionality for other steps
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
    skip2 = False
    skip3 = False

    skip_steps = request.form.get('skip_steps')
    if skip_steps:
        # Parse the steps to skip (format "2,3")
        steps_to_skip = [int(s) for s in skip_steps.split(',') if s.isdigit()]
        
        # Mark steps as completed in session
        completed_steps = get_completed_steps(user_id)
        for step in steps_to_skip:
            if step == 2: 
                skip2 = True
            if step == 3:
                skip3 = True
            if step not in completed_steps:
                completed_steps.append(step)
        
        # Store updated completed steps
        store_completed_steps(user_id, completed_steps)
        
        # Update current step to the target step
        store_current_step(user_id, step_id)
        
        # Log the skip action
        current_app.logger.info(f"Skipping steps {skip_steps} to step {step_id} for user {user_id}")
        flash(f"Skipped steps {skip_steps} and moved to step {step_id + 1}", "info")
    
    # Validate if user can process this step (should be the current step)
    current_step_id = get_current_step_from_session(user_id)
    if (step_id != current_step_id) and (current_step_id not in [6, 7, 8]):
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
                success = True
                flash("Step 3 (Duplication Resolution) automatically completed.", "info")

        elif step_id == 3:

            validated_data = get_validated_data(user_id)
            if not validated_data:
                raise ValueError("No validated data available. Please complete Step 1 first.")

            if skip2 and skip3:
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
                success = True
            
            else:
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
            
            if skip2 and skip3:
                # If we skipped steps 2 and 3, we need to ensure we have the necessary data
                # This is a special case where we assume the user has already handled these steps
                infor_im_matches = get_infor_im_matches(user_id) # Use helper
                if not infor_im_matches:
                    raise ValueError("Infor Item Master matching not completed. Please run the matching process first.")
                
                uom_qoe_validation = get_uom_qoe_validation(user_id)
                if not uom_qoe_validation:  
                    raise ValueError("UOM and QOE validation not completed. Please run the validation process first.")
                
                success = True
            
            else:
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
            change_simulation = get_change_simulation_results(user_id) # Use helper
            if not change_simulation:
                raise ValueError("Change simulation results not available. Please run the simulation first.")
            # check if we have commited the results (we should be able get the _task_id from session)
            task_id = session.get('_task_id')
            if not task_id:
                raise ValueError("No task ID found for this task. Please re-run the pre-processor.")
            
            # if we reach here, meaning now the process for sourcing is all completed.
            # we will clear out session data that are not needed anymore and we will also drop the temp_contract_table created
            # drop temp table
            table_to_drop = f'temp_contract_{user_id}'
            try:
                conn = get_db_connection() # Use helper
                success_drop, drop_error_msg = drop_temp_table(table_to_drop, conn)
                print("temp table drop: ", table_to_drop)
                if not success_drop:
                    raise ValueError(f"Error dropping temporary table {table_to_drop}: {drop_error_msg}")
            except Exception as e:
                current_app.logger.exception(f"Error dropping temporary table {table_to_drop}: {e}")
                raise ValueError(f"Error dropping temporary table {table_to_drop}: {str(e)}")
            
            # clear session data
            session['_commit_task_id'] = session.get('_task_id', None) # Store the latest commit task ID for reference
            for key in list(session.keys()):
                if key not in ['_user_id', '_fresh', '_id', 
                               f'current_step_id_{user_id}', f'completed_steps_{user_id}',
                               '_commit_task_id']:
                    session.pop(key, None)
            
            success = True
            flash(f"Step 5 completed. Your task ID is {task_id}", "success")

            if current_user.role == 'sourcing' or current_user.role == 'admin':
                return redirect(url_for('common.home'))

        elif step_id == 6:
            # Step 6: Export Changes completion check
            # before we do anything, we will update the completed steps to include step 6  
            success = True
            flash("Step 6 marked as completed.", "success")

            # Explicitly mark step 6 as completed before redirecting
            completed_steps = get_completed_steps(user_id)
            if step_id not in completed_steps:
                completed_steps.append(step_id)
                store_completed_steps(user_id, completed_steps)

             # Check for redirect parameters based on user role
            if request.form.get('redirect_to_history') == 'true' and current_user.role in ['admin', 'mdm']:
                # Admin and MDM users get redirected to history page
                return redirect(url_for('data_export.view_history'))
            
            elif request.form.get('redirect_to_home') == 'true' and current_user.role == 'sourcing':
                # Sourcing users get redirected to home page
                return redirect(url_for('common.home'))

        elif step_id == 7:
            # Step 7: Synchronization Inspection completion check
            # Add checks relevant to step 7
            success = True
            flash("Step 7 marked as completed.", "success")

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