from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, send_file, current_app
from flask_login import login_required, current_user
import os
import pandas as pd
import zipfile
import tempfile
from datetime import datetime
import time
from ..common.db import (get_db_connection, 
                         get_task_history, 
                         get_data_export_line, 
                         get_im_link_line, 
                         get_contract_to_close,
                         data_persistence_after_export,
                         get_contracts_to_link,
                         commit_contract_link,
                         delete_task_by_task_id,
                         get_task_by_task_id,
                         get_task_errors,
                         get_task_error_edits,
                         save_error_edit,
                         revert_error_edit,
                         hold_task_by_task_id,
                         unhold_task_by_task_id,
                         delete_all_error_edits,
                         check_task_owner,
                         get_sync_percentages,
                         get_base_data_last_updateDT,
                         mark_task_reprocess,
                         apply_approved_edits,
                         log_task_file,
                         get_latest_task_file,
                         get_completion_timestamps_by_task_id)
from ..common.session import store_current_step, store_completed_steps, get_completed_steps
from ..common.utils_export_data import (make_batch_upload_excel,
                                        make_single_contract_excel,
                                        make_infor_direct_excel,
                                        make_batch_upload_csv,
                                        make_single_contract_csv,
                                        make_infor_direct_csv1,
                                        make_infor_direct_csv2)
from ..common.utils import error_edit_check, apply_edits_to_errors
from zoneinfo import ZoneInfo

# # Application timezone: New York local time
# NYC_TZ = ZoneInfo("America/New_York")

data_export_bp = Blueprint('data_export', __name__,
                           url_prefix='/data-export',
                           template_folder='templates')



@data_export_bp.route('/history')
@login_required
def view_history():
    """View task history for current user"""
    try:
        # Get database connection
        conn = get_db_connection()
        if not conn:
            flash("Could not connect to database.", "danger")
            return redirect(url_for('common.home'))
        
        # Get user ID and role
        user_id = current_user.id

        # current_completed_steps = get_completed_steps(user_id)

        # if 6 not in current_completed_steps:
        #     # automatically mark 1-5 as completed
        #     completed_steps = [1,2,3,4,5]
        #     store_completed_steps(user_id, completed_steps)
        #     store_current_step(user_id, 6)  # Set current step to 6 (Export Changes)
        #     session.modified = True
        # else:
        #     completed_steps = [1,2,3,4,5,6]
        #     store_completed_steps(user_id, completed_steps)
        #     store_current_step(user_id, 6)  
        #     session.modified = True

        
        # # Define the workflow steps for the sidebar
        # workflow_steps = [
        #     {"id": 1, "name": "File Pre-Checking"},
        #     {"id": 2, "name": "Duplication Overview"},
        #     {"id": 3, "name": "Resolve Duplications"},
        #     {"id": 4, "name": "Item Master Matching"},
        #     {"id": 5, "name": "Change Simulation"},
        #     {"id": 6, "name": "Export Changes"},
        #     {"id": 7, "name": "Sync Inspection"},
        #     {"id": 8, "name": "Completion"}
        # ]
        
        # # Set the current step object (not just ID)
        # current_step = {"id": 6, "name": "Export Changes"}
        
        # Fetch task history (list all but delete tasks need to be limited by task owner if user is sourcing)
        success, msg, tasks = get_task_history(conn, user_id=user_id, user_role="general") #hard-coded dummy user role to read all tasks
        
        if not success:
            flash(f"Error: {msg}", "danger")
            return redirect(url_for('common.home'))
        
        # Fetch contracts that need linking
        success_contracts, msg_contracts, contract_linking_data = get_contracts_to_link(conn)
        
        if not success_contracts:
            flash(f"Error retrieving contracts to link: {msg_contracts}", "warning")
            contract_linking_data = []
        
        # Render history template with tasks and workflow information
        return render_template('history.html', 
                              tasks=tasks, 
                              current_step = None,
                              contract_linking_data=contract_linking_data)
    
    except Exception as e:
        flash(f"Error retrieving task history: {str(e)}", "danger")
        return redirect(url_for('common.home'))

@data_export_bp.route('/commit-contract-link', methods=['POST'])
@login_required
def commit_contract_link_route():
    """Handle committing contract links"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        # Extract data from request
        task_id = data.get('task_id')
        export_group = data.get('export_group')
        contract_number_prp = data.get('contract_number_prp')
        erp_vendor_id_prp = data.get('erp_vendor_id_prp')
        contract_number = data.get('contract_number')
        erp_vendor_id_ccx = data.get('erp_vendor_id_ccx')
        user_id = current_user.id
        
        # Validate required fields
        if not task_id or not contract_number_prp or not erp_vendor_id_prp or not contract_number or not erp_vendor_id_ccx:
            return jsonify({
                'success': False,
                'message': 'Missing required fields'
            }), 400
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Database connection failed'
            }), 500
        
        # Commit the contract link
        try:
            success, msg = commit_contract_link(
                conn, 
                task_id, 
                contract_number_prp, 
                erp_vendor_id_prp, 
                contract_number, 
                erp_vendor_id_ccx, 
                export_group, 
                user_id
            )
            
            if not success:
                return jsonify({
                    'success': False,
                    'message': f'Failed to commit contract link: {msg}'
                }), 500
            
            return jsonify({
                'success': True,
                'message': 'Contract link committed successfully'
            })
            
        finally:
            conn.close()
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }), 500


@data_export_bp.route('/task/<task_id>/delete', methods=['POST'])
@login_required
def delete_task(task_id):
    """Mark a task as deleted."""
    try:
        conn = get_db_connection()
        if not conn:
            flash("Could not connect to database.", "danger")
            return redirect(url_for('data_export.view_history'))
        
        # add safe guard to check if task belong to current user 
        # front end should have delete disabled, adding this in case people directly access the URL
        user_id = current_user.id
        user_role = current_user.role
        if user_role not in ['admin', 'mdm']:
            # get user tasks
            success, msg, tasks = get_task_history(conn, user_id=user_id, user_role=user_role)
            if not success:
                flash(f"Error retrieving tasks: {msg}", "danger")
                return redirect(url_for('data_export.view_history'))
            # check if task belong to current user
            if not any(task['TaskID'] == task_id for task in tasks):
                flash("You do not have permission to delete this task.", "danger")
                return redirect(url_for('data_export.view_history'))

        # Use the refactored function to delete the task
        success, error_msg = delete_task_by_task_id(conn, task_id, current_user.id)
        if not success:
            flash(f"Error deleting task: {error_msg}", "danger")
        else:
            flash("Task has been marked as deleted.", "success")
        return redirect(url_for('data_export.view_history'))
    except Exception as e:
        flash(f"Error deleting task: {str(e)}", "danger")
        return redirect(url_for('data_export.view_history'))


@data_export_bp.route('/toggle-task-hold', methods=['POST'])
@login_required
def toggle_task_hold():
    """Toggle a task's status between Pending and Hold"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
            
        task_id = data.get('task_id')
        new_status = data.get('new_status')
        
        # Validate input
        if not task_id or new_status not in ['Pending', 'Hold']:
            return jsonify({'success': False, 'message': 'Invalid request parameters'}), 400
            
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Database connection failed'}), 500
            
        try:
            # Get current task to verify eligibility
            success, _, task = get_task_by_task_id(conn, task_id)
            if not success or not task:
                return jsonify({'success': False, 'message': 'Task not found'}), 404
                
            # Check permissions - only allow if task belongs to user or user is admin/mdm
            if task['UserID'] != current_user.id and current_user.role not in ['admin', 'mdm']:
                return jsonify({'success': False, 'message': 'Permission denied'}), 403
                
            # Update task status based on requested new status
            if new_status == 'Hold':
                success, error_message = hold_task_by_task_id(conn, task_id, current_user.id)
            else:  # new_status == 'Pending'
                success, error_message = unhold_task_by_task_id(conn, task_id, current_user.id)
                
            if not success:
                return jsonify({'success': False, 'message': error_message}), 400
                
            return jsonify({'success': True})
            
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@data_export_bp.route('/task/<task_id>')
@login_required
def view_task(task_id):
    """View details of a specific task."""
    try:
        conn = get_db_connection()
        if not conn:
            flash("Could not connect to database.", "danger")
            return redirect(url_for('data_export.view_history'))

        # Use the refactored function to get task details
        success, error_msg, task = get_task_by_task_id(conn, task_id)
        if not success:
            flash(f"Error retrieving task details: {error_msg}", "danger")
            return redirect(url_for('data_export.view_history'))
        
        has_edit_permission = False
        is_admin = current_user.role in ['admin', 'mdm']
        is_task_owner = task['UserID'] == current_user.id
        has_edit_permission = is_admin or is_task_owner
        
        print(current_user.id, task['UserID'], is_admin, is_task_owner, has_edit_permission) #debug

        # Fetch base data last update timestamp to support front-end scenario detection
        base_last_update = None
        try:
            ok_b, msg_b, base_last_update_ts = get_base_data_last_updateDT(conn)
            if ok_b:
                base_last_update = base_last_update_ts
        except Exception:
            base_last_update = None

        # Fetch completion timestamps including exported datetime
        exported_dt = None
        exported_days_ago = None
        try:
            success_ts, msg_ts, timestamps = get_completion_timestamps_by_task_id(conn, task_id)
            if success_ts and timestamps:
                exported_dt = timestamps.get('exportedDT')
                # Calculate days since export if exported_dt exists
                if exported_dt is not None:
                    try:
                        current_time = datetime.now()
                        
                        # Ensure both datetimes are timezone-naive for comparison
                        if hasattr(exported_dt, 'tzinfo') and exported_dt.tzinfo is not None:
                            exported_dt = exported_dt.replace(tzinfo=None)
                        if hasattr(current_time, 'tzinfo') and current_time.tzinfo is not None:
                            current_time = current_time.replace(tzinfo=None)
                        
                        time_diff = current_time - exported_dt
                        print(f"DEBUG: time_diff={time_diff}, time_diff.days={time_diff.days}")  # Debug
                        exported_days_ago = time_diff.days
                        print(f"DEBUG: Final exported_days_ago={exported_days_ago}")  # Debug
                    except Exception as calc_e:
                        print(f"DEBUG: Error in days calculation: {calc_e}")
                        exported_days_ago = None
                else:
                    print("DEBUG: exported_dt is None")
        except Exception as e:
            print(f"DEBUG: Exception in timestamp calculation: {e}")  # Debug
            exported_dt = None
            exported_days_ago = None

        # If task resolved for reprocess (WithError == 'Rpr'), fetch latest TP reprocess file
        tp_reprocess_url = None
        try:
            if str(task.get('WithError')) == 'Rpr':
                ok_fp, msg_fp, fp = get_latest_task_file(conn, task_id, 'TP_REPROCESS')
                if ok_fp and fp:
                    # Derive URL under static/exports if path is within that folder
                    exports_dir = os.path.join(current_app.static_folder, 'exports')
                    try:
                        # Normalize case and separators for Windows
                        if os.path.commonpath([os.path.abspath(fp), os.path.abspath(exports_dir)]) == os.path.abspath(exports_dir):
                            rel = os.path.relpath(fp, exports_dir).replace('\\', '/')
                            tp_reprocess_url = url_for('static', filename=f'exports/{rel}')
                    except Exception:
                        # Fallback: if filename pattern looks like TP_Reprocess_*.xlsx, use it directly
                        base = os.path.basename(fp)
                        tp_reprocess_url = url_for('static', filename=f'exports/{base}')
        except Exception:
            tp_reprocess_url = None

        return render_template('task_details.html', 
                               task=task,
                               current_step = None,
                               has_edit_permission=has_edit_permission,
                               base_last_update=base_last_update,
                               tp_reprocess_url=tp_reprocess_url,
                               exported_dt=exported_dt,
                               exported_days_ago=exported_days_ago)
    except Exception as e:
        flash(f"Error retrieving task details: {str(e)}", "danger")
        return redirect(url_for('data_export.view_history'))
    

@data_export_bp.route('/task/<task_id>/errors')
@login_required
def get_task_errors_api(task_id):
    """API endpoint to fetch task errors"""
    try:
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Database connection failed'
            }), 500
        
        # Fetch task errors
        success, error_msg, errors = get_task_errors(conn, task_id)

        if not success:
            return jsonify({
                'success': False,
                'message': error_msg
            }), 500

        # Fetch task error edits
        success, error_msg, error_edits = get_task_error_edits(conn, task_id)
        
        if not success:
            return jsonify({
                'success': False,
                'message': error_msg
            }), 500
        
        return jsonify({
            'success': True,
            'errors': errors,
            'error_edits': error_edits
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }), 500
    finally:
        if conn:
            conn.close()


@data_export_bp.route('/task/errors/edit', methods=['POST'])
@login_required
def edit_task_error():
    """API endpoint to save edits to error records"""
    conn = None
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Database connection failed'
            }), 500
        
        # Process each edit
        for edit in data.get('edits', []):
            edit_data = {
                'task_id': data.get('task_id'),
                'data_set': data.get('data_set'),
                'file_row': data.get('file_row'),
                'mfg_part_num': data.get('mfg_part_num', ''),
                'vendor_part_num': data.get('vendor_part_num', ''), 
                'uom': data.get('uom', ''),
                'qoe': data.get('qoe', 0), 
                'contract_number': data.get('contract_number'),
                'pkid': data.get('pkid'),
                'is_drop': data.get('is_drop', 0),
                'is_wrong': data.get('is_wrong', 0),
                'edit_field': edit.get('field'),
                'new_value': edit.get('newValue', ''),
                'edit_by': current_user.id
            }
            
            success, error_msg = save_error_edit(conn, edit_data)
            
            if not success:
                return jsonify({
                    'success': False,
                    'message': error_msg
                }), 500
        
        return jsonify({
            'success': True,
            'message': 'Edits saved successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }), 500
    finally:
        if conn:
            conn.close()


@data_export_bp.route('/task/errors/toggle-drop', methods=['POST'])
@login_required
def toggle_drop_error():
    """API endpoint to toggle the drop state of an error record"""
    conn = None
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Database connection failed'
            }), 500
        
        # Extract data edits from request
        edit_data = {
            'task_id': data.get('task_id'),
            'data_set': data.get('data_set'),
            'file_row': data.get('file_row'),
            'mfg_part_num': data.get('mfg_part_num', ''),
            'vendor_part_num': data.get('vendor_part_num', ''), 
            'uom': data.get('uom', ''),
            'qoe': data.get('qoe', 0), 
            'contract_number': data.get('contract_number'),
            'pkid': data.get('pkid'),
            'is_drop': data.get('is_drop', 0),
            'is_wrong': data.get('is_wrong', 0),
            'edit_field': 'DROP', 
            'new_value': data.get('new_value', ''),
            'edit_by': current_user.id
        }

        # step 1: revert any existing edits for this record
        success, message = revert_error_edit(conn,
                                            edit_data['task_id'],
                                            edit_data['pkid'],
                                            edit_data['data_set'],
                                            edit_data['file_row'],
                                            edit_data['contract_number'])
        if not success:
            return jsonify({
                'success': False,
                'message': message
            }), 500
        
        # step 2: insert the new edits by calling the save_error_edit function
        if edit_data['is_drop'] == 1:
            success, message = save_error_edit(conn, edit_data)

        if not success:
            return jsonify({
                'success': False,
                'message': message
            }), 500
        
        return jsonify({
            'success': True,
            'message': 'Drop state toggled successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }), 500
    finally:
        if conn:
            conn.close()

    

@data_export_bp.route('/task/errors/revert', methods=['POST'])
@login_required
def revert_task_error():
    """API endpoint to revert all changes for a record"""
    conn = None
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        # Extract key parameters
        task_id = data.get('task_id')
        pkid = data.get('pkid')
        data_set = data.get('data_set')
        file_row = data.get('file_row')
        contract_number = data.get('contract_number')
        
        # Validate required fields
        if not all([task_id, pkid, data_set, file_row, contract_number]):
            return jsonify({
                'success': False,
                'message': 'Missing required fields'
            }), 400
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Database connection failed'
            }), 500
        
        # Use the new revert function
        success, message = revert_error_edit(conn, task_id, pkid, data_set, file_row, contract_number)
        
        if not success:
            return jsonify({
                'success': False,
                'message': message
            }), 500
        
        return jsonify({
            'success': True,
            'message': message
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }), 500
    finally:
        if conn:
            conn.close()


@data_export_bp.route('/task/<task_id>/revert-all', methods=['POST'])
@login_required
def revert_all_task_errors(task_id):
    """API endpoint to revert all edits for a task"""
    conn = None
    try:
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Database connection failed'
            }), 500
        
        user_id = current_user.id
        user_role = current_user.role
        is_owner = False

        success, error_msg, is_owner = check_task_owner(conn, task_id, user_id)
        if not success:
            return jsonify({
                'success': False,
                'message': error_msg
            }), 500
        
        if not is_owner and user_role not in ['admin', 'mdm']:
            return jsonify({
                'success': False,
                'message': 'Permission denied. Only the task owner or admin/mdm can revert all edits.'
            }), 403
        
        if is_owner is True or (user_role in ['admin', 'mdm']):
            # Revert all edits for the task
            success, message = delete_all_error_edits(conn, task_id)
            if not success:
                return jsonify({
                    'success': False,
                    'message': message
                }), 500

        
        return jsonify({
            'success': True,
            'message': 'All edits reverted successfully'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }), 500

    finally:
        if conn:
            conn.close()


@data_export_bp.route('/task/<task_id>/sync-status')
@login_required
def get_task_sync_status(task_id):
    """API endpoint to fetch task synchronization status"""
    try:
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Database connection failed'
            }), 500
        
        # Get sync percentages from the database
        success, error_msg, percentages = get_sync_percentages(conn, task_id)
        
        if not success:
            return jsonify({
                'success': False,
                'message': error_msg
            }), 500
        
        # Return the sync percentages
        return jsonify({
            'success': True,
            'maxSync': percentages['maxSync'],
            'ccxTpSync': percentages['ccxTpSync'],
            'ccxChangesSync': percentages['ccxChangesSync'],
            'inforTpSync': percentages['inforTpSync'],
            'inforChangesSync': percentages['inforChangesSync']
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }), 500
    finally:
        if conn:
            conn.close()


@data_export_bp.route('/preview-data', methods=['POST'])
@login_required
def preview_data():
    """Get data preview for export"""
    try:
        # Get request data
        request_data = request.get_json()
        skip_gpo = request_data.get('skip_gpo', False)
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Database connection failed'
            }), 500
        
        # Get export data from database - for preview we get all completed tasks
        # that haven't been exported yet for the current user
        user_id = current_user.id
        user_role = current_user.role
        
        # Get all data eligible for export
        success, msg, export_data = get_data_export_line(conn, 
                                                         user_id=user_id, 
                                                         user_role=user_role)
        
        if not success:
            return jsonify({
                'success': False,
                'message': f'Failed to retrieve export data: {msg}'
            }), 500
        
        # get lines need to be linked to item master item
        success, msg, im_data = get_im_link_line(conn,
                                                    user_id=user_id, 
                                                    user_role=user_role)
        if not success:
            return jsonify({
                'success': False,
                'message': f'Failed to retrieve item master link data: {msg}'
            }), 500
        
        # get contract to be closed
        success, msg, contract_data = get_contract_to_close(conn,
                                                            user_id=user_id, 
                                                            user_role=user_role)
        
        if not success:
            return jsonify({
                'success': False,
                'message': f'Failed to retrieve contract data: {msg}'
            }), 500
        
        # Convert to DataFrame
        df = pd.DataFrame(export_data)
        item_link_df = pd.DataFrame(im_data)
        contract_to_close_df = pd.DataFrame(contract_data)

        if not df.empty:
            # consolidate by select the final rank = 1
            df = df[df['Final Rank'] == 1].copy()
            # sort df so data with issues (date conflicts, missing vendor parts) are at the top
            df.sort_values(by=['Final L Date Check', 'Final H Date Check', 'Create Count', 'Vendor Part Num', 'Expiration Date', 'Contract Number (PrP)', 'CreateDT'], 
                        ascending=[True, True, False, True, False, True, False], inplace=True)
        
            # Process the data
            batch_df = df[df['Export Group'] == 'Batch Upload'].copy()
            single_df = df[df['Export Group'] == 'Single Contract'].copy()

            
            # Handle batch data
            batch_data = []
            if not batch_df.empty:
                # Split GPO records
                gpo_batch = batch_df[batch_df['Source Type'].str.startswith('GPO', na=False)]
                other_batch = batch_df[~batch_df['Source Type'].str.startswith('GPO', na=False)]
                
                # Handle GPO records based on skip_gpo option
                if skip_gpo:
                    # Only include non-GPO records
                    batch_data = other_batch.to_dict('records')
                    gpo_data = gpo_batch.to_dict('records')
                else:
                    # Include all records
                    batch_data = batch_df.to_dict('records')
                    gpo_data = []
            
            # Handle single contract data
            single_data = []
            if not single_df.empty:
                single_data = single_df.to_dict('records')
        
        else:
            batch_data = []
            single_data = []
            gpo_data = []

        # handle item master link data
        item_link_data = []
        if not item_link_df.empty:
            item_link_df.loc[:, 'Item Master Auto Link'] = item_link_df['Active Vendor Item'].apply(lambda x: 'Manual' 
                                                                                                    if (pd.isnull(x) or x == 'No') 
                                                                                                    else 'Auto')
            item_link_df.loc[:, 'Inconsistent Mfg Part Num'] = item_link_df.apply(lambda x: 'Consistent' 
                                                                                  if x['Infor Mfg Part Num'] == x['Mfg Part Num'] 
                                                                                  else 'Inconsistent', axis=1)
            item_link_df.loc[:, 'Invalid Buy UOM'] = item_link_df['Valid Buy UOM'].apply(lambda x: 'Invalid' if pd.isnull(x) else 'valid')
            item_link_df.loc[:, 'Invalid Item'] = item_link_df['Active Vendor Item'].apply(lambda x: 'Invalid' 
                                                                                               if (pd.isnull(x) or x == 'No') else 'valid')
            item_link_data = item_link_df.to_dict('records')
        
        # handle contract to close data
        contract_to_close_data = []
        if not contract_to_close_df.empty:
            contract_to_close_df.columns = ['Contract Number',
                                            'Total Lines (Original)',
                                            'Total Lines (Change Applied)',
                                            'Total Lines Expired',
                                            'TaskID',
                                            'UserID']
            contract_to_close_data = contract_to_close_df.to_dict('records')
        
        return jsonify({
            'success': True,
            'batch_data': batch_data,
            'single_data': single_data,
            'gpo_data': gpo_data if skip_gpo and 'gpo_data' in locals() else [],
            'item_link_data': item_link_data,
            'contract_to_close_data': contract_to_close_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Preview failed: {str(e)}'
        }), 500
    finally:
        if conn:
            conn.close()


@data_export_bp.route('/export-data', methods=['POST'])
@login_required
def export_data():
    """Export function with sorting and row coloring"""
    conn = None
    temp_dir = None
    
    try:
        # Only allow admin and MDM users to export
        if current_user.role not in ['admin', 'mdm']:
            return jsonify({
                'success': False,
                'message': 'You do not have permission to export data'
            }), 403
        
        # Get request data
        user_id = current_user.id
        user_role = current_user.role
        request_data = request.get_json()
        skip_gpo = request_data.get('skip_gpo', False)
        export_format = request_data.get('export_format', 'excel')  # Default to Excel
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Database connection failed'
            }), 500
        
        # Get export data from database
        success, msg, export_data = get_data_export_line(conn, 
                                               user_id=user_id, 
                                               user_role=user_role)
        
        if not success or not export_data:
            return jsonify({
                'success': False,
                'message': f'Failed to retrieve export data: {msg}'
            }), 500
        
        # Convert to DataFrame
        df = pd.DataFrame(export_data)

        # convert price column to numeric to ensure correct reporting in excel
        # it should be made into the correct money format (to cents) in excel, so it should round to 2 decimal places
        df['Contract Price'] = pd.to_numeric(df['Contract Price'], errors='coerce')
        df['Contract Price'] = df['Contract Price'].round(2)
        
        # Filter by Final Rank = 1
        if 'Final Rank' in df.columns:
            df = df[df['Final Rank'] == 1].copy()
        
        # Create temporary directory for files
        temp_dir = tempfile.mkdtemp()
        
        # Get current date for filenames
        current_date = datetime.now().strftime('%Y%m%d')
        
        # Create export directory if it doesn't exist
        export_dir = os.path.join(current_app.static_folder, 'exports')
        os.makedirs(export_dir, exist_ok=True)
        
        # track all generaged files and path
        generated_files = []
        batch_df = df[df['Export Group'] == 'Batch Upload'].copy()
        single_df = df[df['Export Group'] == 'Single Contract'].copy()
        gpo_df = batch_df[batch_df['Source Type'].str.startswith('GPO', na=False)].copy()
        # make tuple of contract number and erp vendor id for single contracts
        unique_contract_vendor_id = single_df[['Contract Number (PrP)', 'ERP Vendor ID (PrP)']].drop_duplicates()
        single_contract_names = list(unique_contract_vendor_id.itertuples(index=False, name=None))

        batch_filepath = []
        single_filepath = []
        infor_direct_filepath = []

        if export_format == 'excel':
            batch_filename = f'batch_upload_{user_id}_{current_date}.xlsx' if not batch_df.empty else f'batch_upload_[NO DATA]_{user_id}_{current_date}.xlsx'
            batch_filepath.append(os.path.join(temp_dir, batch_filename))
            success_batch, error_msg_batch = make_batch_upload_excel(batch_df,
                                                         batch_filepath[-1],
                                                         skip_gpo=skip_gpo)
            
            infor_direct_filename = f'infor_direct_{user_id}_{current_date}.xlsx' if not gpo_df.empty else f'infor_direct_[NO DATA]_{user_id}_{current_date}.xlsx'
            infor_direct_filepath.append(os.path.join(temp_dir, infor_direct_filename))
            success_infor_direct, error_msg_infor_direct = make_infor_direct_excel(gpo_df,
                                                                                 infor_direct_filepath[-1])
            
            success_singles = []
            for contract_name, erp_vendor_id in single_contract_names:
                single_contract_df = single_df[single_df['Contract Number (PrP)'] == contract_name]
                single_filename = f'single_contract_[{erp_vendor_id}]_[{contract_name}]_{user_id}_{current_date}.xlsx' if not single_contract_df.empty else f'single_contract_[NO DATA]_{user_id}_{current_date}.xlsx'
                single_filepath.append(os.path.join(temp_dir, single_filename))
                success_single, error_msg_single = make_single_contract_excel(single_contract_df,
                                                                                single_filepath[-1],
                                                                                contract_name)
                success_singles.append(success_single)
            
            if not all(success_singles) or not success_batch or not success_infor_direct:
                error_messages = []
                if not success_batch:
                    error_messages.append(f"Batch upload export failed: {error_msg_batch}")
                if not success_infor_direct:
                    error_messages.append(f"Infor Direct export failed: {error_msg_infor_direct}")
                if not all(success_singles):
                    error_messages.append(f"Single contract export failed for some contracts: {error_msg_single}")
                
                return jsonify({
                    'success': False,
                    'message': 'Export failed',
                    'errors': error_messages
                }), 500
            
        elif export_format == 'csv':
            batch_filename = f'batch_upload_{user_id}_{current_date}.csv' if not batch_df.empty else f'batch_upload_[NO DATA]_{user_id}_{current_date}.csv'
            batch_filepath.append(os.path.join(temp_dir, batch_filename))
            success_batch, error_msg_batch = make_batch_upload_csv(batch_df,
                                                         batch_filepath[-1],
                                                         skip_gpo=skip_gpo)
            
            infor_direct_filename1 = f'infor_direct_data_{user_id}_{current_date}.csv' if not gpo_df.empty else f'infor_direct_data_[NO DATA]_{user_id}_{current_date}.csv'
            infor_direct_filepath.append(os.path.join(temp_dir, infor_direct_filename1))
            success_infor_direct1, error_msg_infor_direct1 = make_infor_direct_csv1(gpo_df,
                                                                                 infor_direct_filepath[-1])
            
            infor_direct_filename2 = f'infor_direct_GHX_GLD_{user_id}_{current_date}.csv' if not gpo_df.empty else f'infor_direct_GHX_GLD_[NO DATA]_{user_id}_{current_date}.csv'
            infor_direct_filepath.append(os.path.join(temp_dir, infor_direct_filename2))
            success_infor_direct2, error_msg_infor_direct2 = make_infor_direct_csv2(gpo_df,
                                                                                     infor_direct_filepath[-1])
            
            success_singles = []
            
            for contract_name, erp_vendor_id in single_contract_names:
                single_contract_df = single_df[single_df['Contract Number (PrP)'] == contract_name]
                single_filename = f'single_contract_[{erp_vendor_id}]_[{contract_name}]_{user_id}_{current_date}.csv' if not single_contract_df.empty else f'single_contract_[NO DATA]_{user_id}_{current_date}.csv'
                single_filepath.append(os.path.join(temp_dir, single_filename))
                success_single, error_msg_single = make_single_contract_csv(single_contract_df,
                                                                                single_filepath[-1],
                                                                                contract_name)
                success_singles.append(success_single)
            
            if not all(success_singles) or not success_batch or not success_infor_direct1 or not success_infor_direct2:
                error_messages = []
                if not success_batch:
                    error_messages.append(f"Batch upload export failed: {error_msg_batch}")
                if not success_infor_direct1:
                    error_messages.append(f"Infor Direct export failed: {error_msg_infor_direct1}")
                if not success_infor_direct2:
                    error_messages.append(f"Infor Direct export failed: {error_msg_infor_direct2}")
                if not all(success_singles):
                    error_messages.append(f"Single contract export failed for some contracts: {error_msg_single}")
                
                return jsonify({
                    'success': False,
                    'message': 'Export failed',
                    'errors': error_messages
                }), 500
        
        # add generated files to file list
        generated_files = batch_filepath + single_filepath + infor_direct_filepath
        
        # Get int current time for final zip file name for same day export to be unique
        file_id = int(time.time())

        # Create ZIP file of all generated files
        zip_filename = f"preprocessor_executable_changes_{user_id}_{current_date}_{file_id}.zip"
        zip_filepath = os.path.join(export_dir, zip_filename)
        
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            for file_path in generated_files:
                zipf.write(file_path, arcname=os.path.basename(file_path))
        
        # Mark data as exported in database
        if conn:
            success, msg = data_persistence_after_export(conn, user_id = user_id, user_role = user_role, zip_filename = zip_filename)
            if not success:
                return jsonify({
                    'success': False,
                    'message': f'Failed to mark data as exported: {msg}'
                }), 500

        # Generate URL for the ZIP file
        zip_url = url_for('static', filename=f'exports/{zip_filename}')
        
        return jsonify({
            'success': True,
            'message': 'Export completed successfully',
            'zipFile': {
                'name': zip_filename,
                'url': zip_url,
                'description': 'Click to download all file(s)'
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Export failed: {str(e)}'
        }), 500
        
    finally:
        # Clean up resources
        if conn:
            conn.close()
        
        # Clean up temp directory if it was created
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


@data_export_bp.route('/task/<task_id>/try-resolve', methods=['POST'])
@login_required
def try_resolve(task_id):
    """Try Resolve: decide scenario based on base-data timestamp and either reprocess or apply edits."""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"ok": False, "message": "Database connection failed"}), 500

        # Load task
        success, err_msg, task = get_task_by_task_id(conn, task_id)
        if not success or not task:
            return jsonify({"ok": False, "message": err_msg or "Task not found"}), 404

        # Permissions: owner or admin/mdm
        role = current_user.role
        is_admin = role in ['admin', 'mdm']
        is_owner = str(task.get('UserID')) == str(current_user.id)
        if not (is_admin or is_owner):
            return jsonify({"ok": False, "message": "Permission denied"}), 403

        # Disable if task deleted/completed if such status available
        status = str(task.get('Status', '')).lower()
        if status in ['deleted', 'completed']:
            return jsonify({"ok": False, "message": "Task is not eligible for Try Resolve"}), 400

        # Timestamps (America/New_York): base_last_update > createDT and < now => Scenario 1 (non-inclusive)
        task_created_at = task.get('CreateDT')
        if isinstance(task_created_at, str):
            try:
                task_created_at = datetime.fromisoformat(task_created_at)
            except Exception:
                task_created_at = None
        base_ok, base_msg, base_last_update = get_base_data_last_updateDT(conn)
        if not base_ok:
            return jsonify({"ok": False, "message": base_msg or "Failed to get base data timestamp"}), 500
        
        now_ts = datetime.now()

        print("Base last update:", base_last_update, "Task created at:", task_created_at, "Now:", now_ts) #debug

        # Fetch task errors
        success, error_msg, errors = get_task_errors(conn, task_id)
        if not success:
            return jsonify({"ok": False, "message": error_msg or "Failed to fetch task errors"}), 500
        # Fetch task error edits
        success, error_msg, error_edits = get_task_error_edits(conn, task_id)
        if not success:
            return jsonify({"ok": False, "message": error_msg or "Failed to fetch task error edits"}), 500

        # apply the error edits to errors to get the current view of errors
        modified_errors = apply_edits_to_errors(errors or [], error_edits or [])
        df_modified = pd.DataFrame(modified_errors)

        df_modified.to_excel(os.path.join(current_app.root_path, 'temp_files', f'task_{task_id}_modified_errors.xlsx'), index=False) #debug

        pkids_to_reprocess = set(df_modified['PKID'])

        scenario1 = False
        scenario1 = ((base_last_update > task_created_at) and (base_last_update < now_ts)) or (status == 'exported')

        if scenario1:
            # Scenario 1: produce new TP file and mark rows to Reprocess
            print("Scenario 1: base data updated after task creation or task exported") #debug
            ccx_expire = (df_modified['Intended Action'] == 'Expire') & (df_modified['DataSet'] == 'CCX')
            keep3 = (df_modified['Group'] == 'Keep3') & (df_modified['Actual Action'].isin(['Expire then Create (Create)',
                                                                                           'Update (New)']))
            tp_error_non_merge = (df_modified['DataSet'] == 'TP') & (df_modified['Primary Action'] != 'Merge')
            
            df_modified_s1 = df_modified[ccx_expire | keep3 | tp_error_non_merge].copy()

            required_cols = ['Mfg Part Num', 'Vendor Part Num', 'Buyer Part Num', 'Description',
                             'Contract Price', 'UOM', 'QOE', 'Effective Date', 'Expiration Date',
                             'Contract Number', 'ERP Vendor ID', 'Intended Action']
            df_new_tp = df_modified_s1[required_cols].copy()
            df_new_tp.loc[:, 'Source Contract Type'] = df_new_tp['Contract Number'].apply(
                lambda x: 'GPO' if str(x).upper().startswith('PP-') else 'Local')
            df_new_tp.loc[:, 'TaskID Ref'] = task_id
            
            # save to export folder
            export_dir = os.path.join(current_app.static_folder, 'exports')
            os.makedirs(export_dir, exist_ok=True)
            filename = f"TP_Reprocess_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            file_path = os.path.join(export_dir, filename)

            print("Saving new TP to:", file_path) #debug

            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df_new_tp.to_excel(writer, index=False, sheet_name=task_id)

            # log the generated file
            log_ok, log_msg = log_task_file(conn, task_id, 'TP_REPROCESS', file_path)
            if not log_ok:
                current_app.logger.warning(f"Failed to log TP_REPROCESS file for {task_id}: {log_msg}")
            
            # Generate URL for the TP reprocess file
            url_tp_reprocess = url_for('static', filename=f'exports/{filename}')

            upd_ok, upd_msg = mark_task_reprocess(conn, task_id)
            if not upd_ok:
                return jsonify({"ok": False, "message": upd_msg or "Failed to mark rows Reprocess"}), 500

            return jsonify({
                "ok": True,
                "scenario": 1,
                "message": "Base data refreshed: new file for preprocessing generated and previous error rows marked as Reprocess.",
                "download_path": url_tp_reprocess,
            })
        
        else:
            # Scenario 2: load errors + edits, run check, apply approved edits
            err_ok, err_msg, errors = get_task_errors(conn, task_id)
            if not err_ok:
                return jsonify({"ok": False, "message": err_msg or "Failed to load task errors"}), 500
            edits_ok, edits_msg, edits = get_task_error_edits(conn, task_id)
            if not edits_ok:
                return jsonify({"ok": False, "message": edits_msg or "Failed to load task edits"}), 500

            result = error_edit_check(errors or [], edits or [])
            approvals = [r for r in result.get('rows', []) if r.get('pass')]
            failures = [r for r in result.get('rows', []) if not r.get('pass')]

            if not approvals:
                return jsonify({
                    "ok": True,
                    "scenario": 2,
                    "message": "No edits qualify to execute. Review unresolved records.",
                    "approved": 0,
                    "failed": len(failures)
                })

            apply_ok, apply_msg, applied_count = apply_approved_edits(conn, task_id, approvals)
            if not apply_ok:
                return jsonify({"ok": False, "message": apply_msg or "Failed to apply edits"}), 500

            return jsonify({
                "ok": True,
                "scenario": 2,
                "message": "Approved edits applied. Rows marked Execute/PASS.",
                "approved": applied_count,
                "failed": len(failures)
            })

    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass