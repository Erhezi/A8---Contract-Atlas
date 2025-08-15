from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, Response
from flask import current_app, stream_with_context
from flask_login import login_required, current_user
from ..common.db import get_db_connection, get_task_history, get_inspection_summary_count, get_affected_contract_by_task, get_tp_row_sync_by_taskid, get_exported_row_sync_by_taskid
from ..common.session import store_current_step, store_completed_steps, get_completed_steps


data_synchronization_bp = Blueprint('data_synchronization', __name__,
                           url_prefix='/data-synchronization',
                           template_folder='templates')


@data_synchronization_bp.route('/task-headers')
@login_required
def get_task_headers():
    """Get task headers for synchronization inspection"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Could not connect to database'
            }), 500
        
        # Get task history using general role to see all tasks
        success, msg, tasks = get_task_history(conn, user_role="general")
        
        if not success:
            return jsonify({
                'success': False,
                'message': f'Error retrieving tasks: {msg}'
            }), 500
        
        # Filter tasks to only include those relevant for sync inspection
        # Typically tasks that have been exported or completed
        filtered_tasks = []
        for task in tasks:
            # Include tasks that are not in 'Pending' status or have been through export process
            if task.get('Status') not in ['Pending', 'Hold']:
                filtered_tasks.append({
                    'TaskID': task.get('TaskID'),
                    'UserID': task.get('UserID'),
                    'TPFileName': task.get('TPFileName', ''),
                    'Status': task.get('Status', ''),
                    'Status2': task.get('Status2', ''),
                    'WithError': task.get('WithError', False)
                })
        
        return jsonify({
            'success': True,
            'tasks': filtered_tasks
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }), 500
    finally:
        if 'conn' in locals():
            conn.close()


@data_synchronization_bp.route('/sync-details/<task_id>')
@login_required
def get_sync_details(task_id):
    """Get synchronization details for a specific task"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Could not connect to database'
            }), 500
        
        # Get inspection summary counts from the new view
        success, error_msg, summary_data = get_inspection_summary_count(conn, task_id)
        
        if not success:
            return jsonify({
                'success': False,
                'message': f'Error retrieving inspection summary: {error_msg}'
            }), 500
        
        # Prepare summary statistics using the view data
        summary = {
            'totalTPItems': summary_data.get('totalTPLines', 'N/A'),
            'totalTPItemsExecute': summary_data.get('totalExecutableTPLines', 'N/A'),
            'totalTPItemsPending': 'NA' if summary_data.get('totalTPLines') == 'N/A' or summary_data.get('totalExecutableTPLines') == 'N/A' else (
                summary_data.get('totalTPLines', 0) - summary_data.get('totalExecutableTPLines', 0) if 
                isinstance(summary_data.get('totalTPLines'), int) and isinstance(summary_data.get('totalExecutableTPLines'), int) else 'N/A'
            ),
            'totalChangedItems': summary_data.get('totalExportedChangeLines', 'N/A'),
            'totalChangedItemsExecute': summary_data.get('totalExecutableExportedChangeLines', 'N/A'),
            'totalChangedItemsPending': 'NA' if summary_data.get('totalExportedChangeLines') == 'N/A' or summary_data.get('totalExecutableExportedChangeLines') == 'N/A' else (
                summary_data.get('totalExportedChangeLines', 0) - summary_data.get('totalExecutableExportedChangeLines', 0) if 
                isinstance(summary_data.get('totalExportedChangeLines'), int) and isinstance(summary_data.get('totalExecutableExportedChangeLines'), int) else 'N/A'
            ),
            'totalErrorItems': summary_data.get('totalErrorCount', 'N/A'),
            'totalErrorItemsCCX': summary_data.get('totalCCXErrorCount', 'N/A'),
            'totalErrorItemsTP': summary_data.get('totalTPErrorCount', 'N/A')
        }
        
    # Get affected contracts data
        success_contracts, error_msg_contracts, contracts_data = get_affected_contract_by_task(conn, task_id)
        
        if not success_contracts:
            # If contracts data fails, continue with empty list but log the issue
            contracts_data = []
            current_app.logger.warning(f'Could not retrieve contracts data for task {task_id}: {error_msg_contracts}')
        
        # Get TP rows sync data
        success_tp, error_msg_tp, tp_rows = get_tp_row_sync_by_taskid(conn, task_id)
        if not success_tp:
            tp_rows = []
            current_app.logger.warning(f'Could not retrieve TP rows sync data for task {task_id}: {error_msg_tp}')
    
        
        # Exported rows (real data)
        success_exp, error_msg_exp, exported_rows = get_exported_row_sync_by_taskid(conn, task_id)
        if not success_exp:
            exported_rows = []
            current_app.logger.warning(f'Could not retrieve Exported rows sync data for task {task_id}: {error_msg_exp}')
        
        print(f"Retrieved {len(exported_rows)} exported rows for task {task_id}")
        if exported_rows:
            print("Sample exported row:", exported_rows[0])
        else:
            print("No exported rows data available")
        # shape exported rows to match frontend renderer if needed
        # Frontend can consume either the detailed dataset or aggregated; we'll pass detailed list
        sync_details = {
            'taskId': task_id,
            'summary': summary,
            'contractsAffected': contracts_data,
            'tpRowsStatus': tp_rows,
            'exportedRowsStatus': exported_rows
        }
        
        return jsonify({
            'success': True,
            'data': sync_details
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }), 500
    finally:
        if 'conn' in locals():
            conn.close()


@data_synchronization_bp.route('/mark-completed', methods=['POST'])
@login_required
def mark_step_completed():
    """Mark synchronization step as completed manually"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        task_id = data.get('taskId')
        completion_type = data.get('completionType')  # 'CCX' or 'Infor'
        comment = data.get('comment', '')
        
        if not task_id or not completion_type:
            return jsonify({
                'success': False,
                'message': 'Missing required fields'
            }), 400
        
        # Validate completion type
        if completion_type not in ['CCX', 'Infor']:
            return jsonify({
                'success': False,
                'message': 'Invalid completion type'
            }), 400
        
        try:
            conn = get_db_connection()
            if not conn:
                return jsonify({
                    'success': False,
                    'message': 'Database connection failed'
                }), 500
            
            # Here you would implement the actual database update
            # For now, we'll just simulate a successful update
            
            # Update step completion status for current user
            user_id = current_user.id
            completed_steps = get_completed_steps(user_id)
            if 7 not in completed_steps:
                completed_steps.append(7)
                store_completed_steps(user_id, completed_steps)
                store_current_step(user_id, 7)
                session.modified = True
            
            return jsonify({
                'success': True,
                'message': f'Task {task_id} marked as completed for {completion_type} sync'
            })
            
        finally:
            if 'conn' in locals():
                conn.close()
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'An error occurred: {str(e)}'
        }), 500


@data_synchronization_bp.route('/next-step')
@login_required
def proceed_to_next_step():
    """Proceed to Step 8 after completing synchronization inspection"""
    try:
        user_id = current_user.id
        
        # Mark step 7 as completed and advance to step 8
        completed_steps = get_completed_steps(user_id)
        if 7 not in completed_steps:
            completed_steps.append(7)
        
        store_completed_steps(user_id, completed_steps)
        store_current_step(user_id, 8)
        session.modified = True
        
        flash('Synchronization inspection completed. Proceeding to Step 8.', 'success')
        return redirect(url_for('common.workflow', step=8))
        
    except Exception as e:
        flash(f'Error proceeding to next step: {str(e)}', 'danger')
        return redirect(url_for('common.workflow', step=7))