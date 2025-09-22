from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, Response
from flask import current_app, stream_with_context
from flask_login import login_required, current_user
from ..common.db import (
    get_db_connection,
    get_task_history,
    get_inspection_summary_count,
    get_affected_contract_by_task,
    get_tp_row_sync_by_taskid,
    get_exported_row_sync_by_taskid,
    mark_completed_by_task_id,
    add_completion_comment_by_task_id,
    get_completion_count_by_task_id,
    get_completion_timestamps_by_task_id
)
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
            # Include tasks that are exported and are still 'Pending' in status2
            # this means once the task is 'completed', it will be marked as such as no longer appear
            if task.get('Status') == 'Exported' and task.get('Status2') == 'Pending':
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
        json_data = request.get_json(force=True) or {}
        task_id = json_data.get('task_id') or json_data.get('taskId')
        comment = (json_data.get('comment') or '').strip()
        preset = json_data.get('preset')

        if not task_id:
            return jsonify({'success': False, 'message': 'task_id is required'}), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Could not connect to database'}), 500

        user_id = current_user.id

        # Mark completed
        try:
            success_complete = False
            msg_complete = ''
            result_complete = mark_completed_by_task_id(conn, task_id, user_id)
            # Support both (success, msg) or dict style returns
            if isinstance(result_complete, tuple):
                success_complete, msg_complete = result_complete[0], (result_complete[1] if len(result_complete) > 1 else '')
            elif isinstance(result_complete, dict):
                success_complete = result_complete.get('success', False)
                msg_complete = result_complete.get('message', '')
            else:
                success_complete = bool(result_complete)
            if not success_complete:
                return jsonify({'success': False, 'message': f'Unable to mark completed: {msg_complete}'}), 500

            # Store the completed task ID in session for step 8
            session['_commit_task_id'] = task_id
            session.modified = True

            # Optional comment persistence
            if comment:
                try:
                    result_comment = add_completion_comment_by_task_id(conn, task_id, comment)
                    if isinstance(result_comment, tuple) and result_comment and result_comment[0] is False:
                        current_app.logger.warning(f"Comment save failed for task {task_id}: {result_comment[1] if len(result_comment)>1 else ''}")
                except Exception as ce:
                    current_app.logger.warning(f"Error saving completion comment for task {task_id}: {ce}")

            return jsonify({'success': True, 'message': 'Task marked as completed', 'task_id': task_id, 'preset': preset})
        finally:
            conn.close()

    except Exception as e:
        current_app.logger.exception('Error marking step completed')
        return jsonify({'success': False, 'message': str(e)}), 500
    

@data_synchronization_bp.route('/completion-display/<task_id>')
@login_required
def completion_display(task_id):
    """Render completion (step 8) display with timeline and counts.

    Uses helper DB functions:
        get_completion_timestamps_by_task_id(conn, task_id)
        get_completion_count_by_task_id(conn, task_id)

    Falls back gracefully if any part fails.
    """
    try:

        print(task_id)  # Debug print for the task id from session

        if not task_id:
            return jsonify({'success': False, 'message': 'No Task ID provided for completion display'}), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Database connection not available'}), 500

        # Gather timestamps
        ts_success, ts_error, timestamps_result = get_completion_timestamps_by_task_id(conn, task_id)
        if not ts_success:
            current_app.logger.warning(f"Error retrieving completion timestamps for task {task_id}: {ts_error}")
            timestamps_result = {}

        # Gather counts
        counts_success, counts_error, counts_result = get_completion_count_by_task_id(conn, task_id)
        if not counts_success:
            current_app.logger.warning(f"Error retrieving completion counts for task {task_id}: {counts_error}")
            counts_result = {}

        # Use exact key names returned by DB layer (with fallbacks)
        timestamps_result = timestamps_result or {}
        counts_result = counts_result or {}

        # Use exact key names returned by DB layer
        create_ts = timestamps_result.get('createDT')
        export_ts = timestamps_result.get('exportedDT')
        completed_ts = timestamps_result.get('completedDT')

        # Build timeline structure
        def fmt(ts):
            return ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else (ts if ts else None)

        timeline = [
            {'key': 'created', 'label': 'Created', 'timestamp': create_ts, 'display': fmt(create_ts)},
            {'key': 'exported', 'label': 'Exported', 'timestamp': export_ts, 'display': fmt(export_ts)},
            {'key': 'completed', 'label': 'Completed', 'timestamp': completed_ts, 'display': fmt(completed_ts)},
        ]

        processed_items = counts_result.get('totalTPItems')
        affected_items = counts_result.get('totalAffectedItems')

        display_data = {
            'task_id': task_id,
            'timeline': timeline,
            'counts': {
                'processed_items': processed_items,
                'affected_items': affected_items
            },
            'errors': {
                'timestamps': ts_error if not ts_success else None,
                'counts': counts_error if not counts_success else None
            }
        }

        # set the session _commit_task_id to None after rendering
        session['_commit_task_id'] = None
        session.modified = True

        return jsonify({'success': True, 
                        'message': '', 
                        'data': display_data})
    except Exception as e:
        current_app.logger.exception('Error rendering completion display')
        return jsonify({'success': False, 'message': f'Error rendering completion display: {e}'}), 500
    finally:
        if 'conn' in locals():
            try:
                conn.close()
            except Exception:
                pass