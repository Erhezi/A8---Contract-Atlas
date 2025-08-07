from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, Response
from flask import current_app, stream_with_context
from flask_login import login_required, current_user
from ..common.db import get_db_connection, get_task_history, get_sync_percentages
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
        
        # Get sync percentages for the task
        success, error_msg, sync_percentages = get_sync_percentages(conn, task_id)
        
        if not success:
            return jsonify({
                'success': False,
                'message': f'Error retrieving sync percentages: {error_msg}'
            }), 500
        
        # Calculate summary statistics based on sync percentages
        # These are approximations based on the percentage data
        estimated_total_tp = 100  # Base number for percentage calculations
        
        summary = {
            'totalTPItems': estimated_total_tp,
            'totalChangedItems': int(estimated_total_tp * (sync_percentages.get('maxSync', 0) / 100)) if sync_percentages else 0,
            'totalErrorItems': int(estimated_total_tp * (1 - (sync_percentages.get('ccxTpSync', 0) / 100))) if sync_percentages else 0
        }
        
        sync_details = {
            'taskId': task_id,
            'summary': summary,
            'syncPercentages': sync_percentages or {},
            'contractsAffected': [
                # Placeholder data - could be enhanced with actual contract queries
                {
                    'contractNumber': 'Contract data available in detailed views',
                    'vendorId': 'See export history for details',
                    'changesCount': summary['totalChangedItems'],
                    'status': 'Synced' if sync_percentages and sync_percentages.get('ccxTpSync', 0) > 90 else 'Partial'
                }
            ],
            'tpRowsStatus': [
                # Placeholder data with sync percentage info
                {
                    'itemNumber': 'TP Items',
                    'contractNumber': 'CCX Sync',
                    'action': 'Various',
                    'syncStatus': f"{sync_percentages.get('ccxTpSync', 0):.1f}%" if sync_percentages else '0%',
                    'lastUpdated': 'Real-time'
                },
                {
                    'itemNumber': 'TP Items',
                    'contractNumber': 'Infor Sync',
                    'action': 'Various',
                    'syncStatus': f"{sync_percentages.get('inforTpSync', 0):.1f}%" if sync_percentages else '0%',
                    'lastUpdated': 'Real-time'
                }
            ],
            'exportedRowsStatus': [
                # Placeholder data with export sync info
                {
                    'exportGroup': 'CCX Export',
                    'contractNumber': 'Multiple',
                    'itemsCount': summary['totalChangedItems'],
                    'exportStatus': f"{sync_percentages.get('ccxChangesSync', 0):.1f}%" if sync_percentages and sync_percentages.get('ccxChangesSync') is not None else 'N/A',
                    'exportDate': 'Latest export'
                },
                {
                    'exportGroup': 'Infor Export',
                    'contractNumber': 'Multiple',
                    'itemsCount': summary['totalChangedItems'],
                    'exportStatus': f"{sync_percentages.get('inforChangesSync', 0):.1f}%" if sync_percentages and sync_percentages.get('inforChangesSync') is not None else 'N/A',
                    'exportDate': 'Latest export'
                }
            ]
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