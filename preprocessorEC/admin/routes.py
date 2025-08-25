from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from ..common.db import get_db_connection, get_task_history, clear_deleted_tasks, delete_existing_commit, repending_existing_exported_task

# Create the blueprint
admin_blueprint = Blueprint(
    'admin', 
    __name__, 
    url_prefix='/admin',
    template_folder='templates'
)

@admin_blueprint.route('/')
@login_required
def admin_index():
    """Admin main page"""
    if current_user.role != 'admin':
        flash('Access denied: Admins only', 'danger')
        return redirect(url_for('common.home'))
    return render_template('admin.html')

@admin_blueprint.route('/tasks')
@login_required
def list_tasks():
    """list all tasks (admin sees all)"""
    try:
        if current_user.role != 'admin':
            return jsonify({
                'success': False, 
                'message': 'Forbidden'
            }), 403

        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False, 
                'message': 'Database connection failed'
            }), 500

        # For admin perspective, pass user_role='admin' and no user_id to fetch all
        success, msg, tasks = get_task_history(conn, user_role='admin')
        if not success:
            return jsonify({
                'success': False, 
                'message': msg
            }), 500
        
        # Transform database field names to frontend expected names
        transformed_tasks = []
        for task in tasks:
            transformed_task = {
                'task_id': task.get('TaskID', ''),
                'user_name': task.get('UserID', ''),
                'wrike_task_id': task.get('WrikeID', ''),
                'filename': task.get('TPFileName', ''),
                'status_ccx': task.get('Status', ''),  # Status maps to status_ccx
                'status_infor': task.get('Status2', ''),  # Status2 maps to status_infor
                'created_at': str(task.get('CreateDT', '')) if task.get('CreateDT') else '',
                'updated_at': str(task.get('UpdateDT', '')) if task.get('UpdateDT') else '',
                'precheck_mode': task.get('PreCheckMode', ''),
                'dedup_mode': task.get('DedupMode', ''),
                'simulation_mode': task.get('SimulationMode', ''),
                'with_error': task.get('WithError', ''),
                'completed_by': task.get('CompletedBy', ''),
                'exported_by': task.get('ExportedBy', '')
            }
            transformed_tasks.append(transformed_task)
            if task.get('Statu') == 'Deleted':
                print(task)  # Debug print for deleted tasks

        return jsonify({
            'success': True, 
            'tasks': transformed_tasks
        })
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': str(e)
        }), 500

@admin_blueprint.route('/task/<task_id>/delete', methods=['POST'])
@login_required
def admin_delete_task(task_id):
    """Permanent delete of a task (remove records)"""
    try:
        if current_user.role != 'admin':
            return jsonify({
                'success': False, 
                'message': 'Forbidden'
            }), 403

        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False, 
                'message': 'Database connection failed'
            }), 500

        success, msg = delete_existing_commit('admin', task_id, conn)
        status_code = 200 if success else 400
        return jsonify({
            'success': success, 
            'message': msg
        }), status_code
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': str(e)
        }), 500

@admin_blueprint.route('/task/<task_id>/repend', methods=['POST'])
@login_required
def admin_repend_task(task_id):
    """Flip exported task back to pending and cleanup related traces"""
    try:
        if current_user.role != 'admin':
            return jsonify({
                'success': False, 
                'message': 'Forbidden'
            }), 403

        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False, 
                'message': 'Database connection failed'
            }), 500

        success, msg = repending_existing_exported_task('admin', task_id, conn)
        status_code = 200 if success else 400
        return jsonify({
            'success': success, 
            'message': msg
        }), status_code
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': str(e)
        }), 500

@admin_blueprint.route('/tasks/clear_deleted', methods=['POST'])
@login_required
def admin_clear_deleted_tasks():
    """Bulk clear already logically deleted tasks (final purge)"""
    try:
        if current_user.role != 'admin':
            return jsonify({
                'success': False, 
                'message': 'Forbidden'
            }), 403

        payload = request.get_json(silent=True) or {}
        task_ids = payload.get('task_ids') or []
        if not isinstance(task_ids, list) or not task_ids:
            return jsonify({
                'success': False, 
                'message': 'task_ids list required'
            }), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False, 
                'message': 'Database connection failed'
            }), 500

        success, msg = clear_deleted_tasks('admin', task_ids, conn)
        status_code = 200 if success else 400
        return jsonify({
            'success': success, 
            'message': msg
        }), status_code
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': str(e)
        }), 500

