from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from flask_login import login_required, current_user
import os
from datetime import datetime
from ..common.db import get_db_connection, get_task_history
from ..common.session import store_current_step, store_completed_steps

data_export_bp = Blueprint('data_export', __name__,
                           url_prefix='/data_export',
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

        # automatically mark 1-5 as completed
        completed_steps = [1,2,3,4,5]
        store_completed_steps(user_id, completed_steps)
        store_current_step(user_id, 6)  # Set current step to 6 (Export Changes)
        session.modified = True
        
        # Define the workflow steps for the sidebar
        workflow_steps = [
            {"id": 1, "name": "File Pre-Checking"},
            {"id": 2, "name": "Duplication Overview"},
            {"id": 3, "name": "Resolve Duplications"},
            {"id": 4, "name": "Item Master Matching"},
            {"id": 5, "name": "Change Simulation"},
            {"id": 6, "name": "Export Changes"},
            {"id": 7, "name": "Sync Inspection"},
            {"id": 8, "name": "Completion"}
        ]
        
        # Set the current step object (not just ID)
        current_step = {"id": 6, "name": "Export Changes"}
        
        # Fetch task history based on user role
        success, msg, tasks = get_task_history(conn, user_id=user_id, user_role=current_user.role)
        
        if not success:
            flash(f"Error: {msg}", "danger")
            return redirect(url_for('common.home'))
        
        # Render history template with tasks and workflow information
        return render_template('history.html', 
                              tasks=tasks, 
                              workflow_steps=workflow_steps,
                              current_step=current_step,
                              completed_steps=completed_steps)
    
    except Exception as e:
        flash(f"Error retrieving task history: {str(e)}", "danger")
        return redirect(url_for('common.home'))


@data_export_bp.route('/task/<task_id>/delete', methods=['POST'])
@login_required
def delete_task(task_id):
    """Mark a task as deleted"""
    try:
        conn = get_db_connection()
        if not conn:
            flash("Could not connect to database.", "danger")
            return redirect(url_for('data_export.view_history'))
        
        # Check if user has permission to delete this task
        # For non-admin users, they should only be able to delete their own tasks
        if current_user.role not in ['admin', 'mdm']:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT UserID FROM [DM_MONTYNT\\dli2].PreprocessorHeader
                WHERE TaskID = ?
            """, (task_id,))
            
            task_owner = cursor.fetchone()
            
            if not task_owner or task_owner[0] != current_user.id:
                flash("You don't have permission to delete this task.", "danger")
                return redirect(url_for('data_export.view_history'))
        
        # Update task status to 'Deleted' and also update the UpdateDT column
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE [DM_MONTYNT\\dli2].PreprocessorHeader
            SET Status = 'Deleted', UpdateDT = GETDATE()
            WHERE TaskID = ?
        """, (task_id,))
        
        conn.commit()
        
        flash("Task has been marked as deleted.", "success")
        return redirect(url_for('data_export.view_history'))
    
    except Exception as e:
        flash(f"Error deleting task: {str(e)}", "danger")
        return redirect(url_for('data_export.view_history'))