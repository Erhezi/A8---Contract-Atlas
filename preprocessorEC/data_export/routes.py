from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, send_file, current_app
from flask_login import login_required, current_user
import os
import pandas as pd
from datetime import datetime
from ..common.db import get_db_connection, get_task_history, get_data_export_line
from ..common.session import store_current_step, store_completed_steps

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
        
        if not success or not export_data:
            return jsonify({
                'success': False,
                'message': f'Failed to retrieve export data: {msg}'
            }), 500
        
        # Convert to DataFrame
        df = pd.DataFrame(export_data)
        # consolidate by select the final rank = 1
        df = df[df['Final Rank'] == 1].copy()
        
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
        
        return jsonify({
            'success': True,
            'batch_data': batch_data,
            'single_data': single_data,
            'gpo_data': gpo_data if skip_gpo and 'gpo_data' in locals() else []
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
    """Export data for CCX upload based on selected options"""
    try:
        # Only allow admin and MDM users to export
        if current_user.role not in ['admin', 'mdm']:
            return jsonify({
                'success': False,
                'message': 'You do not have permission to export data'
            }), 403
        
        # Get request data
        request_data = request.get_json()
        export_type = request_data.get('export_type')
        skip_gpo = request_data.get('skip_gpo', False)
        export_format = request_data.get('export_format', 'excel')
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Database connection failed'
            }), 500
        
        # Get export data from database
        success, msg, export_data = get_data_export_line(conn)
        
        if not success or not export_data:
            return jsonify({
                'success': False,
                'message': f'Failed to retrieve export data: {msg}'
            }), 500
        
        # Convert to DataFrame
        df = pd.DataFrame(export_data)
        
        # Create export directory if it doesn't exist
        export_dir = os.path.join(current_app.static_folder, 'exports')
        os.makedirs(export_dir, exist_ok=True)
        
        # Get current date for filenames
        current_date = datetime.now().strftime('%Y%m%d')
        
        # List to store generated files
        generated_files = []
        
        # Process based on export type
        if export_type == 'batch' or export_type is None:
            # Part 1: 'Export Group' == 'Batch Upload'
            batch_df = df[df['Export Group'] == 'Batch Upload'].copy()
            
            if not batch_df.empty:
                # Define columns for batch upload
                batch_columns = [
                    'Organization', 'Vendor', 'Manufacturer', 'Contract Number',
                    'Contract Description', 'Tier Level', 'Tier Description', 'Source Type',
                    'Start Date', 'End Date', 'Mfg Part Num', 'Vendor Part Num',
                    'Buyer Part Num', 'Description', 'Contract Price', 'UOM', 'QOE',
                    'Effective Date', 'Expiration Date'
                ]
                
                # Rename columns
                column_mapping = {
                    'Mfg Part Num': 'Mfg Part',
                    'Vendor Part Num': 'Vendor Part',
                    'Buyer Part Num': 'Buyer Part',
                    'Description': 'Item Description',
                    'Contract Price': 'Price',
                    'QOE': 'Qty'
                }
                
                # Split GPO records
                gpo_batch = batch_df[batch_df['Source Type'].str.startswith('GPO', na=False)]
                other_batch = batch_df[~batch_df['Source Type'].str.startswith('GPO', na=False)]
                
                # Handle GPO records
                if not gpo_batch.empty:
                    if skip_gpo:
                        # Export GPO records separately
                        gpo_filename = f"Infor_direct_change_{current_date}.xlsx"
                        gpo_filepath = os.path.join(export_dir, gpo_filename)
                        gpo_batch.to_excel(gpo_filepath, index=False)
                        
                        # Add to generated files
                        gpo_file_url = url_for('static', filename=f'exports/{gpo_filename}')
                        generated_files.append({
                            'name': gpo_filename,
                            'url': gpo_file_url,
                            'description': 'GPO Records (Direct Changes)'
                        })
                    else:
                        # Include GPO records in main batch
                        other_batch = pd.concat([other_batch, gpo_batch])
                
                # Export main batch if not empty
                if not other_batch.empty:
                    # Select and rename columns
                    batch_output = other_batch[batch_columns].copy()
                    batch_output.rename(columns=column_mapping, inplace=True)
                    
                    # Export batch file
                    batch_filename = f"batch_{current_date}.xlsx"
                    batch_filepath = os.path.join(export_dir, batch_filename)
                    batch_output.to_excel(batch_filepath, index=False)
                    
                    # Add to generated files
                    batch_file_url = url_for('static', filename=f'exports/{batch_filename}')
                    generated_files.append({
                        'name': batch_filename,
                        'url': batch_file_url,
                        'description': 'Batch Upload Template'
                    })
        
        if export_type == 'single' or export_type is None:
            # Part 2: 'Export Group' == 'Single Contract'
            single_df = df[df['Export Group'] == 'Single Contract'].copy()
            
            if not single_df.empty:
                # Define columns for single contract
                single_columns = [
                    'Mfg Part Num', 'Vendor Part Num', 'Buyer Part Num',
                    'Description', 'Contract Price', 'UOM', 'QOE',
                    'Effective Date', 'Expiration Date'
                ]
                
                # Group by Contract Number
                contract_groups = single_df.groupby('Contract Number')
                
                # Export each contract to a separate file
                for contract_number, contract_df in contract_groups:
                    # Select columns
                    contract_output = contract_df[single_columns].copy()
                    
                    # Export single contract file
                    contract_filename = f"{contract_number}_{current_date}.xlsx"
                    contract_filepath = os.path.join(export_dir, contract_filename)
                    contract_output.to_excel(contract_filepath, index=False)
                    
                    # Add to generated files
                    contract_file_url = url_for('static', filename=f'exports/{contract_filename}')
                    generated_files.append({
                        'name': contract_filename,
                        'url': contract_file_url,
                        'description': f'Single Contract: {contract_number}'
                    })
        
        # Update task status in the database to mark as exported
        # Note: In this new design we mark all eligible tasks as exported
        # Get all task IDs that were exported
        task_ids = df['TaskID'].unique().tolist()
        
        if task_ids:
            cursor = conn.cursor()
            task_ids_str = ','.join(['?' for _ in task_ids])
            cursor.execute(f"""
                UPDATE [DM_MONTYNT\\dli2].PreprocessorHeader
                SET Status = 'Exported', UpdateDT = GETDATE()
                WHERE TaskID IN ({task_ids_str})
            """, task_ids)
            conn.commit()
        
        return jsonify({
            'success': True,
            'message': 'Export completed successfully',
            'files': generated_files
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Export failed: {str(e)}'
        }), 500
    finally:
        if conn:
            conn.close()