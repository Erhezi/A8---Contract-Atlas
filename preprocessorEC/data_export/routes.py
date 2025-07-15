from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, send_file, current_app
from flask_login import login_required, current_user
import os
import pandas as pd
import io
import zipfile
import tempfile
from datetime import datetime
from ..common.db import (get_db_connection, 
                         get_task_history, 
                         get_data_export_line, 
                         get_im_link_line, 
                         get_contract_to_close,
                         data_persistence_after_export)
from ..common.session import store_current_step, store_completed_steps

data_export_bp = Blueprint('data_export', __name__,
                           url_prefix='/data-export',
                           template_folder='templates')

# utitlity function to apply excel styling
def apply_excel_styling(writer, df, sheet_name, date_conflict_rows=None, missing_vendor_rows=None):
    """Apply conditional formatting to Excel sheets based on data conditions
    
    Args:
        writer: ExcelWriter object
        df: DataFrame being written
        sheet_name: Name of the sheet
        date_conflict_rows: List of row indexes with date conflicts
        missing_vendor_rows: List of row indexes with missing vendor parts
    """
    # Get the xlsxwriter workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    
    # Define formats for different conditions
    date_conflict_format = workbook.add_format({
        'bg_color': '#f8d7da',  # Light red
        'border': 1
    })
    
    missing_vendor_format = workbook.add_format({
        'bg_color': '#fff3cd',  # Light yellow
        'border': 1
    })
    
    # Apply conditional formatting to rows
    if date_conflict_rows:
        for row in date_conflict_rows:
            worksheet.set_row(row + 1, None, date_conflict_format)  # +1 for header row
            
    if missing_vendor_rows:
        for row in missing_vendor_rows:
            # Only apply if not already marked as date conflict
            if date_conflict_rows and row not in date_conflict_rows:
                worksheet.set_row(row + 1, None, missing_vendor_format)  # +1 for header row


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
            df.sort_values(by=['Final L Date Check', 'Final H Date Check', 'Vendor Part Num'], 
                        ascending=[True, True, True], inplace=True)
        
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
    """Export data for CCX upload based on selected options"""
    conn = None
    temp_dir = None
    
    try:
        # Only allow admin and MDM users to export
        if current_user.role not in ['admin', 'mdm']:
            return jsonify({
                'success': False,
                'message': 'You do not have permission to export data'
            }), 403
        
        user_id = current_user.id
        user_role = current_user.role
        # Get request data
        request_data = request.get_json()
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
        
        # Filter by Final Rank = 1
        df = df[df['Final Rank'] == 1].copy()
        
        # Create temporary directory for files
        temp_dir = tempfile.mkdtemp()
        
        # Get current date for filenames
        current_date = datetime.now().strftime('%Y%m%d')
        user_id = current_user.id
        
        # List to store generated files
        generated_files = []
        
        # Create export directory if it doesn't exist
        export_dir = os.path.join(current_app.static_folder, 'exports')
        os.makedirs(export_dir, exist_ok=True)
        
        # 1. Process batch upload data
        batch_df = df[df['Export Group'] == 'Batch Upload'].copy()
        
        if not batch_df.empty:
            # Split GPO records
            gpo_batch = batch_df[batch_df['Source Type'].str.startswith('GPO', na=False)].copy()
            non_gpo_batch = batch_df[~batch_df['Source Type'].str.startswith('GPO', na=False)].copy()
            
            # Sort by Actual Action as requested
            action_order = {
                'Expire': 0,
                'Expire then Create (Expire)': 1,
                'Expire then Create (Create)': 2,
                'Update (New)': 3,
                'Create': 4
            }
            
            # Create a sorting key based on the action order
            def get_action_order(action):
                return action_order.get(action, 999)  # Default high number for unknown actions
            
            # Apply sorting
            if 'Actual Action' in batch_df.columns:
                batch_df['action_sort'] = batch_df['Actual Action'].apply(get_action_order)
                batch_df = batch_df.sort_values('action_sort').drop('action_sort', axis=1)
                
                if not gpo_batch.empty:
                    gpo_batch['action_sort'] = gpo_batch['Actual Action'].apply(get_action_order)
                    gpo_batch = gpo_batch.sort_values('action_sort').drop('action_sort', axis=1)
                
                if not non_gpo_batch.empty:
                    non_gpo_batch['action_sort'] = non_gpo_batch['Actual Action'].apply(get_action_order)
                    non_gpo_batch = non_gpo_batch.sort_values('action_sort').drop('action_sort', axis=1)
            
            # 1. Export batch_upload file (excluding GPO if skip_gpo is True)
            batch_to_export = non_gpo_batch if skip_gpo else batch_df
            
            if not batch_to_export.empty:
                # Define columns for batch upload
                batch_columns = [
                    'Organization', 'Vendor', 'Manufacturer', 'Contract Number',
                    'Contract Description', 'Tier Level', 'Tier Description', 'Source Type',
                    'Start Date', 'End Date', 'Mfg Part Num', 'Vendor Part Num',
                    'Buyer Part Num', 'Description', 'Contract Price', 'UOM', 'QOE',
                    'Effective Date', 'Expiration Date'
                ]
                
                # Select columns that exist in the DataFrame
                available_columns = [col for col in batch_columns if col in batch_to_export.columns]
                batch_output = batch_to_export[available_columns].copy()
                
                # Rename columns
                column_mapping = {
                    'Mfg Part Num': 'Mfg Part',
                    'Vendor Part Num': 'Vendor Part',
                    'Buyer Part Num': 'Buyer Part',
                    'Description': 'Item Description',
                    'Contract Price': 'Price',
                    'QOE': 'Qty'
                }
                batch_output.rename(columns={k: v for k, v in column_mapping.items() if k in batch_output.columns}, inplace=True)
                
                # Create filename
                batch_filename = f"batch_upload_{user_id}_{current_date}.{export_format}"
                batch_filepath = os.path.join(temp_dir, batch_filename)
                
                # Export based on format
                if export_format == 'excel':
                    # Find rows with date conflicts or missing vendor parts
                    date_conflict_rows = []
                    missing_vendor_rows = []
                    
                    for i, row in batch_to_export.iterrows():
                        # Check for date conflicts
                        if (('Final L Date Check' in row and row['Final L Date Check'] != 'pass') or 
                            ('Final H Date Check' in row and row['Final H Date Check'] != 'pass')):
                            date_conflict_rows.append(i)
                        
                        # Check for missing vendor parts with specific action
                        if (pd.isna(row.get('Vendor Part Num')) and 
                            row.get('Actual Action') in ['Create', 'Expire then Create (Create)', 'Update (New)']):
                            missing_vendor_rows.append(i)
                    
                    # Export with styling
                    with pd.ExcelWriter(batch_filepath, engine='xlsxwriter') as writer:
                        batch_output.to_excel(writer, sheet_name='Batch Upload', index=False)
                        apply_excel_styling(writer, batch_output, 'Batch Upload', 
                                           date_conflict_rows, missing_vendor_rows)
                else:
                    batch_output.to_csv(batch_filepath, index=False)
                
                generated_files.append(batch_filepath)
            
            # 2. Export Infor_direct_change file (GPO records)
            if not gpo_batch.empty and skip_gpo:
                # Use all columns for Infor direct change
                gpo_filename = f"Infor_direct_change_{user_id}_{current_date}.{export_format}"
                gpo_filepath = os.path.join(temp_dir, gpo_filename)
                
                if export_format == 'excel':
                    # Find rows with date conflicts or missing vendor parts
                    date_conflict_rows = []
                    missing_vendor_rows = []
                    
                    for i, row in gpo_batch.iterrows():
                        # Check for date conflicts
                        if (('Final L Date Check' in row and row['Final L Date Check'] != 'pass') or 
                            ('Final H Date Check' in row and row['Final H Date Check'] != 'pass')):
                            date_conflict_rows.append(i)
                        
                        # Check for missing vendor parts with specific action
                        if (pd.isna(row.get('Vendor Part Num')) and 
                            row.get('Actual Action') in ['Create', 'Expire then Create (Create)', 'Update (New)']):
                            missing_vendor_rows.append(i)
                    
                    # Export with styling
                    with pd.ExcelWriter(gpo_filepath, engine='xlsxwriter') as writer:
                        gpo_batch.to_excel(writer, sheet_name='GPO Records', index=False)
                        apply_excel_styling(writer, gpo_batch, 'GPO Records', 
                                           date_conflict_rows, missing_vendor_rows)
                else:
                    gpo_batch.to_csv(gpo_filepath, index=False)
                
                generated_files.append(gpo_filepath)
            
            # 3. Export GHX_EXCLUDE_GLD_to_merge file
            # Filter by specific actions
            ghx_filter = batch_df['Actual Action'].isin(['Expire', 'Expire then Create (Expire)', 'Update (New)'])
            ghx_df = batch_df[ghx_filter].copy()
            
            if not ghx_df.empty:
                # Select only required columns
                ghx_columns = ['Contract Number', 'Mfg Part Num', 'Supplier (CCX Sync)', 'UOM', 'TaskID']
                available_ghx_columns = [col for col in ghx_columns if col in ghx_df.columns]
                
                if len(available_ghx_columns) > 0:
                    ghx_output = ghx_df[available_ghx_columns].copy()
                    
                    # Create filename
                    ghx_filename = f"GHX_EXCLUDE_GLD_to_merge_{user_id}_{current_date}.{export_format}"
                    ghx_filepath = os.path.join(temp_dir, ghx_filename)
                    
                    # Export based on format
                    if export_format == 'excel':
                        ghx_output.to_excel(ghx_filepath, index=False)
                    else:
                        ghx_output.to_csv(ghx_filepath, index=False)
                    
                    generated_files.append(ghx_filepath)
        
        # 4. Export single contract files
        single_df = df[df['Export Group'] == 'Single Contract'].copy()
        
        if not single_df.empty:
            # Define columns for single contract (use what's available in preview)
            single_columns = [
                'Mfg Part Num', 'Vendor Part Num', 'Buyer Part Num',
                'Description', 'Contract Price', 'UOM', 'QOE',
                'Effective Date', 'Expiration Date'
            ]
            
            # Available columns
            available_columns = [col for col in single_columns if col in single_df.columns]
            
            # Group by Contract Number
            for contract_number, contract_group in single_df.groupby('Contract Number'):
                if not contract_group.empty:
                    contract_df = contract_group[available_columns].copy()
                    
                    # Create filename
                    contract_filename = f"single_contract_{contract_number}_{user_id}_{current_date}.{export_format}"
                    contract_filepath = os.path.join(temp_dir, contract_filename)
                    
                    # Export based on format
                    if export_format == 'excel':
                        # Find rows with date conflicts or missing vendor parts
                        date_conflict_rows = []
                        missing_vendor_rows = []
                        
                        for i, row in contract_group.iterrows():
                            # Check for date conflicts
                            if (('Final L Date Check' in row and row['Final L Date Check'] != 'pass') or 
                                ('Final H Date Check' in row and row['Final H Date Check'] != 'pass')):
                                date_conflict_rows.append(i)
                            
                            # Check for missing vendor parts with specific action
                            if (pd.isna(row.get('Vendor Part Num')) and 
                                row.get('Actual Action') in ['Create', 'Expire then Create (Create)', 'Update (New)']):
                                missing_vendor_rows.append(i)
                        
                        # Export with styling
                        with pd.ExcelWriter(contract_filepath, engine='xlsxwriter') as writer:
                            contract_df.to_excel(writer, sheet_name=f'Contract {contract_number}', index=False)
                            apply_excel_styling(writer, contract_df, f'Contract {contract_number}', 
                                               date_conflict_rows, missing_vendor_rows)
                    else:
                        contract_df.to_csv(contract_filepath, index=False)
                    
                    generated_files.append(contract_filepath)
        
        # Create ZIP file of all generated files
        zip_filename = f"preprocessor_executable_changes_{user_id}_{current_date}.zip"
        zip_filepath = os.path.join(export_dir, zip_filename)
        
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            for file_path in generated_files:
                zipf.write(file_path, arcname=os.path.basename(file_path))
        
        # Generate URL for the ZIP file
        zip_url = url_for('static', filename=f'exports/{zip_filename}')
        
        # Mark tasks as exported using data_persistence_after_export
        # Assuming this function exists and takes appropriate parameters
        if conn:
            data_persistence_after_export(conn, user_id)
        
        return jsonify({
            'success': True,
            'message': 'Export completed successfully',
            'zipFile': {
                'name': zip_filename,
                'url': zip_url,
                'description': 'All export files (ZIP)'
            },
            'files': []  # We no longer need individual file links since we have a zip
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