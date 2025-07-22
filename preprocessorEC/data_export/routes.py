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
                         data_persistence_after_export)
from ..common.session import store_current_step, store_completed_steps
from ..common.utils_export_data import (make_batch_upload_excel,
                                        make_single_contract_excel,
                                        make_infor_direct_excel,
                                        make_batch_upload_csv,
                                        make_single_contract_csv,
                                        make_infor_direct_csv1,
                                        make_infor_direct_csv2)

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
            SET Status = 'Deleted', UpdateDT = GETDATE(), DeletedBy = ?
            WHERE TaskID = ?
        """, (current_user.id, task_id))
        
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
        export_extention = export_format.lower() if export_format == 'csv' else 'xlsx'
        
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