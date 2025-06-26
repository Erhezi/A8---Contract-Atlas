from flask import Blueprint, session, render_template, request, jsonify, flash, current_app, redirect, url_for
from flask_login import login_required, current_user
from ..common.session import (get_validated_data, get_deduped_results, get_infor_cl_matches, 
                              store_change_simulation_results, get_change_simulation_results)
from ..common.utils import (compute_changes_to_show, 
                            apply_change, 
                            change_simulation_stage1, 
                            change_simulation_stage2, 
                            change_simulation_stage3, 
                            compute_dataset_changes_df,
                            generate_network_graph,
                            change_simulation_stage4,
                            final_expire_item_validation,
                            final_commit,
                            final_errors_before_commit
                            )
from ..common.db import get_db_connection, get_relevant_contract_line
import os
import json
import pandas as pd
import networkx as nx

change_simulation_bp = Blueprint('change_simulation', __name__,
                              url_prefix='/change-simulation',
                              template_folder='templates')

@change_simulation_bp.route("/show-changes", methods=["POST"])
@login_required
def show_changes():
    """API endpoint to show simulated changes between validated data and stacked data"""
    try:
        user_id = current_user.id
        
        # Get update_action_mode from request, default to 'new'
        request_data = request.get_json() or {}
        update_action_mode = request_data.get('update_action_mode', 'new')
        
        # Validate update_action_mode parameter
        if update_action_mode not in ['new', 'legacy']:
            update_action_mode = 'new'  # Default to 'new' if invalid value
        
        # Get validated data from session
        validated_data = get_validated_data(user_id)       
        if not validated_data:
            flash("No validated data found. Please complete the previous steps first.", "danger")
            return jsonify({
                'success': False,
                'message': "No validated data found. Please complete the previous steps first."
            }), 400
        
        # Get stacked data from step3 (deduplication results)
        # this can be empty if there is no duplicates found, and the stacked data in this case need to be handled gracefully
        stacked_data = get_deduped_results(user_id).get('stacked_data', [])
        if not stacked_data or stacked_data == []:
            flash("No duplicates found in duplication dection step, procecedding with validated data only.", "info")
            for item in validated_data:
                row = {
                    'Buyer Part Num': item.get('Buyer Part Num', ''),
                    'Contract Number': item.get('Contract Number', ''),
                    'Contract Price': item.get('Contract Price', ''),
                    'Dataset': 'TP',
                    'Description': item.get('Description', ''),
                    'EA Price': 0.0,
                    'ERP Vendor ID': item.get('ERP Vendor ID', ''),
                    'Effective Date': item.get('Effective Date', ''),
                    'Expiration Date': item.get('Expiration Date', ''),
                    'File Row': item.get('File Row', ''),
                    'Keep': True,
                    'Mfg Part Num': item.get('Mfg Part Num', ''),
                    'Pair ID': 'tcx',
                    'QOE': item.get('QOE', ''),
                    'Rank': 1,
                    'Reduced Mfg Part Num': item.get('Reduced Mfg Part Num', ''),
                    'Source Contract Type': item.get('Source Contract Type', ''),
                    'Total Contract Line Count': 0,
                    'UOM': item.get('UOM', ''),
                    'Vendor Part Num': item.get('Vendor Part Num', '')
                }
                stacked_data.append(row)
        
        # get uom_qoe_validation restults from step4 (uom_qoe_validation)
        # this can be empty if there nothing to be validated (no item master matching found for items)
        merged_data = get_infor_cl_matches(user_id).get("merged_df", [])
        if not merged_data or merged_data == []:
            flash("No Item Master Item seems to attach to these screened items.", "info")
        
        # Convert data to DataFrames
        validated_df = pd.DataFrame(validated_data)
        # The stacked data should be the deduplicated results from step3
        stacked_df = pd.DataFrame(stacked_data)
        # items from step4 when matching to infor contract line with item numbers
        merged_df = pd.DataFrame(merged_data)

        
        # multiple stages to process the change simulation
        origianl_network_df = change_simulation_stage1(validated_df, stacked_df)
        data_change_show_df = change_simulation_stage2(validated_df, stacked_df, update_action_mode=update_action_mode)
        ccx_merge, tp_merge = apply_change(data_change_show_df, validated_df, stacked_df)

        contract_numbers = list(set(data_change_show_df['Contract Number'].dropna().astype(str)))
        conn = get_db_connection()
        if not conn:
             current_app.logger.error(f"Failed to get DB connection for user {user_id} during Infor CL matching.")
             # Consider a more specific error message for the user if appropriate
             return jsonify({'success': False, 'message': 'Database connection error.'}), 500
        
        success, error_msg, contract_line_count = get_relevant_contract_line(contract_numbers, conn)

        if not success:
            current_app.logger.error(f"Error retrieving contract line count: {error_msg}")
            flash("An error occurred while retrieving contract line counts. Please try again.", "danger")
            return jsonify({
                'success': False,
                'message': error_msg
            }), 500


        contract_line_count_df = pd.DataFrame(contract_line_count)
        contract_line_count_df = contract_line_count_df.rename(columns={
            'contract_number': 'Contract Number',
            'total_line_count': 'Total Contract Line Count'
        })
        
        modified_network_df, ccx_line_count_cal, tp_line_count_cal, line_count_before_after = change_simulation_stage3(ccx_merge, 
                                                                                                                       tp_merge, 
                                                                                                                       data_change_show_df,
                                                                                                                       contract_line_count_df)
        
        # Collect contracts and group them by type
        contract_a_set = set()
        contract_b_set = set()

        if not origianl_network_df.empty:
            contract_a_set.update(origianl_network_df['Contract Number_a'].dropna().astype(str))
            contract_b_set.update(origianl_network_df['Contract Number_b'].dropna().astype(str))

        if not modified_network_df.empty:
            contract_a_set.update(modified_network_df['Contract Number_a'].dropna().astype(str))
            contract_b_set.update(modified_network_df['Contract Number_b'].dropna().astype(str))

        # Remove empty strings and None values
        contract_a_set.discard('')
        contract_a_set.discard('nan')
        contract_b_set.discard('')
        contract_b_set.discard('nan')

        # Sort each group individually for consistency
        contract_a_list = sorted(list(contract_a_set))
        contract_b_list = sorted(list(contract_b_set))

        # Combine with Contract A first, then Contract B
        # This ensures Contract A nodes are grouped together in the first half of the circle
        # and Contract B nodes are grouped together in the second half
        all_contracts = contract_a_list + contract_b_list

        # Remove duplicates while preserving order (in case a contract appears in both A and B)
        seen = set()
        all_contracts_ordered = []
        for contract in all_contracts:
            if contract not in seen:
                all_contracts_ordered.append(contract)
                seen.add(contract)

        if all_contracts_ordered:
            master_G = nx.Graph()
            master_G.add_nodes_from(all_contracts_ordered)
            master_pos = nx.circular_layout(master_G)
            
            # Convert numpy arrays to regular Python lists for JSON serialization
            master_pos_serializable = {}
            for node, pos in master_pos.items():
                master_pos_serializable[node] = [float(pos[0]), float(pos[1])]
        else:
            master_pos_serializable = {}
        
        original_graph_json = generate_network_graph(origianl_network_df, fixed_pos=master_pos, show_IUD=False)
        modified_graph_json = generate_network_graph(modified_network_df, fixed_pos=master_pos, show_IUD=True)
        
        changes_to_show_df, reference_for_expire_rows = compute_changes_to_show(data_change_show_df, merged_df)

        # retrun the dataframes to the frontend for display for each change stats card
        ccx_create, ccx_update, ccx_expire, tp_create, tp_mute, tp_merged = compute_dataset_changes_df(data_change_show_df)
        
        data_change_show_df.loc[:, 'Do Not Expire'] = False
        # Store simulation results in the session if needed
        simulation_results = {
            'update_action_mode': update_action_mode,
            'all_changes': data_change_show_df.to_dict(orient='records'),
            'modified_network_df': modified_network_df.to_dict(orient='records'),
            'fixed_pos': master_pos_serializable,
            'changes_to_show': changes_to_show_df.to_dict(orient='records'),
            'reference_for_expire_rows': reference_for_expire_rows.to_dict(orient='records') if not reference_for_expire_rows.empty else [],
            'contract_line_count': contract_line_count_df.to_dict(orient='records')
        }
        store_change_simulation_results(user_id, simulation_results)
        
        # Return JSON response similar to other routes
        return jsonify({
            'success': True,
            'message': "Changes loaded successfully. Processing simulation...",
            'result': {
                'update_action_mode': update_action_mode,
                'original_graph_data': original_graph_json,
                'modified_graph_data': modified_graph_json,
                'data_change_show': changes_to_show_df.to_dict(orient='records'),
                'reference_for_expire_rows': reference_for_expire_rows.to_dict(orient='records') if not reference_for_expire_rows.empty else [],
                'line_count_before_after': line_count_before_after.to_dict(orient='records') if not line_count_before_after.empty else [],
                'ccx_create': ccx_create.to_dict(orient='records') if not ccx_create.empty else [],
                'ccx_update': ccx_update.to_dict(orient='records') if not ccx_create.empty else [],
                'ccx_expire': ccx_expire.to_dict(orient='records') if not ccx_expire.empty else [],
                'tp_create': tp_create.to_dict(orient='records') if not tp_create.empty else [],
                'tp_mute': tp_mute.to_dict(orient='records') if not tp_mute.empty else [],
                'tp_merged': tp_merged.to_dict(orient='records') if not tp_merged.empty else []
                # Add more details about the changes as needed
            }
        })
    except Exception as e:
        current_app.logger.error(f"Error in show_changes: {str(e)}")
        flash("An error occurred while processing changes. Please try again.", "danger")
        return jsonify({
            'success': False,
            'message': str(e)
        })
    finally:
        if conn:
            conn.close()


@change_simulation_bp.route("/update-expire-selections", methods=["POST"])
@login_required
def update_expire_selections():
    """Update the 'Do Not Expire' selections in session data"""
    try:
        user_id = current_user.id
        
        # Get the posted data
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                'success': False,
                'message': "No selection data provided."
            }), 400
            
        expire_selections = request_data.get('expire_selections', [])
        
        # Get simulation results from session
        simulation_results = get_change_simulation_results(user_id)
        if not simulation_results:
            return jsonify({
                'success': False,
                'message': "No simulation results found in session. Please run the simulation first."
            }), 404
        
        # Update both changes_to_show and all_changes data
        changes_to_show = simulation_results.get('changes_to_show', [])
        all_changes = simulation_results.get('all_changes', [])
        
        # Track updates for logging
        update_count = 0
        
        # Create lookup dictionaries with enhanced composite key for faster matching
        update_lookup = {}
        for selection in expire_selections:
            # Create an enhanced composite key with UOM added
            key = (
                selection.get('contract_number', '').strip(),
                selection.get('erp_vendor_id', '').strip(),
                selection.get('mfg_part_num', '').strip(),
                selection.get('vendor_part_num', '').strip(),
                selection.get('uom', '').strip()  # Add UOM to the key
            )
            update_lookup[key] = selection.get('do_not_expire', False)
        
        # Update changes_to_show first
        for item in changes_to_show:
            if item.get('Primary Action') == 'Expire CCX':
                # Create the same enhanced composite key for lookup
                key = (
                    str(item.get('Contract Number', '')).strip(),
                    str(item.get('ERP Vendor ID', '')).strip(),
                    str(item.get('Mfg Part Num', '')).strip(),
                    str(item.get('Vendor Part Num', '')).strip(),
                    str(item.get('UOM', '')).strip()  # Add UOM to the key
                )
                if key in update_lookup:
                    item['Do Not Expire'] = update_lookup[key]
                    update_count += 1
        
        # Then update all_changes
        for item in all_changes:
            if item.get('Primary Action') == 'Expire CCX':
                # Create the same enhanced composite key for lookup
                key = (
                    str(item.get('Contract Number', '')).strip(),
                    str(item.get('ERP Vendor ID', '')).strip(),
                    str(item.get('Mfg Part Num', '')).strip(),
                    str(item.get('Vendor Part Num', '')).strip(),
                    str(item.get('UOM', '')).strip()  # Add UOM to the key
                )
                if key in update_lookup:
                    item['Do Not Expire'] = update_lookup[key]
        
        # Save updated data back to session
        simulation_results['changes_to_show'] = changes_to_show
        simulation_results['all_changes'] = all_changes
        store_change_simulation_results(user_id, simulation_results)
        session.modified = True

        # update ccx_update after the changes applied
        all_changes_df = pd.DataFrame(all_changes)
        ccx_create, ccx_update, ccx_expire, tp_create, tp_mute, tp_merged = compute_dataset_changes_df(all_changes_df)

        current_app.logger.info(f"User {user_id} updated {update_count} 'Do Not Expire' selections")
        
        # Return the updated data for refreshing the UI
        return jsonify({
            'success': True,
            'message': f"Successfully saved {update_count} 'Do Not Expire' selections.",
            'updated_data': changes_to_show,  # Return the updated data
            'ccx_expire': ccx_expire.to_dict(orient='records') if not ccx_expire.empty else []
        })
        
    except Exception as e:
        current_app.logger.error(f"Error updating expire selections: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"An error occurred: {str(e)}"
        }), 500
    

@change_simulation_bp.route("/finalize-changes", methods=["POST"])
@login_required
def finalize_changes():
    """
    Perform safety checks on the changes, especially on 'Do Not Expire' items,
    before allowing the user to proceed to the next step.
    """
    try:
        user_id = current_user.id
        
        # Get simulation results from session
        simulation_results = get_change_simulation_results(user_id)
        if not simulation_results:
            return jsonify({
                'success': False,
                'message': "No simulation results found in session. Please run the simulation first."
            }), 404
        
        all_changes = simulation_results.get('all_changes', [])
        contract_line_count = simulation_results.get('contract_line_count', [])
        changes_to_show = simulation_results.get('changes_to_show', [])
        reference_for_expire_rows = simulation_results.get('reference_for_expire_rows', [])

        all_changes_df = pd.DataFrame(all_changes)
        contract_line_count_df = pd.DataFrame(contract_line_count)
        changes_to_show_df = pd.DataFrame(changes_to_show)
        reference_for_expire_rows_df = pd.DataFrame(reference_for_expire_rows)


        # final expire item validation
        final_expire_item_validated_df, more_changes_to_append_df, item_related_action_df = final_expire_item_validation(changes_to_show_df, 
                                                                                                 reference_for_expire_rows_df, 
                                                                                                 update_action_mode=simulation_results.get('update_action_mode', 'new'))
        
        final_commit_res, final_changes_to_show_df, final_all_changes_df = final_commit(all_changes_df,
                                                                                        changes_to_show_df,
                                                                                        more_changes_to_append_df,
                                                                                        final_expire_item_validated_df)
        
        final_validation = final_errors_before_commit(final_expire_item_validated_df)
        final_errors = final_validation.get('final_validation_errors', [])
        final_warnings = final_validation.get('final_validation_warnings', [])
        final_checks = final_validation.get('final_validation_checks', [])

         # retrieve the 'Do Not Expire' selections and plot
        df_network_r2, line_operations = change_simulation_stage4(final_all_changes_df, contract_line_count_df)
        print(line_operations)
        
        # Get fixed positions and generate graph JSONs
        modified_network_df = pd.DataFrame(simulation_results.get('modified_network_df', []))
        fixed_pos = simulation_results.get('fixed_pos', {})
        
        # Generate JSON data for both graphs
        modified_graph_json = generate_network_graph(modified_network_df, fixed_pos=fixed_pos, show_IUD=True)
        r2_graph_json = generate_network_graph(df_network_r2, fixed_pos=fixed_pos, show_IUD=True)

        return jsonify({
            'success': True,
            'message': "Changes finalized successfully.",
            'result': {
                'modified_graph_data': modified_graph_json,
                'r2_graph_data': r2_graph_json,
                'final_expire_item_validated': final_expire_item_validated_df.to_dict(orient='records'),
                'final_commit_df': final_commit_res.to_dict(orient='records'),
                'final_errors': final_errors.to_dict(orient='records') if not final_errors.empty else [],
                'final_warnings': final_warnings.to_dict(orient='records') if not final_warnings.empty else [],
                'final_checks': final_checks.to_dict(orient='records') if not final_checks.empty else [],
                'line_operations': line_operations.to_dict(orient='records') if not line_operations.empty else [],
            }
        })

    except Exception as e:
        current_app.logger.error(f"Error in finalize_changes: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"An error occurred: {str(e)}"
        }), 500