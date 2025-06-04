from flask import Blueprint, render_template, request, jsonify, flash, current_app, redirect, url_for
from flask_login import login_required, current_user
from ..common.session import get_validated_data, get_deduped_results, get_uom_qoe_validation
from ..common.utils import compute_changes_to_show, apply_change, change_simulation_stage1, change_simulation_stage2, change_simulation_stage3, generate_network_graph
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
        deduped_results = get_deduped_results(user_id)
        if not deduped_results:
            flash("No deduplicated results found. Please complete the deduplication step first.", "warning")
            deduped_results = {}
            # stacked_df in this case will simply take the validated data as the base, and we will map the columns to fit the stacked_df's structure
            stacked_data = []
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
            deduped_results['stacked_data'] = stacked_data
        
        # get uom_qoe_validation restults from step4 (uom_qoe_validation)
        # this can be empty if there nothing to be validated (no item master matching found for items)
        uom_qoe_validation = get_uom_qoe_validation(user_id)
        if not uom_qoe_validation:
            flash("No UOM/QOE validation results found. Please complete the validation step first.", "warning")
            uom_qoe_validation = {}
        
        # Convert data to DataFrames
        validated_df = pd.DataFrame(validated_data)
        # The stacked data should be the deduplicated results from step3
        stacked_df = pd.DataFrame(deduped_results.get('stacked_data', []))
        # analyzed_df from uom_qoe_validation
        analyzed_df = pd.DataFrame(uom_qoe_validation.get('analyzed_df', []))

        
        # multiple stages to process the change simulation
        origianl_network_df = change_simulation_stage1(validated_df, stacked_df)
        data_change_show_df, data_change_df = change_simulation_stage2(validated_df, stacked_df, update_action_mode=update_action_mode)
        ccx_merge, tp_merge = apply_change(data_change_df, validated_df, stacked_df)
        modified_network_df, ccx_line_count_cal, tp_line_count_cal, line_count_before_after = change_simulation_stage3(ccx_merge, tp_merge)

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
        else:
            master_pos = {}
        
        original_graph_json = generate_network_graph(origianl_network_df, fixed_pos=master_pos, show_IUD=False)
        modified_graph_json = generate_network_graph(modified_network_df, fixed_pos=master_pos, show_IUD=True)
        

        changes_to_show_df, reference_for_expire_rows = compute_changes_to_show(data_change_show_df, analyzed_df)
        
        # Store simulation results in the session if needed
        # session.store_simulation_results(user_id, simulation_results)
        
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
                'ccx_line_count_cal': ccx_line_count_cal.to_dict(orient='records') if not ccx_line_count_cal.empty else [],
                'tp_line_count_cal': tp_line_count_cal.to_dict(orient='records') if not tp_line_count_cal.empty else []
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