from flask import Blueprint, render_template, request, jsonify, flash, current_app, redirect, url_for
from flask_login import login_required, current_user
from ..common.session import get_validated_data, get_deduped_results, get_infor_cl_matches
from ..common.utils import make_infor_upload_stack, apply_change, change_simulation_stage1, change_simulation_stage2, change_simulation_stage3, generate_network_graph
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
        
        # Get validated data from session
        validated_data = get_validated_data(user_id)
        deduped_results = get_deduped_results(user_id)
        
        if not validated_data:
            flash("No validated data found. Please complete the previous steps first.", "danger")
            return jsonify({
                'success': False,
                'message': "No validated data found. Please complete the previous steps first."
            }), 400
        
        # Get stacked data from step3 (deduplication results)
        deduped_results = get_deduped_results(user_id)
        if not deduped_results:
            flash("No stacked data found. Please complete the deduplication step first.", "danger")
            return jsonify({
                'success': False,
                'message': "No stacked data found. Please complete the deduplication step first."
            }), 400
        
        # # get merged df from step4 (infor_cl_matching results)
        # merged_results = get_infor_cl_matches(user_id)
        # if not merged_results:
        #     flash("No merged data found. Please complete the item matching step first.", "danger")
        #     return jsonify({
        #         'success': False,
        #         'message': "No merged data found. Please complete the item matching step first."
        #     }), 400
        
        # Convert data to DataFrames
        validated_df = pd.DataFrame(validated_data)
        # The stacked data should be the deduplicated results from step3
        stacked_df = pd.DataFrame(deduped_results.get('stacked_data', []))

        # # the merged data should be the infor_cl_matching results from step4, transform to make it match stacked_df format
        # stacked_df_b = make_infor_upload_stack(merged_results.get('merged_df', []))

        # output to temp_files dir for debugging
        validated_df.to_excel(os.path.join(current_app.root_path, "temp_files", f"validated_data_{user_id}.xlsx"), index=False)
        stacked_df.to_excel(os.path.join(current_app.root_path, "temp_files", f"stacked_data_{user_id}.xlsx"), index=False)
        # stacked_df_b.to_excel(os.path.join(current_app.root_path, "temp_files", f"merged_data_{user_id}.xlsx"), index=False)

        # it is possible to have stacked_df as empty or no stacked_df if there is no duplicates found.
        # under such case, we might still proceed but we need to handle it gracefully.
        
        # multiple stages to process the change simulation
        origianl_network_df = change_simulation_stage1(validated_df, stacked_df)
        data_change_show_df, data_change_df = change_simulation_stage2(validated_df, stacked_df)
        ccx_merge, tp_merge = apply_change(data_change_df, validated_df, stacked_df)
        modified_network_df = change_simulation_stage3(ccx_merge, tp_merge)

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
        
        # Store simulation results in the session if needed
        # session.store_simulation_results(user_id, simulation_results)
        
        # Return JSON response similar to other routes
        return jsonify({
            'success': True,
            'message': "Changes loaded successfully. Processing simulation...",
            'result': {
                'validated_count': len(validated_df),
                'stacked_count': len(stacked_df),
                'original_graph_data': original_graph_json,
                'modified_graph_data': modified_graph_json,
                'data_change_show': data_change_show_df.to_dict(orient='records')
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