from flask import Blueprint, render_template, request, jsonify, flash, current_app, redirect, url_for
from flask_login import login_required, current_user
from ..common.session import get_validated_data, get_deduped_results, get_infor_cl_matches
from ..common.utils import make_infor_upload_stack
import os
import json
import pandas as pd

change_simulation_bp = Blueprint('change_simulation', __name__,
                              url_prefix='/change_simulation',
                              template_folder='templates')

@change_simulation_bp.route("/show_changes", methods=["POST"])
@login_required
def show_changes():
    """API endpoint to show simulated changes between validated data and stacked data"""
    user_id = current_user.id
    
    # Get validated data from session
    validated_data = get_validated_data(user_id)
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
    
    # get merged df from step4 (infor_cl_matching results)
    merged_results = get_infor_cl_matches(user_id)
    if not merged_results:
        flash("No merged data found. Please complete the item matching step first.", "danger")
        return jsonify({
            'success': False,
            'message': "No merged data found. Please complete the item matching step first."
        }), 400
    
    # Convert data to DataFrames
    validated_df = pd.DataFrame(validated_data)
    
    # The stacked data should be the deduplicated results from step3
    stacked_df_a = pd.DataFrame(deduped_results.get('stacked_data', []))

    # the merged data should be the infor_cl_matching results from step4, transform to make it match stacked_df format
    stacked_df_b = make_infor_upload_stack(merged_results.get('merged_df', []))

    # output to temp_files dir for debugging
    validated_df.to_excel(os.path.join(current_app.root_path, "temp_files", f"validated_data_{user_id}.xlsx"), index=False)
    stacked_df_a.to_excel(os.path.join(current_app.root_path, "temp_files", f"stacked_data_a_{user_id}.xlsx"), index=False)
    stacked_df_b.to_excel(os.path.join(current_app.root_path, "temp_files", f"merged_data_b_{user_id}.xlsx"), index=False)
    
    # Basic validation of the DataFrames
    if validated_df.empty:
        flash("Validated data is empty.", "danger")
        return jsonify({
            'success': False,
            'message': "Validated data is empty."
        }), 400
    
    # we can have stacked_df or merged_df empty, not a big deal

    # TODO: Analyze changes between the datasets
    # In a real implementation, you'd want to analyze the differences here
    
    # Store simulation results in the session if needed
    # session.store_simulation_results(user_id, simulation_results)
    
    # Return JSON response similar to other routes
    return jsonify({
        'success': True,
        'message': "Changes loaded successfully. Processing simulation...",
        'result': {
            'validated_count': len(validated_df),
            'stacked_count': len(stacked_df_a),
            'merged_count': len(stacked_df_b),
            # Add more details about the changes as needed
        }
    })