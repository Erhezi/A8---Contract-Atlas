from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session, current_app, Response
from flask import stream_with_context
from flask_login import login_required, current_user
# Import specific session helpers
from ..common.session import get_temp_table_name, store_infor_cl_matches
from ..common.db import match_to_infor_contract_lines, get_db_connection
# Removed unused imports for this specific function
# from ..common.utils import calculate_confidence_score, process_item_comparisons
# from ..common.model_loader import get_sentence_transformer_model
# import threading
# import pandas as pd
# from random import randint


# Create the blueprint
item_matching_bp = Blueprint('item_matching', __name__,
                          url_prefix='/item-matching',
                          template_folder='templates')


@item_matching_bp.route('/match-infor-contract-lines', methods=['POST'])
@login_required
def match_infor_cl():
    """Match items from the user's temporary table to Infor Contract Lines"""
    # Use current_user.id which is standard for Flask-Login
    if not current_user or not current_user.is_authenticated:
         return jsonify({'success': False, 'message': 'User not found or session expired.'}), 401
    user_id = current_user.id

    # --- Retrieve the table name from session using specific helper ---
    table_name = get_temp_table_name(user_id) # Use specific helper

    if not table_name:
        # If the table name isn't in the session, something went wrong in a previous step.
        # Do not try to recreate the table here.
        current_app.logger.error(f"Temporary table name not found in session for user {user_id}.") # Simplified log message
        return jsonify({
            'success': False,
            'message': 'Required data from previous steps is missing. Please ensure you have uploaded and validated your file.'
        }), 400 # Bad Request, as the prerequisite is missing

    current_app.logger.info(f"Attempting to match Infor CL for user {user_id} using table: {table_name}")

    # --- Perform the matching ---
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
             current_app.logger.error(f"Failed to get DB connection for user {user_id} during Infor CL matching.")
             # Consider a more specific error message for the user if appropriate
             return jsonify({'success': False, 'message': 'Database connection error.'}), 500

        # Call the database function to perform the matching
        success, error_msg, contract_list = match_to_infor_contract_lines(table_name, conn)
        
        # do three way matching for infor contract lines to ccx and upload calling functions from utils

        if not success:
            current_app.logger.error(f"match_to_infor_contract_lines failed for user {user_id}, table {table_name}: {error_msg}")
            return jsonify({
                'success': False,
                'message': f'Error during matching: {error_msg}' # Provide the specific error
            }), 500

        # Store results in session for subsequent steps using specific helper
        store_infor_cl_matches(user_id, contract_list) # Use specific helper

        current_app.logger.info(f"Successfully matched Infor CL for user {user_id}. Found {len(contract_list)} contracts/groups.")
        return jsonify({
            'success': True,
            'contracts': contract_list # Ensure the key matches what the JS expects
        })

    except Exception as e:
        # Log the full exception for debugging
        current_app.logger.exception(f"Unexpected error during Infor CL matching for user {user_id}: {e}")
        return jsonify({'success': False, 'message': f'An unexpected server error occurred.'}), 500
    finally:
        if conn:
            conn.close()
            current_app.logger.debug(f"DB connection closed for user {user_id} after Infor CL matching.")

