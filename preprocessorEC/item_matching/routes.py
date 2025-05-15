from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session, current_app, Response
from flask import stream_with_context
from flask_login import login_required, current_user
# Import specific session helpers
from ..common.session import (get_temp_table_name, 
                              get_excluded_contracts, 
                              get_infor_cl_matches,
                              store_infor_cl_matches, 
                              get_comparison_results, 
                              store_infor_im_matches, 
                              get_infor_im_matches, 
                              store_uom_qoe_validation,
                              get_validated_data)
from ..common.db import match_to_infor_contract_lines, get_db_connection, match_to_item_master, get_valid_buying_uoms
# Removed unused imports for this specific function
from ..common.utils import (three_way_contract_line_matching, 
                            three_way_item_master_matching_compute_similarity, 
                            item_catched_in_infor_im_match, 
                            extract_item_numbers_for_validation, 
                            analyze_uom_qoe_discrepancies)
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


        # get the comparison results from session
        comparison_results = get_comparison_results(user_id)
        # get the excluded contracts from session
        excluded_contracts = get_excluded_contracts(user_id)
        # if stacked_data is None, meaning after review we end up having no duplicates in step2, this is fine, we will simply proceed
        # run three way matching
        print("calling three way matching ...")
        merged_df = three_way_contract_line_matching(comparison_results, contract_list, excluded_contracts)
        result = three_way_item_master_matching_compute_similarity(merged_df)
        
        # Store the result in session for later use
        merged_results = {
            'merged_df': merged_df.to_dict(orient='records'),  # Convert DataFrame to dict for JSON serialization
            'merged_to_review': result
        }
        store_infor_cl_matches(user_id, merged_results)  # Use specific helper


        current_app.logger.info(f"Successfully matched Infor CL for user {user_id}. Found {len(contract_list)} contracts/groups.")
        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        # Log the full exception for debugging
        current_app.logger.exception(f"Unexpected error during Infor CL matching for user {user_id}: {e}")
        return jsonify({'success': False, 'message': f'An unexpected server error occurred.'}), 500
    finally:
        if conn:
            conn.close()
            current_app.logger.debug(f"DB connection closed for user {user_id} after Infor CL matching.")


@item_matching_bp.route('/update-infor-cl-false-positives', methods=['POST'])
@login_required
def update_infor_cl_false_positives():
    """Update false positive flags for Infor CL matching results"""
    try:
        user_id = current_user.id
        data = request.json
        
        # Get the false positive updates
        false_positive_items = data.get('false_positive_items', [])
        print(f"Received false positive items: {false_positive_items}") # Debugging line
        
        if not false_positive_items:
            return jsonify({
                'success': False,
                'message': 'No false positive updates provided.'
            })
        
        # Get the stored infor_cl_matches from session
        from ..common.session import get_infor_cl_matches, store_infor_cl_matches
        
        # Retrieve current matches
        infor_cl_matches = get_infor_cl_matches(user_id)
        if not infor_cl_matches or 'merged_df' not in infor_cl_matches:
            return jsonify({
                'success': False,
                'message': 'No Infor CL matches found in session.'
            })
        
        # Get the merged dataframe as a list of dictionaries
        merged_df_list = infor_cl_matches['merged_df']
        
        # Update false positive flags in merged_df
        updated_items = 0
        for item in merged_df_list:
            if item.get('Need Review') == "No":
                continue
            for fp_item in false_positive_items:
                if (str(item.get('File_Row')) == str(fp_item.get('file_row')) and 
                    str(item.get('mfg_part_num_infor')) == str(fp_item.get('infor_mfn'))):
                    item['False Positive'] = fp_item.get('is_false_positive', False)
                    item['Need Review'] = "Reviewed"
                    updated_items += 1
                    break
        
        # Update the merged_to_review data
        if 'merged_to_review' in infor_cl_matches:
            result = infor_cl_matches['merged_to_review']
            
            # Update items in each confidence level
            for level in ['high', 'medium', 'low']:
                if level in result:
                    for item in result[level]:
                        for fp_item in false_positive_items:
                            if (str(item.get('File_Row')) == str(fp_item.get('file_row')) and 
                                str(item.get('mfg_part_num_infor')) == str(fp_item.get('infor_mfn'))):
                                item['False Positive'] = fp_item.get('is_false_positive', False)
                                break
            
            # Recalculate summary counts
            if 'summary' in result:
                high_fp = sum(1 for item in result.get('high', []) if item.get('false_positive'))
                medium_fp = sum(1 for item in result.get('medium', []) if item.get('false_positive'))
                low_fp = sum(1 for item in result.get('low', []) if item.get('false_positive'))
                
                # Update false positive counts
                result['summary']['high']['false_positives'] = high_fp
                result['summary']['medium']['false_positives'] = medium_fp
                result['summary']['low']['false_positives'] = low_fp
                
                # Calculate review counts
                need_review_count = 0
                no_need_review_count = 0
                item_master_matched = []
                for item in merged_df_list:
                    if item.get('False Positive') == False and item.get('item_number_infor') != '':
                        item_master_matched.append((item.get('File_Row'), item.get('item_number_infor')))
                    if item.get('Need Review') == 'Yes':
                        need_review_count += 1
                    elif item.get('Need Review') == 'No':
                        no_need_review_count += 1
                    elif item.get('Need Review') == 'Reviewed':
                        no_need_review_count += 1
                
                # Update summary counts
                result['summary']['review_count'] = need_review_count
                result['summary']['no_need_review_count'] = no_need_review_count
                result['summary']['item_master_count'] = len(set(item_master_matched))
                result['im_catched'] = list(set(item_master_matched))
        
        # Store updated results back in session
        store_infor_cl_matches(user_id, infor_cl_matches)
        session.modified = True  # Mark session as modified
        
        return jsonify({
            'success': True,
            'message': f'Successfully updated {updated_items} items.',
            'summary': infor_cl_matches.get('merged_to_review', {}).get('summary', {})
        })
        
    except Exception as e:
        current_app.logger.error(f"Error updating Infor CL false positives: {str(e)}")
        return jsonify({
            'success': False, 
            'message': f'Error updating false positives: {str(e)}'
        })


@item_matching_bp.route('/match-item-master', methods=['POST'])
@login_required
def match_item_master():
    """Match items from the user's temporary table to Infor Item Master"""
    # Use current_user.id which is standard for Flask-Login
    if not current_user or not current_user.is_authenticated:
         return jsonify({'success': False, 'message': 'User not found or session expired.'}), 401
    user_id = current_user.id

    # --- Retrieve the table name from session using specific helper ---
    table_name = get_temp_table_name(user_id)

    if not table_name:
        current_app.logger.error(f"Temporary table name not found in session for user {user_id}.")
        return jsonify({
            'success': False,
            'message': 'Required data from previous steps is missing. Please ensure you have uploaded and validated your file.'
        }), 400

    # --- Perform the matching ---
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
             current_app.logger.error(f"Failed to get DB connection for user {user_id} during Item Master matching.")
             return jsonify({'success': False, 'message': 'Database connection error.'}), 500

        # Call the database function to perform the matching
        success, error_msg, item_list = match_to_item_master(table_name, conn)

        if not success:
            current_app.logger.error(f"match_to_item_master failed for user {user_id}, table {table_name}: {error_msg}")
            return jsonify({
                'success': False,
                'message': f'Error during matching: {error_msg}'
            }), 500

        # extract matched im items
        im_catched = item_catched_in_infor_im_match(item_list['items'])
        item_list['im_catched'] = im_catched
        # Store the result in session for later use
        store_infor_im_matches(user_id, item_list)

        current_app.logger.info(f"Successfully matched Item Master for user {user_id}. Found {len(item_list)} items.")
        return jsonify({
            'success': True,
            'result': item_list
        })

    except Exception as e:
        # Log the full exception for debugging
        current_app.logger.exception(f"Unexpected error during Item Master matching for user {user_id}: {e}")
        return jsonify({'success': False, 'message': f'An unexpected server error occurred.'}), 500
    finally:
        if conn:
            conn.close()
            current_app.logger.debug(f"DB connection closed for user {user_id} after Item Master matching.")


@item_matching_bp.route('/update-item-master-false-positives', methods=['POST'])
@login_required
def update_item_master_false_positives():
    """Update false positive flags for Item Master matches"""
    user_id = current_user.id
    
    try:
        # Parse JSON data from request
        data = request.get_json()
        
        if not data or 'false_positive_items' not in data:
            return jsonify({'success': False, 'message': 'No false positive data provided'})
        
        # Get the current matches from session
        matches = get_infor_im_matches(user_id)
        
        if not matches or not isinstance(matches, dict) or 'items' not in matches:
            return jsonify({'success': False, 'message': 'No Item Master matches found in session'})
        
        # Get false positive items from request
        false_positive_items = data['false_positive_items']
        
        # Get items from matches
        all_items = matches['items']
        
        # Update false positive flags on the items
        items_updated = 0
        for fp_item in false_positive_items:
            file_row = str(fp_item.get('file_row', '')).strip().replace('N/A', '')
            item_number = str(fp_item.get('item_number', '')).strip().replace('N/A', '')
            infor_mfn = str(fp_item.get('infor_mfn', '')).strip().replace('N/A', '')
            infor_vendor_id = str(fp_item.get('infor_vendor_id', '')).strip().replace('N/A', '')
            is_false_positive = fp_item.get('is_false_positive', False)
            
            # Find matching items using all available identifying information
            for item in all_items:
                item_file_row = str(item.get('File_Row', '')).strip()
                item_number_infor = str(item.get('item_number_infor', '')).strip()
                item_mfn_infor = str(item.get('mfg_part_num_infor', '')).strip()
                item_vendor_id = str(item.get('erp_vendor_id_infor', '')).strip()
                
                if ((not file_row or item_file_row == file_row) and 
                    (not item_number or item_number_infor == item_number) and
                    (not infor_mfn or item_mfn_infor == infor_mfn) and
                    (not infor_vendor_id or item_vendor_id == infor_vendor_id)):
                    
                    # Mark the item
                    item['false_positive'] = is_false_positive
                    items_updated += 1
        
        # Update the false positive count
        false_positive_count = sum(1 for item in all_items if item.get('false_positive'))
        matches['false_positive_count'] = false_positive_count

        # compute im_catched
        im_catched = item_catched_in_infor_im_match(all_items)
        matches['im_catched'] = im_catched
        
        # Save updated matches back to session
        store_infor_im_matches(user_id, matches)
        session.modified = True  # Mark session as modified
        
        # Return success response with updated count
        return jsonify({
            'success': True, 
            'message': f'Updated {items_updated} items',
            'false_positive_count': false_positive_count
        })
        
    except Exception as e:
        current_app.logger.error(f"Error updating item master false positives: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})


@item_matching_bp.route('/validate-uom-qoe', methods=['POST'])
@login_required
def validate_uom_qoe():
    """Validate UOM and QOE against Infor Item Master"""
    user_id = current_user.id
    
    try:    
        # Get the Infor CL matches and Item Master matches
        infor_cl_matches = get_infor_cl_matches(user_id)
        infor_im_matches = get_infor_im_matches(user_id)
        
        # Extract im_catched from both sources
        im_catched_cl = infor_cl_matches.get('merged_to_review', {}).get('im_catched', []) if infor_cl_matches else []
        im_catched_im = infor_im_matches.get('im_catched', []) if infor_im_matches else []
        
        # Extract item numbers for validation
        item_numbers, im_catched_all_df = extract_item_numbers_for_validation(im_catched_cl, im_catched_im)
        
        # it is possible we end up with 0 match, so this could return true and we can proceed
        if not item_numbers:
            return jsonify({
                'success': True, 
                'message': 'No Item master item matched for validation.'
            })
        
        # Get valid buying UOMs from database
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'message': 'Database connection error.'})
            
        try:
            success, error_msg, valid_uoms = get_valid_buying_uoms(item_numbers, conn)
            print(valid_uoms)   # Debugging line
            
            if not success:
                return jsonify({'success': False, 'message': f'Error fetching valid UOMs: {error_msg}'})
            
            success, error_msg, validated_upload = get_validated_data(user_id)
            
            if not success:
                return jsonify({'success': False, 'message': f'Error fetching validated data: {error_msg}'})
                
            # Analyze UOM/QOE discrepancies
            analyzed_df = analyze_uom_qoe_discrepancies(valid_uoms, validated_upload, im_catched_all_df)
            
            # Store the results in the session
            store_uom_qoe_validation(user_id, analyzed_df.to_dict(orient='records'))
            
            # Return success response with summary
            return jsonify({
                'success': True,
                'result': {
                    'analyzed_df': analyzed_df.to_dict(orient='records'),
                }
            })
            
        finally:
            if conn:
                conn.close()
                current_app.logger.debug(f"DB connection closed for user {user_id} after UOM/QOE validation.")
                
    except Exception as e:
        current_app.logger.exception(f"Unexpected error during UOM/QOE validation for user {user_id}: {e}")
        return jsonify({'success': False, 'message': f'An unexpected server error occurred: {str(e)}'})