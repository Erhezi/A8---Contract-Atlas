# Duplicate detection routes
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, Response
from flask import current_app, stream_with_context
from flask_login import login_required, current_user
import pandas as pd
from random import randint
from ..common.db import get_db_connection, find_duplicates_with_ccx
from ..common.model_loader import get_sentence_transformer_model
from ..common.utils import process_item_comparisons, calculate_confidence_score, apply_deduplication_policy
from ..common.session import get_comparison_results, store_deduplication_results, get_deduped_results

# Create the blueprint
duplicate_bp = Blueprint('duplicate_detection', __name__,
                        url_prefix='/duplicate-detection',
                        template_folder='templates')

@duplicate_bp.route('/process-duplicates', methods=['POST'])
@login_required
def process_duplicates():
    """Process the uploaded data to find duplications with CCX data"""
    try:
        # Check if we have validated data
        if 'validated_data' not in session:
            return jsonify({
                'success': False,
                'message': 'No validated data available. Please complete Step 1 first.'
            })
        
        # Convert validated data from session back to dataframe
        validated_df = pd.DataFrame.from_dict(session['validated_data'])
        
        # Get single connection for entire operation
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Failed to connect to database'
            })
        
        try:
            # Create a unique table name
            random_suffix = ''.join([str(randint(0, 9)) for _ in range(4)])
            user_id = current_user.id
            table_name = f"##temp_contract_{user_id}_{random_suffix}"  # Global Temp Table
            table_name = f"temp_contract_{user_id}" # for dev purpose
            print(table_name)
            
            # Create the temp table using the connection
            from ..common.db import create_temp_table
            success, result = create_temp_table(table_name, validated_df, conn)
            
            if not success:
                return jsonify({
                    'success': False,
                    'message': f'Failed to create temporary table: {result}'
                })
            
            # Run duplicate checking using same connection
            success, error_msg, contract_list = find_duplicates_with_ccx(table_name, conn)
            
            if not success:
                return jsonify({
                    'success': False,
                    'message': f'Error finding duplicates: {error_msg}'
                })
            
            # Store results in session
            session['contract_duplicates'] = contract_list
            
            # Commit at the end of all operations
            conn.commit()
            
            return jsonify({
                'success': True,
                'message': 'Duplicate analysis complete',
                'contracts': contract_list
            })
        finally:
            # Always close connection
            conn.close()
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing duplicates: {str(e)}'
        })

@duplicate_bp.route('/get-duplicate-contracts', methods=['GET'])
@login_required
def get_duplicate_contracts():
    """Get the list of contracts with potential duplicates"""
    if 'contract_duplicates' not in session:
        # If not in session yet, process the duplicates
        return jsonify({
            'success': False,
            'message': 'No duplicate data available. Please process duplicates first.'
        })
    
    return jsonify({
        'success': True,
        'contracts': session['contract_duplicates']
    })

@duplicate_bp.route('/get-duplicate-items/<contract_num>', methods=['GET'])
@login_required
def get_duplicate_items(contract_num):
    """Get the items for a specific contract"""
    if 'contract_duplicates' not in session:
        return jsonify({
            'success': False,
            'message': 'No duplicate data available. Please process duplicates first.'
        })
    
    # Find the contract in the session data
    contract_list = session['contract_duplicates']
    contract = next((c for c in contract_list if c['contract_number'] == contract_num), None)
    
    if not contract:
        return jsonify({
            'success': False,
            'message': f'Contract {contract_num} not found'
        })
    
    return jsonify({
        'success': True,
        'contract': contract
    })

@duplicate_bp.route('/include-exclude-contract', methods=['POST'])
@login_required
def include_exclude_contract():
    """Include or exclude a contract from further processing"""
    data = request.json
    contract_num = data.get('contract_num')
    include = data.get('include', True)
    
    if not contract_num:
        return jsonify({
            'success': False,
            'message': 'No contract number provided'
        })
    
    if 'contract_duplicates' not in session:
        return jsonify({
            'success': False,
            'message': 'No duplicate data available. Please process duplicates first.'
        })
    
    # Initialize included/excluded contracts if not present
    if 'included_contracts' not in session:
        session['included_contracts'] = [contract['contract_number'] for contract in session['contract_duplicates']]
    if 'excluded_contracts' not in session:
        session['excluded_contracts'] = []
    
    # Update the lists
    included = session['included_contracts']
    excluded = session['excluded_contracts']
    
    if include:
        if contract_num not in included:
            included.append(contract_num)
        if contract_num in excluded:
            excluded.remove(contract_num)
    else:
        if contract_num not in excluded:
            excluded.append(contract_num)
        if contract_num in included:
            included.remove(contract_num)
    
    # Update session
    session['included_contracts'] = included
    session['excluded_contracts'] = excluded
    
    return jsonify({
        'success': True,
        'included_contracts': included,
        'excluded_contracts': excluded
    })

@duplicate_bp.route('/initialize-included-contracts', methods=['POST'])
@login_required
def initialize_included_contracts():
    """Initialize the list of included contracts"""
    data = request.json
    contract_numbers = data.get('contract_numbers', [])
    
    if 'contract_duplicates' not in session:
        return jsonify({
            'success': False,
            'message': 'No duplicate data available. Please process duplicates first.'
        })
    
    # Set included contracts in session
    session['included_contracts'] = contract_numbers
    
    # Make sure excluded contracts don't contain any included ones
    if 'excluded_contracts' in session:
        excluded = session['excluded_contracts']
        for contract_num in contract_numbers:
            if contract_num in excluded:
                excluded.remove(contract_num)
        session['excluded_contracts'] = excluded
    else:
        session['excluded_contracts'] = []
    
    return jsonify({
        'success': True,
        'included_contracts': session['included_contracts'],
        'excluded_contracts': session.get('excluded_contracts', [])
    })

@duplicate_bp.route('/get-included-contracts', methods=['GET'])
@login_required
def get_included_contracts():
    """Get the current list of included contracts"""
    included = session.get('included_contracts', [])
    excluded = session.get('excluded_contracts', [])
    
    return jsonify({
        'success': True,
        'included_contracts': included,
        'excluded_contracts': excluded,
        'count': len(included)
    })

@duplicate_bp.route('/process-item-comparison-with-progress', methods=['GET'])
@login_required
def process_item_comparison_with_progress():
    """Process item comparison with real-time progress updates via SSE"""
    def generate():
        try:
            print("Starting item comparison with progress updates...")
            # Clear previous results
            if 'item_comparison_results' in session:
                del session['item_comparison_results']
            
            # Force session write to ensure cleared results are saved
            session.modified = True

            # Check if we have contract data and included contracts
            if 'contract_duplicates' not in session:
                yield f"data: {{" \
                      f"\"progress\": 100, \"total\": 0, \"processed\": 0, " \
                      f"\"message\": \"No contract data available\", \"status\": \"error\"}}\n\n"
                return
            
            # Get included contracts
            included_contracts = session.get('included_contracts', [])
            
            if not included_contracts:
                yield f"data: {{" \
                      f"\"progress\": 100, \"total\": 0, \"processed\": 0, " \
                      f"\"message\": \"No contracts included\", \"status\": \"error\"}}\n\n"
                return
            
            # Filter items from only the included contracts
            all_items = []
            contract_dict = {}
            
            # First create a dictionary of contract_number -> contract for faster lookup
            for contract in session['contract_duplicates']:
                contract_dict[contract['contract_number']] = contract
            
            # Then process only included contracts
            for contract_num in included_contracts:
                if contract_num in contract_dict:
                    all_items.extend(contract_dict[contract_num]['items'])
                else:
                    print(f"Warning: Contract {contract_num} not found in session data")
            
            # Log item count for debugging
            print(f"Found {len(all_items)} items from {len(included_contracts)} included contracts")
            
            total_items = len(all_items)
            if not total_items:
                yield f"data: {{" \
                      f"\"progress\": 100, \"total\": 0, \"processed\": 0, " \
                      f"\"message\": \"No items found\", \"status\": \"done\"}}\n\n"
                return

            # Initial progress update
            yield f"data: {{" \
                  f"\"progress\": 0, \"total\": {total_items}, \"processed\": 0, " \
                  f"\"message\": \"Starting comparison...\"}}\n\n"
            
            # Check for transformer model 
            model = None
            if current_app.config.get('TRANSFORMER_MODEL_LOADED', False):
                model = current_app.config.get('TRANSFORMER_MODEL')
                if model:
                    print("Using transformer model from app config")
                    yield f"data: {{" \
                          f"\"progress\": 0, \"total\": {total_items}, \"processed\": 0, " \
                          f"\"message\": \"Using transformer model for enhanced comparisons\"}}\n\n"
                else:
                    print("Warning: Transformer model is None despite being marked as loaded")
                    yield f"data: {{" \
                          f"\"progress\": 0, \"total\": {total_items}, \"processed\": 0, " \
                          f"\"message\": \"WARNING: Model unavailable, using basic comparison\"}}\n\n"
            else:
                print("Transformer model not loaded, using fallback method")
                yield f"data: {{" \
                      f"\"progress\": 0, \"total\": {total_items}, \"processed\": 0, " \
                      f"\"message\": \"Using basic comparison method (transformer model not loaded)\"}}\n\n"
            
            # Process items and calculate confidence scores
            scored_items = []
            for i, item in enumerate(all_items):
                scored_item = calculate_confidence_score(item, model=model)
                scored_items.append(scored_item)
                
                # Send progress update every 5 items or at the end
                if i % 5 == 0 or i == total_items - 1:
                    progress = round(((i + 1) / total_items) * 100)
                    yield f"data: {{" \
                          f"\"progress\": {progress}, \"total\": {total_items}, \"processed\": {i+1}, " \
                          f"\"message\": \"Processing item comparisons...\"}}\n\n"
            
            # Group by confidence level using skip_scoring=True (since we already calculated scores)
            result = process_item_comparisons(scored_items, skip_scoring=True)

            # debug test
            h = len(result.get('high', []))
            m = len(result.get('medium', []))
            l = len(result.get('low', []))
            t = h+m+l
            print(f"DEBUG: High: {h}, Medium: {m}, Low: {l}, Total: {t}")
            
            # Store the results in app cache
            user_id = current_user.id
            storage_key = f"temp_comparison_results_{user_id}"
            current_app.config[storage_key] = result
            
            # Create fresh summary objects
            result['summary'] = {
                'high': {'total': len(result.get('high', [])), 'false_positives': 0},
                'medium': {'total': len(result.get('medium', [])), 'false_positives': 0},
                'low': {'total': len(result.get('low', [])), 'false_positives': 0},
                'total_items': total_items
            }
            print(f"DEBUG: Summary: {result['summary']}")
            
            # Final message
            yield f"data: {{" \
                  f"\"progress\": 100, \"total\": {total_items}, \"processed\": {total_items}, " \
                  f"\"message\": \"Comparison complete!\", \"status\": \"done\"}}\n\n"
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error in process-item-comparison-with-progress: {error_msg}")
            yield f"data: {{" \
                  f"\"progress\": 100, \"message\": \"Error: {error_msg}\", \"status\": \"error\"}}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@duplicate_bp.route('/finalize-item-comparison', methods=['POST'])
@login_required
def finalize_item_comparison():
    """Finalize item comparison results and save to session"""
    try:
        # Get the data from app-level cache
        user_id = current_user.id
        storage_key = f"temp_comparison_results_{user_id}"
        comparison_results = None
        
        if storage_key in current_app.config:
            comparison_results = current_app.config[storage_key]
            # Save to session for future use
            session['item_comparison_results'] = comparison_results
            session.modified = True
            # Clean ap-level cache
            del current_app.config[storage_key]
        
        # If no results found, return error
        if not comparison_results:
            return jsonify({
                'success': False,
                'message': 'No comparison results found. Please try again.'
            })
        
        # Verify counts match
        h_count = len(comparison_results.get('high', []))
        m_count = len(comparison_results.get('medium', []))
        l_count = len(comparison_results.get('low', []))
        total_count = h_count + m_count + l_count
        
        # Verify and update summary if needed
        summary = comparison_results.get('summary', {})
        if summary.get('high', {}).get('total') != h_count or \
           summary.get('medium', {}).get('total') != m_count or \
           summary.get('low', {}).get('total') != l_count:
            
            # Update with corrected summary
            comparison_results['summary'] = {
                'high': {'total': h_count, 'false_positives': summary.get('high', {}).get('false_positives', 0)},
                'medium': {'total': m_count, 'false_positives': summary.get('medium', {}).get('false_positives', 0)},
                'low': {'total': l_count, 'false_positives': summary.get('low', {}).get('false_positives', 0)},
                'total_items': total_count
            }
            
            session['item_comparison_results'] = comparison_results
            session.modified = True
            summary = comparison_results['summary']
        
        return jsonify({
            'success': True,
            'message': 'Item comparison results finalized',
            'summary': summary
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error finalizing results: {str(e)}'
        })

@duplicate_bp.route('/process-item-comparison', methods=['POST'])
@login_required
def process_item_comparison():
    """Process item-level comparison for included contracts"""
    try:
        # Check if we have contract data
        if 'contract_duplicates' not in session:
            return jsonify({
                'success': False,
                'message': 'No contract data available. Please complete Step 2.1 first.'
            })
        
        # Get included contracts
        included_contracts = session.get('included_contracts', [])
        
        if not included_contracts:
            return jsonify({
                'success': False,
                'message': 'No contracts have been included. Please include at least one contract in Step 2.1.'
            })
        
        # Get all contract items
        all_items = []
        contract_list = session['contract_duplicates']
        
        for contract in contract_list:
            if contract['contract_number'] in included_contracts:
                all_items.extend(contract['items'])
        
        if not all_items:
            return jsonify({
                'success': False,
                'message': 'No items found in the included contracts.'
            })
        
        # Get transformer model
        model = None
        if current_app.config.get('TRANSFORMER_MODEL_LOADED', False):
            model = current_app.config.get('TRANSFORMER_MODEL')
        
        # Process comparison
        comparison_results = process_item_comparisons(all_items, model=model)
        
        # Store in session
        session['item_comparison_results'] = comparison_results
        
        return jsonify({
            'success': True,
            'message': 'Item comparison completed successfully',
            'summary': comparison_results['summary']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing item comparison: {str(e)}'
        })

@duplicate_bp.route('/get-item-comparison-summary', methods=['GET'])
@login_required
def get_item_comparison_summary():
    """Get summary of item comparison results"""
    try:
        # Check in the session first
        if 'item_comparison_results' in session:
            comparison_results = session['item_comparison_results']
        else:
            # Check if we have results in the app-level cache
            user_id = current_user.id
            storage_key = f"temp_comparison_results_{user_id}"
            
            if storage_key in current_app.config:
                # Found in app cache, store in session for future
                comparison_results = current_app.config[storage_key]
                session['item_comparison_results'] = comparison_results
                session.modified = True
            else:
                # No data found
                return jsonify({
                    'success': False,
                    'message': 'No comparison results available. Please complete the item comparison first.'
                })
        
        # Verify the integrity of the data structure
        if not isinstance(comparison_results, dict):
            return jsonify({
                'success': False,
                'message': 'Invalid comparison results format in session.'
            })
        
        # Count actual items in each confidence level
        h_count = len(comparison_results.get('high', []))
        m_count = len(comparison_results.get('medium', []))
        l_count = len(comparison_results.get('low', []))
        total_count = h_count + m_count + l_count
        
        # Get current summary
        current_summary = comparison_results.get('summary', {})
        
        # Check if summary matches actual counts
        if ('summary' not in comparison_results) or \
           (current_summary.get('high', {}).get('total') != h_count) or \
           (current_summary.get('medium', {}).get('total') != m_count) or \
           (current_summary.get('low', {}).get('total') != l_count):
            
            # Preserve false positives from existing summary
            high_fp = current_summary.get('high', {}).get('false_positives', 0)
            med_fp = current_summary.get('medium', {}).get('false_positives', 0)
            low_fp = current_summary.get('low', {}).get('false_positives', 0)
            
            # Create corrected summary
            summary = {
                'high': {'total': h_count, 'false_positives': high_fp},
                'medium': {'total': m_count, 'false_positives': med_fp},
                'low': {'total': l_count, 'false_positives': low_fp},
                'total_items': total_count
            }
            
            # Update in stored results
            comparison_results['summary'] = summary
            session['item_comparison_results'] = comparison_results
            session.modified = True
        else:
            summary = current_summary
        
        return jsonify({
            'success': True,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting summary: {str(e)}'
        })

@duplicate_bp.route('/get-comparison-items/<confidence_level>', methods=['GET'])
@login_required
def get_comparison_items(confidence_level):
    """Get items for a specific confidence level"""
    try:
        if 'item_comparison_results' not in session:
            return jsonify({
                'success': False,
                'message': 'No comparison results available. Please complete the item comparison first.'
            })
        
        if confidence_level not in ['high', 'medium', 'low']:
            return jsonify({
                'success': False,
                'message': 'Invalid confidence level. Must be high, medium, or low.'
            })
        
        comparison_results = session['item_comparison_results']
        items = comparison_results[confidence_level]
        
        return jsonify({
            'success': True,
            'items': items,
            'count': len(items)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching items: {str(e)}'
        })

@duplicate_bp.route('/update-false-positives', methods=['POST'])
def update_false_positives():
    """Update false positive status for selected items"""
    try:
        data = request.json
        confidence_level = data.get('confidence_level')
        item_updates = data.get('item_updates', [])
        
        # Get the item data from session
        if 'item_comparison_results' not in session:
            return jsonify({'success': False, 'message': 'No comparison results found in session'})
        
        comparison_results = session['item_comparison_results']
        
        # Get items for this confidence level
        if confidence_level not in comparison_results:
            return jsonify({'success': False, 'message': f'No items found for {confidence_level} confidence level'})
        
        level_items = comparison_results[confidence_level]
        
        # Apply updates
        for update in item_updates:
            index = update.get('index')
            is_false_positive = update.get('is_false_positive')
            
            if 0 <= index < len(level_items):
                level_items[index]['false_positive'] = is_false_positive
        
        # Save back to session
        comparison_results[confidence_level] = level_items
        
        # Recalculate summary
        summary = {
            'high': {
                'total': len(comparison_results.get('high', [])),
                'false_positives': sum(1 for item in comparison_results.get('high', []) if item.get('false_positive'))
            },
            'medium': {
                'total': len(comparison_results.get('medium', [])),
                'false_positives': sum(1 for item in comparison_results.get('medium', []) if item.get('false_positive'))
            },
            'low': {
                'total': len(comparison_results.get('low', [])),
                'false_positives': sum(1 for item in comparison_results.get('low', []) if item.get('false_positive'))
            },
            'total_items': len(comparison_results.get('high', [])) + 
                          len(comparison_results.get('medium', [])) + 
                          len(comparison_results.get('low', []))
        }
        
        # Update the summary in the comparison_results
        comparison_results['summary'] = summary
        
        # Save the entire updated results back to session
        session['item_comparison_results'] = comparison_results
        session.modified = True  # Ensure session changes are saved
        
        # Print debug info to confirm summary updated
        print(f"Updated summary for false positives: {comparison_results['summary']}")
        
        return jsonify({
            'success': True, 
            'summary': summary
        })
        
    except Exception as e:
        print(f"Error in update_false_positives: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})


@duplicate_bp.route('/apply-resolution', methods=['POST'])
@login_required
def apply_resolution():
    """Apply deduplication resolution policy to identified duplicates"""
    try:
        # Get policy selection from form
        deduplication_policy = request.form.get('resolution_strategy', 'keep_latest')
        
        # Get custom sort options if policy is custom
        custom_sort_fields = []
        custom_sort_directions = []
        
        if deduplication_policy == 'custom':
            # Extract priorities from form data
            priorities = {}
            preferences = {}
            
            for param in ['contract_source', 'unit_price', 'contract_status', 'expiration']:
                priority = request.form.get(f'{param}_priority')
                preference = request.form.get(f'{param}_preference')
                
                if priority and preference:
                    priorities[param] = int(priority)
                    preferences[param] = preference
            
            # Sort parameters by priority (1=highest)
            sorted_params = sorted(priorities.keys(), key=lambda k: priorities[k])
            
            # Map parameters to field names and directions
            param_to_field_map = {
                'contract_source': 'Source Contract Type',
                'unit_price': 'EA Price',
                'contract_status': 'Dataset',  # Using Dataset as proxy for status
                'expiration': 'Expiration Date'
            }
            
            # Map preferences to sort directions
            pref_to_direction_map = {
                'gpo_first': 'asc',  # GPO comes before Local alphabetically
                'local_first': 'desc',
                'lowest_first': 'asc',
                'highest_first': 'desc',
                'new_first': 'desc',  # TP (uploaded) comes after CCX alphabetically
                'existing_first': 'asc',
                'soonest_first': 'asc',
                'farthest_first': 'desc'
            }
            
            # Build sort fields and directions
            for param in sorted_params:
                field = param_to_field_map.get(param)
                direction = pref_to_direction_map.get(preferences[param], 'asc')
                
                if field:
                    custom_sort_fields.append(field)
                    custom_sort_directions.append(direction)
        
        comparison_results = get_comparison_results()
        
        if not comparison_results:
            return jsonify({
                'success': False,
                'message': 'No comparison results found. Please complete Step 2 first.'
            })
        
        # Process data with deduplication policy
        stacked_df, results_summary = apply_deduplication_policy(
            comparison_results, 
            deduplication_policy,
            custom_sort_fields,
            custom_sort_directions
        )
        
        # Store results in session
        store_deduplication_results({
            'stacked_data': stacked_df.to_dict('records') if not stacked_df.empty else [],
            'summary': results_summary,
            'policy': {
                'type': deduplication_policy,
                'custom_fields': custom_sort_fields,
                'custom_directions': custom_sort_directions
            }
        })
        
        # Return successful response with data
        return jsonify({
            'success': True,
            'message': 'Deduplication policy applied successfully',
            'summary': results_summary
        })
        
    except Exception as e:
        print(f"Error applying resolution: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error applying resolution: {str(e)}'
        })

@duplicate_bp.route('/get-deduplication-results', methods=['GET'])
@login_required
def get_deduplication_results():
    """Get the deduplication results for display""" 
    deduplication_results = get_deduped_results()
    if not deduplication_results:
        return jsonify({
            'success': False,
            'message': 'No deduplication results found. Please apply a resolution policy first.'
        })
    
    return jsonify({
        'success': True,
        'data': deduplication_results
    })

@duplicate_bp.route('/complete-step3', methods=['POST'])
@login_required
def complete_step3():
    """Complete Step 3 and move to Step 4"""
    
    deduplication_results = get_deduped_results()
    if not deduplication_results:
        return jsonify({
            'success': False,
            'message': 'No deduplication results found. Please apply a resolution policy first.'
        })
    
    # Mark step as completed
    try:
        current_app.mark_step_complete(3)
        # Update current step in session
        session['current_step_id'] = 4
        session.modified = True
        
        return jsonify({
            'success': True,
            'message': 'Step 3 completed successfully',
            'redirect': url_for('dashboard')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error completing step: {str(e)}'
        })