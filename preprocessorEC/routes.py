from flask import (Blueprint, render_template, request, redirect, url_for, flash, session, 
                   current_app, send_from_directory, jsonify, send_file, Response, stream_with_context)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import pandas as pd
from utils import (validate_file, save_error_file, get_db_connection, 
                  db_transaction, create_temp_table, find_duplicates_with_ccx,
                  calculate_confidence_score, process_item_comparisons)
import os
import time
import re
from random import randint
import json

main_blueprint = Blueprint('main', __name__, template_folder='templates')

@main_blueprint.route('/process-duplicates', methods=['POST'])
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
            table_name = f"test_temp_contract_{user_id}" # for testing purpose - TEST TEST TEST
            print(table_name) # Debugging line - TEST TEST TEST
            
            # Create the temp table using the connection
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

@main_blueprint.route('/get-duplicate-contracts', methods=['GET'])
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

@main_blueprint.route('/get-duplicate-items/<contract_num>', methods=['GET'])
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

@main_blueprint.route('/include-exclude-contract', methods=['POST'])
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
        session['included_contracts'] = [contract['contract_number'] for contract in contract_num]
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

@main_blueprint.route('/initialize-included-contracts', methods=['POST'])
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

@main_blueprint.route('/get-included-contracts', methods=['GET'])
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

@main_blueprint.route('/process-item-comparison-with-progress', methods=['GET'])
@login_required
def process_item_comparison_with_progress():
    """Process item comparison with real-time progress updates via SSE"""
    def generate():
        try:
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
            
            # Get included contracts - USING THE SAME LOGIC AS process_item_comparison()
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
            print(f"DEBUG: High: {h}, Medium: {m}, Low: {l}, Total: {t}") # Debugging line TEST TEST TEST
            
            # Store the results in session AND a temp app cache 
            session['item_comparison_results'] = None # set to none to clear previous results, we will use the cache
            user_id = current_user.id
            storage_key = f"temp_comparison_results_{user_id}"
            current_app.config[storage_key] = result  # Store in app cache for quick access
            # Mark session as modified to ensure it gets saved
            session.modified = True

            # create fresh summary objects
            result['summary'] = {
                'high': {'total': len(result.get('high', [])), 'false_positives': 0},
                'medium': {'total': len(result.get('medium', [])), 'false_positives': 0},
                'low': {'total': len(result.get('low', [])), 'false_positives': 0},
                'total_items': total_items
            }
            print(f"DEBUG: Summary: {result['summary']}") # Debugging line TEST TEST TEST
            
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

@main_blueprint.route('/finalize-item-comparison', methods=['POST'])
@login_required
def finalize_item_comparison():
    """Finalize item comparison results and save to session"""
    try:
        # Data should be saved to session by process_item_comparison_with_progress
        # but due to SSE limitations, we need to explicitly finalize it here
        comparison_results = None

        # force re-sync session with app-level cache
        if not comparison_results:
            user_id = current_user.id
            storage_key = f"temp_comparison_results_{user_id}"
            if storage_key in current_app.config:
                comparison_results = current_app.config[storage_key]
                # save to session for future use
                session['item_comparison_results'] = comparison_results
                session.modified = True
                # clean ap-level cache
                del current_app.config[storage_key]
                print(comparison_results['summary']) # Debugging line TEST TEST TEST

        # if still no results, return error
        # This should not happen if the process_item_comparison_with_progress was successful
        if not comparison_results:
            return jsonify({
                'success': False,
                'message': 'No comparison results found. Please try again. SSE process may have failed.'
            })
        
        if comparison_results:
            h_count = len(comparison_results.get('high', []))
            m_count = len(comparison_results.get('medium', []))
            l_count = len(comparison_results.get('low', []))
            total_count = h_count + m_count + l_count
            
            # Verify and update summary if needed
            summary = comparison_results.get('summary', {})
            if summary.get('high', {}).get('total') != h_count or \
            summary.get('medium', {}).get('total') != m_count or \
            summary.get('low', {}).get('total') != l_count:
                print(f"Correcting mismatched summary. Expected: {h_count}/{m_count}/{l_count}, Found: {summary}")
                
                comparison_results['summary'] = {
                    'high': {'total': h_count, 'false_positives': summary.get('high', {}).get('false_positives', 0)},
                    'medium': {'total': m_count, 'false_positives': summary.get('medium', {}).get('false_positives', 0)},
                    'low': {'total': l_count, 'false_positives': summary.get('low', {}).get('false_positives', 0)},
                    'total_items': total_count
                }
                session['item_comparison_results'] = comparison_results
                session.modified = True
                
                # Update summary variable for response
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

@main_blueprint.route('/process-item-comparison', methods=['POST'])
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
        
        model = None
        if current_app.config.get('TRANSFORMER_MODEL_LOADED', False):
            model = current_app.config.get('TRANSFORMER_MODEL')
            if model:
                print("Using transformer model from app config")
            else:
                print("Warning: Transformer model is None despite being marked as loaded")
        else:
            print("Transformer model not loaded, using fallback method")


        # Process comparison
        from utils import process_item_comparisons
        comparison_results = process_item_comparisons(all_items)
        
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


@main_blueprint.route('/get-item-comparison-summary', methods=['GET'])
@login_required
def get_item_comparison_summary():
    """Get summary of item comparison results"""
    try:
        print("Getting item comparison summary...")
        
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
            print(f"Invalid comparison_results type: {type(comparison_results)}")
            return jsonify({
                'success': False,
                'message': 'Invalid comparison results format in session.'
            })
        
        print(f"Comparison keys available: {list(comparison_results.keys())}")
        
        # IMPORTANT: Count actual items in each confidence level
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
            
            print(f"REBUILDING SUMMARY - Current: {current_summary}, Actual counts: high={h_count}, medium={m_count}, low={l_count}")
            
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
            print(f"CORRECTED summary: {summary}")
        else:
            summary = current_summary
            print(f"Using VALID summary: {summary}")
        
        return jsonify({
            'success': True,
            'summary': summary
        })
        
    except Exception as e:
        print(f"Error in get_item_comparison_summary: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error getting summary: {str(e)}'
        })


@main_blueprint.route('/get-comparison-items/<confidence_level>', methods=['GET'])
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

@main_blueprint.route('/update-false-positives', methods=['POST'])
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
        print(f"Updated summary: {comparison_results['summary']}")
        
        return jsonify({
            'success': True, 
            'summary': summary
        })
        
    except Exception as e:
        print(f"Error in update_false_positives: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})


@main_blueprint.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    return redirect(url_for('auth.landing'))

@main_blueprint.route('/index')
@login_required
def index():
    return render_template('index.html')

@main_blueprint.route('/dashboard')
@login_required
def dashboard():
    # Check if we need to restart the process
    if request.args.get('restart'):
        session['current_step_id'] = 1
        session['completed_steps'] = []
        flash('Starting a new process', 'info')
    
    # Get current step and all steps
    current_step = current_app.get_current_step()
    all_steps = current_app.get_all_steps()
    
    return render_template('dashboard.html', current_step=current_step, steps=all_steps)

@main_blueprint.route('/goto-step/<int:step_id>', methods=['GET'])
@login_required
def goto_step(step_id):
    """Navigate to a specific step"""
    # Check if the requested step is accessible
    is_allowed, current_step_id, message = current_app.validate_step_progress(step_id)
    
    if is_allowed:
        # Update the current step in the session without affecting completion status
        session['current_step_id'] = step_id
        return redirect(url_for('main.dashboard'))
    else:
        flash(message, 'warning')
        return redirect(url_for('main.dashboard'))

@main_blueprint.route('/step/<int:step_id>')
@login_required
def step_view(step_id):
    # Validate step access
    is_allowed, current_id, message = current_app.validate_step_progress(step_id)
    
    if not is_allowed:
        flash(message, 'danger')
        return redirect(url_for('main.dashboard'))
    
    step_id = max(1, min(step_id, 8))  # Clamp between 1 and 8
    session['viewed_step_id'] = step_id  # Track which step is being viewed
    
    return render_template('dashboard.html')

@main_blueprint.route('/process-step/<int:step_id>', methods=['POST'])
@login_required
def process_step(step_id):
    # Validate if user can process this step
    is_allowed, current_id, message = current_app.validate_step_progress(step_id)
    
    if not is_allowed:
        flash(message, 'danger')
        return redirect(url_for('main.dashboard'))
    
    # Process step based on its ID
    success = False
    error_msg = None
    
    try:
        if step_id == 1:
            # Step 1: File validation
            # Check if we have validated data in the session
            if 'validated_data' not in session or 'column_mapping' not in session:
                raise ValueError("File validation incomplete. Please upload and validate a file first.")
            
            # Get validated data from session
            validated_data = session.get('validated_data')
            
            # If we have validated data, then the file has been processed successfully
            if validated_data:
                success = True
                flash("File validated successfully!", "success")
            else:
                raise ValueError("File validation failed or no validated data available.")
            
        elif step_id == 2:
            # Step 2: Duplication Overview - Real business logic
            if 'validated_data' not in session:
                raise ValueError("No validated data available. Please complete Step 1 first.")
            
            # Convert validated data from session back to dataframe
            validated_df = pd.DataFrame.from_dict(session['validated_data'])
            
            # Check if we have contract data in session
            if 'contract_duplicates' not in session:
                raise ValueError("No contract data available. Please process duplicates first.")
            
            contract_list = session['contract_duplicates']
            
            # Get duplication results from session
            if 'item_comparison_results' not in session:
                raise ValueError("Item comparison not completed. Please complete the comparison process first.")
            
            comparison_results = session['item_comparison_results']
            
            # Calculate true duplicates (total duplicates minus false positives)
            high_total = comparison_results.get('summary', {}).get('high', {}).get('total', 0)
            high_fp = comparison_results.get('summary', {}).get('high', {}).get('false_positives', 0)
            
            medium_total = comparison_results.get('summary', {}).get('medium', {}).get('total', 0)
            medium_fp = comparison_results.get('summary', {}).get('medium', {}).get('false_positives', 0)
            
            low_total = comparison_results.get('summary', {}).get('low', {}).get('total', 0)
            low_fp = comparison_results.get('summary', {}).get('low', {}).get('false_positives', 0)
            
            true_duplicates = (high_total - high_fp) + (medium_total - medium_fp) + (low_total - low_fp)
            
            # Find contracts that actually have true duplicates
            contracts_with_true_duplicates = set()
            
            # Check high confidence items
            for i, item in enumerate(comparison_results.get('high', [])):
                if not item.get('false_positive', False):
                    contracts_with_true_duplicates.add(item.get('Contract_Number', ''))
            
            # Check medium confidence items
            for i, item in enumerate(comparison_results.get('medium', [])):
                if not item.get('false_positive', False):
                    contracts_with_true_duplicates.add(item.get('Contract_Number', ''))
            
            # Check low confidence items
            for i, item in enumerate(comparison_results.get('low', [])):
                if not item.get('false_positive', False):
                    contracts_with_true_duplicates.add(item.get('Contract_Number', ''))
            
            # Count unique contracts with true duplicates
            total_contracts_with_duplicates = len(contracts_with_true_duplicates)
            
            # Set success flag for this step
            success = True
            
            # Handle flow based on true duplicates count
            if true_duplicates > 0:
                flash(f"Found {total_contracts_with_duplicates} contracts with {true_duplicates} true duplicate items.", "success")
            else:
                # No true duplicates found - auto-complete step 3
                flash("No true duplicates found after false positive review.", "info")
                
                # Mark step 3 as complete automatically
                current_app.mark_step_complete(3)
                
                # Update next step to be step 4
                session['current_step_id'] = 4
                session.modified = True
                
                flash("Step 3 (Duplication Resolution) automatically completed as no true duplicates were found.", "info")
        
        elif step_id == 3:
            # Step 3: Duplication Resolution
            
            # Check if item comparison was completed
            if 'item_comparison_results' not in session:
                raise ValueError("Item comparison not completed. Please complete Step 2 first.")

            comparison_results = session['item_comparison_results']
            if not comparison_results or 'summary' not in comparison_results:
                raise ValueError("Invalid comparison results. Please rerun the item comparison.")

            resolution_strategy = request.form.get('resolution_strategy')
            if not resolution_strategy:
                raise ValueError("No resolution strategy selected")
                
            # Process resolution (dummy implementation)
            time.sleep(1)  # Simulate processing time
            success = True
            
        elif step_id == 4:
            # Item master matching logic
            matching_method = request.form.get('matching_method')
            if not matching_method:
                raise ValueError("No matching method selected")
                
            # Process matching (dummy implementation)
            time.sleep(1)  # Simulate processing time
            success = True
            
        elif step_id == 5:
            # Change simulation logic
            # Process simulation approval
            time.sleep(1)  # Simulate processing time
            success = True
            
        elif step_id == 6:
            # Export changes logic
            export_format = request.form.get('export_format')
            if not export_format:
                raise ValueError("No export format selected")
                
            # Process export (dummy implementation)
            time.sleep(1)  # Simulate processing time
            success = True
            
        elif step_id == 7:
            # Synchronization inspection logic
            # Process synchronization verification
            time.sleep(1)  # Simulate processing time
            success = True
            
        elif step_id == 8:
            # Completion verification - no processing needed
            success = True
            
        # If successful, mark step as complete and advance
        if success:
            next_step = current_app.mark_step_complete(step_id)
            flash(f"Step {step_id} completed successfully!", "success")
            return redirect(url_for('main.dashboard'))
            
    except ValueError as e:
        error_msg = str(e)
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        
    if error_msg:
        flash(error_msg, "danger")
        
    return redirect(url_for('main.dashboard'))

@main_blueprint.route('/map-columns', methods=['POST'])
@login_required
def map_columns():
    """Save column mapping to session"""
    if 'file_info' not in session:
        return jsonify({'success': False, 'message': 'No file uploaded. Please upload a file first.'})
    
    # Get column mapping from POST data
    column_mapping = {}
    for field, column in request.form.items():
        if column:  # Only include fields that are mapped to a column
            column_mapping[field] = column
    
    # Validate required fields
    required_fields = ['Mfg Part Num', 'Vendor Part Num', 'Description', 
                       'Contract Price', 'UOM', 'QOE', 
                       'Effective Date', 'Expiration Date', 'Contract Number', 'ERP Vendor ID']
    
    missing_fields = [field for field in required_fields if field not in column_mapping and field != 'Buyer Part Num']
    
    if missing_fields:
        return jsonify({
            'success': False, 
            'message': f'Missing required field mapping: {", ".join(missing_fields)}'
        })
    
    # Store mapping in session
    session['column_mapping'] = column_mapping
    
    return jsonify({
        'success': True,
        'message': 'Column mapping saved successfully'
    })

@main_blueprint.route('/validate-file', methods=['POST'])
@login_required
def validate_file_route():
    """Validate the uploaded file columns against required fields"""
    if 'file_info' not in session:
        return jsonify({'success': False, 'message': 'No file uploaded. Please upload a file first.'})
    
    file_info = session['file_info']
    column_mapping = {}
    
    # Get column mapping from POST data
    for field, column in request.form.items():
        if column:  # Only include fields that are mapped to a column
            column_mapping[field] = column
    
    # Load the file
    file_path = os.path.join(current_app.static_folder, 'uploads', file_info['saved_name'])
    
    try:
        # Read file
        if file_info['type'] == 'csv':
            df = pd.read_csv(file_path, dtype=str)
        else:
            df = pd.read_excel(file_path, dtype=str)

        # Get duplicate checking mode from session or form
        duplicate_mode = request.form.get('duplicate_mode', session.get('duplicate_check_mode', 'default'))
        session['duplicate_check_mode'] = duplicate_mode
        
        # Validate the file
        valid_df, error_df, has_errors = validate_file(df, column_mapping, duplicate_mode)
        
        if isinstance(has_errors, str):
            # This means there was an error in the validation process
            return jsonify({'success': False, 'message': has_errors})
        
        # Save the full result dataframe for user download
        if has_errors:
           # Calculate statistics
            total_rows = int(len(error_df))  # Convert to Python int
            error_rows = int(error_df['Has Error'].sum())  # Convert to Python int
            duplicate_rows = int(len(error_df[(error_df['Warning-Potential Duplicates'] != '')]))  # Convert to Python int
            
            # Save the full result dataframe for user download
            error_file = save_error_file(error_df, current_user.id, file_info['name'])
            session['error_file'] = error_file
            
            # Convert error dataframe to HTML table for display
            # First, get only the rows with errors
            error_rows_df = error_df[error_df['Has Error']].copy()

            # Move the File Row column to the front for better visibility
            cols = ['File Row'] + [col for col in error_rows_df.columns if col != 'File Row']
            error_rows_df = error_rows_df[cols]

            # Generate HTML table properly by stripping out pandas default attributes
            html_content = error_rows_df.to_html(classes="", index=False, header=True)
            # Replace the entire table tag, not just part of it
            html_content = re.sub(r'<table[^>]*>', '', html_content)
            html_content = html_content.replace('</table>', '')
            error_table = f'<div class="table-responsive-container"><table id="error_table" class="table table-striped table-bordered">{html_content}</table></div>'
            
            return jsonify({
                'success': False, 
                'message': 'Validation failed. See the errors below. Review and make changes accordingly, then try again.',
                'error_table': error_table,
                'stats': {
                    'total_rows': total_rows,
                    'error_rows': error_rows,
                    'duplicate_rows': duplicate_rows
                }
            })
                
        # No errors, store validated dataframe in session
        # Since we can't store the dataframe directly in session, convert to dict
        session.pop('error_file', None)  # Clear any previous error file
        total_rows = len(valid_df)
        session['validated_data'] = valid_df.to_dict()
        
        return jsonify({
            'success': True,
            'message': 'File validation passed successfully!',
            'stats': {
                'total_rows': total_rows,
                'error_rows': 0,
                'duplicate_rows': 0
            },
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error validating file: {str(e)}'})

@main_blueprint.route('/download-error-file')
@login_required
def download_error_file():
    """Download the error file"""
    # Get the filename from the session
    if 'error_file' not in session:
        flash('No error file available for download', 'danger')
        return redirect(url_for('main.dashboard'))
    
    file_path = session['error_file']
    
    # Verify the file belongs to the current user
    expected_user_dir = f'user_{current_user.id}'
    if expected_user_dir not in file_path:
        flash('Access denied', 'danger')
        return redirect(url_for('main.dashboard'))
    
    # Get directory and filename
    directory = os.path.join(current_app.root_path, os.path.dirname(file_path))
    filename = os.path.basename(file_path)
    
    # Return the file
    return send_from_directory(directory=directory, path=filename, as_attachment=True)

@main_blueprint.route('/download-result-file')
@login_required
def download_result_file():
    """Download the result file"""
    # Get the filename from the session
    if 'result_file' not in session:
        flash('No result file available for download', 'danger')
        return redirect(url_for('main.dashboard'))
    
    file_path = session['result_file']
    
    # Verify the file belongs to the current user
    expected_user_dir = f'user_{current_user.id}'
    if expected_user_dir not in file_path:
        flash('Access denied', 'danger')
        return redirect(url_for('main.dashboard'))
    
    # Get directory and filename
    directory = os.path.join(current_app.root_path, os.path.dirname(file_path))
    filename = os.path.basename(file_path)
    
    # Return the file
    return send_from_directory(directory=directory, path=filename, as_attachment=True)

@main_blueprint.route('/upload-file', methods=['POST'])
@login_required
def upload_file():
    """Upload a file and return column headers"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file:
        # Ensure upload directory exists
        upload_dir = os.path.join(current_app.static_folder, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        saved_name = f"{int(time.time())}_{filename}"
        file_path = os.path.join(upload_dir, saved_name)
        file.save(file_path)
        
        # Determine file type
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        # Read headers
        try:
            if file_ext == 'csv':
                df = pd.read_csv(file_path, nrows=0)
            else:
                df = pd.read_excel(file_path, nrows=0)
                
            headers = df.columns.tolist()
            
            # Store file info in session
            session['file_info'] = {
                'name': filename,
                'saved_name': saved_name,
                'path': file_path,
                'type': file_ext
            }
            
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully',
                'filename': filename,
                'headers': headers
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error processing file: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'File upload failed'})

@main_blueprint.route('/get-duplicates', methods=['GET'])
@login_required
def get_duplicates():
    """Return duplicate data for display in the UI"""
    if 'duplicates' not in session or 'duplicate_table' not in session:
        return jsonify({
            'success': False,
            'message': 'No duplicate data available. Please complete Step 2 first.'
        })
    
    duplicates = session['duplicates']
    duplicate_table = session['duplicate_table']
    
    return jsonify({
        'success': True,
        'count': duplicates['count'],
        'keys': duplicates['keys'],
        'table': duplicate_table
    })

@main_blueprint.route('/download-template')
def download_template():
    """Download the upload template file"""
    try:
        # Use the same path structure as your UOM file in utils.py
        template_path = os.path.join(current_app.root_path, 'data', 'upload_template.xlsx')
        # Check if file exists
        if not os.path.exists(template_path):
            print(f"Template file not found at: {template_path}")
            flash("Template file not found. Please contact the administrator.", "error")
            return redirect(url_for('main.dashboard'))
        
        # Log that we found the file
        print(f"Serving template from: {template_path}")
        
        # Return the file as a download attachment
        return send_file(
            template_path,
            as_attachment=True,
            download_name='contract_price_template.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        # Log the exception
        print(f"Error downloading template: {str(e)}")
        flash(f"Error downloading template: {str(e)}", "error")
        return redirect(url_for('main.dashboard'))