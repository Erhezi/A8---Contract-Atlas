# Session management utilities
from flask import session, current_app
from flask_login import current_user

# --- Generic Helpers (Can remain for non-user/workflow specific data) ---
def get_session_data(key, default=None):
    """Get data from session with a default value if not found"""
    return session.get(key, default)

def store_session_data(key, value):
    """Store data in session and mark as modified"""
    session[key] = value
    session.modified = True

def clear_session_data(key):
    """Remove data from session if it exists"""
    if key in session:
        session.pop(key)
        session.modified = True

# --- User Specific Helpers ---

# --- File Processing Data ---
def store_file_info(user_id, info_dict):
    """Store file information for a user."""
    key = f'file_info_{user_id}'
    session[key] = info_dict
    session.modified = True

def get_file_info(user_id):
    """Get file information for a user."""
    key = f'file_info_{user_id}'
    return session.get(key)

def store_column_mapping(user_id, mapping):
    """Store the column mapping for a user."""
    key = f'column_mapping_{user_id}'
    session[key] = mapping
    session.modified = True

def get_column_mapping(user_id):
    """Get the column mapping for a user."""
    key = f'column_mapping_{user_id}'
    return session.get(key)

def store_validated_data(user_id, data_list):
    """Store validated data (as list of dicts) for a user."""
    key = f'validated_data_{user_id}'
    session[key] = data_list
    session.modified = True

def get_validated_data(user_id):
    """Get validated data (as list of dicts) for a user."""
    key = f'validated_data_{user_id}'
    return session.get(key)

def store_error_file_path(user_id, path):
    """Store the path to the error file for a user."""
    key = f'error_file_path_{user_id}'
    session[key] = path
    session.modified = True

def get_error_file_path(user_id):
    """Get the path to the error file for a user."""
    key = f'error_file_path_{user_id}'
    return session.get(key)

def clear_error_file_path(user_id):
    """Clear the error file path for a specific user."""
    key = f'error_file_path_{user_id}'
    if key in session:
        session.pop(key)
        session.modified = True

# --- Duplicate Detection Data ---
def store_temp_table_name(user_id, table_name):
    """Store the temporary table name for a user."""
    key = f'temp_contract_table_{user_id}'
    session[key] = table_name
    session.modified = True

def get_temp_table_name(user_id):
    """Get the temporary table name for a user."""
    key = f'temp_contract_table_{user_id}'
    return session.get(key)

def store_contract_duplicates(user_id, contract_list):
    """Store the initial contract duplicate list for a user."""
    key = f'contract_duplicates_{user_id}'
    session[key] = contract_list
    session.modified = True

def get_contract_duplicates(user_id):
    """Get the initial contract duplicate list for a user."""
    key = f'contract_duplicates_{user_id}'
    return session.get(key)

def store_included_contracts(user_id, included_list):
    """Store the list of included contracts for a user."""
    key = f'included_contracts_{user_id}'
    session[key] = included_list
    session.modified = True

def get_included_contracts(user_id):
    """Get the list of included contracts for a user."""
    key = f'included_contracts_{user_id}'
    return session.get(key, []) # Default to empty list

def store_excluded_contracts(user_id, excluded_list):
    """Store the list of excluded contracts for a user."""
    key = f'excluded_contracts_{user_id}'
    session[key] = excluded_list
    session.modified = True

def get_excluded_contracts(user_id):
    """Get the list of excluded contracts for a user."""
    key = f'excluded_contracts_{user_id}'
    return session.get(key, []) # Default to empty list

def store_comparison_results(user_id, results):
    """Store comparison results in session for a specific user."""
    key = f'item_comparison_results_{user_id}'
    session[key] = results
    session.modified = True

def get_comparison_results(user_id):
    """Get comparison results from session for a specific user."""
    key = f'item_comparison_results_{user_id}'
    return session.get(key)

def clear_comparison_results(user_id):
    """Clear comparison results for a specific user."""
    key = f'item_comparison_results_{user_id}'
    if key in session:
        session.pop(key)
        session.modified = True

def store_deduplication_results(user_id, results):
    """Store deduplication results in session for a specific user."""
    key = f'deduplication_results_{user_id}'
    session[key] = results
    session.modified = True

def get_deduped_results(user_id):
    """Get deduplication results from session for a specific user."""
    key = f'deduplication_results_{user_id}'
    return session.get(key)

def clear_deduped_results(user_id):
    """Clear deduplication results for a specific user."""
    key = f'deduplication_results_{user_id}'
    if key in session:
        session.pop(key)
        session.modified = True

def store_infor_cl_matches(user_id, match_list):
    """Store the Infor Contract Line match results for a user."""
    key = f'infor_cl_matches_{user_id}'
    session[key] = match_list
    session.modified = True

def get_infor_cl_matches(user_id):
    """Get the Infor Contract Line match results for a user."""
    key = f'infor_cl_matches_{user_id}'
    return session.get(key)

def clear_infor_cl_matches(user_id):
    """Clear the Infor Contract Line match results for a user."""
    key = f'infor_cl_matches_{user_id}'
    if key in session:
        session.pop(key)
        session.modified = True

# --- Workflow State ---
def store_current_step(user_id, step_id):
    """Store the user's current step ID."""
    key = f'current_step_id_{user_id}'
    session[key] = step_id
    session.modified = True

def get_current_step_from_session(user_id):
    """Get the user's current step ID from session."""
    key = f'current_step_id_{user_id}'
    return session.get(key, 1) # Default to step 1

def store_completed_steps(user_id, steps_list):
    """Store the list of completed steps for a user."""
    key = f'completed_steps_{user_id}'
    session[key] = steps_list
    session.modified = True

def get_completed_steps(user_id):
    """Get the list of completed steps for a user."""
    key = f'completed_steps_{user_id}'
    return session.get(key, []) # Default to empty list

def store_error_file_path(user_id, path):
    """Store the path to the error file for a user."""
    key = f'error_file_path_{user_id}'
    session[key] = path
    session.modified = True

def get_error_file_path(user_id):
    """Get the path to the error file for a user."""
    key = f'error_file_path_{user_id}'
    return session.get(key)

def store_uploaded_file_path(user_id, path):
    """Store the path to the uploaded file for a user."""
    key = f'uploaded_file_path_{user_id}'
    session[key] = path
    session.modified = True

def get_uploaded_file_path(user_id):
    """Get the path to the uploaded file for a user."""
    key = f'uploaded_file_path_{user_id}'
    return session.get(key)

def store_column_mapping(user_id, mapping):
    """Store the column mapping for a user."""
    key = f'column_mapping_{user_id}'
    session[key] = mapping
    session.modified = True

def get_column_mapping(user_id):
    """Get the column mapping for a user."""
    key = f'column_mapping_{user_id}'
    return session.get(key)

# --- Functions that calculate based on session data ---

def update_true_duplicates_count(user_id):
    """Calculate and return true duplicates (total - false positives) for a user."""
    comparison_results = get_comparison_results(user_id) # Use helper
    if not comparison_results:
        return 0

    # Calculate true duplicates (total duplicates minus false positives)
    high_total = comparison_results.get('summary', {}).get('high', {}).get('total', 0)
    high_fp = comparison_results.get('summary', {}).get('high', {}).get('false_positives', 0)

    medium_total = comparison_results.get('summary', {}).get('medium', {}).get('total', 0)
    medium_fp = comparison_results.get('summary', {}).get('medium', {}).get('false_positives', 0)

    low_total = comparison_results.get('summary', {}).get('low', {}).get('total', 0)
    low_fp = comparison_results.get('summary', {}).get('low', {}).get('false_positives', 0)

    true_duplicates = (high_total - high_fp) + (medium_total - medium_fp) + (low_total - low_fp)

    return true_duplicates

def get_contracts_with_true_duplicates(user_id):
    """Get list of contract numbers that have true duplicates (non-false positives) for a user."""
    comparison_results = get_comparison_results(user_id) # Use helper
    if not comparison_results:
        return set()

    contracts_with_true_duplicates = set()

    # Check high confidence items
    for item in comparison_results.get('high', []):
        if not item.get('false_positive', False):
            contracts_with_true_duplicates.add(item.get('contract_number_ccx', ''))

    # Check medium confidence items
    for item in comparison_results.get('medium', []):
        if not item.get('false_positive', False):
            contracts_with_true_duplicates.add(item.get('contract_number_ccx', ''))

    # Check low confidence items
    for item in comparison_results.get('low', []):
        if not item.get('false_positive', False):
            contracts_with_true_duplicates.add(item.get('contract_number_ccx', ''))

    return contracts_with_true_duplicates