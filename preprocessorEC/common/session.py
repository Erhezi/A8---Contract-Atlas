# Session management utilities
from flask import session

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
        
def store_comparison_results(results):
    """Store comparison results in session"""
    session['item_comparison_results'] = results
    session.modified = True
    
def get_comparison_results():
    """Get comparison results from session"""
    return session.get('item_comparison_results')

def update_true_duplicates_count():
    """Calculate and return true duplicates (total - false positives)"""
    if 'item_comparison_results' not in session:
        return 0
        
    comparison_results = session['item_comparison_results']
    
    # Calculate true duplicates (total duplicates minus false positives)
    high_total = comparison_results.get('summary', {}).get('high', {}).get('total', 0)
    high_fp = comparison_results.get('summary', {}).get('high', {}).get('false_positives', 0)
    
    medium_total = comparison_results.get('summary', {}).get('medium', {}).get('total', 0)
    medium_fp = comparison_results.get('summary', {}).get('medium', {}).get('false_positives', 0)
    
    low_total = comparison_results.get('summary', {}).get('low', {}).get('total', 0)
    low_fp = comparison_results.get('summary', {}).get('low', {}).get('false_positives', 0)
    
    true_duplicates = (high_total - high_fp) + (medium_total - medium_fp) + (low_total - low_fp)
    
    return true_duplicates

def get_contracts_with_true_duplicates():
    """Get list of contract numbers that have true duplicates (non-false positives)"""
    if 'item_comparison_results' not in session:
        return set()
        
    comparison_results = session['item_comparison_results']
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