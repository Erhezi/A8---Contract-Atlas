# common/workflow.py
# Common workflow functionality
from flask import session
from flask_login import current_user

def inject_workflow_steps():
    """Inject workflow steps into templates."""
    workflow_steps = [
        {'id': 1, 'name': 'Pre-Checking'},
        {'id': 2, 'name': 'Duplication Overview'},
        {'id': 3, 'name': 'Duplication Resolution'},
        {'id': 4, 'name': 'Item Master Matching'},
        {'id': 5, 'name': 'Change Simulation'},
        {'id': 6, 'name': 'Export Changes'},
        {'id': 7, 'name': 'Synchronization Inspection'},
        {'id': 8, 'name': 'Completion Verified'}
    ]
    
    # Determine current step if authenticated
    current_step = None
    current_step_index = 0
    completed_steps = []
    
    if current_user.is_authenticated:
        current_step_id = session.get('current_step_id', 1)
        completed_steps = session.get('completed_steps', [])
        
        # Find the current step object
        current_step = next((step for step in workflow_steps if step['id'] == current_step_id), workflow_steps[0])
        current_step_index = current_step_id
    
    return {
        'workflow_steps': workflow_steps,
        'current_step': current_step,
        'current_step_index': current_step_index,
        'completed_steps': completed_steps
    }