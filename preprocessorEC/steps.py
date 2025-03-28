from flask import session

class StepManager:
    def __init__(self):
        self.steps = [
            {'id': 1, 'name': 'File Pre-Checking', 'description': 'Validate file format and contents'},
            {'id': 2, 'name': 'Duplication Overview', 'description': 'Identify duplicate items in the data'},
            {'id': 3, 'name': 'Resolve Duplications', 'description': 'Select resolution strategy for duplicates'},
            {'id': 4, 'name': 'Item Master Matching', 'description': 'Match items to the existing catalog'},
            {'id': 5, 'name': 'Change Simulation', 'description': 'Preview proposed changes before applying'},
            {'id': 6, 'name': 'Export Changes', 'description': 'Export changes to target systems'},
            {'id': 7, 'name': 'Synchronization', 'description': 'Verify synchronization with target systems'},
            {'id': 8, 'name': 'Completion', 'description': 'Final verification and completion'}
        ]
    
    def validate_step_progress(self, step_id):
        """Check if user can access a specific step"""
        # Initialize current step to 1 if not set
        if 'current_step_id' not in session:
            session['current_step_id'] = 1
            
        # Initialize completed steps if not set
        if 'completed_steps' not in session:
            session['completed_steps'] = []
            
        current_id = session.get('current_step_id', 1)
        completed_steps = session.get('completed_steps', [])
        
        # User can access a step if:
        # 1. It's the current step
        # 2. It's a completed step
        # 3. It's the next step after the current step
        if step_id == current_id or step_id in completed_steps or step_id == current_id + 1:
            return True, current_id, "Access allowed"
        else:
            return False, current_id, f"You must complete previous steps before accessing Step {step_id}"
    
    def mark_step_complete(self, step_id):
        """Mark a step as complete and advance to the next step"""
        # Make sure completed_steps exists
        if 'completed_steps' not in session:
            session['completed_steps'] = []
            
        # Add to completed steps if not already there
        if step_id not in session['completed_steps']:
            session['completed_steps'].append(step_id)
            
        # If this is the current step, advance to the next step
        if session.get('current_step_id', 1) == step_id:
            next_step_id = min(step_id + 1, 8)  # Limit to max step 8
            session['current_step_id'] = next_step_id
            return next_step_id
            
        return session.get('current_step_id', 1)
        
    def get_current_step(self):
        """Get the current step object"""
        step_id = session.get('current_step_id', 1)
        return next((step for step in self.steps if step['id'] == step_id), self.steps[0])
        
    def get_step_status(self, step_id):
        """Get the status of a step (completed, current, future)"""
        current_id = session.get('current_step_id', 1)
        completed_steps = session.get('completed_steps', [])
        
        if step_id in completed_steps:
            return "completed"
        elif step_id == current_id:
            return "current"
        else:
            return "future"
        
    def get_all_steps(self):
        """Get all steps with their status"""
        result = []
        for step in self.steps:
            step_copy = step.copy()
            step_copy['status'] = self.get_step_status(step['id'])
            result.append(step_copy)
        return result