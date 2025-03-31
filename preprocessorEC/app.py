from flask import Flask, redirect, url_for, session, flash, request
from flask_login import LoginManager, current_user
from auth import auth_blueprint
from routes import main_blueprint
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from flask_session import Session
import tempfile
import os
from datetime import timedelta


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure key in production

# Configure session to use filesystem (or Redis, etc.)
app.config['SESSION_TYPE'] = 'filesystem'   
app.config['SESSION_FILE_DIR'] = tempfile.gettempdir()  # Use temp directory for session files
app.config['SESSION_PERMANENT'] = False  # Session will expire when the browser is closed
app.config['SESSION_USE_SIGNER'] = True  # Sign the session cookie
app.config['SESSION_FILE_DIR'] = os.path.join(app.root_path, 'temp_files', 'flask_session')  # Directory for session files
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
print(app.root_path)
Session(app)  # Initialize the session

# Initialize the database engine with connection pooling
app.config['DB_ENGINE'] = create_engine(
    'mssql+pyodbc:///?odbc_connect=' + 
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=MISCPrdAdhocDB;'
    'DATABASE=PRIME;'
    'Trusted_Connection=yes;',
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

# Set up login manager
login_manager = LoginManager()
login_manager.login_view = 'auth.landing'  # Redirect to landing page instead of login
login_manager.init_app(app)

# User loader stub
from models import User
@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# Register blueprints
app.register_blueprint(auth_blueprint)

# Step validation functions
def validate_step_progress(requested_step):
    """
    Validates if a user can access the requested step based on their progress.
    Returns (is_allowed, current_step, message)
    """
    # Get the current completed step from session
    completed_steps = session.get('completed_steps', [])
    current_step_id = session.get('current_step_id', 1)
    
    # If no steps completed, initialize with first step
    if not completed_steps:
        session['completed_steps'] = []
        session['current_step_id'] = 1
        
    # User can access completed steps or the next step in sequence
    if requested_step <= current_step_id:
        return True, current_step_id, None
    else:
        message = f"You must complete step {current_step_id} before accessing step {requested_step}."
        return False, current_step_id, message

def mark_step_complete(step_id):
    """Mark a step as complete and advance to the next step"""
    # Get current completed steps
    completed_steps = session.get('completed_steps', [])
    
    # Add current step to completed if not already there
    if step_id not in completed_steps:
        completed_steps.append(step_id)
        session['completed_steps'] = completed_steps
    
    # Advance current step if this is the current step
    if session.get('current_step_id') == step_id:
        session['current_step_id'] = min(step_id + 1, 8)  # Don't go beyond last step
        
    return session['current_step_id']

def get_current_step():
    """Get the current step object"""
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
    
    current_step_id = session.get('current_step_id', 1)
    return next((step for step in workflow_steps if step['id'] == current_step_id), workflow_steps[0])

def get_all_steps():
    """Get all steps with their status"""
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
    
    current_step_id = session.get('current_step_id', 1)
    completed_steps = session.get('completed_steps', [])
    
    # Add status to each step
    for step in workflow_steps:
        if step['id'] in completed_steps:
            step['status'] = 'completed'
        elif step['id'] == current_step_id:
            step['status'] = 'current'
        else:
            step['status'] = 'future'
    
    return workflow_steps

# Register validators with the app context
app.jinja_env.globals.update(validate_step_progress=validate_step_progress)
app.jinja_env.globals.update(mark_step_complete=mark_step_complete)
app.jinja_env.globals.update(get_current_step=get_current_step)
app.jinja_env.globals.update(get_all_steps=get_all_steps)

# Make the functions available to the app object directly
app.validate_step_progress = validate_step_progress
app.mark_step_complete = mark_step_complete
app.get_current_step = get_current_step  # Add this line
app.get_all_steps = get_all_steps  # Add this line

app.register_blueprint(main_blueprint)


@app.context_processor
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

if __name__ == '__main__':
    app.run(debug=True)
