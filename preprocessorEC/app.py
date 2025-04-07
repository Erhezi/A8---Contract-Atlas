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

# Initialize the model cache
from utils import _MODEL_CACHE
from threading import Thread

# Initialize app config with model status
app.config['TRANSFORMER_MODEL_LOADING'] = False
app.config['TRANSFORMER_MODEL_LOADED'] = False

def load_transformer_model():
    """Load transformer model in background thread"""
    try:
        # Set flag to indicate loading is in progress
        app.config['TRANSFORMER_MODEL_LOADING'] = True
        print("Pre-loading sentence transformer model...")
        from sentence_transformers import SentenceTransformer
        import os
        
        # Define path to local model storage
        local_model_path = os.path.join(app.root_path, 'models', 'all-MiniLM-L6-v2')
        
        # Check if model exists locally
        if os.path.exists(local_model_path):
            print("Loading model from local path...")
            model = SentenceTransformer(local_model_path)
            # Store in both places for backward compatibility
            _MODEL_CACHE['transformer_model'] = model
            app.config['TRANSFORMER_MODEL'] = model
            app.config['TRANSFORMER_MODEL_LOADED'] = True
            print("Sentence transformer model loaded successfully from local path")
        else:
            # First time: download and save the model
            print("Downloading model and saving to local path...")
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            model = SentenceTransformer('all-MiniLM-L6-v2')
            model.save(local_model_path)
            # Store in both places for backward compatibility
            _MODEL_CACHE['transformer_model'] = model
            app.config['TRANSFORMER_MODEL'] = model
            app.config['TRANSFORMER_MODEL_LOADED'] = True
            print("Sentence transformer model downloaded and loaded successfully")
    except ImportError:
        print("Warning: sentence-transformers package not available. Using fallback similarity.")
        _MODEL_CACHE['transformer_model'] = None
        app.config['TRANSFORMER_MODEL'] = None
        app.config['TRANSFORMER_MODEL_LOADED'] = False
        print("Warning: Fallback similarity will be used. This may affect performance.")
    except Exception as e:
        print(f"Error loading transformer model: {str(e)}")
        _MODEL_CACHE['transformer_model'] = None
        app.config['TRANSFORMER_MODEL'] = None
        app.config['TRANSFORMER_MODEL_LOADED'] = False
    finally:
        # Mark loading as complete (success or failure)
        app.config['TRANSFORMER_MODEL_LOADING'] = False
    
    print(f"model loaded in cache {id(model)}") # debugging line TEST TEST TEST

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
    print("Loading transformer model synchronously before starting the application...")
    # Load model synchronously instead of in a thread
    load_transformer_model()
    
    print(f"Model loading complete. Status: {app.config['TRANSFORMER_MODEL_LOADED']}")
    print("Starting Flask application...")
    app.run(debug=True)