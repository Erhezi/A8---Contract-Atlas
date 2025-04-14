# Application Factory for preprocessorEC
from flask import Flask, redirect, url_for, session, flash, request
from flask_login import LoginManager, current_user
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from flask_session import Session
import tempfile
import os
from datetime import timedelta
import importlib.util


def create_app(config_name=None, test_config=None):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Load configuration - allow for test config override
    if test_config is None:
        # Load the configuration based on environment
        from .config import get_config
        config = get_config(config_name)
        app.config.from_object(config)
        print(f"Using configuration: {config.__class__.__name__}")
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)
        print("Using test configuration")
    
    # Ensure session directory exists
    os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
    
    print(f"App root path: {app.root_path}")
    Session(app)  # Initialize the session

    # Initialize the database engine with connection pooling
    app.config['DB_ENGINE'] = create_engine(
        app.config['DB_CONN_STRING'],
        poolclass=QueuePool,
        pool_size=app.config.get('DB_POOL_SIZE', 10),
        max_overflow=app.config.get('DB_MAX_OVERFLOW', 20),
        pool_pre_ping=app.config.get('DB_POOL_PRE_PING', True)
    )

    # Set up login manager
    login_manager = LoginManager()
    login_manager.login_view = 'auth.landing'  # Redirect to landing page instead of login
    login_manager.init_app(app)

    # User loader stub
    from .models import User
    @login_manager.user_loader
    def load_user(user_id):
        return User.get(user_id)

    # Import and register blueprints
    from .auth import auth_blueprint
    from .common import common_bp
    from .file_processing import file_bp
    from .duplicate_detection import duplicate_bp
    
    app.register_blueprint(auth_blueprint)
    app.register_blueprint(common_bp)
    app.register_blueprint(file_bp, url_prefix='/file-processing')
    app.register_blueprint(duplicate_bp, url_prefix='/duplicate-detection')

    # Initialize model management
    # Initialize app config with model status
    app.config['TRANSFORMER_MODEL_LOADING'] = False
    app.config['TRANSFORMER_MODEL_LOADED'] = False
    
    # Import and register StepManager
    from .steps import StepManager
    step_manager = StepManager()
    
    # Make step functions available to the app
    app.validate_step_progress = step_manager.validate_step_progress
    app.mark_step_complete = step_manager.mark_step_complete
    app.get_current_step = step_manager.get_current_step
    app.get_all_steps = step_manager.get_all_steps
    
    # Register validators with the jinja environment
    app.jinja_env.globals.update(validate_step_progress=step_manager.validate_step_progress)
    app.jinja_env.globals.update(mark_step_complete=step_manager.mark_step_complete)
    app.jinja_env.globals.update(get_current_step=step_manager.get_current_step)
    app.jinja_env.globals.update(get_all_steps=step_manager.get_all_steps)

    @app.context_processor
    def inject_workflow_steps():
        """Inject workflow steps into templates."""
        # Get all steps with their status
        workflow_steps = step_manager.get_all_steps()
        
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
    
    # Basic routes
    @app.route('/')
    def index():
        return redirect(url_for('common.home'))
    
    # Load models
    with app.app_context():
        def load_transformer_model(app):
            """Load sentence transformer model"""
            try:
                # Set flag to indicate loading is in progress
                app.config['TRANSFORMER_MODEL_LOADING'] = True
                print("Pre-loading sentence transformer model...")
                from sentence_transformers import SentenceTransformer
                
                # Define path to local model storage
                model_name = app.config.get('MODEL_NAME', 'all-MiniLM-L6-v2')
                local_model_path = os.path.join(app.root_path, 'models', model_name)
                
                # Check if model exists locally
                if os.path.exists(local_model_path):
                    print("Loading model from local path...")
                    model = SentenceTransformer(local_model_path)
                    app.config['TRANSFORMER_MODEL'] = model
                    app.config['TRANSFORMER_MODEL_LOADED'] = True
                    print("Sentence transformer model loaded successfully from local path")
                else:
                    # First time: download and save the model
                    print("Downloading model and saving to local path...")
                    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
                    model = SentenceTransformer(model_name)
                    model.save(local_model_path)
                    app.config['TRANSFORMER_MODEL'] = model
                    app.config['TRANSFORMER_MODEL_LOADED'] = True
                    print("Sentence transformer model downloaded and loaded successfully")
            except ImportError:
                print("Warning: sentence-transformers package not available. Using fallback similarity.")
                app.config['TRANSFORMER_MODEL'] = None
                app.config['TRANSFORMER_MODEL_LOADED'] = False
                print("Warning: Fallback similarity will be used. This may affect performance.")
            except Exception as e:
                print(f"Error loading transformer model: {str(e)}")
                app.config['TRANSFORMER_MODEL'] = None
                app.config['TRANSFORMER_MODEL_LOADED'] = False
            finally:
                # Mark loading as complete (success or failure)
                app.config['TRANSFORMER_MODEL_LOADING'] = False
        
        # Loading of model can be enabled/disabled here
        # For development, we might defer loading until needed
        load_transformer_model(app)
        print(f"Model loading status: {app.config['TRANSFORMER_MODEL_LOADED']}")
        # print(f"TRANSFORMER_MODEL_LOADED: {app.config.get('TRANSFORMER_MODEL_LOADED')}")
        # print(f"TRANSFORMER_MODEL: {app.config.get('TRANSFORMER_MODEL')}")

    return app