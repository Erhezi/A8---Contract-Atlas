# common/model_loader.py
import pip_system_certs.wrapt_requests
import os
from flask import current_app

# Global model cache for backward compatibility
_MODEL_CACHE = {}

def load_transformer_model(app):
    """Load transformer model synchronously"""
    try:
        # Set flag to indicate loading is in progress
        app.config['TRANSFORMER_MODEL_LOADING'] = True
        print("Pre-loading sentence transformer model...")
        from sentence_transformers import SentenceTransformer
        
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
    
    if app.config['TRANSFORMER_MODEL_LOADED']:
        print(f"Model loaded in cache {id(model)}")  # debugging line

def get_sentence_transformer_model():
    """Get the sentence transformer model from app config or local cache if available"""
    from flask import current_app
    
    # Try to get model from app config first
    if current_app.config.get('TRANSFORMER_MODEL_LOADED', False):
        model = current_app.config.get('TRANSFORMER_MODEL')
        if model:
            return model
    
    # Fall back to module cache if not in app config
    return _MODEL_CACHE.get('transformer_model')