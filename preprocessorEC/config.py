import os
import tempfile
from datetime import timedelta

class Config:
    """Base configuration class. Contains default settings."""
    # General settings
    SECRET_KEY = 'your_secret_key_here'  # Replace with a secure key in production
    
    # Session configuration
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Configure database connection
    DB_POOL_SIZE = 10
    DB_MAX_OVERFLOW = 20
    DB_POOL_PRE_PING = True
    
    # Model settings
    MODEL_NAME = 'all-MiniLM-L6-v2'
    
    @property
    def DB_CONN_STRING(self):
        """Default database connection string"""
        return ('mssql+pyodbc:///?odbc_connect=' + 
                'DRIVER={ODBC Driver 17 for SQL Server};'
                'SERVER=MISCPrdAdhocDB;'
                'DATABASE=PRIME;'
                'Trusted_Connection=yes;')
                
    @property
    def SESSION_FILE_DIR(self):
        """Directory for session files"""
        return os.path.join(
            os.path.abspath(os.path.dirname(__file__)), 
            'temp_files', 
            'flask_session'
        )
        
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    TESTING = False
    
    # Additional dev-specific settings
    @property
    def DB_CONN_STRING(self):
        """Development database connection string - can be overridden for local dev"""
        return super().DB_CONN_STRING
    
class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = False
    TESTING = True
    
    # Use in-memory or temporary database for testing
    @property
    def DB_CONN_STRING(self):
        """Test database connection string - in-memory SQLite for fast tests"""
        return 'sqlite:///:memory:'
        
    @property
    def SESSION_FILE_DIR(self):
        """Use a temporary directory for testing sessions"""
        return os.path.join(tempfile.gettempdir(), 'contract_atlas_test_sessions')
    
class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    TESTING = False
    
    # In production, use environment variables for sensitive values
    @property
    def SECRET_KEY(self):
        return os.environ.get('SECRET_KEY', super().SECRET_KEY)
    
    # Production database settings can come from environment variables
    @property
    def DB_CONN_STRING(self):
        driver = os.environ.get('DB_DRIVER', 'ODBC Driver 17 for SQL Server')
        server = os.environ.get('DB_SERVER', 'MISCPrdAdhocDB')
        database = os.environ.get('DB_NAME', 'PRIME')
        
        # Use connection string with environment variables
        return ('mssql+pyodbc:///?odbc_connect=' + 
                f'DRIVER={{{driver}}};'
                f'SERVER={server};'
                f'DATABASE={database};'
                'Trusted_Connection=yes;')

def get_config(config_name='default'):
    """Return the appropriate configuration object based on the environment"""
    config_map = {
        'development': DevelopmentConfig,
        'testing': TestingConfig,
        'production': ProductionConfig,
        'default': DevelopmentConfig
    }
    
    # Get configuration class or default to development
    return config_map.get(config_name, DevelopmentConfig)()