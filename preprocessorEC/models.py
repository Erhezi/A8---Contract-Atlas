# User model
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask import current_app
import uuid

# In-memory user storage for development
# In production, this would be replaced with a database
_USERS = {
    # Default admin user
    'admin': {
        'id': 'admin',
        'username': 'admin',
        'email': 'admin@example.com',
        'name': 'Administrator',
        'password_hash': generate_password_hash('admin'),  # Password: admin -- in real we need to hash it
        'role': 'admin'
    },
    # add another user for testing purpose
    'testuser': {
        'id': 'testuser',
        'username': 'testuser',
        'email': 'dli2@montefiore.org',
        'name': 'Test User',
        'password_hash': generate_password_hash('testuser'),  # Password: testuser -- in real we need to hash it
        'role': 'user'
    }
}

class User(UserMixin):
    """User model for authentication"""
    
    def __init__(self, username, email, name="", password_hash="", role="user"):
        self.id = username
        self.username = username
        self.email = email
        self.name = name
        self.password_hash = password_hash
        self.role = role
        
    def save(self):
        """Save user to storage"""
        try:
            _USERS[self.username] = {
                'id': self.username,
                'username': self.username,
                'email': self.email,
                'name': self.name,
                'password_hash': self.password_hash,
                'role': self.role
            }
            return True
        except Exception as e:
            print(f"Error saving user: {str(e)}")
            return False
            
    @classmethod
    def get(cls, user_id):
        """Get user by ID"""
        user_data = _USERS.get(user_id)
        if user_data:
            return cls(
                username=user_data['username'],
                email=user_data['email'],
                name=user_data.get('name', ''),
                password_hash=user_data['password_hash'],
                role=user_data.get('role', 'user')
            )
        return None
        
    @classmethod
    def get_by_username(cls, username):
        """Get user by username"""
        return cls.get(username)
        
    @classmethod
    def get_all(cls):
        """Get all users"""
        return [cls.get(user_id) for user_id in _USERS.keys()]
    
    @classmethod
    def check_password(cls, username, password):
        """Check user password"""
        user = cls.get_by_username(username)
        if user and user.password_hash:
            return check_password_hash(user.password_hash, password)
        return False