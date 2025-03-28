from flask_login import UserMixin

# Simple in-memory user store for demo purposes
USERS = {
    'admin': {'id': 'admin', 'password': 'admin123'}
}

class User(UserMixin):
    def __init__(self, id):
        self.id = id

    @staticmethod
    def get(user_id):
        if user_id in USERS:
            return User(user_id)
        return None

    def check_password(self, password):
        return USERS.get(self.id, {}).get('password') == password