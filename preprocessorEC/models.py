"""User model and persistence layer.

Replaces previous in-memory _USERS dict by integrating with SQL Server
table [DM_MONTYNT\\dli2].PreprocessorUser. Retains the in-memory map as a
fallback (e.g. during early startup or if DB temporarily unavailable).

Schema (as provided):
    create table PreprocessorUser (
        user_id int IDENTITY(1, 100) PRIMARY KEY not null,
        email varchar(255) not null,
        [id] varchar(120) not null,
        username varchar(120) not null,
        password_hash varchar(255) not null,
        [role] varchar(50) not null default 'sourcing',
        created_at datetime default getdate(),
        last_login_at datetime null,
        reset_code varchar(10) null,
        reset_code_expiry datetime null
    )

We treat column [id] as the stable external identifier (same value as
username for now), and keep user_id as the internal surrogate key.
"""

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask import current_app
from datetime import datetime
from typing import Optional, Tuple

# Fallback in-memory users (legacy / dev only)
_USERS = {
    'admin': {
        'id': 'admin',
        'username': 'admin',
        'email': 'admin@example.com',
        'name': 'Administrator',
        'password_hash': generate_password_hash('admin'),
        'role': 'admin'
    },
    'testuser': {
        'id': 'testuser',
        'username': 'testuser',
        'email': 'dli2@montefiore.org',
        'name': 'Test Sourcing User',
        'password_hash': generate_password_hash('testuser'),
        'role': 'sourcing'
    },
    'testmdm': {
        'id': 'testmdm',
        'username': 'testmdm',
        'email': 'xyzmdm@montefiore.org',
        'name': 'Test MDM User',
        'password_hash': generate_password_hash('testmdm'),
        'role': 'mdm'
    }
}

class User(UserMixin):
    """User model providing DB-backed persistence."""

    def __init__(self, username: str, email: str, name: str = "", password_hash: str = "", role: str = "user", user_id: Optional[int] = None):
        # Flask-Login uses .id as unique identifier; keep aligned with username (legacy behavior)
        self.id = username
        self.user_id = user_id  # surrogate key in table
        self.username = username
        self.email = email
        self.name = name or ""
        self.password_hash = password_hash
        self.role = role or "admin"

    # --- Internal helpers ---
    @staticmethod
    def _get_connection():
        try:
            engine = current_app.config.get('DB_ENGINE')
            if not engine:
                return None
            return engine.raw_connection()
        except Exception as e:
            print(f"User model DB connection error: {e}")
            return None

    @classmethod
    def from_row(cls, row):
        if not row:
            return None
        return cls(
            username=row['username'],
            email=row['email'],
            name=row.get('name', ''),  # name not in table currently; safe fallback
            password_hash=row['password_hash'],
            role=row.get('role', 'sourcing'),
            user_id=row.get('user_id')
        )

    # --- CRUD operations ---
    def save(self) -> bool:
        """Insert a new user row into PreprocessorUser.

        If DB unavailable, falls back to in-memory storage.
        """
        conn = self._get_connection()
        if conn is None:
            # fallback
            _USERS[self.username] = {
                'id': self.username,
                'username': self.username,
                'email': self.email,
                'name': self.name,
                'password_hash': self.password_hash,
                'role': self.role
            }
            return True
        try:
            cursor = conn.cursor()
            # Table does not have name column; we only store provided schema fields.
            # We'll store id same as username for now.
            cursor.execute(
                """
                INSERT INTO [DM_MONTYNT\\dli2].PreprocessorUser (email, [id], username, password_hash, [role])
                VALUES (?, ?, ?, ?, ?)
                """,
                (self.email, self.username, self.username, self.password_hash, self.role)
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error inserting user: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
            return False
        finally:
            try:
                conn.close()
            except Exception:
                pass

    @classmethod
    def get(cls, user_id: str):
        """Get user by external id (maps to column [id] / username)."""
        # DB first
        conn = cls._get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT user_id, email, [id], username, password_hash, [role] FROM [DM_MONTYNT\\dli2].PreprocessorUser WHERE [id]=? OR username=?",
                    (user_id, user_id)
                )
                row = cursor.fetchone()
                if row:
                    columns = [c[0] for c in cursor.description]
                    data = dict(zip(columns, row))
                    return cls.from_row(data)
            except Exception as e:
                print(f"User.get DB error: {e}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        # fallback
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
    def get_by_username(cls, username: str):
        return cls.get(username)

    @classmethod
    def get_by_email(cls, email: str):
        conn = cls._get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT user_id, email, [id], username, password_hash, [role] FROM [DM_MONTYNT\\dli2].PreprocessorUser WHERE email=?",
                    (email,)
                )
                row = cursor.fetchone()
                if row:
                    cols = [c[0] for c in cursor.description]
                    return cls.from_row(dict(zip(cols, row)))
            except Exception as e:
                print(f"User.get_by_email DB error: {e}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        # fallback search
        for u in _USERS.values():
            if u['email'].lower() == email.lower():
                return cls(
                    username=u['username'],
                    email=u['email'],
                    name=u.get('name',''),
                    password_hash=u['password_hash'],
                    role=u.get('role','user')
                )
        return None

    @classmethod
    def get_all(cls):
        conn = cls._get_connection()
        users = []
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT user_id, email, [id], username, password_hash, [role] FROM [DM_MONTYNT\\dli2].PreprocessorUser")
                rows = cursor.fetchall()
                columns = [c[0] for c in cursor.description]
                for row in rows:
                    users.append(cls.from_row(dict(zip(columns, row))))
                return users
            except Exception as e:
                print(f"User.get_all DB error: {e}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        # fallback
        return [cls.get(u) for u in _USERS.keys()]

    @classmethod
    def check_password(cls, username: str, password: str) -> Tuple[bool, str]:
        """Validate credentials, update last_login_at on success."""
        user = cls.get_by_username(username)
        if not user:
            return False, "User does not exist"
        if not user.password_hash:
            return False, "Password not set for user"
        if check_password_hash(user.password_hash, password):
            # Update last_login_at in DB (best effort)
            conn = cls._get_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE [DM_MONTYNT\\dli2].PreprocessorUser SET last_login_at = GETDATE() WHERE username = ?",
                        (username,)
                    )
                    conn.commit()
                except Exception as e:
                    print(f"Failed updating last_login_at: {e}")
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
            return True, "Login successful"
        return False, "Incorrect password"

    # Utility to create a user safely (hash password)
    @classmethod
    def create(cls, username: str, email: str, password: str, role: str = "sourcing") -> Tuple[bool, str]:
        if cls.get_by_username(username):
            return False, "Username already exists"
        if cls.get_by_email(email):
            return False, "Email already registered"
        if role not in {"admin","sourcing","mdm"}:
            return False, "Invalid role"
        password_hash = generate_password_hash(password)
        user = cls(username=username, email=email, password_hash=password_hash, role=role)
        if user.save():
            return True, "User created"
        return False, "Failed to create user"
