import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any

# Constants
STATIC_PASSWORD = "123456"  # Static password for all users

class AuthHandler:
    def __init__(self, db_manager):
        # Database connection
        self.db_manager = db_manager

    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'db_manager'):
            self.db_manager.close()

    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def generate_session_id(self) -> str:
        """Generate a secure session ID"""
        return secrets.token_urlsafe(32)

    def create_session(self, user_id: str) -> str:
        """Create a new session for a user"""
        session_id = self.generate_session_id()
        expires_at = datetime.now() + timedelta(hours=24)  # 24 hour session

        self.db_manager.create_session(session_id, user_id, expires_at)

        return session_id

    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate session and return user_id if valid"""
        session = self.db_manager.get_session(session_id)
        if not session:
            return None

        # Check if session is expired
        expires_at = datetime.fromisoformat(session['expires_at'])
        if datetime.now() > expires_at:
            # Session expired, remove it
            self.db_manager.delete_session(session_id)
            return None

        return session['user_id']

    def login_or_create_user(self, data: list[Dict]) -> list[Dict]:
        """Login existing user or create new user automatically"""
        user_id = data[0].get('user_id', '')
        print(f"[AUTH] login_or_create_user called for: {user_id}")

        if not user_id:
            data[0]['status'] = 'error'
            data[0]['message'] = 'User ID is required'
            return data

        # Check if user exists
        user_exists = self.db_manager.user_exists(user_id)
        print(f"[AUTH] User {user_id} exists: {user_exists}")

        if user_exists:
            # User exists - perform login
            print(f"[AUTH] Logging in existing user: {user_id}")
            return self._login_existing_user(data)
        else:
            # User doesn't exist - create new user automatically
            print(f"[AUTH] Creating new user: {user_id}")
            return self._create_new_user(data)

    def _login_existing_user(self, data: list[Dict]) -> list[Dict]:
        """Login an existing user"""
        user_id = data[0].get('user_id', '')
        print(f"[AUTH] _login_existing_user for: {user_id}")

        user = self.db_manager.get_user(user_id)
        print(f"[AUTH] User data retrieved: {user is not None}")

        if not user:
            data[0]['status'] = 'error'
            data[0]['message'] = 'User not found'
            return data

        # Create session
        session_id = self.create_session(user_id)
        print(f"[AUTH] Session created: {session_id}")

        data[0]['session_id'] = session_id
        data[0]['classifier_cog_db_history'], data[0]['operator_cog_db_history'] = self.db_manager.get_user_db_history(
            user_id)
        data[0]['status'] = 'success'
        data[0]['message'] = 'Login successful'
        print(f"[AUTH] Login successful for: {user_id}")
        return data

    def _create_new_user(self, data: list[Dict]) -> list[Dict]:
        """Create a new user automatically"""
        user_id = data[0].get('user_id', '')
        print(f"[AUTH] _create_new_user for: {user_id}")

        # Create new user with static password
        password_hash = self.hash_password(STATIC_PASSWORD)
        self.db_manager.create_user(user_id, password_hash)
        print(f"[AUTH] User created in database: {user_id}")

        # Create session
        session_id = self.create_session(user_id)
        print(f"[AUTH] Session created for new user: {session_id}")

        data[0]['session_id'] = session_id
        data[0]['classifier_cog_db_history'], data[0]['operator_cog_db_history'] = '', ''  # New user has no history
        data[0]['status'] = 'success'
        data[0]['message'] = f'New user "{user_id}" created successfully'
        print(f"[AUTH] New user creation successful: {user_id}")
        return data

    def logout_user(self, data: list[Dict]) -> list[Dict]:
        """Logout a user"""
        session_id = data[0].get('session_id', '')
        user_id = data[0].get('user_id', '')
        print(f"[AUTH] logout_user for user: {user_id}, session: {session_id}")

        # Always try to delete the session, even if it's invalid
        if session_id:
            self.db_manager.delete_session(session_id)
            print(f"[AUTH] Session deleted: {session_id}")
        else:
            print(f"[AUTH] No session ID provided for logout")

        # Clear user data regardless of session validity
        data[0]['classifier_cog_db_history'], data[0]['operator_cog_db_history'] = '', ''
        data[0]['status'] = 'success'
        data[0]['message'] = 'Logout successful'
        print(f"[AUTH] Logout successful for user: {user_id}")
        return data


def handle_auth_request(data: list[Dict], auth_handler: AuthHandler) -> list[Dict]:
    """
    Main function to handle authentication requests from the frontend.
    This should be integrated into your existing HAL backend message handling.
    """
    bl_input_channel = data[0].get('bl_input_channel')
    print(f"[AUTH] Handling request: {bl_input_channel}")

    if bl_input_channel == 'login':
        user_id = data[0].get('user_id', '')
        print(f"[AUTH] Login attempt for user: {user_id}")

        if not user_id:
            data[0]['status'] = 'error'
            data[0]['message'] = 'User ID is required'
            return data

        result = auth_handler.login_or_create_user(data)
        print(f"[AUTH] Login result: {result[0]['status']} - {result[0]['message']}")
        return result

    elif bl_input_channel == 'logout':
        user_id = data[0].get('user_id', '')
        session_id = data[0].get('session_id', '')
        print(f"[AUTH] Logout attempt for user: {user_id}, session: {session_id}")

        if not user_id or not session_id:
            data[0]['status'] = 'error'
            data[0]['message'] = 'Missing user ID or session ID'
            return data

        result = auth_handler.logout_user(data)
        print(f"[AUTH] Logout result: {result[0]['status']} - {result[0]['message']}")
        return result

    else:
        data[0]['status'] = 'error'
        data[0]['message'] = f'Unknown authentication channel: {bl_input_channel}'
        return data
