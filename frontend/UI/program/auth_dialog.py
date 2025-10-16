from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QWidget, QMessageBox
)
from PyQt5.QtCore import Qt
import os


class AuthDialog(QDialog):
    def __init__(self, parent=None):
        super(AuthDialog, self).__init__(parent)
        self.setWindowTitle("VISION Authentication")
        self.setFixedSize(400, 250)
        self.setModal(True)

        # Initialize variables
        self.user_id = ""
        self.is_authenticated = False
        self.is_guest_mode = False
        self.is_new_user = False

        self.setup_ui()

    def setup_ui(self):
        # Main layout
        layout = QVBoxLayout()

        # Title
        title = QLabel("Login to VISION")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)

        # Description
        description = QLabel("Enter your User ID to login or create a new account")
        description.setAlignment(Qt.AlignCenter)
        description.setStyleSheet("font-size: 12px; color: #666; margin: 5px;")
        layout.addWidget(description)

        # User ID
        self.user_id_label = QLabel("User ID:")
        self.user_id_input = QLineEdit()
        self.user_id_input.setPlaceholderText("Enter your user ID")
        self.user_id_input.returnPressed.connect(self.handle_login)
        layout.addWidget(self.user_id_label)
        layout.addWidget(self.user_id_input)

        # Login button
        self.login_button = QPushButton("Login / Create Account")
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        self.login_button.clicked.connect(self.handle_login)
        layout.addWidget(self.login_button)

        # Continue as Guest button
        self.guest_button = QPushButton("Continue as Guest")
        self.guest_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        self.guest_button.clicked.connect(self.handle_guest_mode)
        layout.addWidget(self.guest_button)

        # Add some spacing
        layout.addStretch()

        self.setLayout(layout)

    def handle_login(self):
        user_id = self.user_id_input.text().strip()
        print(f"[AUTH_DIALOG] handle_login called with user_id: {user_id}")

        if not user_id:
            QMessageBox.warning(self, "Error", "Please enter a User ID.")
            return

        # Create login data structure
        login_data = {
            'bl_input_channel': 'login',
            'user_id': user_id,
            'password': '123456',  # Static password for all users
            'session_id': '',
            'status': 'success',
            'message': ''
        }

        print(f"[AUTH_DIALOG] Created login data: {login_data}")

        # Store the login data for the main UI to use
        self.user_id = user_id
        self.is_authenticated = True
        self.login_data = login_data

        print(f"[AUTH_DIALOG] Setting dialog state - user_id: {self.user_id}, is_authenticated: {self.is_authenticated}")
        self.accept()

    def handle_guest_mode(self):
        """Handle guest mode - no authentication required"""
        print("[AUTH_DIALOG] handle_guest_mode called")

        self.user_id = "guest"
        self.is_authenticated = True
        self.is_guest_mode = True

        # Create guest data structure
        self.guest_data = {
            'bl_input_channel': 'guest_login',
            'user_id': 'guest',
            'password': '',
            'session_id': '',
            'status': 'success',
            'message': 'Guest mode activated'
        }

        print(f"[AUTH_DIALOG] Guest mode data: {self.guest_data}")
        self.accept()

    def closeEvent(self, event):
        """Handle window close event - continue as guest"""
        print("[AUTH_DIALOG] closeEvent triggered - continuing as guest")
        self.handle_guest_mode()
        event.accept()

    def get_auth_data(self):
        """Return the authentication data for the main UI"""
        print(f"[AUTH_DIALOG] get_auth_data called")
        print(f"[AUTH_DIALOG] has login_data: {hasattr(self, 'login_data')}")
        print(f"[AUTH_DIALOG] has guest_data: {hasattr(self, 'guest_data')}")

        if hasattr(self, 'login_data'):
            print(f"[AUTH_DIALOG] Returning login_data: {self.login_data}")
            return self.login_data
        elif hasattr(self, 'guest_data'):
            print(f"[AUTH_DIALOG] Returning guest_data: {self.guest_data}")
            return self.guest_data
        print("[AUTH_DIALOG] No auth data found, returning None")
        return None