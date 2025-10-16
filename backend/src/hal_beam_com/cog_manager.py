#!/usr/bin/python3
"""
Cog Manager - Main Controller for HAL's AI Processing Pipeline

This module serves as the central controller for processing AI-related tasks in the HAL system.
It manages the workflow execution, data processing, and database logging for various AI cog functions.

Key Responsibilities:
1. Manages communication between beamline interface and AI processing workflows
2. Orchestrates different AI workflows based on input channel type
3. Tracks and measures performance metrics for each workflow execution
4. Maintains a persistent log of all operations in SQLite database
5. Handles model selection and configuration for different cog tasks

Database Schema:
- events_final: Database table that tracks complete interaction cycles with VISION. Each record represents a single user interaction from start to finish
- cog_table_final: Stores detailed information about individual cog operations
"""

import os
import sys
import sqlite3
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import traceback
import threading

import pandas as pd

# Configure project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.hal_beam_com.user_auth import AuthHandler, handle_auth_request
from src.hal_beam_com.file_utils import get_temp_dir, get_db_dir
from src.hal_beam_com.utils import InputChannel
from Base import *
from CustomS3 import MultiClientQueue_AI
from workflows import (
    add_context_functions_workflow,
    command_workflow,
    chatbot_workflow,
    simulation_workflow,
    evaluation_workflow,
)
from utils import (
    append_to_command_examples,
    get_model_config,
    get_finetuned_config,
    cog_output_fields,
    CogType
)
# Import the simulation workflow functions
from workflows.simulation_workflow import get_simulation_events, clear_simulation_events

# Constants
ERROR_FOLDER = "./errors"
DATABASE_NAME = 'all_events_logs.db'
DB_PATH = os.path.join(get_db_dir(), DATABASE_NAME)

class DatabaseManager:
    """Handles all database operations for the cog manager."""
    
    def __init__(self):
        """Initialize database connection and create tables if they don't exist."""
        # Use thread-local storage for database connections
        self._local = threading.local()
        self._create_tables()

    def get_connection(self):
        """Get a thread-safe database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(DB_PATH)
            self._local.cursor = self._local.conn.cursor()
        return self._local.conn, self._local.cursor

    def _create_tables(self) -> None:
        """Create necessary database tables if they don't exist."""
        # Create a temporary connection for table creation (this runs in main thread)
        temp_conn = sqlite3.connect(DB_PATH)
        temp_cursor = temp_conn.cursor()

        # Updated users table without email field
        temp_cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                custom_func TEXT DEFAULT NULL,
                created_at TEXT NOT NULL
            );
        """)

        # Sessions table
        temp_cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            );
        """)

        temp_cursor.execute("""
            CREATE TABLE IF NOT EXISTS events_final (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT,
                input TEXT,
                last_cog_output TEXT,
                final_output TEXT,
                start_time TEXT,
                end_time TEXT,
                hal_time TEXT,
                execution_time TEXT,
                status TEXT CHECK(status IN ('success', 'failure'))
            );
        """)

        temp_cursor.execute("""
            CREATE TABLE IF NOT EXISTS cog_table_final(
                cog_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER,
                cog TEXT,
                model TEXT,
                system_prompt TEXT,
                input TEXT, 
                output TEXT,
                start_time TEXT,
                end_time TEXT,
                execution_time TEXT,
                FOREIGN KEY (event_id) REFERENCES events_final(event_id)
            );
        """)

        # Add user_id column indirectly to the tables to preserve the original data and table structure
        # Add user_id column to events_final table
        try:
            temp_cursor.execute("""
                ALTER TABLE events_final
                ADD COLUMN user_id TEXT REFERENCES users (user_id)
            """)
            print("Added user_id column to events_final table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("user_id column already exists in events_final table")
            else:
                print(f"Could not add user_id to events_final: {e}")

        # Add user_id column to cog_table_final table
        try:
            temp_cursor.execute("""
                ALTER TABLE cog_table_final
                ADD COLUMN user_id TEXT REFERENCES users (user_id)
            """)
            print("Added user_id column to cog_table_final table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("user_id column already exists in cog_table_final table")
            else:
                print(f"Could not add user_id to cog_table_final: {e}")

        # Create indexes for better performance
        temp_cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_user_id
            ON events_final(user_id)
        """)

        temp_cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cog_user_id
            ON cog_table_final(user_id)
        """)
        temp_conn.commit()
        temp_conn.close()

    def user_exists(self, user_id: str) -> bool:
        """Check if a user exists by user_id"""
        conn, cursor = self.get_connection()
        cursor.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
        exists = cursor.fetchone() is not None
        print(f"[DB] User {user_id} exists: {exists}")
        return exists

    def create_user(self, user_id: str, password_hash: str) -> None:
        """Create a new user"""
        conn, cursor = self.get_connection()
        created_at = datetime.now().isoformat()
        print(f"[DB] Creating user: {user_id}")
        cursor.execute("""
            INSERT INTO users (user_id, password_hash, created_at)
            VALUES (?, ?, ?)
        """, (user_id, password_hash, created_at))
        conn.commit()
        print(f"[DB] User created successfully: {user_id}")

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by user_id"""
        conn, cursor = self.get_connection()
        cursor.execute("""
            SELECT user_id, password_hash, created_at 
            FROM users WHERE user_id = ?
        """, (user_id,))
        row = cursor.fetchone()
        if row:
            return {
                'user_id': row[0],
                'password_hash': row[1],
                'created_at': row[2]
            }
        return None

    def get_user_db_history(self, user_id: str) -> tuple[str, str]:
        conn, cursor = self.get_connection()
        cursor.execute("""
            SELECT cog, input, output
            FROM cog_table_final
            WHERE user_id = ?
            """, (user_id,))

        rows = cursor.fetchall()

        # Load data into a pandas DataFrame
        df = pd.DataFrame(rows, columns=["cog", "input", "output"])

        classifier_cog_df = df[df["cog"] == CogType.CLASSIFIER.value]
        operation_cog_df = df[df["cog"] == CogType.OP.value]

        classifier_db_history = ""
        for _, row in classifier_cog_df.iterrows():
            print("Fetch DB: classifier")
            print(f"Human: {row['input']}\nAI: {row['output']}\n")
            classifier_db_history += f"Human: {row['input']}\nAI: {row['output']}\n"

        operation_db_history = ""
        for _, row in operation_cog_df.iterrows():
            print("Fetch DB: operation")
            print(f"Human: {row['input']}\nAI: {row['output']}\n")
            operation_db_history += f"Human: {row['input']}\nAI: {row['output']}\n"

        return classifier_db_history, operation_db_history

    def add_custom_func(self, user_id: str, custom_func_path: str) -> None:
        """
        Update the custom_func column for a specific user.

        Args:
            user_id: The ID of the user whose custom_func should be updated
            custom_func_path: The path of JSON file for user's custom_func (use None to set NULL)
        """
        conn, cursor = self.get_connection()
        cursor.execute("""
            UPDATE users
            SET custom_func = ?
            WHERE user_id = ?
        """, (custom_func_path, user_id))
        conn.commit()

    def get_custom_func_path(self, user_id: str) -> str:
        """ Get the custom_func value for a specific user. """
        conn, cursor = self.get_connection()
        cursor.execute("""
            SELECT custom_func
            FROM users
            WHERE user_id = ?
        """, (user_id,))

        row = cursor.fetchone()
        return row[0] if row else None


    def create_session(self, session_id: str, user_id: str, expires_at: datetime) -> None:
        """Create a new session"""
        conn, cursor = self.get_connection()
        created_at = datetime.now().isoformat()
        expires_at_str = expires_at.isoformat()
        print(f"[DB] Creating session: {session_id} for user: {user_id}")
        cursor.execute("""
            INSERT INTO sessions (session_id, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (session_id, user_id, created_at, expires_at_str))
        conn.commit()
        print(f"[DB] Session created successfully: {session_id}")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by session_id"""
        conn, cursor = self.get_connection()
        cursor.execute("""
            SELECT session_id, user_id, created_at, expires_at 
            FROM sessions WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        if row:
            return {
                'session_id': row[0],
                'user_id': row[1],
                'created_at': row[2],
                'expires_at': row[3]
            }
        return None

    def delete_session(self, session_id: str) -> None:
        """Delete a session"""
        conn, cursor = self.get_connection()
        print(f"[DB] Deleting session: {session_id}")
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        rows_deleted = cursor.rowcount
        conn.commit()
        print(f"[DB] Session deletion result: {rows_deleted} rows deleted")

    def cleanup_expired_sessions(self) -> None:
        """Remove expired sessions from database"""
        conn, cursor = self.get_connection()
        current_time = datetime.now().isoformat()
        cursor.execute("DELETE FROM sessions WHERE expires_at < ?", (current_time,))
        conn.commit()

    def log_cog_operation(self, event_id: int, cog_type: CogType, user_id: str,
                          field_keys: List[str], data: list[Dict]) -> None:
        """
        Log individual cog operation details to database.
        
        Args:
            event_id: ID of the parent event
            cog_type: Type of cog operation
            user_id: ID of the user
            field_keys: List of field keys for this cog operation
            data: Data dictionary containing operation details
        """
        conn, cursor = self.get_connection()
        db_user_id = None if user_id == 'guest' else user_id
        cog_details = {
            'model': data[0].get(field_keys[0]),
            'input': data[0].get(field_keys[1]),
            'output': data[0].get(field_keys[2]),
            'start_time': data[0].get(field_keys[3]),
            'end_time': data[0].get(field_keys[4]),
            'execution_time': data[0].get(field_keys[5]),
            'system_prompt': data[0].get(field_keys[6]),
            'user_id': db_user_id
        }
        
        if not cog_details['output']:
            return

        cursor.execute("""
            INSERT INTO cog_table_final (event_id, cog, model, system_prompt, input, output, 
                                 start_time, end_time, execution_time, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (event_id, cog_type.value, cog_details['model'], cog_details['system_prompt'],
              cog_details['input'], cog_details['output'], cog_details['start_time'],
              cog_details['end_time'], cog_details['execution_time'], cog_details['user_id']))

    def log_event(self, data: list[Dict], times: Dict[str, Any],
                  status: str) -> int:
        """
        Log event details to database.
        
        Returns:
            event_id: ID of the created event
        """
        conn, cursor = self.get_connection()

        # Handle missing text_input field
        text_input = data[0].get('text_input', '')
        voice_cog_output = data[0].get('voice_cog_output', '')
        input_text = text_input + voice_cog_output

        last_cog_id = data[0].get('last_cog_id')
        if last_cog_id is None:
            print("Warning: 'last_cog_id' is missing in data[0]. Skipping event logging.")
            return None
        last_cog_output = data[0][cog_output_fields[last_cog_id]]
        final_output = data[0]['final_output']
        project_path = data[0]["project_path"]
        user_id = data[0]["user_id"]
        db_user_id = None if user_id == 'guest' else user_id

        if last_cog_output != final_output:
            status = "failure"

        cursor.execute("""
            INSERT INTO events_final(project_id, input, last_cog_output, final_output, start_time, end_time, 
                             hal_time, execution_time, status, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            project_path, 
            input_text, 
            last_cog_output,
            final_output,
            f"{times['start_time']} seconds",
            f"{times['end_time']}",
            f"{times['hal_time']:.3f} seconds",
            f"{times['execution_time']:.3f} seconds",
            status,
            db_user_id
        ))
        
        return cursor.lastrowid

    def commit(self):
        """Commit changes to database."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.commit()

    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
            self._local.cursor = None

class WorkflowProcessor:
    """Handles all workflow processing operations."""
    
    @staticmethod
    def process_workflow(user_auth: AuthHandler, input_channel: InputChannel, data: List[Dict],
                        queue, start_time: float) -> Tuple[List[Dict], str, float, bool]:
        """
        Process workflow based on input channel type.
        
        Args:
            user_auth: Class for handling user authentication.
            input_channel: Type of input channel
            data: Input data
            queue: Queue for communication
            start_time: Start time of processing
        
        Returns:
            Tuple containing:
            - Processed data
            - Status
            - Processing time on HAL
        """
        status = "success"
        processing_time = 0.0
        skip_logging = False

        print(data[0])

        try:
            match input_channel:
                case InputChannel.REGISTER | InputChannel.LOGIN | InputChannel.LOGOUT:
                    print(f"[BACKEND] Handling auth request: {input_channel}")
                    data = handle_auth_request(data, user_auth)
                    status = data[0]['status']
                    processing_time = time.time() - start_time
                    print(f"[BACKEND] Auth request result: {status}")
                    print(f"[BACKEND] Publishing auth response: {data}")
                    # queue.publish(data)
                    print(f"[BACKEND] Auth response published")
                    skip_logging = True
                case InputChannel.COMMAND:
                    data = command_workflow(
                        data,
                        # queue,
                        audio_base_model=get_model_config("audio", "transcription"),
                        audio_finetuned=get_finetuned_config("audio"),
                        text_base_model=get_model_config("text", "default"),
                        text_finetuned=get_finetuned_config("text"),
                        use_static_prompt=False
                    )
                    status = data[0]['status']
                    processing_time = time.time() - start_time
                    # queue.publish(data)
                    skip_logging = False

                case InputChannel.ADD_CONTEXT:
                    data = add_context_functions_workflow(
                        data,
                        # queue,
                        audio_base_model=get_model_config("audio", "transcription"),
                        audio_finetuned=get_finetuned_config("audio"),
                        text_base_model=get_model_config("text", "refinement"),
                        text_finetuned=get_finetuned_config("text")
                    )
                    status = data[0]['status']
                    processing_time = time.time() - start_time
                    # queue.publish(data)
                    skip_logging = False

                case InputChannel.CONFIRM_CONTEXT:
                    append_to_command_examples(data)
                    # queue.publish(data)
                    skip_logging = False

                case InputChannel.CHATBOT:
                    data = chatbot_workflow(
                        data,
                        # queue,
                        audio_base_model=get_model_config("audio", "transcription"),
                        audio_finetuned=get_finetuned_config("audio"),
                        text_base_model=get_model_config("text", "default"),
                        text_finetuned=get_finetuned_config("text")
                    )
                    # queue.publish(data)
                    skip_logging = False

                case InputChannel.SIMULATE:
                    data = simulation_workflow(
                        data,
                        queue,
                    )
                    # streaming already handled – nothing to log here
                    skip_logging = True

                case InputChannel.SIMULATE_ABORT:
                    # forward abort message to the same workflow; nothing
                    # to log or store in DB
                    data = simulation_workflow(data, queue)
                    status = data[0].get('status', 'success')
                    processing_time = time.time() - start_time
                    skip_logging = True

                case InputChannel.EVALUATE_SIMULATION:
                    # Retrieve simulation events for this sim_id
                    sim_id = data[0].get('sim_id', '')
                    pv_events = get_simulation_events(sim_id)
                    data[0]['pv_events'] = pv_events
                    
                    # Run evaluation workflow
                    data = evaluation_workflow(data)
                    status = data[0].get('status', 'success')
                    processing_time = time.time() - start_time
                    
                    # Clear the simulation events for this sim_id to free memory
                    clear_simulation_events(sim_id)
                    
                    skip_logging = True

                case InputChannel.CONFIRM_CODE:
                    # queue.publish(data)
                    skip_logging = False

                case _:
                    raise ValueError(f"Unknown input channel: {input_channel}")

            if processing_time:
                print("Time to proces:", processing_time)

        except Exception as e:
            print(f"Error in workflow processing: {e}")
            traceback.print_exc()  # Prints the full traceback of the error
            status = "failure"
            processing_time = time.time() - start_time

        return data, status, processing_time, skip_logging

class CogManager:
    """Main class for managing cog operations."""

    def __init__(self):
        """Initialize the cog manager."""
        self.db = DatabaseManager()
        self.auth_handler = AuthHandler(self.db)
        self.workflow_processor = WorkflowProcessor()
        self.cog_operations = {
            CogType.VOICE: ['voice_cog_model', 'voice_cog_input', 'voice_cog_output',
                           'voice_cog_start_time', 'voice_cog_end_time', 
                           'voice_cog_execution_time', 'voice_cog_system_prompt'],
            CogType.CLASSIFIER: ['classifier_cog_model', 'classifier_cog_input', 
                               'classifier_cog_output', 'classifier_cog_start_time',
                               'classifier_cog_end_time', 'classifier_cog_execution_time', 
                               'classifier_cog_system_prompt'],
            CogType.OP: ['operator_cog_model', 'operator_cog_input', 'operator_cog_output',
                        'operator_cog_start_time', 'operator_cog_end_time',
                        'operator_cog_execution_time', 'operator_cog_system_prompt'],
            CogType.ANA: ['analysis_cog_model', 'analysis_cog_input', 'analysis_cog_output',
                         'analysis_cog_start_time', 'analysis_cog_end_time',
                         'analysis_cog_execution_time', 'analysis_cog_system_prompt'],
            CogType.REFINE: ['refinement_cog_model', 'refinement_cog_input', 
                            'refinement_cog_output', 'refinement_cog_start_time',
                            'refinement_cog_end_time', 'refinement_cog_execution_time',
                            'refinement_cog_system_prompt'],
            CogType.VERIFIER: ['verifier_cog_model', 'verifier_cog_input', 
                            'verifier_cog_output', 'verifier_cog_start_time',
                            'verifier_cog_end_time', 'verifier_cog_execution_time',
                            'verifier_cog_system_prompt'],

        }
        self.active_threads = {}
        self.thread_lock = threading.Lock()

    def process_client_request(self, data: List[Dict], client_id: str, client_queue, queue) -> None:
        """
        Process a single client request in a separate thread.

        Args:
            data: Request data from client
            client_id: Unique identifier for the client
            client_queue: Client-specific queue for communication
            queue: List of all client queue
        """
        try:
            print(f"[BACKEND] Processing request for client: {client_id}")
            start_time = time.time()
            start_current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Process workflow
            input_channel = InputChannel(data[0]['bl_input_channel'])
            print(f"[BACKEND] Processing input channel: {input_channel} for client: {client_id}")
            data, status, processing_time, skip_logging = self.workflow_processor.process_workflow(
                self.auth_handler, input_channel, data, client_queue, start_time
            )

            if input_channel != InputChannel.SIMULATE and input_channel != InputChannel.SIMULATE_ABORT:
                queue.publish_to_client(data, client_id)

            if skip_logging:
                return

            # Calculate timing information
            end_time = time.time()
            execution_time = end_time - start_time

            times = {
                'start_time': start_current_time,
                'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'hal_time': processing_time,
                'execution_time': execution_time
            }

            # Log event and cog operations
            with self.thread_lock:
                event_id = self.db.log_event(data, times, status)
                if event_id is not None:  # Only log cog operations if event logging succeeded
                    user_id = data[0]["user_id"]

                    for cog_type, field_keys in self.cog_operations.items():
                        self.db.log_cog_operation(event_id, cog_type, user_id, field_keys, data)

                    self.db.commit()
                    print(f"[BACKEND] Database entry completed for client {client_id}! Execution time: {execution_time:.5f} seconds")
                else:
                    print(f"[BACKEND] Skipped database logging for client {client_id} due to missing required fields")

        except Exception as e:
            print(f"[BACKEND] Error processing request for client {client_id}: {e}")
            traceback.print_exc()
        finally:
            # Remove thread from active threads
            with self.thread_lock:
                if client_id in self.active_threads:
                    del self.active_threads[client_id]

    def process(self, queue: MultiClientQueue_AI) -> None:
        """
        Main processing loop for handling AI workflow requests from multiple clients.
        """
        queue.clear()
        print("[BACKEND] Starting multi-client processing loop...")

        try:
            while True:
                print("[BACKEND] Waiting for next command from any client...")

                # Get request from any available client
                data, client_id, client_queue = queue.get()
                print(f"[BACKEND] Received command from client {client_id}: {data[0].get('bl_input_channel', 'unknown')}")

                # Check if we already have an active thread for this client
                with self.thread_lock:
                    if client_id in self.active_threads:
                        print(f"[BACKEND] Client {client_id} already has an active request. Queuing...")
                        # In a more sophisticated implementation, you might want to queue this request
                        # For now, we'll skip it to avoid conflicts
                        continue

                    # Create a new thread for this client
                    thread = threading.Thread(
                        target=self.process_client_request,
                        args=(data, client_id, client_queue, queue),
                        daemon=True
                    )
                    self.active_threads[client_id] = thread
                    thread.start()
                    print(f"[BACKEND] Started processing thread for client: {client_id}")

        except Exception as e:
            print(f"Error in main processing loop: {e}")
            traceback.print_exc()
            raise

        finally:
            # Wait for all active threads to complete
            for thread in self.active_threads.values():
                thread.join()
            self.db.close()

def main():
    """Main entry point for the cog manager."""
    try:
        queue = MultiClientQueue_AI(save_dir=get_temp_dir())
        manager = CogManager()
        manager.process(queue)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
