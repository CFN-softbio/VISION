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
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Configure project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.hal_beam_com.utils import InputChannel
from Base import *
from CustomS3 import Queue_AI
from workflows import (
    add_context_functions_workflow,
    command_workflow,
    chatbot_workflow
)
from utils import (
    append_to_command_examples,
    get_model_config,
    get_finetuned_config,
    cog_output_fields,
    CogType
)

# Constants
DATABASE_NAME = 'all_events_logs.db'
ERROR_FOLDER = "./errors"

class DatabaseManager:
    """Handles all database operations for the cog manager."""
    
    def __init__(self):
        """Initialize database connection and create tables if they don't exist."""
        self.conn = sqlite3.connect(DATABASE_NAME)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self) -> None:
        """Create necessary database tables if they don't exist."""
        self.cursor.execute("""
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

        self.cursor.execute("""
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
        self.conn.commit()

    def log_cog_operation(self, event_id: int, cog_type: CogType, 
                              field_keys: List[str], data: Dict[str, Any]) -> None:
        """
        Log individual cog operation details to database.
        
        Args:
            event_id: ID of the parent event
            cog_type: Type of cog operation
            field_keys: List of field keys for this cog operation
            data: Data dictionary containing operation details
        """
        cog_details = {
            'model': data[0].get(field_keys[0]),
            'input': data[0].get(field_keys[1]),
            'output': data[0].get(field_keys[2]),
            'start_time': data[0].get(field_keys[3]),
            'end_time': data[0].get(field_keys[4]),
            'execution_time': data[0].get(field_keys[5]),
            'system_prompt': data[0].get(field_keys[6])
        }
        
        if not cog_details['output']:
            return

        self.cursor.execute("""
            INSERT INTO cog_table_final (event_id, cog, model, system_prompt, input, output, 
                                 start_time, end_time, execution_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (event_id, cog_type.value, cog_details['model'], cog_details['system_prompt'],
              cog_details['input'], cog_details['output'], cog_details['start_time'],
              cog_details['end_time'], cog_details['execution_time']))

    def log_event(self, data: Dict[str, Any], times: Dict[str, Any], 
                  status: str) -> int:
        """
        Log event details to database.
        
        Returns:
            event_id: ID of the created event
        """
        input_text = data[0]['text_input'] + data[0]['voice_cog_output']
        last_cog_id = data[0]['last_cog_id']
        last_cog_output = data[0][cog_output_fields[last_cog_id]]
        final_output = data[0]['final_output']
        project_path = data[0]["project_path"]

        if last_cog_output != final_output:
            status = "failure"

        self.cursor.execute("""
            INSERT INTO events_final(project_id, input, last_cog_output, final_output, start_time, end_time, 
                             hal_time, execution_time, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            project_path, 
            input_text, 
            last_cog_output,
            final_output,
            f"{times['start_time']} seconds",
            f"{times['end_time']}",
            f"{times['hal_time']:.3f} seconds",
            f"{times['execution_time']:.3f} seconds",
            status
        ))
        
        return self.cursor.lastrowid

    def commit(self):
        """Commit changes to database."""
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()

class WorkflowProcessor:
    """Handles all workflow processing operations."""
    
    @staticmethod
    def process_workflow(input_channel: InputChannel, data: List[Dict], 
                        queue: Queue_AI, start_time: float) -> Tuple[List[Dict], str, float]:
        """
        Process workflow based on input channel type.
        
        Args:
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
                case InputChannel.COMMAND:
                    data = command_workflow.run(
                        data,
                        queue,
                        audio_base_model=get_model_config("audio", "transcription"),
                        audio_finetuned=get_finetuned_config("audio"),
                        text_base_model=get_model_config("text", "default"),
                        text_finetuned=get_finetuned_config("text"),
                        use_static_prompt=True
                    )
                    status = data[0]['status']
                    processing_time = time.time() - start_time
                    queue.publish(data)
                    skip_logging = True

                case InputChannel.ADD_CONTEXT:
                    data = add_context_functions_workflow.run(
                        data,
                        queue,
                        audio_base_model=get_model_config("audio", "transcription"),
                        audio_finetuned=get_finetuned_config("audio"),
                        text_base_model=get_model_config("text", "refinement"),
                        text_finetuned=get_finetuned_config("text")
                    )
                    status = data[0]['status']
                    processing_time = time.time() - start_time
                    queue.publish(data)
                    skip_logging = True

                case InputChannel.CONFIRM_CONTEXT:
                    append_to_command_examples(data)
                    queue.publish(data)
                    skip_logging = True

                case InputChannel.CHATBOT:
                    data = chatbot_workflow.run(
                        data,
                        queue,
                        audio_base_model=get_model_config("audio", "transcription"),
                        audio_finetuned=get_finetuned_config("audio"),
                        text_base_model=get_model_config("text", "default"),
                        text_finetuned=get_finetuned_config("text")
                    )
                    queue.publish(data)
                    skip_logging = True

                case InputChannel.CONFIRM_CODE:
                    queue.publish(data)
                    skip_logging = False

                case _:
                    raise ValueError(f"Unknown input channel: {input_channel}")

        except Exception as e:
            print(f"Error in workflow processing: {e}")
            status = "failure"
            processing_time = time.time() - start_time

        return data, status, processing_time, skip_logging

class CogManager:
    """Main class for managing cog operations."""

    def __init__(self):
        """Initialize the cog manager."""
        self.db = DatabaseManager()
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
                            'refinement_cog_system_prompt']
        }

    def process(self, queue: Queue_AI) -> None:
        """
        Main processing loop for handling AI workflow requests.
        """
        queue.clear()

        try:
            while True:
                print("Waiting for next command")
                data = queue.get()
                start_time = time.time()
                start_current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Process workflow
                input_channel = InputChannel(data[0]['bl_input_channel'])
                data, status, processing_time, skip_logging = self.workflow_processor.process_workflow(
                    input_channel, data, queue, start_time
                )

                if skip_logging:
                    continue

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
                event_id = self.db.log_event(data, times, status)
                
                for cog_type, field_keys in self.cog_operations.items():
                    self.db.log_cog_operation(event_id, cog_type, field_keys, data)

                self.db.commit()
                print(f"Database entry completed! Execution time: {execution_time:.5f} seconds")

        except Exception as e:
            print(f"Error in main processing loop: {e}")
            raise

        finally:
            self.db.close()

def main():
    """Main entry point for the cog manager."""
    try:
        queue = Queue_AI()
        manager = CogManager()
        manager.process(queue)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
