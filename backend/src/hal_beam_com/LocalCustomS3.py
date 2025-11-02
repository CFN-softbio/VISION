#!/usr/bin/python3
import io
import os
import time
import json
import shutil
import uuid
import numpy as np
from pathlib import Path
from Base import Base
from src.hal_beam_com.model_manager import ModelManager
from src.hal_beam_com.multi_client_base import MultiClientQueueBase, ClientSpecificQueueBase
from utils import model_configurations, ACTIVE_CONFIG, get_finetuned_config


def get_shared_comm_dir():
    """
    Get the shared communication directory that works across Docker and host.
    Checks environment variable first, then uses OS-specific defaults.
    
    Priority:
    1. HAL_COMM_DIR environment variable (if set)
    2. OS-specific default in user's home directory
    
    Returns a Path that should be mounted/accessible from both container and host.
    """
    # Check for explicit environment variable
    env_dir = os.environ.get('HAL_COMM_DIR')
    if env_dir:
        print(f"[DEBUG] Using HAL_COMM_DIR from environment: {env_dir}")
        return Path(env_dir)
    
    # Use defaults based on OS
    home = Path.home()
    
    if os.name == 'nt':  # Windows
        # Use AppData\Local for Windows
        base_dir = Path(os.environ.get('LOCALAPPDATA', home / 'AppData' / 'Local'))
        comm_dir = base_dir / 'hal_vision' / 'shared'
        print(f"[DEBUG] Using Windows default directory: {comm_dir}")
        return comm_dir
    else:  # Linux/Mac
        # Use hidden directory in home for Unix-like systems
        comm_dir = home / '.hal_vision' / 'shared'
        print(f"[DEBUG] Using Unix default directory: {comm_dir}")
        return comm_dir


def initialize_models():
    """Initialize models based on current configuration"""
    config = model_configurations[ACTIVE_CONFIG]
    text_models = set(config["text"].values())
    audio_models = [(model, get_finetuned_config("audio"))
                    for model in set(config["audio"].values())]

    print("Initializing models...")
    print(f"Text models: {text_models}")
    print(f"Audio models: {audio_models}")
    ModelManager.load_models(text_models, audio_models)
    print("Model initialization complete")


class CustomS3(Base):
    """
    Local version of CustomS3 that mimics S3 behavior using local filesystem.
    Maintains the same interface as the original CustomS3 class.
    Designed to work across Docker container and host machine via shared volume.
    """

    def __init__(self,
                 username,
                 send='send',
                 receive='receive',
                 endpoint="localhost:8000",
                 secret_key_path=None,
                 secure=True,
                 bucket_name='transmissions',
                 experiment=None,
                 name='S3',
                 save_dir='./',
                 log_verbosity=5,
                 client_id=None,
                 **kwargs):

        super().__init__(name=name, log_verbosity=log_verbosity, **kwargs)

        # Initialize models before starting the queue
        initialize_models()

        if experiment is None:
            # Use only date (not hour) so frontend and backend use same experiment name
            experiment = f'experiment_{self.now(str_format="%Y-%m-%d")}'

        self.experiment = experiment
        self.send = send
        self.receive = receive
        self.save_dir = save_dir
        self.bucket_name = bucket_name
        self.client_id = client_id or str(uuid.uuid4())

        print(f"\n{'='*80}")
        print(f"[DEBUG BACKEND] Initializing CustomS3 queue: {name}")
        print(f"[DEBUG BACKEND] Queue type: {self.__class__.__name__}")
        print(f"[DEBUG BACKEND] Client ID: {self.client_id}")
        print(f"[DEBUG BACKEND] Send queue: {send}, Receive queue: {receive}")
        print(f"[DEBUG BACKEND] Experiment: {experiment}")

        # Use shared communication directory for inter-process/container communication
        self.comm_dir = get_shared_comm_dir()
        self.base_path = self.comm_dir / bucket_name / experiment
        self.send_path_dir = self.base_path / send
        self.receive_path_dir = self.base_path / receive

        print(f"[DEBUG BACKEND] Communication directory: {self.comm_dir}")
        print(f"[DEBUG BACKEND] Base path: {self.base_path}")
        print(f"[DEBUG BACKEND] Send path: {self.send_path_dir}")
        print(f"[DEBUG BACKEND] Receive path: {self.receive_path_dir}")
        print(f"[DEBUG BACKEND] Local save directory: {os.path.abspath(self.save_dir)}")

        # Create all necessary directories
        os.makedirs(self.save_dir, exist_ok=True)
        self.send_path_dir.mkdir(parents=True, exist_ok=True)
        self.receive_path_dir.mkdir(parents=True, exist_ok=True)

        print(f"[DEBUG BACKEND] Directories created successfully")

        # Set appropriate permissions for the communication directory (Unix only)
        if os.name != 'nt':  # Not Windows
            try:
                os.chmod(self.comm_dir, 0o777)
                os.chmod(self.base_path, 0o777)
                os.chmod(self.send_path_dir, 0o777)
                os.chmod(self.receive_path_dir, 0o777)
                print(f"[DEBUG BACKEND] Permissions set to 0o777")
            except (OSError, PermissionError) as e:
                print(f"[DEBUG BACKEND] Warning: Could not set permissions: {e}")
                self.msg(f"Warning: Could not set permissions: {e}", 3, 1)

        print(f"{'='*80}\n")

        self.msg(f"Using shared communication directory: {self.comm_dir}", 4, 1)
        self.msg(f"Send path: {self.send_path_dir}", 5, 2)
        self.msg(f"Receive path: {self.receive_path_dir}", 5, 2)

    def _list_directory_contents(self, dir_path, label="Directory"):
        """Helper method to list and print directory contents for debugging"""
        try:
            if not dir_path.exists():
                print(f"[DEBUG BACKEND] {label} does not exist: {dir_path}")
                return
            
            files = list(dir_path.glob('*'))
            if files:
                print(f"[DEBUG BACKEND] {label} contents ({len(files)} items):")
                for f in sorted(files):
                    size = f.stat().st_size if f.is_file() else 'DIR'
                    print(f"[DEBUG BACKEND]   - {f.name} ({size} bytes)")
            else:
                print(f"[DEBUG BACKEND] {label} is empty: {dir_path}")
        except Exception as e:
            print(f"[DEBUG BACKEND] Error listing {label}: {e}")

    def send_path(self):
        return '{}/{}'.format(self.experiment, self.send)

    def receive_path(self):
        return '{}/{}'.format(self.experiment, self.receive)

    def getS3_floats(self):
        """Maintained for compatibility, uses get_s3_file internally"""
        return self.get_s3_file()

    def publish_s3_floats(self, data):
        """Maintained for compatibility, uses publish_s3_file internally"""
        self.publish_s3_file(data)

    def get_s3_file(self):
        """Wait for and retrieve a file from the receive directory"""
        print(f"[DEBUG BACKEND] Waiting for file in: {self.receive_path_dir}")
        wait_count = 0
        while True:
            files = sorted(self.receive_path_dir.glob('*.npy'))
            if wait_count % 10 == 0:  # Print every 10 iterations
                print(f"[DEBUG BACKEND] Checking for files... (attempt {wait_count + 1})")
                self._list_directory_contents(self.receive_path_dir, "Receive directory")
            
            if files:
                print(f"[DEBUG BACKEND] Found {len(files)} file(s) in receive directory")
                try:
                    print(f"[DEBUG BACKEND] Reading file: {files[0].name}")
                    data = np.load(files[0], allow_pickle=True)
                    print(f"[DEBUG BACKEND] Successfully loaded data from {files[0].name}")
                    
                    # Save a copy to local process directory
                    local_copy = Path(self.save_dir) / f'{self.name}-received.npy'
                    np.save(local_copy, data, allow_pickle=True)
                    print(f"[DEBUG BACKEND] Saved local copy to: {local_copy}")
                    
                    files[0].unlink()  # Remove the file after reading
                    print(f"[DEBUG BACKEND] Deleted original file: {files[0].name}")
                    return data
                except Exception as ex:
                    print(f"[DEBUG BACKEND] Error reading file: {ex}")
                    self.msg_error(f'Error reading file: {ex}', 1, 2)
                    time.sleep(0.1)
            
            wait_count += 1
            time.sleep(0.1)

    def publish_s3_file(self, data):
        """Save data to both the communication directory and local process directory"""
        timestamp = self.now(str_format='%Y%m%d_%H%M%S_%f')

        # Save to communication directory
        comm_file_path = self.send_path_dir / f'obj_{timestamp}.npy'
        print(f"[DEBUG BACKEND] Publishing file to: {comm_file_path}")
        
        np.save(comm_file_path, data, allow_pickle=True)
        file_size = comm_file_path.stat().st_size
        print(f"[DEBUG BACKEND] File written: {comm_file_path.name} ({file_size} bytes)")

        # Save to local process directory
        local_file_path = Path(self.save_dir) / f'{self.name}-sent.npy'
        np.save(local_file_path, data, allow_pickle=True)
        print(f"[DEBUG BACKEND] Local copy saved: {local_file_path}")

        if os.name != 'nt':
            try:
                os.chmod(comm_file_path, 0o666)
                print(f"[DEBUG BACKEND] File permissions set to 0o666")
            except (OSError, PermissionError) as e:
                print(f"[DEBUG BACKEND] Warning: Could not set file permissions: {e}")
                self.msg(f"Warning: Could not set file permissions: {e}", 4, 2)

        self._list_directory_contents(self.send_path_dir, "Send directory (after publish)")
        self.msg(f'Sent local data: {comm_file_path}', 4, 2)

    def publish_status_file(self, file_path, name=None):
        """Upload a status file to the local communication directory"""
        self.msg(f'Uploading status file ({self.now()})...', 4, 1)
        print(f"[DEBUG BACKEND] Publishing status file: {file_path}")

        p = Path(file_path)
        name = p.name if name is None else f'{name}{p.suffix}'
        
        # Create status directory in communication path
        status_dir = self.base_path / 'status' / self.send
        status_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = status_dir / name

        # Copy the file to the status directory
        shutil.copy2(file_path, dest_path)
        print(f"[DEBUG BACKEND] Status file copied to: {dest_path}")
        
        if os.name != 'nt':
            try:
                os.chmod(dest_path, 0o666)
            except (OSError, PermissionError) as e:
                print(f"[DEBUG BACKEND] Warning: Could not set file permissions: {e}")
                self.msg(f"Warning: Could not set file permissions: {e}", 4, 2)

        self.msg(f'Sent local status file: {dest_path}', 4, 2)

    def get_status_files(self, name='status', timestamp=False):
        """Retrieve status files from the local communication directory"""
        status_prefix = self.base_path / name
        now_str = self.now(str_format='%Y-%m-%d_%H%M%S')

        print(f"[DEBUG BACKEND] Getting status files from: {status_prefix}")
        self.msg(f'Getting status files ({self.now()})', 4, 1)
        self.msg(f'recursive searching: {status_prefix}', 4, 2)

        if not status_prefix.exists():
            print(f"[DEBUG BACKEND] Status directory does not exist: {status_prefix}")
            self.msg(f'Status directory does not exist: {status_prefix}', 3, 2)
            return

        # Walk through all files in the status directory
        file_count = 0
        for file_path in status_prefix.rglob('*'):
            if file_path.is_file():
                file_count += 1
                print(f"[DEBUG BACKEND] Downloading status file: {file_path.name}")
                self.msg(f'downloading: {file_path}', 4, 3)

                # Calculate relative path from base_path
                relative_path = file_path.relative_to(self.base_path)
                
                if timestamp:
                    dest_path = Path(self.save_dir) / 'status' / now_str / relative_path.relative_to(name)
                else:
                    dest_path = Path(self.save_dir) / relative_path

                # Create destination directory if needed
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    shutil.copy2(file_path, dest_path)
                    print(f"[DEBUG BACKEND] Copied to: {dest_path}")
                except Exception as ex:
                    print(f"[DEBUG BACKEND] Error copying file: {ex}")
                    self.msg_error('Python Exception in get_status_files', 1, 2)
                    self.print(ex)

        print(f"[DEBUG BACKEND] Downloaded {file_count} status file(s)")
        self.msg('Done.', 4, 2)

    def get(self, save=True, check_interrupted=True, force_load=False):
        """
        Get data from the queue.
        For simple queues, returns just the data.
        For multi-client queues, this is overridden to return (data, client_id, queue_reference).
        """
        print(f"[DEBUG BACKEND] get() called - force_load={force_load}")
        self.msg(f'Getting data/command ({self.now()})', 4, 1)

        if force_load or (check_interrupted and self.interrupted()):
            print(f"[DEBUG BACKEND] Loading data from disk...")
            self.msg('Loading data/command from disk...', 4, 1)
            data = self.load()
        else:
            print(f"[DEBUG BACKEND] Waiting for new data...")
            self._list_directory_contents(self.receive_path_dir, "Receive directory (before get)")
            self.msg(f'Waiting for data/command ({self.now()})...', 4, 1)
            data = self.get_s3_file()

        if isinstance(data, (list, tuple, np.ndarray)):
            print(f"[DEBUG BACKEND] Received list with length {len(data)}")
            self.msg(f'Received: list length {len(data)}', 4, 2)
        else:
            print(f"[DEBUG BACKEND] Received data (type: {type(data).__name__})")
            self.msg('Received.', 4, 2)

        return data

    def publish(self, data, save=True):
        """
        Publish data to the queue.
        For multi-client queues, this is overridden to handle client_id.
        """
        print(f"[DEBUG BACKEND] publish() called")
        self.msg(f'Sending data/command ({self.now()})...', 4, 1)
        self.publish_s3_file(data)
        self.msg('Sent.', 4, 2)

    def republish(self, stype='sent'):
        print(f"[DEBUG BACKEND] republish() called - type: {stype}")
        self.msg(f'Re-publishing the last data/command that was {stype} ({self.now()})', 4, 1)
        self.interrupted()
        data = self.load(stype=stype)
        self.publish(data)

    def interrupted(self):
        received = Path(self.save_dir) / f'{self.name}-received.npy'
        sent = Path(self.save_dir) / f'{self.name}-sent.npy'

        print(f"[DEBUG BACKEND] Checking for interruption...")
        print(f"[DEBUG BACKEND] Received file exists: {received.exists()}")
        print(f"[DEBUG BACKEND] Sent file exists: {sent.exists()}")

        if not received.exists():
            self.msg('No saved received data from prior work-cycle.', 4, 2)
            return False

        if not sent.exists():
            self.msg('No saved sent data from prior work-cycle.', 4, 2)
            return True

        received_time = received.stat().st_mtime
        sent_time = sent.stat().st_mtime

        if sent_time > received_time:
            self.msg('Last work-cycle completed normally (based on file dates).', 4, 2)
            self.msg(f'Received: {self.time_str(received_time)}', 5, 3)
            self.msg(f'Sent: {self.time_str(sent_time)} ({self.time_delta(received_time, sent_time)})', 5, 3)
            return False
        else:
            self.msg('Last work-cycle INTERRUPTED (based on file dates).', 3, 2)
            self.msg(f'Received: {self.time_str(received_time)}', 3, 3)
            self.msg(f'Sent: {self.time_str(sent_time)} ({self.time_delta(received_time, sent_time)})', 3, 3)
            return True

    def load(self, stype='received'):
        file_path = Path(self.save_dir) / f'{self.name}-{stype}.npy'
        print(f"[DEBUG BACKEND] Loading from: {file_path}")
        return np.load(file_path, allow_pickle=True)

    def clear(self, name=None):
        """Remove the saved files."""
        print(f"[DEBUG BACKEND] clear() called")
        self.msg('Clearing queue saved files.', 2, 1)

        if name is None:
            name = self.name

        for stype in ['received', 'sent']:
            file_path = Path(self.save_dir) / f'{name}-{stype}.npy'
            if file_path.exists():
                print(f"[DEBUG BACKEND] Removing {file_path}")
                self.msg(f'Removing {file_path}', 3, 2)
                file_path.unlink()
            else:
                self.msg(f'{stype.capitalize()} data does not exist ({file_path})', 3, 2)


class Queue_AI(CustomS3):
    def __init__(self, username="local", send='ai', receive='user', endpoint="local",
                 secret_key_path=None, experiment=None, name='aiS3',
                 save_dir='./', verbosity=5, **kwargs):
        print(f"[DEBUG BACKEND] Initializing Queue_AI")
        print(f"[DEBUG BACKEND] *** Using simple queue (not multi-client) ***")
        print(f"[DEBUG BACKEND] *** Frontend should use Queue_user (not MultiClientQueue_user) ***")
        super().__init__(username=username, send=send, receive=receive,
                         endpoint=endpoint, secret_key_path=secret_key_path,
                         experiment=experiment, name=name, save_dir=save_dir,
                         log_verbosity=verbosity, **kwargs)


class Queue_user(CustomS3):
    def __init__(self, username="local", send='user', receive='ai', endpoint="local",
                 secret_key_path=None, experiment=None, name='userS3',
                 save_dir='./', verbosity=5, **kwargs):
        print(f"[DEBUG BACKEND] Initializing Queue_user")
        print(f"[DEBUG BACKEND] *** Using simple queue (not multi-client) ***")
        print(f"[DEBUG BACKEND] *** Frontend should use Queue_AI (not MultiClientQueue_AI) ***")
        super().__init__(username=username, send=send, receive=receive,
                         endpoint=endpoint, secret_key_path=secret_key_path,
                         experiment=experiment, name=name, save_dir=save_dir,
                         log_verbosity=verbosity, **kwargs)


class MultiClientQueue_AI(CustomS3, MultiClientQueueBase):
    """
    Multi-client queue for AI server that can handle requests from multiple clients simultaneously.
    Each client gets its own dedicated queue path to avoid conflicts.
    Uses local filesystem for storage.
    
    This class overrides get() to return (data, client_id, queue_reference) for compatibility
    with existing code that expects: data, client_id, client_queue = queue.get()
    """

    def __init__(self, username="local", endpoint="local", secret_key_path=None,
                 experiment=None, name='multiClientAI', save_dir='./', verbosity=5, **kwargs):
        print(f"[DEBUG BACKEND] Initializing MultiClientQueue_AI")
        print(f"[DEBUG BACKEND] *** Using MULTI-CLIENT queue ***")
        print(f"[DEBUG BACKEND] *** Clients should use MultiClientQueue_user ***")
        CustomS3.__init__(self, username=username, send='ai', receive='user', endpoint=endpoint,
                        secret_key_path=secret_key_path, experiment=experiment, name=name,
                        save_dir=save_dir, log_verbosity=verbosity, **kwargs)
        MultiClientQueueBase.__init__(self)

        # Store parameters for creating client queues
        self.username = username
        self.endpoint = endpoint
        self.secret_key_path = secret_key_path
        
        # Track which client we're currently processing
        self._current_client_id = None
        
        # Track if we're currently processing a request (to avoid re-loading from disk)
        self._processing_request = False

    def _discover_clients_from_storage(self):
        """Discover new clients by checking for client-specific directories in filesystem."""
        discovered_clients = set()
        try:
            # Look for directories starting with 'user_' in the base path
            print(f"[DEBUG BACKEND] Discovering clients in: {self.base_path}")
            for path in self.base_path.glob('user_*'):
                if path.is_dir():
                    client_id = path.name[5:]  # Remove 'user_' prefix
                    discovered_clients.add(client_id)
                    print(f"[DEBUG BACKEND] Discovered client: {client_id}")

            if not discovered_clients:
                print(f"[DEBUG BACKEND] No clients discovered")

        except Exception as e:
            print(f"[DEBUG BACKEND] Error discovering clients from filesystem: {e}")

        return discovered_clients

    def _create_client_queue(self, client_id):
        """Create a new local filesystem-based queue instance for a specific client."""
        print(f"[DEBUG BACKEND] Creating client queue for: {client_id}")
        return ClientSpecificQueue(
            username=self.username,
            send=f'ai_{client_id}',
            receive=f'user_{client_id}',
            endpoint=self.endpoint,
            secret_key_path=self.secret_key_path,
            experiment=self.experiment,
            name=f'aiS3_{client_id}',
            save_dir=self.save_dir,
            verbosity=self.log_verbosity,
            client_id=client_id
        )

    def _get_client_queue_name(self, client_id):
        """Get the queue name for a specific client."""
        return f'aiS3_{client_id}'

    def get(self, save=True, check_interrupted=True, force_load=False):
        """
        Multi-client aware get() that scans for messages from any client.
        
        Returns tuple of (data, client_id, queue_reference) for compatibility with:
            data, client_id, client_queue = queue.get()
        
        The queue_reference returned is the client-specific queue object.
        """
        print(f"[DEBUG BACKEND] MultiClientQueue_AI.get() called")
        print(f"[DEBUG BACKEND] _processing_request flag: {self._processing_request}")
        self.msg(f'Getting data/command from any client ({self.now()})', 4, 1)

        # If we're currently processing a request, skip the interruption check
        # This prevents re-loading from disk when get() is called multiple times
        # before publish() is called (e.g., when cog_manager queues requests)
        if self._processing_request:
            print(f"[DEBUG BACKEND] Currently processing a request, skipping interruption check")
            check_interrupted = False

        if force_load or (check_interrupted and self.interrupted()):
            print(f"[DEBUG BACKEND] Loading data from disk...")
            self.msg('Loading data/command from disk...', 4, 1)
            data = self.load()
            # If loading from disk, we don't have client_id info
            # Mark that we're processing this request
            self._processing_request = True
            
            # Discover clients and get the client queue if we have a client_id
            if self._current_client_id:
                self.discover_new_clients()
                client_queue = self.get_client_queue(self._current_client_id)
                return (data, self._current_client_id, client_queue)
            else:
                return (data, None, self)

        print(f"[DEBUG BACKEND] Scanning for messages from all clients...")
        
        wait_count = 0
        while True:
            # Scan for client directories
            try:
                client_dirs = list(self.base_path.glob('user_*'))
                
                if wait_count % 10 == 0:
                    print(f"[DEBUG BACKEND] Scanning for clients... (attempt {wait_count + 1})")
                    print(f"[DEBUG BACKEND] Found {len(client_dirs)} client directory(s)")
                    for cdir in client_dirs:
                        print(f"[DEBUG BACKEND]   - {cdir.name}")
                
                # Check each client directory for messages
                for client_dir in client_dirs:
                    if not client_dir.is_dir():
                        continue
                    
                    client_id = client_dir.name[5:]  # Remove 'user_' prefix
                    
                    # Look for .npy files in this client's directory
                    files = sorted(client_dir.glob('*.npy'))
                    
                    if files:
                        print(f"[DEBUG BACKEND] Found {len(files)} message(s) from client: {client_id}")
                        file_path = files[0]
                        
                        try:
                            print(f"[DEBUG BACKEND] Reading file: {file_path}")
                            data = np.load(file_path, allow_pickle=True)
                            print(f"[DEBUG BACKEND] Successfully loaded data from client {client_id}")
                            
                            # Save current client ID for potential republish
                            self._current_client_id = client_id
                            
                            # Mark that we're now processing a request
                            # This prevents re-loading from disk on subsequent get() calls
                            self._processing_request = True
                            print(f"[DEBUG BACKEND] Set _processing_request flag to True")
                            
                            # Save a copy to local process directory
                            local_copy = Path(self.save_dir) / f'{self.name}-received.npy'
                            np.save(local_copy, data, allow_pickle=True)
                            print(f"[DEBUG BACKEND] Saved local copy to: {local_copy}")
                            
                            # Delete the file after reading
                            file_path.unlink()
                            print(f"[DEBUG BACKEND] Deleted original file: {file_path.name}")
                            
                            if isinstance(data, (list, tuple, np.ndarray)):
                                print(f"[DEBUG BACKEND] Received list with length {len(data)} from client {client_id}")
                                self.msg(f'Received from client {client_id}: list length {len(data)}', 4, 2)
                            else:
                                print(f"[DEBUG BACKEND] Received data from client {client_id} (type: {type(data).__name__})")
                                self.msg(f'Received from client {client_id}.', 4, 2)
                            
                            # Discover/register this client with the base class
                            print(f"[DEBUG BACKEND] Discovering clients to register {client_id}")
                            self.discover_new_clients()
                            
                            # Get the client-specific queue object
                            print(f"[DEBUG BACKEND] Getting client queue for {client_id}")
                            client_queue = self.get_client_queue(client_id)
                            print(f"[DEBUG BACKEND] Client queue retrieved: {type(client_queue).__name__}")
                            
                            # Return tuple: (data, client_id, client_queue)
                            # This allows: data, client_id, client_queue = queue.get()
                            # And then: client_queue.publish(response)
                            return (data, client_id, client_queue)
                            
                        except Exception as ex:
                            print(f"[DEBUG BACKEND] Error reading file from client {client_id}: {ex}")
                            self.msg_error(f'Error reading file from client {client_id}: {ex}', 1, 2)
                            continue
                
                # No messages found, wait and try again
                wait_count += 1
                time.sleep(0.1)
                
            except Exception as ex:
                print(f"[DEBUG BACKEND] Error scanning for clients: {ex}")
                self.msg_error(f'Error scanning for clients: {ex}', 1, 2)
                time.sleep(0.1)

    def publish(self, data, client_id=None, save=True):
        """
        Publish data to a specific client.
        If client_id is None, uses the last client that sent a message.
        
        This method should not normally be called - publish should be called
        on the client-specific queue object returned by get().
        """
        if client_id is None:
            client_id = self._current_client_id
        
        if client_id is None:
            raise ValueError("No client_id specified and no previous client to reply to")
        
        print(f"[DEBUG BACKEND] MultiClientQueue_AI.publish() called for client: {client_id}")
        self.msg(f'Sending data/command to client {client_id} ({self.now()})...', 4, 1)
        
        # Get the client-specific queue and call publish on it
        try:
            client_queue = self.get_client_queue(client_id)
            client_queue.publish(data, save=save)
            
            # Clear the processing flag - we're done with this request
            self._processing_request = False
            print(f"[DEBUG BACKEND] Set _processing_request flag to False")
            
            self.msg(f'Sent to client {client_id}.', 4, 2)
        except Exception as ex:
            print(f"[DEBUG BACKEND] Error publishing to client {client_id}: {ex}")
            self.msg_error(f'Error publishing to client {client_id}: {ex}', 1, 2)
            raise


class ClientSpecificQueue(CustomS3, ClientSpecificQueueBase):
    """
    Queue for a specific client with non-blocking get capability.
    Uses local filesystem for storage.
    """

    def __init__(self, **kwargs):
        print(f"[DEBUG BACKEND] Initializing ClientSpecificQueue")
        CustomS3.__init__(self, **kwargs)

    def get_non_blocking(self):
        """
        Non-blocking version of get() that returns None if no data is available.
        """
        print(f"[DEBUG BACKEND] get_non_blocking() called")
        try:
            # Check if there are any files in the receive path
            files = sorted(self.receive_path_dir.glob('*.npy'))

            if not files:
                print(f"[DEBUG BACKEND] No files available in receive directory")
                return None

            # Get the first available file
            file_path = files[0]
            print(f"[DEBUG BACKEND] Found file: {file_path.name}")

            try:
                data = np.load(file_path, allow_pickle=True)
                print(f"[DEBUG BACKEND] Successfully loaded data from {file_path.name}")
                # Delete the file after reading
                file_path.unlink()
                print(f"[DEBUG BACKEND] Deleted file: {file_path.name}")
                return data

            except Exception as ex:
                print(f"[DEBUG BACKEND] Error reading file {file_path}: {ex}")
                self.msg_error(f'Error reading file {file_path}: {ex}', 1, 2)
                return None

        except Exception as ex:
            print(f"[DEBUG BACKEND] Error in get_non_blocking: {ex}")
            self.msg_error(f'Error in get_non_blocking: {ex}', 1, 2)
            return None

    def publish(self, data, save=True):
        """
        Publish data to this specific client.
        Overrides the base class method to use the correct send path.
        """
        print(f"[DEBUG BACKEND] ClientSpecificQueue.publish() called")
        print(f"[DEBUG BACKEND] Send path: {self.send_path_dir}")
        
        # Just call the parent's publish method - the send_path_dir is already
        # set to the client-specific directory (ai_{client_id})
        super().publish(data, save=save)


class MultiClientQueue_user(CustomS3):
    """
    Multi-client queue for user clients that can send requests to the AI server.
    Uses local filesystem for storage.
    
    NOTE: This creates client-specific queue names (user_{client_id} and ai_{client_id}).
    The backend MUST use MultiClientQueue_AI to handle these client-specific queues.
    If the backend uses simple Queue_user, there will be a mismatch!
    """

    def __init__(self, client_id=None, username="local", endpoint="local", secret_key_path=None,
                 experiment=None, name='multiClientUser', save_dir='./', verbosity=5, **kwargs):
        # Generate client_id if not provided
        if client_id is None:
            client_id = str(uuid.uuid4())

        print(f"[DEBUG BACKEND] Initializing MultiClientQueue_user with client_id: {client_id}")
        print(f"[DEBUG BACKEND] *** Using MULTI-CLIENT queue ***")
        print(f"[DEBUG BACKEND] *** Frontend MUST use MultiClientQueue_AI ***")
        print(f"[DEBUG BACKEND] *** If frontend uses simple Queue_user, communication will FAIL ***")

        # Use client-specific queue names
        send_queue = f'user_{client_id}'
        receive_queue = f'ai_{client_id}'

        print(f"[DEBUG BACKEND] Client-specific queues: send={send_queue}, receive={receive_queue}")

        super().__init__(username=username, send=send_queue, receive=receive_queue, endpoint=endpoint,
                        secret_key_path=secret_key_path, experiment=experiment, name=name, save_dir=save_dir,
                        log_verbosity=verbosity, client_id=client_id, **kwargs)

        print(f"[DEBUG BACKEND] Full send path: {self.send_path_dir}")
        print(f"[DEBUG BACKEND] Full receive path: {self.receive_path_dir}")
