#!/usr/bin/python3
import io
import os
import time
import json
import shutil
import uuid
import numpy as np
from pathlib import Path
from .Base import Base
from .multi_client_base import MultiClientQueueBase, ClientSpecificQueueBase


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

        if experiment is None:
            # Use only date (not hour) so frontend and backend use same experiment name
            experiment = f'experiment_{self.now(str_format="%Y-%m-%d")}'

        self.experiment = experiment
        self.send = send
        self.receive = receive
        self.save_dir = save_dir
        self.bucket_name = bucket_name
        self.client_id = client_id or str(uuid.uuid4())

        print(f"\n{'=' * 80}")
        print(f"[DEBUG FRONTEND] Initializing CustomS3 queue: {name}")
        print(f"[DEBUG FRONTEND] Queue type: {self.__class__.__name__}")
        print(f"[DEBUG FRONTEND] Client ID: {self.client_id}")
        print(f"[DEBUG FRONTEND] Send queue: {send}, Receive queue: {receive}")
        print(f"[DEBUG FRONTEND] Experiment: {experiment}")

        # Use shared communication directory for inter-process/container communication
        self.comm_dir = get_shared_comm_dir()
        self.base_path = self.comm_dir / bucket_name / experiment
        self.send_path_dir = self.base_path / send
        self.receive_path_dir = self.base_path / receive

        print(f"[DEBUG FRONTEND] Communication directory: {self.comm_dir}")
        print(f"[DEBUG FRONTEND] Base path: {self.base_path}")
        print(f"[DEBUG FRONTEND] Send path: {self.send_path_dir}")
        print(f"[DEBUG FRONTEND] Receive path: {self.receive_path_dir}")
        print(f"[DEBUG FRONTEND] Local save directory: {os.path.abspath(self.save_dir)}")

        # Create all necessary directories
        os.makedirs(self.save_dir, exist_ok=True)
        self.send_path_dir.mkdir(parents=True, exist_ok=True)
        self.receive_path_dir.mkdir(parents=True, exist_ok=True)

        print(f"[DEBUG FRONTEND] Directories created successfully")

        # Set appropriate permissions for the communication directory (Unix only)
        if os.name != 'nt':  # Not Windows
            try:
                os.chmod(self.comm_dir, 0o777)
                os.chmod(self.base_path, 0o777)
                os.chmod(self.send_path_dir, 0o777)
                os.chmod(self.receive_path_dir, 0o777)
                print(f"[DEBUG FRONTEND] Permissions set to 0o777")
            except (OSError, PermissionError) as e:
                print(f"[DEBUG FRONTEND] Warning: Could not set permissions: {e}")
                self.msg(f"Warning: Could not set permissions: {e}", 3, 1)

        # Track processed files to avoid re-processing
        self._seen_objects = set()

        print(f"{'=' * 80}\n")

        self.msg(f"Using shared communication directory: {self.comm_dir}", 4, 1)
        self.msg(f"Send path: {self.send_path_dir}", 5, 2)
        self.msg(f"Receive path: {self.receive_path_dir}", 5, 2)

    def _list_directory_contents(self, dir_path, label="Directory"):
        """Helper method to list and print directory contents for debugging"""
        try:
            if not dir_path.exists():
                print(f"[DEBUG FRONTEND] {label} does not exist: {dir_path}")
                return

            files = list(dir_path.glob('*'))
            if files:
                print(f"[DEBUG FRONTEND] {label} contents ({len(files)} items):")
                for f in sorted(files):
                    size = f.stat().st_size if f.is_file() else 'DIR'
                    print(f"[DEBUG FRONTEND]   - {f.name} ({size} bytes)")
            else:
                print(f"[DEBUG FRONTEND] {label} is empty: {dir_path}")
        except Exception as e:
            print(f"[DEBUG FRONTEND] Error listing {label}: {e}")

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
        print(f"[DEBUG FRONTEND] Waiting for file in: {self.receive_path_dir}")
        wait_count = 0
        while True:
            files = sorted(self.receive_path_dir.glob('*.npy'))
            if wait_count % 10 == 0:  # Print every 10 iterations
                print(f"[DEBUG FRONTEND] Checking for files... (attempt {wait_count + 1})")
                self._list_directory_contents(self.receive_path_dir, "Receive directory")

            if files:
                print(f"[DEBUG FRONTEND] Found {len(files)} file(s) in receive directory")
                try:
                    print(f"[DEBUG FRONTEND] Reading file: {files[0].name}")
                    data = np.load(files[0], allow_pickle=True)
                    print(f"[DEBUG FRONTEND] Successfully loaded data from {files[0].name}")

                    # Save a copy to local process directory
                    local_copy = Path(self.save_dir) / f'{self.name}-received.npy'
                    np.save(local_copy, data, allow_pickle=True)
                    print(f"[DEBUG FRONTEND] Saved local copy to: {local_copy}")

                    files[0].unlink()  # Remove the file after reading
                    print(f"[DEBUG FRONTEND] Deleted original file: {files[0].name}")
                    return data
                except Exception as ex:
                    print(f"[DEBUG FRONTEND] Error reading file: {ex}")
                    self.msg_error(f'Error reading file: {ex}', 1, 2)
                    time.sleep(0.1)

            wait_count += 1
            time.sleep(0.1)

    def publish_s3_file(self, data):
        """Save data to both the communication directory and local process directory"""
        timestamp = self.now(str_format='%Y%m%d_%H%M%S_%f')

        # Save to communication directory
        comm_file_path = self.send_path_dir / f'obj_{timestamp}.npy'
        print(f"[DEBUG FRONTEND] Publishing file to: {comm_file_path}")

        np.save(comm_file_path, data, allow_pickle=True)
        file_size = comm_file_path.stat().st_size
        print(f"[DEBUG FRONTEND] File written: {comm_file_path.name} ({file_size} bytes)")

        # Save to local process directory
        local_file_path = Path(self.save_dir) / f'{self.name}-sent.npy'
        np.save(local_file_path, data, allow_pickle=True)
        print(f"[DEBUG FRONTEND] Local copy saved: {local_file_path}")

        if os.name != 'nt':
            try:
                os.chmod(comm_file_path, 0o666)
                print(f"[DEBUG FRONTEND] File permissions set to 0o666")
            except (OSError, PermissionError) as e:
                print(f"[DEBUG FRONTEND] Warning: Could not set file permissions: {e}")
                self.msg(f"Warning: Could not set file permissions: {e}", 4, 2)

        self._list_directory_contents(self.send_path_dir, "Send directory (after publish)")
        self.msg(f'Sent local data: {comm_file_path}', 4, 2)

    def publish_status_file(self, file_path, name=None):
        """Upload a status file to the local communication directory"""
        self.msg(f'Uploading status file ({self.now()})...', 4, 1)
        print(f"[DEBUG FRONTEND] Publishing status file: {file_path}")

        p = Path(file_path)
        name = p.name if name is None else f'{name}{p.suffix}'

        # Create status directory in communication path
        status_dir = self.base_path / 'status' / self.send
        status_dir.mkdir(parents=True, exist_ok=True)

        dest_path = status_dir / name

        # Copy the file to the status directory
        shutil.copy2(file_path, dest_path)
        print(f"[DEBUG FRONTEND] Status file copied to: {dest_path}")

        if os.name != 'nt':
            try:
                os.chmod(dest_path, 0o666)
            except (OSError, PermissionError) as e:
                print(f"[DEBUG FRONTEND] Warning: Could not set file permissions: {e}")
                self.msg(f"Warning: Could not set file permissions: {e}", 4, 2)

        self.msg(f'Sent local status file: {dest_path}', 4, 2)

    def get_status_files(self, name='status', timestamp=False):
        """Retrieve status files from the local communication directory"""
        status_prefix = self.base_path / name
        now_str = self.now(str_format='%Y-%m-%d_%H%M%S')

        print(f"[DEBUG FRONTEND] Getting status files from: {status_prefix}")
        self.msg(f'Getting status files ({self.now()})', 4, 1)
        self.msg(f'recursive searching: {status_prefix}', 4, 2)

        if not status_prefix.exists():
            print(f"[DEBUG FRONTEND] Status directory does not exist: {status_prefix}")
            self.msg(f'Status directory does not exist: {status_prefix}', 3, 2)
            return

        # Walk through all files in the status directory
        file_count = 0
        for file_path in status_prefix.rglob('*'):
            if file_path.is_file():
                file_count += 1
                print(f"[DEBUG FRONTEND] Downloading status file: {file_path.name}")
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
                    print(f"[DEBUG FRONTEND] Copied to: {dest_path}")
                except Exception as ex:
                    print(f"[DEBUG FRONTEND] Error copying file: {ex}")
                    self.msg_error('Python Exception in get_status_files', 1, 2)
                    self.print(ex)

        print(f"[DEBUG FRONTEND] Downloaded {file_count} status file(s)")
        self.msg('Done.', 4, 2)

    def get(self, save=True, check_interrupted=True, force_load=False, return_all=False):
        """Get the current item being published, optionally returning all available messages"""
        print(f"[DEBUG FRONTEND] get() called - force_load={force_load}, return_all={return_all}")
        self.msg(f'Getting data/command ({self.now()})', 4, 1)

        if force_load or (check_interrupted and self.interrupted()):
            print(f"[DEBUG FRONTEND] Loading data from disk...")
            self.msg('Loading data/command from disk...', 4, 1)
            data = self.load()

            if isinstance(data, (list, tuple, np.ndarray)):
                print(f"[DEBUG FRONTEND] Loaded list with length {len(data)}")
                self.msg(f'Loaded: list length {len(data)}', 4, 2)
            else:
                print(f"[DEBUG FRONTEND] Loaded data (type: {type(data).__name__})")
                self.msg('Loaded.', 4, 2)
        else:
            print(f"[DEBUG FRONTEND] Waiting for new messages...")
            self._list_directory_contents(self.receive_path_dir, "Receive directory (before get)")

            messages = []
            while not messages:  # wait until at least one new file is available
                # List all files and sort chronologically
                all_files = sorted(self.receive_path_dir.glob('*.npy'), key=lambda f: f.stat().st_mtime)

                if all_files:
                    print(f"[DEBUG FRONTEND] Found {len(all_files)} file(s) in receive directory")

                for file_path in all_files:
                    file_str = str(file_path)
                    if file_str in self._seen_objects:
                        continue  # already consumed

                    print(f"[DEBUG FRONTEND] Processing new file: {file_path.name}")
                    tmp_file = Path(self.save_dir) / f'_tmp_{file_path.name}'
                    try:
                        shutil.copy2(file_path, tmp_file)
                        msg = np.load(tmp_file, allow_pickle=True)
                        messages.append(msg)
                        self._seen_objects.add(file_str)
                        print(f"[DEBUG FRONTEND] Successfully loaded message from {file_path.name}")

                        if tmp_file.exists():
                            tmp_file.unlink()
                            self.msg(f"{tmp_file} deleted.", 4, 3)
                    except Exception as ex:
                        print(f"[DEBUG FRONTEND] Error processing file {file_path.name}: {str(ex)}")
                        self.msg_error(f'Error processing file {file_path.name}: {str(ex)}', 1, 2)

                if not messages:
                    time.sleep(0.05)  # short pause before retrying

            # If caller wants everything (streaming) return the full list.
            # Otherwise deliver only the newest message to avoid flooding
            # single-shot request/reply calls with a backlog.
            if return_all:
                data = messages
                print(f"[DEBUG FRONTEND] Returning all {len(messages)} message(s)")
            else:
                data = messages[-1]  # newest (list is chronological)
                print(f"[DEBUG FRONTEND] Returning newest message (out of {len(messages)})")

            if isinstance(data, (list, tuple, np.ndarray)):
                if return_all:
                    self.msg(f'Received: {len(data)} messages', 4, 2)
                else:
                    self.msg(f'Received: list length {len(data)}', 4, 2)
            else:
                self.msg('Received.', 4, 2)

        return data

    def publish(self, data, save=True):
        print(f"[DEBUG FRONTEND] publish() called")
        self.msg(f'Sending data/command ({self.now()})...', 4, 1)
        self.publish_s3_file(data)
        self.msg('Sent.', 4, 2)

    def republish(self, stype='sent'):
        print(f"[DEBUG FRONTEND] republish() called - type: {stype}")
        self.msg(f'Re-publishing the last data/command that was {stype} ({self.now()})', 4, 1)
        self.interrupted()
        data = self.load(stype=stype)
        self.publish(data)

    def interrupted(self):
        received = Path(self.save_dir) / f'{self.name}-received.npy'
        sent = Path(self.save_dir) / f'{self.name}-sent.npy'

        print(f"[DEBUG FRONTEND] Checking for interruption...")
        print(f"[DEBUG FRONTEND] Received file exists: {received.exists()}")
        print(f"[DEBUG FRONTEND] Sent file exists: {sent.exists()}")

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
        print(f"[DEBUG FRONTEND] Loading from: {file_path}")
        return np.load(file_path, allow_pickle=True)

    def flush(self):
        """
        Mark all current files in the receive queue as already seen.
        This ensures that only new files (created after this call)
        are delivered on next get().
        """
        print(f"[DEBUG FRONTEND] flush() called")
        try:
            files = list(self.receive_path_dir.glob('*.npy'))
            print(f"[DEBUG FRONTEND] Flushing {len(files)} file(s)")
            for file_path in files:
                self._seen_objects.add(str(file_path))
        except Exception as ex:
            print(f"[DEBUG FRONTEND] flush() failed: {ex}")
            self.msg_warning(f'flush() failed: {ex}', 2, 1)

    def clear(self):
        """Remove saved files for this queue and reset seen-object tracking"""
        print(f"[DEBUG FRONTEND] clear() called")
        self.msg('Clearing queue saved files.', 2, 1)

        for stype in ['received', 'sent']:
            file_path = Path(self.save_dir) / f'{self.name}-{stype}.npy'
            if file_path.exists():
                print(f"[DEBUG FRONTEND] Removing {file_path}")
                self.msg(f'Removing {file_path}', 3, 2)
                file_path.unlink()
            else:
                self.msg(f'{stype.capitalize()} data does not exist ({file_path})', 3, 2)

        # Reset the seen-object tracking
        self._seen_objects.clear()
        print(f"[DEBUG FRONTEND] Cleared seen objects tracking")


class Queue_AI(CustomS3):
    def __init__(self, username="local", send='ai', receive='user', endpoint="local",
                 secret_key_path=None, experiment=None, name='aiS3',
                 save_dir='./', verbosity=5, **kwargs):
        print(f"[DEBUG FRONTEND] Initializing Queue_AI")
        print(f"[DEBUG FRONTEND] *** Using simple queue (not multi-client) ***")
        print(f"[DEBUG FRONTEND] *** Backend should use Queue_user (not MultiClientQueue_user) ***")
        super().__init__(username=username, send=send, receive=receive,
                         endpoint=endpoint, secret_key_path=secret_key_path,
                         experiment=experiment, name=name, save_dir=save_dir,
                         log_verbosity=verbosity, **kwargs)


class Queue_user(CustomS3):
    def __init__(self, username="local", send='user', receive='ai', endpoint="local",
                 secret_key_path=None, experiment=None, name='userS3',
                 save_dir='./', verbosity=5, **kwargs):
        print(f"[DEBUG FRONTEND] Initializing Queue_user")
        print(f"[DEBUG FRONTEND] *** Using simple queue (not multi-client) ***")
        print(f"[DEBUG FRONTEND] *** Backend should use Queue_AI (not MultiClientQueue_AI) ***")
        super().__init__(username=username, send=send, receive=receive,
                         endpoint=endpoint, secret_key_path=secret_key_path,
                         experiment=experiment, name=name, save_dir=save_dir,
                         log_verbosity=verbosity, **kwargs)


class MultiClientQueue_AI(CustomS3, MultiClientQueueBase):
    """
    Multi-client queue for AI server that can handle requests from multiple clients simultaneously.
    Each client gets its own dedicated queue path to avoid conflicts.
    Uses local filesystem for storage.
    """

    def __init__(self, username="local", endpoint="local", secret_key_path=None,
                 experiment=None, name='multiClientAI', save_dir='./', verbosity=5, **kwargs):
        print(f"[DEBUG FRONTEND] Initializing MultiClientQueue_AI")
        print(f"[DEBUG FRONTEND] *** Using MULTI-CLIENT queue ***")
        print(f"[DEBUG FRONTEND] *** Clients should use MultiClientQueue_user ***")
        CustomS3.__init__(self, username=username, send='ai', receive='user', endpoint=endpoint,
                          secret_key_path=secret_key_path, experiment=experiment, name=name,
                          save_dir=save_dir, log_verbosity=verbosity, **kwargs)
        MultiClientQueueBase.__init__(self)

        # Store parameters for creating client queues
        self.username = username
        self.endpoint = endpoint
        self.secret_key_path = secret_key_path

    def _discover_clients_from_storage(self):
        """Discover new clients by checking for client-specific directories in filesystem."""
        discovered_clients = set()
        try:
            # Look for directories starting with 'user_' in the base path
            print(f"[DEBUG FRONTEND] Discovering clients in: {self.base_path}")
            for path in self.base_path.glob('user_*'):
                if path.is_dir():
                    client_id = path.name[5:]  # Remove 'user_' prefix
                    discovered_clients.add(client_id)
                    print(f"[DEBUG FRONTEND] Discovered client: {client_id}")

            if not discovered_clients:
                print(f"[DEBUG FRONTEND] No clients discovered")

        except Exception as e:
            print(f"[DEBUG FRONTEND] Error discovering clients from filesystem: {e}")

        return discovered_clients

    def _create_client_queue(self, client_id):
        """Create a new local filesystem-based queue instance for a specific client."""
        print(f"[DEBUG FRONTEND] Creating client queue for: {client_id}")
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


class ClientSpecificQueue(CustomS3, ClientSpecificQueueBase):
    """
    Queue for a specific client with non-blocking get capability.
    Uses local filesystem for storage.
    """

    def __init__(self, **kwargs):
        print(f"[DEBUG FRONTEND] Initializing ClientSpecificQueue")
        CustomS3.__init__(self, **kwargs)

    def get_non_blocking(self):
        """
        Non-blocking version of get() that returns None if no data is available.
        """
        print(f"[DEBUG FRONTEND] get_non_blocking() called")
        try:
            # Check if there are any files in the receive path
            files = sorted(self.receive_path_dir.glob('*.npy'))

            if not files:
                print(f"[DEBUG FRONTEND] No files available in receive directory")
                return None

            # Get the first available file
            file_path = files[0]
            print(f"[DEBUG FRONTEND] Found file: {file_path.name}")

            try:
                data = np.load(file_path, allow_pickle=True)
                print(f"[DEBUG FRONTEND] Successfully loaded data from {file_path.name}")
                # Delete the file after reading
                file_path.unlink()
                print(f"[DEBUG FRONTEND] Deleted file: {file_path.name}")
                return data

            except Exception as ex:
                print(f"[DEBUG FRONTEND] Error reading file {file_path}: {ex}")
                self.msg_error(f'Error reading file {file_path}: {ex}', 1, 2)
                return None

        except Exception as ex:
            print(f"[DEBUG FRONTEND] Error in get_non_blocking: {ex}")
            self.msg_error(f'Error in get_non_blocking: {ex}', 1, 2)
            return None


class MultiClientQueue_user(CustomS3):
    """
    Multi-client queue for user clients that can send requests to the AI server.
    Uses local filesystem for storage.

    NOTE: This creates client-specific queue names (user_{client_id} and ai_{client_id}).
    The backend MUST use MultiClientQueue_AI to handle these client-specific queues.
    If the backend uses simple Queue_AI, there will be a mismatch!
    """

    def __init__(self, client_id=None, username="local", endpoint="local", secret_key_path=None,
                 experiment=None, name='multiClientUser', save_dir='./', verbosity=5, **kwargs):
        # Generate client_id if not provided
        if client_id is None:
            client_id = str(uuid.uuid4())

        print(f"[DEBUG FRONTEND] Initializing MultiClientQueue_user with client_id: {client_id}")
        print(f"[DEBUG FRONTEND] *** Using MULTI-CLIENT queue ***")
        print(f"[DEBUG FRONTEND] *** Backend MUST use MultiClientQueue_AI ***")
        print(f"[DEBUG FRONTEND] *** If backend uses simple Queue_AI, communication will FAIL ***")

        # Use client-specific queue names
        send_queue = f'user_{client_id}'
        receive_queue = f'ai_{client_id}'

        print(f"[DEBUG FRONTEND] Client-specific queues: send={send_queue}, receive={receive_queue}")

        super().__init__(username=username, send=send_queue, receive=receive_queue, endpoint=endpoint,
                         secret_key_path=secret_key_path, experiment=experiment, name=name, save_dir=save_dir,
                         log_verbosity=verbosity, client_id=client_id, **kwargs)

        print(f"[DEBUG FRONTEND] Full send path: {self.send_path_dir}")
        print(f"[DEBUG FRONTEND] Full receive path: {self.receive_path_dir}")
