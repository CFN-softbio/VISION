#!/usr/bin/python3
import io
import os
import shutil
import time
import uuid
from pathlib import Path
import paramiko
import numpy as np
from Base import Base
from src.hal_beam_com.model_manager import ModelManager
from src.hal_beam_com.multi_client_base import MultiClientQueueBase, ClientSpecificQueueBase
from utils import model_configurations, ACTIVE_CONFIG, get_finetuned_config


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
    def __init__(self,
                 username,
                 send='send',
                 receive='receive',
                 endpoint="localhost",
                 password=None,
                 port=22,
                 remote_base_path=None,  # Will default to user's home directory
                 bucket_name='transmissions',
                 experiment=None,
                 name='SSH',
                 save_dir='./',
                 log_verbosity=5,
                 clear_on_start=True,
                 client_id=None,
                 **kwargs):

        super().__init__(name=name, log_verbosity=log_verbosity, **kwargs)

        # Initialize models before starting the queue
        if not hasattr(self, '_models_initialized'):
            initialize_models()
            CustomS3._models_initialized = True

        if clear_on_start:
            self.clear()  # Clear any leftover files from previous sessions

        self._last_processed_time = 0

        # Get password from env variables if not provided
        if password is None:
            password = os.environ.get('SSH_PASSWORD')
            if password is None:
                raise EnvironmentError("SSH_PASSWORD environment variable is not set.")

        self._password = password  # Store for potential reconnection
        self._username = username
        self._endpoint = endpoint
        self._port = port

        # Setup SSH client
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.ssh.connect(endpoint, port, username, password)
            self.sftp = self.ssh.open_sftp()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to SSH server: {str(e)}")

        # Get user's home directory if remote_base_path not specified
        if remote_base_path is None:
            try:
                self.remote_base_path = self.sftp.normalize('.')  # Get current (home) directory
            except:
                self.remote_base_path = f'/home/{username}'  # Fallback
        else:
            self.remote_base_path = remote_base_path

        self.bucket_name = bucket_name
        if experiment is None:
            experiment = f'experiment_{self.now(str_format="%Y-%m-%d_%H")}'
        self.experiment = experiment
        self.send = send
        self.receive = receive
        self.save_dir = save_dir
        self.client_id = client_id or str(uuid.uuid4())

        print(f"\n{'='*80}")
        print(f"[DEBUG BACKEND] Initializing CustomS3 queue: {name}")
        print(f"[DEBUG BACKEND] Queue type: {self.__class__.__name__}")
        print(f"[DEBUG BACKEND] Client ID: {self.client_id}")
        print(f"[DEBUG BACKEND] Send queue: {send}, Receive queue: {receive}")
        print(f"[DEBUG BACKEND] Experiment: {experiment}")
        print(f"[DEBUG BACKEND] Remote base path: {self.remote_base_path}")
        print(f"{'='*80}\n")

        # Create remote directories if they don't exist
        self._ensure_remote_paths()

    def _ensure_remote_paths(self):
        """Create necessary remote directories"""
        paths = [
            f"{self.remote_base_path}/vision_data",
            f"{self.remote_base_path}/vision_data/{self.bucket_name}",
            f"{self.remote_base_path}/vision_data/{self.bucket_name}/{self.experiment}",
            f"{self.remote_base_path}/vision_data/{self.bucket_name}/{self.experiment}/{self.send}",
            f"{self.remote_base_path}/vision_data/{self.bucket_name}/{self.experiment}/{self.receive}"
        ]

        for path in paths:
            try:
                try:
                    self.sftp.stat(path)
                except FileNotFoundError:
                    self.msg(f'Creating directory: {path}', 4, 2)
                    self.sftp.mkdir(path)
            except Exception as e:
                self.msg_error(f'Failed to create directory {path}: {str(e)}', 1, 2)
                raise

    def send_path(self):
        return f'{self.remote_base_path}/vision_data/{self.bucket_name}/{self.experiment}/{self.send}'

    def receive_path(self):
        return f'{self.remote_base_path}/vision_data/{self.bucket_name}/{self.experiment}/{self.receive}'

    def get_s3_file(self):
        """Wait for new file in receive directory and download it"""
        receive_path = self.receive_path()
        local_path = f'{self.save_dir}/{self.name}-received.npy'

        print(f"[DEBUG BACKEND] Waiting for file in: {receive_path}")
        current_time = time.time()
        # Only consider files newer than last request
        self._last_processed_time = max(self._last_processed_time, current_time - 30)  # 30 second window

        wait_count = 0
        while True:
            try:
                if wait_count % 10 == 0:
                    print(f"[DEBUG BACKEND] Checking for files... (attempt {wait_count + 1})")
                    self.msg(f'Checking directory: {receive_path}', 4, 2)

                try:
                    self.sftp.stat(receive_path)
                except FileNotFoundError:
                    self._ensure_remote_paths()
                    time.sleep(1)
                    wait_count += 1
                    continue

                # When filtering files:
                files = []
                for f in self.sftp.listdir(receive_path):
                    try:
                        stat = self.sftp.stat(f'{receive_path}/{f}')
                        if stat.st_mtime > self._last_processed_time:
                            files.append((f, stat.st_mtime))
                    except Exception as e:
                        self.msg_warning(f'Failed to stat file {f}: {str(e)}', 3, 2)
                        continue

                # Sort by modification time
                files.sort(key=lambda x: x[1])

                if files:
                    print(f"[DEBUG BACKEND] Found {len(files)} new file(s)")
                    self.msg(f'Found {len(files)} new files', 4, 2)
                    for fname, mtime in files:
                        self.msg(f'File: {fname}, Modified: {time.ctime(mtime)}', 4, 2)

                    # Get newest file
                    newest_file, newest_time = files[-1]
                    remote_file = f'{receive_path}/{newest_file}'

                    print(f"[DEBUG BACKEND] Processing file: {newest_file}")
                    self.msg(f'Processing file: {remote_file}', 4, 2)

                    # Download and verify
                    self.sftp.get(remote_file, local_path)

                    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                        raise ValueError("Downloaded file is empty or missing")

                    # Load data
                    loaded_data = np.load(local_path, allow_pickle=True).item()
                    if loaded_data['timestamp'] <= self._last_processed_time:
                        self.msg_warning('Received old data, waiting for new data...', 3, 2)
                        continue
                    
                    print(f"[DEBUG BACKEND] Successfully loaded data from {newest_file}")
                    return loaded_data['data']

                else:
                    if wait_count % 10 == 0:
                        print(f"[DEBUG BACKEND] No new files found")
                    self.msg('No new files found, waiting...', 4, 2)

            except Exception as ex:
                print(f"[DEBUG BACKEND] Error in get_s3_file: {ex}")
                self.msg_error(f'Error in get_s3_file: {str(ex)}', 1, 2)
                # Try to reconnect if needed
                self._try_reconnect()

            wait_count += 1
            time.sleep(1)

    def _cleanup_remote_files(self, path, files):
        """Clean up processed files"""
        try:
            # Keep only the most recent file
            for fname, mtime in files[:-1]:
                try:
                    remote_file = f'{path}/{fname}'
                    self.msg(f'Removing old file: {remote_file}', 4, 2)
                    self.sftp.remove(remote_file)
                except Exception as e:
                    self.msg_warning(f'Failed to remove file {fname}: {str(e)}', 3, 2)
        except Exception as e:
            self.msg_error(f'Error during cleanup: {str(e)}', 2, 2)

    def publish_s3_file(self, data):
        """Save data to file and upload to remote server"""
        # Add timestamp to data
        timestamped_data = {
            'timestamp': time.time(),
            'data': data
        }

        try:
            # Clear old files before publishing new ones
            self.clear()

            local_path = f'{self.save_dir}/{self.name}-sent.npy'
            max_retries = 10

            print(f"[DEBUG BACKEND] Publishing file to remote server")
            try:
                # Save local file
                self.msg(f'Saving data to local file: {local_path}', 4, 2)
                np.save(local_path, timestamped_data, allow_pickle=True)

                # Verify local file exists and has content
                if not os.path.exists(local_path):
                    raise FileNotFoundError("Failed to create local file")
                if os.path.getsize(local_path) == 0:
                    raise ValueError("Local file is empty")

                # Verify remote path exists
                send_path = self.send_path()
                try:
                    self.sftp.stat(send_path)
                except FileNotFoundError:
                    self.msg_error(f'Send directory not found, recreating: {send_path}', 3, 2)
                    self._ensure_remote_paths()

                remote_filename = f'obj_{self.now(str_format="%Y%m%d_%H%M%S_%f")}.npy'
                remote_path = f'{send_path}/{remote_filename}'

                # Upload file with retries
                for attempt in range(max_retries):
                    try:
                        # Upload file
                        self.msg(f'Uploading to remote path (attempt {attempt + 1}/{max_retries}): {remote_path}', 4, 2)
                        self.sftp.put(local_path, remote_path)

                        # Wait a bit for the file system to catch up
                        time.sleep(0.5)

                        # Verify remote file exists and has correct size
                        try:
                            remote_stat = self.sftp.stat(remote_path)
                            local_size = os.path.getsize(local_path)
                            remote_size = remote_stat.st_size

                            self.msg(f'Local file size: {local_size} bytes', 4, 2)
                            self.msg(f'Remote file size: {remote_size} bytes', 4, 2)

                            if remote_size == local_size:
                                print(f"[DEBUG BACKEND] Successfully uploaded file (size: {remote_size} bytes)")
                                self.msg(f'Successfully uploaded file (size: {remote_size} bytes)', 4, 2)
                                return  # Success!
                            else:
                                raise ValueError(f"Size mismatch: local={local_size}, remote={remote_size}")

                        except FileNotFoundError:
                            if attempt < max_retries - 1:
                                self.msg_warning(f'Remote file not found after upload, retrying...', 3, 2)
                                time.sleep(1)  # Wait longer before retry
                                continue
                            else:
                                raise

                    except Exception as ex:
                        if attempt < max_retries - 1:
                            self.msg_warning(f'Upload attempt {attempt + 1} failed: {str(ex)}', 3, 2)
                            time.sleep(1)  # Wait before retry
                            continue
                        else:
                            raise

                raise Exception(f"Failed to upload file after {max_retries} attempts")

            except Exception as ex:
                print(f"[DEBUG BACKEND] Error in publish_s3_file: {ex}")
                self.msg_error(f'Error in publish_s3_file: {str(ex)}', 1, 2)
                raise
        except Exception as ex:
            self.msg_error(f'Error in publish_s3_file: {str(ex)}', 1, 2)
            raise

    def publish_status_file(self, file_path, name=None):
        """Upload a status file to the remote server via SFTP"""
        self.msg(f'Uploading status file ({self.now()})...', 4, 1)

        p = Path(file_path)
        name = p.name if name is None else f'{name}{p.suffix}'
        
        # Create remote status directory path
        remote_status_dir = f'{self.remote_base_path}/vision_data/{self.bucket_name}/{self.experiment}/status/{self.send}'
        remote_file_path = f'{remote_status_dir}/{name}'

        try:
            # Ensure remote status directory exists
            self._ensure_remote_directory(remote_status_dir)

            # Upload the file
            self.sftp.put(str(file_path), remote_file_path)
            
            self.msg(f'Sent SSH status file: {remote_file_path}', 4, 2)

        except Exception as ex:
            self.msg_error(f'Error uploading status file: {str(ex)}', 1, 2)
            raise

    def get_status_files(self, name='status', timestamp=False):
        """Download status files from the remote server via SFTP"""
        remote_prefix = f'{self.remote_base_path}/vision_data/{self.bucket_name}/{self.experiment}/{name}'
        now_str = self.now(str_format='%Y-%m-%d_%H%M%S')

        self.msg(f'Getting status files ({self.now()})', 4, 1)
        self.msg(f'recursive searching: {remote_prefix}', 4, 2)

        try:
            # Check if remote directory exists
            try:
                self.sftp.stat(remote_prefix)
            except FileNotFoundError:
                self.msg(f'Status directory does not exist: {remote_prefix}', 3, 2)
                return

            # Recursively download files
            self._download_directory_recursive(remote_prefix, name, timestamp, now_str)

            self.msg('Done.', 4, 2)

        except Exception as ex:
            self.msg_error(f'Error in get_status_files: {str(ex)}', 1, 2)

    def _download_directory_recursive(self, remote_path, base_name, timestamp, now_str):
        """Recursively download all files from a remote directory"""
        try:
            # List all items in the directory
            for item in self.sftp.listdir_attr(remote_path):
                remote_item_path = f'{remote_path}/{item.filename}'
                
                # Calculate relative path
                relative_path = remote_item_path.split(f'{self.experiment}/{base_name}/')[-1]
                
                if self._is_directory(item):
                    # Recursively process subdirectories
                    self._download_directory_recursive(remote_item_path, base_name, timestamp, now_str)
                else:
                    # Download file
                    self.msg(f'downloading: {remote_item_path}', 4, 3)
                    
                    if timestamp:
                        local_file_path = Path(self.save_dir) / base_name / now_str / relative_path
                    else:
                        local_file_path = Path(self.save_dir) / base_name / relative_path

                    # Create local directory if needed
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        self.sftp.get(remote_item_path, str(local_file_path))
                    except Exception as ex:
                        self.msg_error(f'Error downloading {remote_item_path}: {str(ex)}', 1, 2)

        except Exception as ex:
            self.msg_error(f'Error in _download_directory_recursive: {str(ex)}', 1, 2)

    def _is_directory(self, item):
        """Check if an SFTP item is a directory"""
        import stat
        return stat.S_ISDIR(item.st_mode)

    def _ensure_remote_directory(self, path):
        """Ensure a remote directory exists, creating it if necessary"""
        try:
            self.sftp.stat(path)
        except FileNotFoundError:
            # Directory doesn't exist, create it recursively
            parent = os.path.dirname(path)
            if parent and parent != path:
                self._ensure_remote_directory(parent)
            self.msg(f'Creating remote directory: {path}', 4, 2)
            self.sftp.mkdir(path)

    def _try_reconnect(self):
        """Try to reconnect SFTP session"""
        try:
            self.sftp.stat('.')
        except:
            self.msg_warning('Lost SFTP connection, attempting to reconnect...', 2, 2)
            try:
                self.sftp.close()
                self.sftp = self.ssh.open_sftp()
            except:
                self.ssh.connect(self._endpoint, self._port, self._username, self._password)
                self.sftp = self.ssh.open_sftp()

    def clear(self, name=None):
        """Remove all files in remote and local directories"""
        try:
            if name is None:
                name = self.name

            # Clear remote files
            receive_path = self.receive_path()
            send_path = self.send_path()

            for path in [receive_path, send_path]:
                try:
                    files = self.sftp.listdir(path)
                    for f in files:
                        try:
                            self.msg(f'Removing file: {path}/{f}', 4, 2)
                            self.sftp.remove(f'{path}/{f}')
                        except Exception as e:
                            self.msg_warning(f'Failed to remove {f}: {str(e)}', 3, 2)
                except Exception as e:
                    self.msg_warning(f'Failed to clear path {path}: {str(e)}', 3, 2)

            # Clear local files
            for stype in ['received', 'sent']:
                path = f'{self.save_dir}/{name}-{stype}.npy'
                if os.path.exists(path):
                    os.remove(path)
                    self.msg(f'Removed local file: {path}', 4, 2)

            # Reset timestamp
            self._last_processed_time = time.time()  # Update to current time

        except Exception as e:
            self.msg_error(f'Error during clear: {str(e)}', 1, 2)

    def __del__(self):
        """Cleanup SSH connections"""
        try:
            if hasattr(self, 'sftp'):
                self.sftp.close()
            if hasattr(self, 'ssh'):
                self.ssh.close()
        except:
            pass

    def close(self):
        """Safely close connections"""
        try:
            if hasattr(self, 'sftp'):
                self.sftp.close()
            if hasattr(self, 'ssh'):
                self.ssh.close()
            self.msg('Connections closed successfully', 4, 2)
        except Exception as ex:
            self.msg_error(f'Error closing connections: {str(ex)}', 1, 2)

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
        received = f'{self.save_dir}/{self.name}-received.npy'
        sent = f'{self.save_dir}/{self.name}-sent.npy'

        print(f"[DEBUG BACKEND] Checking for interruption...")
        print(f"[DEBUG BACKEND] Received file exists: {os.path.exists(received)}")
        print(f"[DEBUG BACKEND] Sent file exists: {os.path.exists(sent)}")

        if not os.path.exists(received):
            self.msg('No saved received data from prior work-cycle.', 4, 2)
            return False

        if not os.path.exists(sent):
            self.msg('No saved sent data from prior work-cycle.', 4, 2)
            return True

        received_time = os.path.getmtime(received)
        sent_time = os.path.getmtime(sent)

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
        file_path = f'{self.save_dir}/{self.name}-{stype}.npy'
        print(f"[DEBUG BACKEND] Loading from: {file_path}")
        return np.load(file_path, allow_pickle=True)


# Two-party connection:
########################################

VERBOSITY = 4
ENDPOINT = ""  # Your SSH endpoint (ip or domain)
USERNAME = ""  # Your SSH username
PASSWORD = ""  # Or use SSH_PASSWORD env variable
EXPERIMENT = "experiment"  # Can be anything, just make sure to match it on both sides

# You can specify a custom path if needed
REMOTE_PATH = None  # Will default to user's home directory


class Queue_AI(CustomS3):
    def __init__(self, username=USERNAME, send='ai', receive='user', endpoint=ENDPOINT, password=PASSWORD,
                 experiment=EXPERIMENT, name='aiSSH', save_dir='./', verbosity=VERBOSITY,
                 remote_base_path=REMOTE_PATH, **kwargs):
        print(f"[DEBUG BACKEND] Initializing Queue_AI")
        print(f"[DEBUG BACKEND] *** Using simple queue (not multi-client) ***")
        print(f"[DEBUG BACKEND] *** Frontend should use Queue_user (not MultiClientQueue_user) ***")
        super().__init__(username=username, send=send, receive=receive, endpoint=endpoint, password=password,
                         experiment=experiment, name=name, save_dir=save_dir, log_verbosity=verbosity,
                         remote_base_path=remote_base_path, **kwargs)


class Queue_user(CustomS3):
    def __init__(self, username=USERNAME, send='user', receive='ai', endpoint=ENDPOINT, password=PASSWORD,
                 experiment=EXPERIMENT, name='userSSH', save_dir='./', verbosity=VERBOSITY,
                 remote_base_path=REMOTE_PATH, **kwargs):
        print(f"[DEBUG BACKEND] Initializing Queue_user")
        print(f"[DEBUG BACKEND] *** Using simple queue (not multi-client) ***")
        print(f"[DEBUG BACKEND] *** Frontend should use Queue_AI (not MultiClientQueue_AI) ***")
        super().__init__(username=username, send=send, receive=receive, endpoint=endpoint, password=password,
                         experiment=experiment, name=name, save_dir=save_dir, log_verbosity=verbosity,
                         remote_base_path=remote_base_path, **kwargs)


class MultiClientQueue_AI(CustomS3, MultiClientQueueBase):
    """
    Multi-client queue for AI server that can handle requests from multiple clients simultaneously.
    Each client gets its own dedicated queue path to avoid conflicts.
    Uses SSH/SFTP backend for storage.
    
    This class overrides get() to return (data, client_id, queue_reference) for compatibility
    with existing code that expects: data, client_id, client_queue = queue.get()
    """

    def __init__(self, username=USERNAME, endpoint=ENDPOINT, password=PASSWORD, port=22,
                 experiment=EXPERIMENT, name='multiClientAI', save_dir='./', verbosity=VERBOSITY,
                 remote_base_path=REMOTE_PATH, **kwargs):
        print(f"[DEBUG BACKEND] Initializing MultiClientQueue_AI")
        print(f"[DEBUG BACKEND] *** Using MULTI-CLIENT queue ***")
        print(f"[DEBUG BACKEND] *** Clients should use MultiClientQueue_user ***")
        CustomS3.__init__(self, username=username, send='ai', receive='user', endpoint=endpoint,
                        password=password, port=port, experiment=experiment, name=name,
                        save_dir=save_dir, log_verbosity=verbosity, remote_base_path=remote_base_path,
                        clear_on_start=False, **kwargs)
        MultiClientQueueBase.__init__(self)

        # Store parameters for creating client queues
        self.username = username
        self.endpoint = endpoint
        self.password = password
        self.port = port
        self.remote_base_path_param = remote_base_path
        
        # Track which client we're currently processing
        self._current_client_id = None
        
        # Track if we're currently processing a request (to avoid re-loading from disk)
        self._processing_request = False

    def _discover_clients_from_storage(self):
        """Discover new clients by checking for client-specific directories via SSH."""
        discovered_clients = set()
        try:
            # Look for directories starting with 'user_' in the base experiment path
            base_path = f'{self.remote_base_path}/vision_data/{self.bucket_name}/{self.experiment}'
            
            print(f"[DEBUG BACKEND] Discovering clients in: {base_path}")
            try:
                # List all items in the directory
                for item in self.sftp.listdir_attr(base_path):
                    if self._is_directory(item) and item.filename.startswith('user_'):
                        client_id = item.filename[5:]  # Remove 'user_' prefix
                        discovered_clients.add(client_id)
                        print(f"[DEBUG BACKEND] Discovered client: {client_id}")
            except FileNotFoundError:
                self.msg(f'Base path does not exist yet: {base_path}', 4, 2)
                print(f"[DEBUG BACKEND] Base path does not exist yet: {base_path}")
            except Exception as e:
                self.msg_error(f'Error listing directory {base_path}: {str(e)}', 2, 2)

            if not discovered_clients:
                print(f"[DEBUG BACKEND] No clients discovered")

        except Exception as e:
            print(f"[DEBUG BACKEND] Error discovering clients from SSH: {e}")

        return discovered_clients

    def _create_client_queue(self, client_id):
        """Create a new SSH-based queue instance for a specific client."""
        print(f"[DEBUG BACKEND] Creating client queue for: {client_id}")
        return ClientSpecificQueue(
            username=self.username,
            send=f'ai_{client_id}',
            receive=f'user_{client_id}',
            endpoint=self.endpoint,
            password=self.password,
            port=self.port,
            experiment=self.experiment,
            name=f'aiSSH_{client_id}',
            save_dir=self.save_dir,
            verbosity=self.log_verbosity,
            remote_base_path=self.remote_base_path_param,
            client_id=client_id,
            clear_on_start=False
        )

    def _get_client_queue_name(self, client_id):
        """Get the queue name for a specific client."""
        return f'aiSSH_{client_id}'

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
                base_path = f'{self.remote_base_path}/vision_data/{self.bucket_name}/{self.experiment}'
                
                client_dirs = []
                try:
                    for item in self.sftp.listdir_attr(base_path):
                        if self._is_directory(item) and item.filename.startswith('user_'):
                            client_dirs.append(item.filename)
                except FileNotFoundError:
                    pass  # Base path doesn't exist yet
                
                if wait_count % 10 == 0:
                    print(f"[DEBUG BACKEND] Scanning for clients... (attempt {wait_count + 1})")
                    print(f"[DEBUG BACKEND] Found {len(client_dirs)} client directory(s)")
                    for cdir in client_dirs:
                        print(f"[DEBUG BACKEND]   - {cdir}")
                
                # Check each client directory for messages
                for client_dir in client_dirs:
                    client_id = client_dir[5:]  # Remove 'user_' prefix
                    client_path = f'{base_path}/{client_dir}'
                    
                    # Look for .npy files in this client's directory
                    try:
                        files = []
                        for f in self.sftp.listdir(client_path):
                            if f.endswith('.npy'):
                                try:
                                    stat = self.sftp.stat(f'{client_path}/{f}')
                                    files.append((f, stat.st_mtime))
                                except:
                                    continue
                        
                        # Sort by modification time
                        files.sort(key=lambda x: x[1])
                        
                        if files:
                            print(f"[DEBUG BACKEND] Found {len(files)} message(s) from client: {client_id}")
                            file_name, _ = files[0]  # Get oldest file
                            file_path = f'{client_path}/{file_name}'
                            
                            try:
                                print(f"[DEBUG BACKEND] Reading file: {file_path}")
                                
                                # Download file
                                local_temp = f'{self.save_dir}/_tmp_{file_name}'
                                self.sftp.get(file_path, local_temp)
                                
                                # Load data
                                loaded_data = np.load(local_temp, allow_pickle=True).item()
                                data = loaded_data['data']
                                
                                print(f"[DEBUG BACKEND] Successfully loaded data from client {client_id}")
                                
                                # Save current client ID for potential republish
                                self._current_client_id = client_id
                                
                                # Mark that we're now processing a request
                                # This prevents re-loading from disk on subsequent get() calls
                                self._processing_request = True
                                print(f"[DEBUG BACKEND] Set _processing_request flag to True")
                                
                                # Save a copy to local process directory
                                local_copy = f'{self.save_dir}/{self.name}-received.npy'
                                np.save(local_copy, data, allow_pickle=True)
                                print(f"[DEBUG BACKEND] Saved local copy to: {local_copy}")
                                
                                # Delete the remote file after reading
                                self.sftp.remove(file_path)
                                print(f"[DEBUG BACKEND] Deleted remote file: {file_path}")
                                
                                # Clean up local temp file
                                if os.path.exists(local_temp):
                                    os.remove(local_temp)
                                
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
                    
                    except Exception as ex:
                        print(f"[DEBUG BACKEND] Error checking client {client_id}: {ex}")
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
    Uses SSH/SFTP backend for storage.
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
            receive_path = self.receive_path()
            
            # Check if directory exists
            try:
                self.sftp.stat(receive_path)
            except FileNotFoundError:
                print(f"[DEBUG BACKEND] Receive directory does not exist")
                return None

            # Get list of files
            files = []
            for f in self.sftp.listdir(receive_path):
                try:
                    stat = self.sftp.stat(f'{receive_path}/{f}')
                    files.append((f, stat.st_mtime))
                except Exception as e:
                    self.msg_warning(f'Failed to stat file {f}: {str(e)}', 3, 2)
                    continue

            if not files:
                print(f"[DEBUG BACKEND] No files available in receive directory")
                return None

            # Sort by modification time and get oldest file
            files.sort(key=lambda x: x[1])
            oldest_file, _ = files[0]
            remote_file = f'{receive_path}/{oldest_file}'

            print(f"[DEBUG BACKEND] Found file: {oldest_file}")

            # Download file
            local_path = f'{self.save_dir}/{self.name}-received.npy'
            
            try:
                self.sftp.get(remote_file, local_path)
                
                if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                    return None

                # Load data
                loaded_data = np.load(local_path, allow_pickle=True).item()
                data = loaded_data['data']

                print(f"[DEBUG BACKEND] Successfully loaded data from {oldest_file}")

                # Delete the remote file after reading
                self.sftp.remove(remote_file)
                print(f"[DEBUG BACKEND] Deleted remote file: {remote_file}")
                
                return data

            except Exception as ex:
                print(f"[DEBUG BACKEND] Error reading file {remote_file}: {ex}")
                self.msg_error(f'Error reading file {remote_file}: {ex}', 1, 2)
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
        print(f"[DEBUG BACKEND] Send path: {self.send_path()}")
        
        # Just call the parent's publish method - the send_path() is already
        # set to the client-specific directory (ai_{client_id})
        super().publish(data, save=save)


class MultiClientQueue_user(CustomS3):
    """
    Multi-client queue for user clients that can send requests to the AI server.
    Uses SSH/SFTP backend for storage.
    
    NOTE: This creates client-specific queue names (user_{client_id} and ai_{client_id}).
    The backend MUST use MultiClientQueue_AI to handle these client-specific queues.
    If the backend uses simple Queue_user, there will be a mismatch!
    """

    def __init__(self, client_id=None, username=USERNAME, endpoint=ENDPOINT, password=PASSWORD, port=22,
                 experiment=EXPERIMENT, name='multiClientUser', save_dir='./', verbosity=VERBOSITY,
                 remote_base_path=REMOTE_PATH, **kwargs):
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
                        password=password, port=port, experiment=experiment, name=name, save_dir=save_dir,
                        log_verbosity=verbosity, remote_base_path=remote_base_path, client_id=client_id,
                        clear_on_start=False, **kwargs)

        print(f"[DEBUG BACKEND] Full send path: {self.send_path()}")
        print(f"[DEBUG BACKEND] Full receive path: {self.receive_path()}")
