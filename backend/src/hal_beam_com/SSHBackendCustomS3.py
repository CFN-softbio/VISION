#!/usr/bin/python3
import io
import os
import shutil
import time
from pathlib import Path
import paramiko
import numpy as np
from Base import Base


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
                 **kwargs):

        super().__init__(name=name, log_verbosity=log_verbosity, **kwargs)

        if clear_on_start:
            self.clear()  # Clear any leftover files from previous sessions

        self._last_processed_time = 0

        # Get password from env variables if not provided
        if password is None:
            password = os.environ.get('SSH_PASSWORD')
            if password is None:
                raise EnvironmentError("SSH_PASSWORD environment variable is not set.")

        self._password = password  # Store for potential reconnection

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

        current_time = time.time()
        # Only consider files newer than last request
        self._last_processed_time = max(self._last_processed_time, current_time - 30)  # 30 second window

        while True:
            try:
                self.msg(f'Checking directory: {receive_path}', 4, 2)

                try:
                    self.sftp.stat(receive_path)
                except FileNotFoundError:
                    self._ensure_remote_paths()
                    time.sleep(1)
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
                    self.msg(f'Found {len(files)} new files', 4, 2)
                    for fname, mtime in files:
                        self.msg(f'File: {fname}, Modified: {time.ctime(mtime)}', 4, 2)

                    # Get newest file
                    newest_file, newest_time = files[-1]
                    remote_file = f'{receive_path}/{newest_file}'

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
                    return loaded_data['data']

                else:
                    self.msg('No new files found, waiting...', 4, 2)

            except Exception as ex:
                self.msg_error(f'Error in get_s3_file: {str(ex)}', 1, 2)
                # Try to reconnect if needed
                self._try_reconnect()

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
                self.ssh.connect(self.ssh.get_transport().getpeername()[0],
                                 self.ssh.get_transport().getpeername()[1],
                                 self.ssh._transport.get_username(),
                                 self._password)
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
        if force_load or (check_interrupted and self.interrupted()):
            self.msg('Loading data/command from disk...', 4, 1)
            data = self.load()
        else:
            self.msg('Waiting for data/command...', 4, 1)
            data = self.get_s3_file()
        return data

    def publish(self, data, save=True):
        self.msg('Sending data/command...', 4, 1)
        self.publish_s3_file(data)
        self.msg('Sent.', 4, 2)

    def republish(self, stype='sent'):
        self.msg(f'Re-publishing the last data/command that was {stype}', 4, 1)
        data = self.load(stype=stype)
        self.publish(data)

    def interrupted(self):
        received = f'{self.save_dir}/{self.name}-received.npy'
        sent = f'{self.save_dir}/{self.name}-sent.npy'

        if not os.path.exists(received):
            return False
        if not os.path.exists(sent):
            return True

        received_time = os.path.getmtime(received)
        sent_time = os.path.getmtime(sent)

        return sent_time < received_time

    def load(self, stype='received'):
        return np.load(f'{self.save_dir}/{self.name}-{stype}.npy', allow_pickle=True)


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
        super().__init__(username=username, send=send, receive=receive, endpoint=endpoint, password=password,
                         experiment=experiment, name=name, save_dir=save_dir, verbosity=verbosity,
                         remote_base_path=remote_base_path, **kwargs)


class Queue_user(CustomS3):
    def __init__(self, username=USERNAME, send='user', receive='ai', endpoint=ENDPOINT, password=PASSWORD,
                 experiment=EXPERIMENT, name='userSSH', save_dir='./', verbosity=VERBOSITY,
                 remote_base_path=REMOTE_PATH, **kwargs):
        super().__init__(username=username, send=send, receive=receive, endpoint=endpoint, password=password,
                         experiment=experiment, name=name, save_dir=save_dir, verbosity=verbosity,
                         remote_base_path=remote_base_path, **kwargs)
