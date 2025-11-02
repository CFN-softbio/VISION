#!/usr/bin/python3
import io
import os
import time
import json
import shutil
import numpy as np
from pathlib import Path
from .Base import Base


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
        return Path(env_dir)

    # Use defaults based on OS
    home = Path.home()

    if os.name == 'nt':  # Windows
        # Use AppData\Local for Windows
        base_dir = Path(os.environ.get('LOCALAPPDATA', home / 'AppData' / 'Local'))
        return base_dir / 'hal_vision' / 'shared'
    else:  # Linux/Mac
        # Use hidden directory in home for Unix-like systems
        return home / '.hal_vision' / 'shared'


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
            experiment = f'experiment_{self.now(str_format="%Y-%m-%d_%H")}'

        self.experiment = experiment
        self.send = send
        self.receive = receive
        self.save_dir = save_dir
        self.bucket_name = bucket_name
        self.client_id = client_id

        # Use shared communication directory for inter-process/container communication
        self.comm_dir = get_shared_comm_dir()
        self.base_path = self.comm_dir / bucket_name / experiment
        self.send_path_dir = self.base_path / send
        self.receive_path_dir = self.base_path / receive

        # Create all necessary directories
        os.makedirs(self.save_dir, exist_ok=True)
        self.send_path_dir.mkdir(parents=True, exist_ok=True)
        self.receive_path_dir.mkdir(parents=True, exist_ok=True)

        # Set appropriate permissions for the communication directory (Unix only)
        if os.name != 'nt':  # Not Windows
            try:
                os.chmod(self.comm_dir, 0o777)
                os.chmod(self.base_path, 0o777)
                os.chmod(self.send_path_dir, 0o777)
                os.chmod(self.receive_path_dir, 0o777)
            except (OSError, PermissionError) as e:
                self.msg(f"Warning: Could not set permissions: {e}", 3, 1)

        # Track processed files to avoid re-processing
        self._seen_objects = set()

        self.msg(f"Using shared communication directory: {self.comm_dir}", 4, 1)
        self.msg(f"Send path: {self.send_path_dir}", 5, 2)
        self.msg(f"Receive path: {self.receive_path_dir}", 5, 2)

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
        while True:
            files = sorted(self.receive_path_dir.glob('*.npy'))
            if files:
                try:
                    data = np.load(files[0], allow_pickle=True)
                    # Save a copy to local process directory
                    local_copy = Path(self.save_dir) / f'{self.name}-received.npy'
                    np.save(local_copy, data, allow_pickle=True)
                    files[0].unlink()  # Remove the file after reading
                    return data
                except Exception as ex:
                    self.msg_error(f'Error reading file: {ex}', 1, 2)
                    time.sleep(0.1)
            time.sleep(0.1)

    def publish_s3_file(self, data):
        """Save data to both the communication directory and local process directory"""
        timestamp = self.now(str_format='%Y%m%d_%H%M%S_%f')

        # Save to communication directory
        comm_file_path = self.send_path_dir / f'obj_{timestamp}.npy'
        np.save(comm_file_path, data, allow_pickle=True)

        # Save to local process directory
        local_file_path = Path(self.save_dir) / f'{self.name}-sent.npy'
        np.save(local_file_path, data, allow_pickle=True)

        if os.name != 'nt':
            try:
                os.chmod(comm_file_path, 0o666)
            except (OSError, PermissionError) as e:
                self.msg(f"Warning: Could not set file permissions: {e}", 4, 2)

        self.msg(f'Sent local data: {comm_file_path}', 4, 2)

    def publish_status_file(self, file_path, name=None):
        """Upload a status file to the local communication directory"""
        self.msg(f'Uploading status file ({self.now()})...', 4, 1)

        p = Path(file_path)
        name = p.name if name is None else f'{name}{p.suffix}'

        # Create status directory in communication path
        status_dir = self.base_path / 'status' / self.send
        status_dir.mkdir(parents=True, exist_ok=True)

        dest_path = status_dir / name

        # Copy the file to the status directory
        shutil.copy2(file_path, dest_path)

        if os.name != 'nt':
            try:
                os.chmod(dest_path, 0o666)
            except (OSError, PermissionError) as e:
                self.msg(f"Warning: Could not set file permissions: {e}", 4, 2)

        self.msg(f'Sent local status file: {dest_path}', 4, 2)

    def get_status_files(self, name='status', timestamp=False):
        """Retrieve status files from the local communication directory"""
        status_prefix = self.base_path / name
        now_str = self.now(str_format='%Y-%m-%d_%H%M%S')

        self.msg(f'Getting status files ({self.now()})', 4, 1)
        self.msg(f'recursive searching: {status_prefix}', 4, 2)

        if not status_prefix.exists():
            self.msg(f'Status directory does not exist: {status_prefix}', 3, 2)
            return

        # Walk through all files in the status directory
        for file_path in status_prefix.rglob('*'):
            if file_path.is_file():
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
                except Exception as ex:
                    self.msg_error('Python Exception in get_status_files', 1, 2)
                    self.print(ex)

        self.msg('Done.', 4, 2)

    def get(self, save=True, check_interrupted=True, force_load=False, return_all=False):
        """Get the current item being published, optionally returning all available messages"""
        self.msg(f'Getting data/command ({self.now()})', 4, 1)

        if force_load or (check_interrupted and self.interrupted()):
            self.msg('Loading data/command from disk...', 4, 1)
            data = self.load()

            if isinstance(data, (list, tuple, np.ndarray)):
                self.msg(f'Loaded: list length {len(data)}', 4, 2)
            else:
                self.msg('Loaded.', 4, 2)
        else:
            messages = []
            while not messages:  # wait until at least one new file is available
                # List all files and sort chronologically
                all_files = sorted(self.receive_path_dir.glob('*.npy'), key=lambda f: f.stat().st_mtime)

                for file_path in all_files:
                    file_str = str(file_path)
                    if file_str in self._seen_objects:
                        continue  # already consumed

                    tmp_file = Path(self.save_dir) / f'_tmp_{file_path.name}'
                    try:
                        shutil.copy2(file_path, tmp_file)
                        msg = np.load(tmp_file, allow_pickle=True)
                        messages.append(msg)
                        self._seen_objects.add(file_str)

                        if tmp_file.exists():
                            tmp_file.unlink()
                            self.msg(f"{tmp_file} deleted.", 4, 3)
                    except Exception as ex:
                        self.msg_error(f'Error processing file {file_path.name}: {str(ex)}', 1, 2)

                if not messages:
                    time.sleep(0.05)  # short pause before retrying

            # If caller wants everything (streaming) return the full list.
            # Otherwise deliver only the newest message to avoid flooding
            # single-shot request/reply calls with a backlog.
            if return_all:
                data = messages
            else:
                data = messages[-1]  # newest (list is chronological)

            if isinstance(data, (list, tuple, np.ndarray)):
                if return_all:
                    self.msg(f'Received: {len(data)} messages', 4, 2)
                else:
                    self.msg(f'Received: list length {len(data)}', 4, 2)
            else:
                self.msg('Received.', 4, 2)

        return data

    def publish(self, data, save=True):
        self.msg(f'Sending data/command ({self.now()})...', 4, 1)
        self.publish_s3_file(data)
        self.msg('Sent.', 4, 2)

    def republish(self, stype='sent'):
        self.msg(f'Re-publishing the last data/command that was {stype} ({self.now()})', 4, 1)
        self.interrupted()
        data = self.load(stype=stype)
        self.publish(data)

    def interrupted(self):
        received = Path(self.save_dir) / f'{self.name}-received.npy'
        sent = Path(self.save_dir) / f'{self.name}-sent.npy'

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
        return np.load(file_path, allow_pickle=True)

    def flush(self):
        """
        Mark all current files in the receive queue as already seen.
        This ensures that only new files (created after this call)
        are delivered on next get().
        """
        try:
            for file_path in self.receive_path_dir.glob('*.npy'):
                self._seen_objects.add(str(file_path))
        except Exception as ex:
            self.msg_warning(f'flush() failed: {ex}', 2, 1)

    def clear(self):
        """Remove saved files for this queue and reset seen-object tracking"""
        self.msg('Clearing queue saved files.', 2, 1)

        for stype in ['received', 'sent']:
            file_path = Path(self.save_dir) / f'{self.name}-{stype}.npy'
            if file_path.exists():
                self.msg(f'Removing {file_path}', 3, 2)
                file_path.unlink()
            else:
                self.msg(f'{stype.capitalize()} data does not exist ({file_path})', 3, 2)

        # Reset the seen-object tracking
        self._seen_objects.clear()


class Queue_AI(CustomS3):
    def __init__(self, username="local", send='ai', receive='user', endpoint="local",
                 secret_key_path=None, experiment=None, name='aiS3',
                 save_dir='./', verbosity=5, **kwargs):
        super().__init__(username=username, send=send, receive=receive,
                         endpoint=endpoint, secret_key_path=secret_key_path,
                         experiment=experiment, name=name, save_dir=save_dir,
                         log_verbosity=verbosity, **kwargs)


class Queue_user(CustomS3):
    def __init__(self, username="local", send='user', receive='ai', endpoint="local",
                 secret_key_path=None, experiment=None, name='userS3',
                 save_dir='./', verbosity=5, **kwargs):
        super().__init__(username=username, send=send, receive=receive,
                         endpoint=endpoint, secret_key_path=secret_key_path,
                         experiment=experiment, name=name, save_dir=save_dir,
                         log_verbosity=verbosity, **kwargs)
