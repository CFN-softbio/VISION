#!/usr/bin/python3
import io
import os
import time
import json
import numpy as np
from pathlib import Path
from Base import Base
from src.hal_beam_com.model_manager import ModelManager
from utils import model_configurations, ACTIVE_CONFIG, get_finetuned_config


def get_system_temp_dir():
    """Get the appropriate system-wide temp directory"""
    if os.name == 'nt':  # Windows
        return os.path.join(os.environ.get('TEMP', 'C:\\Temp'), 'hal_communication')
    else:  # Linux/Mac
        return os.path.join('/tmp', 'hal_communication')


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
                 **kwargs):

        super().__init__(name=name, log_verbosity=log_verbosity, **kwargs)

        # Initialize models before starting the queue
        initialize_models()

        if experiment is None:
            experiment = f'experiment_{self.now(str_format="%Y-%m-%d_%H")}'

        self.experiment = experiment
        self.send = send
        self.receive = receive
        self.save_dir = save_dir
        self.bucket_name = bucket_name

        # Use system temp directory for inter-process communication
        self.comm_dir = Path(get_system_temp_dir())
        self.base_path = self.comm_dir / bucket_name / experiment
        self.send_path_dir = self.base_path / send
        self.receive_path_dir = self.base_path / receive

        # Create all necessary directories
        os.makedirs(self.save_dir, exist_ok=True)
        self.send_path_dir.mkdir(parents=True, exist_ok=True)
        self.receive_path_dir.mkdir(parents=True, exist_ok=True)

        # Set appropriate permissions for the communication directory
        if os.name != 'nt':  # Not Windows
            os.chmod(self.comm_dir, 0o777)
            os.chmod(self.base_path, 0o777)
            os.chmod(self.send_path_dir, 0o777)
            os.chmod(self.receive_path_dir, 0o777)

        self.msg(f"Using communication directory: {self.comm_dir}", 4, 1)

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
            os.chmod(comm_file_path, 0o666)

        self.msg(f'Sent local data: {comm_file_path}', 4, 2)

    def get(self, save=True, check_interrupted=True, force_load=False):
        self.msg(f'Getting data/command ({self.now()})', 4, 1)

        if force_load or (check_interrupted and self.interrupted()):
            self.msg('Loading data/command from disk...', 4, 1)
            data = self.load()
        else:
            self.msg(f'Waiting for data/command ({self.now()})...', 4, 1)
            data = self.get_s3_file()

        if isinstance(data, (list, tuple, np.ndarray)):
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

    def clear(self):
        self.msg('Clearing queue saved files.', 2, 1)

        for stype in ['received', 'sent']:
            file_path = Path(self.save_dir) / f'{self.name}-{stype}.npy'
            if file_path.exists():
                self.msg(f'Removing {file_path}', 3, 2)
                file_path.unlink()
            else:
                self.msg(f'{stype.capitalize()} data does not exist ({file_path})', 3, 2)


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
