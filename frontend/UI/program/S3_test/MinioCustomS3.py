#!/usr/bin/python3
import os
import time, datetime
from pathlib import Path
import os

import numpy as np

import io
# minio (S3 API) Python client reference:
# https://docs.min.io/docs/python-client-api-reference.html
from minio import Minio
from minio.error import S3Error

from .Base import Base

class CustomS3(Base):
    """
    CustomS3 is a helper class that simplifies communication with an S3-compatible storage server, enabling data transmission
    and retrieval in scientific workflows or distributed systems. This class manages secure file storage, temporary file handling,
    and structured storage of experimental data, facilitating the transfer of NumPy array data and other binary files to and
    from the S3 server.

    Attributes:
        username (str): The S3 access key for authentication.
        send (str): Queue name for publishing data; default is 'send'.
        receive (str): Queue name for receiving data; default is 'receive'.
        endpoint (str): Endpoint URL for the S3 or Lustre server, defaulting to 'localhost:8000'.
        secret_key_path (str): Path to the file storing the secret key for authentication. Only used if SECRET_S3_KEY is not defined in the env.
        secure (bool): Whether to use SSL for secure communication; default is True.
        bucket_name (str): S3 bucket for storing data; default is 'transmissions'.
        experiment (str): Name of the experiment for organizational storage; auto-generated if not specified.
        name (str): Name identifier for this instance; defaults to 'S3'.
        save_dir (str): Directory path for temporary storage; default is './'.
        log_verbosity (int): Controls logging verbosity; default is 5.

    Methods:
        send_path(): Returns the S3 path for sending data based on experiment and send queue.
        receive_path(): Returns the S3 path for receiving data based on experiment and receive queue.
        getS3_floats(): Retrieves and processes a single data event containing NumPy array floats.
        publishS3_floats(data): Publishes NumPy array data to the S3 server as floats.
        getS3_file(): Retrieves a file from S3 and loads its content into a NumPy array.
        publishS3_file(data): Saves a NumPy array as a local file and uploads it to the S3 server.
        publish_status_file(file_path, name=None): Uploads a status file with metadata to the S3 server.
        get_status_files(name='status', timestamp=False): Retrieves status files from S3 storage.
        get(save=True, check_interrupted=True, force_load=False): Fetches the most recent data item, either from local or S3.
        publish(data, save=True): Publishes a data item to S3.
        republish(stype='sent'): Re-sends the last published data item.
        interrupted(): Checks if there was an interruption in the last work cycle based on file timestamps.
        load(stype='received'): Loads data from a local saved file based on type ('received' or 'sent').
        clear(): Deletes locally stored files in the save directory to reset the queue.
    """

    def __init__(self,
                 username,  # S3 access_key
                 send='send',  # Queue to send/publish to
                 receive='receive',  # Queue to watch for signals
                 endpoint="localhost:8000",  # S3/lustre server
                 secret_key_path=None,  # Authentication key
                 secure=True,  # Require SSL?
                 bucket_name='transmissions',  # S3 bucket to use
                 experiment=None,  # Name in S3 storage for the datasets
                 name='S3',  # Name of this object (for print/log purposes)
                 save_dir='./',  # Location for tmp save files
                 log_verbosity=5,  # If a 'common' object is defined
                 **kwargs  # Additional keywork arguments
                 ):
        
        super().__init__(name=name, log_verbosity=log_verbosity, **kwargs) # class Base()

        # Get secret key from env variables (preferred)
        # This way there's no risk of pushing keys to GitHub, or LLMs
        secret_key = os.environ.get('SECRET_S3_KEY')

        # If it was not defined in the env variables, we will load it via the specified path
        if secret_key is None:
            with open(self.__get_secret_key_path(secret_key_path)) as fin:
                secret_key = fin.readline().strip()
            
            
        self.client = Minio(endpoint, access_key=username, secret_key=secret_key, secure=secure)
        
        self.bucket_name = bucket_name
        if experiment is None:
            experiment = 'experiment_{}'.format(self.now(str_format='%Y-%m-%d_%H'))
        self.experiment = experiment
        self.send = send
        self.receive = receive
        
        self.save_dir = save_dir

    def __get_secret_key_path(self, secret_key_path):
        if secret_key_path is None:
            # Search in 'typical' paths
            p = Path('./S3_secret_key.txt')  # Local to script execution
            if not p.exists():
                p = Path.home() / ".secret/S3_secret_key.txt"  # User's home dir
        else:
            p = Path(secret_key_path)

        if not p.exists():
            self.msg_error('Specified secret_key path is not valid.', 1, 0)

        return p

    def send_path(self):
        return '{}/{}'.format(self.experiment, self.send)
    def receive_path(self):
        return '{}/{}'.format(self.experiment, self.receive)

    def getS3_floats(self):
        
        import json
        
        events = self.client.listen_bucket_notification(
            bucket_name=self.bucket_name,
            prefix='{}/'.format(self.receive_path()),
            events=["s3:ObjectCreated:Put"],
        )
        
        
        for event in events:
            #self.print_d(event)
            assert len(event["Records"]) == 1

            record = event['Records'][0]
            #self.print_d(record)
            content_type = record['s3']['object']['contentType']
            assert content_type == "AE/custom"
            
            object_name = record['s3']['object']['key']
            timestamp = record["s3"]["object"]["userMetadata"]["X-Amz-Meta-Timestamp"]
            shape = record['s3']['object']['userMetadata']['X-Amz-Meta-Shape']
            shape = tuple(json.loads(f'[{shape}]'))
            self.msg('Received S3 data: {}'.format(object_name), 4, 2)
            
            sucessful = False
            try:
                data_stream = self.client.get_object(self.bucket_name, object_name)
                data_bytes = data_stream.data
                data_np = np.frombuffer(data_bytes).reshape(shape)
                sucessful = True
                
            except Exception as ex:
                self.msg_error('Python Exception in getS3', 1, 2)
                self.print(ex)
                
            finally:
                data_stream.close()
                data_stream.release_conn()

            if not sucessful:
                self.msg_warning('S3 data retrieval failed.', 2, 2)
            
            break # Process just that one event
        
        return data_np
            
        
    
    def publishS3_floats(self, data):
        
        now_str = self.now(str_format='%Y%m%d_%H%M%S_%f')
        object_name = '{}/obj{}'.format(self.send_path(), now_str)
        
        data_bytes = data.tobytes()
        data_stream = io.BytesIO(data_bytes)
        
        
        result = self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=data_stream,
            length=len(data_bytes),
            content_type="AE/custom",
            metadata={
                'timestamp': self.now(),
                'shape': data.shape,
                }
            )

        self.msg('Sent S3 data: {}'.format(object_name), 4, 2)


    def getS3_file(self):
        
        events = self.client.listen_bucket_notification(
            bucket_name=self.bucket_name,
            prefix='{}/'.format(self.receive_path()),
            events=["s3:ObjectCreated:Put"],
        )
        
        
        file_path = '{}/{}-received.npy'.format(self.save_dir, self.name)
        
        
        for event in events:
            #self.print_d(event)
            assert len(event["Records"]) == 1

            record = event['Records'][0]
            #self.print_d(record)
            content_type = record['s3']['object']['contentType']
            assert content_type == "AE/custom"
            
            object_name = record['s3']['object']['key']
            timestamp = record["s3"]["object"]["userMetadata"]["X-Amz-Meta-Timestamp"]
            self.msg('Received S3 data: {}'.format(object_name), 4, 2)
            
            sucessful = False
            try:
                self.client.fget_object(
                    self.bucket_name, 
                    object_name,
                    file_path
                    )
                data = np.load(file_path, allow_pickle=True)
                sucessful = True
                
            except Exception as ex:
                self.msg_error('Python Exception in getS3', 1, 2)
                self.print(ex)
                

            if not sucessful:
                self.msg_warning('S3 data retrieval failed.', 2, 2)
            else:
                np.save(file_path, data, allow_pickle=True)
            
            break # Process just that one event
        
        
        return data
            
        
    
    def publishS3_file(self, data):
        
        file_path = '{}/{}-sent.npy'.format(self.save_dir, self.name)
        np.save(file_path, data, allow_pickle=True)
        
        now_str = self.now(str_format='%Y%m%d_%H%M%S_%f')
        object_name = '{}/obj{}'.format(self.send_path(), now_str)
        
        
        result = self.client.fput_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            file_path=file_path,
            content_type="AE/custom",
            metadata={
                'timestamp': self.now(),
                }
            )

        self.msg('Sent S3 data: {}'.format(object_name), 4, 2)


    def publish_status_file(self, file_path, name=None):
        self.msg('Uploading status file ({})...'.format(self.now()), 4, 1)
        
        p = Path(file_path)
        name = p.name if name is None else '{}{}'.format(name, p.suffix)
        object_name = '{}/status/{}/{}'.format(self.experiment, self.send, name)
        
        
        result = self.client.fput_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            file_path=file_path,
            content_type="AE/status",
            metadata={
                'timestamp': self.now(),
                'mtime': p.stat().st_mtime,
                'ctime': p.stat().st_ctime,
                'filesize': p.stat().st_size,
                }
            )

        self.msg('Sent S3 file: {}'.format(object_name), 4, 2)


    def get_status_files(self, name='status', timestamp=False):
        
        prefix = '{}/{}/'.format(self.experiment, name)
        now_str = self.now(str_format='%Y-%m-%d_%H%M%S')
        
        self.msg('Getting status files ({})'.format(self.now()), 4, 1)
        self.msg('recursive searching: {}'.format(prefix), 4, 2)
        
        objects = self.client.list_objects(
            bucket_name=self.bucket_name,
            prefix=prefix,
            recursive=True,
            )
        
        for obj in objects:
            self.msg('downloading: {}'.format(obj.object_name), 4, 3)
            
            if timestamp:
                obj_str = obj.object_name[len(self.experiment):][len('/status/'):]
                file_path = '{}/status/{}/{}'.format( self.save_dir, now_str, obj_str )
            else:
                file_path = '{}{}'.format( self.save_dir, obj.object_name[len(self.experiment):] )
            
            try:
                self.client.fget_object(
                    self.bucket_name, 
                    obj.object_name,
                    file_path
                    )

            except Exception as ex:
                self.msg_error('Python Exception in get_status_files', 1, 2)
                self.print(ex)
            
        self.msg('Done.', 4, 2)
    

    def get(self, save=True, check_interrupted=True, force_load=False):
        '''Get the current item being published.'''
        #message = self.from_socket.recv()
        
        self.msg('Getting data/command ({})'.format(self.now()), 4, 1)
        
        if force_load or (check_interrupted and self.interrupted()):
            self.msg('Loading data/command from disk...'.format(self.now()), 4, 1)
            data = self.load()
            
            if isinstance(data, (list, tuple, np.ndarray)):
                self.msg('Loaded: list length {}'.format(len(data)), 4, 2)
            else:
                self.msg('Loaded.', 4, 2)
        
        else:
            self.msg('Waiting for data/command ({})...'.format(self.now()), 4, 1)
            #data = self.getS3_floats()
            data = self.getS3_file()
            
            if isinstance(data, (list, tuple, np.ndarray)):
                self.msg('Received: list length {}'.format(len(data)), 4, 2)
            else:
                self.msg('Received.', 4, 2)
            
            #if save:
                #np.save('{}/{}-received.npy'.format(self.save_dir, self.name), data, allow_pickle=True)

        return data


    
    def publish(self, data, save=True):
        
        self.msg('Sending data/command ({})...'.format(self.now()), 4, 1)
        #self.publishS3_floats(data)
        self.publishS3_file(data)
        self.msg('Sent.', 4, 2)

        #if save:
            #np.save('{}/{}-sent.npy'.format(self.save_dir, self.name), data, allow_pickle=True)

    def republish(self, stype='sent'):
        '''Re-send the last data/command.
        You can use this to force the loop to restart, if you interrupted it.'''
        
        self.msg('Re-publishing the last data/command that was {} ({})'.format(stype, self.now()), 4, 1)
        self.interrupted()
        
        data = self.load(stype=stype)
        self.publish(data)
        
        
    def interrupted(self):
        
        received = '{}/{}-received.npy'.format(self.save_dir, self.name)
        
        if not os.path.exists(received):
            self.msg('No saved received data from prior work-cycle.', 4, 2)
            return False
        
        received = os.path.getmtime(received)
        sent = '{}/{}-sent.npy'.format(self.save_dir, self.name)
        if os.path.isfile(sent):
            sent = os.path.getmtime(sent)
        else:
            self.msg('No saved sent data from prior work-cycle.', 4, 2)
            return True
        
        if sent>received:
            # As expected, the file we sent is newer
            self.msg('Last work-cycle completed normally (based on file dates).', 4, 2)
            self.msg('Received: {}'.format(self.time_str(received)), 5, 3)
            self.msg('Sent: {} ({})'.format(self.time_str(sent), self.time_delta(received, sent)), 5, 3)
            return False
        else:
            self.msg('Last work-cycle INTERRUPTED (based on file dates).', 3, 2)
            self.msg('Received: {}'.format(self.time_str(received)), 3, 3)
            self.msg('Sent: {} ({})'.format(self.time_str(sent), self.time_delta(received, sent)), 3, 3)
            return True


    def load(self, stype='received'):
        data = np.load('{}/{}-{}.npy'.format(self.save_dir, self.name, stype), allow_pickle=True)
        return data


    def clear(self):
        '''Remove the saved filed.'''
        self.msg('Clearing queue saved files.', 2, 1)
        
        received = '{}/{}-received.npy'.format(self.save_dir, self.name)
        if os.path.exists(received):
            self.msg('Removing {}'.format(received), 3, 2)
            os.remove(received)
        else:
            self.msg('Received data does not exist ({})'.format(received), 3, 2)

        sent = '{}/{}-sent.npy'.format(self.save_dir, self.name)
        if os.path.exists(sent):
            self.msg('Removing {}'.format(sent), 3, 2)
            os.remove(sent)
        else:
            self.msg('Sent data does not exist ({})'.format(sent), 3, 2)


# Two-party connection:
########################################

# TODO: Remove sensitive urls and keys before pushing to GitHub
VERBOSITY=4
ENDPOINT="REDACTED"
USERNAME='REDACTED'
# USERNAME='VISION'
SECRET_KEY_PATH=None # Use default path
# SECRET_KEY_PATH='/nsls2/xf11bm/data/2021_3/UShell/S3_secret_key.txt' # Specify path
# SECRET_KEY_PATH='/nsls2/data/cms/legacy/xf11bm/data/2024_2/beamline/PTA/test/S3_test/S3_secret_key.txt'
# SECRET_KEY_PATH='/nsls2/data/cms/legacy/xf11bm/software/ETsai/VISION_0/Model3.0/program/S3_test/S3_secret_key_old.txt'
# SECRET_KEY_PATH='/Users/shraymathur/MyFolders/BNL/work/S3_secret_key.txt'



# SECRET_KEY = None
EXPERIMENT='REDACTED'
#EXPERIMENT=None # Default is to use day and hour


class Queue_AI(CustomS3): # AI on server
    def __init__(self, username=USERNAME, send='ai', receive='user', endpoint=ENDPOINT, secret_key_path=SECRET_KEY_PATH, experiment=EXPERIMENT, name='aiS3', save_dir='./', verbosity=VERBOSITY, **kwargs):
        super().__init__(username=username, send=send, receive=receive, endpoint=endpoint, secret_key_path=secret_key_path, experiment=experiment, name=name, save_dir=save_dir, verbosity=verbosity, **kwargs)

class Queue_user(CustomS3): # user at beamline
    def __init__(self, username=USERNAME, send='user', receive='ai', endpoint=ENDPOINT, secret_key_path=SECRET_KEY_PATH, experiment=EXPERIMENT, name='userS3', save_dir='./', verbosity=VERBOSITY, **kwargs):
        super().__init__(username=username, send=send, receive=receive, endpoint=endpoint, secret_key_path=secret_key_path, experiment=experiment, name=name, save_dir=save_dir, verbosity=verbosity, **kwargs)






########################################
# The usage would be:
########################################

# AI server:
########################################
#from CustomS3 import Queue_AI
#from Base import *
#def process(queue):
    #while True:
        ## Wait for a signal from user
        #data = queue.get()
        ## Do arbitrary amounts of work/processing
        #data[0]['AI'] = 'Did some work at timestamp: {}'.format(Base().now())
        ## Send updated results back to user
        #queue.publish(data)
#q = Queue_AI()
#process(q)



# Beamline:
########################################
#from CustomS3 import Queue_user
#from Base import *
#q = Queue_user()
#data = send(q, data)


