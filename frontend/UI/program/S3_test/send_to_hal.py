#!/usr/bin/python3

''' VISION calls this file to send the recorded audio to HAL.
The send_audio function is used by VISION (This was 
basically copied from the main function of run_sm.py)'''

from .CustomS3 import MultiClientQueue_user
from .Base import *
import time
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

from pydub import AudioSegment

from .file_utils import FileUtil

file_util = FileUtil()

# Create a multi-client queue with auto-generated client_id
q = MultiClientQueue_user(save_dir=file_util.get_temp_dir())
q.clear()                     # drop leftovers from previous runs

# Create a thread pool executor for running blocking operations
_executor = ThreadPoolExecutor(max_workers=4)


def cleanup_executor():
    """
    Clean up the thread pool executor and worker. Call this when shutting down the application.
    """
    global _executor
    if _executor:
        _executor.shutdown(wait=True)
        _executor = None


def receive(queue=q, check_interrupted=False):
    return queue.get(save=True,
                     check_interrupted=check_interrupted,
                     force_load=False,
                     return_all=True)      # new parameter: return all pending messages


def send(data, queue=q):
    print(f"[SEND] send function called with data: {data}")
    print(f"[SEND] Using client_id: {queue.client_id}")

    # Discard all existing replies, so only fresh reply will be delivered below
    if hasattr(queue, "flush"):
        queue.flush()
    queue.publish(data)  # actually sends data
    print(f"[SEND] Data published to queue")

    # Get the reply
    print("[SEND] Waiting for reply...")
    data = queue.get(check_interrupted=False) # wait for fresh reply
    print(f"[SEND] Received reply: {data}")

    # Handle basic numpy array unwrapping if needed
    import numpy as np
    if isinstance(data, np.ndarray):
        if data.shape == () and isinstance(data.item(), dict):
            # Unwrap singleton ndarray containing dict
            data = data.item()
        elif data.dtype == object and data.shape and all(isinstance(x, dict) for x in data):
            # Convert ndarray of dicts to list
            data = list(data)
    
    # Flatten list for normal request/reply pattern
    if isinstance(data, list):
        if len(data) == 1:
            data = data[0]
        elif len(data) > 1:
            # Keep the last message if multiple
            data = data[-1]
    
    print(f"[SEND] Returning processed data: {data}")
    return data


def send_audio(audio):
    data = {
        'session_name': 'testing',
        'datetime': Base().now(),
        'audio': audio
    }
    data = [data]
    
    data = send(data, q)
    
    return data['transcription']


async def send_audio_async(audio):
    data = {
        'session_name': 'testing',
        'datetime': Base().now(),
        'audio': audio
    }
    data = [data]

    result = await send_hal_async(data)

    return result['transcription']


def send_hal(data):
    # re-use the global queue `q` defined at import
    data = [data]                       # always send a list
    return send(data, q)                # send and return the simplified reply


async def send_hal_async(data):
    """
    Asynchronous version of send_hal that runs the blocking send operation
    in a thread pool to prevent UI blocking.

    Args:
        data: The data to send to HAL

    Returns:
        The response from HAL
    """
    print(f"[SEND_HAL] send_hal_async called with data: {data}")
    print(f"[SEND_HAL] Using client_id: {q.client_id}")

    # re-use the global queue `q` defined at import
    data = [data]                       # always send a list
    print(f"[SEND_HAL] Sending data list: {data}")

    # Run the blocking send operation in a thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, send, data, q)

    print(f"[SEND_HAL] Received result: {result}")
    return result


def send_audio_file(data, file_path):
    audio = AudioSegment.from_file(file_path)
    data['voice_cog_input'] = audio
    return send_hal(data)


async def send_audio_file_async(data, file_path):
    # Load audio file in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    audio = await loop.run_in_executor(_executor, AudioSegment.from_file, file_path)

    data['voice_cog_input'] = audio
    return await send_hal_async(data)


def speed_test(queue, data=None, n=10):
    '''Test the round-trip time for sending a signal to AI server,
    and getting a reply.'''

    # TODO: Do we want hardcoded paths?
    file_path = '/nsls2/data/cms/legacy/xf11bm/software/SMathur/data/example.wav'
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    data = {
        'session_name': 'testing',
        'datetime': Base().now(),
        'audio': audio

        }
    data = [data]

    spans = []
    for i in range(n):
        start = time.time()
        data = send(q, data)
        print(data[0]['transcription'])
        end = time.time()
        span = end-start
        print('    Test {:d}, round-trip S3 signalling took: {:.3f} seconds'.format(i+1, span))
        spans.append(span)

    avg, std = np.average(spans), np.std(spans)
    print('Average loop time: {:.3f} ± {:.3f} s'.format(avg, std))
