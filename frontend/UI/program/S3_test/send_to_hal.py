#!/usr/bin/python3

''' VISION calls this file to send the recorded audio to HAL.
The send_audio function is used by VISION (This was 
basically copied from the main function of run_sm.py)'''

from .CustomS3 import Queue_user
from .Base import *
import time

from pydub import AudioSegment

q = Queue_user()

def receive(queue = q):

    data = queue.get()


    return data

def send(data, queue = q):
    # Send signal to AI
    queue.publish(data)

    # Wait for reply
    data = queue.get()

    print(data)

    return data

def send_audio(audio):

    # q = Queue_user()
    data = {
        'session_name': 'testing',
        'datetime': Base().now(),
        'audio':audio
        }
    data = [data]

    data = send(data, q)
    # print(data)

    return data[0]['transcription']

def send_hal(data):
    q = Queue_user()

    data = [data]

    data = send(data, q)

    print(data)

    return data[0]



def send_audio_file(data, file_path):

    # q = Queue_user()
    audio = AudioSegment.from_file(file_path)

    data['voice_cog_input'] = audio

    data = send_hal(data)

    # data = [data]

    # data = send(q, data)

    return data



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
        'audio':audio

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
