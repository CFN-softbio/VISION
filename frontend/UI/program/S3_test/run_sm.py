#!/usr/bin/python3

from CustomS3 import Queue_user
from Base import *
import time

from pydub import AudioSegment



def send(queue, data):
    # Send signal to AI
    queue.publish(data)

    # Wait for reply
    data = queue.get()

    return data


def speed_test(queue, data=None, n=10):
    '''Test the round-trip time for sending a signal to AI server,
    and getting a reply.'''

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




if __name__ == '__main__':

    # The user side of the loop sends signals and gets a reply
    q = Queue_user()


    speed_test(q)
    #Result:
    # Average loop time: 0.334 ± 0.080 s

    # Example of sending data to AI server (and getting a reply):

    # Path to the audio file

    # file_path = '/nsls2/data/cms/legacy/xf11bm/software/SMathur/data/example.wav'
    # # Load the audio file
    # audio = AudioSegment.from_file(file_path)
    # # Get the raw audio data as a bytestring
    # # raw_data = audio.raw_data
    # # # Get the frame rate
    # # sample_rate = audio.frame_rate

    # data = {
    #     'session_name': 'testing',
    #     'datetime': Base().now(),
    #     'audio':audio
    #     # 'audio': raw_data,
    #     # 'sample_rate': sample_rate

    #     # Can add whatever other meta-data, np arrays, etc.
    #     }
    # data = [data]

    # data = send(q, data)
    # print(data)
