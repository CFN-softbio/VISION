#!/usr/bin/python3

from CustomS3 import Queue_user
from Base import *
import time


def send(queue, data):
    # Send signal to AI
    queue.publish(data)

    # Wait for reply
    data = queue.get()

    return data


def speed_test(queue, data=None, n=10):
    '''Test the round-trip time for sending a signal to AI server,
    and getting a reply.'''

    if data is None:
        data = {
            'session_name': 'testing',
            'datetime': Base().now(),
        }
    data = [data]

    spans = []
    for i in range(n):
        start = time.time()
        data = send(q, data)
        end = time.time()
        span = end-start
        print('    Test {:d}, round-trip S3 signalling took: {:.3f} seconds'.format(i+1, span))
        spans.append(span)

    avg, std = np.average(spans), np.std(spans)
    print('Average loop time: {:.3f} ± {:.3f} s'.format(avg, std))




if __name__ == '__main__':

    # The user side of the loop sends signals and gets a reply
    q = Queue_user()


    #speed_test(q)
    #Result:
    # Average loop time: 0.334 ± 0.080 s

    # Example of sending data to AI server (and getting a reply):
    data = {
        'session_name': 'testing',
        'datetime': Base().now(),

        # Can add whatever other meta-data, np arrays, etc.
        }
    data = [data]

    data = send(q, data)
    print(data)
