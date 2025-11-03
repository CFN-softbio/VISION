from bluesky.plans import count
from bluesky.plans import bps


def pil_test():
    uid = RE(count([pilatus2M]))
    h = db[uid[0]]
    # print(uid)
    print(h.start)
    return h


def motor_test(pos=0):
    RE(bps.mov(smx, pos))
