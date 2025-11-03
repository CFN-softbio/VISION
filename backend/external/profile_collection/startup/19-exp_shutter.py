print(f'Loading {__file__}')

##### Experimental shutters #####
# updated by RL, 20210901
# These shutters are controlled by sending a TTL pulse via Ecat controller.

trigger_new_pv = EpicsSignal("XF:11BM-ES{Shutter}")
shutter_sts1_pv = EpicsSignal("XF:11BM-ES{Psh_blade1}Pos")
shutter_sts2_pv = EpicsSignal("XF:11BM-ES{Psh_blade2}Pos")

photonshutter_sts = EpicsSignal("XF:11BMA-PPS{PSh}Sts:FailOpn-Sts")
photonshutter_open = EpicsSignal("XF:11BMA-PPS{PSh}Cmd:Opn-Cmd")
photonshutter_cls = EpicsSignal("XF:11BMA-PPS{PSh}Cmd:Cls-Cmd")


def shutter_on(verbosity=3):
    # ii = 0
    # yield from bps.mv(trigger_new_pv, 1)
    # time.sleep(0.1)
    while shutter_state(verbosity=0) != 1:
        yield from bps.mv(trigger_new_pv, 1)
        time.sleep(0.01)
        # ii += 1

    # print(ii)
    if verbosity >= 3:
        shutter_state(verbosity=verbosity)


def shutter_off(verbosity=3):
    while shutter_state(verbosity=0) != 0:
        yield from bps.mv(trigger_new_pv, 0)
        time.sleep(0.01)

    if verbosity >= 3:
        shutter_state(verbosity=verbosity)


def shutter_state(verbosity=3):
    if shutter_sts1_pv.get() == 1 and shutter_sts2_pv.get() == 1:
        status = 1
        if verbosity >= 3:
            print("Shutter is OPEN.")
    else:
        status = 0
        if verbosity >= 3:
            print("Shutter is CLOSED.")

    return status


# old control, abandoned in 2021C3
# These shutters are controlled by sending a 5V pulse via QEM output on the Delta Tau controller MC06
# (the same unit that controls slits S5). Both the opening and closing of the shutter are triggered
# by the rise of the pulse.
#
# Note:
# - PV for the QEM output on MC06 is:
# 	XF:11BMB-CT{MC:06}Asyn.AOUT
# - This PV is located under Slit 5/Asyn --> asynRecord/More... --> asynOctet interface I/O --> ASCII
# - 'M112=1' sets the state to high
# - 'M112=0' sets the state to low
# - 'M111=1' launches the change in state
# - A sleep time of ~2 ms between successive caput commands is needed to get proper response; 1 ms is too short
#####

# global xshutter_state
xshutter_state = 0  ## TODO: read the shutter state and set this accordingly


## Open shutter
def xshutter_trigger():
    sleep_time = 0.005
    caput("XF:11BMB-CT{MC:06}Asyn.AOUT", "M112=1")
    sleep(sleep_time)
    caput("XF:11BMB-CT{MC:06}Asyn.AOUT", "M111=1")
    sleep(sleep_time)
    caput("XF:11BMB-CT{MC:06}Asyn.AOUT", "M112=0")
    sleep(sleep_time)
    caput("XF:11BMB-CT{MC:06}Asyn.AOUT", "M111=1")


trigger_pv = EpicsSignal("XF:11BMB-CT{MC:06}Asyn.AOUT")
# shutter_sts1_pv = EpicsSignal('XF:11BMB-OP{PSh:2}Pos:1-Sts')
# shutter_sts2_pv = EpicsSignal('XF:11BMB-OP{PSh:2}Pos:2-Sts')


def xshutter_trigger_RE(verbosity=3):
    yield from bps.mv(trigger_pv, "M112=1")
    yield from bps.mv(trigger_pv, "M111=1")
    yield from bps.mv(trigger_pv, "M112=0")
    yield from bps.mv(trigger_pv, "M111=1")
    if verbosity >= 3:
        value = yield from bps.read(shutter_sts_pv.read())
        print(value[shutter_sts_pv.name]["value"])


def xshutter(inout, q=0):
    global xshutter_state

    if inout == "o" or inout == "open" or inout == 1:
        if xshutter_state == 0:
            xshutter_trigger()
            xshutter_state = 1
            if q == 0:
                print("Experimental shutter opened")
                return xshutter_state
        elif xshutter_state == 1:
            print("Experimental shutter is already open; no changes made")
        else:
            print("xshutter_state is neither 0 nor 1; no changes made")

    if inout == "c" or inout == "close" or inout == 0:
        if xshutter_state == 1:
            xshutter_trigger()
            xshutter_state = 0
            if q == 0:
                print("Experimental shutter closed")
                return xshutter_state
        elif xshutter_state == 0:
            print("Experimental shutter is already closed; no changes made")
        else:
            print("xshutter_state is neither 0 nor 1; no changes made")
