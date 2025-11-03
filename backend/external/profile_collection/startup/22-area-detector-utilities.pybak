def xp_set(seconds):
    #    sleep_time=0.002
    # caput('XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime',seconds)
    caput("XF:11BMB-ES{}:cam1:AcquireTime".format(pilatus_Epicsname), seconds)
    #    sleep(sleep_time)
    caput("XF:11BMB-ES{}:cam1:AcquirePeriod".format(pilatus_Epicsname), seconds + 0.1)


def xp(seconds):
    sleep_time = 0.1
    caput("XF:11BMB-ES{}:cam1:Acquire".format(pilatus_Epicsname), 1)
    time.sleep(seconds + sleep_time)
