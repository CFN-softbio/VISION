# XF:11BMB-CT{DIODE-Local:3}OutCh00:Data-SP

# PV list of Diode box: AO, Analog Output
# class Diode_AIpv(object):
# def __init__(self, ii):
# self.name = 'AI_Chan{}'.format(ii)
# self.sts = 'XF:11BMB-ES{}AI:{}-I'.format('{IO}', ii)
# Diode_AI=[None]
# for ii in range(1, 9):
# Diode_AI.append(AIpv(ii))


Diode_AO = EpicsSignal("XF:11BMB-CT{DIODE-Local:3}OutCh00:Data-SP")
Diode_A1 = EpicsSignal("XF:11BMB-CT{DIODE-Local:3}OutCh01:Data-SP")
Diode_A2 = EpicsSignal("XF:11BMB-CT{DIODE-Local:3}OutCh02:Data-SP")
Diode_A3 = EpicsSignal("XF:11BMB-CT{DIODE-Local:3}OutCh03:Data-SP")
# caput('XF:11BMB-CT{DIODE-Local:3}OutCh00:Data-SP.DESC', 'Diode AO 1')
# Diode_AO.get()
# Diode_AO.set()
