print(f'Loading {__file__}')

import ophyd

##### FOE #####
quad_electrometer1_1 = ophyd.EpicsSignalRO(
    "XF:11BMA-BI{IM:1}EM180:Current1:MeanValue_RBV", name="quad_electrometer1_1"
)
quad_electrometer1_2 = ophyd.EpicsSignalRO(
    "XF:11BMA-BI{IM:1}EM180:Current2:MeanValue_RBV", name="quad_electrometer1_2"
)
quad_electrometer1_3 = ophyd.EpicsSignalRO(
    "XF:11BMA-BI{IM:1}EM180:Current3:MeanValue_RBV", name="quad_electrometer1_3"
)
quad_electrometer1_4 = ophyd.EpicsSignalRO(
    "XF:11BMA-BI{IM:1}EM180:Current4:MeanValue_RBV", name="quad_electrometer1_4"
)

bim1 = ophyd.EpicsSignalRO("XF:11BMA-BI{IM:1}EM180:Current1:MeanValue_RBV", name="bim1")
bim2 = ophyd.EpicsSignalRO("XF:11BMA-BI{IM:1}EM180:Current2:MeanValue_RBV", name="bim2")


##### Endstation #####
## TODO: fix 'precision' and 'units' at EPICS level
ion_chamber1 = ophyd.EpicsSignalRO("XF:11BMB-BI{IM:3}:IC1_MON", name="ion_chamber1")
ion_chamber2 = ophyd.EpicsSignalRO("XF:11BMB-BI{IM:3}:IC2_MON", name="ion_chamber2")
ion_chamber3 = ophyd.EpicsSignalRO("XF:11BMB-BI{IM:3}:IC3_MON", name="ion_chamber3")
ion_chamber4 = ophyd.EpicsSignalRO("XF:11BMB-BI{IM:3}:IC4_MON", name="ion_chamber4")


class ScaleSignal(ophyd.signal.DerivedSignal):
    def __init__(self, *args, factor, **kwargs):
        self._factor = factor
        super().__init__(*args, **kwargs)

    def inverse(self, value):
        return self._factor * value

    def forward(self, value):
        return value / self._factor

    def describe(self):
        desc = super().describe()
        wd = desc[self.name]
        wd["derived_type"] = "ScaleSignal"
        wd["factor"] = self._factor
        return desc


scaled_ic1 = ScaleSignal(ion_chamber1, factor=1e9, name="scaled_ic1")
scaled_ic2 = ScaleSignal(ion_chamber2, factor=1e9, name="scaled_ic2")
scaled_ic3 = ScaleSignal(ion_chamber3, factor=1e9, name="scaled_ic3")
scaled_ic4 = ScaleSignal(ion_chamber4, factor=1e9, name="scaled_ic4")


quad_electrometer2_1 = ophyd.EpicsSignalRO(
    "XF:11BMB-BI{IM:2}EM180:Current1:MeanValue_RBV", name="quad_electrometer2_1"
)
quad_electrometer2_2 = ophyd.EpicsSignalRO(
    "XF:11BMB-BI{IM:2}EM180:Current2:MeanValue_RBV", name="quad_electrometer2_2"
)
quad_electrometer2_3 = ophyd.EpicsSignalRO(
    "XF:11BMB-BI{IM:2}EM180:Current3:MeanValue_RBV", name="quad_electrometer2_3"
)
quad_electrometer2_4 = ophyd.EpicsSignalRO(
    "XF:11BMB-BI{IM:2}EM180:Current4:MeanValue_RBV", name="quad_electrometer2_4"
)


# bim6 is the monitor after the sample (called dsmon on X9)
# The monitor sits on an arm on the DET system, so it can be moved with DETx and DETy
# bim6 = ophyd.EpicsSignalRO("XF:11BMB-BI{IM:2}EM180:Current1:MeanValue_RBV", name='bim6')
class EpicsSignalROWait(ophyd.EpicsSignalRO):
    """Customized version of EpicsSignal that has a 'wait_time' in the 'read()'
    function. This can be used for signals that need some time to settle before
    a value is read."""

    def __init__(self, *args, wait_time=None, **kwargs):
        if wait_time is not None:
            self._wait_time = wait_time
        else:
            self._wait_time = 0

        super().__init__(*args, **kwargs)

    def read(self, *args, **kwargs):
        # print('waiting {} s'.format(self._wait_time))
        sleep(self._wait_time)
        return super().read(*args, **kwargs)


bim6 = EpicsSignalROWait("XF:11BMB-BI{IM:2}EM180:Current1:MeanValue_RBV", wait_time=1, name="bim6")


class EpicsSignalROIntegrate(ophyd.EpicsSignalRO):
    """Customized version of EpicsSignal that has manually integrates (averages
    a few values). This can be used for signals that are otherwise too erratic."""

    def __init__(self, *args, wait_time=None, integrate_num=1, integrate_delay=0.01, **kwargs):
        if wait_time is not None:
            self._wait_time = wait_time
        else:
            self._wait_time = 0

        self._integrate_num = integrate_num
        self._integrate_delay = integrate_delay

        super().__init__(*args, **kwargs)

    def read(self, *args, **kwargs):
        # print('waiting {} s'.format(self._wait_time))
        sleep(self._wait_time)

        value = 0.0
        num = 0.0
        for i in range(self._integrate_num):
            value_current = super().read(*args, **kwargs)[self.name]["value"]
            # print(value_current)
            value += value_current
            num += 1.0
            sleep(self._integrate_delay)

        value /= num

        ret = super().read(*args, **kwargs)
        ret[self.name]["value"] = value

        return ret


bim6_integrating = EpicsSignalROIntegrate(
    "XF:11BMB-BI{IM:2}EM180:Current1:MeanValue_RBV",
    wait_time=0.5,
    integrate_num=8,
    integrate_delay=0.1,
    name="bim6",
)
