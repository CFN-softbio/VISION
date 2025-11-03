#!/usr/bin/python
# -*- coding: utf-8 -*-
# vi: ts=4 sw=4


################################################################################
#  Short-term settings (specific to a particular user/experiment) can
# be placed in this file. You may instead wish to make a copy of this file in
# the user's data directory, and use that as a working copy.
################################################################################


# logbooks_default = ['User Experiments']
# tags_default = ['CFN Soft-Bio']

import pickle

from ophyd import EpicsSignal
from bluesky.suspenders import SuspendFloor

beam_current = EpicsSignal("SR:OPS-BI{DCCT:1}I:Real-I")
sus = SuspendFloor(beam_current, 100, resume_thresh=101)
RE.install_suspender(sus)
# RE.clear_suspenders()

INTENSITY_EXPECTED_050 = 12065.0
INTENSITY_EXPECTED_025 = INTENSITY_EXPECTED_050 * 0.5


if False:
    # The following shortcuts can be used for unit conversions. For instance,
    # for a motor operating in 'mm' units, one could instead do:
    #     sam.xr( 10*um )
    # To move it by 10 micrometers. HOWEVER, one must be careful if using
    # these conversion parameters, since they make implicit assumptions.
    # For instance, they assume linear axes are all using 'mm' units. Conversely,
    # you will not receive an error if you try to use 'um' for a rotation axis!
    m = 1e3
    cm = 10.0
    mm = 1.0
    um = 1e-3
    nm = 1e-6

    inch = 25.4
    pixel = 0.172  # Pilatus

    deg = 1.0
    rad = np.degrees(1.0)
    mrad = np.degrees(1e-3)
    urad = np.degrees(1e-6)


def get_default_stage():
    return stg


class SampleTSAXS(SampleTSAXS_Generic):
    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)
        self.naming_scheme = ["name", "extra", "exposure_time"]


class SampleGISAXS(SampleGISAXS_Generic):
    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)
        self.naming_scheme = ["name", "extra", "x", "y", "exposure_time"]
        self.naming_scheme = ["name", "extra", "clock", "temperature", "exposure_time"]


class Sample(SampleTSAXS):
    # class Sample(SampleGISAXS):

    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)

        # self.WAXSx_pos_default=-16.9
        # self.DETx_pos_default=0

        # self.WAXSx_pos=-14.9
        # self.DETx_pos=0
        # self.WAXS_pos=[self.WAXSx_pos, self.DETx_pos]

        # self.detector='SAXS'

        self.SAXS_exposure_time = 30
        self.WAXS_exposure_time = 30

        # self.naming_scheme = ['name', 'extra', 'clock', 'temperature', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'x', 'y', 'exposure_time']
        self.naming_scheme = ["name", "extra", "exposure_time"]
        # self.naming_scheme = ['name', 'extra',  'clock','exposure_time']
        # for GISAXS/GIWAXS
        # self.naming_scheme = ['name', 'extra',  'temperature', 'clock','exposure_time']

        self.md["exposure_time"] = 30

        # self.incident_angles_default = [0.08, 0.10, 0.12, 0.15, 0.20]
        # self.incident_angles_default = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        self.incident_angles_default = [0.05, 0.08, 0.10, 0.12, 0.15]
        # self.incident_angles_default = [0.02, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1, 0.12, 0.15]
        # self.incident_angles_default = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15]
        # self.incident_angles_default = [0.0]

        self.x_pos_default = [-1, 0, 1]

    def _set_axes_definitions(self):
        """Internal function which defines the axes for this stage. This is kept
        as a separate function so that it can be over-ridden easily."""
        super()._set_axes_definitions()

        self._axes_definitions.append(
            {
                "name": "phi",
                "motor": srot,
                "enabled": True,
                "scaling": +1.0,
                "units": "deg",
                "hint": None,
            }
        )
        self._axes_definitions.append(
            {
                "name": "trans2",
                "motor": strans2,
                "enabled": True,
                "scaling": +1.0,
                "units": "deg",
                "hint": None,
            }
        )

    def _measureTimeSeries(
        self,
        exposure_time=None,
        num_frames=10,
        wait_time=None,
        extra=None,
        measure_type="measureTimeSeries",
        verbosity=3,
        **md,
    ):
        self.naming_scheme_hold = self.naming_scheme
        self.naming_scheme = ["name", "extra", "clock", "exposure_time"]
        super().measureTimeSeries(
            exposure_time=exposure_time,
            num_frames=num_frames,
            wait_time=wait_time,
            extra=extra,
            measure_type=measure_type,
            verbosity=verbosity,
            **md,
        )
        self.naming_scheme = self.naming_scheme_hold

    def goto(self, label, verbosity=3, **additional):
        super().goto(label, verbosity=verbosity, **additional)
        # You can add customized 'goto' behavior here

    def scan_SAXSdet(self, exposure_time=None):
        SAXS_pos = [-73, 0, 73]
        # SAXSx_pos=[-65, 0, 65]

        RE.md["stitchback"] = True

        for SAXSx_pos in SAXS_pos:
            for SAXSy_pos in SAXS_pos:
                mov(SAXSx, SAXSx_pos)
                mov(SAXSy, SAXSy_pos)
                self.measure(10)

    def do(self, step=0, verbosity=3, **md):
        """Performs the "default action" for this sample. This usually means
        aligning the sample, and taking data.

        The 'step' argument can optionally be given to jump to a particular
        step in the sequence."""

        if verbosity >= 4:
            print("  doing sample {}".format(self.name))

        if step <= 1:
            if verbosity >= 5:
                print("    step 1: goto origin")
            self.xo()  # goto origin
            self.yo()
            # self.gotoAlignedPosition()

        # if step<=5:
        # self.align()

        if step <= 10:
            if verbosity >= 5:
                print("    step 10: measuring")
            saxs_on()
            self.measure(exposure_time=self.SAXS_exposure_time)

            waxs_on()
            self.measure(exposure_time=self.WAXS_exposure_time)

    def doCustom(self, step=0, exposure_time=1, verbosity=3, **md):
        """Performs the "default action" for this sample. This usually means
        aligning the sample, and taking data.

        The 'step' argument can optionally be given to jump to a particular
        step in the sequence."""

        if verbosity >= 4:
            print("  doing sample {}".format(self.name))

        if step <= 1:
            if verbosity >= 5:
                print("    step 1: goto origin")
            self.xo()  # goto origin
            self.yo()
            # self.gotoAlignedPosition()

        # if step<=5:
        # self.align()

        if step <= 10:
            if verbosity >= 5:
                print("    step 10: measuring")
            self.measure(exposure_time=exposure_time)

    # def do(self, step=0, align_step=0, **md):

    # if step<=1:
    # get_beamline().modeAlignment()
    # saxs_on()

    # if step<=2:
    # self.xo() # goto origin

    # if step<=4:
    # self.yo()
    # self.tho()

    # if step<=5:
    # self.align(step=align_step, reflection_angle=0.12)
    ##self.setOrigin(['y','th']) # This is done within align

    ##if step<=7:
    ##self.xr(0.2)

    # if step<=8:
    # get_beamline().modeMeasurement()

    # if step<=10:
    # saxs_on()
    ##for detector in get_beamline().detector:
    ##detector.setExposureTime(self.SAXS_exposure_time)
    ##self.measureIncidentAngles(self.incident_angles_default, **md)
    ##self.thabs(0.0)
    # waxs_on()
    # for detector in get_beamline().detector:
    # detector.setExposureTime(self.WAXS_exposure_time)
    # self.measureIncidentAngles(self.incident_angles_default, **md)
    # self.thabs(0.0)

    def alignVeryQuick(
        self,
        intensity=INTENSITY_EXPECTED_025,
        align_step=9,
        reflection_angle=0.07,
        mode_control=True,
        verbosity=3,
    ):
        if mode_control:
            get_beamline().modeAlignment()
            beam.on()
            caput("XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime", 0.25)
            caput("XF:11BMB-ES{Det:PIL2M}:cam1:AcquirePeriod", 0.3)

        # self.yo()
        self.tho()

        # fit_scan(smy, 0.4, 13, fit='HMi')
        # fit_scan(sth, 0.8, 21, fit='COM')

        self.ysearch(step_size=0.05, min_step=0.01, intensity=intensity, target=0.5, polarity=-1)
        self.thsearch(step_size=0.1, min_step=0.01, target="max")

        self.setOrigin(["y", "th"])
        # self.align(step=align_step, reflection_angle=reflection_angle, verbosity=verbosity)

        if mode_control:
            beam.off()
            get_beamline().modeMeasurement()

    def IC_int(self):
        ion_chamber_readout1 = caget("XF:11BMB-BI{IM:3}:IC1_MON")
        ion_chamber_readout2 = caget("XF:11BMB-BI{IM:3}:IC2_MON")
        ion_chamber_readout3 = caget("XF:11BMB-BI{IM:3}:IC3_MON")
        ion_chamber_readout4 = caget("XF:11BMB-BI{IM:3}:IC4_MON")

        ion_chamber_readout = (
            ion_chamber_readout1 + ion_chamber_readout2 + ion_chamber_readout3 + ion_chamber_readout4
        )

        return ion_chamber_readout > 1 * 5e-08


class CapillaryHolderCustom(CapillaryHolder):
    def __init__(self, name="CapillaryHolder", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)
        # for TSAXS/WAXS
        self._axes["x"].origin = -16.9  # vacuum
        self._axes["y"].origin = -1.97
        # for thermal TSAXS/WAXS
        # self._axes['x'].origin = -16.2
        # self._axes['y'].origin = -3.87

        # temperature ramp is defined here
        # self.temp_series_cooling = np.arange(95, 40-.1, -5)
        self.temp_series_heating1 = np.arange(30, 100 + 0.1, 10)
        self.temp_series_heating2 = np.arange(100, 150 + 0.1, 5)
        self.temp_series_heating3 = np.arange(150, 220 + 0.1, 2)
        # self.temp_series_heating2 = np.arange(100, 150, 5)
        # self.temp_series_heating3 = np.arange(150, 235+.1, 2)
        self.temp_series = np.concatenate(
            (
                self.temp_series_heating1,
                self.temp_series_heating2,
                self.temp_series_heating3,
            ),
            axis=0,
        )
        # self.temp_series_cooling = np.arange(230, 100-.1, -5)
        # self.temp_series = np.concatenate((self.temp_series_heating, self.temp_series_cooling), axis=0)
        # self.temp_series = self.temp_series_heating2

    def doSamples(self):
        swaxs_on()
        for sample in self.getSamples():
            sample.doCustom(exposure_time=sample.SAXS_exposure_time)
        # waxs_on()
        # for sample in self.getSamples():
        # sample.doCustom(exposure_time=sample.WAXS_exposure_time)

    def doSamplesTime(self, num=3000, wait_time=None):
        # saxs_on()
        # for sample in self.getSamples():
        # sample.doCustom(exposure_time=sample.SAXS_exposure_time)
        for sample in self.getSamples():
            sample.reset_clock()
        # waxs_on()
        for ii in range(num):
            for sample in self.getSamples():
                sample.do()
            if wait_time is not None:
                time.sleep(wait_time)

    def doTemperature_cool(
        self,
        step=0,
        exposure_time_SAXS=30.0,
        exposure_time_WAXS=30.0,
        temperature_tolerance=0.2,
        wait_time=60,
        verbosity=3,
        x_offset=-1,
        x_step=+0.10,
        **md,
    ):
        for sample in self.getSamples():
            # sample.incident_angles_default = [0.08, 0.1, 0.12]
            sample.naming_scheme = [
                "name",
                "extra",
                "temperature",
                "clock",
                "exposure_time",
            ]

        ##run the Bar at RT
        # if step < 1:
        # self.doSamples()

        ##1st heating up to 200C directly
        # if step < 5:

        # self.setTemperature(95)
        # while abs(self.temperature(verbosity=0) - 95)>temperature_tolerance:
        # if verbosity>=3:
        # print('  setpoint = {:.3f}°C, Temperature = {:.3f}°C          \r'.format(self.temperature_setpoint()-273.15, self.temperature(verbosity=0)), end='')
        # time.sleep(2)
        # if wait_time is not None:
        # time.sleep(wait_time)

        # self.doSamples()

        # cooling process
        if step < 10:
            flow_max()
            for temperature in self.temp_series_cooling:
                self.setTemperature(temperature)
                while abs(self.temperature(verbosity=0) - temperature) > temperature_tolerance:
                    if verbosity >= 3:
                        print(
                            "  setpoint = {:.3f}°C, Temperature = {:.3f}°C          \r".format(
                                self.temperature_setpoint() - 273.15,
                                self.temperature(verbosity=0),
                            ),
                            end="",
                        )
                    time.sleep(2)
                if wait_time is not None:
                    time.sleep(wait_time)

                # Measure
                get_beamline().modeMeasurement()
                saxs_on()
                for sample in self.getSamples():
                    sample.gotoOrigin(["x", "y", "th"])
                    # sample.xr(x_offset)
                    sample.measure(exposure_time=exposure_time_SAXS)
                waxs_on()
                for sample in self.getSamples():
                    sample.gotoOrigin(["x", "y", "th"])
                    sample.measure(exposure_time=exposure_time_WAXS)
        ## heating process
        # if step < 15:
        # flow_off()
        # for temperature in self.temp_series_cooling :
        # self.setTemperature(temperature)
        # while abs(self.temperature(verbosity=0) - temperature)>temperature_tolerance:
        # if verbosity>=3:
        # print('  setpoint = {:.3f}°C, Temperature = {:.3f}°C          \r'.format(self.temperature_setpoint()-273.15, self.temperature(verbosity=0)), end='')
        # time.sleep(2)
        # if wait_time is not None:
        # time.sleep(wait_time)

        ## Measure
        # get_beamline().modeMeasurement()
        # saxs_on()
        # for sample in self.getSamples():
        # sample.gotoOrigin(['x','y','th'])
        ##sample.xr(x_offset)
        # sample.measure(exposure_time=exposure_time_SAXS)
        # waxs_on()
        # for sample in self.getSamples():
        # sample.gotoOrigin(['x','y','th'])
        # sample.measure(exposure_time=exposure_time_WAXS)

        flow_off()

    def doTemperature_heat(
        self,
        step=0,
        exposure_time_SAXS=30.0,
        exposure_time_WAXS=30.0,
        temperature_probe="A",
        temperature_tolerance=0.4,
        wait_time=120,
        verbosity=3,
        x_offset=-1,
        x_step=+0.10,
        **md,
    ):
        for sample in self.getSamples():
            sample.naming_scheme = [
                "name",
                "extra",
                "temperature",
                "clock",
                "exposure_time",
            ]
            # sample.naming_scheme = ['name', 'extra', 'temperature_B', 'clock', 'exposure_time']

        # run the Bar at RT
        if step < 1:
            self.doSamples()

        # heating process
        if step < 15:
            # flow_off()
            for temperature in self.temp_series_heating:
                self.setTemperature(temperature)
                while (
                    abs(self.temperature(temperature_probe=temperature_probe, verbosity=0) - temperature)
                    > temperature_tolerance
                ):
                    if verbosity >= 3:
                        print(
                            "  setpoint = {:.3f}°C, Temperature = {:.3f}°C          \r".format(
                                self.temperature_setpoint() - 273.15,
                                self.temperature(verbosity=0),
                            ),
                            end="",
                        )
                    time.sleep(2)
                if wait_time is not None:
                    time.sleep(wait_time)

                # Measure
                get_beamline().modeMeasurement()
                saxs_on()
                for sample in self.getSamples():
                    sample.gotoOrigin(["x", "y", "th"])
                    # sample.xr(x_offset)
                    sample.measure(exposure_time=exposure_time_SAXS)
                waxs_on()
                for sample in self.getSamples():
                    sample.gotoOrigin(["x", "y", "th"])
                    sample.measure(exposure_time=exposure_time_WAXS)
        self.setTemperature(25)

    def doTemperature_ramp(
        self,
        step=0,
        exposure_time_SAXS=30.0,
        exposure_time_WAXS=30.0,
        temperature_probe="A",
        temperature_tolerance=0.4,
        wait_time=120,
        verbosity=3,
        x_offset=-1,
        x_step=+0.10,
        **md,
    ):
        for sample in self.getSamples():
            sample.naming_scheme = [
                "name",
                "extra",
                "temperature",
                "clock",
                "exposure_time",
            ]
            # sample.naming_scheme = ['name', 'extra', 'temperature_B', 'clock', 'exposure_time']

        # run the Bar at RT
        # if step < 1:
        # self.doSamples()

        # heating process
        if step < 5:
            # flow_off()
            for temperature in self.temp_series:
                self.setTemperature(temperature)
                while (
                    abs(self.temperature(temperature_probe=temperature_probe, verbosity=0) - temperature)
                    > temperature_tolerance
                ):
                    if verbosity >= 3:
                        print(
                            "  setpoint = {:.3f}°C, Temperature = {:.3f}°C          \r".format(
                                self.temperature_setpoint() - 273.15,
                                self.temperature(verbosity=0),
                            ),
                            end="",
                        )
                    time.sleep(2)
                if wait_time is not None:
                    time.sleep(wait_time)

                # Measure
                get_beamline().modeMeasurement()
                swaxs_on()
                for sample in self.getSamples():
                    sample.gotoOrigin(["x", "y", "th"])
                    # sample.xr(x_offset)
                    sample.measure(exposure_time=exposure_time_SAXS)
                # waxs_on()
                # for sample in self.getSamples():
                # sample.gotoOrigin(['x','y','th'])
                # sample.measure(exposure_time=exposure_time_WAXS)
        self.setTemperature(25)

    def doTemperature_static(
        self,
        step=0,
        T_target=120,
        exposure_time_SAXS=30.0,
        exposure_time_WAXS=30.0,
        temperature_tolerance=0.2,
        wait_time=240,
        verbosity=3,
        x_offset=-1,
        x_step=+0.10,
        **md,
    ):
        for sample in self.getSamples():
            # sample.incident_angles_default = [0.08, 0.1, 0.12]
            sample.naming_scheme = [
                "name",
                "extra",
                "temperature",
                "clock",
                "exposure_time",
            ]

        ##run the Bar at RT
        # if step < 1:
        # self.doSamples()

        # keep T at 120degC
        if step < 10:
            # flow_max()
            self.setTemperature(T_target)
            for ii in range():
                get_beamline().modeMeasurement()
                swaxs_on()
                for sample in self.getSamples():
                    sample.gotoOrigin(["x", "y", "th"])
                    # sample.xr(x_offset)
                    sample.measure(exposure_time=exposure_time_SAXS)
                time.sleep(wait_time)

        if step < 10:
            # flow_max()
            self.setTemperature(25)


class GIBarCustom(GIBar_long_thermal):
    # class GIBarCustom(GIBar):

    def __init__(self, name="GIBar", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        temp_series0 = np.arange(30, 200 + 0.1, 10)
        # temp_series_cooling1 = np.arange(200, 150-.1, -5)
        # temp_series_cooling2 = np.arange(140, 70-.1, -10)
        # temp_series_cooling3 = np.arange(65, 40-.1, -5)

        # self.temp_series_cooling1 = np.arange(190, 30-.1, -10)
        # self.temp_series_heating1 = np.arange(30, 200+.1, 10)

        self.temp_series_cooling1 = np.array([160, 120, 80, 40])
        self.temp_series_heating1 = np.array([80, 120, 160, 200])

        temp_series1 = np.arange(40, 65 + 0.1, 5)
        temp_series2 = np.arange(70, 140 + 0.1, 10)
        temp_series3 = np.arange(150, 200 + 0.1, 5)

        temp_series_cooling2 = np.arange(200, 30 - 0.1, -10)

        # self.temp_series = np.arange(30, 200+.1, 10)
        # self.temp_series = np.arange(130, 200+.1, 10)
        # self.temp_series = np.arange(140, 200+.1, 10)

        # self.temp_series = np.concatenate((temp_series_cooling, temp_series1, temp_series2, temp_series3), axis=0)
        # self.temp_series = np.concatenate((temp_series1, temp_series2, temp_series3), axis=0)

        # heating only
        # self.temp_series = np.concatenate((temp_series1, temp_series2, temp_series3, temp_series_cooling2), axis=0)
        # self.temp_align = [60, 90, 120, 150, 180]
        self.temp_align = [40, 80, 120, 160, 200]
        # self.temperature_series = []

    def doTemperature_series(
        self,
        step=0,
        exposure_time_SAXS=10.0,
        exposure_time_WAXS=40.0,
        temperature_tolerance=0.4,
        wait_time=200,
        verbosity=3,
        x_offset=-1,
        x_step=+0.10,
        **md,
    ):
        self.incident_angles_default = [0.08, 0.1, 0.12]
        self.naming_scheme = [
            "name",
            "extra",
            "temperature",
            "th",
            "x",
            "exposure_time",
        ]

        # run the sample at RT
        # if step < 1:
        # self.doSamples()

        ##1st heating up to 200C directly
        if step < 5:
            if wait_time is not None:
                time.sleep(wait_time)

            self.measureIncidentAngles()

        # cooling process
        if step < 10:
            flow_max()
            for temperature in self.temp_series_cooling1:
                self.setTemperature(temperature)
                while abs(self.temperature(verbosity=0) - temperature) > temperature_tolerance:
                    if verbosity >= 3:
                        print(
                            "  setpoint = {:.3f}°C, Temperature = {:.3f}°C          \r".format(
                                self.temperature_setpoint() - 273.15,
                                self.temperature(verbosity=0),
                            ),
                            end="",
                        )
                    time.sleep(2)
                if wait_time is not None:
                    time.sleep(wait_time)

                if temperature in self.temp_align:
                    saxs_on()

                    x_offset += x_step

                    # Realign
                    if False:
                        for sample in self.getSamples():
                            sample.gotoOrigin(["x", "y", "th"])
                            sample.xr(x_offset)
                            sample.alignQuick(align_step=8, reflection_angle=0.07)
                    else:
                        # Very quick align
                        get_beamline().modeAlignment()
                        beam.on()
                        caput("XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime", 0.25)
                        caput("XF:11BMB-ES{Det:PIL2M}:cam1:AcquirePeriod", 0.30)

                        for sample in self.getSamples():
                            sample.gotoOrigin(["x", "y", "th"])
                            sample.xr(x_offset)
                            sample.alignVeryQuick(intensity=INTENSITY_EXPECTED_025, mode_control=False)

                        get_beamline().modeMeasurement()

                # Measure
                get_beamline().modeMeasurement()
                saxs_on()
                for sample in self.getSamples():
                    sample.gotoOrigin(["x", "y", "th"])
                    sample.xr(x_offset)
                    sample.measureIncidentAngles(exposure_time=exposure_time_SAXS, **md)
                    sample.thabs(0.0)

                waxs_on()
                for sample in self.getSamples():
                    sample.gotoOrigin(["x", "y", "th"])
                    sample.xr(x_offset)
                    if sample.name == "2B":
                        sample.measureIncidentAngles(exposure_time=exposure_time_WAXS / 3, **md)
                    else:
                        sample.measureIncidentAngles(exposure_time=exposure_time_WAXS, **md)
                    sample.thabs(0.0)

        # heating process
        if step < 15:
            flow_off()
            for temperature in self.temp_series_heating1:
                self.setTemperature(temperature)
                while abs(self.temperature(verbosity=0) - temperature) > temperature_tolerance:
                    if verbosity >= 3:
                        print(
                            "  setpoint = {:.3f}°C, Temperature = {:.3f}°C          \r".format(
                                self.temperature_setpoint() - 273.15,
                                self.temperature(verbosity=0),
                            ),
                            end="",
                        )
                    time.sleep(2)
                if wait_time is not None:
                    time.sleep(wait_time)

                if temperature in self.temp_align:
                    saxs_on()

                    x_offset += x_step

                    # Realign
                    if False:
                        for sample in self.getSamples():
                            sample.gotoOrigin(["x", "y", "th"])
                            sample.xr(x_offset)
                            sample.alignQuick(align_step=8, reflection_angle=0.07)
                    else:
                        # Very quick align
                        get_beamline().modeAlignment()
                        beam.on()
                        caput("XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime", 0.25)
                        caput("XF:11BMB-ES{Det:PIL2M}:cam1:AcquirePeriod", 0.30)

                        for sample in self.getSamples():
                            sample.gotoOrigin(["x", "y", "th"])
                            sample.xr(x_offset)
                            sample.alignVeryQuick(intensity=INTENSITY_EXPECTED_025, mode_control=False)

                        beam.off()
                        get_beamline().modeMeasurement()

                # Measure
                get_beamline().modeMeasurement()
                saxs_on()
                for sample in self.getSamples():
                    sample.gotoOrigin(["x", "y", "th"])
                    sample.xr(x_offset)
                    sample.measureIncidentAngles(exposure_time=exposure_time_SAXS, **md)
                    sample.thabs(0.0)

                waxs_on()
                for sample in self.getSamples():
                    sample.gotoOrigin(["x", "y", "th"])
                    sample.xr(x_offset)
                    if sample.name == "2B":
                        sample.measureIncidentAngles(exposure_time=exposure_time_WAXS / 3, **md)
                    else:
                        sample.measureIncidentAngles(exposure_time=exposure_time_WAXS, **md)
                    sample.thabs(0.0)

        if step < 20:
            self.setTemperature(25)

            # Measure
            get_beamline().modeMeasurement()
            saxs_on()
            for sample in self.getSamples():
                sample.gotoOrigin(["x", "y", "th"])
                sample.xr(x_offset)
                sample.measureIncidentAngles(exposure_time=exposure_time_SAXS, **md)
                sample.thabs(0.0)

            waxs_on()
            for sample in self.getSamples():
                sample.gotoOrigin(["x", "y", "th"])
                sample.xr(x_offset)
                sample.measureIncidentAngles(exposure_time=exposure_time_WAXS, **md)
                sample.thabs(0.0)

            saxs_on()


def waxs_on():
    detselect(pilatus800)
    WAXSx.move(-200)
    WAXSy.move(20)


def saxs_on():
    detselect(pilatus2M)
    # WAXSx.move(-193)
    WAXSy.move(24)


def swaxs_on():
    detselect([pilatus2M, pilatus800])
    # WAXSx.move(-193)
    WAXSx.move(-200)
    WAXSy.move(24)


# cms.SAXS.setCalibration([764, 1680-579], 2.0, [-60, -71]) # 13.5 keV

cms.SAXS.setCalibration([780, 1680 - 603], 5.03, [-60, -73])  # 13.5 keV
# cms.WAXS.setCalibration([460, 1043-432], 0.371, [-194.5, 13.635515625])


RE.md["experiment_group"] = "Chris Li (Drexel)"
RE.md["experiment_alias_directory"] = "/GPFS/xf11bm/data/2019_2/CLi/"
RE.md["experiment_proposal_number"] = "303632"
RE.md["experiment_SAF_number"] = "304391"
RE.md["experiment_user"] = "various"
RE.md["experiment_type"] = "TSAXS"  # TSAXS, GISAXS, GIWAXS, etc.
RE.md["experiment_project"] = "Crystallization at L/L interface"


if False:
    # Example of a multi-sample holder

    md = {
        "owner": "beamline",
        "series": "various",
    }

    # hol = GIBar(base=stg)
    hol = GIBarCustom(base=stg)

    # hol.addSampleSlotPosition( Sample('Brush_0', **md), 1, 3)
    # hol.addSampleSlotPosition( Sample('Brush_0.25', **md), 2, 25-8 )
    # hol.addSampleSlotPosition( Sample('Brush_0.5', **md), 3, 46-10-3)
    hol.addSampleSlotPosition(Sample("PLLA_b_PEO emulsion", **md), 5, 66 - 10 - 6)
    hol.addSampleSlotPosition(Sample("LA-PEO5k_S_GISAXS", **md), 5, 86 - 20 - 1)
    hol.addSampleSlotPosition(Sample("LA-PEO5k_H_GISAXS", **md), 6, 90 - 20 + 14)
    hol.addSampleSlotPosition(Sample("LA-PEO5k_B_GISAXS", **md), 7, 105 - 3)
    # hol.addSampleSlotPosition( Sample('Yen_2018-1_sampleD', **md), 8, 85)
    # hol.addSampleSlotPosition( Sample('JH_60_40_arearatio_S6_8nm_S3_16nm_solid', **md), 9, 82.3, 'SAXS' )
    # hol.addSampleSlotPosition( Sample('JH_none_arearatio_S6_8nm_S3_16nm_solid', **md), 10, 91.9, 'SAXS' )


# hol.addSampleSlotPosition( Sample('AgBH_5m_insitu+GIBar', **md), 7, 70 )


if True:
    # hol = CapillaryHolder(base=stg)
    hol = CapillaryHolderCustom(base=stg)

    hol.addSampleSlot(Sample("Sarah_T24"), 1.0)
    hol.addSampleSlot(Sample("Sarah_T48"), 2.0)
    hol.addSampleSlot(Sample("Sarah_T72"), 3.0)
    hol.addSampleSlot(Sample("Sarah_U4"), 4.0)
    hol.addSampleSlot(Sample("Sarah_U12"), 5.0)
    hol.addSampleSlot(Sample("Sarah_U24"), 6.0)
    hol.addSampleSlot(Sample("Sarah_U48"), 7.0)
    hol.addSampleSlot(Sample("Sarah_U72"), 8.0)
    hol.addSampleSlot(Sample("Yongwei_1M_PEO_2%"), 9.0)
    hol.addSampleSlot(Sample("Yongwei_2M_PEO_2%"), 10.0)
    hol.addSampleSlot(Sample("Yongwei_5M_PEO_2%"), 11.0)
    hol.addSampleSlot(Sample("Yongwei_2_PAN"), 12.0)
    hol.addSampleSlot(Sample("Yongwei_5_PAN"), 13.0)
    hol.addSampleSlot(Sample("Yongwei_10_PAN"), 14.0)
    # hol.addSampleSlot( Sample('Sarah_T12'), 15.0 )
    # hol.addSampleSlot( Sample('Sarah_A4'), 15.0 )h

if False:
    # hol = CapillaryHolder(base=stg)
    hol = CapillaryHolderCustom(base=stg)

    hol.addSampleSlot(Sample("FLScreen"), 5.0)
    hol.addSampleSlot(Sample("Sarah_B4"), 6.0)
    hol.addSampleSlot(Sample("Sarah_B14"), 7.0)
    hol.addSampleSlot(Sample("AgBH_5m_cali"), 8.0)
    hol.addSampleSlot(Sample("Sarah_C4"), 9.0)
    hol.addSampleSlot(Sample("Sarah_C14"), 10.0)
    hol.addSampleSlot(Sample("Empty"), 11.0)

    # hol.addSampleSlot( Sample('Sarah_A4'), 15.0 )h


# %run -i /GPFS/xf11bm/data/2018_1/beamline/user.py

# robot.listGarage()

"""
The time-resolved expt. at 125degC
Four samples, two fiber and two polymer sheets, are put on the bottom of dia 2mm capillaries. 
Heat the stage to 125C. add solvent (MCB:DMF, 19:1) into Capillary and put them directly onto the hot stage. 
The fiber samples need to locate the scattering position and then run hol.doSamplesTime()
Unfortuenately, it doesnot work as planned, as the sample was dissolved in the solvent and lose the structure


"""


"""
Protocol for capillary holder
cms.ventChamber()
--load samples
cms.pumpChamber()
--name samples
%run -i /nsls2/xf11bm/data/2019_2/CLi/user.py
hol.listSamples()
hol.doSamples()
"""
