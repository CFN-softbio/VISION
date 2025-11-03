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
import os
from shutil import copyfile

from ophyd import EpicsSignal
from bluesky.suspenders import SuspendFloor, SuspendCeil

ring_current = EpicsSignal("SR:OPS-BI{DCCT:1}I:Real-I")
sus = SuspendFloor(ring_current, 100, resume_thresh=400, sleep=600)
RE.install_suspender(sus)

# absorber_pos = EpicsSignal( 'XF:11BMB-ES{SM:1-Ax:ArmR}Mtr.RBV')
# sus_abs_low = SuspendFloor(absorber_pos, -56, resume_thresh=-55)
# sus_abs_hi = SuspendCeil(absorber_pos, -54, resume_thresh=-55)
# RE.install_suspender(sus_abs_low)
# RE.install_suspender(sus_abs_hi)
# from ophyd import EpicsSignal
# from bluesky.suspenders import SuspendFloor

# beam_current = EpicsSignal('SR:OPS-BI{DCCT:1}I:Real-I')
# sus = SuspendFloor(beam_current, 100, resume_thresh=101)
# RE.install_suspender(sus)
# RE.clear_suspenders()


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
        self.naming_scheme = ["name", "extra", "th", "exposure_time"]
        self.naming_scheme = ["name", "extra", "clock", "temperature", "exposure_time"]


# class Sample(SampleTSAXS):
class Sample(SampleGISAXS):
    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)

        # SVA_chamber
        # self._axes['x'].origin = -4
        # self._axes['y'].origin = 14.5
        # self._axes['th'].origin = 0

        # self.detector='SAXS'

        # self.naming_scheme = ['name', 'extra', 'clock', 'temperature', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'y', 'th', 'clock', 'exposure_time']
        self.naming_scheme = [
            "name",
            "extra",
            "clock",
            "humidity",
            "th",
            "exposure_time",
        ]
        # self.naming_scheme = ['name', 'extra', 'clock', 'temperature', 'exposure_time']

        self.md["exposure_time"] = 1
        self.SAXS_time = 10
        self.WAXS_time = 15

        # self.incident_angles_default = [0.08, 0.10, 0.12, 0.15, 0.20]
        # self.incident_angles_default = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        # self.incident_angles_default = [0.08, 0.12, 0.14, 0.20, 0.26, 0.32] #for 17kev/15kev
        # self.incident_angles_default = [0.10, 0.16, 0.20] #for 17kev/15kev
        # self.incident_angles_default = [0.08, 0.12, 0.15, 0.18, 0.20] #for 10kev
        # self.incident_angles_default = [0.08, 0.12, 0.15, 0.18] #for 10kev LJR
        # self.incident_angles_default = [0.12, 0.16, 0.20, 0.24] #for 10kev, Perovskites
        self.incident_angles_default = [0.08, 0.1, 0.12, 0.15]
        # self.incident_angles_default = [0.12, 0.16, 0.20, 0.24] #for 10kev, Perovskites
        # self.incident_angles_default = [0.02, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1, 0.12, 0.15]
        # self.incident_angles_default = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15]
        # self.incident_angles_default = [0.0]

        self.x_pos_default = [-1, 0, 1]

        self.total_flow = 20
        self.wetflow_default = self.total_flow * np.arange(0.1, 0.51, 0.1)
        self.wetwait_default = [1200, 1200, 1200, 1200, 1200]

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

    def do_ljr(self, step=0, align_step=0, **md):
        # NOTE: if align_step =8 is not working, try align_step=4

        if step <= 1:
            saxs_on()
            get_beamline().modeAlignment()

        if step <= 2:
            self.xo()  # goto origin

        if step <= 4:
            self.yo()
            self.tho()

        if step <= 5:
            self.align(step=align_step, reflection_angle=0.12)
            # self.setOrigin(['y','th']) # This is done within align

        # if step<=7:
        # self.xr(0.2)

        if step <= 8:
            get_beamline().modeMeasurement()

        if step <= 10:
            self.thabs(0.0)

    def do(self, step=0, align_step=0, **md):
        # NOTE: if align_step =8 is not working, try align_step=4

        if step <= 1:
            saxs_on()
            get_beamline().modeAlignment()

        if step <= 2:
            self.xo()  # goto origin

        if step <= 4:
            self.yo()
            self.tho()

        if step <= 5:
            self.align(step=align_step, reflection_angle=0.12)
            # self.setOrigin(['y','th']) # This is done within align

        # if step<=7:
        # self.xr(0.2)

        if step <= 8:
            get_beamline().modeMeasurement()

        if step <= 10:
            if self.incident_angles == None:
                incident_angles = self.incident_angles_default
            else:
                incident_angles = self.incident_angles

            # saxs_on()
            ##self._test2_measureIncidentAngles(incident_angles, exposure_time=self.SAXS_time, tiling='ygaps', **md)
            # self.measureIncidentAngles(incident_angles, exposure_time=self.SAXS_time, **md)
            swaxs_on()
            self._test2_measureIncidentAngles(incident_angles, exposure_time=self.WAXS_time, tiling="ygaps", **md)
            # self.measureIncidentAngles(incident_angles, exposure_time=self.WAXS_time, **md)
            # waxs_on()
            # self._test2_measureIncidentAngles(self.incident_angles_default, exposure_time=self.WAXS_time, tiling='ygaps', **md)

            self.thabs(0.0)

    def do_SAXS(self, step=0, align_step=0, **md):
        # NOTE: if align_step =8 is not working, try align_step=4

        if step <= 1:
            saxs_on()
            get_beamline().modeAlignment()

        if step <= 2:
            self.xo()  # goto origin

        if step <= 4:
            self.yo()
            self.tho()

        if step <= 5:
            self.align(step=align_step, reflection_angle=0.12)
            # self.setOrigin(['y','th']) # This is done within align

        # if step<=7:
        # self.xr(0.2)

        if step <= 8:
            get_beamline().modeMeasurement()

        if step <= 10:
            if self.incident_angles == None:
                incident_angles = self.incident_angles_default
            else:
                incident_angles = self.incident_angles

            # saxs_on()
            ##self._test2_measureIncidentAngles(incident_angles, exposure_time=self.SAXS_time, tiling='ygaps', **md)
            # self.measureIncidentAngles(incident_angles, exposure_time=self.SAXS_time, **md)
            saxs_on()
            self._test2_measureIncidentAngles(incident_angles, exposure_time=self.SAXS_time, tiling="ygaps", **md)
            # self.measureIncidentAngles(incident_angles, exposure_time=self.WAXS_time, **md)
            # waxs_on()
            # self._test2_measureIncidentAngles(self.incident_angles_default, exposure_time=self.WAXS_time, tiling='ygaps', **md)

            self.thabs(0.0)

    def IC_int(self):
        ion_chamber_readout1 = caget("XF:11BMB-BI{IM:3}:IC1_MON")
        ion_chamber_readout2 = caget("XF:11BMB-BI{IM:3}:IC2_MON")
        ion_chamber_readout3 = caget("XF:11BMB-BI{IM:3}:IC3_MON")
        ion_chamber_readout4 = caget("XF:11BMB-BI{IM:3}:IC4_MON")

        ion_chamber_readout = (
            ion_chamber_readout1 + ion_chamber_readout2 + ion_chamber_readout3 + ion_chamber_readout4
        )

        return ion_chamber_readout > 1 * 5e-08

    def do_TSAXS(self, step=0, align_step=0, **md):
        if step <= 1:
            saxs_on()

        if step <= 2:
            self.xo()  # goto origin

        if step <= 4:
            self.yo()
            self.tho()

        if step <= 8:
            get_beamline().modeMeasurement()

        if step <= 10:
            self.measure(exposure_time=self.SAXS_time, **md)

            # if self.exposure_time_SAXS==None:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.SAXS_time, tiling=self.tiling, **md)
            # else:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.exposure_time_SAXS, tiling=self.tiling, **md)

            self.thabs(0.0)

    def align_y(self, step=0, reflection_angle=0.12, verbosity=3):
        """Align the sample with respect to the beam. GISAXS alignment involves
        vertical translation to the beam center, and rocking theta to get the
        sample plane parralel to the beam. Finally, the angle is re-optimized
        in reflection mode.

        The 'step' argument can optionally be given to jump to a particular
        step in the sequence."""

        if verbosity >= 4:
            print("  Aligning Y {}".format(self.name))

        if step <= 0:
            # Prepare for alignment

            if RE.state != "idle":
                RE.abort()

            if get_beamline().current_mode != "alignment":
                # if verbosity>=2:
                # print("WARNING: Beamline is not in alignment mode (mode is '{}')".format(get_beamline().current_mode))
                print("Switching to alignment mode (current mode is '{}')".format(get_beamline().current_mode))
                get_beamline().modeAlignment()

            get_beamline().setDirectBeamROI()

            beam.on()

        if step <= 9 and reflection_angle is not None:
            # Final alignment using reflected beam
            if verbosity >= 4:
                print("    align: reflected beam")
            get_beamline().setReflectedBeamROI(total_angle=reflection_angle * 2.0)
            # get_beamline().setReflectedBeamROI(total_angle=reflection_angle*2.0, size=[12,2])

            self.thabs(reflection_angle)

            fit_scan(smy, 0.2, 21, fit="max")
            self.setOrigin(["y"])

        if step <= 10:
            self.thabs(0.0)
            beam.off()
            get_beamline().modeMeasurement()

    def do_WAXS_only(self, step=0, align_step=0, **md):
        if step < 5:
            self.xo()
            self.yo()
            self.tho()
            get_beamline().modeMeasurement()
        if step <= 10:
            if self.incident_angles == None:
                incident_angles = self.incident_angles_default
            else:
                incident_angles = self.incident_angles

            waxs_on()
            # for detector in get_beamline().detector:
            # detector.setExposureTime(self.MAXS_time)
            self._test2_measureIncidentAngles(incident_angles, exposure_time=self.WAXS_time, tiling="ygaps", **md)

            self.yr(-1)
            self.measure(exposure_time=self.WAXS_time, tiling="ygaps", extra="BKG", **md)

            # if self.exposure_time_MAXS==None:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.MAXS_time, tiling=self.tiling, **md)
            # else:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.exposure_time_MAXS, tiling=self.tiling, **md)

            self.thabs(0.0)
            self.yo()

    def do_WAXS(self, step=0, align_step=0, **md):
        if step <= 1:
            saxs_on()
            get_beamline().modeAlignment()

        if step <= 2:
            self.xo()  # goto origin

        if step <= 4:
            self.yo()
            self.tho()

        if step <= 5:
            self.align(step=align_step)
            # self.setOrigin(['y','th']) # This is done within align

        # if step<=7:
        # self.xr(0.2)

        if step <= 8:
            get_beamline().modeMeasurement()

        if step <= 10:
            if self.incident_angles == None:
                incident_angles = self.incident_angles_default
            else:
                incident_angles = self.incident_angles

            waxs_on()  # edited from waxs_on 3/25/19 through a saxs_on error
            # for detector in get_beamline().detector:
            # detector.setExposureTime(self.MAXS_time)
            self._test2_measureIncidentAngles(incident_angles, exposure_time=self.WAXS_time, tiling="ygaps", **md)

            # if self.exposure_time_MAXS==None:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.MAXS_time, tiling=self.tiling, **md)
            # else:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.exposure_time_MAXS, tiling=self.tiling, **md)
            self.thabs(0.0)


class HumidityStageCumstom(HumidityStage):
    def __init__(self, name="GIBarCustom", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._axes["x"].origin = -54.3
        ##position for calibration
        # self._axes['y'].origin = 19
        # self._axes['th'].origin = 0.23

        # CREATE DIRECTORIES TO SAVE DATA
        # holder_data_folder = os.path.join(parent_data_folder, self.name)
        # os.makedirs(holder_data_folder, exist_ok=True)
        # os.makedirs(os.path.join(holder_data_folder, 'waxs'), exist_ok=True)
        # os.makedirs(os.path.join(holder_data_folder, 'saxs'), exist_ok=True)
        # os.makedirs(os.path.join(holder_data_folder, 'waxs/raw'), exist_ok=True)
        # os.makedirs(os.path.join(holder_data_folder, 'saxs/raw'), exist_ok=True)
        # RE.md['experiment_alias_directory'] = holder_data_folder

        #### COPY CURRENT STATE OF user.py TO SAMPLE DIRECTORY
        # copyfile(os.path.join(parent_data_folder, 'user.py'), os.path.join(holder_data_folder,'user.py'))

    def doSamples(self, exposure_time=1, verbosity=3):
        # maxs_on()
        for sample in self.getSamples():
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            if sample.detector == "SAXS" or sample.detector == "BOTH":
                sample.do_SAXS()

        for sample in self.getSamples():
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            if sample.detector == "BOTH":
                sample.do_WAXS_only()

        for sample in self.getSamples():
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            if sample.detector == "WAXS":
                sample.do_WAXS()

    def alignSamples(self, step=0, align_step=0, verbosity=3, **md):
        for sample in self.getSamples():
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))

            if step <= 1:
                saxs_on()
                get_beamline().modeAlignment()

            if step <= 2:
                sample.xo()  # goto origin
            if step <= 4:
                sample.yo()
                sample.tho()

            if step <= 5:
                sample.align(step=align_step)

    def alignSamplesQuick(self, step=0, align_step=0, verbosity=3, **md):
        for sample in self.getSamples():
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))

            if step <= 1:
                saxs_on()
                get_beamline().modeAlignment()

            if step <= 2:
                sample.xo()  # goto origin
            if step <= 4:
                sample.yo()
                sample.tho()

            if step <= 5:
                sample.align_y(step=align_step)

    def measureSamples(self, step=0, verbosity=3, **md):
        cms.modeMeasurement()
        saxs_on()
        for sample in self.getSamples():
            sample.gotoOrigin()

            if sample.incident_angles == None:
                incident_angles = sample.incident_angles_default
            else:
                incident_angles = sample.incident_angles

            sample.measureIncidentAngles_Stitch(
                incident_angles, exposure_time=sample.SAXS_time, tiling="ygaps", **md
            )

        waxs_on()  # edited from waxs_on 3/25/19 through a saxs_on error

        for sample in self.getSamples():
            sample.gotoOrigin()

            if sample.incident_angles == None:
                incident_angles = sample.incident_angles_default
            else:
                incident_angles = sample.incident_angles
            # for detector in get_beamline().detector:
            # detector.setExposureTime(self.MAXS_time)
            sample.measureIncidentAngles_Stitch(
                incident_angles, exposure_time=sample.WAXS_time, tiling="ygaps", **md
            )

            sample.gotoOrigin()
            sample.yr(-1)
            sample.measure(exposure_time=sample.WAXS_time, extra="TWAXS", tiling="ygaps", **md)
            sample.gotoOrigin()

    def do_WAXS_only(self, step=0, align_step=0, **md):
        if step < 5:
            self.xo()
            self.yo()
            self.tho()
            get_beamline().modeMeasurement()
        if step <= 10:
            if self.incident_angles == None:
                incident_angles = self.incident_angles_default
            else:
                incident_angles = self.incident_angles

            waxs_on()
            # for detector in get_beamline().detector:
            # detector.setExposureTime(self.MAXS_time)
            self.measureIncidentAngles_Humidity(
                incident_angles, exposure_time=self.WAXS_time, tiling="ygaps", **md
            )

            self.yr(-1)
            self.measure(exposure_time=self.WAXS_time, tiling="ygaps", extra="BKG", **md)

            # if self.exposure_time_MAXS==None:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.MAXS_time, tiling=self.tiling, **md)
            # else:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.exposure_time_MAXS, tiling=self.tiling, **md)

            self.thabs(0.0)
            self.yo()

    def _test_doSamples(self, exposure_time=1, verbosity=3):
        if caget("XF:11BMA-PPS{PSh}Sts:FailOpn-Sts") == 1:
            print("The SHUTTER is closed. Please open it")
            return

        # maxs_on()
        for sample in self.getSamples():
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            if sample.detector == "SAXS" or sample.detector == "BOTH":
                sample.do_SAXS()

        for sample in self.getSamples():
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            if sample.detector == "BOTH":
                sample.do_WAXS_only()

        for sample in self.getSamples():
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            if sample.detector == "WAXS":
                sample.do_WAXS()

    def doSamples_WAXS_only(self, verbosity=3):
        for sample in self.getSamples():
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            sample.do_WAXS_only()

    def setDryFlow(self, voltage=None):
        if voltage == None or voltage > 5 or voltage < 0:
            print("Input voltage betwee 0 and 5V")
        self.setFlow(1, voltage=voltage)

    def setWetFlow(self, voltage=0):
        if voltage == None or voltage > 5 or voltage < 0:
            print("Input voltage betwee 0 and 5V")
        self.setFlow(2, voltage=voltage)


class CapillaryHolderCustom(CapillaryHolder):
    def __init__(self, name="GIBarCustom", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._axes["x"].origin = -16.7
        ##position for calibration
        self._axes["y"].origin = -1.8
        # self._axes['th'].origin = 0.23

    def doSamples(self, exposure_time_WAXS=25, exposure_time_SAXS=25, verbosity=3):
        # maxs_on()
        waxs_on()
        for sample in self.getSamples():
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            sample.gotoOrigin()
            sample.measure(exposure_time_WAXS)

        saxs_on()
        for sample in self.getSamples():
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            sample.gotoOrigin()
            sample.measure(exposure_time_SAXS)


def saxs_on():
    detselect(pilatus2M)
    WAXSx.move(-210)
    # WAXSx.move(-196)
    WAXSy.move(30)


def waxs_on():
    detselect(pilatus800)
    WAXSx.move(-193)
    WAXSy.move(18)


# def maxs_on():
# detselect(pilatus300)
# MAXSx.move(-72.5)
# MAXSy.move(58)

# def maxs_on_P3HT():
# detselect(pilatus300)
# MAXSx.move(-74)
# MAXSy.move(58)


def swaxs_on():
    detselect([pilatus2M, pilatus800])
    # WAXSx.move(-210)
    WAXSx.move(-196)
    WAXSy.move(24)


RE.md["experiment_alias_directory"] = "/nsls2/xf11bm/data/2020_2/QFu2/"


# cms.SAXS.setCalibration([737, 1089], 2.8, [-65, -73])  #20190320,
# cms.SAXS.setCalibration([755, 1075], 5.05, [-65, -73])  #20190320
# cms.SAXS.setCalibration([755, 1072], 5.05, [-65, -73])  #13.5 keV
# cms.SAXS.setCalibration([756, 1679-608], 5.05, [-65, -73])
# cms.SAXS.setCalibration([831, 1679-547], 2, [-50, -65])

cms.SAXS.setCalibration([756, 1074], 5.05, [-65, -73])

if True:
    # Example of a multi-sample holder

    md = {
        "owner": "B. Ocko (BNL)",
        "series": "various",
    }

    # QFhumidbar = HumidityStageCumstom(base=stg)
    # oe=-5
    # QFhumidbar.addSampleSlotPosition( Sample('QF_PA on PS-35 -2', **md),1, 28+oe, 'WAXS', incident_angles=[0.12])
    # QFhumidbar.addSampleSlotPosition( Sample('QF_PA on PS-35 soaked -2', **md),2, 38+oe, 'WAXS', incident_angles=[0.12])
    # QFhumidbar.addSampleSlotPosition( Sample('QF_PA on PS-24 soaked and compacted', **md),3, 45+oe, 'WAXS', incident_angles=[0.12])
    # QFhumidbar.addSampleSlotPosition( Sample('QF_untreat-not-dried', **md),4, 53+oe, 'WAXS', incident_angles=[0.12])
    # QFhumidbar.addSampleSlotPosition( Sample('QF_untreat-dried', **md),5, 65+oe, 'WAXS', incident_angles=[0.12])
    # QFhumidbar.addSampleSlotPosition( Sample('QF_treat-not-dried', **md),6, 75+oe, 'WAXS', incident_angles=[0.12])
    # QFhumidbar.addSampleSlotPosition( Sample('QF_treat-dried', **md),7, 87+oe, 'WAXS', incident_angles=[0.12])
    # QFhumidbar.addSampleSlotPosition( Sample('QF_silicon wafer-BKG', **md),8, 96+oe, 'WAXS', incident_angles=[0.12])

    # untreathumidity_bar = HumidityStageCumstom(base=stg)
    ##compacted.addGaragePosition(-1,-1)
    # oe=-8
    # untreathumidity_bar.addSampleSlotPosition( Sample('untreat-orig-1', **md),1, 35+oe, 'WAXS', incident_angles=[0.12])
    # untreathumidity_bar.addSampleSlotPosition( Sample('untreat-orig-2', **md),2, 45+oe, 'WAXS', incident_angles=[0.12])
    # untreathumidity_bar.addSampleSlotPosition( Sample('untreat-BA', **md),3, 57+oe, 'WAXS', incident_angles=[0.12])
    # untreathumidity_bar.addSampleSlotPosition( Sample('untreat-chloroform', **md),4, 67+oe, 'WAXS', incident_angles=[0.12])
    # untreathumidity_bar.addSampleSlotPosition( Sample('untreat-THF', **md),5, 80+oe, 'WAXS', incident_angles=[0.12])
    # untreathumidity_bar.addSampleSlotPosition( Sample('untreat-DCM', **md),6, 90+oe, 'WAXS', incident_angles=[0.12])

    # BWXLE_bar = HumidityStageCumstom(base=stg)
    ##compacted.addGaragePosition(-1,-1)
    # oe=-32
    # BWXLE_bar.addSampleSlotPosition( Sample('BWXLE-DCM-1', **md),1, 50+oe, 'WAXS', incident_angles=[0.12])
    # BWXLE_bar.addSampleSlotPosition( Sample('BWXLE-DCM-2', **md),2, 60+oe, 'WAXS', incident_angles=[0.12])
    # BWXLE_bar.addSampleSlotPosition( Sample('BWXLE-THF-1', **md),3, 67+oe, 'WAXS', incident_angles=[0.12])
    # BWXLE_bar.addSampleSlotPosition( Sample('BWXLE-THF-2', **md),4, 78+oe, 'WAXS', incident_angles=[0.12])
    # BWXLE_bar.addSampleSlotPosition( Sample('PAonPS35-orig-DCM-1', **md),5, 86+oe, 'WAXS', incident_angles=[0.12])
    # BWXLE_bar.addSampleSlotPosition( Sample('PAonPS35-orig-DCM-2', **md),6, 95+oe, 'WAXS', incident_angles=[0.12])

    # PAonPS35_bar = HumidityStageCumstom(base=stg)
    ##compacted.addGaragePosition(-1,-1)
    # oe=-34
    # PAonPS35_bar.addSampleSlotPosition( Sample('untreat-water', **md),1, 55+oe, 'WAXS', incident_angles=[0.12])
    # PAonPS35_bar.addSampleSlotPosition( Sample('untreat-heat-2', **md),2, 65+oe, 'WAXS', incident_angles=[0.12])
    # PAonPS35_bar.addSampleSlotPosition( Sample('PAonPS35-soaked-1', **md),3, 75+oe, 'WAXS', incident_angles=[0.12])
    # PAonPS35_bar.addSampleSlotPosition( Sample('PAonPS35-soaked-2', **md),4, 84+oe, 'WAXS', incident_angles=[0.12])
    # PAonPS35_bar.addSampleSlotPosition( Sample('PAonPS35-compacted-soaked-1', **md),5, 95+oe, 'WAXS', incident_angles=[0.12])
    # PAonPS35_bar.addSampleSlotPosition( Sample('PAonPS35-compacted-soaked-2', **md),6, 105+oe, 'WAXS', incident_angles=[0.12])

    hol1 = HumidityStageCumstom(base=stg)
    # compacted.addGaragePosition(-1,-1)
    oe = -34
    hol1.addSampleSlotPosition(Sample("QF_Bulk-M1-1", **md), 1, 13, "BOTH", incident_angles=[0.1, 0.12, 0.15])
    hol1.addSampleSlotPosition(Sample("QF_Bulk-0p25-1", **md), 2, 28, "BOTH", incident_angles=[0.1, 0.12, 0.15])
    hol1.addSampleSlotPosition(Sample("QF_Bulk-0p5-1", **md), 3, 43, "BOTH", incident_angles=[0.1, 0.12, 0.15])
    hol1.addSampleSlotPosition(Sample("QF_Bulk-M2-1", **md), 4, 56, "BOTH", incident_angles=[0.1, 0.12, 0.15])
    hol1.addSampleSlotPosition(Sample("QF_Bulk-M4-2", **md), 5, 69, "BOTH", incident_angles=[0.1, 0.12, 0.15])

    hol2 = HumidityStageCumstom(base=stg)
    # compacted.addGaragePosition(-1,-1)
    oe = -8
    hol2.addSampleSlotPosition(
        Sample("QF_FS_prinstine_1-1", **md),
        1,
        5 + oe,
        "BOTH",
        incident_angles=[0.1, 0.12, 0.15],
    )
    hol2.addSampleSlotPosition(
        Sample("QF_FS_prinstine_2-1", **md),
        2,
        15 + oe,
        "BOTH",
        incident_angles=[0.1, 0.12, 0.15],
    )
    hol2.addSampleSlotPosition(
        Sample("QF_FS_chloroform-2", **md),
        3,
        25 + oe,
        "BOTH",
        incident_angles=[0.1, 0.12, 0.15],
    )
    hol2.addSampleSlotPosition(
        Sample("QF_FS_BA-1", **md),
        4,
        37 + oe,
        "BOTH",
        incident_angles=[0.1, 0.12, 0.15],
    )
    hol2.addSampleSlotPosition(
        Sample("QF_FS_DCM-1", **md),
        5,
        45 + oe,
        "BOTH",
        incident_angles=[0.1, 0.12, 0.15],
    )
    hol2.addSampleSlotPosition(
        Sample("QF_FS_DMF-2", **md),
        6,
        55 + oe,
        "BOTH",
        incident_angles=[0.1, 0.12, 0.15],
    )
    hol2.addSampleSlotPosition(
        Sample("QF_FS_water-2", **md),
        7,
        65 + oe,
        "BOTH",
        incident_angles=[0.1, 0.12, 0.15],
    )

    # hol2 = GIBar_Custom(base=stg)
    # hol2.addGaragePosition(1,2)
    # hol2.addSampleSlotPosition( Sample('NV_1_OMIC_DW', **md),1, 7, 'WAXS', incident_angles=[0.05, 0.08, 0.1,0.12,0.15])
    # hol2.addSampleSlotPosition( Sample('NV_5_OMIC_DW', **md),2, 22, 'WAXS', incident_angles=[0.05, 0.08, 0.1,0.12,0.15])
    # hol2.addSampleSlotPosition( Sample('NV_7_OMIC_DW', **md),3, 35, 'WAXS', incident_angles=[0.05, 0.08, 0.1,0.12,0.15])
    # hol2.addSampleSlotPosition( Sample('NV_10_OMIC_DW', **md),4, 47, 'WAXS', incident_angles=[0.05, 0.08, 0.1,0.12,0.15])
    # hol2.addSampleSlotPosition( Sample('NV_1_BMIC_DW', **md),5, 65, 'WAXS', incident_angles=[0.05, 0.08, 0.1,0.12,0.15])
    # hol2.addSampleSlotPosition( Sample('NV_5_BMIC_DW', **md),6, 75, 'WAXS', incident_angles=[0.05, 0.08, 0.1,0.12,0.15])
    # hol2.addSampleSlotPosition( Sample('NV_7_BMIC_DW', **md),7, 91, 'WAXS', incident_angles=[0.05, 0.08, 0.1,0.12,0.15])
    # hol2.addSampleSlotPosition( Sample('NV_10_BMIC_DW', **md),8, 104, 'WAXS', incident_angles=[0.05, 0.08, 0.1,0.12,0.15])

    # Garage_holders=[hol3, hol4, hol5, hol6, hol7, hol8, hol9, hol10]  #, hol11, hol12]
    # Garage_holders=[hol11, hol1, hol2, hol12, hol13, hol16, hol17, hol9, hol10, hol18, hol14, hol15]  #, hol11, hol12]
    # robot.run()


if 0:
    hol = CapillaryHolder(base=stg)
    # hol = CapillaryHolderCustom(base=stg)

    hol.addSampleSlot(Sample("test1"), 1.0)
    hol.addSampleSlot(Sample("test2"), 2.0)
    hol.addSampleSlot(Sample("test2"), 3.0)
    # hol.addSampleSlot( Sample('YT_Sep08_2017_sample903'), 3.0 )
    # hol.addSampleSlot( Sample('YT_Sep08_2017_sample904'), 4.0 )
    hol.addSampleSlot(Sample("FL_screen"), 5.0)
    # hol.addSampleSlot( Sample('YT_Sep08_2017_sample906'), 6.0 )
    # hol.addSampleSlot( Sample('YT_Sep08_2017_sample907'), 7.0 )
    hol.addSampleSlot(Sample("AgBH_cali_5m"), 8.0)
    # hol.addSampleSlot( Sample('AgBH_3m_cali'), 8.0 )
    # hol.addSampleSlot( Sample('YT_Sep08_2017_sample909'), 9.0 )
    # hol.addSampleSlot( Sample('YT_Sep08_2017_sample910'), 10.0 )
    # hol.addSampleSlot( Sample('YT_Sep08_2017_sample911'), 11.0 )
    hol.addSampleSlot(Sample("test"), 11.0)
    # hol.addSampleSlot( Sample('YT_Sep08_2017_sample912'), 12.0 )
    # hol.addSampleSlot( Sample('YT_Sep08_2017_sample913'), 13.0 )
    # hol.addSampleSlot( Sample('YT_Sep08_2017_sample914'), 14.0 )
    # hol.addSampleSlot( Sample('YT_Sep08_2017_sample915'), 15.0 )


# %run -i /GPFS/xf11bm/data/2018_1/beamline/user.py

# robot.listGarage()


"""
## for GI
sam=hol0.gotoSample(7)
sam.xr(0.2)
sam.thabs(0.12)
sam.measure(120)

sam.do()

## for transmission
robot_pickup(<raw_1-4>, <slot_1-3>)
hol#.gotoOrigin()
hol#.xr(...)    ## to center the holder central hole
hol#.yr(...)    ## to center the holder central hole
hol#.setOrigin(['x','y'])
hol#.doSamples()        ## does 10sec WAXS, and then 10sec SAXS
   --or--
hol#.doSamples(exposure_time_WAXS=20, exposure_time_SAXS=20)    # for non-default exposure times
robot_return(<raw_1-4>, <slot_1-3>)

#humdity_list=[0,20,40,60,80,100,0]
wetFlow_list= [ 0, 2.8, 3,  3.9,  3.0,   5, 0]
dryFlow_list= [ 5, 3.0, 3,  3.0,  2.3,   0, 5]
ritikahumidbar.alignSamples()
for ii in range(7):    
    ritikahumidbar.setDryFlow(dryFlow_list[ii])
    ritikahumidbar.setWetFlow(wetFlow_list[ii])
    time.sleep(600)
    ritikahumidbar.measureSamples()

#humdity_list=[100,80,60,40,20,0,100,0]
wetFlow_list= [ 5, 1.5,   1.5,  1.5,  1.35,   0, 5, 0]
dryFlow_list= [ 0, 1,     1.55, 1.7,  2.0,    5, 0, 5]
#untreathumidity_bar.alignSamples()
for ii in range(8):    
    untreathumidity_bar.setDryFlow(dryFlow_list[ii])
    untreathumidity_bar.setWetFlow(wetFlow_list[ii])
    time.sleep(1200)
    untreathumidity_bar.measureSamples()
    
    
#humdity_list=[100,80,60,40,20,0,100,0]
wetFlow_list= [ 5, 1.5,   1.5,  1.5,  1.35,   0, 5, 0]
dryFlow_list= [ 0, 1,     1.55, 1.7,  2.0,    5, 0, 5]
BWXLE_bar.alignSamples()
for ii in range(8):    
    BWXLE_bar.setDryFlow(dryFlow_list[ii])
    BWXLE_bar.setWetFlow(wetFlow_list[ii])
    time.sleep(1200)
    BWXLE_bar.measureSamples()
    
#humdity_list=[100,80,60,40,20,0,100,0]
wetFlow_list= [ 5, 1.5,   1.5,  1.5,  1.35,   0, 5, 0]
dryFlow_list= [ 0, 1,     1.55, 1.7,  2.0,    5, 0, 5]
PAonPS35_bar.alignSamples()
for ii in range(8):    
    PAonPS35_bar.setDryFlow(dryFlow_list[ii])
    PAonPS35_bar.setWetFlow(wetFlow_list[ii])
    time.sleep(1200)
    PAonPS35_bar.measureSamples()
    

#humdity_list=[100,50]
wetFlow_list= [ 5, 1.5]
dryFlow_list= [ 0, 1.65]

#set RH=100%, align and scan
kinetics_bar.setDryFlow(dryFlow_list[0])
kinetics_bar.setWetFlow(wetFlow_list[0])
kinetics_bar.alignSamples()
kinetics_bar.measureSamples(exposure_time=15)  

#set RH=50%, align and scan
kinetics_bar.setDryFlow(dryFlow_list[1])
kinetics_bar.setWetFlow(wetFlow_list[1])

for align_ii in range(10):
    kinetics_bar.alignSamplesQuick() #quick_align samples in every ~24min
    for ii in range(8):  #1min for data collection + 2min for wait = 3min for each measurement. 3*8=24min for alignment
        #sleep for 2 min
        kinetics_bar.measureSamples(exposure_time=15)  
        time.sleep(2*60)

    
"""

"""

humidity list


humidity,    wet,     dry         2nd day, recheck
48           3         3

71           3         2.5                      
80           3         2.4/2.35        ###        

56           3         2.8  
60           3         2.7              ### 

43           3         3.3              
41           3         3.4              
40           3         3.45/3.6       ### 

20       2.5/2.4       3             ### 

humidity_list=[100, 80, 60,   40,  20, 0,100,0]
wetFlow_list= [ 5,  3,   3,   3,  2.5, 0, 5, 0]
dryFlow_list= [ 0, 2.35, 2.7, 3.6, 3,  5, 0, 5]
hol.alignSamples()
for ii in range(8):    
    hol1.setDryFlow(dryFlow_list[ii])
    hol1.setWetFlow(wetFlow_list[ii])
    post_to_slack('set to humidity  {}'.format(humidity_list[ii]))
    time.sleep(30*60)
    post_to_slack('start the measurement at humidity {}'.format(hol1.humidity(verbosity=0)))
    hol1.measureSamples()
"""
"""


humidity_list=[100,  75,   50,  25,   0,   100,0]
wetFlow_list= [ 5,   3,    3,   2.5,  0,    5, 0]
dryFlow_list= [ 0,  2.47,  3,    3,   5,    0, 5]
hol.alignSamples()
for ii in range(8):    
    hol1.setDryFlow(dryFlow_list[ii])
    hol1.setWetFlow(wetFlow_list[ii])
    post_to_slack('set to humidity  {}'.format(humidity_list[ii]))
    time.sleep(30*60)
    post_to_slack('start the measurement at humidity {}'.format(hol1.humidity(verbosity=0)))
    hol1.measureSamples()"""
