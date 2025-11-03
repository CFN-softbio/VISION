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
from bluesky.suspenders import SuspendFloor, SuspendCeil

# RE.clear_suspender()

ring_current = EpicsSignal("SR:OPS-BI{DCCT:1}I:Real-I")
sus = SuspendFloor(ring_current, 100, resume_thresh=101)
RE.install_suspender(sus)

# absorber_pos = EpicsSignal( 'XF:11BMB-ES{SM:1-Ax:ArmR}Mtr.RBV')
# sus_abs_low = SuspendFloor(absorber_pos, -56, resume_thresh=-55)
# sus_abs_hi = SuspendCeil(absorber_pos, -54, resume_thresh=-55)
# RE.install_suspender(sus_abs_low)
# RE.install_suspender(sus_abs_hi)
# RE.clear_suspender()


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


INTENSITY_EXPECTED_050 = 51034.0
INTENSITY_EXPECTED_025 = INTENSITY_EXPECTED_050 * 0.5


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


# class Sample(SampleTSAXS):
class Sample(SampleXR_WAXS):
    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)
        # self.naming_scheme = ['name', 'extra', 'clock', 'temperature', 'exposure_time']
        self.naming_scheme = ["name", "extra", "th", "exposure_time"]

        # self.md['exposure_time'] = 20 #SAXS exposure time
        self.SAXS_time = 30
        self.WAXS_time = 10

        self.incident_angles_default = [0.08, 0.10, 0.12]
        # self.incident_angles_default = [0.10]
        # -116.0000 mm
        self.x_pos_default = [-1, 0, 1]

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

    def align(self, step=0, reflection_angle=0.08, verbosity=3):
        """Align the sample with respect to the beam. GISAXS alignment involves
        vertical translation to the beam center, and rocking theta to get the
        sample plane parralel to the beam. Finally, the angle is re-optimized
        in reflection mode.

        The 'step' argument can optionally be given to jump to a particular
        step in the sequence."""

        if verbosity >= 4:
            print("  Aligning {}".format(self.name))

        if step <= 0:
            # Prepare for alignment

            if RE.state != "idle":
                RE.abort()

            if get_beamline().current_mode != "alignment":
                if verbosity >= 2:
                    print(
                        "WARNING: Beamline is not in alignment mode (mode is '{}')".format(
                            get_beamline().current_mode
                        )
                    )
                # get_beamline().modeAlignment()

            get_beamline().setDirectBeamROI()

            beam.on()

        if step <= 2:
            if verbosity >= 4:
                print("    align: searching")

            # Estimate full-beam intensity
            value = None
            if True:
                # You can eliminate this, in which case RE.md['beam_intensity_expected'] is used by default
                self.yr(-2)
                # detector = gs.DETS[0]
                detector = get_beamline().detector[0]
                value_name = get_beamline().TABLE_COLS[0]
                RE(count([detector]))
                value = detector.read()[value_name]["value"]
                self.yr(+2)

            if "beam_intensity_expected" in RE.md and value < RE.md["beam_intensity_expected"] * 0.75:
                print(
                    "WARNING: Direct beam intensity ({}) lower than it should be ({})".format(
                        value, RE.md["beam_intensity_expected"]
                    )
                )

            # Find the step-edge
            self.ysearch(
                step_size=0.5,
                min_step=0.005,
                intensity=value,
                target=0.5,
                verbosity=verbosity,
                polarity=-1,
            )

            # Find the peak
            self.thsearch(step_size=0.4, min_step=0.01, target="max", verbosity=verbosity)

        if step <= 4:
            if verbosity >= 4:
                print("    align: fitting")

            fit_scan(smy, 1.2, 21, fit="HMi")
            # time.sleep(2)
            fit_scan(sth, 1.5, 21, fit="max")
            # time.sleep(2)

        # if step<=5:
        #    #fit_scan(smy, 0.6, 17, fit='sigmoid_r')
        #    fit_edge(smy, 0.6, 17)
        #    fit_scan(sth, 1.2, 21, fit='max')

        if step <= 8:
            fit_scan(smy, 0.6, 21, fit="sigmoid_r")
            time.sleep(2)
            fit_scan(sth, 0.8, 21, fit="COM")
            time.sleep(2)
            self.setOrigin(["y", "th"])

        if step <= 9 and reflection_angle is not None:
            # Final alignment using reflected beam
            if verbosity >= 4:
                print("    align: reflected beam")
            get_beamline().setReflectedBeamROI(total_angle=reflection_angle * 2.0)
            # get_beamline().setReflectedBeamROI(total_angle=reflection_angle*2.0, size=[12,2])

            self.thabs(reflection_angle)

            result = fit_scan(sth, 0.2, 41, fit="max")
            # result = fit_scan(sth, 0.2, 81, fit='max') #it's useful for alignment of SmarAct stage
            sth_target = result.values["x_max"] - reflection_angle

            if result.values["y_max"] > 50:
                th_target = self._axes["th"].motor_to_cur(sth_target)
                self.thsetOrigin(th_target)

            # fit_scan(smy, 0.2, 21, fit='max')
            self.setOrigin(["y"])

        if step <= 10:
            self.thabs(0.0)
            beam.off()

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

    def do(self, step=0, align_step=0, **md):
        if step <= 1:
            get_beamline().modeAlignment()
            saxs_on()

        if step <= 2:
            self.xo()  # goto origin

        if step <= 4:
            self.yo()
            self.tho()

        if step <= 5:
            # self.align(step=align_step, reflection_angle=0.12)
            # self.setOrigin(['y','th']) # This is done within align
            self.XR_align()

        # if step<=7:
        # self.xr(0.2)

        # if step<=8:
        # get_beamline().modeMeasurement()

        if step <= 10:
            self.XR_check_alignment(th_angle=1, roi_size=[10, 10])
            # self.XR_scan(theta_range=[0, .15], theta_delta=0.01, roi_size=[10,10], exposure_time=1)
            # self.XR_scan(theta_range=[0, .5], theta_delta=0.015, roi_size=[10,10], exposure_time=1)
            self.XR_scan(theta_range=[0, 3], theta_delta=0.04, roi_size=[10, 10], exposure_time=1)
            self.tho()
            # saxs_on()
            # self.measureIncidentAngles(self.incident_angles_default, exposure_time=self.SAXS_time , **md)
            # waxs_on()
            # self.measureIncidentAngles(self.incident_angles_default, exposure_time=self.WAXS_time ,tiling='ygaps', **md)
            # self.thabs(0.0)
            # waxs_on()
            # for detector in get_beamline().detector:
            # detector.setExposureTime(self.waxs_exposure_time)
            # self.measureIncidentAngles(self.incident_angles_default, **md)
            # self.thabs(0.0)

    def do_thscan(self, step=0, align_step=0, **md):
        if step <= 1:
            get_beamline().modeAlignment()
            saxs_on()

        if step <= 2:
            self.xo()  # goto origin

        if step <= 4:
            self.yo()
            self.tho()

        if step <= 5:
            # self.align(step=align_step, reflection_angle=0.12)
            # self.setOrigin(['y','th']) # This is done within align
            self.XR_align()
        # if step<=7:
        # self.xr(0.2)

        if step <= 8:
            get_beamline().modeMeasurement()

        if step <= 10:
            self.XR_check_alignment(th_angle=1, roi_size=[10, 3])
            self.th2thscan(theta_range=[0.25, 6], theta_delta=0.05, roi_size=[10, 3])
            self.tho()
            # waxs_on()
            # self.measureIncidentAngles(self.incident_angles_default,  exposure_time=self.WAXS_time, **md)


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

        self._axes["y"].origin = 8

        self.temp_series_cooling1 = np.array([160, 120, 80, 40])
        self.temp_series_heating1 = np.array([80, 120, 160, 200])

        temp_series1 = np.arange(40, 65 + 0.1, 5)
        temp_series2 = np.arange(70, 140 + 0.1, 10)
        temp_series3 = np.arange(150, 200 + 0.1, 5)

        temp_series_cooling2 = np.arange(200, 30 - 0.1, -10)

        self.temp_series = np.array([200, 50])
        # self.temp_series = np.array([ 170, 200])

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

    def doTemperature_series_heat(
        self,
        step=0,
        exposure_time_SAXS=30.0,
        exposure_time_WAXS=120.0,
        temperature_tolerance=0.4,
        wait_time=200,
        verbosity=3,
        x_step=0.20,
        **md,
    ):
        # will do heat only
        # TODO change beamstop
        #     check waxs_on and saxs_on

        for sample in self.getSamples():
            sample.incident_angles_default = [0.1]
            sample.naming_scheme = [
                "name",
                "extra",
                "temperature",
                "th",
                "x",
                "exposure_time",
            ]

        # run the Bar at RT
        if step < 1:
            self.doSamples()

        ##1st heating up to 200C directly
        # if step < 5:

        # self.setTemperature(200)
        # while abs(self.temperature(verbosity=0) - 200)>temperature_tolerance:
        # if verbosity>=3:
        # print('  setpoint = {:.3f}°C, Temperature = {:.3f}°C          \r'.format(self.temperature_setpoint()-273.15, self.temperature(verbosity=0)), end='')
        # time.sleep(2)
        # if wait_time is not None:
        # time.sleep(wait_time)

        # self.doSamples()

        # cooling and heating process
        if step < 10:
            for ct, temperature in np.ndenumerate(self.temp_series):
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

                self.doSamples()

        if step < 20:
            self.setTemperature(25)


cms.SAXS.setCalibration([780, 1680 - 605], 5.03, [-60, -73])  # 13.5 keV
# cms.WAXS.setCalibration([460, 1043-425], 0.371, [-194.5, 16.39])

cms.WAXS.setCalibration([460, 1043 - 426], 0.371, [-194.5, 14])

RE.md["experiment_group"] = "LLoo (Princeton)"
RE.md["experiment_alias_directory"] = "/GPFS/xf11bm/data/2019_2/LLoo2/"
RE.md["experiment_proposal_number"] = "303342"
RE.md["experiment_SAF_number"] = "304276"
RE.md["experiment_user"] = "Loo et al"
RE.md["experiment_type"] = "XR"  # TSAXS, GISAXS, GIWAXS, etc.
RE.md["experiment_project"] = "polymeric semiconductor thin films"


# this is the 2nd try to run only RT and 70deg to check the radiation damage.

print("\n\n\nReminders:")
print("    Define your detectors using, e.g.: detselect(pilatus2M)")
print("    Reload your user-specific script, e.g.: %run -i /GPFS/xf11bm/data/2017_2/user_group/user.py")
print("\n")


def wbs():
    print("bsx = {}".format(bsx.position))
    print("bsy = {}".format(bsy.position))
    print("bsphi = {}".format(bsphi.position))


def wsam():
    print("smx = {}".format(smx.position))
    print("smy = {}".format(smy.position))
    print("sth = {}".format(sth.position))


def wWAXS():
    print("WAXSx = {}".format(WAXSx.position))
    print("WAXSy = {}".format(WAXSy.position))
    print("WAXSz = {}".format(WAXSz.position))


def wSAXS():
    print("SAXSx = {}".format(SAXSx.position))
    print("SAXSy = {}".format(SAXSy.position))


def waxs_on():
    detselect(pilatus800)
    WAXSx.move(-194.5)
    WAXSy.move(21)

    # WAXSx.move(-250)
    # WAXSy.move(18)


def saxs_on():
    detselect(pilatus2M)
    # WAXSx.move(-193)
    WAXSy.move(24)


def cooling(wait_time=5):
    sam1.gotoOrigin()
    sam1.measureIncidentAngle(0.1, exposure_time=1)
    sam2.gotoOrigin()
    sam2.measureIncidentAngle(0.1, exposure_time=1)
    time.sleep(wait_time)


# This series have longer sample-detector distance.
# WAXSy = 0
if True:
    # Example of a multi-sample holder

    md = {
        "owner": "L.Loo",
        "series": "various",
    }

    hol = GIBarCustom(base=stg)
    # hol = GIBar(base=stg)
    cb = 0

    # hol.addSampleSlotPosition( Sample('T_cali', **md), 1, 62+cb )
    # hol.addSampleSlotPosition( Sample('YX_XR_C8_C12_T1', **md), 2, 7+cb )
    # hol.addSampleSlotPosition( Sample('YX_XR_C8_C12_T2', **md), 3, 22+cb )
    # hol.addSampleSlotPosition( Sample('YX_XR_C8_C12_T3', **md), 4, 39+cb )
    # hol.addSampleSlotPosition( Sample('YX_XR_C8_C12_T4', **md), 5, 48+7+cb )
    # hol.addSampleSlotPosition( Sample('YX_XR_C8_C12_T5', **md), 6, 11+cb )
    # hol.addSampleSlotPosition( Sample('YX_XR_C8_C12_T6', **md), 6, 20+cb )
    # hol.addSampleSlotPosition( Sample('YX_XR_C8_C12_T7', **md), 7, 39-8+cb )
    # hol.addSampleSlotPosition( Sample('YX_XR_C8_C12_T8', **md), 6, 16+13+cb )
    # hol.addSampleSlotPosition( Sample('YX_XR_C8_C12_T9', **md), 7, 39-8+18+cb )
    hol.addSampleSlotPosition(Sample("YX-C8-SC", **md), 1, 4 + 2)
    hol.addSampleSlotPosition(Sample("YX-C12-SC", **md), 2, 15 + 12)
    hol.addSampleSlotPosition(Sample("YX-C8-control", **md), 3, 52)
    hol.addSampleSlotPosition(Sample("YX-C12-control", **md), 4, 66)
    hol.addSampleSlotPosition(Sample("YX-C8_C12-1C-OnSi", **md), 5, 82)

if False:
    # Example of a multi-sample holder

    md = {
        "owner": "N. Jiang",
        "series": "various",
    }

    # hol = GIBar(base=stg)
    hol = GIBarCustom(base=stg)
    cb = -20 + 2

    hol.addSampleSlotPosition(Sample("19B_oct100_HSi", **md), 1, 17 + cb)
    hol.addSampleSlotPosition(Sample("20E_eh100_HSi", **md), 2, 28 + cb)
    hol.addSampleSlotPosition(Sample("lM100O10_Si", **md), 3, 39 + cb)
    hol.addSampleSlotPosition(Sample("3_lDe", **md), 4, 49 + cb)
    hol.addSampleSlotPosition(Sample("lOct50DL", **md), 5, 60 + cb)
    hol.addSampleSlotPosition(Sample("cOct50DL", **md), 6, 70 + cb)
    hol.addSampleSlotPosition(Sample("lM100D10_THF", **md), 7, 78 + cb)
    hol.addSampleSlotPosition(Sample("cM200d20_MeOH", **md), 8, 88 + cb)
    hol.addSampleSlotPosition(Sample("13_2D_lDe200", **md), 9, 96 + cb)
    hol.addSampleSlotPosition(Sample("13_2B_cDe200", **md), 10, 107 + cb)
    hol.addSampleSlotPosition(Sample("23D", **md), 11, 118 + cb)
    hol.addSampleSlotPosition(Sample("M150D50_1wtSi", **md), 12, 129 + cb)

if False:
    md = {
        "owner": "N. Jiang",
        "series": "various",
    }
    RE.md["experiment_type"] = "TSAXS"

    hol = CapillaryHolder(base=stg)
    hol.addSampleSlot(Sample("FL", **md), 5)
    hol.addSampleSlot(Sample("AgBH_5m_cali_13.5kev", **md), 8)
    hol.addSampleSlot(Sample("Empty", **md), 11)
    # hol.addSampleSlot( Sample('run17_SAXS_ThermalRamp_kl04-94-1', **md), 7)


if False:
    hol = CapillaryHolder(base=stg)

    # hol.addSampleSlot( Sample('2nd_80pW12-500mMLiCl_a'), 1.0 )
    # hol.addSampleSlot( Sample('2nd_120pW12-500mMLiCl_a'), 2.0 )
    # hol.addSampleSlot( Sample('2nd_160pW12-500mMLiCl_a'), 3.0 )
    # hol.addSampleSlot( Sample('2nd_200pW12-500mMLiCl_a'), 4.0 )
    hol.addSampleSlot(Sample("FS"), 5.0)
    # hol.addSampleSlot( Sample('2nd_160pW12-300mMLiCl_a'), 6.0 )
    # hol.addSampleSlot( Sample('2nd_160pW12-400mMLiCl_a'), 7.0 )
    hol.addSampleSlot(Sample("AgBH_cali_5m_13.5keV"), 8.0)
    # hol.addSampleSlot( Sample('2nd_80pW12-500mMLiCl_b'), 9.0 )
    # hol.addSampleSlot( Sample('2nd_120pW12-500mMLiCl_b'), 10.0 )
    hol.addSampleSlot(Sample("Empty"), 11.0)
    # hol.addSampleSlot( Sample('2nd_200pW12-500mMLiCl_b'), 12.0 )
    # hol.addSampleSlot( Sample('2nd_220pW12-500mMLiCl_b'), 13.0 )
    # hol.addSampleSlot( Sample('2nd_160pW12-300mMLiCl_b'), 14.0 )
    # hol.addSampleSlot( Sample('2nd_160pW12-400mMLiCl_b'), 15.0 )
    # sam = hol.getSample(1)


# %run -i /GPFS/xf11bm/data/2018_1/beamline/user.py

# robot.listGarage()

"""

#XR protocol
sam=SampleXR_WAXS('test')
sam.XR_align()
sam.XR_check_alignment(th_angle=1, roi_size=[10, 10]) 
sam.XR_scan(theta_range=[0, 3], theta_delta=0.02, roi_size=[10,10], exposure_time=1)    



for sample in hol.getSamples():
    sample.gotoOrigin()
    sample.measureIncidentAngles(exposure_time=10, tiling='ygaps')
"""
