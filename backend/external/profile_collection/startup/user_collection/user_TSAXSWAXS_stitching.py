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

### DEFINE YOUR PARENT DATA FOLDER HERE


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

        # self.naming_scheme = ['name', 'extra', 'clock', 'temperature', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'y', 'th', 'clock', 'exposure_time']
        self.naming_scheme = ["name", "extra", "x", "y", "exposure_time"]
        # self.naming_scheme = ['name', 'extra', 'clock', 'temperature', 'exposure_time']

        self._axes["y"].origin = 10
        self._axes["th"].origin = 1.1

        self.md["exposure_time"] = 1
        # self.SAXS_time = 10
        # self.WAXS_time = 10
        self.SAXS_time = 5
        self.WAXS_time = 20

        # self.incident_angles_default = [0.08, 0.10, 0.12, 0.15, 0.20]
        # self.incident_angles_default = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        # self.incident_angles_default = [0.08, 0.12, 0.14, 0.20, 0.26, 0.32] #for 17kev/15kev
        # self.incident_angles_default = [0.10, 0.16, 0.20] #for 17kev/15kev
        # self.incident_angles_default = [0.08, 0.12, 0.15, 0.18, 0.20] #for 10kev
        # self.incident_angles_default = [0.08, 0.12, 0.15, 0.18] #for 10kev LJR
        # self.incident_angles_default = [0.12, 0.16, 0.20, 0.24] #for 10kev, Perovskites
        self.incident_angles_default = [
            0.12,
            0.16,
            0.20,
            0.24,
        ]  # for 10kev, Perovskites
        # self.incident_angles_default = [0.02, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1, 0.12, 0.15]
        # self.incident_angles_default = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15]
        # self.incident_angles_default = [0.0]

        self.x_pos_default = [-1, 0, 1]

        self.total_flow = 20
        self.wetflow_default = self.total_flow * np.arange(0.1, 0.51, 0.1)
        self.wetwait_default = [1200, 1200, 1200, 1200, 1200]

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

    def doTemperatures_series(
        self,
        exposure_time=15,
        output_channel="3",
        T1=160,
        T2=145,
        wait_time=60,
        wait_time2=40,
        temperature_tolerance=1,
        temperature_probe="temperature_C",
        verbosity=3,
    ):
        # increasing
        T_ramp_set(5)
        for temperature in self.T_series:
            self.setTemperature(temperature, output_channel=output_channel, verbosity=verbosity)

            # Wait until we reach the temperature
            # while abs(self.temperature(verbosity=0) - temperature)>temperature_tolerance:
            while (
                abs(self.temperature(temperature_probe=temperature_probe, verbosity=0) - temperature)
                > temperature_tolerance
            ):
                if verbosity >= 3:
                    print(
                        "  setpoint = {:.3f}°C, Temperature = {:.3f}°C          \r".format(
                            self.temperature_setpoint() - 273.15,
                            self.temperature(temperature_probe=temperature_probe, verbosity=0),
                        ),
                        end="",
                    )
                time.sleep(poling_period)

            # Allow for additional equilibration at this temperature
            if wait_time is not None:
                time.sleep(wait_time)

            self.measure(exposure_time=exposure_time)

        # First cooling to T1
        self.setTemperature(T1, output_channel=output_channel, verbosity=verbosity)

        # Wait until we reach the temperature
        # while abs(self.temperature(verbosity=0) - temperature)>temperature_tolerance:
        while abs(self.temperature(temperature_probe=temperature_probe, verbosity=0) - T1) > temperature_tolerance:
            if verbosity >= 3:
                print(
                    "  setpoint = {:.3f}°C, Temperature = {:.3f}°C          \r".format(
                        self.temperature_setpoint() - 273.15,
                        self.temperature(temperature_probe=temperature_probe, verbosity=0),
                    ),
                    end="",
                )
            time.sleep(poling_period)

        # Allow for additional equilibration at this temperature
        if wait_time is not None:
            time.sleep(wait_time)

        self.measure(exposure_time=exposure_time)

        # 2nd cooling to T2
        T_ramp_set(2)
        self.setTemperature(T2, output_channel=output_channel, verbosity=verbosity)

        # Wait until we reach the temperature
        # while abs(self.temperature(verbosity=0) - temperature)>temperature_tolerance:
        while abs(self.temperature(temperature_probe=temperature_probe, verbosity=0) - T2) > temperature_tolerance:
            if verbosity >= 3:
                print(
                    "  setpoint = {:.3f}°C, Temperature = {:.3f}°C          \r".format(
                        self.temperature_setpoint() - 273.15,
                        self.temperature(temperature_probe=temperature_probe, verbosity=0),
                    ),
                    end="",
                )
            time.sleep(poling_period)

        # Allow for additional equilibration at this temperature
        if wait_time is not None:
            time.sleep(wait_time)

        self.measure(exposure_time=exposure_time)

        # 3rd cooling process
        T_ramp_set(Tramp)
        self.setTemperature(T3, output_channel=output_channel, verbosity=verbosity)

        while (
            abs(self.temperature(temperature_probe=temperature_probe, verbosity=0) - (T3 - 1))
            > temperature_tolerance
        ):
            self.measure(exposure_time=exposure_time)
            time.sleep(wait_time2)

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

            swaxs_on()
            self.measureIncidentAngles_Stitch(incident_angles, exposure_time=self.SAXS_time, tiling="ygaps", **md)
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

    def do_SAXS(self, step=0, align_step=0, **md):
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

            saxs_on_det()
            self.measureIncidentAngles_Stitch(incident_angles, exposure_time=self.SAXS_time, tiling="ygaps", **md)

            # if self.exposure_time_SAXS==None:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.SAXS_time, tiling=self.tiling, **md)
            # else:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.exposure_time_SAXS, tiling=self.tiling, **md)

            self.thabs(0.0)

    def align_y(self, step=0, reflection_angle=0.08, verbosity=3):
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
            self.measureIncidentAngles_Stitch(incident_angles, exposure_time=self.WAXS_time, tiling="ygaps", **md)

            # if self.exposure_time_MAXS==None:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.MAXS_time, tiling=self.tiling, **md)
            # else:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.exposure_time_MAXS, tiling=self.tiling, **md)

            self.thabs(0.0)

    def do_WAXS(self, step=0, align_step=0, reflection_angle=0.12, tiling="ygaps", **md):
        if step <= 1:
            saxs_on()
            get_beamline().modeAlignment()

        if step <= 2:
            self.xo()  # goto origin

        if step <= 4:
            self.yo()
            self.tho()

        if step <= 5:
            self.align(step=align_step, reflection_angle=reflection_angle)
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
            self.measureIncidentAngles_Stitch(incident_angles, exposure_time=self.WAXS_time, tiling=tiling, **md)

            # if self.exposure_time_MAXS==None:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.MAXS_time, tiling=self.tiling, **md)
            # else:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.exposure_time_MAXS, tiling=self.tiling, **md)
            self.thabs(0.0)

    def alignQuick(self, align_step=8, reflection_angle=0.12, verbosity=3):
        saxs_on()
        get_beamline().modeAlignment()
        # self.yo()
        self.tho()
        # beam.on()

        if verbosity >= 4:
            print("    align: reflected beam")
        get_beamline().setReflectedBeamROI(total_angle=reflection_angle * 2.0)

        self.thabs(reflection_angle)

        result = fit_scan(sth, 0.2, 41, fit="max")
        # result = fit_scan(sth, 0.2, 81, fit='max') #it's useful for alignment of SmarAct stage
        sth_target = result.values["x_max"] - reflection_angle
        if result.values["y_max"] > 10:
            th_target = self._axes["th"].motor_to_cur(sth_target)
            self.thsetOrigin(th_target)

        result = fit_scan(smy, 0.3, 21, fit="max")
        smy_target = result.values["x_max"]
        if result.values["y_max"] > 10:
            smy_target = self._axes["y"].motor_to_cur(smy_target)
            self.ysetOrigin(smy_target)

        get_beamline().modeMeasurement()

    def align_crazy(
        self,
        reflection_angle=0.12,
        ROI_size=[10, 180],
        th_range=0.3,
        int_threshold=10,
        verbosity=3,
    ):
        # def ROI3 in 160pixels with the center located at reflection beam
        get_beamline().setReflectedBeamROI(total_angle=reflection_angle * 2, size=ROI_size)  # set ROI3

        rel_th = 1
        ct = 0
        cycle = 0
        self.thabs(reflection_angle)
        while abs(rel_th) > 0.005 and ct < 5:
            self.snap(0.5)
            refl_ypos = caget("XF:11BMB-ES{Det:PIL2M}:Stats3:MaxY_RBV")
            refl_ypos_center = caget("XF:11BMB-ES{Det:PIL2M}:Stats3:CentroidY_RBV")

            while abs(refl_ypos - refl_ypos_center) > 10:
                print("The center does not match to the max")
                self.thr(0.05)
                self.snap(0.5)
                refl_ypos = caget("XF:11BMB-ES{Det:PIL2M}:Stats3:MaxY_RBV")
                refl_ypos_center = caget("XF:11BMB-ES{Det:PIL2M}:Stats3:CentroidY_RBV")

            # refl_ypos = caget('XF:11BMB-ES{Det:PIL2M}:Stats3:SigmaY_RBV')
            int_max = caget("XF:11BMB-ES{Det:PIL2M}:Stats3:MaxValue_RBV")

            while int_max < int_threshold and cycle <= np.round(th_range / 0.1):
                self.thabs(reflection_angle + th_range - cycle * 0.1)
                cycle += 1
                self.snap(0.5)
                refl_ypos = caget("XF:11BMB-ES{Det:PIL2M}:Stats3:MaxY_RBV")
                int_max = caget("XF:11BMB-ES{Det:PIL2M}:Stats3:MaxValue_RBV")
                # return False
            cycle = 0

            if int_max < int_threshold:
                return False

            rel_ypos = refl_ypos - ROI_size[1] / 2

            rel_th = rel_ypos / get_beamline().SAXS.distance / 1000 * 0.172 / np.pi * 180 / 2

            print("The th offset is {}".format(rel_th))
            self.thr(rel_th)

            ct += 1
        if abs(rel_th) > 0.005:
            print("Fast alignment failed after {} times of aligment".format(ct))

            return False
        else:
            self.thset(reflection_angle)

            return True

    def align_custom(self, step=0, reflection_angle=0.08, verbosity=3):
        """Align the sample with respect to the beam. GISAXS alignment involves
        vertical translation to the beam center, and rocking theta to get the
        sample plane parralel to the beam. Finally, the angle is re-optimized
        in reflection mode.

        The 'step' argument can optionally be given to jump to a particular
        step in the sequence."""

        if verbosity >= 3:
            print("  Aligning {}".format(self.name))

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

            if "beam_intensity_expected" in RE.md:
                if value < RE.md["beam_intensity_expected"] * 0.75:
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
            ##time.sleep(2)
            fit_scan(sth, 1.5, 21, fit="max")
            ##time.sleep(2)

        # if step<=5:
        #    #fit_scan(smy, 0.6, 17, fit='sigmoid_r')
        #    fit_edge(smy, 0.6, 17)
        #    fit_scan(sth, 1.2, 21, fit='max')

        if step <= 8:
            # fit_scan(smy, 0.3, 21, fit='sigmoid_r')

            fit_edge(smy, 0.6, 21)
            # fit_edge(smy, 0.3, 16)
            # time.sleep(2)
            # fit_edge(smy, 0.4, 21)
            # fit_scan(sth, 0.8, 21, fit='COM')
            # time.sleep(2)
            # self.setOrigin(['y', 'th'])
            self.setOrigin(["y"])

        if step <= 9 and reflection_angle is not None:
            # Final alignment using reflected beam
            if verbosity >= 4:
                print("    align: reflected beam")

            if self.align_crazy() == False:
                get_beamline().setReflectedBeamROI(total_angle=reflection_angle * 2.0)
                # get_beamline().setReflectedBeamROI(total_angle=reflection_angle*2.0, size=[12,2])

                self.thabs(reflection_angle)

                result = fit_scan(sth, 0.4, 41, fit="max")
                # result = fit_scan(sth, 0.2, 81, fit='max') #it's useful for alignment of SmarAct stage
                sth_target = result.values["x_max"] - reflection_angle

                if result.values["y_max"] > 50:
                    th_target = self._axes["th"].motor_to_cur(sth_target)
                    self.thsetOrigin(th_target)

            get_beamline().setReflectedBeamROI(total_angle=reflection_angle * 2.0)
            self.thabs(reflection_angle)
            fit_scan(smy, 0.3, 16, fit="max")
            self.setOrigin(["y"])

        if step <= 10:
            self.thabs(0.0)
            beam.off()


class TransOffCenteredCustom(OffCenteredHoder):
    def __init__(self, name="OffCenteredHoderCustom", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)
        self._axes["x"].origin = -17.2  # -0.3
        self._axes["y"].origin = 10.76  # -0.1
        ##self._axes['x'].origin = -59.3
        ###position for calibration
        # self._axes['x'].origin = -71.5+.2
        # self._axes['y'].origin = 10
        # self._axes['th'].origin = 1.1

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

    # def doSamples(self, verbosity=3):

    ##maxs_on()
    # for sample in self.getSamples():
    # if verbosity>=3:
    # print('Doing sample {}...'.format(sample.name))
    # if sample.detector=='SAXS' or sample.detector=='BOTH':
    # sample.do_SAXS()

    # for sample in self.getSamples():
    # if verbosity>=3:
    # print('Doing sample {}...'.format(sample.name))
    # if sample.detector=='BOTH':
    # sample.do_WAXS_only()

    # for sample in self.getSamples():
    # if verbosity>=3:
    # print('Doing sample {}...'.format(sample.name))
    # if sample.detector=='WAXS':
    # sample.do_WAXS()

    def doSamples(
        self,
        sequence="Outer",
        exposure_WAXS_time=10,
        exposure_SAXS_time=60,
        verbosity=3,
    ):
        if sequence == "Outer":
            step = 0
        elif sequence == "Inner":
            step = 5
        else:
            return print("Please define the first measurement: Outer or Inner")
        if step < 1:
            waxs_on_outer()
            for sample in self.getSamples():
                sample.gotoOrigin()
                sample.measure(exposure_WAXS_time, extra="outer-normal", tiling="ygaps")

            detselect(pilatus2M)
            for sample in self.getSamples():
                sample.gotoOrigin()
                sample.measure(exposure_SAXS_time, tiling="ygaps")

        if step < 10:
            waxs_on_inner()
            for sample in self.getSamples():
                sample.gotoOrigin()
                sample.measure(exposure_WAXS_time, extra="inner-normal", tiling="ygaps")

        if step > 1:
            waxs_on_outer()
            for sample in self.getSamples():
                sample.gotoOrigin()
                sample.measure(exposure_WAXS_time, extra="outer-normal", tiling="ygaps")

            detselect(pilatus2M)
            for sample in self.getSamples():
                sample.gotoOrigin()
                sample.measure(exposure_SAXS_time, tiling="ygaps")

    def doSamples_test(self, exposure_WAXS_time=10, exposure_SAXS_time=60, verbosity=3):
        for sample in self.getSamples():
            sample.gotoOrigin()
            sample.measure(1)


def saxs_on():
    detselect(pilatus2M)
    # WAXSx.move(-195)
    # WAXSy.move(24)


def saxs_on_det():
    detselect(pilatus2M)
    WAXSx.move(-200)
    WAXSy.move(30)


def waxs_on():
    detselect(pilatus800)
    WAXSx.move(-195)
    WAXSy.move(18)


def waxs_on_outer():  # for inner-outer stitching
    detselect(pilatus800)
    WAXSx.move(-245)
    WAXSy.move(50)


def waxs_on_inner():  # for inner-outer stitching
    detselect(pilatus800)
    WAXSx.move(-210)
    WAXSy.move(18)


# cms.SAXS.setCalibration([737, 1089], 2.8, [-65, -73])  #20190320,
# cms.SAXS.setCalibration([737, 1089], 5.05, [-65, -73])  #20190320
# cms.SAXS.setCalibration([755, 1072], 5.05, [-65, -73])  #20191020
# cms.SAXS.setCalibration([748, 1680-590], 2.01, [-64, -73]) # 13.5 keV
cms.SAXS.setCalibration([761, 1680 - 606], 5.0, [-65, -73])

RE.md["experiment_alias_directory"] = "/nsls2/data/cms/legacy/xf11bm/data/2022_1/LZhu/"

if True:
    cali = CapillaryHolder(base=stg)
    # hol = CapillaryHolderCustom(base=stg)

    cali.addSampleSlot(Sample("FL_screen"), 5.0)
    cali.addSampleSlot(Sample("AgBH_cali_5m_17kev"), 8.0)
    cali.addSampleSlot(Sample("Empty"), 11.0)

if True:
    # Example of a multi-sample holder

    md = {
        "owner": "L. Zhu",
        "series": "various",
    }

    # edge on
    hol1 = CapillaryHolder(base=stg)
    hol1.addGaragePosition(1, 1)
    hol1.name = "hol1"
    hol1.addSampleSlot(Sample("Guanchun_flaton_PEEK-COOH"), 1)
    hol1.addSampleSlot(Sample("Guanchun_flaton_PEEK-SO2"), 2)
    hol1.addSampleSlot(Sample("Guanchun_flaton_PEEK-CN"), 3)
    hol1.addSampleSlot(Sample("Guanchun_flaton_PAEN-COOH"), 4)
    hol1.addSampleSlot(Sample("Guanchun_flaton_PAEN-SO2"), 5)
    hol1.addSampleSlot(Sample("Guanchun_flaton_PAEN-CN"), 6)
    hol1.addSampleSlot(Sample("Guanchun_flaton_PAES-COOH"), 7)
    hol1.addSampleSlot(Sample("Guanchun_flaton_PAES-SO2"), 8)
    hol1.addSampleSlot(Sample("Guanchun_flaton_PAES-CN"), 9)
    hol1.addSampleSlot(Sample("Guanchun_flaton_Nate PPS UR"), 10)
    hol1.addSampleSlot(Sample("Guanchun_flaton_Nate PPS R"), 11)

    hol2 = TransOffCenteredCustom(base=stg)
    hol2.addGaragePosition(1, 2)
    hol2.name = "hol2"
    hol2.addSampleSlot(Sample("Guanchun_edgeon_Michael1"), 0)
    hol2.addSampleSlot(Sample("Guanchun_edgeon_Michael2"), 1)
    hol2.addSampleSlot(Sample("Guanchun_edgeon_Michael3"), 2)
    hol2.addSampleSlot(Sample("Guanchun_edgeon_64L"), 3)
    hol2.addSampleSlot(Sample("Guanchun_edgeon_128L"), 4)
    hol2.addSampleSlot(Sample("Guanchun_edgeon_256L"), 5)
    hol2.addSampleSlot(Sample("Guanchun_edgeon_512L"), 6)
    hol2.addSampleSlot(Sample("Guanchun_edgeon_1024L"), 7)
    hol2.addSampleSlot(Sample("Guanchun_edgeon_2048L"), 8)
    hol2.addSampleSlot(Sample("Guanchun_edgeon_64L_QA1"), 9)
    hol2.addSampleSlot(Sample("Guanchun_edgeon_Nate_R_PPS_edgeon"), 10)

    hol2.addSampleSlot(Sample("Guanchun_edgeon_Michael_2h_anneal_D1"), 12)
    hol2.addSampleSlot(Sample("Guanchun_edgeon_Michael_2h_anneal_D2"), 13)

    # hol2 = TransOffCenteredCustom(base=stg)
    # hol2.addGaragePosition(1,2)
    # hol2.name = 'hol2'
    # hol2.addSampleSlot( Sample('Guanchun_8020 ES'),1)
    # hol2.addSampleSlot( Sample('Guanchun_8020 ESP'),2)
    # hol2.addSampleSlot( Sample('Guanchun_8020 ESA'),3)
    # hol2.addSampleSlot( Sample('Guanchun_8020 ESAP'),4)

    # hol5 = GIBar_Custom(base=stg)
    # hol5.addGaragePosition(2,2)
    # hol5.name = 'hol5'
    # hol5.addSampleSlot( Sample('Zhiwen_TerpolymerQ'),0)
    # hol5.addSampleSlot( Sample('Zhiwen_TerpolymerQA60'),1)
    # hol5.addSampleSlot( Sample('Zhiwen_TerpolymerQA80'),2)
    # hol5.addSampleSlot( Sample('Zhiwen_TerpolymerQS'),3)
    # hol5.addSampleSlot( Sample('Zhiwen_TerpolymerQSA60'),4)
    # hol5.addSampleSlot( Sample('Zhiwen_TerpolymerQSA80'),5)
    # hol5.addSampleSlot( Sample('Zhiwen_TerpolymerQSA6048h'),6)

    que = Queue(base=stg)
    # que.addHolderIntoQueue(hol1, [1, 2], 1)
    que.addHolderIntoQueue(hol2, [1, 2], 2)
    # que.addHolderIntoQueue(hol3, [1, 3], 3)
    # que.addHolderIntoQueue(hol4, [2, 1], 4)
    # que.addHolderIntoQueue(hol5, [2, 2], 5)
    # que.addHolderIntoQueue(hol6, [2, 3], 6)
    ##que.addHolderIntoQueue(hol7, [3, 1], 7)
    # que.addHolderIntoQueue(hol8, [3, 2], 8)
    # que.addHolderIntoQueue(hol9, [3, 3], 9)
    # que.addHolderIntoQueue(hol10, [4, 1], 10)
    # que.addHolderIntoQueue(hol11, [1, 2], 2)
    # que.addHolderIntoQueue(hol10, [4, 1], 10)
    # que.addHolderIntoQueue(hol11, [4, 2], 11)
    # que.addHolder(hol3, [2, 3])
    # que.addHolderIntoQueue(cali, [4, 1], 10)
    que.setSequence()
    que.checkStatus(verbosity=5)


# %run -i /GPFS/xf11bm/data/2018_1/beamline/user.py

# robot.listGarage()


"""
New alignment procedure for holder:

1. move to the sample close to the center (cali_sample)
2. align the cali_sample and set the other samples position (y and th) as cali_sample
3. align the other samples one by one
   --1>> y scan in narrow range (0.6 in 21 steps) 
   --2>> 
   --3>> if 4 is not working, use reflection to align th. 
   and second, 0.3 in 16 steps)
4. 



"""

"""
2021_2

round bs
In [1272]: wbs()
bsx = -14.955786
bsy = -9.999943
bsphi = -16.000004999999987

rod bs
In [1286]: wbs()
bsx = -14.000031
bsy = 17.000000999999997
bsphi = -223.4




"""
