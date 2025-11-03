"""
TODO:

1. Save Linkam parameters into the metadata  
   -almost done. The bsx needs to be added in. 
2. Integrate the Linkam tensile stage and create a standard method for modes (step, force, velocity)
   -seperate in different modes. 
   -setp mode is done. 
3. Multiple threads for Linkam and Detectors. (if the detector is running in burst mode.)
   -it requires 1s delay for each stage. 
    #TODO: ask Tom about the setting updates. 
"""


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

# ring_current = EpicsSignal('SR:OPS-BI{DCCT:1}I:Real-I')
# sus = SuspendFloor(ring_current, 100, resume_thresh=400, sleep=600)
# RE.install_suspender(sus)


RE.md["experiment_alias_directory"] = "/nsls2/data/cms/legacy/xf11bm/data/2023_1/beamline"
# cms.SAXS.setCalibration([737, 1680-582], 3, [-65, -73]) #3m, 13.5kev
# cms.SAXS.setCalibration([738, 1097], 3.0, [-65, -73])   #3m,13.5kev
# cms.SAXS.setCalibration([738, 1680-590], 2, [-65, -73])
cms.SAXS.setCalibration([738, 1096], 3.0, [-65, -73])  # 20201021, 13.5 keV


def swaxs_on():
    detselect([pilatus2M, pilatus800])
    WAXSx.move(-227)
    WAXSy.move(27)


#### FOR LAST SAMPLE
# def swaxs_on():
#     detselect([pilatus2M, pilatus800])
#     WAXSx.move(-227)
#     WAXSy.move(32.56)


def saxs_on():
    detselect(pilatus2M)
    WAXSx.move(-227)
    WAXSy.move(27)


def waxs_on():
    detselect(pilatus800)
    WAXSx.move(-227)
    WAXSy.move(27)


# cms.setDirectBeamROI()


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


class Sample(SampleGISAXS_Generic):
    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)
        self.naming_scheme = ["name", "extra", "th", "exposure_time"]
        self.naming_scheme = ["name", "extra", "clock", "temperature", "exposure_time"]

    def setLinkamRate(self, rate):
        caput("XF:11BM-ES:{LINKAM}:RAMPRATE:SET", rate)
        return rate

    def setLinkamTemperature(self, temperature):
        caput("XF:11BM-ES:{LINKAM}:SETPOINT:SET", temperature)
        return temperature

    def setLinkamOn(self):
        caput("XF:11BM-ES:{LINKAM}:STARTHEAT", 1)
        return 1

    def setLinkamOff(self):
        caput("XF:11BM-ES:{LINKAM}:STARTHEAT", 0)
        return 0

    def linkamStatus(self):
        return caget("XF:11BM-ES:{LINKAM}:STATUS")

    def linkamTemperature(self):
        return caget("XF:11BM-ES:{LINKAM}:TEMP")

    def linkamTensilePos(self):
        return caget("XF:11BM-ES:{LINKAM}:TST_MOTOR_POS")


# class Sample(Sample_Generic):
# class Sample(SampleGISAXS):
class Sample(SampleTSAXS):
    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)

        # self.naming_scheme = ['name', 'extra', 'clock', 'temperature', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'y', 'th', 'clock', 'exposure_time']
        #    self.naming_scheme = ['name', 'extra', 'id', 'clock', 'Linkam_temperature', 'exposure_time']
        #    self.naming_scheme = ['name', 'extra', 'id', 'clock', 'Linkam_temperature','Linkam_velocity','Linkam_pos','Linkam_', 'exposure_time']
        self.naming_scheme = [
            "name",
            "extra",
            "LTensile_temperature",
            "LTensile_position",
            "LTensile_force",
            "LTensile_strain",
        ]  # , 'exposure_time']

        self._axes["x"].origin = -16.9
        self._axes["y"].origin = 9.5
        self._axes["th"].origin = 0

        self.md["exposure_time"] = 1
        self.SAXS_time = 60

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

    def get_attribute(self, attribute):
        if attribute == "LTensile_temperature":
            return LTensile.temperature()
        if attribute == "LTensile_position":
            return LTensile.POS.get()
        if attribute == "LTensile_force":
            return LTensile.FORCE.get()
        if attribute == "LTensile_strain":
            return LTensile.STRAIN.get()
        if attribute == "LTensile_stress":
            return LTensile.STRESS.get()

        return super().get_attribute(attribute)

    def get_naming_string(self, attribute):
        # Handle special cases of formatting the text
        super().get_naming_string(attribute)

        if attribute == "LTensile_temperature":
            return "{:.1f}C".format(self.get_attribute(attribute))
        if attribute == "LTensile_position":
            return "{:.1f}mm".format(self.get_attribute(attribute))
        if attribute == "LTensile_force":
            return "{:.1f}N".format(self.get_attribute(attribute))
        if attribute == "LTensile_strain":
            return "Strain{:.1f}".format(self.get_attribute(attribute))
        if attribute == "LTensile_stress":
            return "Stress{:.1f}".format(self.get_attribute(attribute))

        return super().get_naming_string(attribute)

    def get_naming_attribute(self, attribute):
        super().get_naming_string(attribute)
        # if attribute=='temperature_Linkam':
        #     return LThermal.temperature()

    # def setLinkamRate(self, rate):
    #     caput('XF:11BM-ES:{LINKAM}:RAMPRATE:SET', rate)
    #     return rate
    # def setLinkamTemperature(self,temperature ):
    #     caput('XF:11BM-ES:{LINKAM}:SETPOINT:SET', temperature)
    #     return temperature
    # def setLinkamOn(self):
    #     caput('XF:11BM-ES:{LINKAM}:STARTHEAT', 1)
    #     return 1

    # def setLinkamOff(self):
    #     caput('XF:11BM-ES:{LINKAM}:STARTHEAT', 0)
    #     return 0

    # def linkamStatus(self):
    #     return caget('XF:11BM-ES:{LINKAM}:STATUS')

    # def linkamTemperature(self):
    #     return caget('XF:11BM-ES:{LINKAM}:TEMP')

    def measureTimeSeries_custom(self, maxTime=60 * 60 * 6, exposure_time=10, interval=20, reset_clock=True):
        if reset_clock == True:
            self.reset_clock()

        start_time = np.ceil(self.clock() / interval) * interval
        trigger_time = np.arange(start_time, maxTime, interval)

        # while self.clock()<maxTime:
        for trigger in trigger_time:
            while self.clock() < trigger:
                time.sleep(0.2)
            self.measure(exposure_time)

    # def temperature_series_Linkam(self,step=0, temp_sequence = [122, 20, 200, 126, 20, 200], wait_sequence=[1800, 60, 60, 5400, 60, 60], rate_sequence=[30, 10, 10, 30, 10, 10], maxTime=60*60*10, exposure_time=10,interval=20, reset_clock=True):
    # def LinkamTemperatureSeries(self,step=0, temp_sequence = [200, 130, 0, 200, 140, 0, 200, 150, 0, 200], wait_sequence=[60, 60*20, 60*5, 60*1, 60*40, 60*5, 60*1, 60*120, 60*5, 60], rate_sequence=[10, 30, 10, 10, 30, 10, 10, 30, 10, 10], maxTime=60*60*10, exposure_time=10,interval=20, reset_clock=True):

    # def measureTensile(self, exposure_time=None, extra=None, measure_type='snap', verbosity=3, **md):
    #     '''Take a quick exposure (without saving data).'''

    #     self.expose(exposure_time=exposure_time, extra=extra, measure_type=measure_type, verbosity=verbosity, handlefile=False, **md)
    #     remove_last_Pilatus_series()

    def measureTensile(self, exposure_time=None, extra=None, measure_type="measure", verbosity=3, **md):
        """Measure data by triggering the area detectors.

        Parameters
        ----------
        exposure_time : float
            How long to collect data
        extra : string, optional
            Extra information about this particular measurement (which is typically
            included in the savename/filename).
        """

        if exposure_time != None:
            self.set_attribute("exposure_time", exposure_time)
        # else:
        # exposure_time = self.get_attribute('exposure_time')

        savename = self.get_savename(savename_extra=extra)

        if verbosity >= 2 and (get_beamline().current_mode != "measurement"):
            print(
                "WARNING: Beamline is not in measurement mode (mode is '{}')".format(get_beamline().current_mode)
            )

        if verbosity >= 1 and len(get_beamline().detector) < 1:
            print("ERROR: No detectors defined in cms.detector")
            return

        md_current = self.get_md()
        # md_current.update(self.get_measurement_md())
        md_current["sample_savename"] = savename
        md_current["measure_type"] = measure_type
        md_current["filename"] = "{:s}_{:06d}".format(savename, RE.md["scan_id"])
        md_current["LTensile_temperatures"] = LTensile.temperature()
        md_current["LTensile_position"] = LTensile.POS.get()
        md_current["LTensile_force"] = LTensile.FORCE.get()
        md_current["LTensile_strain"] = LTensile.STRAIN.get()
        md_current["LTensile_stress"] = LTensile.STRESS.get()
        md_current.update(md)

        self.expose(exposure_time, extra=extra, verbosity=verbosity, handlefile=True, **md_current)
        # self.expose(exposure_time, extra=extra, verbosity=verbosity, **md)

        self.md["measurement_ID"] += 1

    def LinkamTemperatureSeries(
        self,
        step=0,
        temp_sequence=[30, 40, 50],
        wait_sequence=[5, 5, 5],
        rate_sequence=[30, 10, 30],
        maxTime=60 * 60 * 10,
        exposure_time=10,
        interval=20,
        reset_clock=True,
    ):
        # swaxs_on()
        LThermal.on()
        # temp_sequence = [25, 200, 25, 200]

        # if step<1: #RT
        #     self.doSamples()

        if reset_clock == True:
            self.reset_clock()

        # if step<1:
        # self.measure(exposure_time)
        if step < 3:
            # for index, temperature in enumerate(temp_sequence):
            for temperature, rate, wait_time in zip(temp_sequence, rate_sequence, wait_sequence):
                start_time = np.ceil(self.clock() / interval) * interval
                trigger_time = np.arange(start_time, maxTime, interval)
                LThermal.setTemperatureRate(rate)
                LThermal.setTemperature(temperature)

                # self.setLinkamRate(rate_sequence[index])
                # time.sleep(0.1)
                # self.setLinkamTemperature(temperature)

                # time.sleep(0.2)

                currentTime = []

                for trigger in trigger_time:
                    while self.clock() < trigger:
                        time.sleep(0.2)
                    # self.measure(exposure_time)

                    if abs(LThermal.temperature() - temperature) < 0.5:
                        # if abs(self.linkamTemperature()-temperature)<0.5:
                        currentTime.append(self.clock())
                        print(currentTime[0])
                        # if wait_sequence(index)==0 or self.clock()- currentTime[0] > wait_sequence[index]:
                        if self.clock() - currentTime[0] > wait_time:
                            break

        # if step<5:
        #     self.setLinkamTemperature(20)
        #     time.sleep(0.2)
        #     self.setLinkamOff()

    def LinkamTensileSeries_velocity(
        self,
        step=0,
        temp_sequence=[30, 40, 50],
        wait_sequence=[5, 5, 5],
        rate_sequence=[30, 10, 30],
        maxTime=60 * 60 * 10,
        exposure_time=10,
        interval=20,
        reset_clock=True,
    ):
        # swaxs_on()
        LThermal.on()

        # TODO: check the mode. If not velocity, quit

    def LinkamTensile_step_builder(self):
        """Build the steps fhr the LinkamTensile stage in step mode
        #input is a matrix for each step: (in csv or pandas?)

        #        --- Thermal --------- | --- Tensile ---
        #        T,  T-rate, wait_time, position, velocity (um/s)
        #step1:  30   5        0         500          0
        #step2:  30   20       30        1000        100
        #step3:  40   10       10        1500        200

        # self.LTensile_steps_default = np.array[[30, 5, 10, 0, 0], [40, 20, 30, 500, 100], [50, 10, 10, 500, 0] ]

        """

        # print out the steps for review
        if self.LinkamTensile_stages_default != None:
            print("The current steps are: \n")
            print(self.LinkamTensile_stages_default)
            if input("Do you like to start a new step protocol? y or n") == "n":
                return self.LinkamTensile_stages_default

        # number of elements
        stages = input("How many stages for this run?")
        # steps = int(steps)
        stages_df = pd.DataFrame()

        for step in range(int(stages)):
            # Below line read inputs from user using map() function
            print("step NO {}".format(step))
            para = list(
                map(
                    int,
                    input("\nEnter the numbers for [Temperature, T-rate, wait time, position, velocity] : ")
                    .strip()
                    .split(),
                )
            )[:n]
            current_data = {
                "a_stepNo": step,
                "b_temperature": para[0],
                "c_temperatureRate": para[1],
                "d_waitTime": para[2],
                "e_position": para[3],
                "f_velocity": para[4],
                # 'g_exposure_seconds': exposure_time
            }
            df = pd.DataFrame(data=current_data, index=[step])
            stages_df = pd.concat([stages_df, df])

        # check through the matrix, make sure the motor could move to the correct position in the wait time
        for ii in range(1, len(steps_df)):
            r_distance = stages_df["e_position"][ii] - stages_df["e_position"][ii - 1]
            # print(r_distance)
            if r_distance / stages_df["f_velocity"][ii] > stages_df["d_waitTime"][ii]:
                print("ALERT: you do NOT have enough time to finish step [ {} ].".format(ii))

        self.LinkamTensile_stages_default = stages_df
        # print out the steps for review
        print("The current steps are: \n")
        print(self.LinkamTensile_stages_default)

        return self.LinkamTensile_stages_default

    def LinkamTensileMeasure_step(
        self,
        step=0,
        maxTime=60 * 60 * 10,
        exposure_time=1,
        interval=3,
        reset_clock=True,
    ):
        self.LinkamTensile_stages_default
        #

        # swaxs_on()
        LTensile.on()

        # TODO: check the mode. If not step mode, quit

        # if step<1: #RT
        #     self.doSamples()

        if reset_clock == True:
            self.reset_clock()

        # if step<1:
        # self.measure(exposure_time)
        if step < 3:
            # for index, temperature in enumerate(temp_sequence):
            stages, paras = self.LinkamTensile_stages_default.shape

            # for temperature, rate, wait_time in zip(temp_sequence, rate_sequence, wait_sequence):
            for stage in range(stages):
                (
                    stepNo,
                    temperature,
                    rate,
                    wait_time,
                    position,
                    velocity,
                ) = self.LinkamTensile_stages_default.loc[stage]

                print("Stage {} : {}".format(stage, self.LinkamTensile_stages_default.loc[stage]))

                start_time = np.ceil(self.clock() / interval) * interval
                trigger_time = np.arange(start_time, maxTime, interval)

                LTensile.setTemperatureRate(rate)
                # time.sleep(0.1)
                LTensile.setTemperature(temperature)
                # time.sleep(0.1)

                LTensile.mov(position=position, velocity=velocity)

                currentTime = []

                print("Stage {} : complete the parameter settings and start the measurement".format(stage))
                for trigger in trigger_time:
                    while self.clock() < trigger:
                        time.sleep(0.2)
                    self.measureTensile(exposure_time)

                    # check if the stage is done.
                    #       move done                                     temperature done
                    if int(LTensile.status_code_Tensile.get()) & 4 and int(LTensile.status_code.get()) & 2:
                        break

    def LinkamTensileSeries_step(
        self, num_frames, exposure_time=0.095, exposure_period=0.1, detectors=None, extra=None, verbosity=3, **md
    ):
        """
        Continueous shots with internal trigger of detectors.
        ref: spin-coater code in /2022_3/beamline/

        Parameters
        ----------
        speeds: list
            The list of spinner speeds in multiple steps.
        periods: list
            The lasting time for corresponding spinner speed.
        num_frames : int
            The number of data points.
        exposure_time: float
            The exposure time for single point
        exposure_period: float
            The exposure period for single point. should be at least 0.05s longer than exposure_time
        md : dict, optional
            metadata
        """

        if exposure_period < exposure_time + 0.005:
            return print("Error: exposure period should be at least 0.05s more than exposure time.")

        if detectors == None:
            detectors = cms.detector

        if exposure_time is not None:
            self.set_attribute("exposure_time", exposure_time)

        # Set exposure time

        for detector in get_beamline().detector:
            yield from detector.setExposureTime(exposure_time)
            yield from detector.setExposurePeriod(exposure_period)
            yield from detector.setExposureNumber(num_frames)

        # bec.disable_plots()
        # bec.disable_table()

        self.LinkamTensile_stages_default
        stages, paras = self.LinkamTensile_stages_default.shape
        LTensile.on()

        # TODO: check the mode. If not step mode, quit

        # if reset_clock==True:
        #     self.reset_clock()

        savename = self.get_savename(savename_extra=extra)

        # caput('XF:11BMB-ES{Det:SAXS}:cam1:FileName', savename)

        if verbosity >= 2 and (get_beamline().current_mode != "measurement"):
            print(
                "WARNING: Beamline is not in measurement mode (mode is '{}')".format(get_beamline().current_mode)
            )

        if verbosity >= 1 and len(cms.detector) < 1:
            print("ERROR: No detectors defined in cms.detector")
            return

        md_current = self.get_md()
        md_current["sample_savename"] = savename
        md_current["series"] = "series_measure"
        # md_current.update(self.get_measurement_md())
        # md_current['filename'] = '{:s}_{:04d}.tiff'.format(savename, md_current['detector_sequence_ID'])
        md_current["measure_series_num_frames"] = num_frames
        md_current["filename"] = "{:s}_{:04d}.tiff".format(savename, RE.md["scan_id"])
        md_current["exposure_time"] = exposure_time
        # md_current['measure_series_motor'] = motor.name
        # md_current['measure_series_positions'] = [start, stop]

        # md_current['fileno'] = '{:s}_{:04d}.tiff'.format(savename, RE.md['scan_id'])
        md_current.update(md)

        # turn on shutter
        yield from shutter_on()

        # Perform the scan
        @bpp.stage_decorator(detectors)
        def inner(paras, **md):
            # for detector in detectors:
            #     status = yield from bps.trigger(detector, group='det')

            for stage in range(stages):
                (
                    stepNo,
                    temperature,
                    rate,
                    wait_time,
                    position,
                    velocity,
                ) = self.LinkamTensile_stages_default.loc[stage]

                print("Stage {} : {}".format(stage, self.LinkamTensile_stages_default.loc[stage]))

                # TODO:
                yield from bps.mv(LTensile.temperature_rate_setpoint, rate)
                yield from bps.mv(LTensile.temperature_setpoint, temperature)
                # yield from bps.mv(LTensile.velocity_setpoint, velocity)
                yield from LTensile._mov(position=position, velocity=velocity)
                # yield from bps.mv(LTensile.position_setpoint, position)
                print("1")
                time.sleep(1)
                while (int(LTensile.status_code_Tensile.get()) & 4 and int(LTensile.status_code.get()) & 2) == 0:
                    print("2")
                    print(LTensile.POS.get())

                    time.sleep(0.2)  #    break
                print("3")

                time.sleep(wait_time)

            yield from bps.wait(group="det")

        yield from bpp.run_wrapper(inner(paras, md={"LinkamTensile": stages}))

        # close out
        yield from shutter_off()

        self.md["measurement_ID"] += 1

        # data collected, link uid to file name
        # for detector in cms.detector:
        #     self.handle_fileseries(detector, num_frames=num_frames, extra=extra, verbosity=verbosity, **md)

        # reset the num_frame back to 1
        for detector in get_beamline().detector:
            yield from detector.setExposureNumber(1)

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

    def measureTimeSeries_custom(self, maxTime=60 * 60 * 1.6, exposure_time=6, interval=12, reset_clock=True):
        if reset_clock == True:
            self.reset_clock()

        start_time = np.ceil(self.clock() / interval) * interval
        trigger_time = np.arange(start_time, maxTime, interval)

        # while self.clock()<maxTime:
        for trigger in trigger_time:
            while self.clock() < trigger:
                time.sleep(0.2)
            self.measure(exposure_time)

    def scan_SAXSdet(self, exposure_time=None):
        SAXS_pos = [-73, 0, 73]
        # SAXSx_pos=[-65, 0, 65]

        RE.md["stitchback"] = True

        for SAXSx_pos in SAXS_pos:
            for SAXSy_pos in SAXS_pos:
                mov(SAXSx, SAXSx_pos)
                mov(SAXSy, SAXSy_pos)
                self.measure(10)

    def intMeasure(self, output_file, exposure_time=1):
        """Measure the transmission intensity of the sample by ROI4.
        The intensity will be saved in output_file
        """
        if abs(beam.energy(verbosity=0) - 13.5) < 0.1:
            # beam.setAbsorber(4)
            beam.setTransmission(1e-4)
        elif abs(beam.energy(verbosity=0) - 17) < 0.1:
            beam.setTransmission(1e-6)

        # print('Absorber is moved to position {}'.format(beam.absorber()[0]))

        detselect([pilatus2M])
        # if beam.absorber()[0]>=4:
        bsx.move(bsx.position + 6)
        # beam.setTransmission(1)

        self.measure(exposure_time)

        temp_data = self.transmission_data_output(4)

        cms.modeMeasurement()
        # beam.setAbsorber(0)
        # beam.absorber_out()

        # output_data = output_data.iloc[0:0]

        # create a data file to save the INT data
        INT_FILENAME = "{}/data/{}.csv".format(os.path.dirname(__file__), output_file)

        if os.path.isfile(INT_FILENAME):
            output_data = pds.read_csv(INT_FILENAME, index_col=0)
            output_data = output_data.append(temp_data, ignore_index=True)
            output_data.to_csv(INT_FILENAME)
        else:
            temp_data.to_csv(INT_FILENAME)

    def transmission_data_output(self, slot_pos):
        """Output the tranmission of direct beam"""
        h = db[-1]
        dtable = h.table()

        # beam.absorber_transmission_list = [1, 0.041, 0.0017425, 0.00007301075, 0.00000287662355, 0.000000122831826, 0.00000000513437]
        scan_id = h.start["scan_id"]
        I_bim5 = h.start["beam_int_bim5"]  # beam intensity from bim5
        I0 = dtable.pilatus2M_stats4_total
        filename = h.start["sample_name"]
        exposure_time = h.start["sample_exposure_time"]
        # I2 = dtable.pilatus2M_stats2_total
        # I3 = 2*dtable.pilatus2M_stats1_total - dtable.pilatus2M_stats2_total
        # In = I3 / beam.absorber_transmission_list[slot_pos] / exposure_time

        current_data = {
            "a_filename": filename,
            "b_scanID": scan_id,
            "c_I0": I0,
            "d_I_bim5": I_bim5,
            "e_absorber_slot": slot_pos,
            #'f_absorber_ratio': beam.absorber_transmission_list[slot_pos],
            "f_absorber_ratio": 0.000001,
            "g_exposure_seconds": exposure_time,
        }

        return pds.DataFrame(data=current_data)


# if True:

#     cali = CapillaryHolder(base=stg)
#     #hol = CapillaryHolderCustom(base=stg)
#     cali.name = 'cali'
#     cali.addSampleSlot( Sample('FL_screen'), 5.0 )
#     cali.addSampleSlot( Sample('AgBH_cali_3m_13.5kev'), 8.0 )
#     cali.addSampleSlot( Sample('Empty'), 11.0 )

# if True:

#     # Example of a multi-sample holder

#     ago6 = CapillaryHolderCustom(base=stg)
#     ago6.addGaragePosition(1,1)
#     ago6.addSampleSlot( Sample('NW_VM6102'),3)
#     ago6.addSampleSlot( Sample('NW_M516G_code4'),7)
#     ago6.addSampleSlot( Sample('NW_M516G_code2'),10)
#     # ago6.addSampleSlot( Sample('leftoverbar_slot4'),4)


# %run -i /GPFS/xf11bm/data/2018_1/beamline/user.py

# robot.listGarage()

"""
rod bs
In [424]: wbs()
bsx = -9.2
bsy = 17.000341
bsphi = -223.40267500000002

sphere bs
In [416]: wbs()
In [408]: wbs()
bsx = -10.480053
bsy = -11.899877
bsphi = -16.001704999999987

linkam stage
In [56]: sam=Sample('AML_4_167_6')

In [57]: sam.setOrigin(['x', 'y'])

In [58]: wsam()
smx = -16.2
smy = 9.55
sth = 0.0


"""
