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


RE.md["experiment_alias_directory"] = "/nsls2/data/cms/legacy/xf11bm/data/2023_1/beamline/Commissioning"
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


class SampleLinkamTensile(SampleGISAXS_Generic):
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


class Sample(SampleTSAXS):
    # class Sample(SampleGISAXS):

    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)

        # self.naming_scheme = ['name', 'extra', 'clock', 'temperature', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'y', 'th', 'clock', 'exposure_time']
        self.naming_scheme = [
            "name",
            "extra",
            "id",
            "clock",
            "temperature_Linkam",
            "exposure_time",
        ]
        # self.naming_scheme = ['name', 'extra', 'clock', 'temperature', 'exposure_time']

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

    #    self.attributes['temperature_Linkam'] = self.linkamTemperature()

    def get_attribute(self, attribute):
        if attribute == "temperature_Linkam":
            return LThermal.temperature()
        return super().get_attribute(attribute)

    def get_naming_string(self, attribute):
        # Handle special cases of formatting the text
        if attribute == "temperature_Linkam":
            return "Linkam{:.1f}C".format(self.get_attribute(attribute))
        return super().get_naming_string(attribute)

    def series_measure(
        self,
        num_frames,
        exposure_time=None,
        exposure_period=None,
        detectors=None,
        extra=None,
        per_step=None,
        wait_time=None,
        measure_type="Series_measure",
        verbosity=3,
        fill_gaps=False,
        **md,
    ):
        """
        Continueous shots with internal trigger of detectors. (burst mode)

        Parameters
        ----------
        num_frames : int
            The number of data points.
        exposure_time: float
            The exposure time for single point
        exposure_period: float
            The exposure period for single point. should be at least 0.05s longer than exposure_time
        md : dict, optional
            metadata
        """
        # span = abs(stop-start)
        # positions, dp = np.linspace(start, stop, num, endpoint=True, retstep=True)

        if detectors is None:
            detectors = cms.detector

        if exposure_time is not None:
            self.set_attribute("exposure_time", exposure_time)

        # Set exposure time
        for detector in get_beamline().detector:
            if exposure_time != detector.cam.acquire_time.get():
                RE(detector.setExposureTime(exposure_time))
                # detector.cam.acquire_time.put(exposure_time)
            # detector.cam.acquire_period.put(exposure_period)
            # detector.cam.num_images.put(num_frames)
            RE(detector.setExposurePeriod(exposure_period))
            RE(detector.setExposureNumber(num_frames))

            # if detector.name is 'pilatus2M':

            #     if exposure_time != caget('XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime'):
            #         caput('XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime', exposure_time)
            #     caput('XF:11BMB-ES{Det:PIL2M}:cam1:AcquirePeriod', exposure_period)
            #     caput('XF:11BMB-ES{Det:PIL2M}:cam1:NumImages', num_frames)

            # if detector.name is 'pilatus800':
            #     if exposure_time != caget('XF:11BMB-ES{Det:PIL800K}:cam1:AcquireTime'):
            #         caput('XF:11BMB-ES{Det:PIL800K}:cam1:AcquireTime', exposure_time)
            #     caput('XF:11BMB-ES{Det:PIL800K}:cam1:AcquirePeriod', exposure_period)
            #     caput('XF:11BMB-ES{Det:PIL800K}:cam1:NumImages', num_frames)

            # if detector.name is 'pilatus300' :
            #     if exposure_time != caget('XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime'):
            #         caput('XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime', exposure_time)
            #     caput('XF:11BMB-ES{Det:SAXS}:cam1:AcquirePeriod', exposure_period)
            #     caput('XF:11BMB-ES{Det:SAXS}:cam1:NumImages', num_frames)
            # extra wait time when changing the exposure time.
            # time.sleep(2)
            # elif detector.name is 'PhotonicSciences_CMS':
            # detector.setExposureTime(exposure_time, verbosity=verbosity)

        # for detector in get_beamline().detector:
        # detector.cam.acquire_time.value=exposure_time
        # detector.cam.acquire_period.value=exposure_period
        # detector.cam.num_images.value=num_frames

        # bec.disable_plots()
        # bec.disable_table()

        savename = self.get_savename(savename_extra=extra)
        if verbosity >= 2 and (get_beamline().current_mode != "measurement"):
            print(
                "WARNING: Beamline is not in measurement mode (mode is '{}')".format(get_beamline().current_mode)
            )

        if verbosity >= 1 and len(get_beamline().detector) < 1:
            print("ERROR: No detectors defined in cms.detector")
            return

        md_current = self.get_md()
        # md_current['sample_savename'] = savename
        md_current["measure_type"] = measure_type
        md_current["series"] = "series_measure"
        md_current.update(self.get_measurement_md())
        # md_current['filename'] = '{:s}_{:04d}.tiff'.format(savename, md_current['detector_sequence_ID'])
        md_current["measure_series_num_frames"] = num_frames
        md_current["filename"] = "{:s}_{:04d}.tiff".format(savename, RE.md["scan_id"])
        # md_current['filename'] = '{:s}_{:04d}.tiff'.format(savename, RE.md['scan_id']+1)
        md_current["exposure_time"] = exposure_time
        md_current["exposure_period"] = exposure_period
        # md_current['measure_series_motor'] = motor.name
        # md_current['measure_series_positions'] = [start, stop]

        # md_current['fileno'] = '{:s}_{:04d}.tiff'.format(savename, RE.md['scan_id'])
        md_current.update(md)

        print(RE.md["scan_id"])

        # Perform the scan
        # get_beamline().beam._test_on(wait_time=0.1)
        get_beamline().beam.on()
        RE(count(get_beamline().detector, md=md_current))
        get_beamline().beam.off()

        self.md["measurement_ID"] += 1
        # reset the num_frame back to 1
        for detector in get_beamline().detector:
            RE(detector.setExposureNumber(1))

        # data collected, link uid to file name
        for detector in cms.detector:
            print("handling the file names")
            self.handle_fileseries(detector, num_frames=num_frames, extra=extra, verbosity=verbosity, **md)

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


class GIBar_Custom(GIBar):
    def __init__(self, name="GIBarCustom", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        # self._axes['x'].origin = -59.3
        ##position for calibration
        self._axes["x"].origin = -71.5
        self._axes["y"].origin = 10
        self._axes["th"].origin = 1.1

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

    def doSamples(self, verbosity=3):
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

    # def doSamples(self, verbosity=3):

    ##maxs_on()
    # for sample in self.getSamples():
    # if verbosity>=3:
    # print('Doing sample {}...'.format(sample.name))
    # sample.do()

    # def doSamples_WAXS_only(self,  verbosity=3):
    # for sample in self.getSamples():
    # if verbosity>=3:
    # print('Doing sample {}...'.format(sample.name))
    # if sample.detector=='BOTH':
    # sample.do_WAXS_only()


class CapillaryHolderCustom(CapillaryHolder):
    def __init__(self, name="CapillaryHolderCustom", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._axes["x"].origin = -17.2  # -0.3
        self._axes["y"].origin = -2

    # def doSamples(self, tiling=None):

    #     # swaxs_on()
    #     for sample in self.getSamples():
    #         sample.gotoOrigin()
    #         time.sleep(0.2)
    #         sample.measure(sample.SAXS_time, tiling=tiling)

    def doSamples(
        self,
        step=0,
        exposure_WAXS_time=10,
        exposure_SAXS_time=1,
        int_measure=True,
        output_file="Transmission_output",
        verbosity=3,
    ):
        if step < 5:
            saxs_on()
            for sample in self.getSamples():
                sample.gotoOrigin()
                while smx.moving == True:
                    time.sleep(0.2)
                sample.measure(exposure_time=exposure_SAXS_time, **md)
                # self.doSamples(SAXS_expo_time=SAXS_expo_time, verbosity=verbosity, **md)
            waxs_on()
            for sample in self.getSamples():
                sample.gotoOrigin()
                while smx.moving == True:
                    time.sleep(0.2)
                sample.measure(exposure_time=exposure_WAXS_time, **md)

        if int_measure and step < 10:
            for sample in self.getSamples():
                sample.gotoOrigin()
                while smx.moving == True:
                    time.sleep(0.2)
                sample.intMeasure(output_file=output_file)

    def doSamples_washer(
        self,
        step=0,
        exposure_WAXS_time=10,
        exposure_SAXS_time=10,
        int_measure=True,
        output_file="Transmission_output",
        verbosity=3,
        **md,
    ):
        if step < 5:
            swaxs_on()
            for sample in self.getSamples():
                sample.gotoOrigin()
                while smx.moving == True:
                    time.sleep(0.2)
                sample.measure(exposure_time=exposure_SAXS_time, **md)
                # self.doSamples(SAXS_expo_time=SAXS_expo_time, verbosity=verbosity, **md)

        if int_measure and step < 10:
            for sample in self.getSamples():
                sample.gotoOrigin()
                while smx.moving == True:
                    time.sleep(0.2)
                sample.intMeasure(output_file=output_file)

        # waxs_on()
        # for sample in self.getSamples():
        #     sample.gotoOrigin()
        #     sample.measure(exposure_time=exposure_WAXS_time, **md)


if True:
    cali = CapillaryHolder(base=stg)
    # hol = CapillaryHolderCustom(base=stg)
    cali.name = "cali"
    cali.addSampleSlot(Sample("FL_screen"), 5.0)
    cali.addSampleSlot(Sample("AgBH_cali_3m_13.5kev"), 8.0)
    cali.addSampleSlot(Sample("Empty"), 11.0)

if True:
    # Example of a multi-sample holder

    ago6 = CapillaryHolderCustom(base=stg)
    ago6.addGaragePosition(1, 1)
    ago6.addSampleSlot(Sample("NW_VM6102"), 3)
    ago6.addSampleSlot(Sample("NW_M516G_code4"), 7)
    ago6.addSampleSlot(Sample("NW_M516G_code2"), 10)
    # ago6.addSampleSlot( Sample('leftoverbar_slot4'),4)


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
