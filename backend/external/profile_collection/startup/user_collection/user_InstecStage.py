#!/usr/bin/python
# -*- coding: utf-8 -*-
# vi: ts=4 sw=4


# This doc is created for user group (COsuji from Yale) at 06/02/17
# Please note any setup change druing their visit.

################################################################################
#  Short-term settings (specific to a particular user/experiment) can
# be placed in this file. You may instead wish to make a copy of this file in
# the user's data directory, and use that as a working copy.
################################################################################

import math

# logbooks_default = ['User Experiments']
# tags_default = ['CFN Soft-Bio']


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
        # self.naming_scheme = ['name', 'extra', 'temperature', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'clock', 'temperature_B', 'exposure_time']
        self.naming_scheme = ["name", "extra", "temperature", "y", "exposure_time"]

        self.md["exposure_time"] = 20.0


class SampleGISAXS(SampleGISAXS_Generic):
    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)
        self.naming_scheme = ["name", "extra", "th", "exposure_time"]
        # self.temperatures_default = [85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 28]
        self.temperatures_default = [
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
            90,
            95,
            100,
            105,
            110,
            115,
            120,
            125,
            130,
            135,
            140,
            145,
            150,
            155,
            160,
            165,
            170,
            165,
            160,
            155,
            150,
            145,
            140,
            135,
            130,
            125,
            120,
            115,
            110,
            105,
            100,
            95,
            90,
            85,
            80,
            75,
            70,
            65,
            60,
            55,
            50,
            45,
            40,
            35,
            30,
            25,
        ]


class Sample(SampleTSAXS):
    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)

        # self.naming_scheme = ['name', 'extra', 'clock', 'temperature', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'y', 'th', 'clock', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'th', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'x', 'y', 'exposure_time']
        self.naming_scheme = ["name", "extra", "temperature_C", "exposure_time"]
        # self.naming_scheme = ['name', 'extra', 'clock', 'temperature', 'ewxposure_time']
        # self.naming_scheme = ['name', 'extra', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'temperature_B', 'y', 'exposure_time']
        # self.naming_scheme = ['name', 'extra', 'id', 'clock', 'temperature_B', 'x', 'y', 'exposure_time']

        self.md["exposure_time"] = 10.0

        self.mesh_y_range = np.arange(-2.5, 2.5 + 0.01, 0.1)

        # self.incident_angles_default = [0.08, 0.10, 0.12, 0.15, 0.20]
        # self.incident_angles_default = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        self.incident_angles_default = [0.05, 0.08, 0.10, 0.12, 0.15]
        # self.incident_angles_default = [0.02, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1, 0.12, 0.15]
        # self.incident_angles_default = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15]

    def measure_mesh_scan(
        self,
        num_spots=50 + 1,
        translation_amount_y=0.1,
        translation_amount_x=0.1,
        exposure_time=3,
    ):
        self.naming_scheme = ["name", "extra", "x", "y", "exposure_time"]

        for y_pos in self.mesh_y_range:
            self.yabs(y_pos)
            self.xabs(-2.5)
            time.sleep(2)
            sam.measureSpots(
                num_spots=num_spots,
                translation_amount=translation_amount_x,
                exposure_time=exposure_time,
            )

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
                detector = gs.DETS[0]
                value_name = gs.TABLE_COLS[0]
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
            fit_scan(sth, 1.5, 21, fit="max")

        # if step<=5:
        #    #fit_scan(smy, 0.6, 17, fit='sigmoid_r')
        #    fit_edge(smy, 0.6, 17)
        #    fit_scan(sth, 1.2, 21, fit='max')

        if step <= 8:
            # fit_scan(smy, 0.3, 21, fit='sigmoid_r')
            fit_edge(smy, 0.5, 21)
            fit_scan(sth, 1.1, 21, fit="max")

            self.setOrigin(["y", "th"])

        if step <= 9:
            # fit_scan(smy, 0.3, 21, fit='sigmoid_r')
            fit_edge(smy, 0.4, 21)
            fit_scan(sth, 0.9, 21, fit="max")

            self.setOrigin(["y", "th"])

        # if step<=9 and reflection_angle is not None:
        ## Final alignment using reflected beam
        # if verbosity>=4:
        # print('    align: reflected beam')
        # get_beamline().setReflectedBeamROI(total_angle=reflection_angle*2.0)

        # self.thabs(reflection_angle)

        # result = fit_scan(sth, 0.7, 41, fit='max')
        # sth_target = result.values['x_max']-reflection_angle

        # if result.values['y_max']>10:
        # th_target = self._axes['th'].motor_to_cur(sth_target)
        # self.thsetOrigin(th_target)

        if step <= 10:
            self.thabs(0.0)
            beam.off()

    def measureTimeSeriesAngles_multix(
        self,
        exposure_time=None,
        num_frames=10,
        wait_time=None,
        extra=None,
        measure_type="measureTimeSeries",
        verbosity=3,
        tiling=False,
        fix_name=True,
        **md,
    ):
        # Initial sam.do() alignment of the sample seems to dewet (ionize the air around) the sample so multix time series should not include the x position of the alignment for best measurements.
        if fix_name and ("clock" not in self.naming_scheme):
            self.naming_scheme_hold = self.naming_scheme
            self.naming_scheme = self.naming_scheme_hold.copy()
            self.naming_scheme.insert(-1, "clock")

        md["measure_series_num_frames"] = num_frames

        for i in range(num_frames):
            if verbosity >= 3:
                print(
                    "Measuring frame {:d}/{:d} ({:.1f}% complete).".format(
                        i + 1, num_frames, 100.0 * i / num_frames
                    )
                )

            md["measure_series_current_frame"] = i + 1
            print("Angles in measure include: {}".format(sam.incident_angles_default))
            self.measureIncidentAngles(exposure_time=exposure_time, extra=extra, **md)
            movr(smx, 0.2)
            if i > 1 and i % 10 == 0:
                sam.xo()
            if wait_time is not None:
                sleep(wait_time)

    def temperature_cool_series(
        self,
        step=0,
        temp_cool_seq=None,
        temperature_probe="C",
        exposure_time=30,
        wait_time_cool=200,
        temp_tolerance=0.5,
    ):
        if temp_cool_seq is None:
            temp_cool_seq1 = np.arange(190, 155 - 0.1, -5)
            temp_cool_seq2 = np.arange(153, 151 - 0.1, -2)
            temp_cool_seq3 = np.arange(150, 140 - 0.1, -1)
            temp_cool_seq4 = np.arange(135, 80 - 0.1, -5)

            temp_cool_seq = np.concatenate(
                (temp_cool_seq1, temp_cool_seq2, temp_cool_seq3, temp_cool_seq4), axis=0
            )

        # cooling
        if step < 5:
            for temperature in temp_cool_seq:
                self.setTemperature(temperature, output_channel="3")
                while (
                    abs(self.temperature(temperature_probe=temperature_probe, output_channel="3") - temperature)
                    > temp_tolerance
                ):
                    time.sleep(2)
                    self.temperature(temperature_probe=temperature_probe, output_channel="3")

                print(
                    "reach the set temperature: {}C and start waiting for {}s".format(temperature, wait_time_cool)
                )
                time.sleep(wait_time_cool)
                self.measure(exposure_time=exposure_time)

        if step < 10:
            self.setTemperature(25, output_channel="3")  # def temperature_seires(self):

    ## for Instec 402 stage
    def tscan(
        self,
        temperature_start,
        temperature_final,
        num_intervals,
        wait_time,
        temp_update_time=5,
        exposure_time=0,
    ):
        if temperature_start == None or temperature_start < 0.0 or temperature_start >= 250:
            print("temperature_start must be set between 0 and 250 degC.\n")
            return 0

        if temperature_final == None or temperature_final < 0.0 or temperature_final >= 250:
            print("temperature_final must be set between 0 and 250 degC.\n")
            return 0

        temperature_step = (temperature_final - temperature_start) / abs(num_intervals)

        if temperature_final < temperature_start:
            temperature_series = np.arange(temperature_start, temperature_final - 0.0001, temperature_step)
        else:
            temperature_series = np.arange(temperature_start, temperature_final + 0.0001, temperature_step)

        tscan_zero_time = time.time()
        self.tscan_seconds = []
        self.tscan_degC = []
        self.tscan_data = []

        # dump all the (seconds, degC) data into a file; use filename like "<sam.name>_tscan_<first_ID>-<last_ID>.csv" under *user/tscan directory
        self.tscan_filename = "{}/tscan/{}_tscan_{}.csv".format(
            RE.md["experiment_alias_directory"], self.name, RE.md["scan_id"]
        )
        self.tscan_data = pds.DataFrame(columns=["scan_id", "degC", "seconds"])
        # f=open(filename,'w')

        for temperature_setpoint in temperature_series:
            self.setTemperature(temperature_setpoint, output_channel="3")
            self.temperature(temperature_probe="C", output_channel="3")
            # f.write('# scan_ID seconds degC\n')

            for t_wait in np.arange(0, wait_time, temp_update_time):
                time.sleep(temp_update_time)
                current_time = time.time() - tscan_zero_time
                current_temperature = self.temperature(temperature_probe="C", output_channel="3", verbosity=2)
                self.tscan_seconds.append(current_time)
                self.tscan_degC.append(current_temperature)
                print("{:.3f} {:.3f}".format(current_time, current_temperature))
                # f.write('{:d} {:.3f} {:.3f}\n'.format(RE.md['scan_id'], current_time, current_temperature))
                self.tscan_data = self.tscan_data.append(
                    {
                        "scan_id": RE.md["scan_id"],
                        "degC": current_temperature,
                        "seconds": current_time,
                    },
                    ignore_index=True,
                )

            # while self.IC_int() == False:
            # print('The beam intensity is lower than it should be. Beam may be lost.')
            # sleep(120)

            if exposure_time > 0:
                saxs_on()
                self.measure(exposure_time)
                waxs_on()
                self.measure(exposure_time)

        self.tscan_data.to_csv(self.tscan_filename)
        # f.close()

    # def tscan_save to recover tscan data when prematurely terminated
    def tscan_save(self):
        # filename = '{}/tscan/{}_tscan_{}.csv'.format(RE.md['experiment_alias_directory'], self.name, RE.md['scan_id']-1)
        ##I'm not sure whether tscan_data is saved or not when terminating tscans. If not, try to load the data into datafrme again.
        # tscan_data = pds.DataFrame(columns=['scan_id', 'degC', 'seconds'])
        # for ii in len(self.tscan_seconds):
        # tscan_data = tscan_data.append({'scan_id':RE.md['scan_id']-1-len(self.tscan_seconds)+ii, 'degC':self.tscan_degC, 'seconds':self.tscan_seconds}, ignore_index=True)
        self.tscan_data.to_csv(self.tscan_filename)

    def IC_int(self):
        ion_chamber_readout1 = caget("XF:11BMB-BI{IM:3}:IC1_MON")
        ion_chamber_readout2 = caget("XF:11BMB-BI{IM:3}:IC2_MON")
        ion_chamber_readout3 = caget("XF:11BMB-BI{IM:3}:IC3_MON")
        ion_chamber_readout4 = caget("XF:11BMB-BI{IM:3}:IC4_MON")

        ion_chamber_readout = (
            ion_chamber_readout1 + ion_chamber_readout2 + ion_chamber_readout3 + ion_chamber_readout4
        )

        return ion_chamber_readout > 1 * 1e-08


class CapillaryHoldeCustom(CapillaryHolder):
    def __init__(self, name="CapillaryHolder", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._axes["y"].origin = 13.4  # smy position for slot 4 of 7-capillary cassette
        self._axes["x"].origin = -17.6  # smx position for slot 4 of 7-capillary cassette

        # self.temperatures_default = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35]

        # special CapillaryHolder for instec stage with 7 capillaries in 2" seperation.
        ## NEED to update
        ## slot  1; smx = +26.60
        ## slot  4; smx = -17.80
        ## slot  7; smx = -61.94
        self.x_spacing = 0.2 * 25.4  # 0.2" seperation in x

    def tscan(
        self,
        temperature_start,
        temperature_final,
        num_intervals,
        wait_time,
        temp_update_time=5,
        exposure_time=0,
    ):
        if temperature_start == None or temperature_start < 0.0 or temperature_start >= 250:
            print("temperature_start must be set between 0 and 250 degC.\n")
            return 0

        if temperature_final == None or temperature_final < 0.0 or temperature_final >= 250:
            print("temperature_final must be set between 0 and 250 degC.\n")
            return 0

        temperature_step = (temperature_final - temperature_start) / abs(num_intervals)

        if temperature_final < temperature_start:
            temperature_series = np.arange(temperature_start, temperature_final - 0.0001, temperature_step)
        else:
            temperature_series = np.arange(temperature_start, temperature_final + 0.0001, temperature_step)

        tscan_zero_time = time.time()
        self.tscan_seconds = []
        self.tscan_degC = []
        self.tscan_data = []

        # dump all the (seconds, degC) data into a file; use filename like "<sam.name>_tscan_<first_ID>-<last_ID>.csv" under *user/tscan directory
        self.tscan_filename = "{}/tscan/{}_tscan_{}.csv".format(
            RE.md["experiment_alias_directory"], self.name, RE.md["scan_id"]
        )
        self.tscan_data = pds.DataFrame(columns=["scan_id", "degC", "seconds"])
        # f=open(filename,'w')

        for temperature_setpoint in temperature_series:
            self.setTemperature(temperature_setpoint, output_channel="3")
            self.temperature(temperature_probe="C", output_channel="3")
            # f.write('# scan_ID seconds degC\n')

            for t_wait in np.arange(0, wait_time, temp_update_time):
                time.sleep(temp_update_time)
                current_time = time.time() - tscan_zero_time
                current_temperature = self.temperature(temperature_probe="C", output_channel="3", verbosity=2)
                self.tscan_seconds.append(current_time)
                self.tscan_degC.append(current_temperature)
                print("{:.3f} {:.3f}".format(current_time, current_temperature))
                # f.write('{:d} {:.3f} {:.3f}\n'.format(RE.md['scan_id'], current_time, current_temperature))
                self.tscan_data = self.tscan_data.append(
                    {
                        "scan_id": RE.md["scan_id"],
                        "degC": current_temperature,
                        "seconds": current_time,
                    },
                    ignore_index=True,
                )

            # while self.IC_int() == False:
            # print('The beam intensity is lower than it should be. Beam may be lost.')
            # sleep(120)

            if exposure_time > 0:
                self.measure(exposure_time)

        self.tscan_data.to_csv(self.tscan_filename)
        # f.close()

    # def tscan_save to recover tscan data when prematurely terminated
    def tscan_save(self):
        # filename = '{}/tscan/{}_tscan_{}.csv'.format(RE.md['experiment_alias_directory'], self.name, RE.md['scan_id']-1)
        ##I'm not sure whether tscan_data is saved or not when terminating tscans. If not, try to load the data into datafrme again.
        # tscan_data = pds.DataFrame(columns=['scan_id', 'degC', 'seconds'])
        # for ii in len(self.tscan_seconds):
        # tscan_data = tscan_data.append({'scan_id':RE.md['scan_id']-1-len(self.tscan_seconds)+ii, 'degC':self.tscan_degC, 'seconds':self.tscan_seconds}, ignore_index=True)
        self.tscan_data.to_csv(self.tscan_filename)

    def get_slot_position(self, slot):
        """Return the motor position for the requested slot number."""
        # This method should be over-ridden in sub-classes, so as to properly
        # implement the positioning appropriate for that holder.

        position_x = 0.0 + slot - 4

        return position_x * self.x_spacing

    # def addSampleSlot(self, sample, slot):
    #'''Adds a sample to the specified "slot" (defined/numbered sample
    # holding spot on this holder).'''

    # self.addSample(sample, sample_number=slot)
    # sample.setOrigin( ['x'], [self.get_slot_position(slot)[0]] )
    # sample.setOrigin( ['yy'], [self.get_slot_position(slot)[1]] )

    def temperature_seires(self):
        temperatures_increase = np.arange(27, 200.1, 0.5)
        for temperature in temperatures_increase:
            self.setTemperature(temperature)
            self.temperature(temperature_probe="B")
            time.sleep(30)
            while self.IC_int() == False:
                print("The beam intensity is lower than it should be. Beam may be lost.")
                sleep(120)
            # for tt in self.temperature_default:
            # if abs(temperature-tt)<0.01:

            # self.measure(1)

        time.sleep(600)

        temperatures_decrease = np.arange(200, 27, -0.5)
        for temperature in temperatures_decrease:
            self.setTemperature(temperature)
            self.temperature(temperature_probe="B")
            time.sleep(30)
            while self.IC_int() == False:
                print("The beam intensity is lower than it should be. Beam may be lost.")
                time.sleep(120)

    def do_series(self, y_pos=2.6, exposure_time=10, num_frames=5000, wait_time=0, verbosity=3):
        for ii in range(num_frames):
            for sample in self.getSamples():
                if verbosity >= 3:
                    print("Doing sample {}...".format(sample.name))
                sample.xo()
                # sample.yo()
                sample.yabs(y_pos)
                sample.measure(exposure_time=exposure_time)

            if wait_time > 0:
                time.sleep(wait_time)

            # while self.IC_int() == False:
            # print('The beam intensity is lower than it should be. Beam may be lost.')
            # time.sleep(120)

    def do_yscan_series(
        self,
        y_pos_start=0,
        y_pos_end=2.8,
        translation_amount=0.4,
        exposure_time=10,
        num_frames=5000,
        wait_time=0,
        verbosity=3,
    ):
        for ii in range(num_frames):
            for sample in self.getSamples():
                if verbosity >= 3:
                    print("Doing sample {}...".format(sample.name))
                sample.xo()
                # sample.yo()
                sample.yabs(y_pos_start)
                # sample.measure(exposure_time=exposure_time)
                num_spots = int(np.ceil(y_pos_end - y_pos_start) / translation_amount)
                sample.measureSpots(
                    num_spots=num_spots,
                    translation_amount=translation_amount,
                    axis="y",
                    exposure_time=exposure_time,
                )

            if wait_time > 0:
                time.sleep(wait_time)

    def reset_clock(self):
        for sample in self.getSamples():
            sample.reset_clock()

    def do_series_smarter(self, num_frames=5000, wait_time=0):
        for ii in range(num_frames):
            if self.temperature(temperature_probe="B") > 25 and self.temperature(temperature_probe="B") < 70:
                detselect([pilatus2M, psccd])
                self.doSamples()
            elif self.temperature(temperature_probe="B") > 70:
                detselect([pilatus2M])
                self.doSamples()
                time.sleep(80)

            if wait_time > 0:
                time.sleep(wait_time)

            while self.IC_int() == False:
                print("The beam intensity is lower than it should be. Beam may be lost.")
                sleep(120)

    def do_series_temperature(self, temperature=120, num_frames=5000, wait_time=0):
        self.setTemperature(temperature)
        flow_on(voltage=5)
        for ii in range(num_frames):
            self.doSamples()

            if wait_time > 0.0:
                time.sleep(wait_time)

            while self.IC_int() == False:
                print("The beam intensity is lower than it should be. Beam may be lost.")
                time.sleep(120)

            if self.temperature < temperature + 1:
                flow_off()

    def do_series_multix(self, num_frames=5000, wait_time=0):
        for ii in range(num_frames):
            if ii % 3 != 0:
                self.xo()
                self.xr(0.2)
                self.setOrigin(["x"])

            if ii % 3 == 0:
                self.xo()
                self.xr(-0.2 * 2)
                self.setOrigin(["x"])

            self.doSamples()

            if wait_time > 0:
                time.sleep(wait_time)

            while self.IC_int() == False:
                print("The beam intensity is lower than it should be. Beam may be lost.")
                sleep(120)

    def IC_int(self):
        ion_chamber_readout1 = caget("XF:11BMB-BI{IM:3}:IC1_MON")
        ion_chamber_readout2 = caget("XF:11BMB-BI{IM:3}:IC2_MON")
        ion_chamber_readout3 = caget("XF:11BMB-BI{IM:3}:IC3_MON")
        ion_chamber_readout4 = caget("XF:11BMB-BI{IM:3}:IC4_MON")

        ion_chamber_readout = (
            ion_chamber_readout1 + ion_chamber_readout2 + ion_chamber_readout3 + ion_chamber_readout4
        )

        return ion_chamber_readout > 1 * 1e-08


def save_to_csv(self):
    data = []
    for i in range(5):
        current = {"test1": i, "test2": i - 1}
        current_DF = pds.DataFrame(data=current)
        data.append(current_DF)


def saxs_on():
    detselect([pilatus2M])
    WAXSx.move(-215)
    WAXSy.move(40)


def waxs_on():
    detselect([pilatus800])
    WAXSx.move(-191)
    WAXSy.move(16)


def stage_center():
    # smx.move(-17.4)
    # smx.move(-16.25)
    smx.move(-16.98)
    # smy.move(7.5)
    # smy.move(6.5)
    # smy.move(6.9)
    smy.move(8.7)
    sam.setOrigin(["x", "y"])


# cms.SAXS.setCalibration([738, 1092], 2, [-65, -73])

cms.SAXS.setCalibration([770, 1089], 2.02, [-60, -73])  # 17kev

RE.md["experiment_alias_directory"] = "/nsls2/xf11bm/data/2019_2/SSprunt2"


if False:
    hol = CapillaryHolder(base=stg)
    # hol = CapillaryHolderCurrent(base=stg)
    # hol = CapillaryHolderCustom(base=stg)

    hol.addSampleSlot(SampleTSAXS("RTD"), 1.0)
    hol.addSampleSlot(Sample("L4T"), 2.0)
    hol.addSampleSlot(Sample("RL7T-2nd"), 3.0)
    hol.addSampleSlot(Sample("RUL4T-2nd"), 4.0)
    hol.addSampleSlot(Sample("BLG20T"), 5.0)
    hol.addSampleSlot(Sample("RUL10T-2nd"), 6.0)
    hol.addSampleSlot(Sample("UL7T"), 7.0)


"""   
Instec 402 stage setup:

In [1698]: wbs()                                                                                                               
bsx = -5.62885
bsy = -9.09994
bsphi = -16.001724999999993

In [1699]: beam.energy(); beam.divergence(); beam.size()                                                                       
E = 17.00 keV, wavelength = 0.7293 Å, Bragg = 0.018143 rad = 1.0395 deg
Beam divergence:
  horizontal = 0.100 mrad
  vertical   = 0.100 mrad
Beam size:
  horizontal = 0.200 mm
  vertical   = 0.200 mm
Out[1699]: (0.19999050000000018, 0.19999725000000002)






"""


""" OLD stuff
- To load the new sample names in holder, update the sample names in the list above and run in python shell:

In [228]: %run -i /GPFS/xf11bm/data/2017_3/SSprunt2/user.py

- To list samples in holder:

In [229]: hol.listSamples()
3.0: Buffer
4.0: Empty_cap
5.0: DNA_unaligned

- To go to a sample:
In [230]: sam=hol.gotoSample(5)
DNA_unaligned.x = 0.000 mm       
In [231]: sam.name
Out[231]: 'DNA_unaligned'

- To move to different spots on sample:
sam.yr()
sam.xr()

- To go to origin of sample, check sample position, etc:

In [232]: sam.xo(); sam.yo()

In [233]: sam.pos()
DNA_unaligned.th = 0.000 deg (origin = 0.000)
DNA_unaligned.x = 0.000 mm (origin = 5.080)
DNA_unaligned.y = 0.000 mm (origin = 0.000)
Out[233]: {'th': 0.0, 'x': 0.0, 'y': 1.7763568394002505e-15}

- To set new origin for a given sample:
In [240]: sam.setOrigin(['x','y'])




hol.do_yscan_series(y_pos_start=-.3, y_pos_end=2.9, exposure_time=10, num_frames=12, wait_time=0)
"""
