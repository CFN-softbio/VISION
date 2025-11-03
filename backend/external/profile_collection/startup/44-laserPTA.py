from epics import caput, caget


class PhotoThermalAnnealer:
    def __init__(self, print_code="PTA> "):
        # self.controlTTL_PV = 'XF:11BMB-ES{IO}AO:3-SP'
        # self.controlTTL_PV = 'XF:11BM-ES{Ecat:DO1_2}'
        # self.powerV_PV = 'XF:11BMB-ES{IO}AO:4-SP'
        # self.powerV_PV = 'XF:11BM-ES{Ecat:AO1}1'

        # setting at Feb 2023
        self.powerV_PV = "XF:11BM-ES{Ecat:AO2}1"
        self.controlTTL_PV = "XF:11BM-ES{Ecat:DO1_2}"

        self.print_code = print_code
        # if verbosity>=3:
        # print('{}'.format(self.print_code))

        self.laserOff()

        # self._use_calibration_NoFlow()
        # self._use_calibration_N2()

        self.laser_calib_m, self.laser_calib_b, self.power_limit_low = (
            7.4632,
            -5.8773,
            0.13,
        )  # BNL 70W IR Laser factory calibration (2019-Mar)
        # self.laser_calib_m, self.laser_calib_b, self.power_limit_low = 7.5376, -6.9667, 0.38 # UW 70W IR Laser factory calibration (2019-Apr)

        # self.power_limit_high = 65.1 # For RTP
        self.power_limit_high = 70.0  # For LZA

        self.calibration_m = 234 / 70
        self.calibration_b = 0

    def linear_regression(self, data, verbosity=3):
        # import numpy as np
        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.asarray(data)[:, 0], np.asarray(data)[:, 1]
        )
        if verbosity >= 5:
            print("    m = {:.3f} ; b = {:.3f} [R = {:.2f}]".format(slope, intercept, r_value))

        return slope, intercept, r_value

    # Laser control
    ########################################

    def laserOn(self):
        caput(self.controlTTL_PV, 1)

    def laserOff(self):
        caput(self.controlTTL_PV, 0)

    def laserPulse(self, duration, power_W=None):
        if power_W is not None:
            self.setLaserPower(power_W)

        self.laserOn()
        print("{}  Laser on for {:.2f} s".format(self.print_code, float(duration)))
        time.sleep(duration)
        self.laserOff()
        print("{}  Laser off".format(self.print_code))

    def off(self):
        self.laserOff()

    def getVoltage(self, verbosity=3):
        Voltage = float(caget(self.powerV_PV))
        power_W = self.laser_calib_m * Voltage + self.laser_calib_b

        if verbosity >= 3:
            print("{}    Voltage = {:.2f} V ({:.2f} W)".format(self.print_code, Voltage, power_W))

        return Voltage

    def getLaserPower(self, verbosity=3):
        Voltage = float(caget(self.powerV_PV))
        power_W = self.laser_calib_m * Voltage + self.laser_calib_b

        if verbosity >= 3:
            print("{}    Power = {:.2f} W ({:.2f} V)".format(self.print_code, power_W, Voltage))

        return power_W

    def setLaserPower(self, power_W, verbosity=3):
        if power_W >= self.power_limit_high:
            print(
                "{}Desired laser power must be less than {:.2f} W (your input was {:.2f} W).".format(
                    self.print_code, self.power_limit_high, power_W
                )
            )
            return

        if power_W <= self.power_limit_low:
            print(
                "{}Desired laser power must be greater than {:.2f} W (your input was {:.2f} W).".format(
                    self.print_code, self.power_limit_low, power_W
                )
            )
            power_W = 0

        power_V = (power_W - self.laser_calib_b) / self.laser_calib_m

        if power_V > 10.25 or power_V < 0:
            print(
                "{}Desired laser power ({:.2f} W) requires {:.2f} V, which is not possible.".format(
                    self.print_code, power_W, power_V
                )
            )
            return

        if verbosity >= 3:
            print(
                "{}Setting laser to {:.2f} W (using control voltage of {:.2f} V)".format(
                    self.print_code, power_W, power_V
                )
            )
        caput(self.powerV_PV, power_V)

    # Temperature calibration
    # (conversion from laser power into temperature)
    ########################################

    # def _use_calibration_NoFlow(self, verbosity=3):

    #     # Normal RTP height, no air flow, open GISAXS windows
    #     # [ power_W, Temperature_C ] , # power_V
    #     data = [
    #         [0, 30.0] , # 0.0
    #         [10, 175.9] , # 2.13
    #         [12.43, 206.7] , # 2.45
    #         [15.0, 239.9] , # 2.80
    #         [16.38, 257.2] , # 2.98
    #         [18.35, 279.9] , # 3.25
    #         [20.0, 298.4] , # 3.47
    #         [25.0, 350.1] , # 4.14
    #         #[50.0, >400] , # 7.49
    #         ]

    #     m, b, r_value = self.linear_regression(data, verbosity=verbosity)
    #     if verbosity>=3:
    #         print("{}    Calibration updated to: m = {:.3f}°C/W; b = {:.3f}°C    [R = {:.2f}]".format(self.print_code, m, b, r_value))
    #     self.calibration_m, self.calibration_b = m, b

    # def _use_calibration_N2(self, verbosity=3):

    #     # Normal RTP height, N2 flow, Kapton GISAXS windows
    #     # [ power_W, Temperature_C ] , # power_V
    #     data = [
    #         [0, 24.2] , # 0.0
    #         [9.24, 164.9] , # 2.02
    #         [10.79, 188.0] , # 2.23
    #         [10.79, 190.9] , # 2.23
    #         [12.34, 211.4] , # 2.44
    #         [13.89, 227] , # 2.65
    #         [15.44, 252] , # 2.65
    #         [16.99, 271] , # 3.06
    #         ]

    #     m, b, r_value = self.linear_regression(data, verbosity=verbosity)
    #     if verbosity>=3:
    #         print("{}    Calibration updated to: m = {:.3f}°C/W; b = {:.3f}°C    [R = {:.2f}]".format(self.print_code, m, b, r_value))
    #     self.calibration_m, self.calibration_b = m, b

    def getTemperature(self, T_base=100, verbosity=3):
        power_W = self.getLaserPower(verbosity=0)
        m, b = self.calibration_m, self.calibration_b  # [°C/W] , [°C]
        T = m * power_W + b + T_base

        if verbosity >= 3:
            print("{}Current set temperature is {:.2f}°C (using {:.2f} W)".format(self.print_code, T, power_W))
            if caget(self.controlTTL_PV) == True:
                print("{}  Note: Laser is not turned on.".format(self.print_code))

        return T

    def setTemperature(self, T_target, T_base=100, verbosity=3):
        T_excess = T_target - T_base

        m, b = self.calibration_m, self.calibration_b  # [°C/W] , [°C]

        power_W = (T_excess - b) / m

        if verbosity >= 3:
            print("{}Setting temperature to {:.2f}°C".format(self.print_code, T_target))
        self.setLaserPower(power_W)

        if verbosity >= 3:
            if caget(self.controlTTL_PV) == True:
                print("{}  Note: Laser is not turned on.".format(self.print_code))

    # def setVoltage(self, V_target, verbosity=3):

    #     m, b = self.calibration_m, self.calibration_b # [°C/W] , [°C]

    #     power_W = (T_target-b)/m

    #     if verbosity>=3:
    #         print("{}Setting temperature to {:.2f}°C".format(self.print_code, T_target))
    #     self.setLaserPower(power_W)

    #     if verbosity>=3:
    #         if abs(int(caget(self.controlTTL_PV))-5)>0.1:
    #             print("{}  Note: Laser is not turned on.")

    # RTP protocols
    ########################################

    def controlTemperature(self, T_target, adjust_strength=0.5, delay_time=0.1, adjust_clip=1, verbosity=3):
        m, b = self.calibration_m, self.calibration_b  # [°C/W] , [°C]

        power_W_nominal = (T_target - b) / m

        power_W_current = power_W_nominal

        while True:
            T_current = 0  # TODO: Obtain T from beamline

            adjust = adjust_strength * m * (T_target - T_current)
            adjust = np.clip(adjust, adjust - adjust_clip, adjust + adjust_clip)

            if verbosity >= 3:
                print(
                    "{}        T = {:.2f}°C; Adjusting laser power by {:+.2f} W (from {:.2f} W to {:.2f} W)".format(
                        self.print_code,
                        T_current,
                        adjust,
                        power_W_current,
                        power_W_current + adjust,
                    )
                )

            power_W_current += adjust
            self.setLaserPower(power_W_current)

            time.sleep(delay_time)

    def jumpTemperature(
        self,
        T_target,
        T_current=30,
        dwell_time=None,
        ramp_usage_factor=0.80,
        ramp_power=50,
        rate_initial=None,
        verbosity=3,
    ):
        if rate_initial is None:
            # Calculate the expected rate from prior calibration

            # T_i = 35C, power_W = 50, rate_initial = (358.9-47.2)/(68.905-59.829) = 34.34 °C/s
            rate_initial = 35 * (ramp_power / 50)
            # T_i = 35C, power_W = 65, rate_initial = 42 °C/s
            # rate_initial = 42*(ramp_power/65)

        ramp_time = ramp_usage_factor * (T_target - T_current) / rate_initial

        if verbosity >= 4:
            print("{}    ramp_time = {:.1f} s".format(self.print_code, ramp_time))

        # Fast ramp
        self.setLaserPower(ramp_power)
        start_time = time.time()
        self.laserOn()

        i = 0
        while (time.time() - start_time) < ramp_time:
            if verbosity >= 3:
                current_time = time.time() - start_time
                if i % 50 == 0:
                    print(
                        "{}  FAST RAMP @ {:.1f}°C/s to T = {:.1f}°C; {:.1f} s (ramp_time {:.1f} s) {:.1f}%".format(
                            self.print_code,
                            rate_initial,
                            T_target,
                            current_time,
                            ramp_time,
                            100.0 * current_time / ramp_time,
                        )
                    )

            i += 1
            time.sleep(0.01)

        # Stabilize at desired temperature
        self.setTemperature(T_target, verbosity=verbosity)

        if dwell_time is not None:
            if verbosity >= 3:
                print("{}  Holding at T = {:.1f}°C for {:.1f} s...".format(self.print_code, T_target, dwell_time))
            time.sleep(dwell_time)
            self.laserOff()
            if verbosity >= 3:
                print("{}  Laser off.".format(self.print_code))

    # LZA helpers
    ########################################
    def msg(self, txt, priority=1, indent=0, indent_char="  "):
        indent_str = indent_char * (indent)

        txt = "LZA> {}{}".format(indent_str, txt)
        if priority >= 1:
            print(txt)

    # def date_stamp(self, priority=1, indent=0):
    # self.log.date_stamp(priority=priority, indent=indent)

    def start_timing(self, predicted_duration):
        """Allows one to define the predicted length of time for an event (or sequence of events),
        so one can thereafter check on the progress."""

        self.timing = True
        self.timing_predicted_duration = predicted_duration
        self.timing_start_time = time.time()

    def end_timing(self):
        self.timing = False

    def is_timing(self):
        return self.timing

    def get_run_timing(self):
        return time.time() - self.timing_start_time

    def get_remain_timing(self):
        # Usage: time_left = self.get_remain_timing()
        return self.timing_predicted_duration - self.get_run_timing()

    def timing_txt(self):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        time_left = self.get_remain_timing()

        hours = int(time_left / (60 * 60))
        minutes = int((time_left - hours * 60 * 60) / (60))
        seconds = time_left % 60

        # txt = '%s (ETC: %02d:%02d:%02d)' % (date_str, hours, minutes, seconds)
        left_str = str(datetime.timedelta(seconds=round(time_left, 0)))
        txt = "%s (%s)" % (date_str, left_str)

        return txt

    def timing_msg(self, priority=2, indent=0):
        txt = self.timing_txt()
        self.msg(txt, priority=priority, indent=indent)

    def timing_prediction_txt(self, predicted_duration):
        finish = time.time() + predicted_duration
        date_str = datetime.datetime.fromtimestamp(finish).strftime("%Y-%m-%d %H:%M:%S")

        time_left = predicted_duration

        hours = int(time_left / (60 * 60))
        minutes = int((time_left - hours * 60 * 60) / (60))
        seconds = time_left % 60

        left_str = str(datetime.timedelta(seconds=round(time_left, 0)))
        txt = "%s (%s)" % (date_str, left_str)

        return txt

    # LZA annealing protocols
    ########################################

    def single_sweep(self, length, velocity, delay_at_end=0.1):
        """LZA sweep by moving the sample."""

        velocity = round(velocity, 4)

        self.msg("sweep %.1f mm (@ %.4f mm/s)" % (length, velocity), priority=1, indent=1)

        sweep_time = abs(length / velocity)
        # self.stage.move_relative( +length, velocity=velocity )
        # time.sleep( sweep_time + delay_at_end )

        # self.stage.move_relative_wait_stop( +length, velocity=velocity )

        self.laserOn()
        self.xr(length, velocity=velocity, wait=False)
        time.sleep(sweep_time)
        self.laserOff()

        # Wait for stage to finish moving
        while smx.moving:
            time.sleep(0.05)

        time.sleep(delay_at_end)

    def single_sweep_laser(self, length, velocity, start=None, delay_at_end=0.1):
        """LZA sweep by moving the laser."""
        # move laser head in x direction

        if start == None:
            laserx_origin = laserx.position
        else:
            laserx_origin = start + laserx.position
            self.laserx_speedreset()
            laserx.move(laserx_origin)

        velocity = round(velocity, 4)

        self.msg("sweep %.1f mm (@ %.4f mm/s)" % (length, velocity), priority=1, indent=1)

        sweep_time = abs(length / velocity)
        # self.stage.move_relative( +length, velocity=velocity )
        # time.sleep( sweep_time + delay_at_end )

        # self.stage.move_relative_wait_stop( +length, velocity=velocity )

        self.laserOn()
        self.laserxr(length, velocity=velocity, wait=False)
        time.sleep(sweep_time)
        self.laserOff()

        # Wait for stage to finish moving
        while laserx.moving:
            time.sleep(0.05)

        time.sleep(delay_at_end)

    def double_sweep(self, length, velocity, delay_at_end=0.1):
        self.single_sweep(+length, velocity, delay_at_end=delay_at_end)
        self.single_sweep(-length, velocity, delay_at_end=delay_at_end)

    def double_sweep_laser(self, length, velocity, start=None, delay_at_end=0.1):
        self.single_sweep_laser(+length, velocity, start=start, delay_at_end=delay_at_end)
        self.single_sweep_laser(-length, velocity, start=None, delay_at_end=delay_at_end)

    def anneal_cyclic(self, length, velocity, num_cycles, delay_at_end=0.1):
        self.msg(
            "Cycling Anneal (%.1fmm X %.1f, @ %.4f mm/s)" % (length, num_cycles, velocity),
            1,
        )

        # Predict how long the anneal will take
        predicted_duration = ((length / velocity) + delay_at_end) * 2 * num_cycles  # Stage motion
        # predicted_duration += ( self.stage.com_wait_time*2 )*2*num_cycles # Serial communications
        self.start_timing(predicted_duration)

        # finish = time.time() + predicted_duration
        # date_str = datetime.datetime.fromtimestamp( finish ).strftime("%Y-%m-%d %H:%M:%S")
        self.msg(
            "Will finish at %s" % (self.timing_prediction_txt(predicted_duration)),
            indent=1,
        )

        fractional_sweeps = num_cycles - int(num_cycles)
        if fractional_sweeps == 0.5:
            num_cycles = int(num_cycles - fractional_sweeps)
        elif fractional_sweeps != 0.0:
            self.error("Cannot interpret number of cycles (%f)." % (num_cycles))

        # Anneal cycles
        for cycle in range(num_cycles):
            self.msg(
                "Cycle #: %d/%d (%.1f%% done)" % (cycle + 1, num_cycles, (100.0 * cycle / num_cycles)),
                1,
            )
            # self.date_stamp(indent=1)
            if self.is_timing():
                self.timing_msg(indent=1)

            self.double_sweep(length, velocity, delay_at_end=delay_at_end)

        # Final sweep (if any)
        if fractional_sweeps == 0.5:
            self.single_sweep(+length, velocity, delay_at_end=delay_at_end)

        self.end_timing()
        self.msg("Run complete (%s)" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def anneal_cyclic_laser(
        self,
        length,
        velocity,
        num_cycles,
        start=None,
        delay_at_end=0.1,
        x_step=0.05,
        measure_method=None,
        measures=None,
        cycles_done=0,
    ):
        self.msg(
            "Cycling Anneal (%.1fmm X %.1f, @ %.4f mm/s)" % (length, num_cycles, velocity),
            1,
        )

        if measures is None:
            measures = [1, 2, 4, 8, 16]

        if start == None:
            laserx_origin = laserx.position
        else:
            laserx_origin = start + laserx.position
            self.laserx_speedreset()
            laserx.move(laserx_origin)

        velocity = round(velocity, 4)
        # Predict how long the anneal will take
        predicted_duration = ((length / velocity) + delay_at_end) * 2 * num_cycles  # Stage motion
        # predicted_duration += ( self.stage.com_wait_time*2 )*2*num_cycles # Serial communications
        self.start_timing(predicted_duration)

        # finish = time.time() + predicted_duration
        # date_str = datetime.datetime.fromtimestamp( finish ).strftime("%Y-%m-%d %H:%M:%S")
        self.msg(
            "Will finish at %s" % (self.timing_prediction_txt(predicted_duration)),
            indent=1,
        )

        fractional_sweeps = num_cycles - int(num_cycles)
        if fractional_sweeps == 0.5:
            num_cycles = int(num_cycles - fractional_sweeps)
        elif fractional_sweeps != 0.0:
            self.error("Cannot interpret number of cycles (%f)." % (num_cycles))

        # Anneal cycles
        for cycle in range(num_cycles):
            self.msg(
                "Cycle #: %d/%d (%.1f%% done)" % (cycle + 1, num_cycles, (100.0 * cycle / num_cycles)),
                1,
            )
            # self.date_stamp(indent=1)
            if self.is_timing():
                self.timing_msg(indent=1)

            self.double_sweep_laser(length, velocity, start=None, delay_at_end=delay_at_end)

            # Do the GISAXS measurement
            if (cycle + 1 + cycles_done) in measures:
                if measure_method is None:
                    extra = "{:d}X".format(cycle + 1 + cycles_done)
                    self.LZA_measure(extra=extra, x_step=x_step)
                else:
                    self.measure_method()

        # Final sweep (if any)
        if fractional_sweeps == 0.5:
            self.single_sweep_laser(length, velocity, delay_at_end=delay_at_end)

        self.end_timing()
        self.msg("Run complete (%s)" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    def LZA_measure(self, extra=None, incident_angles=None, exposure_time=10, x_step=0.05):
        sam.xr(x_step)
        if incident_angles is None:
            # incident_angles = sam.incident_angles_default
            incident_angles = [0.12]
        sam.measureIncidentAngles(incident_angles, exposure_time=exposure_time, extra=extra)

    def single_sweep_measure(self, length=4, velocity=0.1, delay_at_end=0.1, exposure_time=1, extra=None):
        """LZA sweep by moving the sample."""

        velocity = round(velocity, 4)

        self.msg("sweep %.1f mm (@ %.4f mm/s)" % (length, velocity), priority=1, indent=1)

        sweep_time = abs(length / velocity)
        # self.stage.move_relative( +length, velocity=velocity )
        # time.sleep( sweep_time + delay_at_end )

        # self.stage.move_relative_wait_stop( +length, velocity=velocity )

        cms.modeMeasurement()
        sam.thabs(0.12)

        print("Laser on")
        self.laserOn()

        # self.xr(length, velocity=velocity, wait=False) # Move stage
        self.laserxr(-length, velocity=velocity, wait=False)  # Move laser
        start_time = time.time()

        # time.sleep(sweep_time)
        num_images = int((length / velocity) / (exposure_time + 0.3))
        sam.measure_burst(exposure_time=exposure_time, num_images=num_images, extra=extra)

        while (time.time() - start_time) < sweep_time:
            time.sleep(0.05)

        print("Laser off")
        self.laserOff()

        # Wait for stage to finish moving
        while smx.moving:
            time.sleep(0.05)

        time.sleep(delay_at_end)

    # Other measurement styles
    ########################################

    def thermal_gradient_T_conversion(self, position, power_fractional=1.00, T_ambient=25):
        T_est = 492 - abs(position) * 24.7  # P_frac = 0.45
        T_est = 528 - abs(position) * 26.6  # P_frac = 0.50
        T_est = 950 - abs(position) * 53.2  # P_frac = 1.00

        # T_excess = (T_est-T_ambient)*(power_fractional/0.45)
        # T_est = T_excess + T_ambient
        # print(T_est)
        # T_est = 831.24*power_fractional + 118.68 - (43.958*power_fractional + 4.9539)*position
        # print(T_est)

        return T_est

    def thermal_gradient_getAlignment(self, position):
        # For sam.x() = 0.0; state = {'origin': {'x': -41.0, 'y': 15.14, 'th': 0.3978: , 'phi': 0.0}}
        # For sam.x() = -14.0; state = {'origin': {'x': -41.0, 'y': 15.18, 'th': 0.3664, 'phi': 0.0}}

        x1, y1 = [0, 14.99]
        x2, y2 = [-14, 15.18]
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        return 0.0

    def thermal_gradient_measure(
        self,
        exposure_time=10,
        power_fractional=1.00,
        x_offset=0,
        x_step=-0.05,
        already_heated=0,
    ):
        testing = False

        positions = [0, -2, -4, -6]
        heating_times = [2] * 10

        cms.modeMeasurement()

        print("thermal gradient annealing using P_frac = {:.2f}".format(power_fractional))
        self.centerLaser()
        self.setLaserPower(68.5 * power_fractional)

        for i, heating_time in enumerate(heating_times):
            total_time = np.sum(heating_times[: i + 1]) + already_heated

            # Anneal sample
            print(" PTA will anneal sample for +{:.2f}s ({:.2f} s total)".format(heating_time, total_time))
            if not testing:
                sam.xo()
                # self.centerLaser()
                self.laserOn()
                time.sleep(heating_time)
                self.laserOff()

                time.sleep(10)

            if i in [0, 1, 4, 9]:
                # Measure spots
                for position in positions:
                    T_est = self.thermal_gradient_T_conversion(position + x_offset)
                    print("     Sample move to {:.4f} mm (T_est = {:.1f}°C)".format(position + x_offset, T_est))

                    extra = "P{:.2f}T{:.1f}Ct{:.1f}s".format(power_fractional, T_est, total_time)
                    print('      will append to filename: "{}"'.format(extra))

                    if not testing:
                        sam.xabs(position)
                        sam.xr(x_offset)

                        sam.measureIncidentAngle(0.12, exposure_time=exposure_time, extra=extra)

                x_offset += x_step

    # PTA stage/motion control
    ########################################

    def xabs(self, position, velocity=None, wait=True):
        if position > +124 or position < -124:
            print("ERROR: Cannot move sample x to that position")
            return

        if velocity is not None:
            caput("XF:11BMB-ES{PTA:Sample-Ax:X}Mtr.VELO", float(velocity))
        # smx.move(position)
        caput("XF:11BMB-ES{PTA:Sample-Ax:X}Mtr.VAL", position, wait=wait)

    def xr(self, amount, velocity=None, wait=True):
        self.xabs(smx.position + amount, velocity=velocity, wait=wait)

    def laserxabs(self, position, velocity=None, wait=True):
        if position > +124 or position < -124:
            print("ERROR: Cannot move sample x to that position")
            return

        if velocity is not None:
            caput("XF:11BMB-ES{PTA:Laser-Ax:X}Mtr.VELO", float(velocity))

        caput("XF:11BMB-ES{PTA:Laser-Ax:X}Mtr.VAL", position, wait=wait)

    def laserx_speedreset(self):
        caput("XF:11BMB-ES{PTA:Laser-Ax:X}Mtr.VELO", 5)

    def laserxr(self, amount, velocity=None, wait=True):
        self.laserxabs(laserx.position + amount, velocity=velocity, wait=wait)

    # def laserxabs(self, position):
    # if position>0 and position<+120:
    # laserx.move(position)
    # else:
    # print('ERROR: Cannot move laser x to that position')

    # def laserxr(self, amount):
    # self.laserxabs( laserx.position + amount )

    def laseryabs(self, position):
        if position > -21 and position <= 0:
            lasery.move(position)
        else:
            print("ERROR: Cannot move laser y to that position")

    def laseryr(self, amount):
        self.laseryabs(lasery.position + amount)

    def loadSample(self):
        print("1. Turning laser off")
        self.laserOff()

        print("2. Moving sample stage to loading position")
        self.xabs(+120.0, velocity=10)

        print("3. User: Set vacuum valve from 'pump' to 'vent'")
        print("4. User: Open laser box")

    def startSample(self):
        self.laserOff()
        self.xabs(-75, velocity=10)
        self.centerLaser()

    def centerLaser(self):
        self.laserxabs(-67.1, velocity=10)

    def checkLaser(self, duration=2, power_W=5):
        self.setLaserPower(power_W)
        self.laserPulse(duration)


pta = PhotoThermalAnnealer()
