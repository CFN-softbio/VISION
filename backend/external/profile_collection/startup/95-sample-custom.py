################################################################################
# Code for customer-made holders and sample stages,
# Samples include :
# 1. SampleTSAXS_Generic : TSAXS/TWAXS
# 2. SampleGISAXS_Generic : GISAXS/GIWAXS
# 3. SampleXR_WAXS : XR and th-2th scan
# 4. SampleCDSAXS_Generic : CD SAXS/GISAXS
# 5. SampleGonio_Generic : SmartAct Goniometer
# 6. SampleSecondStage : 2nd sample stage

# Holders include :
# 1. CapillaryHolder : TSAXS
# 2. GIBar : GISAXS/GIWAXS bar
# 3. CapillaryHolderHeated :  heat stage for transmission
# 4. GIBar_long_thermal : heat stage for GI
# 5. WellPlateHolder: well plate
# 6. PaloniThermalStage: special stage made in brass by UCSB group.
# 7. DSCStage : tranmission stage for DSC capsules
# 8. InstecStage60 : HCS 60, From Sam Sprunt (Kent State U. )
# 9. InstecStage402 : HCS 402, beamline owned
# 10.OffCenteredHoder: GTSAXS
# 11.HumidityStage: GI humidity stage
# 12.HumidityTransmissionStage: Trans humidity stage
# 13.GIBar_Linkam : Linkam GI holder with N2 protective dome

# TODO: some clean-ups.
# 1. make rotational stage as custom stage with phi, strans, strans2, tilt, tilt2 stages.
#
################################################################################


# custmer-made samples.
class SampleTSAXS_Generic(Sample_Generic):
    ################# Direct beam transmission measurement ####################
    def intMeasure(self, output_file, exposure_time):
        """Measure the transmission intensity of the sample by ROI4.
        The intensity will be saved in output_file
        """
        if abs(beam.energy(verbosity=0) - 13.5) < 0.1:
            beam.setAbsorber(4)
        elif abs(beam.energy(verbosity=0) - 17) < 0.1:
            beam.setAbsorber(6)

        print("Absorber is moved to position {}".format(beam.absorber()[0]))

        detselect([pilatus2M])
        if beam.absorber()[0] >= 4:
            bsx.move(bsx.position + 6)
            beam.setTransmission(1)

        self.measure(exposure_time)

        temp_data = self.transmission_data_output(beam.absorber()[0])

        cms.modeMeasurement()
        # beam.setAbsorber(0)
        beam.absorber_out()

        # output_data = output_data.iloc[0:0]

        # create a data file to save the INT data
        INT_FILENAME = "{}/data/{}.csv".format(os.path.dirname(__file__), output_file)

        if os.path.isfile(INT_FILENAME):
            output_data = pds.read_csv(INT_FILENAME, index_col=0)
            # output_data = output_data.append(temp_data, ignore_index=True)
            output_data = pds.concat([output_data, temp_data], ignore_index=True)
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
            "f_absorber_ratio": beam.absorber()[1],
            "g_exposure_seconds": exposure_time,
        }

        return pds.DataFrame(data=current_data)


class SampleGISAXS_Generic(Sample_Generic):
    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)
        self.naming_scheme = ["name", "extra", "th", "exposure_time"]
        self.incident_angles_default = [0.08, 0.10, 0.12, 0.15, 0.20]
        self.measure_setting = {}
        self.alignDone = False

    def measureSpots(
        self,
        num_spots=2,
        translation_amount=0.1,
        axis="x",
        exposure_time=None,
        extra=None,
        measure_type="measureSpots",
        **md,
    ):
        super().measureSpots(
            num_spots=num_spots,
            translation_amount=translation_amount,
            axis=axis,
            exposure_time=exposure_time,
            extra=extra,
            measure_type=measure_type,
            **md,
        )

    def measureIncidentAngle(self, angle, exposure_time=None, extra=None, tiling=None, **md):
        self.thabs(angle)
        while sth.moving == True:
            time.sleep(0.1)
        self.measure(exposure_time=exposure_time, extra=extra, tiling=tiling, **md)

    def measureIncidentAngles(self, angles=None, exposure_time=None, extra=None, tiling=None, **md):
        # measure the incident angles first and then change the tiling features.
        if angles is None:
            angles = self.incident_angles_default
        for angle in angles:
            self.measureIncidentAngle(angle, exposure_time=exposure_time, extra=extra, tiling=tiling, **md)

    def measureIncidentAngles_Stitch(
        self, angles=None, exposure_time=None, extra=None, tiling=None, verbosity=3, **md
    ):
        # measure the incident angles first and then change the tiling features.
        if tiling == None:
            if angles is None:
                angles = self.incident_angles_default
            for angle in angles:
                self.measureIncidentAngle(angle, exposure_time=exposure_time, extra=extra, tiling=tiling, **md)

        elif tiling == "ygaps":
            if angles is None:
                angles = self.incident_angles_default
            # pos1
            for angle in angles:
                self.thabs(angle)
                while sth.moving == True:
                    time.sleep(0.1)
                time.sleep(0.5)
                extra_current = "pos1" if extra is None else "{}_pos1".format(extra)
                md["detector_position"] = "lower"
                self.measure_single(
                    exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
                )

            # pos2
            SAXSy_o = SAXSy.user_readback.value
            SAXSx_o = SAXSx.user_readback.value
            WAXSy_o = WAXSy.user_readback.value
            WAXSx_o = WAXSx.user_readback.value
            MAXSy_o = MAXSy.user_readback.value
            if pilatus2M in cms.detector:
                SAXSy.move(SAXSy_o + 5.16)
            if pilatus800 in cms.detector:
                WAXSy.move(WAXSy_o + 5.16)
            if pilatus8002 in cms.detector:
                MAXSy.move(MAXSy_o + 5.16)

            for angle in angles:
                self.thabs(angle)
                while sth.moving == True:
                    time.sleep(0.1)
                time.sleep(0.5)

                extra_current = "pos2" if extra is None else "{}_pos2".format(extra)
                md["detector_position"] = "upper"
                self.measure_single(
                    exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
                )

            if SAXSy.user_readback.value != SAXSy_o:
                SAXSy.move(SAXSy_o)
            if WAXSy.user_readback.value != WAXSy_o:
                WAXSy.move(WAXSy_o)
            if MAXSy.user_readback.value != MAXSy_o:
                MAXSy.move(MAXSy_o)

        elif tiling == "xygaps":
            if angles is None:
                angles = self.incident_angles_default
            # pos1
            for angle in angles:
                self.thabs(angle)
                time.sleep(0.5)
                extra_current = "pos1" if extra is None else "{}_pos1".format(extra)
                md["detector_position"] = "lower_left"
                self.measure_single(
                    exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
                )

            # pos2
            SAXSy_o = SAXSy.user_readback.value
            SAXSx_o = SAXSx.user_readback.value
            WAXSy_o = WAXSy.user_readback.value
            WAXSx_o = WAXSx.user_readback.value
            # MAXSy_o = MAXSy.user_readback.value

            for angle in angles:
                self.thabs(angle)
                time.sleep(0.2)
                if pilatus2M in cms.detector:
                    SAXSy.move(SAXSy_o + 5.16)
                if pilatus800 in cms.detector:
                    WAXSy.move(WAXSy_o + 5.16)
                if pilatus300 in cms.detector:
                    MAXSy.move(MAXSy_o + 5.16)

                extra_current = "pos2" if extra is None else "{}_pos2".format(extra)
                md["detector_position"] = "upper"
                self.measure_single(
                    exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
                )

            # pos4  #comment out to save time
            for angle in angles:
                self.thabs(angle)
                time.sleep(0.2)

                if pilatus2M in cms.detector:
                    SAXSx.move(SAXSx_o + 5.16)
                    SAXSy.move(SAXSy_o + 5.16)
                if pilatus800 in cms.detector:
                    WAXSx.move(WAXSx_o - 5.16)
                    WAXSy.move(WAXSy_o + 5.16)
                extra_current = "pos4" if extra is None else "{}_pos4".format(extra)
                md["detector_position"] = "upper_right"
                self.measure_single(
                    exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
                )

            # pos3
            for angle in angles:
                self.thabs(angle)
                time.sleep(0.2)

                if pilatus2M in cms.detector:
                    SAXSx.move(SAXSx_o + 5.16)
                    SAXSy.move(SAXSy_o)
                if pilatus800 in cms.detector:
                    WAXSx.move(WAXSx_o - 5.16)
                    WAXSy.move(WAXSy_o)

                extra_current = "pos3" if extra is None else "{}_pos3".format(extra)
                md["detector_position"] = "lower_right"
                self.measure_single(
                    exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
                )

            if WAXSx.user_readback.value != WAXSx_o:
                WAXSx.move(WAXSx_o)
            if WAXSy.user_readback.value != WAXSy_o:
                WAXSy.move(WAXSy_o)

            if SAXSx.user_readback.value != SAXSx_o:
                SAXSx.move(SAXSx_o)
            if SAXSy.user_readback.value != SAXSy_o:
                SAXSy.move(SAXSy_o)

    ################# Direct beam transmission measurement ####################
    def intMeasure(self, output_file, exposure_time):
        """Measure the transmission intensity of the sample by ROI4.
        The intensity will be saved in output_file
        """
        if abs(beam.energy(verbosity=0) - 13.5) < 0.1:
            beam.setAbsorber(4)
        elif abs(beam.energy(verbosity=0) - 17) < 0.1:
            beam.setAbsorber(6)

        print("Absorber is moved to position {}".format(beam.absorber()[0]))

        detselect([pilatus2M])
        if beam.absorber()[0] >= 4:
            bsx.move(bsx.position + 6)
            beam.setTransmission(1)

        self.measure(exposure_time)

        temp_data = self.transmission_data_output(beam.absorber()[0])

        cms.modeMeasurement()
        # beam.setAbsorber(0)
        beam.absorber_out()

        # output_data = output_data.iloc[0:0]

        # create a data file to save the INT data
        INT_FILENAME = "{}/data/{}.csv".format(os.path.dirname(__file__), output_file)

        if os.path.isfile(INT_FILENAME):
            output_data = pds.read_csv(INT_FILENAME, index_col=0)
            # output_data = output_data.append(temp_data, ignore_index=True)
            output_data = pds.concat([output_data, temp_data], ignore_index=True)
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
            "f_absorber_ratio": beam.absorber()[1],
            "g_exposure_seconds": exposure_time,
        }

        return pds.DataFrame(data=current_data)

    def _alignOld(self, step=0):
        """Align the sample with respect to the beam. GISAXS alignment involves
        vertical translation to the beam center, and rocking theta to get the
        sample plane parralel to the beam.

        The 'step' argument can optionally be given to jump to a particular
        step in the sequence."""

        # TODO: Deprecate and delete

        if step <= 0:
            # TODO: Check what mode we are in, change if necessary...
            # get_beamline().modeAlignment()
            beam.on()

        # TODO: Improve implementation
        if step <= 2:
            # fit_scan(smy, 2.6, 35, fit='HM')
            fit_scan(smy, 2.6, 35, fit="sigmoid_r")

        if step <= 4:
            # fit_scan(smy, 0.6, 17, fit='HM')
            fit_scan(smy, 0.6, 17, fit="sigmoid_r")
            fit_scan(sth, 1.2, 21, fit="max")

        # if step<=6:
        #    fit_scan(smy, 0.3, 17, fit='sigmoid_r')
        #    fit_scan(sth, 1.2, 21, fit='COM')

        if step <= 8:
            fit_scan(smy, 0.2, 17, fit="sigmoid_r")
            fit_scan(sth, 0.8, 21, fit="gauss")

        if step <= 9:
            # self._testing_refl_pos()
            # movr(sth,.1)
            # fit_scan(sth, 0.2, 41, fit='gauss')
            # fit_scan(smy, 0.2, 21, fit='gauss')
            # movr(sth,-.1)

            beam.off()


    def align(self, step=0, reflection_angle=0.15, verbosity=3):
        '''Align the sample with respect to the beam. GISAXS alignment involves
        vertical translation to the beam center, and rocking theta to get the
        sample plane parralel to the beam. Finally, the angle is re-optimized
        in reflection mode.
        
        The 'step' argument can optionally be given to jump to a particular
        step in the sequence.'''
        start_time = time.time()
        alignment = 'Success'
        initial_y = smy.position
        initial_th = sth.position

        align_crazy = self.swing(reflection_angle=reflection_angle)
        crazy_y = smy.position
        crazy_th = sth.position
        cms.setDirectBeamROI()

        if  align_crazy[0] == False:
            alignment = 'Failed'
            if step<=4:
                if verbosity>=4:
                    print('    align: fitting')
                
                fit_scan(smy, 1.2, 21, fit='HMi')
                ##time.sleep(2)
                fit_scan(sth, 1.5, 21, fit='max')
                ##time.sleep(2)            
                
            #if step<=5:
            #    #fit_scan(smy, 0.6, 17, fit='sigmoid_r')
            #    fit_edge(smy, 0.6, 17)
            #    fit_scan(sth, 1.2, 21, fit='max')


            if step<=8:
                #fit_scan(smy, 0.3, 21, fit='sigmoid_r')
                
                fit_edge(smy, 0.6, 21)
                #time.sleep(2)
                #fit_edge(smy, 0.4, 21)
                fit_scan(sth, 0.8, 21, fit='COM')
                #time.sleep(2)            
                self.setOrigin(['y', 'th'])
        
        
            if step<=9 and reflection_angle is not None:
                # Final alignment using reflected beam
                if verbosity>=4:
                    print('    align: reflected beam')
                get_beamline().setReflectedBeamROI(total_angle=reflection_angle*2.0)
                #get_beamline().setReflectedBeamROI(total_angle=reflection_angle*2.0, size=[12,2])
                
                self.thabs(reflection_angle)
                
                result = fit_scan(sth, 0.4, 41, fit='max') 
                #result = fit_scan(sth, 0.2, 81, fit='max') #it's useful for alignment of SmarAct stage
                sth_target = result.values['x_max']-reflection_angle
                
                if result.values['y_max']>50:
                    th_target = self._axes['th'].motor_to_cur(sth_target)
                    self.thsetOrigin(th_target)

                #fit_scan(smy, 0.2, 21, fit='max')
                #self.setOrigin(['y'])            

            if step<=10:
                self.thabs(0.0)
                beam.off()

        ### save the alignment information
        align_time = time.time() - start_time

        current_data = {'a_sample': self.name,
                        'b_quick_alignment': alignment, 
                        'c_align_time': align_time, 
                        'd_offset_y': smy.position - initial_y,
                        'e_offset_th': sth.position - initial_th, 
                        'f_crazy_offset_y': smy.position - crazy_y,
                        'g_crazy_offset_th': sth.position - crazy_th, 
                        'h_search_no': align_crazy[1]}
        
        temp_data = pds.DataFrame([current_data])

        INT_FILENAME='{}/data/{}.csv'.format(os.path.dirname(__file__) , 'alignment_results.csv')            
    
        if os.path.isfile(INT_FILENAME):
            output_data = pds.read_csv(INT_FILENAME, index_col=0)
            output_data = pds.concat([output_data, temp_data])    
            output_data.to_csv(INT_FILENAME)
        else:
            temp_data.to_csv(INT_FILENAME)

    def swing(self, step=0, reflection_angle=0.12, ROI_size=[10, 180], th_range=0.3, int_threshold=10, verbosity=3):

        #setting parameters
        rel_th = 1
        ct = 0
        cycle = 0
        intenisty_threshold = 10

        #re-assure the 3 ROI positon
        get_beamline().setDirectBeamROI()
        get_beamline().setReflectedBeamROI(total_angle=reflection_angle*2)

        #set ROI2 as a fixed area
        get_beamline().setROI2ReflectBeamROI(total_angle=reflection_angle*2, size=ROI_size)
        pilatus2M.roi2.size.y.set(190)
        pilatus2M.roi2.min_xyz.min_y.set(852)


        #def ROI3 in 160pixels with the center located at reflection beam
        # get_beamline().setReflectedBeamROI(total_angle = reflection_angle*2, size=ROI_size) #set ROI3

        # self.thabs(reflection_angle)
        if verbosity>=4:
            print('  Aligning {}'.format(self.name))
        
        # if step<=0:
        #     # Prepare for alignment
            
        #     if RE.state!='idle':
        #         RE.abort()
                
        #     if get_beamline().current_mode!='alignment':
        #         #if verbosity>=2:
        #             #print("WARNING: Beamline is not in alignment mode (mode is '{}')".format(get_beamline().current_mode))
        #         print("Switching to alignment mode (current mode is '{}')".format(get_beamline().current_mode))
        #         get_beamline().modeAlignment()
                
                
            get_beamline().setDirectBeamROI()
            
            beam.on()

        
        if step<=2:
            # if verbosity>=4:
            #     print('    align: searching')
                
            # Estimate full-beam intensity
            value = None
            if True:
                # You can eliminate this, in which case RE.md['beam_intensity_expected'] is used by default
                self.yr(-2)
                #detector = gs.DETS[0]
                detector = get_beamline().detector[0]
                # value_name = get_beamline().TABLE_COLS[0]
                beam.on()
                RE(count([detector]))
                value = detector.read()['pilatus2M_stats4_total']['value']
                self.yr(2)
            
            # if 'beam_intensity_expected' in RE.md:
            #     if value<RE.md['beam_intensity_expected']*0.75:
            #         print('WARNING: Direct beam intensity ({}) lower than it should be ({})'.format(value, RE.md['beam_intensity_expected']))
                
            #check the last value:
            # value=20000
            ii = 0
            while abs(pilatus2M.stats4.total.get() - value)/value < 0.1 and ii < 3: 
                ii += 1
                # Find the step-edge
                RE(self.search_plan(motor=smy, step_size=.1, min_step=0.01, target=0.5, intensity=20000, polarity=-1,detector_suffix='_stats4_total'))                
                # Find the peak
                # self.thsearch(step_size=0.2, min_step=0.01, target='max', verbosity=verbosity)
                RE(self.search_plan(motor=sth, step_size=.2, min_step=0.01, target='max',detector_suffix='_stats4_total'))
            #last check for height
            # self.ysearch(step_size=0.05, min_step=0.005, intensity=value, target=0.5, verbosity=verbosity, polarity=-1)
            RE(self.search_plan(motor=smy, step_size=0.05, min_step=0.005, target=0.5, intensity=20000, polarity=-1,detector_suffix='_stats4_total'))
                   
        #check reflection beam
        self.thr(reflection_angle)
        RE(count([detector]))
        
        if abs(detector.stats2.max_xy.get().y - detector.stats2.centroid.get().y) < 20 and detector.stats2.max_value.get() > intenisty_threshold:

            #continue the fast alignment 
            print('The reflective beam is found! Continue the fast alignment')
            
            while abs(rel_th) > 0.005 and ct < 5:
            # while detector.roi3.max_value.get() > 50 and ct < 5:
                
                #absolute beam position 
                refl_beam = detector.roi2.min_xyz.min_y.get() + detector.stats2.max_xy.y.get()

                #roi3 position
                roi3_beam = detector.roi3.min_xyz.min_y.get() + detector.roi3.size.y.get()/2

                #distance from current postion to the center of roi2 (the disired rel beam position)
                # rel_ypos = detector.stats2.max_xy.get().y - detector.stats2.size.get().y
                rel_ypos = refl_beam - roi3_beam

                rel_th = rel_ypos/get_beamline().SAXS.distance/1000*0.172/np.pi*180/2
                
                print('The th offset is {}'.format(rel_th))
                self.thr(rel_th)
                
                ct += 1
                RE(count([detector]))

            if detector.stats3.total.get()>50:

                print('The fast alignment works!')
                self.thr(-reflection_angle)

    
                self.setOrigin(['y', 'th'])

                beam.off()

                return True, ii

            else:
                print('Alignment Error: Cannot Locate the reflection beam')
                self.thr(-reflection_angle)
                beam.off()

                return False, ii


        elif abs(detector.stats2.max_xy.get().y - detector.stats2.centroid.get().y) > 5:
            print('Max and Centroid dont Match!')

            #perform the full alignment
            print('Alignment Error: No reflection beam is found!')
            self.thr(-reflection_angle)
            beam.off()
            return False, ii

        else:
            print('Intensiy < threshold!')

            #perform the full alignment
            print('Alignment Error: No reflection beam is found!')
            self.thr(-reflection_angle)
            beam.off()
            return False, ii

    def search_plan(self, motor=smy, step_size=1.0, min_step=0.05, intensity=None, target=0.5, detector=None, detector_suffix=None, polarity=+1, verbosity=3):
        '''Moves this axis, searching for a target value.
        
        Parameters
        ----------
        step_size : float
            The initial step size when moving the axis
        min_step : float
            The final (minimum) step size to try
        intensity : float
            The expected full-beam intensity readout
        target : 0.0 to 1.0
            The target ratio of full-beam intensity; 0.5 searches for half-max.
            The target can also be 'max' to find a local maximum.
        detector, detector_suffix
            The beamline detector (and suffix, such as '_stats4_total') to trigger to measure intensity
        polarity : +1 or -1
            Positive motion assumes, e.g. a step-height 'up' (as the axis goes more positive)
        '''
        print("HERE!!")
        
        if detector is None:
            #detector = gs.DETS[0]
            detector = get_beamline().detector[0]
        if detector_suffix is None:
            #value_name = gs.TABLE_COLS[0]
            value_name = get_beamline().TABLE_COLS[0]
        else:
            value_name = detector.name + detector_suffix

        print(f"detector={detector}")

        @bpp.stage_decorator([detector])
        @bpp.run_decorator(md={})
        def inner_search():
            nonlocal intensity, target, step_size

            if not get_beamline().beam.is_on():
                print('WARNING: Experimental shutter is not open.')
            
            
            if intensity is None:
                intensity = RE.md['beam_intensity_expected']

            # bec.disable_table()
            
            
            # Check current value
            vv = yield from bps.trigger_and_read([detector, motor])
            value = vv[value_name]['value']
            # RE(count([detector]))
            # value = detector.read()[value_name]['value']


            if target == 'max':
                
                if verbosity>=5:
                    print("Performing search on axis '{}' target is 'max'".format(self.name))
                
                max_value = value
                # max_position = self.get_position(verbosity=0)
                
                
                direction = +1*polarity
                
                while step_size>=min_step:
                    if verbosity>=4:
                        print("        move {} by {} × {}".format(self.name, direction, step_size))

                    #  pos = yield from bps.rd(motor)
                    yield from bps.mvr(motor, direction*step_size)
                    # self.move_relative(move_amount=direction*step_size, verbosity=verbosity-2)

                    prev_value = value
                    yield from bps.trigger_and_read([detector, motor])
                    # RE(count([detector]))
                    
                    value = detector.read()[value_name]['value']
                    # if verbosity>=3:
                    #     print("      {} = {:.3f} {}; value : {}".format(self.name, self.get_position(verbosity=0), self.units, value))
                        
                    if value>max_value:
                        max_value = value
                        # max_position = self.get_position(verbosity=0)
                        
                    if value>prev_value:
                        # Keep going in this direction...
                        pass
                    else:
                        # Switch directions!
                        direction *= -1
                        step_size *= 0.5
                    
                    
            elif target == 'min':
                
                if verbosity>=5:
                    print("Performing search on axis '{}' target is 'min'".format(self.name))
                
                direction = +1*polarity
                
                while step_size>=min_step:
                    if verbosity>=4:
                        print("        move {} by {} × {}".format(self.name, direction, step_size))

                    # pos = yield from bps.rd(motor)
                    yield from bps.mvr(motor, direction*step_size)
                    # self.move_relative(move_amount=direction*step_size, verbosity=verbosity-2)

                    prev_value = value
                    yield from bps.trigger_and_read([detector, motor])
                    # RE(count([detector]))
                    value = detector.read()[value_name]['value']
                    if verbosity>=3:
                        print("      {} = {:.3f} {}; value : {}".format(self.name, self.get_position(verbosity=0), self.units, value))
                        
                    if value<prev_value:
                        # Keep going in this direction...
                        pass
                    else:
                        # Switch directions!
                        direction *= -1
                        step_size *= 0.5
                                    
            else:

                target_rel = target
                target = target_rel*intensity

                if verbosity>=5:
                    print("Performing search on axis '{}' target {} × {} = {}".format(self.name, target_rel, intensity, target))
                if verbosity>=4:
                    print("      value : {} ({:.1f}%)".format(value, 100.0*value/intensity))
                
                
                # Determine initial motion direction
                if value>target:
                    direction = -1*polarity
                else:
                    direction = +1*polarity
                    
                while step_size>=min_step:
                    
                    if verbosity>=4:
                        print("        move {} by {} × {}".format(self.name, direction, step_size))
                    
                    # pos = yield from bps.rd(motor)
                    yield from bps.mvr(motor, direction*step_size)
                    # self.move_relative(move_amount=direction*step_size, verbosity=verbosity-2)
                    
                    yield from bps.trigger_and_read([detector, motor])
                    # RE(count([detector]))
                    value = detector.read()[value_name]['value']
                    #if verbosity>=3:
                    #    print("      {} = {:.3f} {}; value : {} ({:.1f}%)".format(self.name, self.get_position(verbosity=0), self.units, value, 100.0*value/intensity))
                        
                    # Determine direction
                    if value>target:
                        new_direction = -1.0*polarity
                    else:
                        new_direction = +1.0*polarity
                        
                    if abs(direction-new_direction)<1e-4:
                        # Same direction as we've been going...
                        # ...keep moving this way
                        pass
                    else:
                        # Switch directions!
                        direction *= -1
                        step_size *= 0.5
        
            # bec.enable_table()

        yield from inner_search()


    def _test_align(self, step=0, reflection_angle=0.12, verbosity=3):
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
                # if verbosity>=2:
                # print("WARNING: Beamline is not in alignment mode (mode is '{}')".format(get_beamline().current_mode))
                print("Switching to alignment mode (current mode is '{}')".format(get_beamline().current_mode))
                get_beamline().modeAlignment()
            get_beamline().setDirectBeamROI()
        # if step<=2:
        # if verbosity>=4:
        # print('    align: searching')
        # beam.on()
        ## Estimate full-beam intensity
        # value = None
        # if True:
        ## You can eliminate this, in which case RE.md['beam_intensity_expected'] is used by default
        # self.yr(-2)
        ##detector = gs.DETS[0]
        # detector = get_beamline().detector[0]
        # value_name = get_beamline().TABLE_COLS[0]
        # RE(count([detector]))
        # value = detector.read()[value_name]['value']
        # self.yr(+2)

        # if 'beam_intensity_expected' in RE.md and value<RE.md['beam_intensity_expected']*0.75:
        # print('WARNING: Direct beam intensity ({}) lower than it should be ({})'.format(value, RE.md['beam_intensity_expected']))

        ## Find the step-edge
        # self.ysearch(step_size=0.5, min_step=0.005, intensity=value, target=0.5, verbosity=verbosity, polarity=-1)

        ## Find the peak
        # self.thsearch(step_size=0.4, min_step=0.01, target='max', verbosity=verbosity)

        if step <= 4:
            if verbosity >= 4:
                print("    align: fitting")

            fit_scan(smy, 1.2, 21, fit="HMi")
            ##time.sleep(2)
            fit_scan(sth, 1.5, 21, fit="max")
            ##time.sleep(2)

        # if step<=8 and reflection_angle==None:
        # fit_edge(smy, 0.6, 21)
        ##time.sleep(2)
        # fit_scan(sth, 0.8, 21, fit='COM')
        # self.setOrigin(['y', 'th'])

        if step <= 9 and reflection_angle is not None:
            # Final alignment using reflected beam
            if verbosity >= 4:
                print("    align: reflected beam")
            get_beamline().setReflectedBeamROI(total_angle=reflection_angle * 2.0)
            # sth scan
            self.thabs(reflection_angle)
            result = fit_scan(sth, 0.4, 41, fit="max")
            sth_target = result.values["x_max"] - reflection_angle

            if result.values["y_max"] > 50:
                th_target = self._axes["th"].motor_to_cur(sth_target)
                self.thsetOrigin(th_target)
            # y scan
            fit_scan(smy, 0.2, 21, fit="max")
            self.setOrigin(["y"])

        if step <= 10:
            self.thabs(0.0)
            beam.off()

    def alignQuick(self, align_step=8, reflection_angle=0.08, verbosity=3):
        get_beamline().modeAlignment()
        # self.yo()
        self.tho()
        beam.on()
        self.align(step=align_step, reflection_angle=reflection_angle, verbosity=verbosity)

    def level(self, step=0, pos_x_left=-5, pos_x_right=5):
        # TODO: Move this code. (This should be a property of the GIBar object.)

        # level sample by checking bar height at pos_left and pos_right
        print("checking the level of Sample")
        if step <= 1:
            cms.modeAlignment()
            self.tho()
            self.yo()

        self.xabs(pos_x_left)
        # beam.on()
        fit_edge(smy, 0.6, 17)  # it's better not to move smy after scan but only the center position
        time.sleep(1)
        pos_y_left = smy.position
        # pos_y_left=smy.user_readback.value
        print("BEFORE LEVEL: pos_y_left = {}".format(pos_y_left))

        self.xabs(pos_x_right)
        fit_edge(smy, 0.6, 17)  # it's better not to move smy after scan but only the center position
        time.sleep(1)
        pos_y_right = smy.position
        print("BEFORE LEVEL: pos_y_right = {}".format(pos_y_right))

        offset_schi = (pos_y_right - pos_y_left) / (pos_x_right - pos_x_left) * 180 / np.pi
        print("The schi offset is {} degrees".format(offset_schi))
        schi.move(schi.position - offset_schi)

        # double-check the chi offset
        self.xabs(pos_x_left)
        fit_edge(smy, 0.6, 17)  # it's better not to move smy after scan but only the center position
        time.sleep(1)
        pos_y_left = smy.position
        print("AFTER LEVEL: pos_y_left = {}".format(pos_y_left))

        self.xabs(pos_x_right)
        fit_edge(smy, 0.6, 17)  # it's better not to move smy after scan but only the center position
        time.sleep(1)
        pos_y_right = smy.position
        print("AFTER LEVEL: pos_y_right = {}".format(pos_y_right))
        # beam.off()

        self.xo()
        offset_schi = (pos_y_right - pos_y_left) / (pos_x_right - pos_x_left) * 180 / np.pi

        if offset_schi <= 0.1:
            print("schi offset is aligned successfully!")

        else:
            print("schi offset is WRONG. Please redo the level command")
        fit_edge(smy, 0.6, 17)  # it's better not to move smy after scan but only the center position
        self.setOrigin(["y"])

    def do(self, step=0, align_step=0, **md):
        if step <= 1:
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
            # detselect([pilatus300, psccd])
            # detselect(psccd)
            # detselect(pilatus300)
            detselect(pilatus2M)
            for detector in get_beamline().detector:
                if detector.name == "pilatus2M":
                    RE(detector.setExposureTime(self.md["exposure_time"]))
                else:
                    detector.setExposureTime(self.md["exposure_time"])
            self.measureIncidentAngles(self.incident_angles_default, **md)
            self.thabs(0.0)

    def backup_do_SAXS(self, step=0, align_step=0, measure_setting=None, **md):
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
            if measure_setting == None:
                measure_setting = measure_setting

            if self.incident_angles == None:
                incident_angles = self.incident_angles_default
            else:
                incident_angles = self.incident_angles
            self.measureIncidentAngles_Stitch(incident_angles, exposure_time=self.SAXS_time, tiling="ygaps", **md)

    def do_SAXS(self, step=0, align_step=0, measure_setting=None, **md):
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
            self.alignDone = True

        if step <= 10:
            if self.measure_setting["incident_angles"] == None:
                incident_angles = self.incident_angles_default
            else:
                incident_angles = self.measure_setting["incident_angles"]
            if self.measure_setting["exposure_time"] == None:
                exposure_time = self.SAXS_time
            else:
                if type(self.measure_setting["exposure_time"]) == list:
                    exposure_time = self.measure_setting["exposure_time"][0]
                else:
                    exposure_time = self.measure_setting["exposure_time"]
            tiling = self.measure_setting["tiling"]

            self.measureIncidentAngles_Stitch(incident_angles, exposure_time=exposure_time, tiling=tiling, **md)

    def do_WAXS_only(self, step=0, align_step=0, **md):
        if step < 5:
            self.xo()
            self.yo()
            self.tho()
            get_beamline().modeMeasurement()
        if step <= 10:
            if self.measure_setting["incident_angles"] == None:
                incident_angles = self.incident_angles_default
            else:
                incident_angles = self.measure_setting["incident_angles"]
            if self.measure_setting["exposure_time"] == None:
                exposure_time = self.SAXS_time
            else:
                if type(self.measure_setting["exposure_time"]) == list:
                    exposure_time = self.measure_setting["exposure_time"][0]
                else:
                    exposure_time = self.measure_setting["exposure_time"]
            tiling = self.measure_setting["tiling"]
            waxs_on()
            self.measureIncidentAngles_Stitch(incident_angles, exposure_time=exposure_time, tiling=tiling, **md)
            self.thabs(0.0)

    def _backup_do_WAXS(self, step=0, align_step=0, **md):
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

            waxs_on()  # edited from waxs_on 3/25/19 through a saxs_on error
            # for detector in get_beamline().detector:
            # detector.setExposureTime(self.MAXS_time)
            self._test2_measureIncidentAngles(incident_angles, exposure_time=self.WAXS_time, tiling="ygaps", **md)

            # if self.exposure_time_MAXS==None:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.MAXS_time, tiling=self.tiling, **md)
            # else:
            # self.measureIncidentAngles(incident_angles, exposure_time=self.exposure_time_MAXS, tiling=self.tiling, **md)
            self.thabs(0.0)

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
            self.align(step=align_step, reflection_angle=0.12)
            # self.setOrigin(['y','th']) # This is done within align

        # if step<=7:
        # self.xr(0.2)

        if step <= 8:
            get_beamline().modeMeasurement()

        if step <= 10:
            if self.measure_setting["incident_angles"] == None:
                incident_angles = self.incident_angles_default
            else:
                incident_angles = self.measure_setting["incident_angles"]
            if self.measure_setting["exposure_time"] == None:
                exposure_time = self.SAXS_time
            else:
                if type(self.measure_setting["exposure_time"]) == list:
                    exposure_time = self.measure_setting["exposure_time"][0]
                else:
                    exposure_time = self.measure_setting["exposure_time"]
            tiling = self.measure_setting["tiling"]
            waxs_on()
            self.measureIncidentAngles_Stitch(incident_angles, exposure_time=exposure_time, tiling=tiling, **md)
            self.thabs(0.0)


class SampleCDSAXS_Generic(Sample_Generic):
    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)
        self.naming_scheme = ["name", "extra", "phi", "exposure_time"]
        self.rot_angles_default = np.arange(-45, +45 + 1, +1)
        # self.rot_angles_default = np.linspace(-45, +45, num=90, endpoint=True)

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

    def measureAngle(self, angle, exposure_time=None, extra=None, measure_type="measure", **md):
        self.phiabs(angle)
        self.measure(exposure_time=exposure_time, extra=extra, measure_type=measure_type, **md)

    def measureAngles(self, angles=None, exposure_time=None, extra=None, measure_type="measureAngles", **md):
        if angles is None:
            angles = self.rot_angles_default
        for angle in angles:
            self.measureAngle(angle, exposure_time=exposure_time, extra=extra, measure_type=measure_type, **md)


class SampleXR_WAXS(SampleGISAXS_Generic):
    ################# Specular reflectivity (XR) measurement ####################

    def XR_scan(
        self,
        scan_type="theta_scan",
        theta_range=[0, 1.6],
        theta_delta=0.1,
        theta_list=None,
        qz_list=None,
        roi_size=[12, 30],
        exposure_time=1,
        threshold=20000,
        max_exposure_time=10,
        extra="XR_scan",
        output_file=None,
        verbosity=5, 
        **md,
    ):
        """Run x-ray reflectivity measurement for thin film samples on WAXS pilatus800k.
        There will be two WAXSy positions for XR.
        The 1st position is the beam shining directly on the detector with maximum attenuation.
        This position is defined by cms.WAXS.setCalibration([734, 1090],0.255, [-65, -73])
        The 2nd position is the beam out of the WAXS detector.
        The detector will be moved to the 2nd position when the reflected beam out of beam stop.

        Parameters
        ----------
        scan_type : list
            theta_scan: in step of theta
            q_scan: in step of q
        theta_range: list
            The scanning range. It can be single section or multiple sections with various step_size.
            Examples:
            [0, 1.6] or
            [[0, .3],[0.3, 1], [1, 1.6]]
        theta_delta: float or list
            The scaning step. Examples:
            0.02    or
            [0.005, 0.1, 0.2]
        roi_size: float
            The szie of ROI1.
        exposure_time: float
            The mininum exposure time
        min_step : float
            The final (minimum) step size to try
        intensity : float
            The expected full-beam intensity readout
        threshold : float
            The threshold of minimum intensity. Exposure time increases automatically if < max_exposure_time
        max_exposure_time : float
            The maximum of exposure time to limit the total time.

        """
        # TODO:
        # if theta_end < theta_start:
        #    print("The theta_end is larger than theta_start!!!")

        # disable the besteffortcallback and plot all ROIs
        # bec.disable_table()
        cms.modeXRMeasurement()
        cms.definePos(size=roi_size)

        bec.disable_plots()

        cms.WAXS.detector.stats1.total.kind = "hinted"
        cms.WAXS.detector.stats2.total.kind = "hinted"

        self.naming_scheme_hold = self.naming_scheme
        self.naming_scheme = ["name", "extra", "x", "th", "exposure_time"]
        # default_WAXSy = WAXSy.position

        # move in absorber and move out the beamstop
        slot_pos = 5
        beam.setAbsorber(slot_pos)
        if beam.absorber()[0] >= 4:
            bsx.move(bsx.position + 6)
            beam.setTransmission(1)

        # create a clean dataframe and a direct beam images
        self.yr(-2)
        self.tho()
        # Energy = 13.5kev
        if abs(beam.energy(verbosity=1) - 13.5) < 0.1:
            direct_beam_slot = 4
        # Energy = 17kev
        if abs(beam.energy(verbosity=1) - 17) < 0.1:
            direct_beam_slot = 6
            slot_pos = 6

        # Energy = 10kev
        if abs(beam.energy(verbosity=1) - 10) < 0.1:
            direct_beam_slot = 5

        beam.setAbsorber(direct_beam_slot)
        # TODO:move detector to the 1st position and setROI
        get_beamline().setWAXSpos(total_angle=0, roi=cms.XR_pos1)
        get_beamline().setXRROI(total_angle=0, size=roi_size, default_WAXSy=None)
        self.measure(exposure_time, extra="direct_beam")
        self.yo()

        output_data = self.XR_data_output(direct_beam_slot, exposure_time)
        # output_data = output_data.iloc[0:0]

        # create a data file to save the XRR data
        if output_file is None:
            header = db[-1]
            # XR_FILENAME='{}/data/{}.csv'.format(os.path.dirname(__file__) , header.get('start').get('scan_id')+1)
            # XR_FILENAME='{}/data/{}.csv'.format(header.start['experiment_alias_directory'], header.get('start').get('scan_id')+1)
            # XR_FILENAME='{}/data/{}_{}.csv'.format(header.start['experiment_alias_directory'],header.start['sample_name'], header.get('start').get('scan_id')+1)
            XR_FILENAME = "{}/data/{}.csv".format(
                header.start["experiment_alias_directory"], header.start["filename"]
            )
            print("FILENAME= {}".format(XR_FILENAME))
        else:
            XR_FILENAME = "{}/data/{}.csv".format(header.start["experiment_alias_directory"], output_file)
        
        if verbosity>=5:
            theta_output_data = None


        # load theta positions in scan
        if scan_type == "theta_scan":
            # list the sth positions in scan
            if theta_list == None:
                theta_list = np.arange(theta_range[0], theta_range[1], theta_delta)
            else:
                theta_list = theta_list

            #
            """
            if np.size(theta_range) == 2:
                theta_list=np.arange(theta_range[0], theta_range[1], theta_delta)
            #multiple sections for measurement
            else: 
                theta_list=[]
                if np.shape(theta_range)[0] != np.size(theta_delta):
                    print("The theta_range does not match theta_delta")
                    return
                if np.shape(theta_range)[-1] != 2:
                    print("The input of theta_range is incorrect.")
                    return                
                for number, item in enumerate(theta_range):
                    theta_list_temp = np.arange(item[0], item[1], theta_delta[number])
                    theta_list.append(theta_list_temp)
                theta_list = np.hstack(theta_list)
            
            """
        elif scan_type == "qz_scan":
            if qz_list is not None:
                qz_list = qz_list
            else:
                qz_list = self.qz_list_default
            theta_list = np.rad2deg(np.arcsin(qz_list * header.start["calibration_wavelength_A"] / 4 / np.pi))

        pos_flag = 0
        for theta in theta_list:
            self.thabs(theta)
            # get_beamline().setSpecularReflectivityROI(total_angle=theta*2,size=roi_size,default_SAXSy=-73)

            # if cms.out_of_beamstop(total_angle=theta*2, size=roi_size):
            if cms.beamOutXR(total_angle=theta * 2, roi=cms.XR_pos2) == False:
                get_beamline().setWAXSpos(total_angle=0, roi=cms.XR_pos1)
                get_beamline().setXRROI(total_angle=theta * 2, size=roi_size)
                print("WAXS in POS1 for XR")
            elif pos_flag == 0:
                get_beamline().setWAXSpos(total_angle=0, roi=cms.XR_pos2)
                get_beamline().setXRROI(total_angle=theta * 2, size=roi_size)
                print("WAXS in POS2 for XR")
                pos_flag = 1
            else:
                get_beamline().setXRROI(total_angle=theta * 2, size=roi_size)
                print("WAXS in POS2 for XR")
                pos_flag = 1

            self.measure(exposure_time, extra=extra)
            temp_data = self.XR_data_output(slot_pos, exposure_time)
            print("data = {}".format(temp_data))
            # initial exposure period
            N = 1
            N_last = 1
            if threshold is not None and type(threshold) == int:
                while temp_data["e_I1"][temp_data.index[-1]] < threshold and N < max_exposure_time:
                    if slot_pos > 0:
                        if (
                            temp_data["e_I1"][temp_data.index[-1]] < threshold
                        ):  # The count is too small to evaluate the next slot_pos.
                            slot_pos = slot_pos - 1
                        else:
                            slot_current = (
                                beam.absorber_transmission_list[slot_pos]
                                * threshold
                                / temp_data["e_I1"][temp_data.index[-1]]
                            )
                            for slot_no in np.arange(5, 0, -1):
                                if slot_current > beam.absorber_transmission_list[slot_no]:
                                    slot_pos = slot_no - 1

                        beam.setAbsorber(slot_pos)
                        print("The absorber is slot {}\n".format(slot_pos))
                        print("The theta is {}\n".format(theta))
                        if verbosity>=5:
                            THETA_FILENAME = "{}/data/Theta_profile_{}.csv".format(
                                header.start["experiment_alias_directory"], header.start["filename"]
                            )

                            theta_temp_output = {'a_scanID':db[-1].start["scan_id"]+1, "b_theta": theta }
                            theta_temp_output_data = pds.DataFrame(data=theta_temp_output)

                            if theta_output_data == None: 
                                theta_output_data = theta_temp_output_data

                            theta_output_data = pds.concat([theta_output_data, theta_temp_output_data], ignore_index=True)
                            # save to file
                            theta_output_data.to_csv(THETA_FILENAME)

                        self.measure(exposure_time, extra=extra)
                        temp_data = self.XR_data_output(slot_pos, exposure_time)
                    else:
                        if (
                            threshold / float(temp_data["e_I1"][temp_data.index[-1]]) < max_exposure_time
                            and N_last < max_exposure_time
                        ):
                            N = np.ceil(N_last * threshold / float(temp_data["e_I1"][temp_data.index[-1]]))
                            print("e_I1={}".format(float(temp_data["e_I1"][temp_data.index[-1]])))
                            print("N={}".format(N))
                            print("exposure time  = {}".format(N * exposure_time))
                        else:
                            N = max_exposure_time
                            print("exposure time is MAX")
                        print("The absorber is slot {}\n".format(slot_pos))
                        print("The theta is {}\n".format(theta))

                        self.measure(N * exposure_time, extra=extra)
                        temp_data = self.XR_data_output(slot_pos, N * exposure_time)
                        N_last = N

            elif len(threshold) > 1 and temp_data["e_I1"][temp_data.index[-1]] > threshold[-1]:
                slot_pos = slot_pos + 1
                print("The absorber is slot {}\n".format(slot_pos))
                print("The theta is {}\n".format(theta))
                beam.setAbsorber(slot_pos)
                self.measure(exposure_time, extra=extra)
                temp_data = self.XR_data_output(slot_pos, exposure_time)

            # output_data = output_data.append(temp_data, ignore_index=True)
            output_data = pds.concat([output_data, temp_data], ignore_index=True)
            # save to file
            output_data.to_csv(XR_FILENAME)

        # reset the changed items
        bec.enable_plots()
        # bec.enable_table()
        self.naming_scheme = self.naming_scheme_hold
        # remove the absorber completely out of the beam
        beam.absorber_out()

    def XR_abort(self):
        """Reset the beamline status back to origin before XRR measurement."""
        beam.off()
        cms.modeMeasurement()
        beam.setAbsorber(0)
        # remove the absorber completely out of the beam
        beam.absorber_out()

        self.xo()
        self.yo()
        self.tho()

        bec.enable_plots()
        bec.enable_table()
        pilatus_name.hints = {"fields": ["pilatus800_stats3_total", "pilatus800_stats4_total"]}

    def XR_data_output(self, slot_pos, exposure_time):
        """XRR data output in DataFrame format, including:
        'a_qz': qz,                  #qz
        'b_th':sth_pos,              #incident angle
        'c_scanID': scan_id,         #scan ID
        'd_I0': I0,                  #bim5 flux
        'e_I1': I1,                  #ROI1
        'f_I2': I2,                  #ROI2
        'g_I3': I3,                  #2*ROI1-ROI2
        'h_In': In,                  #reflectivity
        'i_absorber_slot': slot_pos, #absorption slot No.
        'j_exposure_seconds': exposure_time}   #exposure time
        """

        h = db[-1]
        dtable = h.table()

        # beam.absorber_transmission_list = [1, 0.041, 0.0017425, 0.00007301075, 0.00000287662355, 0.000000122831826, 0.00000000513437]

        # Energy = 13.5kev
        if abs(beam.energy(verbosity=1) - 13.5) < 0.1:
            beam.absorber_transmission_list = beam.absorber_transmission_list_13p5kev

        # Energy = 17kev
        elif abs(beam.energy(verbosity=1) - 17) < 0.1:
            beam.absorber_transmission_list = beam.absorber_transmission_list_17kev

        elif abs(beam.energy(verbosity=1) - 10) < 0.1:
            beam.absorber_transmission_list = beam.absorber_transmission_list
        else:
            print("The absorber has not been calibrated under current Energy!!!")

        sth_pos = h.start["sample_th"]
        qz = 4 * np.pi * np.sin(np.deg2rad(sth_pos)) / h.start["calibration_wavelength_A"]
        scan_id = h.start["scan_id"]
        I0 = h.start["beam_int_bim5"]  # beam intensity from bim5
        I1 = dtable.pilatus800_stats1_total
        I2 = dtable.pilatus800_stats2_total
        I3 = 2 * dtable.pilatus800_stats1_total - dtable.pilatus800_stats2_total
        In = I3 / beam.absorber_transmission_list[slot_pos] / exposure_time

        current_data = {
            "a_qz": qz,  # qz
            "b_th": sth_pos,  # incident angle
            "c_scanID": scan_id,  # scan ID
            "d_I0": I0,  # bim5 flux
            "e_I1": I1,  # ROI1
            "f_I2": I2,  # ROI2
            "g_I3": I3,  # 2*ROI1-ROI2
            "h_In": In,  # reflectivity
            "i_absorber_slot": slot_pos,  # absorption slot No.
            "j_exposure_seconds": exposure_time,
        }  # exposure time

        return pds.DataFrame(data=current_data)

    def XR_align(self, step=0, reflection_angle=0.15, verbosity=3):
        """Specific alignment for XRR

        Align the sample with respect to the beam. XR alignment involves
        vertical translation to the beam center, and rocking theta to get the
        sample plane parralel to the beam. Finally, the angle is re-optimized
        in reflection mode.

        The 'step' argument can optionally be given to jump to a particular
        step in the sequence."""

        #

        if verbosity >= 4:
            print("  Aligning {}".format(self.name))

        if step <= 0:
            cms.modeXRAlignment()
            cms.modeAlignment()
            beam.setTransmission(1e-4)
            detselect(pilatus2M)
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

        if step <= 8:
            # fit_scan(smy, 0.6, 21, fit='sigmoid_r')

            fit_edge(smy, 0.6, 21)
            # time.sleep(2)
            # fit_edge(smy, 0.4, 21)
            fit_scan(sth, 0.8, 21, fit="COM")
            # time.sleep(2)
            self.setOrigin(["y", "th"])

        if step <= 9 and reflection_angle is not None:
            # Final alignment using reflected beam
            if verbosity >= 4:
                print("    align: reflected beam")

            if abs(beam.energy(verbosity) - 17) < 0.1:
                reflection_angle = 0.15

            get_beamline().setReflectedBeamROI(total_angle=reflection_angle * 2.0)
            # get_beamline().setReflectedBeamROI(total_angle=reflection_angle*2.0, size=[12,2])

            self.thabs(reflection_angle)

            result = fit_scan(sth, 0.1, 41, fit="max")
            # result = fit_scan(sth, 0.2, 81, fit='max') #it's useful for alignment of SmarAct stage
            sth_target = result.values["x_max"] - reflection_angle

            if result.values["y_max"] > 50:
                th_target = self._axes["th"].motor_to_cur(sth_target)
                self.thsetOrigin(th_target)

            # fit_scan(smy, 0.2, 21, fit='max')
            # self.setOrigin(['y'])

        if step <= 10:
            self.thabs(0.0)
            beam.off()
            detselect(pilatus800)

    def XR_check_alignment(self, th_angle=1, exposure_time=1, roi_size=[10, 10]):
        """Check the alignment of the XR.
        The total_angle is the incident angle.
        The reflection spot should be located in the center of ROI2"""
        cms.modeXRMeasurement()
        # TODO: set a default position
        get_beamline().setWAXSpos(total_angle=0, roi=cms.XR_pos2)
        get_beamline().setXRROI(total_angle=th_angle * 2, size=roi_size)
        # sam.xo()
        sam.yo()
        sam.thabs(th_angle)
        sam.measure(exposure_time)
        print(
            "===========sam.th moves to {}deg and ROI1 is set at {}deg. ============".format(
                th_angle, th_angle * 2
            )
        )
        print("======Please check the ROI whether at the reflected position. =======")
        print("========If not, modify sam.th or schi to meet the reflected beam. ===========")

    # define a theta-2theta scan by rotating sample by sth and accordingly changing roi1 and roi2 at 2theta position

    def th2thscan(
        self,
        scan_type="theta_scan",
        theta_range=[1, 4],
        theta_delta=0.1,
        qz_list=None,
        roi_size=[10, 10],
        exposure_time=1,
        threshold=20000,
        max_exposure_time=10,
        extra="th2th_scan",
        output_file=None,
    ):
        """Run x-ray reflectivity measurement for thin film samples on WAXS pilatus800k.
        There will be two WAXSy positions for XR.
        The 1st position is the beam shining directly on the detector with maximum attenuation.
        This position is defined by cms.WAXS.setCalibration([734, 1090],0.255, [-65, -73])
        The 2nd position is the beam out of the WAXS detector.
        The detector will be moved to the 2nd position when the reflected beam out of beam stop.

        Parameters
        ----------
        scan_type : list
            theta_scan: in step of theta
            q_scan: in step of q

        theta_range: list
            The scanning range. It can be single section or multiple sections with various step_size.
            Examples:
            [0, 1.6] or
            [[0, .3],[0.3, 1], [1, 1.6]]
        theta_delta: float or list
            The scaning step. Examples:
            0.02    or
            [0.005, 0.1, 0.2]
        roi_size: float
            The szie of ROI1.
        exposure_time: float
            The mininum exposure time
        min_step : float
            The final (minimum) step size to try
        intensity : float
            The expected full-beam intensity readout
        threshold : float
            The threshold of minimum intensity. Exposure time increases automatically if < max_exposure_time
        max_exposure_time : float
            The maximum of exposure time to limit the total time.

        """
        # TODO:

        [theta_start, theta_end] = theta_range
        if theta_end < theta_start:
            print("The theta_end is larger than theta_start!!!")
        if theta_start < 0.25:
            print("For th2th scan, the start of the scan has to be larger than 0.5deg")
            return

        # disable the besteffortcallback and plot all ROIs
        # bec.disable_table()
        cms.modeXRMeasurement()
        cms.definePos(size=roi_size)

        bec.disable_plots()

        cms.WAXS.detector.stats1.total.kind = "hinted"
        cms.WAXS.detector.stats2.total.kind = "hinted"

        self.naming_scheme_hold = self.naming_scheme
        self.naming_scheme = ["name", "extra", "th", "exposure_time"]
        # default_WAXSy = WAXSy.position

        # move in absorber and move out the beamstop
        slot_pos = 6
        beam.setAbsorber(slot_pos)
        if beam.absorber()[0] >= 4:
            bsx.move(bsx.position + 6)
            beam.setTransmission(1)

        # create a clean dataframe and a direct beam images
        self.yr(-2)
        self.tho()
        # Energy = 13.5kev
        if abs(beam.energy(verbosity=1) - 13.5) < 0.1:
            direct_beam_slot = 4
        # Energy = 17kev
        if abs(beam.energy(verbosity=1) - 17) < 0.1:
            direct_beam_slot = 5
            slot_pos = 5
        beam.setAbsorber(direct_beam_slot)
        # move detector to the 1st position and setROI
        get_beamline().setWAXSpos(total_angle=0, roi=cms.XR_pos1)
        get_beamline().setXRROI(total_angle=0, size=roi_size, default_WAXSy=None)
        self.measure(exposure_time, extra="direct_beam")
        self.yo()
        # move detector to POS2 and start the th2th scan
        get_beamline().setWAXSpos(total_angle=0, roi=cms.XR_pos2)

        output_data = self.XR_data_output(direct_beam_slot, exposure_time)
        # output_data = output_data.iloc[0:0]

        # create a data file to save the XRR data
        if output_file is None:
            header = db[-1]
            # XR_FILENAME='{}/data/{}.csv'.format(os.path.dirname(__file__) , header.get('start').get('scan_id')+1)
            # XR_FILENAME='{}/data/{}.csv'.format(header.start['experiment_alias_directory'], header.get('start').get('scan_id')+1)
            th2th_FILENAME = "{}/data/{}_{}.csv".format(
                header.start["experiment_alias_directory"],
                header.start["sample_name"],
                header.get("start").get("scan_id") + 1,
            )
        else:
            th2th_FILENAME = "{}/data/{}.csv".format(header.start["experiment_alias_directory"], output_file)

        # load theta positions in scan
        if scan_type == "theta_scan":
            # list the sth positions in scan
            theta_list = np.arange(theta_start, theta_end + 0.001, theta_delta)
        elif scan_type == "qz_scan":
            if qz_list is not None:
                qz_list = qz_list
            else:
                qz_list = self.qz_list_default
            theta_list = np.rad2deg(np.arcsin(qz_list * header.start["calibration_wavelength_A"] / 4 / np.pi))

        pos_flag = 0
        for theta in theta_list:
            self.thabs(theta)
            # th2th scan starts with POS2 directly
            get_beamline().setXRROI_update(total_angle=theta * 2, size=roi_size)

            self.measure(exposure_time, extra=extra)
            temp_data = self.XR_data_output(slot_pos, exposure_time)

            # initial exposure period
            if threshold is not None and type(threshold) == int:
                if slot_pos > 0:
                    if (
                        temp_data["e_I1"][temp_data.index[-1]] < 10
                    ):  # The count is too small to evaluate the next slot_pos.
                        slot_pos = slot_pos - 1
                    else:
                        slot_current = (
                            beam.absorber_transmission_list[slot_pos]
                            * threshold
                            / temp_data["e_I1"][temp_data.index[-1]]
                        )
                        for slot_no in np.arange(5, 0, -1):
                            if slot_current > beam.absorber_transmission_list[slot_no]:
                                slot_pos = slot_no - 1

                    beam.setAbsorber(slot_pos)
                    print("The absorber is slot {}\n".format(slot_pos))
                    print("The theta is {}\n".format(theta))
                    self.measure(exposure_time, extra=extra)
                    temp_data = self.XR_data_output(slot_pos, exposure_time)
                # else:
                ##self.measure(exposure_time, extra=extra)
                # temp_data = self.XR_data_output(slot_pos, exposure_time)

            elif len(threshold) > 1 and temp_data["e_I1"][temp_data.index[-1]] > threshold[-1]:
                slot_pos = slot_pos + 1
                print("The absorber is slot {}\n".format(slot_pos))
                print("The theta is {}\n".format(theta))
                beam.setAbsorber(slot_pos)
                self.measure(exposure_time, extra=extra)
                temp_data = self.XR_data_output(slot_pos, exposure_time)

            # output_data = output_data.append(temp_data, ignore_index=True)
            output_data = pds.concat([output_data, temp_data], ignore_index=True)
            # save to file
            output_data.to_csv(th2th_FILENAME)

        # reset the changed items
        bec.enable_plots()
        # bec.enable_table()
        self.naming_scheme = self.naming_scheme_hold
        # remove the absorber completely out of the beam
        beam.absorber_out()


class SampleGonio_Generic(SampleGISAXS_Generic):
    """Goniometer stage made by SmarAct"""

    def __init__(self, name="Goniometer", base=None, **md):
        super().__init__(name=name, base=base, **md)
        self._axes["x"].origin = -17.7
        self._axes["y"].origin = 9

    def _set_axes_definitions(self):
        """Internal function which defines the axes for this stage. This is kept
        as a separate function so that it can be over-ridden easily."""
        super()._set_axes_definitions()
        # The _axes_definitions array holds a list of dicts, each defining an axis
        self._axes_definitions.append = [
            {
                "name": "phi",
                "motor": srot,
                "enabled": True,
                "scaling": +1.0,
                "units": "deg",
                "hint": None,
            },
            {
                "name": "trans",
                "motor": strans,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": None,
            },
            {
                "name": "trans2",
                "motor": strans2,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": None,
            },
        ]


class SampleSecondStage(SampleGISAXS_Generic):
    """The second sample stage with the large open space
    Note: steps to initiallize it.
    1. Break vacuum of both chamber and pipe.
    2. Remove the connected tubing and
    3. open 10-motors.py. set beamline_stage = 'open_MAXS'
    """

    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)
        self._axes["x"].origin = -1.9
        self._axes["y"].origin = 18.97
        self._axes["th"].origin = 0.348


# custmer-made holders


class GIBar(PositionalHolder):
    """This class is a sample bar for grazing-incidence (GI) experiments."""

    # Core methods
    ########################################

    def __init__(self, name="GIBar", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._positional_axis = "x"

        #before 2024-2
        # self.xsetOrigin(-71.89405)
        # # self.ysetOrigin(10.37925)
        # self.ysetOrigin(5.38)
        
        #changed in 2024-3
        self._axes['x'].origin = -76.1
        self._axes['y'].origin = 7   
        

        self.mark("right edge", x=+108.2)
        self.mark("left edge", x=0)
        self.mark("center", x=54.1, y=0)

        # measurement measure_settings
        self.detector = None
        self.exposure_time = None
        self.incident_angles = None
        self.tiling = None
        self.measure_setting = {}

    def _backup_addSampleSlotPosition(
        self,
        sample,
        slot,
        position,
        detector_opt="SAXS",
        incident_angles=None,
        account_substrate=True,
        exposure_time=None,
        thickness=0,
    ):
        """Adds a sample to the specified "slot" (defined/numbered sample
        holding spot on this holder)."""

        super().addSampleSlotPosition(
            sample=sample,
            slot=slot,
            position=position,
            detector_opt=detector_opt,
            incident_angles=incident_angles,
        )

        # Adjust y-origin to account for substrate thickness
        if account_substrate and "substrate_thickness" in sample.md:
            sample.ysetOrigin(-1.0 * sample.md["substrate_thickness"])

        sample.detector = detector_opt
        sample.exposure_time = exposure_time
        sample.thickness = exposure_time

        # Adjust y-origin to account for substrate thickness
        if thickness != 0:
            sample.ysetOrigin(-1.0 * thickness)

    def addSampleSlotPosition(
        self,
        sample,
        slot,
        position,
        detector_opt=None,
        incident_angles=None,
        exposure_time=None,
        tiling=None,
        thickness=0,
    ):
        """Adds a sample to the specified "slot" (defined/numbered sample
        holding spot on this holder)."""

        super().addSampleSlotPosition(
            sample=sample,
            slot=slot,
            position=position,
            detector_opt=detector_opt,
            incident_angles=incident_angles,
        )
        # Adjust y-origin to account for substrate thickness
        if thickness != 0:
            sample.ysetOrigin(-1.0 * thickness)

        if detector_opt == None:
            sample.measure_setting["detector"] = self.detector
        else:
            sample.measure_setting["detector"] = detector_opt
        if exposure_time == None:
            sample.measure_setting["exposure_time"] = self.exposure_time
        else:
            sample.measure_setting["exposure_time"] = exposure_time
        if incident_angles == None:
            sample.measure_setting["incident_angles"] = self.incident_angles
        else:
            sample.measure_setting["incident_angles"] = incident_angles
        if tiling == None:
            sample.measure_setting["tiling"] = self.tiling
        else:
            sample.measure_setting["tiling"] = tiling

    def setMeasure(self, detector, incident_angles, exposure_time, tiling):
        """define the measurement setting for the holder, including:
        detector:  'SAXS', 'WAXS', or 'BOTH';
        incident_angles: [0.05, 0.10.2];
        exposure_time: in second, corresponding to the detector setting;
        tiling: 'x', 'y', 'xy', None

        """
        if detector == "BOTH":
            detector = ["SAXS", "WAXS"]
            if type(exposure_time) == int:
                exposure_time = [exposure_time, exposure_time]
        self.measure_setting["detector"] = detector
        self.measure_setting["incident_angles"] = incident_angles
        self.measure_setting["exposure_time"] = exposure_time
        self.measure_setting["tiling"] = tiling
        self.detector = detector
        self.incident_angles = incident_angles
        self.exposure_time = exposure_time
        self.tiling = tiling

        return self.measure_setting

    def alignSamples(self, range=None, step=0, align_step=0, x_offset=0, verbosity=3, **md):
        """Iterates through the samples on the holder, aligning each one."""

        if step <= 0:
            get_beamline().modeAlignment()

        if step <= 5:
            for sample in self.getSamples(range=range):
                sample.gotoOrigin(["x", "y", "th"])
                sample.gotoOrigin(["x"])
                sample.xr(x_offset)
                sample.align(step=align_step, reflection_angle=0.12)

        if step <= 10:
            if verbosity >= 3:
                print("Alignment complete.")
                for i, sample in enumerate(self.getSamples()):
                    print("Sample {} ({})".format(i + 1, sample.name))
                    print(sample.save_state())

    def alignSamplesQuick(self, range=None, step=0, x_offset=0, verbosity=3, **md):
        """Iterates through the samples on the holder, aligning each one."""

        if step <= 0:
            get_beamline().modeAlignment()

        if step <= 5:
            for sample in self.getSamples(range=range):
                sample.gotoOrigin(["x", "y", "th"])
                sample.gotoOrigin(["x"])
                sample.xr(x_offset)
                sample.alignQuick(reflection_angle=0.12)  # =0.07)

        if step <= 10:
            if verbosity >= 3:
                print("Alignment complete.")
                for i, sample in enumerate(self.getSamples()):
                    print("Sample {} ({})".format(i + 1, sample.name))
                    print(sample.save_state())

    def alignSamplesVeryQuick(self, range=None, step=0, x_offset=0, verbosity=3, **md):
        """Iterates through the samples on the holder, aligning each one."""

        if step <= 0:
            get_beamline().modeAlignment()
            beam.on()
            caput("XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime", 0.25)
            caput("XF:11BMB-ES{Det:SAXS}:cam1:AcquirePeriod", 0.30)

        if step <= 5:
            for sample in self.getSamples(range=range):
                sample.gotoOrigin(["x", "y", "th"])
                sample.gotoOrigin(["x"])
                sample.xr(x_offset)
                sample.alignVeryQuick(intensity=INTENSITY_EXPECTED_025, mode_control=False)

        if step <= 8:
            beam.off()

        if step <= 10:
            if verbosity >= 3:
                print("Alignment complete.")
                for i, sample in enumerate(self.getSamples()):
                    print("Sample {} ({})".format(i + 1, sample.name))
                    print(sample.save_state())

    def measureSamples(self, range=None, step=0, angles=None, exposure_time=15, x_offset=0, verbosity=3, **md):
        """Measures all the samples.

        If the optional range argument is provided (2-tuple), then only sample
        numbers within that range (inclusive) are run. If range is instead a
        string, then all samples with names that match are returned."""

        if step <= 0:
            get_beamline().modeMeasurement()

        if step <= 5:
            for sample in self.getSamples(range=range):
                if verbosity >= 3:
                    print("Measuring sample {}...".format(sample.name))

                sample.gotoOrigin(["x", "y", "th"])
                sample.gotoOrigin(["x"])
                sample.xr(x_offset)
                sample.measureIncidentAngles(angles=angles, verbosity=verbosity, exposure_time=exposure_time, **md)

    def printSaveStates(self, range=None, verbosity=3, **md):
        if range is None:
            range_start = 0
        else:
            range_start = range[0]

        save_string = "origins = [\n"

        for i, sample in enumerate(self.getSamples(range=range)):
            sample_id = range_start + i + 1

            save_string += "    {} , # Sample {}\n".format(sample.save_state(), sample_id)
            # save_string += '    {} , # Sample {} ({})\n'.format(sample.save_state(), sample_id, sample.name)

        save_string += "    ]\n"
        save_string += "for origin, sample in zip(origins, hol.getSamples()):\n"
        save_string += "    sample.restore_state(origin)\n"

        print(save_string)

    def _backup_doSamples(self, range=None, verbosity=3):
        # saxs_on()
        for sample in self.getSamples(range=range):
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            if sample.detector == "SAXS" or sample.detector == "BOTH":
                sample.do_SAXS()

        for sample in self.getSamples(range=range):
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            if sample.detector == "BOTH":
                sample.do_WAXS_only()

        for sample in self.getSamples(range=range):
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            if sample.detector == "WAXS":
                sample.do_WAXS()

    def doSamples(self, range=None, verbosity=3):
        # saxs_on()
        for sample in self.getSamples(range=range):
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            if "SAXS" in sample.detector or sample.detector == "BOTH":
                sample.do_SAXS()

        for sample in self.getSamples(range=range):
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            if "SAXS" in sample.detector and "WAXS" in sample.detector:
                sample.do_WAXS_only()
            elif sample.detector == "BOTH":
                sample.do_WAXS_only()

        for sample in self.getSamples(range=range):
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            if sample.detector == "WAXS":
                sample.do_WAXS()

    def doSamples_Stitch(self, angles=None, exposure_time=None, extra=None, tiling=None, verbosity=3, **md):
        if exposure_time == None:
            exposure_time = self.exposure_time

        # measure the incident angles first and then change the tiling features.
        if tiling == None:
            if angles is None:
                if sample.incident_angles == None:
                    incident_angles = self.incident_angles_default
                else:
                    incident_angles = self.incident_angles
            for angle in angles:
                self.measureIncidentAngle(angle, exposure_time=exposure_time, extra=extra, tiling=tiling, **md)

        elif tiling == "ygaps":
            SAXSy_o = SAXSy.user_readback.value
            SAXSx_o = SAXSx.user_readback.value
            WAXSy_o = WAXSy.user_readback.value
            WAXSx_o = WAXSx.user_readback.value

            # pos1
            for sample in self.getSamples():
                sample.gotoOrigin()
                if angles is None:
                    if sample.incident_angles == None:
                        incident_angles = self.incident_angles_default
                    else:
                        incident_angles = self.incident_angles
                for angle in angles:
                    sample.thabs(angle)
                    time.sleep(0.5)
                    extra_current = "pos1" if extra is None else "{}_pos1".format(extra)
                    md["detector_position"] = "lower"
                    sample.measure_single(
                        exposure_time=exposure_time,
                        extra=extra_current,
                        verbosity=verbosity,
                        stitchback=True,
                        **md,
                    )

            # pos2
            # MAXSy_o = MAXSy.user_readback.value
            if pilatus2M in cms.detector:
                SAXSy.move(SAXSy_o + 5.16)
            if pilatus800 in cms.detector:
                WAXSy.move(WAXSy_o + 5.16)
            if pilatus300 in cms.detector:
                MAXSy.move(MAXSy_o + 5.16)

            for sample in self.getSamples():
                sample.gotoOrigin()
                if angles is None:
                    if sample.incident_angles == None:
                        incident_angles = self.incident_angles_default
                    else:
                        incident_angles = self.incident_angles
                for angle in angles:
                    sample.thabs(angle)
                    time.sleep(0.5)
                    extra_current = "pos2" if extra is None else "{}_pos2".format(extra)
                    md["detector_position"] = "upper"
                    sample.measure_single(
                        exposure_time=exposure_time,
                        extra=extra_current,
                        verbosity=verbosity,
                        stitchback=True,
                        **md,
                    )

            if SAXSy.user_readback.value != SAXSy_o:
                SAXSy.move(SAXSy_o)
            if WAXSy.user_readback.value != WAXSy_o:
                WAXSy.move(WAXSy_o)
            # if MAXSy.user_readback.value != MAXSy_o:
            # MAXSy.move(MAXSy_o)

        elif tiling == "xygaps":
            SAXSy_o = SAXSy.user_readback.value
            SAXSx_o = SAXSx.user_readback.value
            WAXSy_o = WAXSy.user_readback.value
            WAXSx_o = WAXSx.user_readback.value

            # pos1
            for sample in self.getSamples():
                if angles is None:
                    if sample.incident_angles == None:
                        incident_angles = self.incident_angles_default
                    else:
                        incident_angles = self.incident_angles

                sample.gotoOrigin()
                for angle in angles:
                    sample.thabs(angle)
                    time.sleep(0.5)
                    extra_current = "pos1" if extra is None else "{}_pos1".format(extra)
                    md["detector_position"] = "lower_left"
                    sample.measure_single(
                        exposure_time=exposure_time,
                        extra=extra_current,
                        verbosity=verbosity,
                        stitchback=True,
                        **md,
                    )

            # pos2
            if pilatus2M in cms.detector:
                SAXSy.move(SAXSy_o + 5.16)
            if pilatus800 in cms.detector:
                WAXSy.move(WAXSy_o + 5.16)
            if pilatus300 in cms.detector:
                MAXSy.move(MAXSy_o + 5.16)
            for sample in self.getSamples():
                sample.gotoOrigin()
                for angle in angles:
                    sample.thabs(angle)
                    time.sleep(0.2)
                    extra_current = "pos2" if extra is None else "{}_pos2".format(extra)
                    md["detector_position"] = "upper"
                    sample.measure_single(
                        exposure_time=exposure_time,
                        extra=extra_current,
                        verbosity=verbosity,
                        stitchback=True,
                        **md,
                    )

            # pos4  #comment out to save time
            if pilatus2M in cms.detector:
                SAXSx.move(SAXSx_o + 5.16)
                SAXSy.move(SAXSy.o + 5.16)
            if pilatus800 in cms.detector:
                WAXSx.move(WAXSx_o - 5.16)
                WAXSy.move(WAXSy_o + 5.16)
            for sample in self.getSamples():
                sample.gotoOrigin()
                if angles is None:
                    if sample.incident_angles == None:
                        incident_angles = self.incident_angles_default
                    else:
                        incident_angles = self.incident_angles
                for angle in angles:
                    sample.thabs(angle)
                    time.sleep(0.2)
                    extra_current = "pos4" if extra is None else "{}_pos4".format(extra)
                    md["detector_position"] = "upper_right"
                    sample.measure_single(
                        exposure_time=exposure_time,
                        extra=extra_current,
                        verbosity=verbosity,
                        stitchback=True,
                        **md,
                    )

            # pos3
            if pilatus2M in cms.detector:
                SAXSx.move(SAXSx_o + 5.16)
                SAXSy.move(SAXSy_o)
            if pilatus800 in cms.detector:
                WAXSx.move(WAXSx_o - 5.16)
                WAXSy.move(WAXSy_o)
            for sample in self.getSamples():
                sample.gotoOrigin()
                if angles is None:
                    if sample.incident_angles == None:
                        incident_angles = self.incident_angles_default
                    else:
                        incident_angles = self.incident_angles
                for angle in angles:
                    sample.thabs(angle)
                    time.sleep(0.2)
                    extra_current = "pos3" if extra is None else "{}_pos3".format(extra)
                    md["detector_position"] = "lower_right"
                    sample.measure_single(
                        exposure_time=exposure_time,
                        extra=extra_current,
                        verbosity=verbosity,
                        stitchback=True,
                        **md,
                    )

            if WAXSx.user_readback.value != WAXSx_o:
                WAXSx.move(WAXSx_o)
            if WAXSy.user_readback.value != WAXSy_o:
                WAXSy.move(WAXSy_o)

            if SAXSx.user_readback.value != SAXSx_o:
                SAXSx.move(SAXSx_o)
            if SAXSy.user_readback.value != SAXSy_o:
                SAXSy.move(SAXSy_o)


class CapillaryHolder(PositionalHolder):
    """This class is a sample holder that has 15 slots for capillaries."""

    # Core methods
    ########################################

    def __init__(self, name="CapillaryHolder", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._positional_axis = "x"

        self.x_spacing = 6.342  # 3.5 inches / 14 spaces

        # slot  1; smx = +26.60
        # slot  8; smx = -17.80
        # slot 15; smx = -61.94

        # Set the x and y origin to be the center of slot 8
        # self.xsetOrigin(-16.7)
        # self.ysetOrigin(-2.36985)
        # self.ysetOrigin(-2.36985)
        # self.xsetOrigin(-16.7+-0.3)
        # self.ysetOrigin(-1.8)
        # self.xsetOrigin(-17.2)


        self.ysetOrigin(-4.8)
        self.xsetOrigin(-20)


        self.mark("right edge", x=+54.4)
        self.mark("left edge", x=-54.4)
        self.mark("bottom edge", y=-12.71)
        self.mark("center", x=0, y=0)

    def get_slot_position(self, slot):
        """Return the motor position for the requested slot number."""

        return +1 * self.x_spacing * (slot - 8)

    def measure_Stitch(self, exposure_time=None, extra=None, tiling=None, verbosity=3, **md):
        # measure the incident angles first and then change the tiling features.
        if tiling == None:
            for sample in self.getSamples():
                sample.gotoOrigin()
                sample.measure_single(
                    exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
                )

        elif tiling == "ygaps":
            # pos1
            for sample in self.getSamples():
                sample.gotoOrigin()
                time.sleep(0.2)
                extra_current = "pos1" if extra is None else "{}_pos1".format(extra)
                md["detector_position"] = "lower"
                sample.measure_single(
                    exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
                )

            # pos2
            SAXSy_o = SAXSy.user_readback.value
            SAXSx_o = SAXSx.user_readback.value
            WAXSy_o = WAXSy.user_readback.value
            WAXSx_o = WAXSx.user_readback.value
            # MAXSy_o = MAXSy.user_readback.value

            for sample in self.getSamples():
                sample.gotoOrigin()
                time.sleep(0.2)

                if pilatus2M in cms.detector:
                    SAXSy.move(SAXSy_o + 5.16)
                if pilatus800 in cms.detector:
                    WAXSy.move(WAXSy_o + 5.16)
                if pilatus300 in cms.detector:
                    MAXSy.move(MAXSy_o + 5.16)

                extra_current = "pos2" if extra is None else "{}_pos2".format(extra)
                md["detector_position"] = "upper"
                sample.measure_single(
                    exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
                )

            if SAXSy.user_readback.value != SAXSy_o:
                SAXSy.move(SAXSy_o)
            if WAXSy.user_readback.value != WAXSy_o:
                WAXSy.move(WAXSy_o)
            # if MAXSy.user_readback.value != MAXSy_o:
            # MAXSy.move(MAXSy_o)

        elif tiling == "xygaps":
            # pos1
            for sample in self.getSamples():
                sample.gotoOrigin()
                time.sleep(0.2)
                extra_current = "pos1" if extra is None else "{}_pos1".format(extra)
                md["detector_position"] = "lower_left"
                sample.measure_single(
                    exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
                )

            # pos2
            SAXSy_o = SAXSy.user_readback.value
            SAXSx_o = SAXSx.user_readback.value
            WAXSy_o = WAXSy.user_readback.value
            WAXSx_o = WAXSx.user_readback.value
            # MAXSy_o = MAXSy.user_readback.value

            for sample in self.getSamples():
                sample.gotoOrigin()
                time.sleep(0.2)
                if pilatus2M in cms.detector:
                    SAXSy.move(SAXSy_o + 5.16)
                if pilatus800 in cms.detector:
                    WAXSy.move(WAXSy_o + 5.16)
                if pilatus300 in cms.detector:
                    MAXSy.move(MAXSy_o + 5.16)

                extra_current = "pos2" if extra is None else "{}_pos2".format(extra)
                md["detector_position"] = "upper"
                sample.measure_single(
                    exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
                )

            # pos4  #comment out to save time
            for sample in self.getSamples():
                sample.gotoOrigin()
                time.sleep(0.2)

                if pilatus2M in cms.detector:
                    SAXSx.move(SAXSx_o + 5.16)
                    SAXSy.move(SAXSy_o + 5.16)
                if pilatus800 in cms.detector:
                    WAXSx.move(WAXSx_o - 5.16)
                    WAXSy.move(WAXSy_o + 5.16)
                extra_current = "pos4" if extra is None else "{}_pos4".format(extra)
                md["detector_position"] = "upper_right"
                sample.measure_single(
                    exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
                )

            # pos3
            for sample in self.getSamples():
                sample.gotoOrigin()
                time.sleep(0.2)

                if pilatus2M in cms.detector:
                    SAXSx.move(SAXSx_o + 5.16)
                    SAXSy.move(SAXSy_o)
                if pilatus800 in cms.detector:
                    WAXSx.move(WAXSx_o - 5.16)
                    WAXSy.move(WAXSy_o)

                extra_current = "pos3" if extra is None else "{}_pos3".format(extra)
                md["detector_position"] = "lower_right"
                sample.measure_single(
                    exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
                )

            if WAXSx.user_readback.value != WAXSx_o:
                WAXSx.move(WAXSx_o)
            if WAXSy.user_readback.value != WAXSy_o:
                WAXSy.move(WAXSy_o)

            if SAXSx.user_readback.value != SAXSx_o:
                SAXSx.move(SAXSx_o)
            if SAXSy.user_readback.value != SAXSy_o:
                SAXSy.move(SAXSy_o)

    def measureSamples(self, range=None, step=0, angles=None, exposure_time=15, x_offset=0, verbosity=3, **md):
        """Measures all the samples.

        If the optional range argument is provided (2-tuple), then only sample
        numbers within that range (inclusive) are run. If range is instead a
        string, then all samples with names that match are returned."""

        if step <= 0:
            get_beamline().modeMeasurement()

        if step <= 5:
            for sample in self.getSamples(range=range):
                if verbosity >= 3:
                    print("Measuring sample {}...".format(sample.name))

                sample.gotoOrigin(["x", "y"])
                sample.gotoOrigin(["x"])
                sample.xr(x_offset)
                sample.measureIncident(exposure_time=exposure_time, verbosity=verbosity, **md)


class CapillaryHolderThreeRows(CapillaryHolder):
    """This class is a sample holder that has 15x3 slots for transmission geometry."""

    # Core methods
    ########################################

    def __init__(self, name="CapillaryHolderThreeRows", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        # Set the x and y origin to be the center of slot 8

        self.mark("right edge", x=+54.4)
        self.mark("left edge", x=-54.4)
        self.mark("bottom edge", y=-12.71)
        self.mark("center", x=0, y=0)

        self._positional_axis = ["x", "y"]

        self._axes["y"].origin = -1.8  # The origin is the #8 hole in the top row
        self._axes["x"].origin = -16.9

        self.x_spacing = 6.342  # 3.5 inches / 14 spaces
        self.y_spacing = 0.25 * 25.4  # 2.5 inches / 14 spaces

    # def _set_axes_definitions(self):
    #     '''Internal function which defines the axes for this stage. This is kept
    #     as a separate function so that it can be over-ridden easily.'''

    #     # The _axes_definitions array holds a list of dicts, each defining an axis
    #     super()._set_axes_definitions()

    #     self._axes_definitions.append ( {'name': 'yy',
    #                         'motor': smy2,
    #                         'enabled': True,
    #                         'scaling': +1.0,
    #                         'units': 'mm',
    #                         'hint': 'positive moves stage up',
    #                         } )

    def slot(self, sample_number):
        """Moves to the selected slot in the holder."""

        getattr(self, self._positional_axis[0] + "abs")(self.get_slot_position(sample_number))

    def get_slot_position(self, slot):
        """Return the motor position for the requested slot number."""
        # This method should be over-ridden in sub-classes, so as to properly
        # implement the positioning appropriate for that holder.
        # This is the critical to define the position for the 10 samples.

        position_y = int((slot - 1) / 15)
        position_x = (slot - 1) % 15 - 7

        return position_x * self.x_spacing, position_y * self.y_spacing

    def addSampleSlot(self, sample, slot):
        """Adds a sample to the specified "slot" (defined/numbered sample
        holding spot on this holder)."""

        self.addSample(sample, sample_number=slot)
        sample.setOrigin(["x"], [self.get_slot_position(slot)[0]])
        sample.setOrigin(["y"], [self.get_slot_position(slot)[1]])


class CapillaryHolderHeated(CapillaryHolder):
    def update_sample_names(self):
        for sample in self.getSamples():
            if "temperature" not in sample.naming_scheme:
                sample.naming_scheme.insert(-1, "temperature")

    def doHeatCool(
        self,
        heat_temps,
        cool_temps,
        exposure_time=None,
        stabilization_time=120,
        temp_tolerance=0.5,
        step=1,
    ):
        if step <= 1:
            for temperature in heat_temps:
                try:
                    self.setTemperature(temperature)

                    while self.temperature(verbosity=0) < temperature - temp_tolerance:
                        time.sleep(5)
                    time.sleep(stabilization_time)

                    for sample in self.getSamples():
                        sample.gotoOrigin()
                        sample.xr(-0.05)
                        sample.measure(exposure_time)

                except HTTPError:
                    pass

        if step <= 5:
            for temperature in heat_temps:
                try:
                    self.setTemperature(temperature)

                    self.setTemperature(temperature)

                    while self.temperature(verbosity=0) > temperature + temp_tolerance:
                        time.sleep(5)
                    time.sleep(stabilization_time)

                    for sample in self.getSamples():
                        sample.gotoOrigin()
                        sample.xr(0.1)
                        sample.measure(exposure_time)

                except HTTPError:
                    pass


class GIBar_long_thermal(GIBar):
    """This class is a sample bar with heating/cooling feature and 6" long bar for grazing-incidence (GI) experiments."""

    # Core methods
    ########################################

    def __init__(self, name="GIBar", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._positional_axis = "x"

        # Set the x and y origin to be the center of slot 8

        # self.xsetOrigin(-71.89405-22.1) # TODO: Update this value
        # self._axes['y'].origin = 7.06
        self._axes["y"].origin = 2.06

        self.mark("right edge", x=+152.4)
        self.mark("left edge", x=0)
        self.mark("center", x=76.2, y=0)

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

    def alignSamples_Custom(self, step=0, align_step=8, verbosity=3, **md):
        # first_sample = self.getSamples()
        # hol_xcenter = 4.3*25.4/2
        cali_sample = self.getSample(1)

        sample_pos_list = []

        for sample in self.getSamples():
            sample_pos_list.append(sample.position)
        hol_xcenter = min(sample_pos_list) / 2 + max(sample_pos_list) / 2

        print(sample_pos_list)

        # locate the sample for alignning the holder
        for sample in self.getSamples():
            if abs(sample.position - hol_xcenter) < abs(cali_sample.position - hol_xcenter):
                cali_sample = sample
            print("The current calibraion sample is {}".format(cali_sample.name))

        print("The calibraion sample is {}".format(cali_sample.name))

        # full alignment on cali_sample
        # cali_sample.align()
        if verbosity >= 3:
            print("Doing holder on sample {}...".format(cali_sample.name))

        saxs_on()
        get_beamline().modeAlignment()

        cali_sample.xo()  # goto origin
        cali_sample.yo()
        cali_sample.tho()
        cali_sample.align(step=0)

        cali_sample.xo()
        cali_sample.yo()
        cali_sample.tho()

        for sample in self.getSamples():
            sample.setOrigin(["th", "y"])

        for sample in self.getSamples():
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            if step <= 1:
                saxs_on()
                get_beamline().modeAlignment()
            if step <= 2:
                sample.xo()  # goto origin
                sample.yo()
                sample.tho()

            if step <= 5:
                sample.align(step=align_step)

    def measureSamples(self, det="SWAXS", tiling=None, verbosity=3, **md):
        cms.modeMeasurement()

        if det == "SAXS":
            saxs_on()
            for sample in self.getSamples():
                sample.gotoOrigin()

                if sample.incident_angles == None:
                    incident_angles = sample.incident_angles_default
                else:
                    incident_angles = sample.incident_angles

                sample.measureIncidentAngles_Stitch(
                    incident_angles, exposure_time=sample.SAXS_time, tiling=tiling, **md
                )

        elif det == "WAXS":
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
                    incident_angles, exposure_time=sample.WAXS_time, tiling=tiling, **md
                )

                sample.gotoOrigin()

        elif det == "SWAXS":
            swaxs_on()
            for sample in self.getSamples():
                sample.gotoOrigin()

                if sample.incident_angles == None:
                    incident_angles = sample.incident_angles_default
                else:
                    incident_angles = sample.incident_angles

                sample.measureIncidentAngles_Stitch(
                    incident_angles, exposure_time=sample.SWAXS_time, tiling=tiling, **md
                )

        elif det == "BOTH":
            saxs_on()
            for sample in self.getSamples():
                sample.gotoOrigin()

                if sample.incident_angles == None:
                    incident_angles = sample.incident_angles_default
                else:
                    incident_angles = sample.incident_angles

                sample.measureIncidentAngles_Stitch(
                    incident_angles, exposure_time=sample.SAXS_time, tiling=tiling, **md
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
                    incident_angles, exposure_time=sample.WAXS_time, tiling=tiling, **md
                )

                sample.gotoOrigin()

        else:
            print("Wrong det input. Options are SAXS/WAXS/SWAXS/BOTH")

    def doTemperatures(
        self,
        temperature_list=None,
        output_file="Transmission_output",
        int_measure=False,
        wait_time=600,
        WAXS_expo_time=5,
        temperature_probe="A",
        output_channel="1",
        temperature_tolerance=1,
        range=None,
        verbosity=3,
        poling_period=2.0,
        **md,
    ):
        # cms.modeMeasurement()
        if temperature_list == None:
            temperature_list = self.temperature_list

        for index, temperature in enumerate(temperature_list):
            # Set new temperature
            # self.setTemperature(temperature, output_channel=output_channel, verbosity=verbosity)

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

            if index % 4 == 0:
                self.alignSamples_Custom()
                # self.alignSamples()
            elif wait_time is not None:
                time.sleep(wait_time)

            post_to_slack("set to Temperature {}".format(temperature))
            self.measureSamples()
            # self.doSamples(SAXS_expo_time=SAXS_expo_time, verbosity=verbosity, **md)
            if int_measure:
                self.intMeasure(output_file=output_file)

        self.setTemperature(25)


class GIBar_Linkam(GIBar):
    """This class is a sample bar with heating/cooling feature and 6" long bar for grazing-incidence (GI) experiments."""

    # Core methods
    ########################################

    def __init__(self, name="GIBar", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._positional_axis = "x"

        # Set the x and y origin to be the center of slot 8

        # self.xsetOrigin(-71.89405-22.1) # TODO: Update this value
        self._axes["y"].origin = -16.8
        self._axes["y"].origin = 2.06


class WellPlateHolder(PositionalHolder):
    """This class is a sample holder for 96 well plate.
    row: A--E; column: 1--12
    The sample names are like: 'A1', 'D3', 'E12'
    It uses two stages, smx and smy2 to locate the sample.

    """

    # Core methods
    ########################################

    def __init__(self, name="WellPlateHolder", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._positional_axis = ["x", "yy"]

        self._axes["y"].origin = -5  # smy stage should be set with the limit [-5.5, -5]
        self._axes["x"].origin = -49.25
        self._axes["yy"].origin = 3.3

        self.x_spacing = 9  # 9mm seperation both in x and yy direction
        self.yy_spacing = 9

    def _set_axes_definitions(self):
        """Internal function which defines the axes for this stage. This is kept
        as a separate function so that it can be over-ridden easily."""

        # The _axes_definitions array holds a list of dicts, each defining an axis
        super()._set_axes_definitions()

        self._axes_definitions.append(
            {
                "name": "yy",
                "motor": smy2,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": "positive moves stage up",
            }
        )

    def slot(self, sample_number):
        """Moves to the selected slot in the holder."""

        getattr(self, self._positional_axis[0] + "abs")(self.get_slot_position(sample_number))

    def get_slot_position(self, slot):
        """Return the motor position for the requested slot number."""
        # This method should be over-ridden in sub-classes, so as to properly
        # implement the positioning appropriate for that holder.
        # slot is like 'A1', 'D12'

        sample_row = ord(slot[0]) - ord("A")
        sample_column = int(slot[1:]) - 1
        # sample_number = sample_row*12 + sample_column

        return sample_column * self.x_spacing, sample_row * self.yy_spacing

    def addSample(self, sample, sample_number=None):
        """Add a sample to this holder/bar."""

        if sample_number is None:
            if len(self._samples) == 0:
                sample_number = 1
            else:
                ki = [int(key) for key in self._samples.keys()]
                sample_number = np.max(ki) + 1

        if sample_number in self._samples.keys():
            print(
                'Warning: Sample number {} is already defined on holder "{:s}". Use "replaceSample" if you are sure you want to eliminate the existing sample from the holder.'.format(
                    sample_number, self.name
                )
            )

        else:
            self._samples[sample_number] = sample

        self._samples[sample_number] = sample

        sample.set_base_stage(self)
        sample.md["holder_sample_number"] = sample_number

    def addSampleSlot(self, sample, slot):
        """Adds a sample to the specified "slot" (defined/numbered sample
        holding spot on this holder)."""

        self.addSample(sample, sample_number=slot)
        sample.setOrigin(["x"], [self.get_slot_position(slot)[0]])
        sample.setOrigin(["yy"], [self.get_slot_position(slot)[1]])

    def listSamplesPositions(self):
        """Print a list of the current samples associated with this holder/
        bar."""

        for sample_number, sample in self._samples.items():
            pos = getattr(sample, self._positional_axis + "pos")(verbosity=0)
            print("%s: %s (%s = %.3f)" % (str(sample_number), sample.name, self._positional_axis, pos))

    def listSamples(self):
        """Print a list of the current samples associated with this holder/
        bar."""

        for sample_number, sample in sorted(self._samples.items()):
            print("{}: {:s}".format(sample_number, sample.name))

    def gotoAlignedPosition(self):
        """Goes to the currently-defined 'aligned' position for this stage. If
        no specific aligned position is defined, then the zero-point for the stage
        is used instead."""

        # TODO: Optional offsets? (Like goto mark?)
        self.gotoOrigin(axes=self._positional_axis)
        # time.sleep(10)

    def getSample(self, sample_number, verbosity=3):
        """Return the requested sample object from this holder/bar.

        One can provide an integer, in which case the corresponding sample
        (from the holder's inventory) is returned. If a string is provided,
        the closest-matching sample (by name) is returned."""

        if sample_number not in self._samples:
            if verbosity >= 1:
                print("Error: Sample {} not defined.".format(sample_number))
            return None

        sample_match = self._samples[sample_number]

        if verbosity >= 3:
            print("{}: {:s}".format(sample_number, sample_match.name))

        return sample_match

    def namingWellPlate(self, name, row_range=["A", "G"], column_range=[1, 12]):
        """Name the samples in the well plate.
        The format is 'NAME_A05_'
        """
        # md = {
        #     'owner' : 'J. Paloni (MIT) group' ,
        #     'series' : 'various' ,
        #     }

        for row_number in range(ord(row_range[0]), ord(row_range[1]) + 1):
            row = chr(row_number)
            # print(row)
            for column in range(column_range[0], column_range[1] + 1):
                # print(column)
                sample_name = "{}_{}{}".format(name, row, column)
                position = "{}".format(row) + "{0:0=2d}".format(column)
                self.addSampleSlot(SampleTSAXS(sample_name), position)


class PaloniThermalStage(CapillaryHolder):
    """This class is a sample holder made by Jason Paloni.
    It's made by copper with chiller cooling.
    The stage has 5(x) by 2(z) = 10 samples in total.
    It uses two stages, smx and smy2 to locate the sample.

    Note: need to enable the smy2 stage (~4inch travel distance in y direction).

    1. %mov smy -5, set the limit as -5
    2. mount the special smy2 stage onto the sample stage and connect the smy2 motor
    3. enable ('yy') in sam._set_axes_definitions of 94-sample.py
    """

    def __init__(self, name="CapillaryHolder", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        # self._axes['y'].origin -= 0.5357125
        self._positional_axis = ["x", "yy"]

        self._axes["y"].origin = -5
        self._axes["x"].origin = -0.25
        self._axes["yy"].origin = 1.5

        self.x_spacing = 1.375 * 25.4  # 1.375 seperation in x
        self.yy_spacing = 32.13 + 23.25  # 2.25 seperation in y
        # self.yy_spacing = 2.25 *25.4 # 2.25 seperation in y

    def _set_axes_definitions(self):
        """Internal function which defines the axes for this stage. This is kept
        as a separate function so that it can be over-ridden easily."""

        # The _axes_definitions array holds a list of dicts, each defining an axis
        super()._set_axes_definitions()

        self._axes_definitions.append(
            {
                "name": "yy",
                "motor": smy2,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": "positive moves stage up",
            }
        )

    def slot(self, sample_number):
        """Moves to the selected slot in the holder."""

        getattr(self, self._positional_axis[0] + "abs")(self.get_slot_position(sample_number))

    def get_slot_position(self, slot):
        """Return the motor position for the requested slot number."""
        # This method should be over-ridden in sub-classes, so as to properly
        # implement the positioning appropriate for that holder.
        # This is the critical to define the position for the 10 samples.

        if slot < 5.5:
            position_x = 0.0 + slot - 3
        if slot > 5.5:
            position_x = 0.0 + slot - 5 - 3
        position_yy = 0.0 + int(slot / 6) * 1.0

        return position_x * self.x_spacing, position_yy * self.yy_spacing

    def addSampleSlot(self, sample, slot):
        """Adds a sample to the specified "slot" (defined/numbered sample
        holding spot on this holder)."""

        self.addSample(sample, sample_number=slot)
        sample.setOrigin(["x"], [self.get_slot_position(slot)[0]])
        sample.setOrigin(["yy"], [self.get_slot_position(slot)[1]])


class DSCStage(CapillaryHolder):
    """This class is a sample holder for DSC pans.
    The stage has 11(x) by 2(z) = 22 samples in total.
    It has the same dimension as the regular capillary holder.
    It uses two stages, smx and smy to locate the sample.
    X origin is the center. Y origin is the top rack.
    """

    def __init__(self, name="CapillaryHolder", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)
        # TODO: search the origin position and check the spacing
        self._axes["y"].origin = -2.2
        self.x_spacing = 8.9  # 0.325 *25.4 # 0.375 seperation in x
        self.y_spacing = 12.7  # 1/2 inch seperation in y

    def slot(self, sample_number):
        """Moves to the selected slot in the holder."""

        getattr(self, self._positional_axis[0] + "abs")(self.get_slot_position(sample_number))

    def get_slot_position(self, slot):
        """Return the motor position for the requested slot number."""
        # This method should be over-ridden in sub-classes, so as to properly
        # implement the positioning appropriate for that holder.
        # This is the critical to define the position for the 10 samples.

        if slot < 11.5:
            position_x = 0.0 + slot - 6
        elif slot > 11.5:
            position_x = 0.0 + slot - 11 - 6
        position_y = 0.0 + int(slot / 11.5) * 1.0

        return position_x * self.x_spacing, position_y * self.y_spacing

    def addSampleSlot(self, sample, slot):
        """Adds a sample to the specified "slot" (defined/numbered sample
        holding spot on this holder)."""

        self.addSample(sample, sample_number=slot)
        sample.setOrigin(["x"], [self.get_slot_position(slot)[0]])
        sample.setOrigin(["y"], [self.get_slot_position(slot)[1]])


class CapillaryHolderThermal(CapillaryHolder):
    """This class is a sample holder for 2row capillary holder.
    The stage has 15(x) by 2(y) = 39 samples in total.
    It has the same dimension as the regular capillary holder.
    It uses two stages, smx and smy to locate the sample.
    X origin is the center (#8). Y origin is the top rack.
    """

    def __init__(self, name="CapillaryHolder", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)
        # TODO: search the origin position and check the spacing
        self._axes["y"].origin = 8.65
        # self.x_spacing = 9.2  #0.325 *25.4 # 0.375 seperation in x
        self.y_spacing = 12.7  # 1/2 inch seperation in y

    def slot(self, sample_number):
        """Moves to the selected slot in the holder."""

        getattr(self, self._positional_axis[0] + "abs")(self.get_slot_position(sample_number))

    def get_slot_position(self, slot):
        """Return the motor position for the requested slot number."""
        # This method should be over-ridden in sub-classes, so as to properly
        # implement the positioning appropriate for that holder.
        # This is the critical to define the position for the 10 samples.

        if slot < 15.5:
            position_x = 0.0 + slot - 8
        elif slot > 15.5:
            position_x = 0.0 + slot - 15 - 6

        position_y = 0.0 + int(slot / 15.5) * 1.0

        return position_x * self.x_spacing, position_y * self.y_spacing

    def addSampleSlot(self, sample, slot):
        """Adds a sample to the specified "slot" (defined/numbered sample
        holding spot on this holder)."""

        self.addSample(sample, sample_number=slot)
        sample.setOrigin(["x"], [self.get_slot_position(slot)[0]])
        sample.setOrigin(["y"], [self.get_slot_position(slot)[1]])


class GIBarSecondStage(GIBar):
    """This class is a sample bar for grazing-incidence (GI) experiments at the 2nd sample stage."""

    # Core methods
    ########################################

    def __init__(self, name="GIBar", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._axes["x"].origin = 8.65
        self._axes["y"].origin = 22.35
        self._axes["th"].origin = 0.348


class HumidityStage(GIBar):
    """This class is for the humidity stage for multiple samples."""

    def __init__(self, name="GIBar", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._axes["x"].origin = -64.4
        self._axes["y"].origin = 10
        self._axes["th"].origin = 0

    def humidity(self, AI_chan=7, temperature=25, verbosity=3):
        # AI_chan=7, the independent sensor
        # AI_chan=3, the integrated sensor in the flow control panel
        return ioL.readRH(AI_chan=AI_chan, temperature=temperature, verbosity=verbosity)

    def setFlow(self, channel, voltage=0):
        # device = 'A1'
        ioL.set(AO[channel], 0)
        time.sleep(1)
        ioL.set(AO[channel], voltage)
        # MFC.setMode(channeldevice=device, mode=1)


class HumidityTransmissionStage(CapillaryHolder):
    """This class is for the humidity transmission stage for multiple samples."""

    def __init__(self, name="HumidityTransmissionStage", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._positional_axis = "x"

        self.x_spacing = 6.43

        # slot  1; smx = +26.60
        # slot  8; smx = -17.80
        # slot 15; smx = -61.94

        self._axes["x"].origin = -64.4
        self._axes["y"].origin = 10
        self._axes["th"].origin = 0

    def humidity(self, AI_chan=8, temperature=25, verbosity=3):
        # AI_chan=7, the independent sensor
        # AI_chan=3, the integrated sensor in the flow control panel
        return ioL.readRH(AI_chan=AI_chan, temperature=temperature, verbosity=verbosity)

    def setFlow(self, channel, voltage=0):
        # device = 'A1'
        ioL.set(AO[channel], 0)
        time.sleep(1)
        ioL.set(AO[channel], voltage)
        # MFC.setMode(channeldevice=device, mode=1)


class InstecStage60(CapillaryHolder):
    def __init__(self, name="CapillaryHolder", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._axes["y"].origin = 13.4  # smy position for slot 4 of 7-capillary cassette
        self._axes["x"].origin = -15.6  # smx position for slot 4 of 7-capillary cassette

        self.y_pos_default = []

        self.x_spacing = 0.2 * 25.4  # 0.2" seperation in x

    def get_slot_position(self, slot):
        """Return the motor position for the requested slot number."""
        # This method should be over-ridden in sub-classes, so as to properly
        # implement the positioning appropriate for that holder.

        position_x = 0.0 + slot - 4

        return position_x * self.x_spacing

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


class OffCenteredHoder(GIBar):
    """The special sample holder for GTSAXS and TSAXS along thin film edges.
    It contains 17 slots with fixed position, similar to Capillary Holders. But the holder dimensions and functions are similar to GIBar.

    """

    def __init__(self, name="OffCenteredHoder", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._positional_axis = "x"

        self.x_spacing = 6.342  # 3.5 inches / 16 spaces

        # slot  1; smx = +26.60
        # slot  8; smx = -17.80
        # slot 15; smx = -61.94

        # Set the x and y origin to be the center of slot 8
        self._axes["x"].origin = -17.2  # -0.3
        self._axes["y"].origin = 5.38  # -0.1
        # self.ysetOrigin(5.38)
        # self.xsetOrigin(-17.2)

        self.mark("right edge", x=+54.4)
        self.mark("left edge", x=-54.4)
        self.mark("bottom edge", y=-12.71)
        self.mark("center", x=0, y=0)

        # measurement measure_settings
        self.detector = None
        self.exposure_time = None
        self.incident_angles = None
        self.tiling = None
        self.measure_setting = {}

    def get_slot_position(self, slot):
        """Return the motor position for the requested slot number."""

        return +1 * self.x_spacing * (slot - 8)
