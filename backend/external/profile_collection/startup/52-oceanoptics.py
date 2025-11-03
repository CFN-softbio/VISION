# XF:11BM-ES{OceanOptics:Spec-1}:BUFF_CAPACITY
# XF:11BM-ES{OceanOptics:Spec-1}:BUFF_CAPACITY_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:BUFF_ELEMENT_COUNT_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:BUFF_MAX_CAPACITY_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:BUFF_MIN_CAPACITY_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:COLLECT_MODE
# XF:11BM-ES{OceanOptics:Spec-1}:COLLECT_MODE_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:COLLECT_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:CONNECTED_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:CORRECTION
# XF:11BM-ES{OceanOptics:Spec-1}:CORRECTION_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:CURR_TEC_TEMP_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:DARK_AVAILABLE_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:EDC
# XF:11BM-ES{OceanOptics:Spec-1}:EDC_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:FEATURES_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:INTEGRATION_TIME
# XF:11BM-ES{OceanOptics:Spec-1}:INTEGRATION_TIME_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:INT_MAX_TIME_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:INT_MIN_TIME_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:LIGHT_SOURCE
# XF:11BM-ES{OceanOptics:Spec-1}:LIGHT_SOURCE_COUNT_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:LIGHT_SOURCE_INTENSITY
# XF:11BM-ES{OceanOptics:Spec-1}:LIGHT_SOURCE_INTENSITY_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:LIGHT_SOURCE_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:MODEL
# XF:11BM-ES{OceanOptics:Spec-1}:NLC
# XF:11BM-ES{OceanOptics:Spec-1}:NLC_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:NUM_SPECTRA
# XF:11BM-ES{OceanOptics:Spec-1}:NUM_SPECTRA_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:OUTPUT
# XF:11BM-ES{OceanOptics:Spec-1}:REF_AVAILABLE_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:SERIAL
# XF:11BM-ES{OceanOptics:Spec-1}:SHUTTER
# XF:11BM-ES{OceanOptics:Spec-1}:SHUTTER_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:SPECTRA_COLLECTED_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:SPECTRUM_TYPE
# XF:11BM-ES{OceanOptics:Spec-1}:SPECTRUM_TYPE_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:STATUS
# XF:11BM-ES{OceanOptics:Spec-1}:STATUS_MSG
# XF:11BM-ES{OceanOptics:Spec-1}:STROBE
# XF:11BM-ES{OceanOptics:Spec-1}:STROBE_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:SUBTRACT_FORMAT
# XF:11BM-ES{OceanOptics:Spec-1}:SUBTRACT_FORMAT_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:TEC
# XF:11BM-ES{OceanOptics:Spec-1}:TEC_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:TEC_TEMP
# XF:11BM-ES{OceanOptics:Spec-1}:TEC_TEMP_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:TRIGGER_MODE
# XF:11BM-ES{OceanOptics:Spec-1}:TRIGGER_MODE_RBV
# XF:11BM-ES{OceanOptics:Spec-1}:X_AXIS
# XF:11BM-ES{OceanOptics:Spec-1}:X_AXIS_FORMAT
# XF:11BM-ES{OceanOptics:Spec-1}:X_AXIS_FORMAT_RBV

class OpticalSpectometer(object):
    
    def __init__(self, ii):
        self.name = "AO_Chan{}".format(ii)
        # self.sp = 'XF:11BM-ES{{Ecat:AO{}}}'.format(ii)
        self.sp = "XF:11BM-ES{{Ecat:AO1}}{}".format(ii)
        # self.sts = 'XF:11BMB-ES{}AO:{}-RB'.format('{IO}', ii)
        self.sts = self.sp
        self.PV = self.sp
        self.signal = EpicsSignal(self.PV)

# from ophyd import Device, Component as Cpt, EpicsSignal, EpicsSignalRO


class OceanOpticsSpectrometer(Device):

    # Define all the EPICS signals and read-only signals for the spectrometer

    buff_capacity = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:BUFF_CAPACITY')

    buff_capacity_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:BUFF_CAPACITY_RBV')

    buff_element_count_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:BUFF_ELEMENT_COUNT_RBV')

    buff_max_capacity_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:BUFF_MAX_CAPACITY_RBV')

    buff_min_capacity_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:BUFF_MIN_CAPACITY_RBV')

    collect_mode = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:COLLECT_MODE')

    collect_mode_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:COLLECT_MODE_RBV')

    collect_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:COLLECT_RBV')

    connected_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:CONNECTED_RBV')

    correction = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:CORRECTION')

    correction_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:CORRECTION_RBV')

    curr_tec_temp_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:CURR_TEC_TEMP_RBV')

    dark_available_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:DARK_AVAILABLE_RBV')

    edc = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:EDC')

    edc_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:EDC_RBV')

    features_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:FEATURES_RBV')

    integration_time = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:INTEGRATION_TIME')

    integration_time_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:INTEGRATION_TIME_RBV')

    int_max_time_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:INT_MAX_TIME_RBV')

    int_min_time_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:INT_MIN_TIME_RBV')

    light_source = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:LIGHT_SOURCE')

    light_source_count_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:LIGHT_SOURCE_COUNT_RBV')

    light_source_intensity = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:LIGHT_SOURCE_INTENSITY')

    light_source_intensity_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:LIGHT_SOURCE_INTENSITY_RBV')

    light_source_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:LIGHT_SOURCE_RBV')

    model = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:MODEL')

    nlc = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:NLC')

    nlc_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:NLC_RBV')

    num_spectra = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:NUM_SPECTRA')

    num_spectra_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:NUM_SPECTRA_RBV')

    output = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:OUTPUT')

    ref_available_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:REF_AVAILABLE_RBV')

    serial = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:SERIAL')

    shutter = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:SHUTTER')

    shutter_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:SHUTTER_RBV')

    spectra_collected_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:SPECTRA_COLLECTED_RBV')

    spectrum_type = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:SPECTRUM_TYPE')

    spectrum_type_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:SPECTRUM_TYPE_RBV')

    status = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:STATUS')

    status_msg = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:STATUS_MSG')

    strobe = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:STROBE')

    strobe_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:STROBE_RBV')

    subtract_format = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:SUBTRACT_FORMAT')

    subtract_format_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:SUBTRACT_FORMAT_RBV')

    tec = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:TEC')

    tec_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:TEC_RBV')

    tec_temp = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:TEC_TEMP')

    tec_temp_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:TEC_TEMP_RBV')

    trigger_mode = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:TRIGGER_MODE')

    trigger_mode_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:TRIGGER_MODE_RBV')

    x_axis = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:X_AXIS')

    x_axis_format = Cpt(EpicsSignal, 'XF:11BM-ES{OceanOptics:Spec-1}:X_AXIS_FORMAT')

    x_axis_format_rbv = Cpt(EpicsSignalRO, 'XF:11BM-ES{OceanOptics:Spec-1}:X_AXIS_FORMAT_RBV')


    # Function to get the value of BUFF_CAPACITY

    def get_buff_capacity(self):

        return self.buff_capacity.get()


    # Function to set the value of BUFF_CAPACITY

    def set_buff_capacity(self, value):

        self.buff_capacity.put(value)


    # Function to get the readback value of BUFF_CAPACITY

    def get_buff_capacity_rbv(self):

        return self.buff_capacity_rbv.get()


    # Function to get the readback value of BUFF_ELEMENT_COUNT

    def get_buff_element_count_rbv(self):

        return self.buff_element_count_rbv.get()


    # Function to get the readback value of BUFF_MAX_CAPACITY

    def get_buff_max_capacity_rbv(self):

        return self.buff_max_capacity_rbv.get()


    # Function to get the readback value of BUFF_MIN_CAPACITY

    def get_buff_min_capacity_rbv(self):

        return self.buff_min_capacity_rbv.get()


    # Function to get the value of COLLECT_MODE

    def get_collect_mode(self):

        return self.collect_mode.get()


    # Function to set the value of COLLECT_MODE

    def set_collect_mode(self, value):

        return self.set_collect_mode.set(value)
    

OOS = OceanOpticsSpectrometer(name='OOS')