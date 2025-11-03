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


config_load()


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


class SampleCDSAXS(SampleCDSAXS_Generic):
    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)
        self.naming_scheme = ["name", "extra", "phi", "exposure_time"]


class Sample(SampleTSAXS):
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


# cms.SAXS.setCalibration([247.5, 528.0], 2.395, [0, 27.52]) # 2017-01-30, 17 keV
# cms.SAXS.setCalibration([263.5, 552.0], 5.038, [0.00, 35.00]) # 2017-02-08, 13.5 keV
cms.SAXS.setCalibration([379.0, 552.0], 5.038, [20.00, 35.00])  # 2017-02-08, 13.5 keV

print("\n\n\nReminders:")
print("    Define your detectors using, e.g.: detselect(pilatus2M)")
print("    Reload your user-specific script, e.g.: %run -i /nsls2/xf11bm/data/2017_2/user_group/user.py")
print("\n")

if False:
    # For testing (and as examples...)
    # %run -i /opt/ipython_profiles/profile_collection/startup/98-user.py

    hol = CapillaryHolder(base=stg)
    hol.addSampleSlot(Sample("test_sample_01"), 1.0)
    hol.addSampleSlot(Sample("test_sample_02"), 3.0)
    hol.addSampleSlot(Sample("test_sample_03"), 5.0)

    sam = hol.getSample(1)
