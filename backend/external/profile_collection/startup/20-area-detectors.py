# import time as ttime  # tea time
# from datetime import datetime
from ophyd import (
    ProsilicaDetector,
    SingleTrigger,
    ImagePlugin,
    DetectorBase,
    HDF5Plugin,
    AreaDetector,
    EpicsSignal,
    EpicsSignalRO,
    ROIPlugin,
    TransformPlugin,
    ProcessPlugin,
    PilatusDetector,
    ProsilicaDetectorCam,
    PilatusDetectorCam,
    StatsPlugin,
)
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.areadetector.plugins import TIFFPlugin_V33
from ophyd.areadetector.base import ADComponent, EpicsSignalWithRBV
from ophyd.areadetector.filestore_mixins import FileStoreTIFFIterativeWrite
from ophyd import Component as Cpt, Signal
from ophyd.utils import set_and_wait
from nslsii.ad33 import SingleTriggerV33, StatsPluginV33

from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite
from ophyd.areadetector.plugins import HDF5Plugin #,register_plugin,PluginBase


def print_area_detector_port_graph(G):
    print("\nArea Detector Port Graph:")
    print("------------------------")
    print("Nodes:")
    for node in sorted(G.nodes()):
        print(f"  • {node}")
    print("\nConnections:")
    for edge in sorted(G.edges()):
        print(f"  • {edge[0]} → {edge[1]}")
    print("------------------------\n")


print(f'Loading {__file__}')

# import filestore.api as fs


# class Elm(SingleTrigger, DetectorBase):
#   pass


Pilatus2M_on = True
Camera_on=True
Pilatus300_on = False
# ONLY 1 Pilatus800 will be turned on at the same time. changed by RL, 20210831
#
print("The current stage is {}".format(beamline_stage))
# beamline_stage = 'default' #for using chamber
# beamline_stage = 'open_MAXS' #for using the upstream stage
if beamline_stage == "open_MAXS" or beamline_stage == "BigHuber":
    Pilatus800_on = False
    Pilatus800_2_on = True
elif beamline_stage == "default":
    Pilatus800_on = True
    Pilatus800_2_on = False

# Pilatus800_on = False
Pilatus800_2_on = True
# Pilatus800_2_on = False
# Pilatus2M_on = False

''' H5Plugin = HDF5Plugin  ''' 



class TIFFPluginWithFileStore(TIFFPlugin_V33, FileStoreTIFFIterativeWrite):
    pass

class HDF5PluginWithFileStore(HDF5Plugin, FileStoreHDF5IterativeWrite):
    pass

class ProsilicaDetectorCamV33(ProsilicaDetectorCam):
    """This is used to update the standard prosilica to AD33."""

    wait_for_plugins = Cpt(EpicsSignal, "WaitForPlugins", string=True, kind="config")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs["wait_for_plugins"] = "Yes"

    def ensure_nonblocking(self):
        self.stage_sigs["wait_for_plugins"] = "Yes"
        for c in self.parent.component_names:
            cpt = getattr(self.parent, c)
            if cpt is self:
                continue
            if hasattr(cpt, "ensure_nonblocking"):
                cpt.ensure_nonblocking()


# class Pilatus2M(SingleTrigger, PilatusDetector):
#     image = Cpt(ImagePlugin, 'image1:')
#     stats1 = Cpt(StatsPlugin, 'Stats1:')
#     stats2 = Cpt(StatsPlugin, 'Stats2:')
#     stats3 = Cpt(StatsPlugin, 'Stats3:')
#     stats4 = Cpt(StatsPlugin, 'Stats4:')
#     stats5 = Cpt(StatsPlugin, 'Stats5:')
#     roi1 = Cpt(ROIPlugin, 'ROI1:')
#     roi2 = Cpt(ROIPlugin, 'ROI2:')
#     roi3 = Cpt(ROIPlugin, 'ROI3:')
#     roi4 = Cpt(ROIPlugin, 'ROI4:')
#     proc1 = Cpt(ProcessPlugin, 'Proc1:')

#     tiff = Cpt(TIFFPluginWithFileStore,
#                suffix='TIFF1:',
#             #    write_path_template='/nsls2/data/cms/legacy/xf11bm/Pilatus2M/%Y/%m/%d/',
#                write_path_template='/nsls2/xf11bm/Pilatus2M/%Y/%m/%d/',
#                root='/nsls2/xf11bm')

#     def setExposureTime(self, exposure_time, verbosity=3):
#         caput('XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime', exposure_time)
#         caput('XF:11BMB-ES{Det:PIL2M}:cam1:AcquirePeriod', exposure_time+0.1)


class StandardProsilica(SingleTrigger, ProsilicaDetector):
    # tiff = Cpt(TIFFPluginWithFileStore,
    #           suffix='TIFF1:',
    #           write_path_template='/XF11ID/data/')
    image = Cpt(ImagePlugin, "image1:")
    stats1 = Cpt(StatsPluginV33, "Stats1:")
    stats2 = Cpt(StatsPluginV33, "Stats2:")
    stats3 = Cpt(StatsPluginV33, "Stats3:")
    stats4 = Cpt(StatsPluginV33, "Stats4:")
    stats5 = Cpt(StatsPluginV33, "Stats5:")
    trans1 = Cpt(TransformPlugin, "Trans1:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")
    proc1 = Cpt(ProcessPlugin, "Proc1:")


class StandardProsilicaV33(SingleTriggerV33, ProsilicaDetector):
    # tiff = Cpt(TIFFPluginWithFileStore,
    #           suffix='TIFF1:',
    #           write_path_template='/XF11ID/data/')
    cam = Cpt(ProsilicaDetectorCamV33, "cam1:")
    image = Cpt(ImagePlugin, "image1:")
    stats1 = Cpt(StatsPluginV33, "Stats1:")
    stats2 = Cpt(StatsPluginV33, "Stats2:")
    stats3 = Cpt(StatsPluginV33, "Stats3:")
    stats4 = Cpt(StatsPluginV33, "Stats4:")
    stats5 = Cpt(StatsPluginV33, "Stats5:")
    trans1 = Cpt(TransformPlugin, "Trans1:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")
    proc1 = Cpt(ProcessPlugin, "Proc1:")


class PilatusDetectorCamV33(PilatusDetectorCam):
    """This is used to update the standard prosilica to AD33."""

    wait_for_plugins = Cpt(EpicsSignal, "WaitForPlugins", string=True, kind="config")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs["wait_for_plugins"] = "Yes"

    def ensure_nonblocking(self):
        self.stage_sigs["wait_for_plugins"] = "Yes"
        for c in self.parent.component_names:
            cpt = getattr(self.parent, c)
            if cpt is self:
                continue
            if hasattr(cpt, "ensure_nonblocking"):
                cpt.ensure_nonblocking()


# class Pilatus(SingleTrigger, PilatusDetector):
#     image = Cpt(ImagePlugin, 'image1:')
#     stats1 = Cpt(StatsPluginV33, 'Stats1:')
#     stats2 = Cpt(StatsPluginV33, 'Stats2:')
#     stats3 = Cpt(StatsPluginV33, 'Stats3:')
#     stats4 = Cpt(StatsPluginV33, 'Stats4:')
#     stats5 = Cpt(StatsPluginV33, 'Stats5:')
#     roi1 = Cpt(ROIPlugin, 'ROI1:')
#     roi2 = Cpt(ROIPlugin, 'ROI2:')
#     roi3 = Cpt(ROIPlugin, 'ROI3:')
#     roi4 = Cpt(ROIPlugin, 'ROI4:')
#     proc1 = Cpt(ProcessPlugin, 'Proc1:')

#     tiff = Cpt(TIFFPluginWithFileStore,
#                suffix='TIFF1:',
#                write_path_template='/nsls2/xf11bm/Pilatus300/%Y/%m/%d/',
#                root='/nsls2/xf11bm')

#     def setExposureTime(self, exposure_time, verbosity=3):
#         caput('XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime', exposure_time)
#         caput('XF:11BMB-ES{Det:SAXS}:cam1:AcquirePeriod', exposure_time+0.1)


class PilatusV33(SingleTriggerV33, PilatusDetector):
    cam = Cpt(PilatusDetectorCamV33, "cam1:")
    image = Cpt(ImagePlugin, "image1:")
    stats1 = Cpt(StatsPluginV33, "Stats1:")
    stats2 = Cpt(StatsPluginV33, "Stats2:")
    stats3 = Cpt(StatsPluginV33, "Stats3:")
    stats4 = Cpt(StatsPluginV33, "Stats4:")
    stats5 = Cpt(StatsPluginV33, "Stats5:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")
    proc1 = Cpt(ProcessPlugin, "Proc1:")

    tiff = Cpt(
        TIFFPluginWithFileStore,
        suffix="TIFF1:",
        write_path_template="/nsls2/xf11bm/Pilatus300/%Y/%m/%d/",
        root="/nsls2/xf11bm",
    )

    def setExposureTime(self, exposure_time, verbosity=3):
        yield from mv(
            self.cam.acquire_time,
            exposure_time,
            self.cam.acquire_period,
            exposure_time + 0.1,
        )
        # caput('XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime', exposure_time)
        # caput('XF:11BMB-ES{Det:SAXS}:cam1:AcquirePeriod', exposure_time+0.1)

    def setExposurePeriod(self, exposure_period, verbosity=3):
        yield from mv(self.cam.acquire_period, exposure_period)

    def setExposureNumber(self, exposure_number, verbosity=3):
        yield from mv(self.cam.num_images, exposure_number)


# class Pilatus800V33(PilatusV33):
#     tiff = Cpt(TIFFPluginWithFileStore,
#                suffix='TIFF1:',
#             #    write_path_template='/Pilatus800K/%Y/%m/%d/',
#             #    read_path_template='/nsls2/xf11bm/Pilatus800K/%Y/%m/%d/',
#             #    root='/nsls2/xf11bm')
#                write_path_template='/nsls2/data/cms/legacy/xf11bm/Pilatus800/%Y/%m/%d/',
#             #    read_path_template='/nsls2/data/cms/legacy/xf11bm/Pilatus800/%Y/%m/%d/',
#                root='/nsls2/data/cms/legacy/xf11bm')


class Pilatus800V33(SingleTriggerV33, PilatusDetector):
    cam = Cpt(PilatusDetectorCamV33, "cam1:")
    image = Cpt(ImagePlugin, "image1:")
    stats1 = Cpt(StatsPluginV33, "Stats1:")
    stats2 = Cpt(StatsPluginV33, "Stats2:")
    stats3 = Cpt(StatsPluginV33, "Stats3:")
    stats4 = Cpt(StatsPluginV33, "Stats4:")
    stats5 = Cpt(StatsPluginV33, "Stats5:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")
    proc1 = Cpt(ProcessPlugin, "Proc1:")

    tiff = Cpt(
        TIFFPluginWithFileStore,
        suffix="TIFF1:",
        #    write_path_template='/nsls2/xf11bm/Pilatus2M/%Y/%m/%d/',     # GPFS client
        # write_path_template='/Pilatus2M/%Y/%m/%d/',                 # NSF-mount of GPFS directory
        #    root='/nsls2/xf11bm'
        read_path_template="/nsls2/data/cms/legacy/xf11bm/Pilatus800/%Y/%m/%d/",
        write_path_template="/nsls2/data/cms/legacy/xf11bm/Pilatus800/%Y/%m/%d/",
        root="/nsls2/data/cms/legacy/xf11bm",
    )
    # root='/')

    def setExposureTime(self, exposure_time, verbosity=3):
        yield from mv(
            self.cam.acquire_time,
            exposure_time,
            self.cam.acquire_period,
            exposure_time + 0.1,
        )
        # self.cam.acquire_time.put(exposure_time)
        # self.cam.acquire_period.put(exposure_time+.1)
        # caput('XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime', exposure_time)
        # caput('XF:11BMB-ES{Det:PIL2M}:cam1:AcquirePeriod', exposure_time+0.1)

    def setExposurePeriod(self, exposure_period, verbosity=3):
        yield from mv(self.cam.acquire_period, exposure_period)

    def setExposureNumber(self, exposure_number, verbosity=3):
        yield from mv(self.cam.num_images, exposure_number)


class Pilatus8002V33(PilatusV33):
    cam = Cpt(PilatusDetectorCamV33, "cam1:")
    tiff = Cpt(
        TIFFPluginWithFileStore,
        suffix="TIFF1:",
        #    write_path_template='/Pilatus800K/%Y/%m/%d/',
        #    read_path_template='/nsls2/xf11bm/Pilatus800K/%Y/%m/%d/',
        #    root='/nsls2/xf11bm')
        write_path_template="/nsls2/data/cms/legacy/xf11bm/Pilatus800_2/%Y/%m/%d/",
        root="/nsls2/data/cms/legacy/xf11bm",
        # write_path_template="/ramdisk/",
        # root="/ramdisk/",
    )


# class Pilatus300V33(PilatusV33):
#     tiff = Cpt(
#         TIFFPluginWithFileStore,
#         suffix="TIFF1:",
#         write_path_template="/nsls2/data/cms/legacy/xf11bm/Pilatus300/%Y/%m/%d/",
#         #    read_path_template='/nsls2/xf11bm/Pilatus300/%Y/%m/%d/',
#         #    root='/nsls2/xf11bm')
#         #    read_path_template='/nsls2/data/cms/legacy/xf11bm/Pilatus300/%Y/%m/%d/',
#         read_path_template="/nsls2/data/cms/legacy/xf11bm/Pilatus300/%Y/%m/%d/",
#         root="/nsls2/data/cms/legacy/xf11bm",
#     )


class Pilatus2M(SingleTrigger, PilatusDetector):
    image = Cpt(ImagePlugin, "image1:")
    stats1 = Cpt(StatsPluginV33, "Stats1:")
    stats2 = Cpt(StatsPluginV33, "Stats2:")
    stats3 = Cpt(StatsPluginV33, "Stats3:")
    stats4 = Cpt(StatsPluginV33, "Stats4:")
    stats5 = Cpt(StatsPluginV33, "Stats5:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")
    proc1 = Cpt(ProcessPlugin, "Proc1:")

    trans1 = Cpt(TransformPlugin, "Trans1:")

    tiff = Cpt(
        TIFFPluginWithFileStore,
        suffix="TIFF1:",
        write_path_template="/nsls2/xf11bm/Pilatus2M/%Y/%m/%d/",
        root="/nsls2/xf11bm",
    )

    def setExposureTime(self, exposure_time, verbosity=3):
        # how to do this with stage_sigs (warning, need to change this every time
        # if you set)
        # RE(pilatus2M.setEposure(1))   ---format
        # self.cam.stage_sigs['acquire_time'] = exposure_time
        # self.cam.stage_sigs['acquire_period'] = exposure_time+.1

        yield from mv(
            self.cam.acquire_time,
            exposure_time,
            self.cam.acquire_period,
            exposure_time + 0.1,
        )
        # yield from mv(self.cam.acquire_period, exposure_time+0.1)


class Pilatus2MV33(SingleTriggerV33, PilatusDetector):
    cam = Cpt(PilatusDetectorCamV33, "cam1:")
    image = Cpt(ImagePlugin, "image1:")
    stats1 = Cpt(StatsPluginV33, "Stats1:")
    stats2 = Cpt(StatsPluginV33, "Stats2:")
    stats3 = Cpt(StatsPluginV33, "Stats3:")
    stats4 = Cpt(StatsPluginV33, "Stats4:")
    stats5 = Cpt(StatsPluginV33, "Stats5:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")
    proc1 = Cpt(ProcessPlugin, "Proc1:")
    trans1 = Cpt(TransformPlugin, "Trans1:")

    tiff = Cpt(
        TIFFPluginWithFileStore,
        suffix="TIFF1:",
        #    write_path_template='/nsls2/xf11bm/Pilatus2M/%Y/%m/%d/',     # GPFS client
        # write_path_template='/Pilatus2M/%Y/%m/%d/',                 # NSF-mount of GPFS directory
        #    root='/nsls2/xf11bm'
        write_path_template="/nsls2/data/cms/legacy/xf11bm/Pilatus2M/%Y/%m/%d/",  # Lustre client
        root="/nsls2/data/cms/legacy/xf11bm",
    )
    # root='/')
    def setExposureTime(self, exposure_time, verbosity=3):
        yield from mv(
            self.cam.acquire_time,
            exposure_time,
            # self.cam.acquire_period,
            # exposure_time + 0.1,
        )    

    def setExposurePeriod(self, exposure_period, verbosity=3):
        yield from mv(self.cam.acquire_period, exposure_period)

    def setExposureNumber(self, exposure_number, verbosity=3):
        yield from mv(self.cam.num_images, exposure_number)

class Pilatus2MV33_h5(SingleTriggerV33, PilatusDetector):
    cam = Cpt(PilatusDetectorCamV33, "cam1:")
    image = Cpt(ImagePlugin, "image1:")
    stats1 = Cpt(StatsPluginV33, "Stats1:")
    stats2 = Cpt(StatsPluginV33, "Stats2:")
    stats3 = Cpt(StatsPluginV33, "Stats3:")
    stats4 = Cpt(StatsPluginV33, "Stats4:")
    stats5 = Cpt(StatsPluginV33, "Stats5:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")
    proc1 = Cpt(ProcessPlugin, "Proc1:")
    trans1 = Cpt(TransformPlugin, "Trans1:")

    # tiff = Cpt(
    #     TIFFPluginWithFileStore,
    #     suffix="TIFF1:",
    #     #    write_path_template='/nsls2/xf11bm/Pilatus2M/%Y/%m/%d/',     # GPFS client
    #     # write_path_template='/Pilatus2M/%Y/%m/%d/',                 # NSF-mount of GPFS directory
    #     #    root='/nsls2/xf11bm'
    #     write_path_template="/nsls2/data/cms/legacy/xf11bm/Pilatus2M/%Y/%m/%d/",  # Lustre client
    #     root="/nsls2/data/cms/legacy/xf11bm",
    # )

    h5 = Cpt(
        HDF5PluginWithFileStore,
        suffix="HDF1:",
        #    write_path_template='/nsls2/xf11bm/Pilatus2M/%Y/%m/%d/',     # GPFS client
        # write_path_template='/Pilatus2M/%Y/%m/%d/',                 # NSF-mount of GPFS directory
        #    root='/nsls2/xf11bm'
        write_path_template="/nsls2/data/cms/legacy/xf11bm/Pilatus2M/%Y/%m/%d/",  # Lustre client
        root="/nsls2/data/cms/legacy/xf11bm",
    )


    def setExposureTime(self, exposure_time, verbosity=3):
        yield from mv(
            self.cam.acquire_time,
            exposure_time,
            self.cam.acquire_period,
            exposure_time + 0.1,
        )
        # self.cam.acquire_time.put(exposure_time)
        # self.cam.acquire_period.put(exposure_time+.1)
        # caput('XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime', exposure_time)
        # caput('XF:11BMB-ES{Det:PIL2M}:cam1:AcquirePeriod', exposure_time+0.1)

    def setExposurePeriod(self, exposure_period, verbosity=3):
        yield from mv(self.cam.acquire_period, exposure_period)

    def setExposureNumber(self, exposure_number, verbosity=3):
        yield from mv(self.cam.num_images, exposure_number)

    def stage(self):
        error = None
        # wrap the staging process in a retry loop
        for retry in range(5):
            try:
                return super().stage()
            except TimeoutError as err:
                # Staging failed becasue the IOC did not answer
                # some request in a resonable time
                # Stash the exception as the variable 'error'
                error = err
            else:
                # Staging worked. Strop retyring.
                break
        else:
            # We exhausted all retires and none worked.
            # Raise the error captured above to produce a useful error message.
            raise error


# print( 'This is the 20-area-dec py.')
# class StandardProsilicaWithTIFF(StandardProsilica):
#    tiff = Cpt(TIFFPluginWithFileStore,
#               suffix='TIFF1:',
#               write_path_template='/nsls2/xf11bm/data/%Y/%m/%d/',
#               root='/nsls2/xf11bm/')


## This renaming should be reversed: no correspondance between CSS screens, PV names and ophyd....
# xray_eye1 = StandardProsilica('XF:11IDA-BI{Bpm:1-Cam:1}', name='xray_eye1')
# xray_eye2 = StandardProsilica('XF:11IDB-BI{Mon:1-Cam:1}', name='xray_eye2')
# xray_eye3 = StandardProsilica('XF:11IDB-BI{Cam:08}', name='xray_eye3')
# xray_eye1_writing = StandardProsilicaWithTIFF('XF:11IDA-BI{Bpm:1-Cam:1}', name='xray_eye1')
# xray_eye2_writing = StandardProsilicaWithTIFF('XF:11IDB-BI{Mon:1-Cam:1}', name='xray_eye2')
# xray_eye3_writing = StandardProsilicaWithTIFF('XF:11IDB-BI{Cam:08}', name='xray_eye3')
# fs1 = StandardProsilica('XF:11IDA-BI{FS:1-Cam:1}', name='fs1')
# fs2 = StandardProsilica('XF:11IDA-BI{FS:2-Cam:1}', name='fs2')
# fs_wbs = StandardProsilica('XF:11IDA-BI{BS:WB-Cam:1}', name='fs_wbs')
# dcm_cam = StandardProsilica('XF:11IDA-BI{Mono:DCM-Cam:1}', name='dcm_cam')
# fs_pbs = StandardProsilica('XF:11IDA-BI{BS:PB-Cam:1}', name='fs_pbs')
# elm = Elm('XF:11IDA-BI{AH401B}AH401B:',)


import time


if Camera_on==True:

    # time.sleep(1)
    # fs1 = StandardProsilicaV33('XF:11BMA-BI{FS:1-Cam:1}', name='fs1')
    time.sleep(1)
    fs2 = StandardProsilicaV33("XF:11BMA-BI{FS:2-Cam:1}", name="fs2")
    time.sleep(2)
    fs3 = StandardProsilicaV33("XF:11BMB-BI{FS:3-Cam:1}", name="fs3")
    time.sleep(2)
    fs4 = StandardProsilicaV33("XF:11BMB-BI{FS:4-Cam:1}", name="fs4")
    time.sleep(2)
    fs5 = StandardProsilicaV33("XF:11BMB-BI{FS:Test-Cam:1}", name="fs5")

    all_standard_pros = [fs2, fs3, fs4]


    for cam_number, fs in zip([2, 3, 4], all_standard_pros):
        G, port_dict = fs.get_asyn_digraph()
        print(fs.cam.port_name.get())
        print_area_detector_port_graph(G)
        print(port_dict)
        cam = port_dict["cam{:02}".format(cam_number)]
        for v in port_dict.values():
            try:
                if v.nd_array_port.get() == "CAM":
                    v.nd_array_port.set("cam{:02}".format(cam_number))
            except AttributeError:
                pass
        fs.validate_asyn_ports()

# class StandardsimDetectorV33(SingleTriggerV33, ProsilicaDetector):
## tiff = Cpt(TIFFPluginWithFileStore,
##           suffix='TIFF1:',
##           write_path_template='/XF11ID/data/')
# cam = Cpt(ProsilicaDetectorCamV33, 'cam1:')
# image = Cpt(ImagePlugin, 'image1:')
# stats1 = Cpt(StatsPluginV33, 'Stats1:')
# stats2 = Cpt(StatsPluginV33, 'Stats2:')
# stats3 = Cpt(StatsPluginV33, 'Stats3:')
# stats4 = Cpt(StatsPluginV33, 'Stats4:')
# stats5 = Cpt(StatsPluginV33, 'Stats5:')
# trans1 = Cpt(TransformPlugin, 'Trans1:')
# roi1 = Cpt(ROIPlugin, 'ROI1:')
# roi2 = Cpt(ROIPlugin, 'ROI2:')
# roi3 = Cpt(ROIPlugin, 'ROI3:')
# roi4 = Cpt(ROIPlugin, 'ROI4:')
# proc1 = Cpt(ProcessPlugin, 'Proc1:')

# simDetector = StandardProsilicaV33('13SIM1:', name='simDetector')


# all_standard_pros = [fs1, fs2, fs3, fs4, fs5, simDetector]
# all_standard_pros = [fs1, fs2, fs3, fs4]
# all_standard_pros = [fs2, fs3, fs4]
# all_standard_pros = [fs2, fs4]


# for camera in all_standard_pros:
#     camera.read_attrs = ['stats1', 'stats2','stats3','stats4','stats5']
#     # camera.tiff.read_attrs = []  # leaving just the 'image'
#     for stats_name in ['stats1', 'stats2','stats3','stats4','stats5']:
#         stats_plugin = getattr(camera, stats_name)
#         stats_plugin.read_attrs = ['total']
#         #camera.stage_sigs[stats_plugin.blocking_callbacks] = 1

# camera.stage_sigs[camera.roi1.blocking_callbacks] = 1
# camera.stage_sigs[camera.trans1.blocking_callbacks] = 1
# camera.cam.ensure_nonblocking()

# camera.stage_sigs[camera.cam.trigger_mode] = 'Fixed Rate'


# for camera in [xray_eye1_writing, xray_eye2_writing, xray_eye3_writing]:
#    camera.read_attrs.append('tiff')
#    camera.tiff.read_attrs = []

# pilatus300 section is marked out as the detector sever cannot be reached after AC power outrage. 121417-RL
# pilatus300 section is unmarked.  032018-MF
# pilatus300 section is marked out for bluesky upgrade.  010819-RL




# pilatus300 section
# if True:
if Pilatus300_on == True:
    pilatus300 = Pilatus300V33("XF:11BMB-ES{Det:SAXS}:", name="pilatus300")
    # pilatus300 = PilatusV33('XF:11BMB-ES{Det:SAXS}:', name='pilatus300')
    pilatus300.tiff.read_attrs = []
    pilatus300.stats3.total.kind = "hinted"
    pilatus300.stats4.total.kind = "hinted"
    STATS_NAMES = ["stats1", "stats2", "stats3", "stats4", "stats5"]
    pilatus300.read_attrs = ["tiff"] + STATS_NAMES
    for stats_name in STATS_NAMES:
        stats_plugin = getattr(pilatus300, stats_name)
        stats_plugin.read_attrs = ["total"]

    pilatus300.cam.ensure_nonblocking()
else:
    pilatus300 = "Pil300ISNOTWORKING"

# pilatus800 section
# if False:
# if True:
if Pilatus800_on == True:
    pilatus800 = Pilatus800V33("XF:11BMB-ES{Det:PIL800K}:", name="pilatus800")
    pilatus800.tiff.read_attrs = []
    pilatus800.stats3.total.kind = "hinted"
    pilatus800.stats4.total.kind = "hinted"
    STATS_NAMES = ["stats1", "stats2", "stats3", "stats4", "stats5"]
    pilatus800.read_attrs = ["tiff"] + STATS_NAMES
    for stats_name in STATS_NAMES:
        stats_plugin = getattr(pilatus800, stats_name)
        stats_plugin.read_attrs = ["total"]

    for item in pilatus800.stats1.configuration_attrs:
        item_check = getattr(pilatus800.stats1, item)
        item_check.kind = "omitted"

    for item in pilatus800.stats2.configuration_attrs:
        item_check = getattr(pilatus800.stats2, item)
        item_check.kind = "omitted"

    for item in pilatus800.stats3.configuration_attrs:
        item_check = getattr(pilatus800.stats3, item)
        item_check.kind = "omitted"

    for item in pilatus800.stats4.configuration_attrs:
        item_check = getattr(pilatus800.stats4, item)
        item_check.kind = "omitted"

    for item in pilatus800.stats5.configuration_attrs:
        item_check = getattr(pilatus800.stats5, item)
        item_check.kind = "omitted"

    for item in pilatus800.tiff.configuration_attrs:
        item_check = getattr(pilatus800.tiff, item)
        item_check.kind = "omitted"

    for item in pilatus800.cam.configuration_attrs:
        item_check = getattr(pilatus800.cam, item)
        item_check.kind = "omitted"
else:
    pilatus800 = "Pil800ISNOTWORKING"

# pilatus800_2 section  changed by RL, 20210831
# if False:
# if True:
if Pilatus800_2_on == True:
    # TODO:
    pilatus8002 = Pilatus8002V33("XF:11BMB-ES{Det:PIL800K2}:", name="pilatus8002")  # change PV
    pilatus8002.tiff.read_attrs = []
    pilatus8002.stats3.total.kind = "hinted"
    pilatus8002.stats4.total.kind = "hinted"
    STATS_NAMES = ["stats1", "stats2", "stats3", "stats4", "stats5"]
    pilatus8002.read_attrs = ["tiff"] + STATS_NAMES
    for stats_name in STATS_NAMES:
        stats_plugin = getattr(pilatus8002, stats_name)
        stats_plugin.read_attrs = ["total"]

    for item in pilatus8002.stats1.configuration_attrs:
        item_check = getattr(pilatus8002.stats1, item)
        item_check.kind = "omitted"

    for item in pilatus8002.stats2.configuration_attrs:
        item_check = getattr(pilatus8002.stats2, item)
        item_check.kind = "omitted"

    for item in pilatus8002.stats3.configuration_attrs:
        item_check = getattr(pilatus8002.stats3, item)
        item_check.kind = "omitted"

    for item in pilatus8002.stats4.configuration_attrs:
        item_check = getattr(pilatus8002.stats4, item)
        item_check.kind = "omitted"

    for item in pilatus8002.stats5.configuration_attrs:
        item_check = getattr(pilatus8002.stats5, item)
        item_check.kind = "omitted"

    for item in pilatus8002.tiff.configuration_attrs:
        item_check = getattr(pilatus8002.tiff, item)
        item_check.kind = "omitted"

    for item in pilatus8002.cam.configuration_attrs:
        item_check = getattr(pilatus8002.cam, item)
        item_check.kind = "omitted"
else:
    pilatus8002 = "Pil800-2ISNOTWORKING"

# pilatus2M section
# if False:
if Pilatus2M_on == True:
    pilatus2M = Pilatus2MV33("XF:11BMB-ES{Det:PIL2M}:", name="pilatus2M")
    pilatus2M.tiff.read_attrs = []

    STATS_NAMES2M = ["stats1", "stats2", "stats3", "stats4"]
    pilatus2M.read_attrs = ["tiff"] + STATS_NAMES2M
    for stats_name in STATS_NAMES2M:
        stats_plugin = getattr(pilatus2M, stats_name)
        stats_plugin.read_attrs = ["total"]
    pilatus2M.cam.ensure_nonblocking()
    pilatus2M.tiff.ensure_blocking()
    pilatus2M.stats3.total.kind = "hinted"
    pilatus2M.stats4.total.kind = "hinted"

    for item in pilatus2M.stats1.configuration_attrs:
        item_check = getattr(pilatus2M.stats1, item)
        item_check.kind = "omitted"

    for item in pilatus2M.stats2.configuration_attrs:
        item_check = getattr(pilatus2M.stats2, item)
        item_check.kind = "omitted"

    for item in pilatus2M.stats3.configuration_attrs:
        item_check = getattr(pilatus2M.stats3, item)
        item_check.kind = "omitted"

    for item in pilatus2M.stats4.configuration_attrs:
        item_check = getattr(pilatus2M.stats4, item)
        item_check.kind = "omitted"

    for item in pilatus2M.stats5.configuration_attrs:
        item_check = getattr(pilatus2M.stats5, item)
        item_check.kind = "omitted"

    for item in pilatus2M.tiff.configuration_attrs:
        item_check = getattr(pilatus2M.tiff, item)
        item_check.kind = "omitted"

    for item in pilatus2M.cam.configuration_attrs:
        item_check = getattr(pilatus2M.cam, item)
        item_check.kind = "omitted"

elif Pilatus2M_on == 'h5': 

    pilatus2M = Pilatus2MV33_h5("XF:11BMB-ES{Det:PIL2M}:", name="pilatus2M")
    pilatus2M.h5.read_attrs = []

    STATS_NAMES2M = ["stats1", "stats2", "stats3", "stats4"]
    pilatus2M.read_attrs = ["h5"] + STATS_NAMES2M
    for stats_name in STATS_NAMES2M:
        stats_plugin = getattr(pilatus2M, stats_name)
        stats_plugin.read_attrs = ["total"]
    pilatus2M.cam.ensure_nonblocking()
    pilatus2M.h5.ensure_blocking()
    pilatus2M.stats3.total.kind = "hinted"
    pilatus2M.stats4.total.kind = "hinted"

    for item in pilatus2M.stats1.configuration_attrs:
        item_check = getattr(pilatus2M.stats1, item)
        item_check.kind = "omitted"

    for item in pilatus2M.stats2.configuration_attrs:
        item_check = getattr(pilatus2M.stats2, item)
        item_check.kind = "omitted"

    for item in pilatus2M.stats3.configuration_attrs:
        item_check = getattr(pilatus2M.stats3, item)
        item_check.kind = "omitted"

    for item in pilatus2M.stats4.configuration_attrs:
        item_check = getattr(pilatus2M.stats4, item)
        item_check.kind = "omitted"

    for item in pilatus2M.stats5.configuration_attrs:
        item_check = getattr(pilatus2M.stats5, item)
        item_check.kind = "omitted"

    # for item in pilatus2M.tiff.configuration_attrs:
    #     item_check = getattr(pilatus2M.tiff, item)
    #     item_check.kind = "omitted"

    for item in pilatus2M.cam.configuration_attrs:
        item_check = getattr(pilatus2M.cam, item)
        item_check.kind = "omitted"
else:
    pilatus2M = "Pil2MISNOTWORKING"


# define the current pilatus detector: pilatus_name and _Epicsname, instead of
# pilatus300 or pilatus2M
# pilatus_name = pilatus800
# pilatus_Epicsname = '{Det:PIL800K}'
pilatus_name = pilatus2M
pilatus_Epicsname = "{Det:PIL2M}"

# pilatus_name = pilatus800
# pilatus_Epicsname = '{Det:PIL800K}'


#######################################################
# These are test functions added by Julien
# We should remove them once we find the source of the
# current "None"type bug at CMS (TRAC ticket [2284]
def get_stage_sigs(dev, dd):
    for cpt_name in dev.component_names:
        cpt = getattr(dev, cpt_name)
        if hasattr(cpt, "stage_sigs"):
            dd.update(cpt.stage_sigs)
        if hasattr(cpt, "component_names"):
            get_stage_sigs(cpt, dd)


def stage_unstage_forever_plan(det):
    i = 0
    print("Started the stage_unstage_plan, running infinite loop...")
    while True:
        i += 1
        # print(f"Staging {i}th time")
        yield from bps.stage(det)
        yield from bps.unstage(det)


def trigger_forever_plan(det):
    i = 0
    print("Started the stage_unstage_plan, running infinite loop...")
    while True:
        i += 1
        # print(f"Staging {i}th time")
        yield from bps.stage(det)
        yield from bps.trigger(det, group="det")
        yield from bps.wait("det")
        yield from bps.unstage(det)


def count_forever_plan(det):
    i = 0
    print("Started the count_forever plan, running infinite loop...")
    while True:
        i += 1
        # print(f"Staging {i}th time")
        yield from bp.count([det])


def stage_unstage_once_plan(det):
    # print(f"Staging {i}th time")
    yield from bps.stage(det)
    yield from bps.unstage(det)


def count_no_save_plan(det):
    # print(f"Staging {i}th time")
    yield from bps.stage(det)
    yield from bps.trigger(det)
    yield from bps.unstage(det)


# to get stage sigs
# from collections import OrderedDict
# stage_sigs = OrderedDict()
# get_stage_sigs(pilatus2M, stage_sigs)


#######################################################

# pilatus_name = pilatus300
# pilatus_Epicsname = '{Det:SAXS}'



print(f'Loading {__file__}DONEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
