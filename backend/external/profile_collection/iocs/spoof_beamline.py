#!/usr/bin/env python3
import asyncio
import os
import re
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
from iocs.linkam import LinkamIOC

from caproto import (ChannelChar, ChannelData, ChannelDouble, ChannelEnum,
                     ChannelInteger, ChannelString, ChannelType)
from caproto.server import ioc_arg_parser, run, pvproperty, PVGroup, SubGroup
from caproto.ioc_examples.fake_motor_record import FakeMotor

PLUGIN_TYPE_PVS = [
    (re.compile('image\\d:'), 'NDPluginStdArrays'),
    (re.compile('Stats\\d:'), 'NDPluginStats'),
    (re.compile('CC\\d:'), 'NDPluginColorConvert'),
    (re.compile('Proc\\d:'), 'NDPluginProcess'),
    (re.compile('Over\\d:'), 'NDPluginOverlay'),
    (re.compile('ROI\\d:'), 'NDPluginROI'),
    (re.compile('Trans\\d:'), 'NDPluginTransform'),
    (re.compile('netCDF\\d:'), 'NDFileNetCDF'),
    (re.compile('TIFF\\d:'), 'NDFileTIFF'),
    (re.compile('JPEG\\d:'), 'NDFileJPEG'),
    (re.compile('Nexus\\d:'), 'NDPluginNexus'),
    (re.compile('HDF\\d:'), 'NDFileHDF5'),
    (re.compile('Magick\\d:'), 'NDFileMagick'),
    (re.compile('TIFF\\d:'), 'NDFileTIFF'),
    (re.compile('HDF\\d:'), 'NDFileHDF5'),
    (re.compile('Current\\d:'), 'NDPluginStats'),
    (re.compile('SumAll'), 'NDPluginStats'),
]


class ReallyDefaultDict(defaultdict):
    def __contains__(self, key):
        # Convert {LINKAM} format to plain format for checking
        key = key.replace("{LINKAM}", "LINKAM")

        if "XF:11BM-ES:LINKAM" in key:
            return True  # Allow Linkam PVs
        if "XF:11BMB-ES{Chm:Smpl-Ax:" in key:
            return False
        if "XF:11BMB-ES{BS-Ax:" in key:
            return False
        if "XF:11BMB-ES{SM:1-Ax:" in key:
            return False
        if "XF:11BMB-ES{Det:PIL2M}" in key:
            return False
        return True

    def __missing__(self, key):
        # Convert {LINKAM} format to plain format for lookup
        lookup_key = key.replace("{LINKAM}", "LINKAM")

        if "XF:11BM-ES:LINKAM" in lookup_key:
            # Forward to the LinkamIOC's PVs
            return self.get(lookup_key, None)
        if "XF:11BMB-ES{Chm:Smpl-Ax:" in key:
            return None
        if "XF:11BMB-ES{BS-Ax:" in key:
            return None
        if "XF:11BMB-ES{Det:PIL2M}" in key:
            return None
        if "XF:11BMB-ES{SM:1-Ax:" in key:
            return None
        if (key.endswith('-SP') or key.endswith('-I') or
                key.endswith('-RB') or key.endswith('-Cmd')):
            key, *_ = key.rpartition('-')
            return self[key]
        if key.endswith('_RBV') or key.endswith(':RBV'):
            return self[key[:-4]]
        ret = self[key] = self.default_factory(key)
        return ret

class Shutter(PVGroup):
    shutter = pvproperty(value=0, name="{{Shutter}}", dtype=ChannelType.INT)
    psh_blade2_pos = pvproperty(value=0, name="{{Psh_blade2}}Pos", dtype=ChannelType.INT)
    psh_blade1_pos = pvproperty(value=0, name="{{Psh_blade1}}Pos", dtype=ChannelType.INT)

    def __init__(self, prefix: str, *args, **kwargs):
        # Initialize the explicit pv properties
        super().__init__(prefix=prefix, *args, **kwargs)

    @shutter.putter # type: ignore
    async def shutter(self, instance, value):
        await asyncio.sleep(0.1)
        await self.psh_blade2_pos.write(value)
        await self.psh_blade1_pos.write(value)
        return value


class CMS_IOC(PVGroup):
    shutter = SubGroup(Shutter, prefix="XF:11BM-ES")
    #motor = SubGroup(FakeMotor, prefix="XF:11BMB-ES{{Chm:Smpl-Ax:X}}Mtr")

    linkam = SubGroup(LinkamIOC, prefix="XF:11BM-ES:LINKAM:")

    def __init__(self, *args, **kwargs):
        super().__init__(prefix="", *args, **kwargs)
        # Initialize the LinkamIOC first
        self.old_pvdb = self.pvdb.copy()
        dummy = ReallyDefaultDict(self.fabricate_channel)
        dummy.update(self.old_pvdb)
        self.pvdb = dummy

    async def startup(self):
        print("[CMS_IOC] Starting up...")
        print("[CMS_IOC] Initializing LinkamIOC...")
        await self.linkam.startup()
        print("[CMS_IOC] Startup complete")

    def fabricate_channel(self, key):
        # If the channel already exists from initialization, return it
        if key in self.old_pvdb:
            return self.old_pvdb[key]
        # Otherwise, fabricate new channels
        if 'PluginType' in key:
            for pattern, val in PLUGIN_TYPE_PVS:
                if pattern.search(key):
                    return ChannelString(value=val)
        elif 'ArrayPort' in key:
            return ChannelString(value=key)
        elif 'PortName' in key:
            return ChannelString(value=key)
        elif 'name' in key.lower():
            return ChannelString(value=key)
        elif 'EnableCallbacks' in key:
            return ChannelEnum(value=0, enum_strings=['Disabled', 'Enabled'])
        elif 'BlockingCallbacks' in key:
            return ChannelEnum(value=0, enum_strings=['No', 'Yes'])
        elif 'Auto' in key:
            return ChannelEnum(value=0, enum_strings=['No', 'Yes'])
        elif 'ImageMode' in key:
            return ChannelEnum(value=0, enum_strings=['Single', 'Multiple', 'Continuous'])
        elif 'WriteMode' in key:
            return ChannelEnum(value=0, enum_strings=['Single', 'Capture', 'Stream'])
        elif 'ArraySize' in key:
            return ChannelData(value=10)
        elif 'TriggerMode' in key:
            return ChannelEnum(value=0, enum_strings=['Internal', 'External'])
        elif 'FileWriteMode' in key:
            return ChannelEnum(value=0, enum_strings=['Single'])
        elif 'FilePathExists' in key:
            return ChannelData(value=1)
        elif 'WaitForPlugins' in key:
            return ChannelEnum(value=0, enum_strings=['No', 'Yes'])
        elif ('file' in key.lower() and 'number' not in key.lower() and
            'mode' not in key.lower()):
            return ChannelChar(value='a' * 250)
        elif ('filenumber' in key.lower()):
            return ChannelInteger(value=0)
        elif 'Compression' in key:
            return ChannelEnum(value=0, enum_strings=['None', 'N-bit', 'szip', 'zlib', 'blosc'])
        return ChannelDouble(value=0.0)


def run_ioc():
    _, run_options = ioc_arg_parser(default_prefix='', desc="PV black hole")
    run_options['interfaces'] = ['127.0.0.1']
    run_options.pop('module_name', None)

    # Instantiate the top-level IOC which includes the LinkamIOC as a SubGroup
    ioc = CMS_IOC()

    # Create an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Start the IOC and properly await startup
        loop.run_until_complete(ioc.startup())

        # Start the device poller (now synchronous)
        ioc.linkam.start_poller()
        print("[run_ioc] IOC startup complete, running main loop...")

        # Run the IOC with the initialized pvdb
        run(ioc.pvdb, **run_options)
    except KeyboardInterrupt:
        print('\nIOC terminated by user')
    except Exception as e:
        print(f'\nError running IOC: {e}')
    finally:
        loop.close()

def main():
    print('''
*** WARNING ***
This script spawns an EPICS IOC which responds to ALL caget, caput, camonitor
requests. As this is effectively a PV black hole, it may affect the
performance and functionality of other IOCs on your network.

The script ignores the --interfaces command line argument, always
binding only to 127.0.0.1, superseding the usual default (0.0.0.0) and any
user-provided value.
*** WARNING ***

Starting IOC...''')

    run_ioc()

if __name__ == '__main__':
    main()
