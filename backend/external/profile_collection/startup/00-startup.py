print(f'Loading {__file__}')

# import logging
# import caproto
# handler = logging.FileHandler('pilatus-trigger-log.txt')
# from caproto._log import LogFormatter, color_log_format, log_date_format
# handler.setFormatter(
#     LogFormatter(color_log_format, datefmt=log_date_format))
# caproto_log = logging.getLogger('caproto')
# caproto_log.handlers.clear()
# caproto_log.addHandler(handler)
# logging.getLogger('caproto.ch').setLevel('DEBUG')
import nslsii
import redis

from redis_json_dict import RedisJSONDict

nslsii.configure_base(get_ipython().user_ns, "cms", publish_documents_with_kafka=True)

from bluesky.magics import BlueskyMagics
from bluesky.preprocessors import pchain

# At the end of every run, verify that files were saved and
# print a confirmation message.
from bluesky.callbacks.broker import verify_files_saved

# RE.subscribe(post_run(verify_files_saved), 'stop')

from pyOlog.ophyd_tools import *

# Uncomment the following lines to turn on verbose messages for
# debugging.
# import logging
# ophyd.logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

# Add a callback that prints scan IDs at the start of each scan.
# def print_scan_ids(name, start_doc):
#     print("Transient Scan ID: {0} @ {1}".format(start_doc['scan_id'],time.strftime("%Y/%m/%d %H:%M:%S")))
#     print("Persistent Unique Scan ID: '{0}'".format(start_doc['uid']))
#
# RE.subscribe(print_scan_ids, 'start')

# - HACK #1 -  patch EpicsSignal.get to retry when timeouts happen stolen from HXN
import ophyd


def _epicssignal_get(self, *, as_string=None, connection_timeout=1.0, **kwargs):
    """Get the readback value through an explicit call to EPICS
    Parameters
    ----------
    count : int, optional
        Explicitly limit count for array data
    as_string : bool, optional
        Get a string representation of the value, defaults to as_string
        from this signal, optional
    as_numpy : bool
        Use numpy array as the return type for array data.
    timeout : float, optional
        maximum time to wait for value to be received.
        (default = 0.5 + log10(count) seconds)
    use_monitor : bool, optional
        to use value from latest monitor callback or to make an
        explicit CA call for the value. (default: True)
    connection_timeout : float, optional
        If not already connected, allow up to `connection_timeout` seconds
        for the connection to complete.
    """
    if as_string is None:
        as_string = self._string

    with self._metadata_lock:
        if not self._read_pv.connected:
            if not self._read_pv.wait_for_connection(connection_timeout):
                raise TimeoutError("Failed to connect to %s" % self._read_pv.pvname)

        ret = None
        attempts = 0
        max_attempts = 4
        while ret is None and attempts < max_attempts:
            attempts += 1
            ret = self._read_pv.get(as_string=as_string, **kwargs)
            if ret is None:
                print(f"*** PV GET TIMED OUT {self._read_pv.pvname} *** attempt #{attempts}/{max_attempts}")
        if ret is None:
            print(f"*** PV GET TIMED OUT {self._read_pv.pvname} *** return `None` as value :(")
            # TODO we really want to raise TimeoutError here, but that may cause more
            # issues in the codebase than we have the time to fix...
            # If this causes issues, remove it to keep the old functionality...
            raise TimeoutError("Failed to get %s after %d attempts" % (self._read_pv.pvname, attempts))
        if attempts > 1:
            print(f"*** PV GET succeeded {self._read_pv.pvname} on attempt #{attempts}")

    if as_string:
        return ophyd.signal.waveform_to_string(ret)

    return ret


from ophyd import EpicsSignal
from ophyd import EpicsSignalRO

# from ophyd import EpicsSignalBase

from ophyd.areadetector import EpicsSignalWithRBV

# Increase the timeout for EpicsSignal.get()
# This beamline was occasionally getting ReadTimeoutErrors
EpicsSignal.set_defaults(timeout=10, connection_timeout=10)
EpicsSignalRO.set_defaults(timeout=10, connection_timeout=10)
ophyd.signal.EpicsSignalBase.set_defaults(timeout=120, connection_timeout=120)


# We have commented this because we would like to identify the PVs that are causing problems.
# Then the controls group can investigate why it is not working as expected.
# Increasing the get() timeout argument is the prefered way to work around this problem.
# EpicsSignal.get = _epicssignal_get
# EpicsSignalRO.get = _epicssignal_get
# EpicsSignalWithRBV.get = _epicssignal_get

from pathlib import Path

import appdirs


try:
    from bluesky.utils import PersistentDict
except ImportError:
    import msgpack
    import msgpack_numpy
    import zict

    class PersistentDict(zict.Func):
        """
        A MutableMapping which syncs it contents to disk.
        The contents are stored as msgpack-serialized files, with one file per item
        in the mapping.
        Note that when an item is *mutated* it is not immediately synced:
        >>> d['sample'] = {"color": "red"}  # immediately synced
        >>> d['sample']['shape'] = 'bar'  # not immediately synced
        but that the full contents are synced to disk when the PersistentDict
        instance is garbage collected.
        """

        def __init__(self, directory):
            self._directory = directory
            self._file = zict.File(directory)
            self._cache = {}
            super().__init__(self._dump, self._load, self._file)
            self.reload()

            # Similar to flush() or _do_update(), but without reference to self
            # to avoid circular reference preventing collection.
            # NOTE: This still doesn't guarantee call on delete or gc.collect()!
            #       Explicitly call flush() if immediate write to disk required.
            def finalize(zfile, cache, dump):
                zfile.update((k, dump(v)) for k, v in cache.items())

            import weakref

            self._finalizer = weakref.finalize(self, finalize, self._file, self._cache, PersistentDict._dump)

        @property
        def directory(self):
            return self._directory

        def __setitem__(self, key, value):
            self._cache[key] = value
            super().__setitem__(key, value)

        def __getitem__(self, key):
            return self._cache[key]

        def __delitem__(self, key):
            del self._cache[key]
            super().__delitem__(key)

        def __repr__(self):
            return f"<{self.__class__.__name__} {dict(self)!r}>"

        @staticmethod
        def _dump(obj):
            "Encode as msgpack using numpy-aware encoder."
            # See https://github.com/msgpack/msgpack-python#string-and-binary-type
            # for more on use_bin_type.
            return msgpack.packb(obj, default=msgpack_numpy.encode, use_bin_type=True)

        @staticmethod
        def _load(file):
            return msgpack.unpackb(file, object_hook=msgpack_numpy.decode, raw=False)

        def flush(self):
            """Force a write of the current state to disk"""
            for k, v in self.items():
                super().__setitem__(k, v)

        def reload(self):
            """Force a reload from disk, overwriting current cache"""
            self._cache = dict(super().items())


# runengine_metadata_dir = "/nsls2/data/cms/shared/config/runengine-metadata"
# RE.md = PersistentDict(runengine_metadata_dir)

# proposed changes by Juan
# runengine_metadata_dir = appdirs.user_data_dir(appname="bluesky") / Path("runengine-metadata")
# # PersistentDict will create the directory if it does not exist
# metadata = PersistentDict(runengine_metadata_dir)
RE.md = RedisJSONDict(redis.Redis("info.cms.nsls2.bnl.gov"), prefix="")
print("Redis JSON:", RE.md)
print("*" * 100)
# RE.md.update(metadata)

# print("a new version of bsui")
# print("sth is happening")

#this replaces RE() <
from bluesky.utils import register_transform
register_transform('RE', prefix='<')

#reload start-up files, modified by Siyu 2024/10/22


# import subprocess

# def run_ipython_script(script_path):
#     subprocess.run([f"{script_path}"])

# def load_py():
#     import glob
#     pyf = sorted( glob.glob ("*.py")  )
#     for py in pyf:
#         %run -i {py}
#         print ( py )  

#     config_load() 

# load_py()