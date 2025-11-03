from IPython import get_ipython
caput(beam.mono_bragg_pv, 1.03953)
pilatus2M.tiff.create_directory.set(-20)

ipython = get_ipython()
ipython.run_line_magic('run', '-i ./startup/user_collection/user_LinkamThermal.py')

class Sample(SampleLinkamTensile, SampleCDSAXS_Generic):
    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)

sam = Sample("test")

detselect(pilatus2M)

RE.md.update({'scan_id': 1})

pilatus2M.cam.num_images.put(1)
