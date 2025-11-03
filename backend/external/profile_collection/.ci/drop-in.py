import numpy as np

caput(beam.mono_bragg_pv, 1.03953)
pilatus2M.tiff.create_directory.set(-20)
sam = SampleGISAXS('test')
detselect(pilatus2M)

RE.md.update({'scan_id': 1})

pilatus2M.cam.num_images.put(1)

uid, = RE(bp.count([pilatus2M], num=3))

hdr = db[uid]
print(hdr.table(fill=True))

data = np.array(list(hdr.data("pilatus2M_image")))
print(f"{data = }\n{data.shape = }")
