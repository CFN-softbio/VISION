cms.setMetadata()
%run -i user_TSAXSWAXS.py

sam = SampleGISAXS('test')
detselect(pilatus2M)
sam.measureIncidentAngle(0.1, exposure_time=1)

sam.xr(1)
sam.thr(0.1)
sam.thabs(0.2)

cms.setMetadata()
sam.measure(1)
cms.modeMeasurement()
wbs()
sam.measure(1)
