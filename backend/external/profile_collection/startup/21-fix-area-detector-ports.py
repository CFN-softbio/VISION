print(f'Loading {__file__}')

# # for cam_number, fs in zip([1,2,3,4], [fs1, fs2, fs3, fs4]):
# for cam_number, fs in zip([2, 3, 4], [fs2, fs3, fs4]):
#     G, port_dict = fs.get_asyn_digraph()
#     cam = port_dict["cam{:02}".format(cam_number)]
#     for v in port_dict.values():
#         try:
#             if v.nd_array_port.get() == "CAM":
#                 v.nd_array_port.set("cam{:02}".format(cam_number))
#         except AttributeError:
#             pass
#     fs.validate_asyn_ports()
