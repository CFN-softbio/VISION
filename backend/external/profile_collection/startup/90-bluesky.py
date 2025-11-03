print(f'Loading {__file__}')


def detselect(detector_object, suffix="_stats4_total"):
    """Switch the active detector and set some internal state"""

    if isinstance(detector_object, (list, tuple)):
        # gs.DETS = detector_object
        # gs.PLOT_Y = detector_object[0].name + suffix
        # gs.TABLE_COLS = [gs.PLOT_Y]
        cms.detector = detector_object
        cms.PLOT_Y = detector_object[0].name + suffix
        cms.TABLE_COLS = [cms.PLOT_Y]

    else:
        cms.detector = [detector_object]
        cms.PLOT_Y = detector_object.name + suffix
        cms.TABLE_COLS = [cms.PLOT_Y]
    return cms.detector
    # only return the detector name than the long list with
    # all attributes since bluesky upgrade in cycle 201802
    detector_list = []
    for detector in cms.detector:
        detector_list.append(detector.name)
    return detector_list
    # return cms.detector


##### I/O devices
from epics import caget, caput


def pneumatic(inout, pv_r, pv_in, pv_out, ss1, quiet):
    if inout == 1:
        caput(pv_in, 1)
        ss2 = "has been inserted"
    elif inout == 0:
        caput(pv_out, 1)
        ss2 = "has been retracted"
    else:
        if caget(pv_r) == 1:
            ss2 = "is IN"
        else:
            ss2 = "is OUT"
    if quiet == 0:
        print(ss1 + " " + ss2)


## Fluorescence screen 1 (FOE)
def io_fs1(inout, q=0):
    pv_r = "XF:11BMA-BI{FS:1}Pos-Sts"
    pv_in = "XF:11BMA-BI{FS:1}Cmd:In-Cmd"
    pv_out = "XF:11BMA-BI{FS:1}Cmd:Out-Cmd"
    ss1 = "Fluorescence screen 1 (FOE)"
    pneumatic(inout, pv_r, pv_in, pv_out, ss1, q)


## Fluorescence screen 3 (Endstation)
def io_fs3(inout, q=0):
    pv_r = "XF:11BMB-BI{FS:3}Pos-Sts"
    pv_in = "XF:11BMB-BI{FS:3}Cmd:In-Cmd"
    pv_out = "XF:11BMB-BI{FS:3}Cmd:Out-Cmd"
    ss1 = "Fluorescence screen 3 (Endstation)"
    pneumatic(inout, pv_r, pv_in, pv_out, ss1, q)


## Fluorescence screen 4 (Endstation)
def io_fs4(inout, q=0):
    pv_r = "XF:11BMB-BI{FS:4}Pos-Sts"
    pv_in = "XF:11BMB-BI{FS:4}Cmd:In-Cmd"
    pv_out = "XF:11BMB-BI{FS:4}Cmd:Out-Cmd"
    ss1 = "Fluorescence screen 3 (Endstation)"
    pneumatic(inout, pv_r, pv_in, pv_out, ss1, q)


## BIM 5 - RIGI (Endstation)
def io_bim5(inout, q=0):
    pv_r = "XF:11BMB-BI{IM:5}Pos-Sts"
    pv_in = "XF:11BMB-BI{IM:5}Cmd:In-Cmd"
    pv_out = "XF:11BMB-BI{IM:5}Cmd:Out-Cmd"
    ss1 = "BIM 5 - RIGI (Endstation)"
    pneumatic(inout, pv_r, pv_in, pv_out, ss1, q)


## Attenuation filter box
def io_atten(pos, inout, q=0):
    if pos >= 1 and pos <= 8:
        pv_r = "XF:11BMB-OP{Fltr:" + str(int(pos)) + "}Pos-Sts"
        pv_in = "XF:11BMB-OP{Fltr:" + str(int(pos)) + "}Cmd:In-Cmd"
        pv_out = "XF:11BMB-OP{Fltr:" + str(int(pos)) + "}Cmd:Out-Cmd"
        ss1 = "Atten filter " + str(int(pos))
        pneumatic(inout, pv_r, pv_in, pv_out, ss1, q)
    else:
        print("Attenuator position must be an integer between 1 and 8")


from math import exp, log


def get_atten_trans():
    E = getE(q=1)  # Current E [keV]

    if E < 6.0 or E > 18.0:
        print("Transmission data not available at the current X-ray enegy.")

    else:
        N = []
        for i in np.arange(8):
            N.append(caget("XF:11BMB-OP{Fltr:" + str(int(i) + 1) + "}Pos-Sts"))

        N_Al = N[0] + 2 * N[1] + 4 * N[2] + 8 * N[3]
        N_Nb = N[4] + 2 * N[5] + 4 * N[6] + 8 * N[7]

        d_Nb = 0.1  # Thickness [mm] of one Nb foil
        d_Al = 0.25  # Thickness [mm] of one Al foil

        # Absorption length [mm] based on fits to LBL CXRO data for 6 < E < 19 keV
        l_Nb = 1.4476e-3 - 5.6011e-4 * E + 1.0401e-4 * E * E + 8.7961e-6 * E * E * E
        l_Al = 5.2293e-3 - 1.3491e-3 * E + 1.7833e-4 * E * E + 1.4001e-4 * E * E * E

        # transmission factors
        tr_Nb = exp(-N_Nb * d_Nb / l_Nb)
        tr_Al = exp(-N_Al * d_Al / l_Al)
        tr_tot = tr_Nb * tr_Al

        print("%dx 0.25mm Al (%.1e) and %dx 0.10mm Nb (%.1e)" % (N_Al, tr_Al, N_Nb, tr_Nb))
        print("Combined transmission is %.1e" % tr_tot)

        return tr_tot


def set_atten_trans(tr):
    E = getE(q=1)  # Current E [keV]

    if E < 6.0 or E > 18.0:
        print("Transmission data not available at the current X-ray enegy.")

    elif tr > 1.0 or tr < 1.0e-10:
        print("Requested attenuator transmission is not valid.")

    else:
        d_Nb = 0.1  # Thickness [mm] of one Nb foil
        d_Al = 0.25  # Thickness [mm] of one Al foil

        # Absorption length [mm] based on fits to LBL CXRO data for 6 < E < 19 keV
        l_Nb = 1.4476e-3 - 5.6011e-4 * E + 1.0401e-4 * E * E + 8.7961e-6 * E * E * E
        l_Al = 5.2293e-3 - 1.3491e-3 * E + 1.7833e-4 * E * E + 1.4001e-4 * E * E * E

        d_l_Nb = d_Nb / l_Nb
        d_l_Al = d_Al / l_Al

        # Number of foils to be inserted (picks a set that gives smallest deviation from requested transmission)
        dev = []
        for i in np.arange(16):
            for j in np.arange(16):
                dev_ij = abs(tr - exp(-i * d_l_Nb) * exp(-j * d_l_Al))
                dev.append(dev_ij)
                if dev_ij == min(dev):
                    N_Nb = i  # number of Nb foils selected
                    N_Al = j  # number of Al foils selected

        N = []
        state = N_Al
        for i in np.arange(4):
            N.append(state % 2)
            state = int(state / 2)

        state = N_Nb
        for i in np.arange(4):
            N.append(state % 2)
            state = int(state / 2)

        for i in np.arange(8):
            io_atten(i + 1, N[i], q=1)

        time.sleep(1.0)
        return get_atten_trans()


### 1-stage valves
def single_valve(cmd, pv_r, pv_op, pv_cl, ss1, quiet):
    if cmd == "st":
        st = caget(pv_r)
        if st == 1:
            ss2 = "valve is open"
        if st == 0:
            ss2 = "valve is closed"
    if cmd == "o" or cmd == "open":
        caput(pv_op, 1)
        ss2 = "valve has been opened"
    if cmd == "c" or cmd == "close":
        caput(pv_cl, 1)
        ss2 = "valve has been closed"
    if quiet == 0:
        print(ss1 + " " + ss2)


### 2-stage valves
def dual_valve(cmd, pv_r_soft, pv_op_soft, pv_cl_soft, pv_r, pv_op, pv_cl, ss1, quiet):
    if cmd == "st":
        st_h = caget(pv_r)
        st_s = caget(pv_r_soft)
        if st_h == 1 and st_s == 1:
            ss2 = "main open and soft open"
        if st_h == 0 and st_s == 1:
            ss2 = "main closed and soft open"
        if st_h == 1 and st_s == 0:
            ss2 = "main open and soft closed"
        if st_h == 0 and st_s == 0:
            ss2 = "main closed and soft closed"
    if cmd == "o" or cmd == "open":
        caput(pv_cl_soft, 1)
        time.sleep(0.2)
        caput(pv_op, 1)
        ss2 = "main has been opened (soft closed)"
    if cmd == "so" or cmd == "soft":
        caput(pv_cl, 1)
        time.sleep(0.5)
        caput(pv_op_soft, 1)
        ss2 = "soft has been opened (main closed)"
    if cmd == "c" or cmd == "close":
        caput(pv_cl, 1)
        time.sleep(0.2)
        caput(pv_cl_soft, 1)
        ss2 = "valve has been closed"
    if quiet == 0:
        print(ss1 + " " + ss2)


## Isolation valve - incident path
def iv_inc(cmd="st", q=0):
    """cmd: 'st' for status, 'o' or 'open' for hard open, 'so' or 'soft' for soft open, 'c' or 'close' for closing"""
    pv_r_soft = "XF:11BMB-VA{Mir:KB-IV:1_Soft}Pos-Sts"
    pv_op_soft = "XF:11BMB-VA{Mir:KB-IV:1_Soft}Cmd:Opn-Cmd"
    pv_cl_soft = "XF:11BMB-VA{Mir:KB-IV:1_Soft}Cmd:Cls-Cmd"
    pv_r = "XF:11BMB-VA{Mir:KB-IV:1}Pos-Sts"
    pv_op = "XF:11BMB-VA{Mir:KB-IV:1}Cmd:Opn-Cmd"
    pv_cl = "XF:11BMB-VA{Mir:KB-IV:1}Cmd:Cls-Cmd"
    ss1 = "Isolation valve for incident path: "
    dual_valve(cmd, pv_r_soft, pv_op_soft, pv_cl_soft, pv_r, pv_op, pv_cl, ss1, q)


## Isolation valve - sample/detector chamber
def iv_chm(cmd="st", q=0):
    """cmd: 'st' for status, 'o' or 'open' for hard open, 'so' or 'soft' for soft open, 'c' or 'close' for closing"""
    pv_r_soft = "XF:11BMB-VA{Chm:Det-IV:1_Soft}Pos-Sts"
    pv_op_soft = "XF:11BMB-VA{Chm:Det-IV:1_Soft}Cmd:Opn-Cmd"
    pv_cl_soft = "XF:11BMB-VA{Chm:Det-IV:1_Soft}Cmd:Cls-Cmd"
    pv_r = "XF:11BMB-VA{Chm:Det-IV:1}Pos-Sts"
    pv_op = "XF:11BMB-VA{Chm:Det-IV:1}Cmd:Opn-Cmd"
    pv_cl = "XF:11BMB-VA{Chm:Det-IV:1}Cmd:Cls-Cmd"
    ss1 = "Isolation valve for sample/WAXS chamber: "
    dual_valve(cmd, pv_r_soft, pv_op_soft, pv_cl_soft, pv_r, pv_op, pv_cl, ss1, q)


## Isolation valve - SAXS flightpath
def iv_pipe(cmd="st", q=0):
    """cmd: 'st' for status, 'o' or 'open' for hard open, 'so' or 'soft' for soft open, 'c' or 'close' for closing"""
    pv_r_soft = "XF:11BMB-VA{BT:SAXS-IV:1_Soft}Pos-Sts"
    pv_op_soft = "XF:11BMB-VA{BT:SAXS-IV:1_Soft}Cmd:Opn-Cmd"
    pv_cl_soft = "XF:11BMB-VA{BT:SAXS-IV:1_Soft}Cmd:Cls-Cmd"
    pv_r = "XF:11BMB-VA{BT:SAXS-IV:1}Pos-Sts"
    pv_op = "XF:11BMB-VA{BT:SAXS-IV:1}Cmd:Opn-Cmd"
    pv_cl = "XF:11BMB-VA{BT:SAXS-IV:1}Cmd:Cls-Cmd"
    ss1 = "Isolation valve for SAXS pipe: "
    dual_valve(cmd, pv_r_soft, pv_op_soft, pv_cl_soft, pv_r, pv_op, pv_cl, ss1, q)


## Vent valve - chamber upstream
def vv_us(cmd="st", q=0):
    """cmd: 'st' for status, 'o' or 'open' for hard open, 'so' or 'soft' for soft open, 'c' or 'close' for closing"""
    pv_r_soft = "XF:11BMB-VA{Chm:Smpl-VV:1_Soft}Pos-Sts"
    pv_op_soft = "XF:11BMB-VA{Chm:Smpl-VV:1_Soft}Cmd:Opn-Cmd"
    pv_cl_soft = "XF:11BMB-VA{Chm:Smpl-VV:1_Soft}Cmd:Cls-Cmd"
    pv_r = "XF:11BMB-VA{Chm:Smpl-VV:1}Pos-Sts"
    pv_op = "XF:11BMB-VA{Chm:Smpl-VV:1}Cmd:Opn-Cmd"
    pv_cl = "XF:11BMB-VA{Chm:Smpl-VV:1}Cmd:Cls-Cmd"
    ss1 = "Upstream vent valve for sample/WAXS chamber: "
    dual_valve(cmd, pv_r_soft, pv_op_soft, pv_cl_soft, pv_r, pv_op, pv_cl, ss1, q)


## Vent valve - chamber downstream
def vv_ds(cmd="st", q=0):
    """cmd: 'st' for status, 'o' or 'open' for hard open, 'so' or 'soft' for soft open, 'c' or 'close' for closing"""
    pv_r_soft = "XF:11BMB-VA{Chm:Det-VV:1_Soft}Pos-Sts"
    pv_op_soft = "XF:11BMB-VA{Chm:Det-VV:1_Soft}Cmd:Opn-Cmd"
    pv_cl_soft = "XF:11BMB-VA{Chm:Det-VV:1_Soft}Cmd:Cls-Cmd"
    pv_r = "XF:11BMB-VA{Chm:Det-VV:1}Pos-Sts"
    pv_op = "XF:11BMB-VA{Chm:Det-VV:1}Cmd:Opn-Cmd"
    pv_cl = "XF:11BMB-VA{Chm:Det-VV:1}Cmd:Cls-Cmd"
    ss1 = "Downstream vent valve for sample/WAXS chamber: "
    dual_valve(cmd, pv_r_soft, pv_op_soft, pv_cl_soft, pv_r, pv_op, pv_cl, ss1, q)


## Gate valve (Endstation) - upstream/small
def gv_us(cmd="st", q=0):
    """cmd: 'st' for status, 'o' or 'open' for open, 'c' or 'close' for closing"""
    pv_r = "XF:11BMB-VA{Slt:4-GV:1}Pos-Sts"
    pv_op = "XF:11BMB-VA{Slt:4-GV:1}Cmd:Opn-Cmd"
    pv_cl = "XF:11BMB-VA{Slt:4-GV:1}Cmd:Cls-Cmd"
    ss1 = "Upstream/small gate valve (Endstation): "
    single_valve(cmd, pv_r, pv_op, pv_cl, ss1, q)


## Gate valve (Endstation) - downstream/large
def gv_ds(cmd="st", q=0):
    """cmd: 'st' for status, 'o' or 'open' for open, 'c' or 'close' for closing"""
    pv_r = "XF:11BMB-VA{Chm:Det-GV:1}Pos-Sts"
    pv_op = "XF:11BMB-VA{Chm:Det-GV:1}Cmd:Opn-Cmd"
    pv_cl = "XF:11BMB-VA{Chm:Det-GV:1}Cmd:Cls-Cmd"
    ss1 = "Downstream/large gate valve (Endstation): "
    single_valve(cmd, pv_r, pv_op, pv_cl, ss1, q)


##### Endstation pumps
## Pump for flightpaths
def pump_fp(onoff, q=0):
    pv_r = "XF:11BMB-VA{BT:SAXS-Pmp:1}Sts:Enbl-Sts"
    pv_w = "XF:11BMB-VA{BT:SAXS-Pmp:1}Cmd:Enbl-Cmd"
    if onoff == 1:
        caput(pv_w, 0)
        time.sleep(0.2)
        caput(pv_w, 1)
        ss = "Flightpath pump has been turned ON"
    elif onoff == 0:
        caput(pv_w, 0)
        ss = "Flightpath pump has been turned OFF"
    else:
        if caget(pv_r) == 1:
            ss = "Flightpath pump is ON"
        else:
            ss = "Flightpath pump is OFF"
    if q == 0:
        print(ss)


## Pump for sample/WAXS chamber
def pump_chm(onoff, q=0):
    pv_r = "XF:11BMB-VA{Chm:Det-Pmp:1}Sts:Enbl-Sts"
    pv_w = "XF:11BMB-VA{Chm:Det-Pmp:1}Cmd:Enbl-Cmd"
    if onoff == 1:
        caput(pv_w, 0)
        time.sleep(0.2)
        caput(pv_w, 1)
        ss = "Chamber pump has been turned ON"
    elif onoff == 0:
        caput(pv_w, 0)
        ss = "Chamber pump has been turned OFF"
    else:
        if caget(pv_r) == 1:
            ss = "Chamber pump is ON"
        else:
            ss = "Chamber pump is OFF"
    if q == 0:
        print(ss)


# PROFILE_ROOT = os.path.dirname(__file__)
# PROFILE_ROOT = '/nsls2/data/cms/legacy/xf11bm/ipython_profiles/profile_collection/startup'
from IPython import get_ipython

PROFILE_ROOT = get_ipython().profile_dir.startup_dir
CMS_CONFIG_FILENAME = os.path.join(PROFILE_ROOT, ".cms_config")

## CMS config file
import pandas as pds


def config_update():
    cms.bsx_pos = bsx.position
    beam.armr_absorber_out = armr.position

    # collect the current positions of motors

    current_config = {
        "bsx_pos": cms.bsx_pos,
        #'armr_absorber_o':beam.armr_absorber_o,
        "_delta_y_hover": robot._delta_y_hover,
        "_delta_y_slot": robot._delta_y_slot,
        "_delta_garage_x": robot._delta_garage_x,
        "_delta_garage_y": robot._delta_garage_y,
        "_position_safe": [robot._position_safe],
        "_position_sample_gripped": [robot._position_sample_gripped],
        "_position_hold": [robot._position_hold],
        "_position_garage": [robot._position_garage],
        "_position_stg_exchange": [robot._position_stg_exchange],
        "_position_stg_safe": [robot._position_stg_safe],
        "time": time.ctime(),
    }

    current_config_DF = pds.DataFrame(data=current_config, index=[1])

    # load the previous config file
    cms_config = pds.read_csv(CMS_CONFIG_FILENAME, index_col=0)
    cms_config_update = pds.concat([cms_config, current_config_DF], ignore_index=True)

    # save to file
    cms_config_update.to_csv(CMS_CONFIG_FILENAME)

    config_load()


def config_load():
    # collect the current positions of motors
    cms_config = pds.read_csv(CMS_CONFIG_FILENAME, index_col=0)
    cms.bsx_pos = cms_config.bsx_pos.values[-1]
    # beam.armr_absorber_o = cms_config.armr_absorber_o.values[-1]

    # robot positions --- with single value
    robot._delta_y_hover = cms_config._delta_y_hover.values[-1]
    robot._delta_y_slot = cms_config._delta_y_slot.values[-1]
    robot._delta_garage_x = cms_config._delta_garage_x.values[-1]
    robot._delta_garage_y = cms_config._delta_garage_y.values[-1]

    # robot positions --- with multiple values in (x, y, r, z, phi)

    # tmp = cms_config._position_safe.values[-1]
    # robot._position_safe = [float(pos) for pos in tmp[1:-1].split(',')]

    robot._position_safe = [float(pos) for pos in cms_config._position_safe.values[-1][1:-1].split(",")]

    # robot._position_safe = cms_config._position_safe.values[-1]
    robot._position_sample_gripped = [
        float(pos) for pos in cms_config._position_sample_gripped.values[-1][1:-1].split(",")
    ]
    robot._position_hold = [float(pos) for pos in cms_config._position_hold.values[-1][1:-1].split(",")]
    robot._position_garage = [float(pos) for pos in cms_config._position_garage.values[-1][1:-1].split(",")]
    robot._position_stg_exchange = [
        float(pos) for pos in cms_config._position_stg_exchange.values[-1][1:-1].split(",")
    ]
    robot._position_stg_safe = [float(pos) for pos in cms_config._position_stg_safe.values[-1][1:-1].split(",")]


## output the scan data and save them in user_folder/data.
def data_output(experiment_cycle=None, experiment_alias_directory=None):
    """
    To output the scan data with the scan_id as name
    Please first create "data" folder under user_folder.
    """

    # headers = db(experiment_cycle='2017_3', experiment_group= 'I. Herman (Columbia U.) group', experiment_alias_directory='/nsls2/xf11bm/data/2017_3/IHerman' )
    if experiment_cycle is not None:
        headers = db(
            experiment_cycle=experiment_cycle,
            experiment_alias_directory=experiment_alias_directory,
        )
    else:
        headers = db(experiment_alias_directory=experiment_alias_directory)

    for header in headers:
        dtable = header.table()
        dtable.to_csv(
            "{}/data/{}.csv".format(
                header.get("start").get("experiment_alias_directory"),
                header.get("start").get("scan_id"),
            )
        )


# def data_output_series(mdkeys, experiment_cycle=None, experiment_alias_directory=None):

#     """
#     To output the scan data with the scan_id as name
#     Please first create "data" folder under user_folder.
#     #updated by RL 01/06/23
#     md is the list of the output metadata.
#     i.e. md = ['clock', 'roi4']
#     """

#     #headers = db(experiment_cycle='2017_3', experiment_group= 'I. Herman (Columbia U.) group', experiment_alias_directory='/nsls2/xf11bm/data/2017_3/IHerman' )
#     if experiment_cycle is not None:
#         headers = db( experiment_cycle=experiment_cycle, experiment_alias_directory=experiment_alias_directory)
#     else:
#         headers = db( experiment_alias_directory=experiment_alias_directory)


#     for header in headers:

#         dtable = header.table()
#         for key in mdkeys:
#             # dtable.to_csv('{}/data/{}.csv'.format(header.get('start').get('experiment_alias_directory') , header.get('start').get('scan_id')))
#             dtable.to_csv('{}/data/{}.csv'.format(header.get('start').get('experiment_alias_directory') , header.get('start').get('scan_id')))


## output the scan data and save them in user_folder/data.
def data_output_seires(id_range):
    """
    To output the scan data with the scan_id as name
    Please first create "data" folder under user_folder.
    id_range = np.arange(55123, 56354)
    """

    for ii in id_range:
        header = db[scan_id]
        dtable = header.table()
        dtable.to_csv(
            "{}/data/{}.csv".format(
                header.get("start").get("experiment_alias_directory"),
                header.get("start").get("scan_id"),
            )
        )


# def XRR_data_output(experiment_ids=None)

# headers = db( experiement_ids )
# for header in headers:
# dtable = header.table()


def metadata_output(output_file, SAF=None, experiment_alias_directory=None):
    """
    To output the scan data with the scan_id as name
    Please first create "data" folder under user_folder.
    SAF: SAF number, like '302914'
    """

    # headers = db(experiment_cycle='2017_3', experiment_group= 'I. Herman (Columbia U.) group', experiment_alias_directory='/nsls2/xf11bm/data/2017_3/IHerman' )
    # if experiment_cycle is not None:
    # headers = db( experiment_cycle=experiment_cycle, experiment_alias_directory=experiment_alias_directory)
    # else:
    # headers = db( experiment_alias_directory=experiment_alias_directory)

    headers = db(experiment_SAF_number=SAF)
    output_data = pds.DataFrame()

    for header in headers:
        if "sample_name" in header.start and "sample_x" in header.start and "sample_clock" in header.start:
            current_data = {
                "a_scan_id": header.start["scan_id"],
                "b_sample_name": header.start["sample_name"],
                "c_clock": header.start["sample_clock"],
                "d_pos_x": header.start["sample_x"],
                "e_pos_th": header.start["sample_th"],
                "f_temperature": header.start["sample_temperature_A"],
            }
            current = pds.DataFrame(data=current_data, index=[1])

        # output_data = output_data.append(current_data, ignore_index=True)
        output_data = pds.concat([output_data, current_data], ignore_index=True)

        # output_data = output_data.iloc[0:0]

    output_data.to_csv(output_file)


# rock/swing a motor contineously while taking images
# added by AWalter and RLi at 20181106
def rock_motor_per_step(detector, motor, step, rock_motor=None, rock_motor_limits=None):
    """
    rock/swing a motor contineously while taking images
    use 'per_step' function in scan plan to rock the motor

    detector: pilatus2M or pilatus300
    motor: this motor is NOT used for measurement. use a motor not related to sample/measurement
    step: this step is NOT useed for measurement. set as 1 for single exposure
    rock_motor: the motor to rock.
    rock_motor_limits: the relative rocking position for rock_motor.


    """

    devices = detector + [motor]
    rewindable = all_safe_rewind(devices)  # if devices can be re-triggered

    current = rock_motor.position

    # define rock to swing rock_motor
    # def rock():
    # yield from mvr(rock_motor, rock_motor_limits)
    # yield from mvr(rock_motor, -rock_motor_limits)
    def rock(current=current):
        yield from mv(rock_motor, current + rock_motor_limits)
        yield from mv(rock_motor, current + -rock_motor_limits)

    def inner_rock_and_read():
        # yield from trigger(detector)
        # status = yield from trigger(detector[0])
        status = detector[0].trigger()
        while not status.done:
            yield from rock()
        yield from mv(rock_motor, current)
        yield from create("primary")

        ret = {}  # collect and return readings to give plan access to them
        for obj in devices:
            reading = yield from read(obj)
            if reading is not None:
                ret.update(reading)
        yield from save()
        return ret

    from bluesky.preprocessors import rewindable_wrapper

    return (yield from rewindable_wrapper(inner_rock_and_read(), rewindable))


# Here is how ot use the rock plan
# our_scan=list_scan([pilatus2M], srot, [1,1], per_step = functools.partial(rock_motor_per_step, rock_motor=strans2, rock_motor_limits=2) )
