print(f'Loading {__file__}')

from math import sin, cos, tan, asin, acos, atan, pi

##### mono utilities #####
hc_over_e = 12.39842
dmm_dsp = 20.1  ## DMM bilayer pitch in [Ang], according to Rigaku metrology report


def getE(q=0):
    """Returns E(keV) based on the current mono_bragg position (q=1 for quiet)"""
    bragg = (pi / 180.0) * caget("XF:11BMA-OP{Mono:DMM-Ax:Bragg}Mtr.RBV")  ## in [rad]
    wavelen = 2.0 * dmm_dsp * sin(bragg)  ## in [Ang]
    E = hc_over_e / wavelen
    if q == 0:
        print(
            "E = %.4g keV, Wavelength = %.4g Ang, DMM d-sp = %.4g Ang, Bragg = %.4g rad (%.4g deg)"
            % (E, wavelen, dmm_dsp, bragg, bragg * 180.0 / pi)
        )
    return E


def moveE(eng):
    """Moves mono_bragg correspong to the specified E(keV)"""
    wavelen = hc_over_e / eng  ## in [Ang]
    bragg = asin(wavelen / (2.0 * dmm_dsp)) * 180.0 / pi  ## in [deg]
    roll2 = -0.02617 - 0.010134 * eng  ## in [deg], based on linear fitting

    print("mono_bragg will move to %.4g deg" % bragg)
    print("mono_roll2 will move to %.4g deg" % roll2)
    yn = input("Are you sure? (y/n): ")
    if yn == "y" or yn == "Y":
        # mov(mono_bragg, bragg)
        # mov(mono_roll2, roll2)
        mono_bragg.move(bragg)
        mono_roll2.move(roll2)

        print("mono_bragg and mono_roll2 were moved")
    else:
        print("No move was made")
    getE()


def moveE_force(eng):
    """Moves mono_bragg correspong to the specified E(keV)"""
    wavelen = hc_over_e / eng  ## in [Ang]
    bragg = asin(wavelen / (2.0 * dmm_dsp)) * 180.0 / pi  ## in [deg]
    roll2 = -0.02617 - 0.010134 * eng  ## in [deg], based on linear fitting
    # mov(mono_bragg, bragg)
    # mov(mono_roll2, roll2)
    mono_bragg.move(bragg)
    mono_roll2.move(roll2)


##### toroidal mirror utilities #####
mir_us_to_ds = 1.0669  ## kinematic support pivot_to_pivot distance along Z in [m]
mir_ib_to_ob = 0.6096  ## kinematic support pivot_to_pivot distance along X in [m]


def movr_mir_pitch(del_mrad):
    """Moves the pitch of the mirror support by specified angle in [mrad]"""
    del_mm = 0.5 * mir_us_to_ds * del_mrad
    # movr(mir_usy,-del_mm)
    # movr(mir_dsyi,del_mm)
    # movr(mir_dsyo,del_mm)
    mir_usy.move(mir_usy.user_readback.value + -del_mm)
    mir_dsyi.move(mir_dsyi.user_readback.value + del_mm)
    mir_dsyo.move(mir_dsyo.user_readback.value + del_mm)


def movr_mir_roll(del_mrad):
    """Moves the roll of the mirror support by specified angle in [mrad]"""
    del_mm = 0.5 * mir_ib_to_ob * del_mrad
    # movr(mir_dsyi,-del_mm)
    # movr(mir_dsyo,del_mm)
    mir_dsyi.move(mir_dsyi.user_readback.value + -del_mm)
    mir_dsyo.move(mir_dsyo.user_readback.value + del_mm)


def movr_mir_yaw(del_mrad):
    """Moves the yaw of the mirror support by specified angle in [mrad]"""
    del_mm = 0.5 * mir_us_to_ds * del_mrad
    # movr(mir_usx,-del_mm)
    # movr(mir_dsx,del_mm)
    mir_usx.move(mir_usx.user_readback.value + -del_mm)
    mir_dsx.move(mir_dsx.user_readback.value + del_mm)


def movr_mir_y(del_mm):
    """Moves the mirror support vertically by specified distance in [mm]"""
    # movr(mir_usy,del_mm)
    # movr(mir_dsyi,del_mm)
    # movr(mir_dsyo,del_mm)
    mir_usy.move(mir_usy.user_readback.value + del_mm)
    mir_dsyi.move(mir_dsyi.user_readback.value + del_mm)
    mir_dsyo.move(mir_dsyo.user_readback.value + del_mm)


def movr_mir_x(del_mm):
    """Moves the mirror support horizontally, normal to beam, by specified distance in [mm]"""
    # movr(mir_usx,del_mm)
    # movr(mir_dsx,del_mm)
    mir_usx.move(mir_usx.user_readback.value + del_mm)
    mir_dsx.move(mir_dsx.user_readback.value + del_mm)


def ave_mir_y():
    """Returns the average height of the toroidal mirror in [mm]"""
    usy = caget("XF:11BMA-OP{Mir:Tor-Ax:YU}Mtr.RBV")
    dsyi = caget("XF:11BMA-OP{Mir:Tor-Ax:YDI}Mtr.RBV")
    dsyo = caget("XF:11BMA-OP{Mir:Tor-Ax:YDO}Mtr.RBV")
    ave_y = 0.5 * (usy + 0.5 * (dsyi + dsyo))
    print("Average mirror support height = %.4f mm relative to nominal zero height (1400 mm)" % ave_y)
    return ave_y


def get_mir_angle():
    '''calculate the incident angle of the toroidal mirror in [mrad]'''
    usy = mir_usy.user_readback.value 
    dsyi = mir_dsyi.user_readback.value
    dsyo = mir_dsyo.user_readback.value
    dsy = (dsyi + dsyo)/2
    alpha_rad = np.arctan((dsy-usy)/mir_us_to_ds/1000)
    alpah_deg = np.rad2deg(alpha_rad)
    print(f"The incident angle of the toroidal mirror is {alpha_rad*1e3: .4f} mrad, equal to {alpah_deg: .4f} deg.")
    return alpha_rad*1e3