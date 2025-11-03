print(f'Loading {__file__}')
########## FOE ##########


########## Endstation ##########
###TODO: change bim3, bim4, bim5 to EPICS signal

## Ion chamber: FMB Oxford I404
from math import exp


def curr_to_flux(amp):
    """Converts Ion Chamber current [A] to flux [ph/s] for FMB Oxford IC filled with gas N2 at 1 atm"""
    E = getE(q=1)  ## E in [keV]
    V_ion = 0.036  ## ionization energy of N2 gas in [keV]
    IC_len = 6.0  ## active length of Ion Chamber in [cm]
    qe = 1.602e-19  ## electron charge in [C]

    ## Absorption length [cm] of gas N2 (1 atm, 1.131 g/L) vs E [keV]
    # based on polynomial fit to the calculated abs length data from: henke.lbl.gov/optical_constants/atten2.html
    # see /home/xf11bm/masa/atten_len_N2*
    abs_len = 355.21 - 112.26 * E + 11.200 * E * E - 0.10611 * E * E * E

    N_abs = amp * V_ion / (qe * E)
    flux = N_abs / (1.0 - exp(-IC_len / abs_len))
    return flux


def get_bim3(q=0):
    """Returns flux at ion chamber in [ph/s] (q=1 for quiet)"""
    bim3_v1 = caget("XF:11BMB-BI{IM:3}:IC1_MON")
    bim3_v2 = caget("XF:11BMB-BI{IM:3}:IC2_MON")
    bim3_h1 = caget("XF:11BMB-BI{IM:3}:IC3_MON")
    bim3_h2 = caget("XF:11BMB-BI{IM:3}:IC4_MON")
    flux_v = curr_to_flux(bim3_v1 + bim3_v2)
    flux_h = curr_to_flux(bim3_h1 + bim3_h2)
    if q == 0:
        print("BIM3 Ion Chamber (Endstation)")
        print("Vertical:")
        print(" Signal 1: %.3e A" % bim3_v1)
        print(" Signal 2: %.3e A" % bim3_v2)
        print(" Signal - total: %.3e A" % (bim3_v1 + bim3_v2))
        print("   Flux - total: %.3e ph/s" % flux_v)
        print("Horizontal:")
        print(" Signal 1: %.3e A" % bim3_h1)
        print(" Signal 2: %.3e A" % bim3_h2)
        print(" Signal - total: %.3e A" % (bim3_h1 + bim3_h2))
        print("   Flux - total: %.3e ph/s" % flux_h)
    return flux_h


##### Following added in July 2017: Check first #####
## Scintillation detector: FMB Oxford C400, channel 1
def get_bim4(q=0):
    """Returns flux at scintillation detector in [cts/s] (q=1 for quiet)"""
    bim4_sec = caget("XF:11BMB-BI{IM:4}:GET_PERIOD")
    bim4_cts = caget("XF:11BMB-BI{IM:4}:C1_1")

    ### Ratio between estimated beam flux to raw scintillator counts (see Olog entry on July 7, 2017)
    # For unslitted, unattenuated beam, BIM4 yields 2.86E5 cts/sec for 1.85E11 ph/s at BIM3:
    # 1.85E11 / 2.86E5 = 647000 (ph/s)/(cts/sec).
    # bim4_factor = 647000.

    ### Ratio between estimated beam flux to raw scintillator counts (see Olog entry on January 18, 2018)
    # For unslitted beam with absorber 4 and evacuated chamber,
    # BIM4 yields 1.978E5 cts/sec for 1.73E11 ph/s at BIM3 and 1.55e11 ph/s at Pilatus2M:
    # Scale factor = (1.545e11 ph/sec) / (1.978e+05 ph/s) = 7.786e5.
    bim4_factor = 7.786e5

    if bim4_sec != 0.0:
        bim4_cps = bim4_cts / bim4_sec
        bim4_flux = bim4_cps * bim4_factor
    else:
        bim4_cps = -1
        bim4_flux = -1

    if q == 0:
        print("BIM4 Scintillation Detector (Endstation)")
        print("Raw count rate: %.3e ph/s" % bim4_cps)
        print("Beam flux: %.3e ph/s" % bim4_flux)
        if bim4_flux == -1:
            print("Counting time seems to be set to zero. Check settigs on FMB Oxford C400.")

    return bim4_flux


## Dectris RIGI diamond diode BPM: FMB Oxford F460, channels 1-4
def get_bim5(q=0):
    """Returns flux at 4-quadrant diamond diode BPM in [cts/s] (q=1 for quiet)"""

    bim5_i0_dark = -3.8e-10  # dark current in A
    bim5_i1_dark = 5.5e-10  # dark current in A
    bim5_i2_dark = 2.3e-10  # dark current in A
    bim5_i3_dark = 5.3e-10  # dark current in A
    bim5_i0 = caget("XF:11BMB-BI{BPM:1}Cur:I0-I") - bim5_i0_dark  # upper left
    bim5_i1 = caget("XF:11BMB-BI{BPM:1}Cur:I1-I") - bim5_i1_dark  # upper right
    bim5_i2 = caget("XF:11BMB-BI{BPM:1}Cur:I2-I") - bim5_i2_dark  # lower left
    bim5_i3 = caget("XF:11BMB-BI{BPM:1}Cur:I3-I") - bim5_i3_dark  # lower right

    ### Ratio between estimated beam flux to raw TOTAL current for the 4 quadrants
    # (see Olog entry on July 7, 2017).
    # For unslitted, unattenuated beam, BIM5 yields a TOTAL current of 4.8E-8 A at ~230 mA ring current,
    # corresponding to 1.38E11 ph/s at BIM3:
    # 1.38E11 / 4.8E-8 = 0.29E19 (ph/s)/A.
    # With dark current (total = 9.3e-10 A = 0.093e-8 A) taken into account,
    # 1.38E11 / 4.7E-8 = 0.294E19 (ph/s)/A.
    # bim5_curr_to_flux = 2.94E18

    ### Ratio between estimated beam flux to raw TOTAL current for the 4 quadrants
    # (see Olog entry on January 18, 2018).
    # For unslitted beam with absorber 4 and evacuated chamber,
    # BIM5 yields 5.09e-8 A for for 1.73E11 ph/s at BIM3 and 1.55e11 ph/s at Pilatus2M:
    # Scale factor = (1.545e11 ph/sec) / (5.0902e-08 A) = 3.025e+18
    bim5_curr_to_flux = 3.025e18

    bim5_T = bim5_i0 + bim5_i1
    bim5_B = bim5_i2 + bim5_i3
    bim5_L = bim5_i0 + bim5_i2
    bim5_R = bim5_i1 + bim5_i3

    bim5_v = (bim5_T - bim5_B) / (bim5_T + bim5_B)
    bim5_h = (bim5_R - bim5_L) / (bim5_R + bim5_L)
    bim5_flux = bim5_curr_to_flux * (bim5_T + bim5_B)

    if q == 0:
        print("BIM5 Diamond Diode BPM (Endstation)")
        print("Vertical:")
        print(" Signal Top: %.3e A" % bim5_T)
        print(" Signal Bottom: %.3e A" % bim5_B)
        print(" Offset [-1(B) to 1(T), 0 at center]: %.3f" % bim5_v)
        print("Horizontal:")
        print(" Signal Right: %.3e A" % bim5_R)
        print(" Signal Left: %.3e A" % bim5_L)
        print(" Offset [-1(L) to 1(R), 0 at center]: %.3f" % bim5_h)
        print("Total Signal: %.3e A" % (bim5_T + bim5_B))
        print("Total Flux: %.3e ph/s" % bim5_flux)

    return bim5_flux
