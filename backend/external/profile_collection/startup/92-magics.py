# BlueskyMagics were imported and registered in 00-startup.py

BlueskyMagics.detectors = [pilatus2M]
BlueskyMagics.positioners = [
    smx,
    smy,
    sth,
    schi,
    sphi,
    srot,
    strans,
    strans2,
    stilt,
    stilt2,
    WAXSx,
    WAXSy,
    WAXSz,
    SAXSx,
    SAXSy,
    bsx,
    bsy,
    bsphi,
    camx,
    camy,
    armz,
    armx,
    armphi,
    army,
    armr,
    bim3y,
    fs3y,
    bim4y,
    bim5y,
    s0.tp,
    s0.bt,
    s0.ob,
    s0.ib,
    s0.xc,
    s0.yc,
    s0.xg,
    s0.yg,
    s1.xc,
    s1.yc,
    s1.xg,
    s1.yg,
    s2.xc,
    s2.yc,
    s2.xg,
    s2.yg,
    s3.xc,
    s3.yc,
    s3.xg,
    s3.yg,
    s4.xc,
    s4.yc,
    s4.xg,
    s4.yg,
    s5.xc,
    s5.yc,
    s5.xg,
    s5.yg,
    mono_bragg,
    mono_pitch2,
    mono_roll2,
    mono_perp2,
    mir_usx,
    mir_dsx,
    mir_usy,
    mir_dsyi,
    mir_dsyo,
    mir_bend,
]

### Override the %wa magic with one that includes offsets.
### Later this will be added to bluesky itself and will not
### need to be customized here.


from IPython.core.magic import Magics, magics_class, line_magic
from operator import attrgetter


# @magics_class
# class CMSCustomMagics(BlueskyMagics):
# @line_magic
# def wa(self, line):
# "List positioner info. 'wa' stands for 'where all'."
# if line.strip():
# positioners = eval(line, self.shell.user_ns)
# else:
# positioners = self.positioners
# positioners = sorted(set(positioners), key=attrgetter('name'))
# values = []
# for p in positioners:
# try:
# values.append(p.position)
# except Exception as exc:
# values.append(exc)

# headers = ['Positioner', 'Value', 'Low Limit', 'High Limit', 'Offset']
# LINE_FMT = '{: <30} {: <10} {: <10} {: <10} {: <10}'
# lines = []
# lines.append(LINE_FMT.format(*headers))
# for p, v in zip(positioners, values):
# if not isinstance(v, Exception):
# try:
# prec = p.precision
# except Exception:
# prec = self.FMT_PREC
# value = np.round(v, decimals=prec)
# try:
# low_limit, high_limit = p.limits
# except Exception as exc:
# low_limit = high_limit = exc.__class__.__name__
# else:
# low_limit = np.round(low_limit, decimals=prec)%
# high_limit = np.round(high_limit, decimals=prec)
# try:
# offset = p.user_offset.get()
# except Exception as exc:
# offset = exc.__class__.__name__
# else:
# offset = np.round(offset, decimals=prec)
# else:
# value = v.__class__.__name__  # e.g. 'DisconnectedError'
# low_limit = high_limit = ''

# lines.append(LINE_FMT.format(p.name, value, low_limit, high_limit,
# offset))
# print('\n'.join(lines))

## This will override the %wa registered from BlueskyMagics
##get_ipython().register_magics(CMSCustomMagics)


def wh_motors(motor=None, verbosity=3):
    if motor == None:
        motor_list1 = [schi, smx, smy, sphi, srot, sth, stilt, stilt2, strans, strans2]
        motor_list2 = [
            s0.xc,
            s0.xg,
            s0.yc,
            s0.yg,
            s1.xc,
            s1.xg,
            s1.yc,
            s1.yg,
            s2.xc,
            s2.xg,
            s2.yc,
            s2.yg,
            s3.xc,
            s3.xg,
            s3.yc,
            s3.yg,
            s4.xc,
            s4.xg,
            s4.yc,
            s4.yg,
            s5.xc,
            s5.xg,
            s5.yc,
            s5.yg,
        ]
        motor_list3 = [
            SAXSx,
            SAXSy,
            WAXSx,
            WAXSy,
            WAXSz,
            armphi,
            armr,
            armx,
            army,
            armz,
            bim3y,
            bim4y,
            bim5y,
            bsphi,
            bsx,
            bsy,
            camx,
            camy,
            fs3y,
            mir_bend,
            mir_dsx,
            mir_dsyi,
            mir_dsyo,
            mir_usx,
            mir_usy,
            mono_bragg,
            mono_perp2,
            mono_pitch2,
            mono_roll2,
        ]
        motor_list = motor_list1 + motor_list2 + motor_list3
    else:
        motor_list = motor

    # for motor in motor_list:
    # print('motor: {} -- position: {} -- limits: {} -- offset: -- {} direction: {}'.format(motor.name, motor.position, motor.limits, motor.user_offset.value, sth.direction_of_travel.value))

    if verbosity >= 1:
        print("|{:*^12}|{:*^12}|{:*^12}|{:*^12}|{:*^12}|{:*^12}|".format("*", "*", "*", "*", "*", "*"))
        print(
            "|{:12}|{:12}|{:12}|{:12}|{:12}|{:12}|".format(
                "motor", "position", "low_limit", "high_limit", "offset", "direction"
            )
        )
        print("|{:*^12}|{:*^12}|{:*^12}|{:*^12}|{:*^12}|{:*^12}|".format("*", "*", "*", "*", "*", "*"))
        # print('|{:12}|{:12.4}|{:8.4}|{:8.4}|{:12}|{:5}|'

    last_z = -100
    beam = True

    flux_expected = None

    for motor in motor_list:
        # print('motor: {} -- position: {} -- limits: {} -- offset: -- {} direction: {}'.format(motor.name, motor.position, motor.limits, motor.user_offset.value, sth.direction_of_travel.value))

        # print('|{:12}  | {:6.6} | {} | {:11.11} | {:5} |'.format(motor.name, motor.position, motor.limits, motor.user_offset.value, sth.direction_of_travel.value))
        ##print('|{:5.1}  | {:16.16} | {:s} | {:11.11} | {:11.11} | |'.format(motor.name, motor.position, motor.limits, motor.user_offset.value, sth.direction_of_travel.value))
        # print('|{:12}  | {:12} | {:12} | {:12} | {:5} |'.format(motor.name, motor.position, motor.limits, motor.user_offset.value, sth.direction_of_travel.value))
        print(
            "|{:12}|{:12.6}|{:12.6}|{:12.6}|{:12}|{:12}|".format(
                motor.name,
                motor.position,
                motor.low_limit,
                motor.high_limit,
                motor.user_offset.value,
                motor.user_offset_dir.value,
            )
        )
    if verbosity >= 1:
        print("|{:*^12}|{:*^12}|{:*^12}|{:*^12}|{:*^12}|{:*^12}|".format("*", "*", "*", "*", "*", "*"))
