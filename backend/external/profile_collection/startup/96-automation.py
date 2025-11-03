#!/usr/bin/python
# -*- coding: utf-8 -*-
# vi: ts=4 sw=4


################################################################################
#  Classes for controlling the robotics and automation on the beamline.
################################################################################
# Known Bugs:
#  N/A
################################################################################
# TODO:
#  Search for "TODO" below.
#
# Config file or globalstore to save information.
################################################################################


class SampleExchangeRobot(Stage):
    def __init__(self, name="SampleExchangeRobot", base=None, use_gs=True, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._sample = None
        self.moving = False

        # The region can be:
        #  'safe' : arm won't collid with anything, it is near the (+,+,+) limit of its travel.
        #  'parking' : arm is close to the parking lot (movement may hit a sample)
        #  'stage' : arm is close to the sample stage/stack (movement may collide with stack, on-axis camera, or downstream window)
        #  'undefined' : position is unknown (do not assume it is safe to move!)
        self._region = "undefined"

        self._status = "inGarage"

        # self.yabs(-82.0) # Good height for 'slotted approach'
        # self.yabs(-77.0) # Good height for 'grip' (grip-screws sitting at bottom of wells)
        # self.yabs(-67.0) # Good height for 'hover' (sample held above stage)
        self._delta_y_hover = 5.0
        self._delta_y_slot = 4.0

        #'SAFE' position of gripper
        self._position_safe = [0, -104.9, 0.0, +90]  # x, y, z, r, phi

        # self._position_sample_gripped = [-100, -104.9, -94.8, 18.0, +90] # x, y, z, r, phi
        # self._position_hold = [0, -104.9, -94.8, 0.0, +90] # x, y, z, r, phi

        # default position of gripper to pick up from Stage
        # self._position_sample_gripped = [ -100.49986249999999, -102.89986875, -93.7, 17.5, 90.0 ] # x, y, z, r, phi
        # self._position_hold = [ 0, -106.89986875, -94.04984999999999, 0, 90.0 ] # x, y, z, r, phi

        # tested without SmarAct stage. smx=50; smy=-2.37
        # self._position_sample_gripped = [ -99, -107, -94, 0.0, 91 ] # x, y, z, r, phi
        # self._position_hold = [ 0, -107, -94, 0, 91 ] # x, y, z, r, phi

        ##tested with the gripper with spring. smx=50; smy=-2.37
        # self._position_sample_gripped = [ -98, -103, -94.5, 0.0, 91 ] # x, y, z, r, phi
        # self._position_hold = [ 0, -103, -94.5, 0, 91 ] # x, y, z, r, phi

        # tested with the gripper with spring. smx=50; smy=-2.37
        self._position_sample_gripped = [-98, -103, -94.5, 91]  # x, y, z,  phi
        self._position_hold = [0, -103, -94.5, 91]  # x, y, z,  phi

        # defacult position of gripper to pick up from Garage(1,1)
        # self._position_garage = [-96, -200, -129.5, 0.0, 0.0] # x, y, z, r, phi
        # self._position_garage = [ -97.5, -201.000121875, -129.9003625, -0.412427, 0.0 ] # x, y, z, r, phi
        # self._position_garage = [ -98.999675, -198, -130.000375, -0.621273, 0.0 ] # x, y, z, r, phi
        # self._position_garage = [ -98, -200, -128, 0.0, 1 ] # x, y, z, r, phi

        # tested with the gripper with spring. smx=50; smy=-2.37
        # self._position_garage = [ -96, -197.5, -127, 0.0, 1 ] # x, y, z, r, phi

        # Manual tweak KY (2017-11-28)
        self._position_garage = [-96, -197.5 - 0.5, -128.5, 1]  # x, y, z, phi

        # default position for stage
        # self._position_stg_exchange = [+30.0, -2.37] # smx, smy
        # self._position_stg_safe = [-30.0, -2.37] # smx, smy
        # self._position_stg_measure = [] # smx, smy

        # default position for stage without SmarAct motor
        # self._position_stg_exchange = [+50.0, -2.37] # smx, smy
        self._position_stg_exchange = [
            +51.5,
            -1.87,
        ]  # smx, smy # Manual tweak KY (2017-11-28)
        self._position_stg_safe = [-30.0, -2.37]  # smx, smy
        self._position_stg_measure = []  # smx, smy

        # garage structure parameter
        self._delta_garage_x = 44.45  # 1.75 inch = 44.45 mm
        self._delta_garage_y = 63.5  # 2.5 inch = 63.5 mm

        # if use_gs and hasattr(gs, 'robot'):
        # if '_position_sample_gripped' in gs.robot:
        # self._position_sample_gripped = gs.robot['_position_sample_gripped']
        # if '_position_hold' in gs.robot:
        # self._position_hold = gs.robot['_position_hold']
        # if '_position_garage' in gs.robot:
        # self._position_garage = gs.robot['_position_garage']
        # else:
        # gs.robot = {}

        for axis_name, axis in self._axes.items():
            axis._move_settle_max_time = 30.0

    def _set_axes_definitions(self):
        """Internal function which defines the axes for this stage. This is kept
        as a separate function so that it can be over-ridden easily."""

        # The _axes_definitions array holds a list of dicts, each defining an axis
        self._axes_definitions = [
            {
                "name": "x",
                "motor": armx,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": "positive moves left/outboard",
            },
            # {'name': 'r',
            #'motor': armr,
            ##'motor': strans,
            #'enabled': True,
            #'scaling': +1.0,
            #'units': 'mm',
            #'hint': 'positive moves radial arm outwards',
            # },
            {
                "name": "y",
                "motor": army,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": "positive moves arm up",
            },
            {
                "name": "z",
                "motor": armz,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": "positive moves arm downstream",
            },
            {
                "name": "phi",
                "motor": armphi,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": "positive moves arm downstream",
            },
        ]

    def home_y(self, verbosity=3, delays=0.5, retries=5, max_wait=2.0):
        if retries < 1:
            print("ERROR: home_y failed (too many retries).")
            return False

        # Activate homing
        caput("XF:11BMB-ES{SM:1-Ax:Y}Start:Home-Cmd", 1)

        # Make sure homing actually started
        start_time = time.time()
        while caget("XF:11BMB-ES{SM:1-Ax:Y}Sts:Homing-Sts") != 1 and time.time() - start_time < max_wait:
            time.sleep(0.01)
            if verbosity >= 5:
                print(
                    "phase, status, homing = {}, {}, {}".format(
                        caget("XF:11BMB-ES{SM:1-Ax:Y}Sts:Home-Sts"),
                        caget("XF:11BMB-ES{SM:1-Ax:Y}Pgm:Home-Sts"),
                        caget("XF:11BMB-ES{SM:1-Ax:Y}Sts:Homing-Sts"),
                    )
                )

        if time.time() - start_time > max_wait:
            # Retry
            self.home_y(
                verbosity=verbosity,
                delays=delays,
                retries=retries - 1,
                max_wait=max_wait,
            )

        # Wait for motion to be complete
        time.sleep(delays)
        while army.moving or caget("XF:11BMB-ES{SM:1-Ax:Y}Sts:Homing-Sts") != 0:
            time.sleep(delays)

        if abs(self.yabs(verbosity=0) - 0) > 0.1:
            print("ERROR: y didn't home (position = {})".format(self.yabs(verbosity=0)))
            return False

        if (
            caget("XF:11BMB-ES{SM:1-Ax:Y}Sts:Home-Sts") != 7
            or caget("XF:11BMB-ES{SM:1-Ax:Y}Pgm:Home-Sts") != 0
            or caget("XF:11BMB-ES{SM:1-Ax:Y}Sts:Homing-Sts") != 0
        ):
            print(
                "phase, status, homing = {}, {}, {}".format(
                    caget("XF:11BMB-ES{SM:1-Ax:Y}Sts:Home-Sts"),
                    caget("XF:11BMB-ES{SM:1-Ax:Y}Pgm:Home-Sts"),
                    caget("XF:11BMB-ES{SM:1-Ax:Y}Sts:Homing-Sts"),
                )
            )
            print("ERROR: y homing failed.")
            return False

        return True

    def home(self, verbosity=3, delays=0.5):
        """Home the axes, so that one can now trust the position values."""

        if not self.checkSafe():
            return

        if self._sample is not None:
            print("ERROR: You shouldn't home the robot with a sample in the gripper.")
            return

        if verbosity >= 3:
            print("Homing robot")

        # army to positive limit (moves arm to top of vertical range); set this to be zero
        success = self.home_y(verbosity=verbosity, delays=delays)
        if not success:
            return False

        # armx to positive limit; set this to be zero
        self.xabs(0)
        caput("XF:11BMB-ES{SM:1-Ax:X}Start:Home-Cmd", 1)
        time.sleep(delays)
        while armx.moving:
            time.sleep(delays)

        # Rotate arm so that it doesn't collide when doing a +z scan
        self.phiabs(+90)  # gripper pointing -x (towards sample stack)
        time.sleep(delays)
        while armphi.moving:
            time.sleep(delays)

        # Home armphi
        caput("XF:11BMB-ES{SM:1-Ax:Yaw}Start:Home-Cmd", 1)
        time.sleep(delays)
        while armphi.moving and caget("XF:11BMB-ES{SM:1-Ax:Yaw}Sts:Homing-Sts") != 0:
            time.sleep(delays)

        # Rotate arm so that it doesn't collide when doing a +z scan
        self.phiabs(+90)  # gripper pointing -x (towards sample stack)
        time.sleep(delays)
        while armphi.moving:
            time.sleep(delays)

        # armz to positive limit (moves arm to downstream of range); set this to be zero
        caput("XF:11BMB-ES{SM:1-Ax:Z}Start:Home-Cmd", 1)
        time.sleep(delays)
        while armz.moving:
            time.sleep(delays)

        # caput('XF:11BMB-ES{SM:1-Ax:ArmR}Mtr.HOMF',1) # armr home forward
        # caput('XF:11BMB-ES{SM:1-Ax:ArmR}Mtr.HOMR',1) # armr home reverse
        # time.sleep(delays)
        # while self._axes['r'].motor.moving:
        # time.sleep(delays)

        self._region = "safe"
        return True

    def checkMove(self, verbosity=3):
        if (
            armx.motor_done_move.value != 1
            or army.motor_done_move.value != 1
            or armz.motor_done_move.value != 1
            or armphi.motor_done_move.value != 1
        ):
            if verbosity >= 3:
                print("Robot is moving!")
            return self.moving == True
        else:
            if verbosity >= 3:
                print("Robot is NOT moving!")
            return self.moving == False

    def checkSafe(self, check_stage=True):
        if self._region is not "safe":
            print(
                "ERROR: Robot arm must start in the 'safe' region of the chamber (current region is '{}'). Move the robot to the safe region (and/or set _region to 'safe').".format(
                    self._region
                )
            )
            return False

        # smx_safe, smy_safe = self._position_stg_safe
        # if check_stage and smx.user_readback.value > (smx_safe+0.1):
        # print("ERROR: smx ({}) is in an unsafe position.".format(smx.user_readback.value))
        # return False

        return True

    def setReferenceSampleGripped(self):
        """The position when the sample (on the stage) is gripped."""

        x = self.xpos(verbosity=0)
        y = self.ypos(verbosity=0)
        z = self.zpos(verbosity=0)
        # r = self.rpos(verbosity=0)
        phi = self.phipos(verbosity=0)
        self._position_sample_gripped = x, y, z, phi  # x, y, z, phi

        print("self._position_sample_gripped = [ {}, {}, {}, {} ] # x, y, z, phi".format(x, y, z, phi))

        # if hasattr(gs, 'robot'):
        # gs.robot['_position_sample_gripped'] = self._position_sample_gripped

        self._position_hold = 0, y, z, phi  # x, y, z, phi

        print("self._position_hold = [ {}, {}, {}, {} ] # x, y, z, phi".format(0, y, z, phi))

        # if hasattr(gs, 'robot'):
        # gs.robot['_position_hold'] = self._position_hold

    def setReferenceGarage(self):
        """The position when the lower-left (1,1) sample of the garage is gripped."""

        x = self.xpos(verbosity=0)
        y = self.ypos(verbosity=0)
        z = self.zpos(verbosity=0)
        # r = self.rpos(verbosity=0)
        phi = self.phipos(verbosity=0)

        self._position_garage = x, y, z, phi  # x, y, z, phi

        print("self._position_garage = [ {}, {}, {}, {} ] # x, y, z, phi".format(x, y, z, phi))

        # if hasattr(gs, 'robot'):
        # gs.robot['_position_garage'] = self._position_garage

    def motionSlot(self, direction):
        self.yr(direction * self._delta_y_slot)

    def motionHover(self, direction):
        self.yr(direction * self._delta_y_hover)

    def sequenceGotoSafe(self, verbosity=3):
        x, y, z, phi = self._position_safe

        # if abs( self.phipos(verbosity=verbosity) - 90 ) < 0.1:
        ## phi = 90deg, prongs pointed at stage
        # pass
        # elif abs( self.phipos(verbosity=verbosity) - 0 ) < 0.1:
        ## phi = 0deg, prongs pointed at stage
        # pass
        # else:
        # pass

        self.phiabs(phi)
        self.yabs(y)
        self.xabs(x)
        self.zabs(z)

        self._region = "safe"

    def _sequenceGotoSafe(self, verbosity=3):
        x, y, z, phi = self._position_safe

        # if abs( self.phipos(verbosity=verbosity) - 90 ) < 0.1:
        ## phi = 90deg, prongs pointed at stage
        # pass
        # elif abs( self.phipos(verbosity=verbosity) - 0 ) < 0.1:
        ## phi = 0deg, prongs pointed at stage
        # pass
        # else:
        # pass

        yield from  self.phiabs(phi)
        yield from  self.yabs(y)
        yield from  self.xabs(x)
        yield from  self.zabs(z)

        self._region = "safe"

    def sequenceGotoSampleStageSlotted(self, x_motion=True, verbosity=3):
        if self._sample is not None:
            print(
                "ERROR: There is already a sample being gripped by robot arm (sample {}.".format(self._sample.name)
            )
            return

        if not self.checkSafe():
            return

        if verbosity >= 3:
            print("TBD")

        # Move sample stage
        x, y = self._position_stg_exchange  # smx, smy
        # mov([smx, smy], [x,y])
        smx.move(x)
        smy.move(y)

        x, y, z, phi = self._position_sample_gripped

        # Pre-align the arm in (y,z)
        self.phiabs(phi, verbosity=verbosity)
        self.zabs(z, verbosity=verbosity)
        self.yabs(y - self._delta_y_slot, verbosity=verbosity)

        self._region = "stage"
        if x_motion:
            # Move arm so that it is slotted over the sample
            self.xabs(x, verbosity=verbosity)
            # self.rabs(r, verbosity=verbosity)

    def _sequenceGotoSampleStageSlotted(self, x_motion=True, verbosity=3):
        if self._sample is not None:
            print(
                "ERROR: There is already a sample being gripped by robot arm (sample {}.".format(self._sample.name)
            )
            return

        if not self.checkSafe():
            return

        if verbosity >= 3:
            print("TBD")

        # Move sample stage
        x, y = self._position_stg_exchange  # smx, smy
        # mov([smx, smy], [x,y])
        yield from bps(mv(smx, x, smy, y, sth, 0))
        # smx.move(x)
        # smy.move(y)

        x, y, z, phi = self._position_sample_gripped

        # Pre-align the arm in (y,z)
        yield from self.phiabs(phi, verbosity=verbosity)
        yield from self.zabs(z, verbosity=verbosity)
        yield from self.yabs(y - self._delta_y_slot, verbosity=verbosity)

        self._region = "stage"
        if x_motion:
            # Move arm so that it is slotted over the sample
            yield from self.xabs(x, verbosity=verbosity)
            # self.rabs(r, verbosity=verbosity)

    def sequencePutSampleOntoStage(self, gotoSafe=True, verbosity=3):
        if self._sample is None:
            print("ERROR: No sample currently being gripped by robot arm.")
            return

        if not self.checkSafe(check_stage=False):
            return

        if verbosity >= 3:
            print("Putting sample onto stage")

        # Move sample stage
        x, y = self._position_stg_exchange  # smx, smy
        # mov([smx, smy], [x,y])
        smx.move(x)
        smy.move(y)
        sth.move(0)

        while smx.moving == True or smy.moving == True or sth.moving == True:
            time.sleep(1)

        x, y, z, phi = self._position_sample_gripped

        # Pre-align the arm in (y,z)
        self.phiabs(phi, verbosity=verbosity)
        self.zabs(z, verbosity=verbosity)
        self.yabs(y + self._delta_y_hover, verbosity=verbosity)

        self._region = "stage"
        # Push the sample out so that it is hovering above the stage
        # r is removed without SmarAct motor
        # mov([armx, self._axes['r'].motor], [x, r])
        # mov(armx,x)
        armx.move(x)

        # Move sample down (-y)
        self.yr(-self._delta_y_hover, verbosity=verbosity)  # Now in contact with stage

        # De-grip
        self.yr(-self._delta_y_slot, verbosity=verbosity)

        self._sample = None
        self._status = "onStage"

        # Move away from stage
        x, y, z, phi = self._position_hold
        # mov([armx, self._axes['r'].motor], [x, r])
        # r is removed without SmarAct motor
        # mov(armx, x)
        armx.move(x)

        if gotoSafe == True:
            self.sequenceGotoSafe(verbosity=verbosity)
        else:
            self._region = "stage"


    def _sequencePutSampleOntoStage(self, gotoSafe=True, verbosity=3):
        if self._sample is None:
            print("ERROR: No sample currently being gripped by robot arm.")
            return

        if not self.checkSafe(check_stage=False):
            return

        if verbosity >= 3:
            print("Putting sample onto stage")

        # Move sample stage
        x, y = self._position_stg_exchange  # smx, smy
        # mov([smx, smy], [x,y])
        
        yield from bps(mv(smx, x, smy, y, sth, 0))
        # yield from bps(mv(smy, y))
        # yield from bps(mv(sth, 0))

        # while smx.moving == True or smy.moving == True or sth.moving == True:
        #     time.sleep(1)

        x, y, z, phi = self._position_sample_gripped

        # Pre-align the arm in (y,z)
        yield from self.phiabs(phi, verbosity=verbosity)
        yield from self.zabs(z, verbosity=verbosity)
        yield from self.yabs(y + self._delta_y_hover, verbosity=verbosity)

        self._region = "stage"
        # Push the sample out so that it is hovering above the stage
        # r is removed without SmarAct motor
        # mov([armx, self._axes['r'].motor], [x, r])
        # mov(armx,x)
        yield from bps(mv(armx, x))

        # Move sample down (-y)
        yield from self.yr(-self._delta_y_hover, verbosity=verbosity)  # Now in contact with stage

        # De-grip
        yield from self.yr(-self._delta_y_slot, verbosity=verbosity)

        self._sample = None
        self._status = "onStage"

        # Move away from stage
        x, y, z, phi = self._position_hold
        # mov([armx, self._axes['r'].motor], [x, r])
        # r is removed without SmarAct motor
        # mov(armx, x)
        # armx.move(x)
        yield from bps(mv(armx, x))


        if gotoSafe == True:
            yield from self._sequenceGotoSafe(verbosity=verbosity)
        else:
            self._region = "stage"


    def sequenceGotoSampleStageSlotted(self, x_motion=True, verbosity=3):
        if self._sample is not None:
            print(
                "ERROR: There is already a sample being gripped by robot arm (sample {}.".format(self._sample.name)
            )
            return

        if not self.checkSafe():
            return

        if verbosity >= 3:
            print("TBD")

        # Move sample stage
        x, y = self._position_stg_exchange  # smx, smy
        # mov([smx, smy], [x,y])
        smx.move(x)
        smy.move(y)

        x, y, z, phi = self._position_sample_gripped

        # Pre-align the arm in (y,z)
        self.phiabs(phi, verbosity=verbosity)
        self.zabs(z, verbosity=verbosity)
        self.yabs(y - self._delta_y_slot, verbosity=verbosity)

        self._region = "stage"
        if x_motion:
            # Move arm so that it is slotted over the sample
            self.xabs(x, verbosity=verbosity)
            # self.rabs(r, verbosity=verbosity)

    def _sequenceGotoSampleStageSlotted(self, x_motion=True, verbosity=3):
        if self._sample is not None:
            print(
                "ERROR: There is already a sample being gripped by robot arm (sample {}.".format(self._sample.name)
            )
            return

        if not self.checkSafe():
            return

        if verbosity >= 3:
            print("TBD")

        # Move sample stage
        x, y = self._position_stg_exchange  # smx, smy
        # mov([smx, smy], [x,y])
        # smx.move(x)
        # smy.move(y)
        yield from bps(mv(smx, x, smy, y, sth, 0))
        # yield from bps.mov(smy, y)

        x, y, z, phi = self._position_sample_gripped

        # Pre-align the arm in (y,z)
        yield from self.phiabs(phi, verbosity=verbosity)
        yield from self.zabs(z, verbosity=verbosity)
        yield from self.yabs(y - self._delta_y_slot, verbosity=verbosity)

        self._region = "stage"
        if x_motion:
            # Move arm so that it is slotted over the sample
            # self.xabs(x, verbosity=verbosity)
            yield from bps.mov(smx, x)

    def sequenceGetSampleFromStage(self, gotoSafe=True, verbosity=3):
        if self._sample is not None:
            print(
                "ERROR: There is already a sample being gripped by robot arm (sample {}.".format(self._sample.name)
            )
            return

        if not self.checkSafe():
            return

        if verbosity >= 3:
            print("Getting sample from stage")

        # Move sample stage
        x, y = self._position_stg_exchange  # smx, smy
        # mov([smx, smy], [x,y])
        smx.move(x)
        smy.move(y)
        sth.move(0)

        x, y, z, phi = self._position_sample_gripped

        # Pre-align the arm in (y,z)
        self.phiabs(phi, verbosity=verbosity)
        self.zabs(z, verbosity=verbosity)
        self.yabs(y - self._delta_y_slot, verbosity=verbosity)

        self._region = "stage"
        # Move arm so that it is slotted over the sample
        self.xabs(x, verbosity=verbosity)
        # self.rabs(r, verbosity=verbosity)

        # Grip sample
        self.yr(+self._delta_y_slot, verbosity=verbosity)

        self._sample = "exists"  # TODO: Fix
        self._status = "onRobot"

        # Pick sample up (+y)
        self.yr(+self._delta_y_hover, verbosity=verbosity)

        # Move away from stage
        x, y, z, phi = self._position_hold
        # mov([armx, self._axes['r'].motor], [x, r])
        # r is removed without SmarAct motor
        # mov(armx, x)
        armx.move(x)

        if gotoSafe == True:
            self.sequenceGotoSafe(verbosity=verbosity)

    def _sequenceGetSampleFromStage(self, gotoSafe=True, verbosity=3):
        if self._sample is not None:
            print(
                "ERROR: There is already a sample being gripped by robot arm (sample {}.".format(self._sample.name)
            )
            return

        if not self.checkSafe():
            return

        if verbosity >= 3:
            print("Getting sample from stage")

        # Move sample stage
        x, y = self._position_stg_exchange  # smx, smy
        # mov([smx, smy], [x,y])
        yield from bps(mv(smx, x, smy, y, sth, 0))
        # smx.move(x)
        # smy.move(y)
        # sth.move(0)

        x, y, z, phi = self._position_sample_gripped

        # Pre-align the arm in (y,z)
        yield from self.phiabs(phi, verbosity=verbosity)
        yield from self.zabs(z, verbosity=verbosity)
        yield from self.yabs(y - self._delta_y_slot, verbosity=verbosity)

        self._region = "stage"
        # Move arm so that it is slotted over the sample
        yield from self.xabs(x, verbosity=verbosity)
        # self.rabs(r, verbosity=verbosity)

        # Grip sample
        yield from self.yr(+self._delta_y_slot, verbosity=verbosity)

        self._sample = "exists"  # TODO: Fix
        self._status = "onRobot"

        # Pick sample up (+y)
        yield from self.yr(+self._delta_y_hover, verbosity=verbosity)

        # Move away from stage
        x, y, z, phi = self._position_hold
        # mov([armx, self._axes['r'].motor], [x, r])
        # r is removed without SmarAct motor
        # mov(armx, x)
        # armx.move(x)
        yield from bps(mv(armx, x))

        if gotoSafe == True:
            yield from self.sequenceGotoSafe(verbosity=verbosity)

    def sequenceGetSampleFromGarage(self, shelf_num, spot_num, gotoSafe=True, verbosity=3):
        if shelf_num < 1 or shelf_num > 4:
            print("ERROR: Invalid shelf {}".format(shelf_num))
            return
        if spot_num < 1 or spot_num > 3:
            print("ERROR: Invalid spot {}".format(spot_num))
            return

        if self._sample is not None:
            print(
                "ERROR: There is already a sample being gripped by robot arm (sample {}.".format(self._sample.name)
            )
            return

        if verbosity >= 3:
            print("Getting sample from garage ({}, {})".format(shelf_num, spot_num))

        x, y, z, phi = self._position_garage

        self.phiabs(phi)

        # Align ourselves with this parking spot
        success = self.sequencePrepGarageXY(shelf_num, spot_num, verbosity=verbosity)
        if not success:
            return
        x = self.xpos(verbosity=0)
        y = self.ypos(verbosity=0)

        # Lower so that the slot is aligned
        self.yr(-self._delta_y_slot, verbosity=verbosity)

        # Move towards parking lot
        self._region = "parking"
        self.zabs(z, verbosity=verbosity)

        # Grip sample
        self.yr(+self._delta_y_slot, verbosity=verbosity)

        self._sample = "exists"  # TODO: Fix
        self._status = "onRobot"

        # Pick sample up (+y)
        self.yr(+self._delta_y_hover, verbosity=verbosity)

        # Move away from parking
        self.zabs(0, verbosity=verbosity)
        self.xabs(0, verbosity=verbosity)
        self.yabs(self._position_safe[1], verbosity=verbosity)
        if gotoSafe == True:
            self.sequenceGotoSafe(verbosity=verbosity)

    def _sequenceGetSampleFromGarage(self, shelf_num, spot_num, gotoSafe=True, verbosity=3):
        if shelf_num < 1 or shelf_num > 4:
            print("ERROR: Invalid shelf {}".format(shelf_num))
            return
        if spot_num < 1 or spot_num > 3:
            print("ERROR: Invalid spot {}".format(spot_num))
            return

        if self._sample is not None:
            print(
                "ERROR: There is already a sample being gripped by robot arm (sample {}.".format(self._sample.name)
            )
            return

        if verbosity >= 3:
            print("Getting sample from garage ({}, {})".format(shelf_num, spot_num))

        x, y, z, phi = self._position_garage

        yield from self.phiabs(phi)

        # Align ourselves with this parking spot
        success = self.sequencePrepGarageXY(shelf_num, spot_num, verbosity=verbosity)
        if not success:
            return
        x = self.xpos(verbosity=0)
        y = self.ypos(verbosity=0)

        # Lower so that the slot is aligned
        yield from  self.yr(-self._delta_y_slot, verbosity=verbosity)

        # Move towards parking lot
        self._region = "parking"
        yield from  self.zabs(z, verbosity=verbosity)

        # Grip sample
        yield from  self.yr(+self._delta_y_slot, verbosity=verbosity)

        self._sample = "exists"  # TODO: Fix
        self._status = "onRobot"

        # Pick sample up (+y)
        yield from  self.yr(+self._delta_y_hover, verbosity=verbosity)

        # Move away from parking
        yield from  self.zabs(0, verbosity=verbosity)
        yield from  self.xabs(0, verbosity=verbosity)
        yield from  self.yabs(self._position_safe[1], verbosity=verbosity)
        if gotoSafe == True:
            yield from self.sequenceGotoSafe(verbosity=verbosity)

    def sequencePutSampleInGarage(self, shelf_num, spot_num, gotoSafe=True, verbosity=3):
        if shelf_num < 1 or shelf_num > 4:
            print("ERROR: Invalid shelf {}".format(shelf_num))
            return
        if spot_num < 1 or spot_num > 3:
            print("ERROR: Invalid spot {}".format(spot_num))
            return

        if self._sample is None:
            print("WARNING: There is no sample being gripped by robot arm.")

        if verbosity >= 3:
            print("Putting sample into garage ({}, {})".format(shelf_num, spot_num))

        x, y, z, phi = self._position_garage

        self.phiabs(phi)

        # Align ourselves with this parking spot
        success = self.sequencePrepGarageXY(shelf_num, spot_num, verbosity=verbosity)
        if not success:
            return
        x = self.xpos(verbosity=0)
        y = self.ypos(verbosity=0)

        # Hover
        self.yr(+self._delta_y_hover, verbosity=verbosity)

        # Move towards parking lot
        self._region = "parking"
        self.zabs(z, verbosity=verbosity)

        # Deposit sample
        self.yr(-self._delta_y_hover, verbosity=verbosity)
        self.yr(-self._delta_y_slot, verbosity=verbosity)

        self._sample = None
        self._status = "inGarage"

        # Move away from parking
        self.zabs(0, verbosity=verbosity)
        self.xabs(0, verbosity=verbosity)
        if gotoSafe == True:
            self.sequenceGotoSafe(verbosity=verbosity)

    def _sequencePutSampleInGarage(self, shelf_num, spot_num, gotoSafe=True, verbosity=3):
        if shelf_num < 1 or shelf_num > 4:
            print("ERROR: Invalid shelf {}".format(shelf_num))
            return
        if spot_num < 1 or spot_num > 3:
            print("ERROR: Invalid spot {}".format(spot_num))
            return

        if self._sample is None:
            print("WARNING: There is no sample being gripped by robot arm.")

        if verbosity >= 3:
            print("Putting sample into garage ({}, {})".format(shelf_num, spot_num))

        x, y, z, phi = self._position_garage

        yield from self.phiabs(phi)

        # Align ourselves with this parking spot
        success = self.sequencePrepGarageXY(shelf_num, spot_num, verbosity=verbosity)
        if not success:
            return
        x = self.xpos(verbosity=0)
        y = self.ypos(verbosity=0)

        # Hover
        yield from self.yr(+self._delta_y_hover, verbosity=verbosity)

        # Move towards parking lot
        self._region = "parking"
        yield from self.zabs(z, verbosity=verbosity)

        # Deposit sample
        yield from self.yr(-self._delta_y_hover, verbosity=verbosity)
        yield from self.yr(-self._delta_y_slot, verbosity=verbosity)

        self._sample = None
        self._status = "inGarage"

        # Move away from parking
        yield from self.zabs(0, verbosity=verbosity)
        yield from self.xabs(0, verbosity=verbosity)
        if gotoSafe == True:
            yield from self.sequenceGotoSafe(verbosity=verbosity)

    def sequencePrepGarageXY(self, shelf_num, spot_num, verbosity=3):
        if shelf_num < 1 or shelf_num > 4:
            print("ERROR: Invalid shelf {}".format(shelf_num))
            return
        if spot_num < 1 or spot_num > 3:
            print("ERROR: Invalid spot {}".format(spot_num))
            return

        if not self.checkSafe():
            return

        if verbosity >= 3:
            print("  Going to garage ({}, {})".format(shelf_num, spot_num))

        if self.zpos(verbosity=0) < -10:
            print("ERROR: z ({}) position unsafe.".format(self.zpos(verbosity=0)))

        # if abs(self.phipos(verbosity=0))>1:
        #     print("ERROR: phi ({}) position unsafe.".format(self.phipos(verbosity=0)))

        x, y, z, phi = self._position_garage

        x += (spot_num - 1) * self._delta_garage_x
        y += (shelf_num - 1) * self._delta_garage_y

        if verbosity >= 4:
            print("    Going to (x,y) = ({}, {})".format(x, y))

        # Do y first to avoid catching cables
        self.yabs(y)
        self.xabs(x)

        xactual = self.xpos(verbosity=0)
        yactual = self.ypos(verbosity=0)

        if abs(x - xactual) > 0.2:
            print("ERROR: x did not reach requested position (request = {}, actual = {})".format(x, xactual))
            return False

        if abs(y - yactual) > 0.2:
            print("ERROR: y did not reach requested position (request = {}, actual = {})".format(y, yactual))
            return False

        return True

    def _sequencePrepGarageXY(self, shelf_num, spot_num, verbosity=3):
        if shelf_num < 1 or shelf_num > 4:
            print("ERROR: Invalid shelf {}".format(shelf_num))
            return
        if spot_num < 1 or spot_num > 3:
            print("ERROR: Invalid spot {}".format(spot_num))
            return

        if not self.checkSafe():
            return

        if verbosity >= 3:
            print("  Going to garage ({}, {})".format(shelf_num, spot_num))

        if self.zpos(verbosity=0) < -10:
            print("ERROR: z ({}) position unsafe.".format(self.zpos(verbosity=0)))

        # if abs(self.phipos(verbosity=0))>1:
        #     print("ERROR: phi ({}) position unsafe.".format(self.phipos(verbosity=0)))

        x, y, z, phi = self._position_garage

        x += (spot_num - 1) * self._delta_garage_x
        y += (shelf_num - 1) * self._delta_garage_y

        if verbosity >= 4:
            print("    Going to (x,y) = ({}, {})".format(x, y))

        # Do y first to avoid catching cables
        yield from self.yabs(y)
        yield from self.xabs(x)

        xactual = self.xpos(verbosity=0)
        yactual = self.ypos(verbosity=0)

        if abs(x - xactual) > 0.2:
            print("ERROR: x did not reach requested position (request = {}, actual = {})".format(x, xactual))
            return False

        if abs(y - yactual) > 0.2:
            print("ERROR: y did not reach requested position (request = {}, actual = {})".format(y, yactual))
            return False

        return True
    
    def loadSample(self, shelf_num, spot_num, verbosity=3):
        # Check if a sample is on stage
        # Unload sample if necessary

        self.sequenceGetSampleFromGarage(shelf_num, spot_num, verbosity=verbosity)
        self.sequencePutSampleOntoStage()

    def _loadSample(self, shelf_num, spot_num, verbosity=3):
        # Check if a sample is on stage
        # Unload sample if necessary

        yield from self.sequenceGetSampleFromGarage(shelf_num, spot_num, verbosity=verbosity)
        yield from self.sequencePutSampleOntoStage()

    def calibrateStage(self, verbosity=3):
        if self._sample is not None:
            print("ERROR: Calibration cannot be done with sample on robot arm.")
            return

        if not self.checkSafe(check_stage=False):
            return

        if verbosity >= 3:
            print("Approaching to stage")

        # Move sample stage
        x, y = self._position_stg_exchange  # smx, smy
        # mov([smx, smy], [x,y])
        smx.move(x)
        smy.move(y)

        x, y, z, phi = self._position_sample_gripped

        # Pre-align the arm in (y,z)
        self.phiabs(phi, verbosity=verbosity)
        self.zabs(z, verbosity=verbosity)
        self.yabs(y - self._delta_y_slot, verbosity=verbosity)

        # Move to a position x is away from sample bar
        self.xabs(-40, verbosity=verbosity)
        # self.rabs(0, verbosity=verbosity)

    def calibrateGarage(self, verbosity=3):
        # use Garage(1,1) to calibration the gripper position
        shelf_num = 1
        spot_num = 1

        if self._sample is not None:
            print("ERROR: Calibration cannot be done with sample on robot arm. ")
            return

        if verbosity >= 3:
            print("Approaching to garage ({}, {})".format(shelf_num, spot_num))

        x, y, z, phi = self._position_garage

        self.phiabs(phi)

        # Align ourselves with this parking spot
        success = self.sequencePrepGarageXY(shelf_num, spot_num, verbosity=verbosity)
        if not success:
            return
        x = self.xpos(verbosity=0)
        y = self.ypos(verbosity=0)

        # Lower so that the slot is aligned
        self.yr(-self._delta_y_slot, verbosity=verbosity)
        self.zabs(-60)


    def pickupHolder(self, slot, gotoSafe=True, verbosity=3):
        # picking up the holer from Garage
        # shelf_num, spot_num: slot number of the holder
        [shelf_num, spot_num] = slot
        if verbosity >= 3:
            print("picking up from garage ({}, {})".format(shelf_num, spot_num))

        if self._sample is not None:
            print(
                "ERROR: There is already a sample being gripped by robot arm (sample {}.".format(self._sample.name)
            )
            return

        self.sequenceGetSampleFromGarage(shelf_num, spot_num, gotoSafe=gotoSafe, verbosity=verbosity)
        time.sleep(1)
        # always go back to safe position from or to the stage
        self.sequencePutSampleOntoStage(verbosity=verbosity)

    def _pickupHolder(self, slot, gotoSafe=True, verbosity=3):
        # picking up the holer from Garage
        # shelf_num, spot_num: slot number of the holder
        [shelf_num, spot_num] = slot
        if verbosity >= 3:
            print("picking up from garage ({}, {})".format(shelf_num, spot_num))

        if self._sample is not None:
            print(
                "ERROR: There is already a sample being gripped by robot arm (sample {}.".format(self._sample.name)
            )
            return

        yield from self.sequenceGetSampleFromGarage(shelf_num, spot_num, gotoSafe=gotoSafe, verbosity=verbosity)
        time.sleep(1)
        # always go back to safe position from or to the stage
        yield from self.sequencePutSampleOntoStage(verbosity=verbosity)

    def returnHolder(self, slot, gotoSafe=True, verbosity=3):
        # returning the holer back to Garage
        # shelf_num, spot_num: slot number of the holder
        [shelf_num, spot_num] = slot
        if verbosity >= 3:
            print("returning back to garage ({}, {})".format(shelf_num, spot_num))

        # always go back to safe position from or to the stage
        self.sequenceGetSampleFromStage(verbosity=verbosity)
        time.sleep(1)
        # only this one need options NOT to return robot to the default position
        self.sequencePutSampleInGarage(shelf_num, spot_num, gotoSafe=gotoSafe, verbosity=verbosity)


    def _returnHolder(self, slot, gotoSafe=True, verbosity=3):
        # returning the holer back to Garage
        # shelf_num, spot_num: slot number of the holder
        [shelf_num, spot_num] = slot
        if verbosity >= 3:
            print("returning back to garage ({}, {})".format(shelf_num, spot_num))

        # always go back to safe position from or to the stage
        yield from self.sequenceGetSampleFromStage(verbosity=verbosity)
        time.sleep(1)
        # only this one need options NOT to return robot to the default position
        yield from self.sequencePutSampleInGarage(shelf_num, spot_num, gotoSafe=gotoSafe, verbosity=verbosity)

    def _stress_test(self, cycles=2, verbosity=5):
        if not self.checkSafe():
            return

        # self.home()

        if not self.checkSafe():
            return

        for i in range(cycles):
            if verbosity >= 2:
                print("Stress test cycle {}".format(i))

            for shelf_num in range(1, 4 + 1):
                for spot_num in range(1, 3 + 1):
                    if verbosity >= 2:
                        print("Stress test garage ({}, {})".format(shelf_num, spot_num))

                    self.sequenceGetSampleFromGarage(shelf_num, spot_num, verbosity=verbosity)
                    self.sequencePutSampleOntoStage(verbosity=verbosity)

                    time.sleep(3)

                    self.sequenceGetSampleFromStage(verbosity=verbosity)
                    self.sequencePutSampleInGarage(shelf_num, spot_num, verbosity=verbosity)

    def run(self, cycles=1, verbosity=3):
        if not self.checkSafe():
            return

        for hol in Garage_holders:
            [shelf_num, spot_num] = hol.GaragePosition
            if verbosity >= 2:
                print("Run test garage ({}, {})".format(shelf_num, spot_num))

            self.sequenceGetSampleFromGarage(shelf_num, spot_num, verbosity=verbosity)
            time.sleep(2)
            self.sequencePutSampleOntoStage(verbosity=verbosity)

            hol.listSamples()
            time.sleep(2)
            hol.doSamples()

            self.sequenceGetSampleFromStage(verbosity=verbosity)
            time.sleep(2)
            self.sequencePutSampleInGarage(shelf_num, spot_num, verbosity=verbosity)
            time.sleep(2)

    def run_test(self, verbosity=3):
        if not self.checkSafe():
            return

        for hol in Garage_holders:
            [shelf_num, spot_num] = hol.GaragePosition
            if verbosity >= 2:
                print("Run test garage ({}, {})".format(shelf_num, spot_num))

            self.sequenceGetSampleFromGarage(shelf_num, spot_num, verbosity=verbosity)
            print("out of garage")
            time.sleep(2)
            self.sequencePutSampleOntoStage(verbosity=verbosity)

            hol.listSamples()
            time.sleep(2)
            hol.doSamples()

            self.sequenceGetSampleFromStage(verbosity=verbosity)
            time.sleep(2)
            self.sequencePutSampleInGarage(shelf_num, spot_num, gotoSafe=False, verbosity=verbosity)
            time.sleep(2)

        self.sequenceGotoSafe(verbosity=verbosity)

    def listGarage(self, verbosity=3):
        for hol in Garage_holders:
            [shelf_num, spot_num] = hol.GaragePosition
            print("In Garage ({}, {})".format(shelf_num, spot_num))
            hol.listSamplesPositions()

    def initiallize(self, verbosity=3):
        
        print('please clean up the robot and any bar from the robotic arm and the sample area.')
        ret3 = input("Is it done? (y/n) ")
        if ret3 == "y" or ret3 == "yes":
            print("Thank you")
            print("The initiallization is done.")
        else:
            print("The initiallization is NOT complete.")
            return
        
        self._region = 'safe'
        self._status = 'inGarage'
        self._sample = None

        if verbosity>=3:
            print('robot._region = {}'.format(robot._region))
            print('robot._status = {}'.format(robot._status))
            print('robot._sample = {}'.format(robot._sample))
        


# class Queue(object):
class Queue(CoordinateSystem):
    """
    Holds the current state of the sample queue, allowing samples settings
    to be 'extracted'; or even allowing a particular sample to be physically
    loaded.
    functions:
    1. check the status of the robot and the holder
    2. define the holders in the garage and the corresponding measurement methods and sequence
    3. run the robot without unecessary movement
    """

    def __init__(self, name="Queue", base=None, **kwargs):
        if base is None:
            base = get_default_stage()

        # super().__init__(name=name, base=base, **kwargs)

        self._holders = {}
        self._current = None  # the current holder on the stage.
        self._currentSample = (
            None  # the current sample measured on the stage, self._currentSample = [holder, sample_number]
        )
        self.status = None  # status of the current robot, including 'onStage', 'onRobot', 'inGarage'
        self._sequence = {}  # the current holder on the stage

        self.reset_clock()

    def clock(self):
        """Return the current value of the "clock" variable. This provides a
        way to set a clock/timer for a sample. For instance, you can call
        "reset_clock" when you initiate some change to the sample. Thereafter,
        the "clock" method lets you check how long it has been since that
        event."""

        clock_delta = time.time() - self.clock_zero
        return clock_delta

    def reset_clock(self):
        """Resets the sample's internal clock/timer to zero."""

        self.clock_zero = time.time()

        return self.clock()

    def checkStatus(self, verbosity=3):
        """check the status of the robot and holder
        including where is the robot
        holder, which holder is on the stage.

        """

        # robot part

        # The region can be:
        #  'safe' : arm won't collid with anything, it is near the (+,+,+) limit of its travel.
        #  'parking' : arm is close to the parking lot (movement may hit a sample)
        #  'stage' : arm is close to the sample stage/stack (movement may collide with stack, on-axis camera, or downstream window)
        #  'undefined' : position is unknown (do not assume it is safe to move!)
        print("robot region = {}".format(robot._region))
        print("robot holding sample = {}".format(robot._sample != None))
        # print('robot moving is {}'.format(robot.checkMove()))

        robot._region = "safe"
        # holder part

        if robot._sample == True:  # holder sitting on the robot
            self.status = "onRobot"
        # elif robot._status == 'inGarage':
        # self.status = 'inGarage'
        # elif robot._status == 'onStage':
        # self.status = 'onStage'
        elif self._current == None:
            self.status = "inGarage"
        else:
            self.status = "onStage"
            # self._current = []

        # promp question to check the status of the current stage
        if verbosity >= 5:
            question = [
                ["Is there any holder on the stage? (y/n)"],
                ["Is there any holder on the robot? (y/n)"],
                ["If yes, what is the holder?"],
            ]
            try:
                print("Tell me more about what is on the stage ::: ")
                ret0 = input("  Q : {} : ".format(question[0]))
                ret1 = input("  Q : {} : ".format(question[1]))
                ret2 = input("  Q : {} : ".format(question[2]))
            except KeyboardInterrupt:
                return

            print("Your input is:")
            print("There is holder on stage ::: {}", ret0)
            print("There is holder on robot ::: {}", ret1)
            print("The holder is ::: {}", ret2)
            ret3 = input("Is it correct? (y/n) ")
            if ret3 == "y" or ret3 == "yes":
                print("Thank you")
            else:
                return

            if ret0 == "y" or ret0 == "yes":
                self.status = "onStage"
                for key, item in que._sequence.items():
                    # for ret2 in self._holders.items()):
                    if item.name == ret2:
                        self._current = item
                        # print('There is no holder with the same name. Please double-check')
                if self._current == None:
                    print("ERROR : The holder name is not in the queue list. Please double-check")
                    return

            elif ret1 == "y" or ret1 == "yes":
                self.status = "parking"
            else:
                self.status = "inGarage"

        if verbosity >= 3:
            print("Current queue status = {}".format(self.status))
            if self._current == None:
                print("The current holder is {}".format(self._current))
            else:
                print("The current holder is {}".format(self._current.name))
            # print('Current robot region is safe')

        return self.status

    def setStatus(self, status, current, verbosity=3):
        # set the current status of the queue
        self.status = status
        self._current = current
        if verbosity >= 3:
            self.checkStatus()

    def addHolder(self, holder, slot):
        """add a holder (assembly of samples) to robot garage
        the slots are listed from slot_pos [1, 1] to [4, 3], corresponding to the slot_number from 1 to 12
        """
        if slot is None:
            if len(self._holders) == 0:
                slot_number = 1
                slot_pos = [1, 1]
            elif len(self._holders) == 1:
                ki = [int(key) for key in self._holders.keys()]
                slot_pos = [np.max(ki) / 3, np.max(ki) / 3 + 1]
                slot_number = slot
        elif len(slot) == 1:
            slot_pos = [int(slot / 3) + 1, slot % 3 + 1]
            slot_number = slot
        elif len(slot) == 2:
            slot_pos = slot
            slot_number = (slot[0] - 1) * 3 + slot[1]

        # print(slot_number)

        if slot_number in self._holders.keys():
            print(
                'Warning: Slot pos {} is already defined". Use "replaceSample" if you are sure you want to eliminate the existing holder from the garage.'.format(
                    slot_pos
                )
            )

        else:
            self._holders[slot_number] = holder

        self._holders[slot_number] = holder

        # holder.set_base_stage(self)
        holder.md["holder_number"] = slot_number
        holder.md["holder_position"] = slot_pos
        holder.slot_number = slot_number
        holder.slot_pos = slot_pos

    def addHolderIntoQueue(self, holder, slot, sequence_number):
        self.addHolder(holder, slot=slot)
        # holder.addGaragePosition(slot)
        holder.sequence_number = sequence_number

    def removeHolder(self, slot):
        """Remove a particular holder from garage."""

        del self._holders[slot]

    def removeHoldersAll(self):
        self._holders = {}

    def replaceHolder(self, holder, slot):
        """Replace a given holder on garage with a different sample."""

        self.removeHolder(slot)
        self.addHolder(holder, slot)

    def getHolder(self, slot, verbosity=3):
        """Return the requested sample object from this holder/bar.
        slot has 3 kinds of inputs:
           1. integer from 1 to 12
           2. list [1, 1] to [4, 3]
           3. holder name
        """

        # TODO :
        if type(slot) is int:
            # slot = [np.max(slot)/3, np.max(slot)/3+1]

            if slot not in self._holders:
                if verbosity >= 1:
                    print("Error: Slot {} not defined.".format(slot))
                return None

            slot_match = self._holders[slot]

            if verbosity >= 3:
                print("{}: {:s}".format(slot, slot_match.name))

            return slot_match

        elif type(slot) is list:
            if len(slot) >= 2 and type(slot[0]) == int and type(slot[1]) == int:
                slot_number = (slot[0] - 1) * 3 + slot[1]
                if slot_number not in self._holders:
                    if verbosity >= 1:
                        print("Error: Slot {} not defined in the garage.".format(slot))
                        # print('sss')
                    return None
                slot_match = self._holders[slot_number]

                if verbosity >= 3:
                    print("{}: {:s}".format(slot, slot_match.name))

                return slot_match
            else:
                if verbosity >= 1:
                    print("Error: Slot {} not defined.".format(slot))
                return None

        elif type(slot) is str:
            # First search for an exact name match
            matches = 0
            slot_match = None
            ct_match = None
            for ct, holder in sorted(self._holders.items()):
                if holder.name == slot:
                    matches += 1
                    if slot_match is None:
                        slot_match = holder
                        ct_match = ct

            if matches == 1:
                if verbosity >= 3:
                    print("{}: {:s}".format(ct_match, slot_match.name))
                return slot_match

            elif matches > 1:
                if verbosity >= 2:
                    print(
                        '{:d} exact matches for "{:s}", returning slot {}: {:s}'.format(
                            matches, slot, ct_match, holder_match.name
                        )
                    )
                return slot_match

        # elif type(slot) is holder:
        # slot_match = slot
        # return slot_match
        # else:
        # return None

        ## Try to find a 'start of name' match
        # for slot_i, slot in sorted(self._holders.items()):
        # if slot.name.startswith(slot):
        # matches += 1
        # if slot_match is None:
        # slot_match = slot
        # slot_i_match = slot_i

        # if matches==1:
        # if verbosity>=3:
        # print('Beginning-name match: {}: {:s}'.format(slot_i_match, slot_match.name))
        # return slot_match

        # elif matches>1:
        # if verbosity>=2:
        # print('{:d} beginning-name matches for "{:s}", returning slot {}: {:s}'.format(matches, slot, slot_i_match, slot_match.name))
        # return slot_match

        ## Try to find a substring match
        # for slot_i, slot in sorted(self._holders.items()):
        # if slot in slot.name:
        # matches += 1
        # if slot_match is None:
        # slot_match = slot
        # slot_i_match = slot_i

        # if matches==1:
        # if verbosity>=3:
        # print('Substring match: {}: {:s}'.format(slot_i_match, slot_match.name))
        # return slot_match

        # elif matches>1:
        # if verbosity>=2:
        # print('{:d} substring matches for "{:s}", returning slot {}: {:s}'.format(matches, slot, slot_i_match, slot_match.name))
        # return slot_match

        # if verbosity>=1:
        # print('No slot has a name matching "{:s}"'.format(slot))
        # return None

        else:
            print('Error: Holder designation "{}" not understood.'.format(slot))
            return None

    import string

    def getHolders(self, holder_range=None, verbosity=3):
        """Get the list of holders associated with this holder.

        If the optional holder_range argument is provided (2-tuple), then only holder
        numbers within that holder_range (inclusive) are run. If holder_range is instead a
        string, then all holders with names that match are returned."""

        holders = []

        if holder_range is None:
            for holder_number in sorted(self._holders):
                holders.append(self._holders[holder_number])

        elif type(holder_range) is list:
            if type(holder_range[0]) is int:
                if len(holder_range) == 2:
                    start, stop = holder_range
                    for holder_number in sorted(self._holders):
                        if holder_number >= start and holder_number <= stop:
                            holders.append(self._holders[holder_number])
                else:
                    for holder_number in sorted(self._holders):
                        for ii in holder_range:
                            if holder_number == ii:
                                holders.append(self._holders[holder_number])

            elif type(holder_range[0]) is str:  # For 96 well holder, format: A1, D2 ...
                for holder_number in sorted(self._holders):
                    holder_row = string.ascii_lowercase(holder_number[0])
                    holder_column = int(holder_number[1:])
                    holder_number = holder_row * 12 + holder_column
                    holders.append(self._holders[holder_number])

        elif type(holder_range) is str:
            for holder_number, holder in sorted(self._holders.items()):
                if holder_range in holder.name:
                    holders.append(holder)

        elif type(holder_range) is int:
            holders.append(self._holders[holder_range])

        else:
            if verbosity >= 1:
                print('Range argument "{}" not understood.'.format(holder_range))
        return holders

    def listHolders(self):
        """Print a list of the current holders in garage."""

        for holder_number, holder in sorted(self._holders.items()):
            print("{}: {:s}".format(holder_number, holder.name))

    def listSamples(self):
        """Print a list of the current samples associated with the holder/
        bars in garage."""

        for holder_number, holder in sorted(self._holders.items()):
            print("{}: {:s}".format(holder_number, holder.name))
            for sample_number, sample in sorted(holder._samples.items()):
                print("{}: {:s}".format(sample_number, sample.name))

    def checkSamples(self, verbosity=3):
        if verbosity > 0:
            self.listSamples()
        # check positions to make sure the samples are in the range of smx
        if verbosity > 3:
            error_signal = False
            for holder_number, holder in sorted(self._holders.items()):
                for sample_number, sample in sorted(holder._samples.items()):
                    sample_xpos = holder.position["x"] + sample.position
                    if sample_xpos < smx.limits[0] or sample_xpos > smx.limits[-1]:
                        print(
                            "ERROR: out of limit of smx motor: holder--{}: sample--{}".format(
                                holder_number, sample_number
                            )
                        )
                        error_signal = True
            if error_signal == True:
                input(" Please ctrl+c and correct the errors.")

    def returnHolder(self, holder="current", gotoSafe=True, force=False):
        """return the holder from stage back to garage.
        holder = None:  retrun to default position
        holder = [2 ,3]:  return to specific garage position
        """
        if type(holder) == list or type(holder) == int:
            holder = self.getHolder(holder)
        if holder != "current":  # and holder is not in sorted(self._holders.items()):
            print("Please use que.returnHoder().")
            # answer = input('Is this slot {} safe for the holder?'.format(slot))
            # if answer == 'yes' or 'y':
            # robot.returnHolder(holder.slot_pos, gotoSafe=gotoSafe)
            ##return

        if self.status == "onStage":
            if holder == "current":
                robot.returnHolder(self._current.slot_pos, gotoSafe=gotoSafe)
            else:
                if self._current == holder:
                    robot.returnHolder(holder.slot_pos, gotoSafe=gotoSafe)
                else:
                    # TODO:
                    for hol in sorted(self._holders.items()):
                        if holder.slot_number == hol.slot_number and hol != self._current:
                            print("There is holder position has been occupied in garage")
                            return

                    answer = input("Is this slot {} safe for the holder?".format(slot))
                    if answer == "yes" or "y":
                        robot.returnHolder(holder.slot_pos, gotoSafe=gotoSafe)
                    else:
                        print("The return of the holder is canceled")
            self.status = "inGarage"
            self._current = None
        else:
            print("There is no holder on the stage")

    def pickupHolder(self, holder, force=False, gotoSafe=True, slack=False):
        """pick up the holder from garage to stage."""
        # holder = self.getHolder(slot, verbosity=0)
        # checkt the status
        self.checkStatus()

        if self.status == "inGarage" and self._current == None:
            robot.pickupHolder(holder.slot_pos, gotoSafe=gotoSafe)
            self._current = holder
            self.status = "onStage"
            post_to_slack(
                text="holder <<<{}>>> is onStage for measurements".format(self._current.name),
                slack=slack,
            )
            return self._current
        elif self.status == "onStage":
            print("There is a holder on the stage.")
            if holder == self._current:
                print("The same holder on the stage")
                self._current = holder
                self.status = "onStage"
                post_to_slack(
                    text="holder <<<{}>>> is onStage for measurements".format(self._current.name),
                    slack=slack,
                )
                return self._current
            if force == True:
                print("Remove the current holder from the stage")
                self.returnHolder()
                robot.pickupHolder(holder.slot_pos, gotoSafe=gotoSafe)
                self._current = holder
                self.status = "onStage"
                post_to_slack(
                    text="holder <<<{}>>> is onStage for measurements".format(self._current.name),
                    slack=slack,
                )
                return self._current
            else:
                print("Nothing happened becasue the stage is occupied. Make sure force=True")
                return None
        else:
            print("There is holder on stage or robot arm. No room for the holder.")

    def gotoSample(self, holder, sample_number, force=False):
        holder = self.pickupHolder(holder, force=force)
        if holder is not None:
            sample = holder.getSample(sample_number, verbosity=0)
            sample.gotoAlignedPosition()
        return sample

    def setSequence(self):
        """Print the sequence to measure the holders associated with this robot."""
        self._sequence = {}
        for slot, holder in sorted(self._holders.items()):
            # print( '{}: {:s}, {:d}'.format(holder.name, holder_slot_number))
            try:
                self._sequence[holder.sequence_number] = holder
                # self._sequence[holder.name] = holder.name
            except AttributeError:
                self._sequence[slot + 100] = holder
                holder.sequence_number = slot + 100
        self._sequenceList = sorted(self._sequence.items())
        self.listSequence()
        return self._sequence

    def listSequence(self):
        if self._sequence == None:
            print("No sequence is defined.")
            print("Run que.setSequence() .")
        else:
            print("Current sequence is defined as: ")
            for seq, holder in sorted(self._sequence.items()):
                print("Sequence NO. {} ======>  {} at Garage {}".format(seq, holder.name, holder.slot_number))

    def runSequence(self, startSample=None, endSample=None, gotoSafeForce=True, currentSample=False):
        """run the sequence of the holders in garage. It allows to start from a given sample.
        startSample: [holder, sample], like [[2, 3], 2]
        endSample: [holder, sample], like [[2, 3], 2]
        """
        self.setSequence()
        # The startSample and endSample have to be a standard list with 2 items, like [holder, sample]
        if startSample != None and len(startSample) != 2:
            print(
                "Please provide the correct sample position to start in garage. The format is like [[2, 3], 2], or [hol1, 3] or [5, 3]"
            )
            return
        if endSample != None and len(endSample) != 2:
            print(
                "Please provide the correct sample position to finish in garage. The format is like [[2, 3], 2], or [hol1, 3] or [5, 3]"
            )
            return

        if startSample == None:  # when the input is None, the startSample from the first sample in the queue
            # startSample = [[0, 0], 0]
            startHolder = min(self._sequence.items())[-1]
            startSample = [startHolder.slot_pos, min(startHolder._samples.items())[0]]
        elif (
            type(startSample[0]) == list or type(startSample[0]) == int
        ):  # when the input is holder position or the queue number
            startHolder = self.getHolder(startSample[0])
        else:  # when the input is holder name
            startHolder = startSample[0]

        if endSample == None:
            endHolder = max(self._sequence.items())[-1]
            endSample = [endHolder.slot_pos, max(endHolder._samples.items())[0]]
        elif type(endSample[0]) == list or type(endSample[0]) == int:
            endHolder = self.getHolder(endSample[0])
        else:
            endHolder = endSample[0]

        # check the corrent sequence of the start and end samples.
        # if startSample[0] != endSample[0]:
        if startHolder != endHolder:
            if startHolder.sequence_number > endHolder.sequence_number:
                return print("The start holder is listed after the last one in the sequence. Please double-check")
        # if the start and end samples are in the same bar
        else:
            if startSample[1] > endSample[1]:
                return print("The start sample is listed after the last one in the holder. Please double-check")

        # start the sequence
        for seq, holder in sorted(self._sequence.items()):
            # determine when the robot moves back to safe position. #TODO: test gotoSafe
            if gotoSafeForce == True:
                gotoSafe = True
            elif holder.slot_number < endHolder.slot_number:
                gotoSafe = False
            else:
                gotoSafe = True

            if holder.sequence_number == startHolder.sequence_number:
                # double-check self._current #TODO:pickupHolder
                if self._current == None or self._current.slot_number != startHolder.slot_number:
                    self.pickupHolder(holder, force=True, gotoSafe=gotoSafe)
                # self._current.doSamples()
                for sample_number, sample in sorted(self._current._samples.items()):
                    if sample_number >= int(startSample[1]):
                        print("working the the sample {} on holder {}".format(sample.name, holder.name))
                        self._currentSample = [holder, sample_number]
                        sample.do()  # run the sample
                        # self._currentSample = []
            elif (
                holder.sequence_number > startHolder.sequence_number
                and holder.sequence_number < endHolder.sequence_number
            ):
                self.pickupHolder(holder, force=True, gotoSafe=gotoSafe)
                # self._current.doSamples()
                for sample_number, sample in sorted(self._current._samples.items()):
                    print("working the the sample {} on holder {}".format(sample.name, holder.name))
                    self._currentSample = [holder, sample_number]
                    sample.do()  # run the sample
            if holder.sequence_number == endHolder.sequence_number:
                self.pickupHolder(holder, force=True, gotoSafe=gotoSafe)
                for sample_number, sample in sorted(self._current._samples.items()):
                    if int(sample_number) <= int(endSample[1]):
                        print("working the the sample {} on holder {}".format(sample.name, holder.name))
                        self._currentSample = [holder, sample_number]
                        sample.do()  # run the sample

    def runHolder(self, holder, slack=False):
        """run a single holder with multiple samples."""
        hol = self.pickupHolder(holder, force=True, slack=slack)
        hol.doSamples()

    def runHolders(self, startHolder=None, endHolder=None, gotoSafeForce=False, slack=False):
        """run a single holder with multiple samples."""
        if startHolder == None:
            startHolder = min(self._sequence.items())[-1]
        elif (
            type(startHolder) == list or type(startHolder) == int
        ):  # when the input is holder position or the queue number
            startHolder = self.getHolder(startHolder)
        else:  # when the input is holder name
            startHolder = startHolder

        if endHolder == None:
            endHolder = max(self._sequence.items())[-1]
        elif type(endHolder) == list or type(endHolder) == int:
            endHolder = self.getHolder(endHolder)
        else:
            endHolder = endHolder

        if abs(gatex.position + 95) > 100:
        #100 to 1
            print("The gate is not OPEN. Please double-check the setting. ")
            input()

        for seq, holder in sorted(self._sequence.items()):
            # determine when the robot moves back to safe position. #TODO: test gotoSafe
            # if gotoSafeForce == False:
            # gotoSafe=False
            # elif holder.sequence_number< endHolder.sequence_number:
            # gotoSafe=False
            # else:
            # gotoSafe=True

            if (
                holder.sequence_number >= startHolder.sequence_number
                and holder.sequence_number <= endHolder.sequence_number
            ):
                self.pickupHolder(holder, force=True, slack=slack)
                # post_to_slack(text='holder <<<{}>>> is onStage for measurements'.format(que._current.name), slack=slack)
                # self.pickupHolder(holder, force=True, gotoSafe=gotoSafe)
                holder.doSamples()

    # This setting is for individual holder. Each sample could set its own setting when adding into holder
    def exposure_setting(
        self,
        exposure_time=None,
        incident_angles=None,
        detectors=None,
        detector_positions=None,
        tiling=None,
    ):
        """define the setting for exposures, including incident angles, exposure time, detector selection, detector positions.
        incident_angles: [0.05, 0.08, 0,1]
        detectors = [pilatus2M, pilatus800]
        detector positions: optimized detector position for individual detectors, like [[60, -73], [-200, 24]] for two detectors, or [[60, -73]] for one detector
        titing: 'ygaps', 'xygaps', None

        """
        # if exposure_time != None:
        # self.exposure_time = exposure_time
        # if incident_angles != None:
        # self.incident_angles = incident_angles
        # if detectors != None:
        # self.detectors = detectors
        if detector_positions != None and detectors != None:
            # self.detector_positions = detector_positions
            if len(detector_positions) != len(detectors):
                print("The quantity of detector does not match the detector positions")
                return
            detector_setting = []
            for ct, detector in enumerate(detectors):
                detector_setting.append = [detector, detector_positions(ct)]
                # detector.position = self.detector_positions(ct)

        # if tiling != None:
        # self.tiling = tiling
        # if exposure_time != None:
        # self.exposure_time = exposure_time
        return {
            "exposure_time": exposure_time,
            "incident_angles": incident_angles,
            "detector_setting": detector_setting,
            "tiling": tiling,
        }


# Note: This will break until class is updated to not use gs at all.
robot = SampleExchangeRobot(use_gs=False)
que = Queue()
# robot._region='safe'

# que_setting1 = que.exposure_setting(exposure_time=10, incident_angles=[0.1, 0.5], detectors=[pilatus2M, pilatus800],  detector_positions = [[60, -73], [-200, 24]], tiling='ygaps')
# sam.measureIncidentAngles(que_setting1.

# config_load()

###############slack communication##############
import requests

slack_channel_xf11bm = "https://hooks.slack.com/services/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"
slack_channel_CMSstatus = "https://hooks.slack.com/services/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"
slack_channel_rp = None

# json_data = {"text":"test"}


def slack_post(json_data):
    requests.post(slack_channel_xf11bm, json=json_data)


def post_to_slack(text, slack=True):
    if slack == True:
        try:
            channel = RE.md["slack_channel"]
        except:
            channel = slack_channel_CMSstatus
        if channel is None:
            channel = slack_channel_CMSstatus
        post = {"text": "{0}".format(text)}
        try:
            # json_data = json.dumps(post)
            # req = request.Request(channel,
            # data=json_data.encode('ascii'),
            # headers={'Content-Type': 'application/json'})
            r = requests.post(
                channel,
                json=post,
                headers={"Content-Type": "application/json", "Accept": "text/plain"},
            )

        except Exception as em:
            print("EXCEPTION: " + str(em))

    else:
        print(text)


# define the info to send to slack
def status_to_slack():
    text = "current holder is "
    return que._current.name
