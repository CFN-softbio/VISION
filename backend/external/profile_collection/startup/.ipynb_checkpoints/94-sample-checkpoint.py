#!/usr/bin/python
# -*- coding: utf-8 -*-
# vi: ts=4 sw=4

################################################################################
#  Code for defining a 'Sample' object, which keeps track of its state, and
# simplifies the task of aligning, measuring, etc.
################################################################################
# Known Bugs:
#  N/A
################################################################################
# TODO:
#  - Search for "TODO" below.
#  - Ability to have a collection of simultaneous motions? (E.g. build up a set
#  of deferred motions?)
#  - Use internal naming scheme to control whether 'saxs'/'waxs' is put in the
# filename
################################################################################


import time
import re
import os
import shutil

import pandas as pds
from datetime import datetime
import functools


class CoordinateSystem(object):
    """
    A generic class defining a coordinate system. Several coordinate systems
    can be layered on top of one another (with a reference to the underlying
    coordinate system given by the 'base_stage' pointer). When motion of a given
    CoordinateSystem is requested, the motion is passed (with coordinate
    conversion) to the underlying stage.
    """

    hint_replacements = {
        "positive": "negative",
        "up": "down",
        "left": "right",
        "towards": "away from",
        "downstream": "upstream",
        "inboard": "outboard",
        "clockwise": "counterclockwise",
        "CW": "CCW",
    }

    # Core methods
    ########################################

    def __init__(self, name="<unnamed>", base=None, **kwargs):
        """Create a new CoordinateSystem (e.g. a stage or a sample).

        Parameters
        ----------
        name : str
            Name for this stage/sample.
        base : Stage
            The stage on which this stage/sample sits.
        """

        self.name = name
        self.base_stage = base

        self.enabled = True

        self.md = {}
        self._marks = {}

        self._set_axes_definitions()
        self._init_axes(self._axes_definitions)
        # self.align_success = True

    def _set_axes_definitions(self):
        """Internal function which defines the axes for this stage. This is kept
        as a separate function so that it can be over-ridden easily."""

        # The _axes_definitions array holds a list of dicts, each defining an axis
        self._axes_definitions = []

    def _init_axes(self, axes):
        """Internal method that generates method names to control the various axes."""

        # Note: Instead of defining CoordinateSystem() having methods '.x', '.xr',
        # '.y', '.yr', etc., we programmatically generate these methods when the
        # class (and subclasses) are instantiated.
        # Thus, the Axis() class has generic versions of these methods, which are
        # appropriated renamed (bound, actually) when a class is instantiated.

        self._axes = {}

        for axis in axes:
            axis_object = Axis(
                axis["name"],
                axis["motor"],
                axis["enabled"],
                axis["scaling"],
                axis["units"],
                axis["hint"],
                self.base_stage,
                stage=self,
            )
            self._axes[axis["name"]] = axis_object

            # Bind the methods of axis_object to appropriately-named methods of
            # the CoordinateSystem() class.
            setattr(self, axis["name"], axis_object.get_position)
            setattr(self, axis["name"] + "abs", axis_object.move_absolute)
            setattr(self, axis["name"] + "r", axis_object.move_relative)
            setattr(self, axis["name"] + "pos", axis_object.get_position)
            setattr(self, axis["name"] + "posMotor", axis_object.get_motor_position)

            setattr(self, axis["name"] + "units", axis_object.get_units)
            setattr(self, axis["name"] + "hint", axis_object.get_hint)
            setattr(self, axis["name"] + "info", axis_object.get_info)

            setattr(self, axis["name"] + "set", axis_object.set_current_position)
            setattr(self, axis["name"] + "o", axis_object.goto_origin)
            setattr(self, axis["name"] + "setOrigin", axis_object.set_origin)
            setattr(self, axis["name"] + "mark", axis_object.mark)

            setattr(self, axis["name"] + "search", axis_object.search)
            setattr(self, axis["name"] + "scan", axis_object.scan)
            setattr(self, axis["name"] + "c", axis_object.center)

    def comment(self, text, logbooks=None, tags=None, append_md=True, **md):
        """Add a comment related to this CoordinateSystem."""

        text += "\n\n[comment for CoordinateSystem: {} ({})].".format(self.name, self.__class__.__name__)

        if append_md:
            md_current = {k: v for k, v in RE.md.items()}  # Global md
            md_current.update(get_beamline().get_md())  # Beamline md

            # Self md
            # md_current.update(self.get_md())

            # Specified md
            md_current.update(md)

            text += "\n\n\nMetadata\n----------------------------------------"
            for key, value in sorted(md_current.items()):
                text += "\n{}: {}".format(key, value)

        logbook.log(text, logbooks=logbooks, tags=tags)

    def set_base_stage(self, base):
        self.base_stage = base
        self._init_axes(self._axes_definitions)

    # Convenience/helper methods
    ########################################

    def multiple_string_replacements(self, text, replacements, word_boundaries=False):
        """Peform multiple string replacements simultaneously. Matching is case-insensitive.

        Parameters
        ----------
        text : str
            Text to return modified
        replacements : dictionary
            Replacement pairs
        word_boundaries : bool, optional
            Decides whether replacements only occur for words.
        """

        # Code inspired from:
        # http://stackoverflow.com/questions/6116978/python-replace-multiple-strings
        # Note inclusion of r'\b' sequences forces the regex-match to occur at word-boundaries.

        if word_boundaries:
            replacements = dict((r"\b" + re.escape(k.lower()) + r"\b", v) for k, v in replacements.items())
            pattern = re.compile("|".join(replacements.keys()), re.IGNORECASE)
            text = pattern.sub(
                lambda m: replacements[r"\b" + re.escape(m.group(0).lower()) + r"\b"],
                text,
            )

        else:
            replacements = dict((re.escape(k.lower()), v) for k, v in replacements.items())
            pattern = re.compile("|".join(replacements.keys()), re.IGNORECfdeASE)
            text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

        return text

    def _hint_replacements(self, text):
        """Convert a motor-hint into its logical inverse."""

        # Generates all the inverse replacements
        replacements = dict((v, k) for k, v in self.hint_replacements.items())
        replacements.update(self.hint_replacements)

        return self.multiple_string_replacements(text, replacements, word_boundaries=True)

    # Control methods
    ########################################
    def setTemperature(self, temperature, verbosity=3):
        if verbosity >= 1:
            print("Temperature functions not implemented in {}".format(self.__class__.__name__))

    def temperature(self, verbosity=3):
        if verbosity >= 1:
            print("Temperature functions not implemented in {}".format(self.__class__.__name__))

        return 0.0

    # Motion methods
    ########################################

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def is_enabled(self):
        return self.enabled

    def pos(self, verbosity=3):
        """Return (and print) the positions of all axes associated with this
        stage/sample."""

        out = {}
        for axis_name, axis_object in sorted(self._axes.items()):
            out[axis_name] = axis_object.get_position(verbosity=verbosity)
            # if verbosity>=2: print('') # \n

        return out

    def origin(self, verbosity=3):
        """Returns the origin for axes."""

        out = {}
        for axis_name, axis_object in sorted(self._axes.items()):
            origin = axis_object.get_origin()
            if verbosity >= 2:
                print("{:s} origin = {:.3f} {:s}".format(axis_name, origin, axis_object.get_units()))
            out[axis_name] = origin

        return out

    def gotoOrigin(self, axes=None):
        """Go to the origin (zero-point) for this stage. All axes are zeroed,
        unless one specifies the axes to move."""

        # TODO: Guard against possibly buggy behavior if 'axes' is a string instead of a list.
        # (Python will happily iterate over the characters in a string.)

        if axes is None:
            axes_to_move = self._axes.values()

        else:
            axes_to_move = [self._axes[axis_name] for axis_name in axes]

        for axis in axes_to_move:
            axis.goto_origin()

    def setOrigin(self, axes, positions=None):
        """Define the current position as the zero-point (origin) for this stage/
        sample. The axes to be considered in this redefinition must be supplied
        as a list.

        If the optional positions parameter is passed, then those positions are
        used to define the origins for the axes."""

        if positions is None:
            for axis in axes:
                getattr(self, axis + "setOrigin")()

        else:
            for axis, pos in zip(axes, positions):
                getattr(self, axis + "setOrigin")(pos)

    def gotoAlignedPosition(self):
        """Goes to the currently-defined 'aligned' position for this stage. If
        no specific aligned position is defined, then the zero-point for the stage
        is used instead."""

        # TODO: Optional offsets? (Like goto mark?)

        if "aligned_position" in self.md and self.md["aligned_position"] is not None:
            for axis_name, position in self.md["aligned_position"].items():
                self._axes[axis_name].move_absolute(position)

        else:
            self.gotoOrigin()

    # Motion logging
    ########################################

    def setAlignedPosition(self, axes):
        """Saves the current position as the 'aligned' position for this stage/
        sample. This allows one to return to this position later. One must
        specify the axes to be considered.

        WARNING: Currently this position data is not saved persistently. E.g. it will
        be lost if you close and reopen the console.
        """

        positions = {}
        for axis_name in axes:
            positions[axis_name] = self._axes[axis_name].get_position(verbosity=0)

        self.attributes["aligned_position"] = positions

    def mark(self, label, *axes, **axes_positions):
        """Set a mark for the stage/sample/etc.

        'Marks' are locations that have been labelled, which is useful for
        later going to a labelled position (using goto), or just to keep track
        of sample information (metadata).

        By default, the mark is set at the current position. If no 'axes' are
        specified, all motors are logged. Alternately, axes (as strings) can
        be specified. If axes_positions are given as keyword arguments, then
        positions other than the current position can be specified.
        """

        positions = {}

        if len(axes) == 0 and len(axes_positions) == 0:
            for axis_name in self._axes:
                positions[axis_name] = self._axes[axis_name].get_position(verbosity=0)

        else:
            for axis_name in axes:
                positions[axis_name] = self._axes[axis_name].get_position(verbosity=0)

            for axis_name, position in axes_positions.items():
                positions[axis_name] = position

        self._marks[label] = positions

    def marks(self, verbosity=3):
        """Get a list of the current marks on the stage/sample/etc. 'Marks'
        are locations that have been labelled, which is useful for later
        going to a labelled position (using goto), or just to keep track
        of sample information (metadata)."""

        if verbosity >= 3:
            print("Marks for {:s} (class {:s}):".format(self.name, self.__class__.__name__))

        if verbosity >= 2:
            for label, positions in self._marks.items():
                print(label)
                for axis_name, position in sorted(positions.items()):
                    print("  {:s} = {:.4f} {:s}".format(axis_name, position, self._axes[axis_name].get_units()))

        return self._marks

    def goto(self, label, verbosity=3, **additional):
        """Move the stage/sample to the location given by the label. For this
        to work, the specified label must have been 'marked' at some point.

        Additional keyword arguments can be provided. For instance, to move
        3 mm from the left edge:
          sam.goto('left edge', xr=+3.0)
        """

        if label not in self._marks:
            if verbosity >= 1:
                print(
                    "Label '{:s}' not recognized. Use '.marks()' for the list of marked positions.".format(label)
                )
            return

        for axis_name, position in sorted(self._marks[label].items()):
            if axis_name + "abs" in additional:
                # Override the marked value for this position
                position = additional[axis_name + "abs"]
                del additional[axis_name + "abs"]

            # relative = 0.0 if axis_name+'r' not in additional else additional[axis_name+'r']
            if axis_name + "r" in additional:
                relative = additional[axis_name + "r"]
                del additional[axis_name + "r"]
            else:
                relative = 0.0

            self._axes[axis_name].move_absolute(position + relative, verbosity=verbosity)

        # Handle any optional motions not already covered
        for command, amount in additional.items():
            if command[-1] == "r":
                getattr(self, command)(amount, verbosity=verbosity)
            elif command[-3:] == "abs":
                getattr(self, command)(amount, verbosity=verbosity)
            else:
                print("Keyword argument '{}' not understood (should be 'r' or 'abs').".format(command))

    # State methods
    ########################################
    def save_state(self):
        """Outputs a string you can use to re-initialize this object back
        to its current state."""
        # TODO: Save to databroker?

        state = {"origin": {}}
        for axis_name, axis in self._axes.items():
            state["origin"][axis_name] = axis.origin

        return state

    def restore_state(self, state):
        """Outputs a string you can use to re-initialize this object back
        to its current state."""

        for axis_name, axis in self._axes.items():
            axis.origin = state["origin"][axis_name]

    # End class CoordinateSystem(object)
    ########################################


class Axis(object):
    """Generic motor axis.

    Meant to be used within a CoordinateSystem() or Stage() object.
    """

    def __init__(self, name, motor, enabled, scaling, units, hint, base, stage=None, origin=0.0):
        self.name = name
        self.motor = motor
        self.enabled = enabled
        self.scaling = scaling
        self.units = units
        self.hint = hint

        self.base_stage = base
        self.stage = stage

        self.origin = 0.0

        self._move_settle_max_time = 10.0
        self._move_settle_period = 0.05
        self._move_settle_tolerance = 0.01

    # Coordinate transformations
    ########################################

    def cur_to_base(self, position):
        """Convert from this coordinate system to the coordinate in the (immediate) base."""

        base_position = self.get_origin() + self.scaling * position

        return base_position

    def base_to_cur(self, base_position):
        """Convert from this base position to the coordinate in the current system."""

        position = (base_position - self.get_origin()) / self.scaling

        return position

    def cur_to_motor(self, position):
        """Convert from this coordinate system to the underlying motor."""

        if self.motor is not None:
            return self.cur_to_base(position)

        else:
            base_position = self.cur_to_base(position)
            return self.base_stage._axes[self.name].cur_to_motor(base_position)

    def motor_to_cur(self, motor_position):
        """Convert a motor position into the current coordinate system."""

        if self.motor is not None:
            return self.base_to_cur(motor_position)

        else:
            base_position = self.base_stage._axes[self.name].motor_to_cur(motor_position)
            return self.base_to_cur(base_position)

    # Programmatically-defined methods
    ########################################
    # Note: Instead of defining CoordinateSystem() having methods '.x', '.xr',
    # '.xp', etc., we programmatically generate these methods when the class
    # (and subclasses) are instantiated.
    # Thus, the Axis() class has generic versions of these methods, which are
    # appropriated renamed (bound, actually) when a class is instantiated.
    def get_position(self, verbosity=3):
        """Return the current position of this axis (in its coordinate system).
        By default, this also prints out the current position."""

        if self.motor is not None:
            base_position = self.motor.position

        else:
            verbosity_c = verbosity if verbosity >= 4 else 0
            base_position = getattr(self.base_stage, self.name + "pos")(verbosity=verbosity_c)

        position = self.base_to_cur(base_position)

        if verbosity >= 2:
            if self.stage:
                stg = self.stage.name
            else:
                stg = "?"

            if verbosity >= 5 and self.motor is not None:
                print("{:s} = {:.3f} {:s}".format(self.motor.name, base_position, self.get_units()))

            print(
                "{:s}.{:s} = {:.3f} {:s} (origin = {:.3f})".format(
                    stg, self.name, position, self.get_units(), self.get_origin()
                )
            )

        return position

    def get_motor_position(self, verbosity=3):
        """Returns the position of this axis, traced back to the underlying
        motor."""

        if self.motor is not None:
            return self.motor.position

        else:
            return getattr(self.base_stage, self.name + "posMotor")(verbosity=verbosity)
            # return self.base_stage._axes[self.name].get_motor_position(verbosity=verbosity)

    def move_absolute(self, position=None, wait=True, verbosity=3):
        """Move axis to the specified absolute position. The position is given
        in terms of this axis' current coordinate system. The "defer" argument
        can be used to defer motions until "move" is called."""

        if position is None:
            # If called without any argument, just print the current position
            return self.get_position(verbosity=verbosity)

        # Account for coordinate transformation
        base_position = self.cur_to_base(position)

        if self.is_enabled():
            if self.motor:
                # mov( self.motor, base_position )
                self.motor.user_setpoint.value = base_position

            else:
                # Call self.base_stage.xabs(base_position)
                getattr(self.base_stage, self.name + "abs")(base_position, verbosity=0)

            if self.stage:
                stg = self.stage.name
            else:
                stg = "?"

            if verbosity >= 2:
                # Show a realtime output of position
                start_time = time.time()
                current_position = self.get_position(verbosity=0)
                while (
                    abs(current_position - position) > self._move_settle_tolerance
                    and (time.time() - start_time) < self._move_settle_max_time
                ):
                    current_position = self.get_position(verbosity=0)
                    print(
                        "{:s}.{:s} = {:5.3f} {:s}      \r".format(
                            stg, self.name, current_position, self.get_units()
                        ),
                        end="",
                    )
                    time.sleep(self._move_settle_period)

            # if verbosity>=1:
            # current_position = self.get_position(verbosity=0)
            # print( '{:s}.{:s} = {:5.3f} {:s}        '.format(stg, self.name, current_position, self.get_units()))

        elif verbosity >= 1:
            print("Axis %s disabled (stage %s)." % (self.name, self.stage.name))

    def move_relative(self, move_amount=None, verbosity=3):
        """Move axis relative to the current position."""

        if move_amount is None:
            # If called without any argument, just print the current position
            return self.get_position(verbosity=verbosity)

        target_position = self.get_position(verbosity=0) + move_amount

        return self.move_absolute(target_position, verbosity=verbosity)

    def _get_position(self, verbosity=3):
        """Return the current position of this axis (in its coordinate system).
        By default, this also prints out the current position."""

        if self.motor is not None:
            base_position = self.motor.position

        else:
            verbosity_c = verbosity if verbosity >= 4 else 0
            base_position = getattr(self.base_stage, self.name + "pos")(verbosity=verbosity_c)

        position = self.base_to_cur(base_position)

        if verbosity >= 2:
            if self.stage:
                stg = self.stage.name
            else:
                stg = "?"

            if verbosity >= 5 and self.motor is not None:
                print("{:s} = {:.3f} {:s}".format(self.motor.name, base_position, self.get_units()))

            print(
                "{:s}.{:s} = {:.3f} {:s} (origin = {:.3f})".format(
                    stg, self.name, position, self.get_units(), self.get_origin()
                )
            )

        return position

    def _move_absolute(self, position=None, wait=True, verbosity=3):
        """Move axis to the specified absolute position. The position is given
        in terms of this axis' current coordinate system. The "defer" argument
        can be used to defer motions until "move" is called."""

        if position is None:
            # If called without any argument, just print the current position
            return self.get_position(verbosity=verbosity)

        # Account for coordinate transformation
        base_position = self.cur_to_base(position)

        if self.is_enabled():
            if self.motor:
                # mov( self.motor, base_position )
                self.motor.user_setpoint.value = base_position

            else:
                # Call self.base_stage.xabs(base_position)
                getattr(self.base_stage, self.name + "abs")(base_position, verbosity=0)

            if self.stage:
                stg = self.stage.name
            else:
                stg = "?"

            if verbosity >= 2:
                # Show a realtime output of position
                start_time = time.time()
                current_position = self._get_position(verbosity=0)
                # while abs(current_position-position)>self._move_settle_tolerance and (time.time()-start_time)<self._move_settle_max_time:
                #     current_position = self.get_position(verbosity=0)
                #     print( '{:s}.{:s} = {:5.3f} {:s}      \r'.format(stg, self.name, current_position, self.get_units()), end='')
                #     time.sleep(self._move_settle_period)

            # if verbosity>=1:
            # current_position = self.get_position(verbosity=0)
            # print( '{:s}.{:s} = {:5.3f} {:s}        '.format(stg, self.name, current_position, self.get_units()))

        elif verbosity >= 1:
            print("Axis %s disabled (stage %s)." % (self.name, self.stage.name))

    def _move_relative(self, move_amount=None, verbosity=3):
        """Move axis relative to the current position."""

        if move_amount is None:
            # If called without any argument, just print the current position
            return self.get_position(verbosity=verbosity)

        target_position = self.get_position(verbosity=0) + move_amount

        return self._move_absolute(target_position, verbosity=verbosity)

    def goto_origin(self):
        """Move axis to the currently-defined origin (zero-point)."""

        self.move_absolute(0)

    def set_origin(self, origin=None):
        """Sets the origin (zero-point) for this axis. If no origin is supplied,
        the current position is redefined as zero. Alternatively, you can supply
        a position (in the current coordinate system of the axis) that should
        henceforth be considered zero."""

        if origin is None:
            # Use current position
            if self.motor is not None:
                self.origin = self.motor.position

            else:
                if self.base_stage is None:
                    print(
                        "Error: %s %s has 'base_stage' and 'motor' set to 'None'."
                        % (self.__class__.__name__, self.name)
                    )
                else:
                    self.origin = getattr(self.base_stage, self.name + "pos")(verbosity=0)

        else:
            # Use supplied value (in the current coordinate system)
            base_position = self.cur_to_base(origin)
            self.origin = base_position

    def set_current_position(self, new_position):
        """Redefines the position value of the current position."""
        current_position = self.get_position(verbosity=0)
        self.origin = self.get_origin() + (current_position - new_position) * self.scaling

    def search(
        self,
        step_size=1.0,
        min_step=0.05,
        intensity=None,
        target=0.5,
        detector=None,
        detector_suffix=None,
        polarity=+1,
        verbosity=3,
    ):
        """Moves this axis, searching for a target value.

        Parameters
        ----------
        step_size : float
            The initial step size when moving the axis
        min_step : float
            The final (minimum) step size to try
        intensity : float
            The expected full-beam intensity readout
        target : 0.0 to 1.0
            The target ratio of full-beam intensity; 0.5 searches for half-max.
            The target can also be 'max' to find a local maximum.
        detector, detector_suffix
            The beamline detector (and suffix, such as '_stats4_total') to trigger to measure intensity
        polarity : +1 or -1
            Positive motion assumes, e.g. a step-height 'up' (as the axis goes more positive)
        """

        if not get_beamline().beam.is_on():
            print("WARNING: Experimental shutter is not open.")

        if intensity is None:
            intensity = RE.md["beam_intensity_expected"]

        if detector is None:
            # detector = gs.DETS[0]
            detector = get_beamline().detector[0]
        if detector_suffix is None:
            # value_name = gs.TABLE_COLS[0]
            value_name = get_beamline().TABLE_COLS[0]
        else:
            value_name = detector.name + detector_suffix

        bec.disable_table()

        # Check current value
        RE(count([detector]))
        value = detector.read()[value_name]["value"]

        if target == "max":
            if verbosity >= 5:
                print("Performing search on axis '{}' target is 'max'".format(self.name))

            max_value = value
            max_position = self.get_position(verbosity=0)

            direction = +1 * polarity

            while step_size >= min_step:
                if verbosity >= 4:
                    print("        move {} by {} × {}".format(self.name, direction, step_size))
                self.move_relative(move_amount=direction * step_size, verbosity=verbosity - 2)

                prev_value = value
                RE(count([detector]))

                value = detector.read()[value_name]["value"]
                if verbosity >= 3:
                    print(
                        "      {} = {:.3f} {}; value : {}".format(
                            self.name, self.get_position(verbosity=0), self.units, value
                        )
                    )

                if value > max_value:
                    max_value = value
                    max_position = self.get_position(verbosity=0)

                if value > prev_value:
                    # Keep going in this direction...
                    pass
                else:
                    # Switch directions!
                    direction *= -1
                    step_size *= 0.5

        elif target == "min":
            if verbosity >= 5:
                print("Performing search on axis '{}' target is 'min'".format(self.name))

            direction = +1 * polarity

            while step_size >= min_step:
                if verbosity >= 4:
                    print("        move {} by {} × {}".format(self.name, direction, step_size))
                self.move_relative(move_amount=direction * step_size, verbosity=verbosity - 2)

                prev_value = value
                RE(count([detector]))
                value = detector.read()[value_name]["value"]
                if verbosity >= 3:
                    print(
                        "      {} = {:.3f} {}; value : {}".format(
                            self.name, self.get_position(verbosity=0), self.units, value
                        )
                    )

                if value < prev_value:
                    # Keep going in this direction...
                    pass
                else:
                    # Switch directions!
                    direction *= -1
                    step_size *= 0.5

        else:
            target_rel = target
            target = target_rel * intensity

            if verbosity >= 5:
                print(
                    "Performing search on axis '{}' target {} × {} = {}".format(
                        self.name, target_rel, intensity, target
                    )
                )
            if verbosity >= 4:
                print("      value : {} ({:.1f}%)".format(value, 100.0 * value / intensity))

            # Determine initial motion direction
            if value > target:
                direction = -1 * polarity
            else:
                direction = +1 * polarity

            while step_size >= min_step:
                if verbosity >= 4:
                    print("        move {} by {} × {}".format(self.name, direction, step_size))
                self.move_relative(move_amount=direction * step_size, verbosity=verbosity - 2)

                RE(count([detector]))
                value = detector.read()[value_name]["value"]
                if verbosity >= 3:
                    print(
                        "      {} = {:.3f} {}; value : {} ({:.1f}%)".format(
                            self.name,
                            self.get_position(verbosity=0),
                            self.units,
                            value,
                            100.0 * value / intensity,
                        )
                    )

                # Determine direction
                if value > target:
                    new_direction = -1.0 * polarity
                else:
                    new_direction = +1.0 * polarity

                if abs(direction - new_direction) < 1e-4:
                    # Same direction as we've been going...
                    # ...keep moving this way
                    pass
                else:
                    # Switch directions!
                    direction *= -1
                    step_size *= 0.5

        bec.enable_table()

    def search_plan(
        self,
        motor=smy,
        step_size=1.0,
        min_step=0.05,
        intensity=None,
        target=0.5,
        detector=None,
        detector_suffix=None,
        polarity=+1,
        fastsearch=False,
        verbosity=3,
    ):
        """Moves this axis, searching for a target value.

        Parameters
        ----------
        step_size : float
            The initial step size when moving the axis
        min_step : float
            The final (minimum) step size to try
        intensity : float
            The expected full-beam intensity readout
        target : 0.0 to 1.0
            The target ratio of full-beam intensity; 0.5 searches for half-max.
            The target can also be 'max' to find a local maximum.
        detector, detector_suffix
            The beamline detector (and suffix, such as '_stats4_total') to trigger to measure intensity
        polarity : +1 or -1
            Positive motion assumes, e.g. a step-height 'up' (as the axis goes more positive)
        """

        @stage_decorator([detector])
        def inner_search():
            if not get_beamline().beam.is_on():
                print("WARNING: Experimental shutter is not open.")

            if intensity is None:
                intensity = RE.md["beam_intensity_expected"]

            if detector is None:
                # detector = gs.DETS[0]
                detector = get_beamline().detector[0]
            if detector_suffix is None:
                # value_name = gs.TABLE_COLS[0]
                value_name = get_beamline().TABLE_COLS[0]
            else:
                value_name = detector.name + detector_suffix

            bec.disable_table()

            # Check current value
            yield from bps.trigger_and_read([detector])
            # RE(count([detector]))
            value = detector.read()[value_name]["value"]

            if fastsearch == True:
                intenisty_threshold = 10
                if (
                    abs(detector.stats2.max_xy.get().y - detector.stats2.centroid.get().y) < 20
                    and detector.stats2.max_value.get() > intenisty_threshold
                ):
                    # continue the fast alignment
                    print("The reflective beam is found! Continue the fast alignment")
                    return

            if target == "max":
                if verbosity >= 5:
                    print("Performing search on axis '{}' target is 'max'".format(self.name))

                max_value = value
                max_position = self.get_position(verbosity=0)

                direction = +1 * polarity

                while step_size >= min_step:
                    if verbosity >= 4:
                        print("        move {} by {} × {}".format(self.name, direction, step_size))

                    pos = yield from bps.rd(motor)
                    yield from bps.mv(motor, pos + direction * step_size)
                    # self.move_relative(move_amount=direction*step_size, verbosity=verbosity-2)

                    prev_value = value
                    yield from bps.trigger_and_read([detector])
                    # RE(count([detector]))

                    value = detector.read()[value_name]["value"]
                    if verbosity >= 3:
                        print(
                            "      {} = {:.3f} {}; value : {}".format(
                                self.name,
                                self.get_position(verbosity=0),
                                self.units,
                                value,
                            )
                        )

                    if value > max_value:
                        max_value = value
                        # max_position = self.get_position(verbosity=0)

                    if value > prev_value:
                        # Keep going in this direction...
                        pass
                    else:
                        # Switch directions!
                        direction *= -1
                        step_size *= 0.5

            elif target == "min":
                if verbosity >= 5:
                    print("Performing search on axis '{}' target is 'min'".format(self.name))

                direction = +1 * polarity

                while step_size >= min_step:
                    if verbosity >= 4:
                        print("        move {} by {} × {}".format(self.name, direction, step_size))

                    pos = yield from bps.rd(motor)
                    yield from bps.mv(motor, pos + direction * step_size)
                    # self.move_relative(move_amount=direction*step_size, verbosity=verbosity-2)

                    prev_value = value
                    RE(count([detector]))
                    value = detector.read()[value_name]["value"]
                    if verbosity >= 3:
                        print(
                            "      {} = {:.3f} {}; value : {}".format(
                                self.name,
                                self.get_position(verbosity=0),
                                self.units,
                                value,
                            )
                        )

                    if value < prev_value:
                        # Keep going in this direction...
                        pass
                    else:
                        # Switch directions!
                        direction *= -1
                        step_size *= 0.5

            else:
                target_rel = target
                target = target_rel * intensity

                if verbosity >= 5:
                    print(
                        "Performing search on axis '{}' target {} × {} = {}".format(
                            self.name, target_rel, intensity, target
                        )
                    )
                if verbosity >= 4:
                    print("      value : {} ({:.1f}%)".format(value, 100.0 * value / intensity))

                # Determine initial motion direction
                if value > target:
                    direction = -1 * polarity
                else:
                    direction = +1 * polarity

                while step_size >= min_step:
                    if verbosity >= 4:
                        print("        move {} by {} × {}".format(self.name, direction, step_size))

                    pos = yield from bps.rd(motor)
                    yield from bps.mv(motor, pos + direction * step_size)
                    # self.move_relative(move_amount=direction*step_size, verbosity=verbosity-2)

                    yield from bps.trigger_and_read([detector])
                    # RE(count([detector]))
                    value = detector.read()[value_name]["value"]
                    if verbosity >= 3:
                        print(
                            "      {} = {:.3f} {}; value : {} ({:.1f}%)".format(
                                self.name,
                                self.get_position(verbosity=0),
                                self.units,
                                value,
                                100.0 * value / intensity,
                            )
                        )

                    # Determine direction
                    if value > target:
                        new_direction = -1.0 * polarity
                    else:
                        new_direction = +1.0 * polarity

                    if abs(direction - new_direction) < 1e-4:
                        # Same direction as we've been going...
                        # ...keep moving this way
                        pass
                    else:
                        # Switch directions!
                        direction *= -1
                        step_size *= 0.5

            bec.enable_table()

        yield from inner_search()

    def _search(
        self,
        step_size=1.0,
        min_step=0.05,
        intensity=None,
        maxInt=40000,
        target=0.5,
        detector=None,
        detector_suffix=None,
        polarity=+1,
        verbosity=3,
    ):
        """Moves this axis, searching for a target value.

        Parameters
        ----------
        step_size : float
            The initial step size when moving the axis
        min_step : float
            The final (minimum) step size to try
        intensity : float
            The expected full-beam intensity readout
        target : 0.0 to 1.0
            The target ratio of full-beam intensity; 0.5 searches for half-max.
            The target can also be 'max' to find a local maximum.
        detector, detector_suffix
            The beamline detector (and suffix, such as '_stats4_total') to trigger to measure intensity
        polarity : +1 or -1
            Positive motion assumes, e.g. a step-height 'up' (as the axis goes more positive)
        """

        if not get_beamline().beam.is_on():
            print("WARNING: Experimental shutter is not open.")

        if intensity is None:
            intensity = RE.md["beam_intensity_expected"]

        if detector is None:
            # detector = gs.DETS[0]
            detector = get_beamline().detector[0]
        if detector_suffix is None:
            # value_name = gs.TABLE_COLS[0]
            value_name = get_beamline().TABLE_COLS[0]
        else:
            value_name = detector.name + detector_suffix

        bec.disable_table()

        # Check current value
        RE(count([detector]))
        value = detector.read()[value_name]["value"]

        if target == "max":
            if verbosity >= 5:
                print("Performing search on axis '{}' target is 'max'".format(self.name))

            max_value = value
            max_position = self.get_position(verbosity=0)

            direction = +1 * polarity

            while step_size >= min_step:
                if verbosity >= 4:
                    print("        move {} by {} × {}".format(self.name, direction, step_size))
                self.move_relative(move_amount=direction * step_size, verbosity=verbosity - 2)

                prev_value = value
                RE(count([detector]))

                value = detector.read()[value_name]["value"]

                if abs(maxInt - value) / maxInt < 0.1:
                    self.align_success = False
                else:
                    self.align_success = True

                if verbosity >= 3:
                    print(
                        "      {} = {:.3f} {}; value : {}".format(
                            self.name, self.get_position(verbosity=0), self.units, value
                        )
                    )

                if value > max_value:
                    max_value = value
                    max_position = self.get_position(verbosity=0)

                if value > prev_value:
                    # Keep going in this direction...
                    pass
                else:
                    # Switch directions!
                    direction *= -1
                    step_size *= 0.5

        elif target == "min":
            if verbosity >= 5:
                print("Performing search on axis '{}' target is 'min'".format(self.name))

            direction = +1 * polarity

            while step_size >= min_step:
                if verbosity >= 4:
                    print("        move {} by {} × {}".format(self.name, direction, step_size))
                self.move_relative(move_amount=direction * step_size, verbosity=verbosity - 2)

                prev_value = value
                RE(count([detector]))
                value = detector.read()[value_name]["value"]
                if verbosity >= 3:
                    print(
                        "      {} = {:.3f} {}; value : {}".format(
                            self.name, self.get_position(verbosity=0), self.units, value
                        )
                    )

                if value < prev_value:
                    # Keep going in this direction...
                    pass
                else:
                    # Switch directions!
                    direction *= -1
                    step_size *= 0.5

        else:
            target_rel = target
            target = target_rel * intensity

            if verbosity >= 5:
                print(
                    "Performing search on axis '{}' target {} × {} = {}".format(
                        self.name, target_rel, intensity, target
                    )
                )
            if verbosity >= 4:
                print("      value : {} ({:.1f}%)".format(value, 100.0 * value / intensity))

            # Determine initial motion direction
            if value > target:
                direction = -1 * polarity
            else:
                direction = +1 * polarity

            while step_size >= min_step:
                if verbosity >= 4:
                    print("        move {} by {} × {}".format(self.name, direction, step_size))
                self.move_relative(move_amount=direction * step_size, verbosity=verbosity - 2)

                RE(count([detector]))
                value = detector.read()[value_name]["value"]
                if verbosity >= 3:
                    print(
                        "      {} = {:.3f} {}; value : {} ({:.1f}%)".format(
                            self.name,
                            self.get_position(verbosity=0),
                            self.units,
                            value,
                            100.0 * value / intensity,
                        )
                    )

                # Determine direction
                if value > target:
                    new_direction = -1.0 * polarity
                else:
                    new_direction = +1.0 * polarity

                if abs(direction - new_direction) < 1e-4:
                    # Same direction as we've been going...
                    # ...keep moving this way
                    pass
                else:
                    # Switch directions!
                    direction *= -1
                    step_size *= 0.5

        bec.enable_table()

    def scan(self):
        print("todo")

    def center(self):
        print("todo")

    def mark(self, label, position=None, verbosity=3):
        """Set a mark for this axis. (By default, the current position is
        used.)"""

        if position is None:
            position = self.get_position(verbosity=0)

        axes_positions = {self.name: position}
        self.stage.mark(label, **axes_positions)

    # Book-keeping
    ########################################

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def is_enabled(self):
        return self.enabled and self.stage.is_enabled()

    def get_origin(self):
        return self.origin

    def get_units(self):
        if self.units is not None:
            return self.units

        else:
            return getattr(self.base_stage, self.name + "units")()

    def get_hint(self, verbosity=3):
        """Return (and print) the "motion hint" associated with this axis. This
        hint gives information about the expected directionality of the motion."""

        if self.hint is not None:
            s = "%s\n%s" % (self.hint, self.stage._hint_replacements(self.hint))
            if verbosity >= 2:
                print(s)
            return s

        else:
            return getattr(self.base_stage, self.name + "hint")(verbosity=verbosity)

    def get_info(self, verbosity=3):
        """Returns information about this axis."""

        self.get_position(verbosity=verbosity)
        self.get_hint(verbosity=verbosity)

    def check_base(self):
        if self.base_stage is None:
            print("Error: %s %s has 'base_stage' set to 'None'." % (self.__class__.__name__, self.name))


class Sample_Generic(CoordinateSystem):
    """
    The Sample() classes are used to define a single, individual sample. Each
    sample is created with a particular name, which is recorded during measurements.
    Logging of comments also includes the sample name. Different Sample() classes
    can define different defaults for alignment, measurement, etc.
    """

    # Core methods
    ########################################
    def __init__(self, name, base=None, **md):
        """Create a new Sample object.

        Parameters
        ----------
        name : str
            Name for this sample.
        base : Stage
            The stage/holder on which this sample sits.
        """

        if base is None:
            base = get_default_stage()
            # print("Note: No base/stage/holder specified for sample '{:s}'. Assuming '{:s}' (class {:s})".format(name, base.name, base.__class__.__name__))

        super().__init__(name=name, base=base)

        self.name = name

        self.md = {
            "exposure_time": 1.0,
            "measurement_ID": 1,
        }
        self.md.update(md)

        self.naming_scheme = ["name", "extra", "exposure_time", "id"]
        self.naming_delimeter = "_"

        # TODO
        # if base is not None:
        # base.addSample(self)

        self.reset_clock()

    def _set_axes_definitions(self):
        """Internal function which defines the axes for this stage. This is kept
        as a separate function so that it can be over-ridden easily."""

        # The _axes_definitions array holds a list of dicts, each defining an axis
        self._axes_definitions = [
            {
                "name": "x",
                "motor": None,
                "enabled": True,
                "scaling": +1.0,
                "units": None,
                "hint": None,
            },
            {
                "name": "y",
                "motor": None,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": None,
            },
            # {'name': 'z',
            #'motor': None,
            #'enabled': False,
            #'scaling': +1.0,
            #'units': 'mm',
            #'hint': None,
            # },
            {
                "name": "th",
                "motor": None,
                "enabled": True,
                "scaling": +1.0,
                "units": "deg",
                "hint": None,
            },
            # {'name': 'chi',
            #'motor': None,
            #'enabled': True,
            #'scaling': +1.0,
            #'units': 'deg',
            #'hint': None,
            # },
            # {'name': 'phi',
            # 'motor': srot,
            # 'enabled': True,
            # 'scaling': +1.0,
            # 'units': 'deg',
            # 'hint': None,
            # },
            # {'name': 'yy',
            #'motor': None,
            #'enabled': True,
            #'scaling': +1.0,
            #'units': 'mm',
            #'hint': None,
            # },
        ]

    # Metadata methods
    ########################################
    # These involve setting or getting values associated with this sample.

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

    def get_attribute(self, attribute):
        """Return the value of the requested md."""

        if attribute in self._axes:
            return self._axes[attribute].get_position(verbosity=0)

        if attribute == "name":
            return self.name

        if attribute == "clock":
            return self.clock()

        if attribute == "temperature":
            return self.temperature(verbosity=0)
        if attribute == "temperature_A":
            return self.temperature(temperature_probe="A", verbosity=0)
        if attribute == "temperature_B":
            return self.temperature(temperature_probe="B", verbosity=0)
        if attribute == "temperature_C":
            return self.temperature(temperature_probe="C", verbosity=0)
        if attribute == "temperature_D":
            return self.temperature(temperature_probe="D", verbosity=0)
        if attribute == "temperature_E":
            return self.temperature(temperature_probe="E", verbosity=0)
        if attribute == "humidity":
            return self.humidity(verbosity=0, AI_chan=7)

        if attribute == "WAXSy":
            return WAXSy.position
        if attribute == "WAXSx":
            return WAXSx.position

        if attribute == "SAXSy":
            return SAXSy.position
        if attribute == "SAXSx":
            return SAXSx.position
        # if attribute=='temperature_Linkam':
        #     # return caget('XF:11BM-ES:{LINKAM}:TEMP')
        #     return LThermal.temperature()
        if attribute in self.md:
            return self.md[attribute]
        if attribute == "energy":
            return "{}kev".format(np.round(beam.energy(verbosity=0), 2))
        if attribute == "dry":
            return "dry{}".format(readDryFlow())
        if attribute == "wet":
            return "wet{}".format(readWetFlow())
        if attribute == "flow":
            if MFC.mode("A1") == 0:  # open
                return "flowMAX"
            if MFC.mode("A1") == 1:  # close
                return "flowOFF"
            if MFC.mode("A1") == 2:
                return "flow{}".format(MFC.flow("A1"))
        if attribute == "voltage":
            return "{0:.3f}V".format(ioL.read(AI[1]))
        for ii in range(9):
            if attribute == "voltage{}".format(ii):
                return "{0:.3f}V".format(ioL.read(AI[ii]))
        if attribute == "applied_v":
            return "{0:.3f}V".format(ioL.read(AO[5]))
        if attribute == "TC":
            temp = 1.0175 * ioL.read(TC[1]) - 4.1286  # temporary for B. Wild run 10/16/19
            return "{0:.3f}C".format(temp)

        replacements = {
            "id": "measurement_ID",
            "ID": "measurement_ID",
            "extra": "savename_extra",
        }

        if attribute == "pos1":
            return "pos1"
        if attribute == "pos2":
            return "pos2"
        if attribute == "pos3":
            return "pos3"
        if attribute == "pos4":
            return "pos4"

        if attribute in replacements:
            return self.md[replacements[attribute]]

        return None

    def set_attribute(self, attribute, value):
        """Arbitrary attributes can be set and retrieved. You can use this to
        store additional meta-data about the sample.

        WARNING: Currently this meta-data is not saved anywhere. You can opt
        to store the information in the sample filename (using "naming").
        """

        self.md[attribute] = value

    def set_md(self, **md):
        self.md.update(md)

    def get_md(self, prefix="sample_", include_marks=True, **md):
        """Returns a dictionary of the current metadata.
        The 'prefix' argument is prepended to all the md keys, which allows the
        metadata to be grouped with other metadata in a clear way. (Especially,
        to make it explicit that this metadata came from the sample.)"""

        # Update internal md
        # self.md['key'] = value

        md_return = self.md.copy()
        md_return["name"] = self.name

        if include_marks:
            for label, positions in self._marks.items():
                md_return["mark_" + label] = positions

        # Add md that varies over time
        md_return["clock"] = self.clock()
        md_return["temperature"] = self.temperature(temperature_probe="A", verbosity=0)
        md_return["temperature_A"] = self.temperature(temperature_probe="A", verbosity=0)
        md_return["temperature_B"] = self.temperature(temperature_probe="B", verbosity=0)
        md_return["temperature_C"] = self.temperature(temperature_probe="C", verbosity=0)
        md_return["temperature_D"] = self.temperature(temperature_probe="D", verbosity=0)
        # md_return['temperature_E'] = self.temperature(temperature_probe='E', verbosity=0)
        # md_return['humidity'] = self.humidity(verbosity=0)

        for axis_name, axis in self._axes.items():
            md_return[axis_name] = axis.get_position(verbosity=0)
            md_return["motor_" + axis_name] = axis.get_motor_position(verbosity=0)

        md_return["savename"] = self.get_savename()  # This should be over-ridden by 'measure'

        # TODO: save the attributes into metadata --061921 RL
        """
        for attribute in self.naming_scheme:
            md_return[attribute] = self.get_attribute(attribute)
            # self.set_attribute(attribute, self.get_attribute(attribute))
        """

        # Include the user-specified metadata
        md_return.update(md)

        # Add an optional prefix
        if prefix is not None:
            md_return = {"{:s}{:s}".format(prefix, key): value for key, value in md_return.items()}

        return md_return

    # Naming scheme methods
    ########################################
    # These allow the user to control how data is named.

    def naming(self, scheme=["name", "extra", "exposure_time", "id"], delimeter="_"):
        """This method allows one to define the naming convention that will be
        used when storing data for this sample. The "scheme" variable is an array
        that lists the various elements one wants to store in the filename.

        Each entry in "scheme" is a string referring to a particular element/
        value. For instance, motor names can be stored ("x", "y", etc.), the
        measurement time can be stored, etc."""

        self.naming_scheme = scheme
        self.naming_delimeter = delimeter

    def get_naming_string(self, attribute):
        # Handle special cases of formatting the text

        if attribute in self._axes:
            return "{:s}{:.3f}".format(attribute, self._axes[attribute].get_position(verbosity=0))

        if attribute == "clock":
            return "{:.1f}s".format(self.get_attribute(attribute))

        if attribute == "exposure_time":
            return "{:.2f}s".format(self.get_attribute(attribute))

        if attribute == "temperature":
            return "T{:.3f}C".format(self.get_attribute(attribute))
        if attribute == "temperature_A":
            return "T{:.3f}C".format(self.get_attribute(attribute))
        if attribute == "temperature_B":
            return "T{:.3f}C".format(self.get_attribute(attribute))
        if attribute == "temperature_C":
            return "T{:.3f}C".format(self.get_attribute(attribute))
        if attribute == "temperature_D":
            return "T{:.3f}C".format(self.get_attribute(attribute))
        if attribute == "temperature_E":
            return "T{:.3f}C".format(self.get_attribute(attribute))
        if attribute == "humidity":
            return "RH{:.3f}".format(self.get_attribute(attribute))
        if attribute == "trigger_time":
            return "{:.1f}s".format(self.get_attribute(attribute))

        if attribute == "WAXSy":
            return "WAXSy{}".format(self.get_attribute(attribute))
        if attribute == "WAXSx":
            return "WAXSx{}".format(self.get_attribute(attribute))

        if attribute == "SAXSy":
            return "SAXSy{}".format(self.get_attribute(attribute))
        if attribute == "SAXSx":
            return "SAXSx{}".format(self.get_attribute(attribute))
        # if attribute=='temperature_Linkam':
        #     return 'Linkam{:.1f}C'.format(self.get_attribute(attribute))
        if attribute == "extra":
            # Note: Don't eliminate this check; it will not be properly handled
            # by the generic call below. When 'extra' is None, we should
            # return None, so that it gets skipped entirely.
            return self.get_attribute("savename_extra")

        if attribute == "spot_number":
            return "spot{:d}".format(self.get_attribute(attribute))

        # Generically: lookup the attribute and convert to string

        att = self.get_attribute(attribute)
        if att is None:
            # If the attribute is not found, simply return the text.
            # This allows the user to insert arbitrary text info into the
            # naming scheme.
            return attribute

        else:
            # TODO: save the attributes into metadata --061921 RL
            """
            for attribute in self.get_attribute(attribute):
                self.set_attribute(attribute, self.get_attribute(attribute))
            """

            return str(att)

    def get_savename(self, savename_extra=None):
        """Return the filename that will be used to store data for the upcoming
        measurement. The method "naming" lets one control what gets stored in
        the filename."""

        if savename_extra is not None:
            self.set_attribute("savename_extra", savename_extra)

        attribute_strings = []
        for attribute in self.naming_scheme:
            s = self.get_naming_string(attribute)
            if s is not None:
                attribute_strings.append(s)

        self.set_attribute("savename_extra", None)

        savename = self.naming_delimeter.join(attribute_strings)

        # Avoid 'dangerous' characters
        savename = savename.replace(" ", "_")
        # savename = savename.replace('.', 'p')
        savename = savename.replace("/", "-slash-")

        return savename

    # Logging methods
    ########################################

    def comment(self, text, logbooks=None, tags=None, append_md=True, **md):
        """Add a comment related to this sample."""

        text += "\n\n[comment for sample: {} ({})].".format(self.name, self.__class__.__name__)

        if append_md:
            md_current = {k: v for k, v in RE.md.items()}  # Global md
            md_current.update(get_beamline().get_md())  # Beamline md

            # Sample md
            md_current.update(self.get_md())

            # Specified md
            md_current.update(md)

            text += "\n\n\nMetadata\n----------------------------------------"
            for key, value in sorted(md_current.items()):
                text += "\n{}: {}".format(key, value)

        logbook.log(text, logbooks=logbooks, tags=tags)

    def log(self, text, logbooks=None, tags=None, append_md=True, **md):
        if append_md:
            text += "\n\n\nMetadata\n----------------------------------------"
            for key, value in sorted(md.items()):
                text += "\n{}: {}".format(key, value)

        logbook.log(text, logbooks=logbooks, tags=tags)

    # Control methods
    ########################################
    def setTemperature(self, temperature, verbosity=3):
        return self.base_stage.setTemperature(temperature, verbosity=verbosity)

    def temperature(self, verbosity=3):
        return self.base_stage.temperature(verbosity=verbosity)

    # Measurement methods
    ########################################

    def get_measurement_md(self, prefix=None, **md):
        # md_current = {}
        md_current = {k: v for k, v in RE.md.items()}  # Global md

        # md_current['detector_sequence_ID'] = caget('XF:11BMB-ES{Det:SAXS}:cam1:FileNumber_RBV')
        # md_current['detector_sequence_ID'] = caget('XF:11BMB-ES{}:cam1:FileNumber_RBV'.format(pilatus_Epicsname))
        if get_beamline().detector[0].name is "pilatus300":
            md_current["detector_sequence_ID"] = caget("XF:11BMB-ES{Det:SAXS}:cam1:FileNumber_RBV")
        elif get_beamline().detector[0].name is "pilatus2M":
            md_current["detector_sequence_ID"] = caget("XF:11BMB-ES{Det:PIL2M}:cam1:FileNumber_RBV")

        md_current.update(get_beamline().get_md())

        md_current.update(md)

        # Add an optional prefix
        if prefix is not None:
            md_return = {"{:s}{:s}".format(prefix, key): value for key, value in md_return.items()}

        return md_current

    def _expose_manual(self, exposure_time=None, verbosity=3, poling_period=0.1, **md):
        """Internal function that is called to actually trigger a measurement."""

        # TODO: Improve this (switch to Bluesky methods)
        # TODO: Store metadata

        if "measure_type" not in md:
            md["measure_type"] = "expose"
        self.log("{} for {}.".format(md["measure_type"], self.name), **md)

        if exposure_time is not None:
            # Prep detector
            # caput('XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime', exposure_time)
            # caput('XF:11BMB-ES{Det:SAXS}:cam1:AcquirePeriod', exposure_time+0.1)
            # caput('XF:11BMB-ES{}:cam1:AcquireTime'.format(pilatus_Epicsname), exposure_time)
            # caput('XF:11BMB-ES{}:cam1:AcquirePeriod'.format(pilatus_Epicsname), exposure_time+0.1)

            if get_beamline().detector[0].name is "pilatus300":
                caput("XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime", exposure_time)
                caput("XF:11BMB-ES{Det:SAXS}:cam1:AcquirePeriod", exposure_time + 0.1)
            elif get_beamline().detector[0].name is "pilatus2M":
                caput("XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime", exposure_time)
                caput("XF:11BMB-ES{Det:PIL2M}:cam1:AcquirePeriod", exposure_time + 0.1)

        get_beamline().beam.on()

        # Trigger acquisition manually
        caput("XF:11BMB-ES{}:cam1:Acquire".format(pilatus_Epicsname), 1)

        if verbosity >= 2:
            start_time = time.time()
            while caget("XF:11BMB-ES{}:cam1:Acquire".format(pilatus_Epicsname)) == 1 and (
                time.time() - start_time
            ) < (exposure_time + 20):
                percentage = 100 * (time.time() - start_time) / exposure_time
                print(
                    "Exposing {:6.2f} s  ({:3.0f}%)      \r".format((time.time() - start_time), percentage),
                    end="",
                )
                time.sleep(poling_period)
        else:
            time.sleep(exposure_time)

        if verbosity >= 3 and caget("XF:11BMB-ES{}:cam1:Acquire".format(pilatus_Epicsname)) == 1:
            print("Warning: Detector still not done acquiring.")

        get_beamline().beam.off()

    def expose(self, exposure_time=None, extra=None, handlefile=True, verbosity=3, poling_period=0.1, **md):
        """Internal function that is called to actually trigger a measurement."""
        """TODO: **md doesnot work in RE(count). """

        if "measure_type" not in md:
            md["measure_type"] = "expose"
        # self.log('{} for {}.'.format(md['measure_type'], self.name), **md)

        # Set exposure time
        if exposure_time is not None:
            exposure_time = abs(exposure_time)
            # for detector in gs.DETS:
            for detector in get_beamline().detector:
                if (
                    exposure_time != detector.cam.acquire_time.get()
                ):  # caget('XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime'):
                    RE(detector.setExposureTime(exposure_time, verbosity=verbosity))
                # if detector.name is 'pilatus800' and exposure_time != detector.cam.acquire_time.get():  #caget('XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime'):
                # RE(detector.setExposureTime(exposure_time, verbosity=verbosity))
                # if detector.name is 'pilatus300' and exposure_time != detector.cam.acquire_time.get():
                # detector.setExposureTime(exposure_time, verbosity=verbosity)
                ##extra wait time when changing the exposure time.
                ##time.sleep(2)
                #############################################
                ##extra wait time for adjusting pilatus2M
                ##this extra wait time has to be added. Otherwise, the exposure will be skipped when the exposure time is increased
                ##Note by 091918
                #############################################
                # time.sleep(2)
                # elif detector.name is 'PhotonicSciences_CMS':
                # detector.setExposureTime(exposure_time, verbosity=verbosity)

        # Do acquisition
        get_beamline().beam.on()

        md["plan_header_override"] = md["measure_type"]
        start_time = time.time()

        # md_current = self.get_md()
        md["beam_int_bim3"] = beam.bim3.flux(verbosity=0)
        md["beam_int_bim4"] = beam.bim4.flux(verbosity=0)
        md["beam_int_bim5"] = beam.bim5.flux(verbosity=0)
        # md['trigger_time'] = self.clock()
        # md.update(md_current)

        # uids = RE(count(get_beamline().detector, 1), **md)
        uids = RE(count(get_beamline().detector), **md)
        # yield from (count(get_beamline().detector), **md)

        # get_beamline().beam.off()
        # print('shutter is off')

        # Wait for detectors to be ready
        max_exposure_time = 0.1
        for detector in get_beamline().detector:
            if detector.name is "pilatus300":
                current_exposure_time = detector.cam.acquire_time.get()
                max_exposure_time = max(max_exposure_time, current_exposure_time)
            elif detector.name is "pilatus2M":
                current_exposure_time = detector.cam.acquire_time.get()
                max_exposure_time = max(max_exposure_time, current_exposure_time)
            elif detector.name is "pilatus800" or detector.name is "pilatus8002":
                current_exposure_time = detector.cam.acquire_time.get()
                max_exposure_time = max(max_exposure_time, current_exposure_time)

            # if detector.name is 'pilatus300':
            #     current_exposure_time = caget('XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime')
            #     max_exposure_time = max(max_exposure_time, current_exposure_time)
            # elif detector.name is 'pilatus2M':
            #     current_exposure_time = caget('XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime')
            #     max_exposure_time = max(max_exposure_time, current_exposure_time)
            # elif detector.name is 'pilatus800':
            #     current_exposure_time = caget('XF:11BMB-ES{Det:PIL800K}:cam1:AcquireTime')
            #     max_exposure_time = max(max_exposure_time, current_exposure_time)
            # elif detector.name is 'PhotonicSciences_CMS':
            # current_exposure_time = detector.exposure_time
            # max_exposure_time = max(max_exposure_time, current_exposure_time)
            else:
                if verbosity >= 1:
                    print("WARNING: Didn't recognize detector '{}'.".format(detector.name))

        if verbosity >= 2:
            status = 0
            while (status == 0) and (time.time() - start_time) < (max_exposure_time + 20):
                percentage = 100 * (time.time() - start_time) / max_exposure_time
                print(
                    "Exposing {:6.2f} s  ({:3.0f}%)      \r".format((time.time() - start_time), percentage),
                    end="",
                )

                time.sleep(poling_period)

                status = 1
                for detector in get_beamline().detector:
                    if detector.cam.acquire.get() == 1:
                        status *= 0

            # print('counting .... percentage = {}'.format(percentage))

        else:
            time.sleep(max_exposure_time)

        # special solution for 2022_1/TKoga2
        if verbosity >= 5:
            print("verbosity = {}.".format(verbosity))
            pct_threshold = 90
            while percentage < pct_threshold:
                print("sth is wrong .... percentage = {} < {}%".format(percentage, pct_threshold))
                start_time = time.time()
                uids = RE(count(get_beamline().detector), **md)
                # yield from (count(get_beamline().detector), **md)

                # get_beamline().beam.off()
                # print('shutter is off')

                # Wait for detectors to be ready
                max_exposure_time = 0.1
                for detector in get_beamline().detector:
                    if detector.name is "pilatus300":
                        current_exposure_time = detector.cam.acquire_time.get()
                        max_exposure_time = max(max_exposure_time, current_exposure_time)
                    elif detector.name is "pilatus2M":
                        current_exposure_time = detector.cam.acquire_time.get()
                        max_exposure_time = max(max_exposure_time, current_exposure_time)
                    elif detector.name is "pilatus800" or detector.name is "pilatus8002":
                        current_exposure_time = detector.cam.acquire_time.get()
                        max_exposure_time = max(max_exposure_time, current_exposure_time)

                percentage = 100 * (time.time() - start_time) / max_exposure_time
                print("After re-exposing .... percentage = {} ".format(percentage))

                # if detector.name is 'pilatus300':
                #     if caget('XF:11BMB-ES{Det:SAXS}:cam1:Acquire')==1:
                #         status *= 0
                # elif detector.name is 'pilatus2M':
                #     if caget('XF:11BMB-ES{Det:PIL2M}:cam1:Acquire')==1:
                #         status *= 0
                # elif detector.name is 'pilatus800':
                #     if caget('XF:11BMB-ES{Det:PIL800K}:cam1:Acquire')==1:
                #         status *= 0
                # elif detector.name is 'PhotonicSciences_CMS':
                # if not detector.detector_is_ready(verbosity=0):
                # status *= 0

        # if verbosity>=3 and caget('XF:11BMB-ES{Det:PIL800K}:cam1:Acquire')==1:
        #     print('Warning: Detector pilatus300 still not done acquiring.')

        # #if verbosity>=3 and caget('XF:11BMB-ES{Det:SAXS}:cam1:Acquire')==1:
        #     #print('Warning: Detector pilatus300 still not done acquiring.')

        # if verbosity>=3 and caget('XF:11BMB-ES{Det:PIL2M}:cam1:Acquire')==1:
        #     print('Warning: Detector pilatus2M still not done acquiring.')

        get_beamline().beam.off()

        # save the percentage information
        # if verbosity>=5:
        #     folder = '/nsls2/data/cms/legacy/xf11bm/data/2022_1/TKoga2/'
        #     # filename = ''

        #     current_data = {'a_sample': self.name,
        #                     'b_exposure_time': detector.cam.acquire_time.get(),
        #                     'c_exposure_percentage': percentage,
        #                     'd_align_time': md['filename']
        #                     }

        #     temp_data = pds.DataFrame([current_data])

        #     # INT_FILENAME='{}/data/{}.csv'.format(os.path.dirname(__file__) , 'alignment_results.csv')
        #     INT_FILENAME='{}/data/{}.csv'.format(folder , 'exposure_info.csv')

        #     if os.path.isfile(INT_FILENAME):
        #         output_data = pds.read_csv(INT_FILENAME, index_col=0)
        #         output_data = output_data.append(temp_data, ignore_index=True)
        #         output_data.to_csv(INT_FILENAME)
        #     else:
        #         temp_data.to_csv(INT_FILENAME)

        if handlefile == True:
            for detector in get_beamline().detector:
                self.handle_file(detector, extra=extra, verbosity=verbosity, **md)
                # self.handle_file(detector, extra=extra, verbosity=verbosity)

    def _expose_test(self, exposure_time=None, extra=None, handlefile=True, verbosity=3, poling_period=0.1, **md):
        """Internal function that is called to actually trigger a measurement."""
        """TODO: **md doesnot work in RE(count). """

        if "measure_type" not in md:
            md["measure_type"] = "expose"
        # self.log('{} for {}.'.format(md['measure_type'], self.name), **md)

        # Set exposure time
        start_time = time.time()
        print("1", time.time() - start_time)
        if exposure_time is not None:
            exposure_time = abs(exposure_time)
            # for detector in gs.DETS:
            for detector in get_beamline().detector:
                if (
                    exposure_time != detector.cam.acquire_time.get()
                ):  # caget('XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime'):
                    RE(detector.setExposureTime(exposure_time, verbosity=verbosity))
                # if detector.name is 'pilatus800' and exposure_time != detector.cam.acquire_time.get():  #caget('XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime'):
                # RE(detector.setExposureTime(exposure_time, verbosity=verbosity))
                # if detector.name is 'pilatus300' and exposure_time != detector.cam.acquire_time.get():
                # detector.setExposureTime(exposure_time, verbosity=verbosity)
                ##extra wait time when changing the exposure time.
                ##time.sleep(2)
                #############################################
                ##extra wait time for adjusting pilatus2M
                ##this extra wait time has to be added. Otherwise, the exposure will be skipped when the exposure time is increased
                ##Note by 091918
                #############################################
                # time.sleep(2)
                # elif detector.name is 'PhotonicSciences_CMS':
                # detector.setExposureTime(exposure_time, verbosity=verbosity)
        print("2", time.time() - start_time)

        # Do acquisition
        get_beamline().beam.on()
        print("beamon", time.time() - start_time)

        md["plan_header_override"] = md["measure_type"]
        # start_time = time.time()

        # md_current = self.get_md()
        md["beam_int_bim3"] = beam.bim3.flux(verbosity=0)
        md["beam_int_bim4"] = beam.bim4.flux(verbosity=0)
        md["beam_int_bim5"] = beam.bim5.flux(verbosity=0)
        # md['trigger_time'] = self.clock()
        # md.update(md_current)

        print("3", time.time() - start_time)
        # uids = RE(count(get_beamline().detector, 1), **md)
        uids = RE(count(get_beamline().detector), **md)
        # yield from (count(get_beamline().detector), **md)
        print("4", time.time() - start_time)

        # get_beamline().beam.off()
        # print('shutter is off')

        # Wait for detectors to be ready
        max_exposure_time = 0.1
        for detector in get_beamline().detector:
            if detector.name is "pilatus300":
                current_exposure_time = caget("XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime")
                max_exposure_time = max(max_exposure_time, current_exposure_time)
            elif detector.name is "pilatus2M":
                current_exposure_time = caget("XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime")
                max_exposure_time = max(max_exposure_time, current_exposure_time)
            elif detector.name is "pilatus800":
                current_exposure_time = caget("XF:11BMB-ES{Det:PIL800K}:cam1:AcquireTime")
                max_exposure_time = max(max_exposure_time, current_exposure_time)
            # elif detector.name is 'PhotonicSciences_CMS':
            # current_exposure_time = detector.exposure_time
            # max_exposure_time = max(max_exposure_time, current_exposure_time)
            else:
                if verbosity >= 1:
                    print("WARNING: Didn't recognize detector '{}'.".format(detector.name))

        print("5", time.time() - start_time)
        if verbosity >= 2:
            status = 0
            while (status == 0) and (time.time() - start_time) < (max_exposure_time + 20):
                percentage = 100 * (time.time() - start_time) / max_exposure_time
                print(
                    "Exposing {:6.2f} s  ({:3.0f}%)      \r".format((time.time() - start_time), percentage),
                    end="",
                )
                time.sleep(poling_period)

                status = 1
                for detector in get_beamline().detector:
                    if detector.name is "pilatus300":
                        if caget("XF:11BMB-ES{Det:SAXS}:cam1:Acquire") == 1:
                            status *= 0
                    elif detector.name is "pilatus2M":
                        if caget("XF:11BMB-ES{Det:PIL2M}:cam1:Acquire") == 1:
                            status *= 0
                    elif detector.name is "pilatus800":
                        if caget("XF:11BMB-ES{Det:PIL800K}:cam1:Acquire") == 1:
                            status *= 0
                    # elif detector.name is 'PhotonicSciences_CMS':
                    # if not detector.detector_is_ready(verbosity=0):
                    # status *= 0
            print("6", time.time() - start_time)

        else:
            time.sleep(max_exposure_time)
        print("7", time.time() - start_time)

        # if verbosity>=3 and caget('XF:11BMB-ES{Det:SAXS}:cam1:Acquire')==1:
        # print('Warning: Detector pilatus300 still not done acquiring.')

        if verbosity >= 3 and caget("XF:11BMB-ES{Det:PIL2M}:cam1:Acquire") == 1:
            print("Warning: Detector pilatus2M still not done acquiring.")

        get_beamline().beam.off()
        print("8", time.time() - start_time)

        # if handlefile == True:
        # for detector in get_beamline().detector:
        # self.handle_file(detector, extra=extra, verbosity=verbosity, **md)
        ##self.handle_file(detector, extra=extra, verbosity=verbosity)

    def handle_file(self, detector, extra=None, verbosity=3, subdirs=True, linksave=True, **md):
        subdir = ""
        if subdirs:
            if detector.name is "pilatus300" or detector.name is "pilatus8002":
                subdir = "/maxs/raw/"
                detname = "maxs"
            elif detector.name is "pilatus2M":
                subdir = "/saxs/raw/"
                detname = "saxs"
            elif detector.name is "pilatus800":
                subdir = "/waxs/raw/"
                detname = "waxs"
            else:
                if verbosity >= 1:
                    print("WARNING: Can't do file handling for detector '{}'.".format(detector.name))
                    return

        filename = detector.tiff.full_file_name.get()  # RL, 20210831
        if os.path.isfile(filename) == False:
            return print("File does not exist")
        # Alternate method to get the last filename
        # filename = '{:s}/{:s}.tiff'.format( detector.tiff.file_path.get(), detector.tiff.file_name.get()  )

        if verbosity >= 3:
            print("  Data saved to: {}".format(filename))

        # if md['measure_type'] is not 'snap':
        if True:
            # self.set_attribute('exposure_time', caget('XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime'))
            self.set_attribute("exposure_time", detector.cam.acquire_time.get())  # RL, 20210831

            # Create symlink
            # link_name = '{}/{}{}'.format(RE.md['experiment_alias_directory'], subdir, md['filename'])
            # savename = md['filename'][:-5]

            # savename = self.get_savename(savename_extra=extra)
            savename = md["filename"]
            # link_name = '{}/{}{}_{:04d}_maxs.tiff'.format(RE.md['experiment_alias_directory'], subdir, savename, RE.md['scan_id']-1)
            link_name = "{}/{}{}_{}.tiff".format(RE.md["experiment_alias_directory"], subdir, savename, detname)

            if os.path.isfile(link_name):
                i = 1
                while os.path.isfile("{}.{:d}".format(link_name, i)):
                    i += 1
                os.rename(link_name, "{}.{:d}".format(link_name, i))
            os.symlink(filename, link_name)

            if verbosity >= 3:
                print("  Data linked as: {}".format(link_name))
                if not os.path.isfile(os.readlink(link_name)): #added by RL, 20231109
                    raise ValueError('NO IMAGE OUTPUT.')

    def _old_handle_file(self, detector, extra=None, verbosity=3, subdirs=True, linksave=True, **md):
        subdir = ""

        if detector.name is "pilatus300" or detector.name is "pilatus8002":
            # chars = caget('XF:11BMB-ES{Det:SAXS}:TIFF1:FullFileName_RBV')
            # filename = ''.join(chr(char) for char in chars)[:-1]
            filename = detector.tiff.full_file_name.get()  # RL, 20210831

            # Alternate method to get the last filename
            # filename = '{:s}/{:s}.tiff'.format( detector.tiff.file_path.get(), detector.tiff.file_name.get()  )

            if verbosity >= 3:
                print("  Data saved to: {}".format(filename))

            if subdirs:
                subdir = "/maxs/raw/"
                # TODO:
                # subdir = '/maxs/raw/'

            # if md['measure_type'] is not 'snap':
            if True:
                # self.set_attribute('exposure_time', caget('XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime'))
                self.set_attribute("exposure_time", detector.cam.acquire_time.get())  # RL, 20210831

                # Create symlink
                # link_name = '{}/{}{}'.format(RE.md['experiment_alias_directory'], subdir, md['filename'])
                # savename = md['filename'][:-5]

                # savename = self.get_savename(savename_extra=extra)
                savename = md["filename"]
                # link_name = '{}/{}{}_{:04d}_maxs.tiff'.format(RE.md['experiment_alias_directory'], subdir, savename, RE.md['scan_id']-1)
                link_name = "{}/{}{}_maxs.tiff".format(RE.md["experiment_alias_directory"], subdir, savename)

                if os.path.isfile(link_name):
                    i = 1
                    while os.path.isfile("{}.{:d}".format(link_name, i)):
                        i += 1
                    os.rename(link_name, "{}.{:d}".format(link_name, i))
                os.symlink(filename, link_name)

                if verbosity >= 3:
                    print("  Data linked as: {}".format(link_name))

        elif detector.name is "pilatus2M":
            foldername = "/nsls2/xf11bm/"

            # chars = caget('XF:11BMB-ES{Det:PIL2M}:TIFF1:FullFileName_RBV')

            # filename = ''.join(chr(char) for char in chars)[:-1]
            # filename = foldername + filename
            filename = detector.tiff.full_file_name.get()  # RL, 20210831

            # chars = caget('XF:11BMB-ES{Det:PIL2M}:TIFF1:FullFileName_RBV')
            # filename = ''.join(chr(char) for char in chars)[:-1]

            # Alternate method to get the last filename
            # filename = '{:s}/{:s}.tiff'.format( detector.tiff.file_path.get(), detector.tiff.file_name.get()  )

            if verbosity >= 3:
                print("  Data saved to: {}".format(filename))

            if subdirs:
                subdir = "/saxs/raw/"
                # TODO:
                # subdir = '/saxs/raw/'

            # if md['measure_type'] is not 'snap':
            if True:
                # self.set_attribute('exposure_time', caget('XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime'))
                self.set_attribute("exposure_time", detector.cam.acquire_time.get())  # RL, 20210831

                # Create symlink
                # link_name = '{}/{}{}'.format(RE.md['experiment_alias_directory'], subdir, md['filename'])
                # savename = md['filename'][:-5]

                # savename = self.get_savename(savename_extra=extra)
                savename = md["filename"]
                link_name = "{}/{}{}_saxs.tiff".format(RE.md["experiment_alias_directory"], subdir, savename)
                # link_name = '{}/{}{}_{:04d}_saxs.tiff'.format(RE.md['experiment_alias_directory'], subdir, savename, RE.md['scan_id']-1)

                if os.path.isfile(link_name):
                    i = 1
                    while os.path.isfile("{}.{:d}".format(link_name, i)):
                        i += 1
                    os.rename(link_name, "{}.{:d}".format(link_name, i))
                os.symlink(filename, link_name)

                if verbosity >= 3:
                    print("  Data linked as: {}".format(link_name))

        elif detector.name is "pilatus800":
            foldername = "/nsls2/xf11bm/"

            # chars = caget('XF:11BMB-ES{Det:PIL800K}:TIFF1:FullFileName_RBV')
            # chars = pilatus800.tiff.full_file_name.get() #RL, 20210831

            # filename = ''.join(chr(char) for char in chars)[:-1]
            # filename = foldername + filename
            filename = detector.tiff.full_file_name.get()  # RL, 20210831
            # Alternate method to get the last filename
            # filename = '{:s}/{:s}.tiff'.format( detector.tiff.file_path.get(), detector.tiff.file_name.get()  )

            if verbosity >= 3:
                print("  Data saved to: {}".format(filename))

            if subdirs:
                subdir = "/waxs/raw/"
                # TODO:
                # subdir = '/waxs/raw/'
            # if md['measure_type'] is not 'snap':
            if True:
                # self.set_attribute('exposure_time', caget('XF:11BMB-ES{Det:PIL800K}:cam1:AcquireTime'))
                self.set_attribute("exposure_time", detector.cam.acquire_time.get())  # RL, 20210831

                # Create symlink
                # link_name = '{}/{}{}'.format(RE.md['experiment_alias_directory'], subdir, md['filename'])
                # savename = md['filename'][:-5]

                # savename = self.get_savename(savename_extra=extra)
                savename = md["filename"]

                link_name = "{}/{}{}_waxs.tiff".format(RE.md["experiment_alias_directory"], subdir, savename)
                # link_name = '{}/{}{}_{:04d}_saxs.tiff'.format(RE.md['experiment_alias_directory'], subdir, savename, RE.md['scan_id']-1)

                if os.path.isfile(link_name):
                    i = 1
                    while os.path.isfile("{}.{:d}".format(link_name, i)):
                        i += 1
                    os.rename(link_name, "{}.{:d}".format(link_name, i))
                os.symlink(filename, link_name)

                if verbosity >= 3:
                    print("  Data linked as: {}".format(link_name))

        # elif detector.name is 'PhotonicSciences_CMS':

        # self.set_attribute('exposure_time', detector.exposure_time)

        # filename = '{:s}/{:s}.tif'.format( detector.file_path, detector.file_name )

        # if subdirs:
        # subdir = '/waxs/'

        ##savename = md['filename'][:-5]
        ##savename = self.get_savename(savename_extra=extra)
        # savename = md['filename']
        ##savename = '{}/{}{}_{:04d}_waxs.tiff'.format(RE.md['experiment_alias_directory'], subdir, savename, RE.md['scan_id']-1)
        # savename = '{}/{}{}_waxs.tiff'.format(RE.md['experiment_alias_directory'], subdir, savename)

        # shutil.copy(filename, savename)
        # if verbosity>=3:
        # print('  Data saved to: {}'.format(savename))

        else:
            if verbosity >= 1:
                print("WARNING: Can't do file handling for detector '{}'.".format(detector.name))
                return

    def snap(self, exposure_time=None, extra=None, measure_type="snap", verbosity=3, **md):
        """Take a quick exposure (without saving data)."""

        self.expose(
            exposure_time=exposure_time,
            extra=extra,
            measure_type=measure_type,
            verbosity=verbosity,
            handlefile=False,
            **md,
        )
        remove_last_Pilatus_series()

    def _measure(
        self,
        exposure_time=None,
        extra=None,
        measure_type="measure",
        verbosity=3,
        tiling=False,
        stitchback=False,
        **md,
    ):
        """Measure data by triggering the area detectors.

        Parameters
        ----------
        exposure_time : float
            How long to collect data
        extra : string, optional
            Extra information about this particular measurement (which is typically
            included in the savename/filename).
        tiling : string
            Controls the detector tiling mode.
              None : regular measurement (single detector position)
              'ygaps' : try to cover the vertical gaps in the Pilatus detector
        """

        if tiling is "xygaps":
            if cms.detector == [pilatus2M]:
                SAXSy_o = SAXSy.user_readback.value
                SAXSx_o = SAXSx.user_readback.value

                extra_current = "pos1" if extra is None else "{}_pos1".format(extra)
                md["detector_position"] = "lower_left"
                self.measure_single(
                    exposure_time=exposure_time,
                    extra=extra_current,
                    measure_type=measure_type,
                    verbosity=verbosity,
                    stitchback=True,
                    **md,
                )

                # extra x movement is needed for pilatus2M.

                SAXSy.move(SAXSy.user_readback.value + 5.16)
                extra_current = "pos2" if extra is None else "{}_pos2".format(extra)
                md["detector_position"] = "upper_left"
                self.measure_single(
                    exposure_time=exposure_time,
                    extra=extra_current,
                    measure_type=measure_type,
                    verbosity=verbosity,
                    stitchback=True,
                    **md,
                )

                # SAXSy.move(SAXSy.user_readback.value + -5.16)
                SAXSx.move(SAXSx.user_readback.value + 5.16)
                extra_current = "pos4" if extra is None else "{}_pos4".format(extra)
                md["detector_position"] = "upper_right"
                self.measure_single(
                    exposure_time=exposure_time,
                    extra=extra_current,
                    measure_type=measure_type,
                    verbosity=verbosity,
                    stitchback=True,
                    **md,
                )

                SAXSy.move(SAXSy.user_readback.value + -5.16)
                extra_current = "pos3" if extra is None else "{}_pos3".format(extra)
                md["detector_position"] = "lower_right"
                self.measure_single(
                    exposure_time=exposure_time,
                    extra=extra_current,
                    measure_type=measure_type,
                    verbosity=verbosity,
                    stitchback=True,
                    **md,
                )

                SAXSx.move(SAXSx.user_readback.value + -5.16)

                if SAXSx.user_readback.value != SAXSx_o:
                    SAXSx.move(SAXSx_o)
                if SAXSy.user_readback.value != SAXSy_o:
                    SAXSy.move(SAXSy_o)
            if cms.detector == [pilatus800]:
                WAXSy_o = WAXSy.user_readback.value
                WAXSx_o = WAXSx.user_readback.value

                extra_current = "pos1" if extra is None else "{}_pos1".format(extra)
                md["detector_position"] = "lower_left"
                self.measure_single(
                    exposure_time=exposure_time,
                    extra=extra_current,
                    measure_type=measure_type,
                    verbosity=verbosity,
                    stitchback=True,
                    **md,
                )

                # extra x movement is needed for pilatus2M.

                WAXSy.move(WAXSy.user_readback.value + 5.16)
                extra_current = "pos2" if extra is None else "{}_pos2".format(extra)
                md["detector_position"] = "upper_left"
                self.measure_single(
                    exposure_time=exposure_time,
                    extra=extra_current,
                    measure_type=measure_type,
                    verbosity=verbosity,
                    stitchback=True,
                    **md,
                )

                WAXSx.move(WAXSx.user_readback.value - 5.16)
                extra_current = "pos4" if extra is None else "{}_pos4".format(extra)
                md["detector_position"] = "upper_right"
                self.measure_single(
                    exposure_time=exposure_time,
                    extra=extra_current,
                    measure_type=measure_type,
                    verbosity=verbosity,
                    stitchback=True,
                    **md,
                )

                WAXSy.move(WAXSy.user_readback.value + -5.16)
                extra_current = "pos3" if extra is None else "{}_pos3".format(extra)
                md["detector_position"] = "lower_right"
                self.measure_single(
                    exposure_time=exposure_time,
                    extra=extra_current,
                    measure_type=measure_type,
                    verbosity=verbosity,
                    stitchback=True,
                    **md,
                )

                WAXSx.move(WAXSx.user_readback.value + 5.16)

                if WAXSx.user_readback.value != WAXSx_o:
                    WAXSx.move(WAXSx_o)
                if WAXSy.user_readback.value != WAXSy_o:
                    WAXSy.move(WAXSy_o)

        elif tiling is "ygaps":
            if cms.detector == [pilatus2M]:
                SAXSy_o = SAXSy.user_readback.value
                SAXSx_o = SAXSx.user_readback.value

                extra_current = "pos1" if extra is None else "{}_pos1".format(extra)
                md["detector_position"] = "lower"
                self.measure_single(
                    exposure_time=exposure_time,
                    extra=extra_current,
                    measure_type=measure_type,
                    verbosity=verbosity,
                    stitchback=True,
                    **md,
                )

                # extra x movement is needed for pilatus2M.

                extra_current = "pos2" if extra is None else "{}_pos2".format(extra)
                md["detector_position"] = "upper"
                SAXSy.move(SAXSy.user_readback.value + 5.16)
                time.sleep(5)
                self.measure_single(
                    exposure_time=exposure_time,
                    extra=extra_current,
                    measure_type=measure_type,
                    verbosity=verbosity,
                    stitchback=True,
                    **md,
                )

                SAXSy.move(SAXSy.user_readback.value - 5.16)
                time.sleep(5)
                if SAXSx.user_readback.value != SAXSx_o:
                    SAXSx.move(SAXSx_o)
                if SAXSy.user_readback.value != SAXSy_o:
                    SAXSy.move(SAXSy_o)

            if cms.detector == [pilatus300]:
                # MAXSy_o = MAXSy.user_readback.value

                extra_current = "pos1" if extra is None else "{}_pos1".format(extra)
                md["detector_position"] = "lower"
                self.measure_single(
                    exposure_time=exposure_time,
                    extra=extra_current,
                    measure_type=measure_type,
                    verbosity=verbosity,
                    stitchback=True,
                    **md,
                )

                MAXSy.move(MAXSy.user_readback.value + 5.16)
                extra_current = "pos2" if extra is None else "{}_pos2".format(extra)
                md["detector_position"] = "upper"
                self.measure_single(
                    exposure_time=exposure_time,
                    extra=extra_current,
                    measure_type=measure_type,
                    verbosity=verbosity,
                    stitchback=True,
                    **md,
                )

                MAXSy.move(MAXSy.user_readback.value + -5.16)

                # if MAXSy.user_readback.value != MAXSy_o:
                # MAXSy.move(MAXSy_o)

            if cms.detector == [pilatus800]:
                WAXSy_o = WAXSy.user_readback.value

                extra_current = "pos1" if extra is None else "{}_pos1".format(extra)
                md["detector_position"] = "lower"
                self.measure_single(
                    exposure_time=exposure_time,
                    extra=extra_current,
                    measure_type=measure_type,
                    verbosity=verbosity,
                    stitchback=True,
                    **md,
                )

                WAXSy.move(WAXSy.user_readback.value + 5.16)
                extra_current = "pos2" if extra is None else "{}_pos2".format(extra)
                md["detector_position"] = "upper"
                self.measure_single(
                    exposure_time=exposure_time,
                    extra=extra_current,
                    measure_type=measure_type,
                    verbosity=verbosity,
                    stitchback=True,
                    **md,
                )

                WAXSy.move(WAXSy.user_readback.value - 5.16)

                # if WAXSy.user_readback.value != WAXSy_o:
                # WAXSy.move(MAXSy_o)
        # if tiling is 'big':
        # TODO: Use multiple images to fill the entire detector motion range

        else:
            # Just do a normal measurement
            self.measure_single(
                exposure_time=exposure_time, extra=extra, measure_type=measure_type, verbosity=verbosity, **md
            )

    def measure(
        self,
        exposure_time=None,
        extra=None,
        measure_type="measure",
        verbosity=3,
        tiling=None,
        stitchback=False,
        **md,
    ):
        """Measure data by triggering the area detectors.

        Parameters
        ----------
        exposure_time : float
            How long to collect data
        extra : string, optional
            Extra information about this particular measurement (which is typically
            included in the savename/filename).
        tiling : string
            Controls the detector tiling mode.
              None : regular measurement (single detector position)
              'ygaps' : try to cover the vertical gaps in the Pilatus detector
        """

        if tiling is "xygaps":
            SAXSy_o = SAXSy.user_readback.value
            SAXSx_o = SAXSx.user_readback.value
            WAXSy_o = WAXSy.user_readback.value
            WAXSx_o = WAXSx.user_readback.value
            # MAXSy_o = MAXSy.user_readback.value

            extra_current = "pos1" if extra is None else "{}_pos1".format(extra)
            md["detector_position"] = "lower_left"
            self.measure_single(
                exposure_time=exposure_time,
                extra=extra_current,
                measure_type=measure_type,
                verbosity=verbosity,
                stitchback=True,
                **md,
            )

            # pos2
            if [pilatus2M] in cms.detector:
                SAXSy.move(SAXSy_o + 5.16)
            if [pilatus800] in cms.detector:
                WAXSy.move(WAXSy_o + 5.16)
            # if [pilatus300] in cms.detector:
            # MAXSy.move(MAXSy_o + 5.16)

            extra_current = "pos2" if extra is None else "{}_pos2".format(extra)
            md["detector_position"] = "upper_left"
            self.measure_single(
                exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
            )

            # pos4
            if pilatus2M in cms.detector:
                SAXSx.move(SAXSx_o + 5.16)
                SAXSy.move(SAXSy.o + 5.16)
            if pilatus800 in cms.detector:
                WAXSx.move(WAXSx_o - 5.16)
                WAXSy.move(WAXSy_o + 5.16)
            extra_current = "pos4" if extra is None else "{}_pos4".format(extra)
            md["detector_position"] = "upper_right"
            self.measure_single(
                exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
            )

            # pos3
            if pilatus2M in cms.detector:
                SAXSx.move(SAXSx_o + 5.16)
                SAXSy.move(SAXSy_o)
            if pilatus800 in cms.detector:
                WAXSx.move(WAXSx_o - 5.16)
                WAXSy.move(WAXSy_o)

            extra_current = "pos3" if extra is None else "{}_pos3".format(extra)
            md["detector_position"] = "lower_right"
            self.measure_single(
                exposure_time=exposure_time, extra=extra_current, verbosity=verbosity, stitchback=True, **md
            )

            if WAXSx.user_readback.value != WAXSx_o:
                WAXSx.move(WAXSx_o)
            if WAXSy.user_readback.value != WAXSy_o:
                WAXSy.move(WAXSy_o)

            if SAXSx.user_readback.value != SAXSx_o:
                SAXSx.move(SAXSx_o)
            if SAXSy.user_readback.value != SAXSy_o:
                SAXSy.move(SAXSy_o)

        elif tiling is "ygaps":
            SAXSy_o = SAXSy.user_readback.value
            SAXSx_o = SAXSx.user_readback.value
            WAXSy_o = WAXSy.user_readback.value
            WAXSx_o = WAXSx.user_readback.value
            ##MAXSy_o = MAXSy.user_readback.value

            extra_current = "pos1" if extra is None else "{}_pos1".format(extra)
            md["detector_position"] = "lower"
            self.measure_single(
                exposure_time=exposure_time,
                extra=extra_current,
                measure_type=measure_type,
                verbosity=verbosity,
                stitchback=True,
                **md,
            )

            if pilatus2M in cms.detector:
                SAXSy.move(SAXSy_o + 5.16)
            if pilatus800 in cms.detector:
                WAXSy.move(WAXSy_o + 5.16)
            # if pilatus300 in cms.detector:
            # MAXSy.move(MAXSy_o + 5.16)

            # extra x movement is needed for pilatus2M.
            extra_current = "pos2" if extra is None else "{}_pos2".format(extra)
            md["detector_position"] = "upper"
            self.measure_single(
                exposure_time=exposure_time,
                extra=extra_current,
                measure_type=measure_type,
                verbosity=verbosity,
                stitchback=True,
                **md,
            )

            if SAXSy.user_readback.value != SAXSy_o:
                SAXSy.move(SAXSy_o)
            if WAXSy.user_readback.value != WAXSy_o:
                WAXSy.move(WAXSy_o)
            # if MAXSy.user_readback.value != MAXSy_o:
            # MAXSy.move(MAXSy_o)

        else:
            # Just do a normal measurement
            self.measure_single(
                exposure_time=exposure_time, extra=extra, measure_type=measure_type, verbosity=verbosity, **md
            )

    def measureRock(
        self,
        incident_angle=None,
        exposure_time=None,
        extra=None,
        measure_type="measure",
        rock_motor=None,
        rock_motor_limits=0.1,
        verbosity=3,
        stitchback=False,
        poling_period=0.2,
        **md,
    ):
        """Measure data when swing the rock_motor.

        Parameters
        ----------
        exposure_time : float
            How long to collect data
        extra : string, optional
            Extra information about this particular measurement (which is typically
            included in the savename/filename).
        rock_motor: string, optional
            Motor to swing in measurement
        rock_motor_limits: float
            The offset in the swing range
        """
        if incident_angle != None:
            self.thabs(incident_angle)

        if exposure_time is not None:
            self.set_attribute("exposure_time", exposure_time)
        # else:
        # exposure_time = self.get_attribute('exposure_time')

        savename = self.get_savename(savename_extra=extra)

        # caput('XF:11BMB-ES{Det:SAXS}:cam1:FileName', savename)

        if verbosity >= 2 and (get_beamline().current_mode != "measurement"):
            print(
                "WARNING: Beamline is not in measurement mode (mode is '{}')".format(get_beamline().current_mode)
            )

        if verbosity >= 1 and len(get_beamline().detector) < 1:
            print("ERROR: No detectors defined in cms.detector")
            return

        md_current = self.get_md()
        md_current.update(self.get_measurement_md())
        md_current["sample_savename"] = savename
        md_current["measure_type"] = measure_type
        # md_current['filename'] = '{:s}_{:04d}.tiff'.format(savename, md_current['detector_sequence_ID'])
        md_current["filename"] = "{:s}_{:04d}.tiff".format(savename, RE.md["scan_id"])
        md_current["beam_int_bim3"] = beam.bim3.flux(verbosity=0)
        md_current["beam_int_bim4"] = beam.bim4.flux(verbosity=0)
        md_current["beam_int_bim5"] = beam.bim5.flux(verbosity=0)
        # md_current['plan_header_override'] = md['measure_type']
        md_current.update(md)

        if "measure_type" not in md:
            md["measure_type"] = "expose"
        # self.log('{} for {}.'.format(md['measure_type'], self.name), **md)

        # Set exposure time
        if exposure_time is not None:
            # for detector in gs.DETS:
            for detector in get_beamline().detector:
                if (
                    exposure_time != detector.cam.acquire_time.get()
                ):  # caget('XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime'):
                    RE(detector.setExposureTime(exposure_time, verbosity=verbosity))
                # if detector.name is 'pilatus300' and exposure_time != detector.cam.acquire_time.get():
                # detector.setExposureTime(exposure_time, verbosity=verbosity)
                ##extra wait time when changing the exposure time.
                ##time.sleep(2)
                #############################################
                ##extra wait time for adjusting pilatus2M
                ##this extra wait time has to be added. Otherwise, the exposure will be skipped when the exposure time is increased
                ##Note by 091918
                #############################################
                # time.sleep(2)
                # elif detector.name is 'PhotonicSciences_CMS':
                # detector.setExposureTime(exposure_time, verbosity=verbosity)

        # Do acquisition
        get_beamline().beam.on()

        start_time = time.time()
        # Define the rock
        # rock_scan=list_scan(get_beamline().detector, armr, [0], per_step = functools.partial(rock_motor_per_step, rock_motor=rock_motor, rock_motor_limits=rock_motor_limits) )
        rock_scan = list_scan(
            get_beamline().detector,
            armr,
            [0],
            per_step=lambda detectors, motor, step: rock_motor_per_step(
                detectors,
                motor,
                step,
                rock_motor=rock_motor,
                rock_motor_limits=rock_motor_limits,
            ),
        )
        # uids = RE(count(get_beamline().detector, 1), **md)
        uids = RE(rock_scan, **md_current)

        # Wait for detectors to be ready
        max_exposure_time = 0
        for detector in get_beamline().detector:
            if detector.name is "pilatus300" or "pilatus800" or "pilatus2M" or "pilatus8002":
                max_exposure_time = detector.cam.acquire_time.get()

            # if detector.name is 'pilatus300':
            #     current_exposure_time = caget('XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime')
            #     max_exposure_time = max(max_exposure_time, current_exposure_time)
            # elif detector.name is 'pilatus2M':
            #     current_exposure_time = caget('XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime')
            #     max_exposure_time = max(max_exposure_time, current_exposure_time)
            # elif detector.name is 'pilatus800':
            #     current_exposure_time = caget('XF:11BMB-ES{Det:PIL800K}:cam1:AcquireTime')
            #     max_exposure_time = max(max_exposure_time, current_exposure_time)
            # elif detector.name is 'PhotonicSciences_CMS':
            # current_exposure_time = detector.exposure_time
            # max_exposure_time = max(max_exposure_time, current_exposure_time)
            else:
                if verbosity >= 1:
                    print("WARNING: Didn't recognize detector '{}'.".format(detector.name))

        if verbosity >= 2:
            status = 0
            while (status == 0) and (time.time() - start_time) < (max_exposure_time + 20):
                percentage = 100 * (time.time() - start_time) / max_exposure_time
                print(
                    "Exposing {:6.2f} s  ({:3.0f}%)      \r".format((time.time() - start_time), percentage),
                    end="",
                )
                time.sleep(poling_period)

                status = 1
                for detector in get_beamline().detector:
                    if detector.cam.acquire.get():
                        status *= 0

                    # if detector.name is 'pilatus300':
                    #     if caget('XF:11BMB-ES{Det:SAXS}:cam1:Acquire')==1:
                    #         status *= 0
                    # elif detector.name is 'pilatus2M':
                    #     if caget('XF:11BMB-ES{Det:PIL2M}:cam1:Acquire')==1:
                    #         status *= 0
                    # elif detector.name is 'pilatus800':
                    #     if caget('XF:11BMB-ES{Det:PIL800K}:cam1:Acquire')==1:
                    #         status *= 0
                    # elif detector.name is 'PhotonicSciences_CMS':
                    #     if not detector.detector_is_ready(verbosity=0):
                    #         status *= 0
            print("")

        else:
            time.sleep(max_exposure_time)

        # if verbosity>=3 and caget('XF:11BMB-ES{Det:SAXS}:cam1:Acquire')==1:
        #     print('Warning: Detector pilatus300 still not done acquiring.')
        # if verbosity>=3 and caget('XF:11BMB-ES{Det:PIL2M}:cam1:Acquire')==1:
        #     print('Warning: Detector pilatus2M still not done acquiring.')
        # if verbosity>=3 and caget('XF:11BMB-ES{Det:PIL800K}:cam1:Acquire')==1:
        #     print('Warning: Detector pilatus800 still not done acquiring.')

        get_beamline().beam.off()

        for detector in get_beamline().detector:
            self.handle_file(detector, extra=extra, verbosity=verbosity, **md_current)
            # self.handle_file(detector, extra=extra, verbosity=verbosity)
        self.md["measurement_ID"] += 1

    def measure_single(self, exposure_time=None, extra=None, measure_type="measure", verbosity=3, **md):
        """Measure data by triggering the area detectors.

        Parameters
        ----------
        exposure_time : float
            How long to collect data
        extra : string, optional
            Extra information about this particular measurement (which is typically
            included in the savename/filename).
        """

        if exposure_time is not None:
            self.set_attribute("exposure_time", exposure_time)
        # else:
        # exposure_time = self.get_attribute('exposure_time')

        savename = self.get_savename(savename_extra=extra)

        if verbosity >= 2 and (get_beamline().current_mode != "measurement"):
            print(
                "WARNING: Beamline is not in measurement mode (mode is '{}')".format(get_beamline().current_mode)
            )

        if verbosity >= 1 and len(get_beamline().detector) < 1:
            print("ERROR: No detectors defined in cms.detector")
            return

        md_current = self.get_md()
        md_current.update(self.get_measurement_md())
        md_current["sample_savename"] = savename
        md_current["measure_type"] = measure_type
        # md_current['filename'] = '{:s}_{:04d}.tiff'.format(savename, md_current['detector_sequence_ID'])
        # md_current['filename'] = '{:s}_{:04d}.tiff'.format(savename, RE.md['scan_id'])
        md_current["filename"] = "{:s}_{:06d}".format(savename, RE.md["scan_id"])
        md_current.update(md)

        self.expose(exposure_time, extra=extra, verbosity=verbosity, **md_current)
        # self.expose(exposure_time, extra=extra, verbosity=verbosity, **md)

        self.md["measurement_ID"] += 1

    def _test_time(self):
        print(time.time())
        time.time()

    def _test_measure_single(
        self, exposure_time=None, extra=None, shutteronoff=True, measure_type="measure", verbosity=3, **md
    ):
        """Measure data by triggering the area detectors.

        Parameters
        ----------
        exposure_time : float
            How long to collect data
        extra : string, optional
            Extra information about this particular measurement (which is typically
            included in the savename/filename).
        """

        # print('1') #0s
        # print(time.time())

        if exposure_time is not None:
            self.set_attribute("exposure_time", exposure_time)
        # else:
        # exposure_time = self.get_attribute('exposure_time')

        savename = self.get_savename(savename_extra=extra)

        # caput('XF:11BMB-ES{Det:SAXS}:cam1:FileName', savename)

        if verbosity >= 2 and (get_beamline().current_mode != "measurement"):
            print(
                "WARNING: Beamline is not in measurement mode (mode is '{}')".format(get_beamline().current_mode)
            )

        if verbosity >= 1 and len(get_beamline().detector) < 1:
            print("ERROR: No detectors defined in cms.detector")
            return

        # print('2') #0.0004s
        # print(time.time())

        md_current = self.get_md()
        md_current["sample_savename"] = savename
        md_current["measure_type"] = measure_type

        md_current.update(self.get_measurement_md())
        # md_current['filename'] = '{:s}_{:04d}.tiff'.format(savename, md_current['detector_sequence_ID'])
        md_current["filename"] = "{:s}_{:04d}.tiff".format(savename, RE.md["scan_id"])
        md_current.update(md)

        # print('3') #0.032s
        # print(time.time())

        self._test_expose(exposure_time, shutteronoff=shutteronoff, extra=extra, verbosity=verbosity, **md_current)

        # print('4') #5.04s
        # print(time.time())

        self.md["measurement_ID"] += 1

        # print('5') #5.0401
        # print(time.time())

    def _test_expose(
        self, exposure_time=None, extra=None, verbosity=3, poling_period=0.1, shutteronoff=True, **md
    ):
        """Internal function that is called to actually trigger a measurement."""

        if "measure_type" not in md:
            md["measure_type"] = "expose"
        # self.log('{} for {}.'.format(md['measure_type'], self.name), **md)

        # Set exposure time
        if exposure_time is not None:
            for detector in get_beamline().detector:
                detector.setExposureTime(exposure_time, verbosity=verbosity)

        # print('1') #5e-5
        # print(self.clock())

        # Do acquisition
        # check shutteronoff, if
        if shutteronoff == True:
            get_beamline().beam.on()
        else:
            print("shutter is disabled")

        # print('2') #3.0
        # print(self.clock())

        md["plan_header_override"] = md["measure_type"]
        start_time = time.time()
        print("2")  # 3.0
        print(self.clock())

        # uids = RE(count(get_beamline().detector, 1), **md)
        # uids = RE(count(get_beamline().detector), **md)
        yield from (count(get_beamline().detector))
        print("3")  # 4.3172
        print(self.clock())

        # get_beamline().beam.off()
        # print('shutter is off')

        # Wait for detectors to be ready
        max_exposure_time = 0
        for detector in get_beamline().detector:
            if detector.name is "pilatus300" or "pilatus2M":
                current_exposure_time = caget("XF:11BMB-ES{}:cam1:AcquireTime".format(pilatus_Epicsname))
                max_exposure_time = max(max_exposure_time, current_exposure_time)
            elif detector.name is "PhotonicSciences_CMS":
                current_exposure_time = detector.exposure_time
                max_exposure_time = max(max_exposure_time, current_exposure_time)
            else:
                if verbosity >= 1:
                    print("WARNING: Didn't recognize detector '{}'.".format(detector.name))

        print("4")  # 4.3193
        print(self.clock())

        if verbosity >= 2:
            status = 0
            print("status1 = ", status)

            while (status == 0) and (time.time() - start_time) < (max_exposure_time + 20):
                percentage = 100 * (time.time() - start_time) / max_exposure_time
                print(
                    "Exposing {:6.2f} s  ({:3.0f}%)      \r".format((time.time() - start_time), percentage),
                    end="",
                )
                print("status2 = ", status)

                time.sleep(poling_period)

                status = 1
                for detector in get_beamline().detector:
                    if detector.name is "pilatus300" or "pilatus2M":
                        print("status2.5 = ", status)
                        if caget("XF:11BMB-ES{}:cam1:Acquire".format(pilatus_Epicsname)) == 1:
                            status = 0
                            print("status3 = ", status)
                        print("status3.5 = ", status)

                    elif detector.name is "PhotonicSciences_CMS":
                        if not detector.detector_is_ready(verbosity=0):
                            status = 0
                print("5")  # 3.0
                print(self.clock())
            print("6")  # 3.0
            print(self.clock())

        else:
            time.sleep(max_exposure_time)

        # print('5') #4.4193
        # print(self.clock())

        if verbosity >= 3 and caget("XF:11BMB-ES{}:cam1:Acquire".format(pilatus_Epicsname)) == 1:
            print("Warning: Detector still not done acquiring.")

        if shutteronoff == True:
            get_beamline().beam.off()
        else:
            print("shutter is disabled")

        # print('6') #4.9564
        # print(self.clock())

        for detector in get_beamline().detector:
            self.handle_file(detector, extra=extra, verbosity=verbosity, **md)

        # print('7') #4.9589
        # print(self.clock())

    def _test_measureSpots(
        self,
        num_spots=4,
        translation_amount=0.2,
        axis="y",
        exposure_time=None,
        extra=None,
        shutteronoff=True,
        measure_type="measureSpots",
        tiling=False,
        **md,
    ):
        """Measure multiple spots on the sample."""

        if "spot_number" not in self.md:
            self.md["spot_number"] = 1

        start_time = time.time()

        for spot_num in range(num_spots):
            self._test_measure_single(
                exposure_time=exposure_time,
                extra=extra,
                measure_type=measure_type,
                shutteronoff=shutteronoff,
                tiling=tiling,
                **md,
            )

            print(spot_num + 1)
            print(time.time() - start_time)
            getattr(self, axis + "r")(translation_amount)
            self.md["spot_number"] += 1
            print("{:d} of {:d} is done".format(spot_num + 1, num_spots))
            print(time.time() - start_time)

    def measureSpots(
        self,
        num_spots=4,
        translation_amount=0.2,
        axis="y",
        exposure_time=None,
        extra=None,
        measure_type="measureSpots",
        tiling=False,
        **md,
    ):
        """Measure multiple spots on the sample."""

        if "spot_number" not in self.md:
            self.md["spot_number"] = 1

        for spot_num in range(num_spots):
            self.measure(exposure_time=exposure_time, extra=extra, measure_type=measure_type, tiling=tiling, **md)

            getattr(self, axis + "r")(translation_amount)
            self.md["spot_number"] += 1
            print("{:d} of {:d} is done".format(spot_num + 1, num_spots))

    def measureTimeSeries(
        self,
        exposure_time=None,
        num_frames=10,
        wait_time=None,
        extra=None,
        measure_type="measureTimeSeries",
        verbosity=3,
        tiling=False,
        fix_name=True,
        **md,
    ):
        if fix_name and ("clock" not in self.naming_scheme):
            self.naming_scheme_hold = self.naming_scheme
            self.naming_scheme = self.naming_scheme_hold.copy()
            self.naming_scheme.insert(-1, "clock")

        md["measure_series_num_frames"] = num_frames

        for i in range(num_frames):
            if verbosity >= 3:
                print(
                    "Measuring frame {:d}/{:d} ({:.1f}% complete).".format(
                        i + 1, num_frames, 100.0 * i / num_frames
                    )
                )

            md["measure_series_current_frame"] = i + 1
            self.measure(
                exposure_time=exposure_time,
                extra=extra,
                measure_type=measure_type,
                verbosity=verbosity,
                tiling=tiling,
                **md,
            )
            if wait_time is not None:
                time.sleep(wait_time)

    # def measureTimeSeriesAngles(self, exposure_time=None, num_frames=10, wait_time=None, extra=None, measure_type='measureTimeSeries', verbosity=3, tiling=False, fix_name=True, **md):

    # if fix_name and ('clock' not in self.naming_scheme):
    # self.naming_scheme_hold = self.naming_scheme
    # self.naming_scheme = self.naming_scheme_hold.copy()
    # self.naming_scheme.insert(-1, 'clock')

    # md['measure_series_num_frames'] = num_frames

    # for i in range(num_frames):

    # if verbosity>=3:
    # print('Measuring frame {:d}/{:d} ({:.1f}% complete).'.format(i+1, num_frames, 100.0*i/num_frames))

    # md['measure_series_current_frame'] = i+1
    # print('Angles in measure include: {}'.format(sam.incident_angles_default))
    # self.measureIncidentAngles(exposure_time=exposure_time, extra=extra, **md)
    # if wait_time is not None:
    # time.sleep(wait_time)
    ##if (i % 2 ==0):
    ##    self.xr(-1)
    ##else:
    ##    self.xr(1)
    ##self.pos()

    def measureTimeSeriesAngles(
        self,
        exposure_time=None,
        num_frames=10,
        wait_time=None,
        extra=None,
        measure_type="measureTimeSeries",
        verbosity=3,
        tiling=False,
        fix_name=True,
        **md,
    ):
        if fix_name and ("clock" not in self.naming_scheme):
            self.naming_scheme_hold = self.naming_scheme
            self.naming_scheme = self.naming_scheme_hold.copy()
            self.naming_scheme.insert(-1, "clock")

        md["measure_series_num_frames"] = num_frames

        for i in range(num_frames):
            if verbosity >= 3:
                print(
                    "Measuring frame {:d}/{:d} ({:.1f}% complete).".format(
                        i + 1, num_frames, 100.0 * i / num_frames
                    )
                )

            md["measure_series_current_frame"] = i + 1
            print("Angles in measure include: {}".format(sam.incident_angles_default))
            self.measureIncidentAngles(exposure_time=exposure_time, extra=extra, **md)
            if wait_time is not None:
                time.sleep(wait_time)

    def measureTemperature(
        self,
        temperature,
        exposure_time=None,
        wait_time=None,
        temperature_probe="A",
        temperature_tolerance=0.4,
        extra=None,
        measure_type="measureTemperature",
        verbosity=3,
        tiling=False,
        poling_period=1.0,
        fix_name=True,
        **md,
    ):
        # Set new temperature
        self.setTemperature(temperature, temperature_probe=temperature_probe, verbosity=verbosity)

        # Wait until we reach the temperature
        while (
            abs(self.temperature(temperature_probe=temperature_probe, verbosity=0) - temperature)
            > temperature_tolerance
        ):
            if verbosity >= 3:
                print(
                    "  setpoint = {:.3f}°C, Temperature = {:.3f}°C          \r".format(
                        self.temperature_setpoint(temperature_probe=temperature_probe) - 273.15,
                        self.temperature(verbosity=0),
                    ),
                    end="",
                )
            time.sleep(poling_period)

        # Allow for additional equilibration at this temperature
        if wait_time is not None:
            time.sleep(wait_time)

        # Measure
        # if fix_name and ('temperature' not in self.naming_scheme):
        #    self.naming_scheme_hold = self.naming_scheme
        #    self.naming_scheme = self.naming_scheme_hold.copy()
        #    self.naming_scheme.insert(-1, 'temperature')

        self.measure(
            exposure_time=exposure_time,
            extra=extra,
            measure_type=measure_type,
            verbosity=verbosity,
            tiling=tiling,
            **md,
        )

        # self.naming_scheme = self.naming_scheme_hold

    def measureTemperatures(
        self,
        temperatures,
        exposure_time=None,
        wait_time=None,
        temperature_probe="A",
        temperature_tolerance=0.4,
        extra=None,
        measure_type="measureTemperature",
        verbosity=3,
        tiling=False,
        poling_period=1.0,
        fix_name=True,
        **md,
    ):
        for temperature in temperatures:
            self.measureTemperature(
                temperature,
                exposure_time=exposure_time,
                wait_time=wait_time,
                temperature_probe=temperature_probe,
                temperature_tolerance=temperature_tolerance,
                measure_type=measure_type,
                verbosity=verbosity,
                tiling=tiling,
                poling_period=poling_period,
                fix_name=fix_name,
                **md,
            )

    def do(self, step=0, verbosity=3, **md):
        """Performs the "default action" for this sample. This usually means
        aligning the sample, and taking data.

        The 'step' argument can optionally be given to jump to a particular
        step in the sequence."""

        if verbosity >= 4:
            print("  doing sample {}".format(self.name))

        if step <= 1:
            if verbosity >= 5:
                print("    step 1: goto origin")
            self.xo()  # goto origin
            self.yo()
            # self.gotoAlignedPosition()

        # if step<=5:
        # self.align()

        if step <= 10:
            if verbosity >= 5:
                print("    step 10: measuring")
            self.measure(self.SAXS_time)

    def scan_measure(
        self,
        motor,
        start,
        stop,
        num_frames,
        exposure_time=None,
        detectors=None,
        extra=None,
        per_step=None,
        wait_time=None,
        measure_type="Scan_measure",
        verbosity=3,
        fill_gaps=False,
        **md,
    ):
        """
        Scans the specified motor and record the detectors with shutter open during the scan.

        Parameters
        ----------
        motor : motor
            The axis/stage/motor that you want to move.
        start, stop : float
            The relative positions of the scan range.
        num_frames : int
            The number of scan points.
        exposure_time: float
            The exposure time for single point
        md : dict, optional
            metadata
        """
        # span = abs(stop-start)
        # positions, dp = np.linspace(start, stop, num, endpoint=True, retstep=True)

        if detectors is None:
            detectors = cms.detector

        if exposure_time is not None:
            self.set_attribute("exposure_time", exposure_time)

        bec.disable_plots()
        bec.disable_table()

        savename = self.get_savename(savename_extra=extra)
        if verbosity >= 2 and (get_beamline().current_mode != "measurement"):
            print(
                "WARNING: Beamline is not in measurement mode (mode is '{}')".format(get_beamline().current_mode)
            )
        if verbosity >= 1 and len(cms.detector) < 1:
            print("ERROR: No detectors defined in cms.detector")
            return

        # set exposure time
        for detector in get_beamline().detector:
            detector.setExposureTime(exposure_time, verbosity=verbosity)
        # set metadata
        md_current = self.get_md()
        md_current["sample_savename"] = savename
        md_current["measure_type"] = measure_type
        md_current["scan"] = "scan_measure"
        md_current.update(self.get_measurement_md())
        md_current["measure_series_num_frames"] = num_frames
        md_current["filename"] = "{:s}_{:04d}.tiff".format(savename, RE.md["scan_id"])
        md_current["measure_series_motor"] = motor.name
        md_current["measure_series_positions"] = [start, stop]
        md_current["exposure_time"] = exposure_time
        md_current.update(md)

        print(RE.md["scan_id"])

        # Perform the scan
        # get_beamline().beam._test_on(wait_time=0.1)
        get_beamline().beam.on()
        # RE(relative_scan(gs.DETS, motor, start, stop, num_frames+1, per_step=per_step, md=md_current))
        RE(
            relative_scan(
                cms.detector,
                motor,
                start,
                stop,
                num_frames + 1,
                per_step=per_step,
                md=md_current,
            ),
            LiveTable([motor, "motor_setpoint"]),
        )

        # if verbosity>=3 and caget('XF:11BMB-ES{Det:SAXS}:cam1:Acquire')==1:
        stage = 1
        for detector in cms.detector:
            if detector.cam.acquire.get() == 1:
                # if verbosity>=3 and caget('XF:11BMB-ES{Det:SAXS}:cam1:Acquire')==1:

                print("Warning: Detector {} still not done acquiring.".format(detector.name))
        # elif verbosity>=3 and caget('XF:11BMB-ES{Det:PIL2M}:cam1:Acquire')==1:
        #     print('Warning: Detector Pilatus2M still not done acquiring.')
        # get_beamline().beam._test_off(wait_time=0.1)
        get_beamline().beam.off()
        self.md["measurement_ID"] += 1

        # data collected, link uid to file name
        for detector in cms.detector:
            # print(detector.name)
            self.handle_fileseries(detector, num_frames=num_frames, extra=extra, verbosity=verbosity, **md)

    def series_measure(
        self,
        num_frames,
        exposure_time=None,
        exposure_period=None,
        detectors=None,
        extra=None,
        per_step=None,
        wait_time=None,
        measure_type="Series_measure",
        verbosity=3,
        fill_gaps=False,
        **md,
    ):
        """
        Continueous shots with internal trigger of detectors. (burst mode)

        Parameters
        ----------
        num_frames : int
            The number of data points.
        exposure_time: float
            The exposure time for single point
        exposure_period: float
            The exposure period for single point. should be at least 0.05s longer than exposure_time
        md : dict, optional
            metadata
        """
        # span = abs(stop-start)
        # positions, dp = np.linspace(start, stop, num, endpoint=True, retstep=True)

        if detectors is None:
            detectors = cms.detector

        if exposure_time is not None:
            self.set_attribute("exposure_time", exposure_time)

        # Set exposure time
        for detector in get_beamline().detector:
            if exposure_time != detector.cam.acquire_time.get():
                RE(detector.setExposureTime(exposure_time))
                # detector.cam.acquire_time.put(exposure_time)
            # detector.cam.acquire_period.put(exposure_period)
            # detector.cam.num_images.put(num_frames)
            RE(detector.setExposurePeriod(exposure_period))
            RE(detector.setExposureNumber(num_frames))

        # bec.disable_plots()
        # bec.disable_table()

        savename = self.get_savename(savename_extra=extra)
        if verbosity >= 2 and (get_beamline().current_mode != "measurement"):
            print(
                "WARNING: Beamline is not in measurement mode (mode is '{}')".format(get_beamline().current_mode)
            )

        if verbosity >= 1 and len(get_beamline().detector) < 1:
            print("ERROR: No detectors defined in cms.detector")
            return

        md_current = self.get_md()
        md_current["sample_savename"] = savename
        md_current["measure_type"] = measure_type
        md_current["series"] = "series_measure"
        md_current.update(self.get_measurement_md())
        # md_current['filename'] = '{:s}_{:04d}.tiff'.format(savename, md_current['detector_sequence_ID'])
        md_current["measure_series_num_frames"] = num_frames
        md_current["filename"] = "{:s}_{:04d}.tiff".format(savename, RE.md["scan_id"])
        # md_current['filename'] = '{:s}_{:04d}.tiff'.format(savename, RE.md['scan_id']+1)
        md_current["exposure_time"] = exposure_time
        md_current["exposure_period"] = exposure_period
        # md_current['measure_series_motor'] = motor.name
        # md_current['measure_series_positions'] = [start, stop]

        # md_current['fileno'] = '{:s}_{:04d}.tiff'.format(savename, RE.md['scan_id'])
        md_current.update(md)

        print(RE.md["scan_id"])

        # Perform the scan
        # get_beamline().beam._test_on(wait_time=0.1)
        get_beamline().beam.on()
        RE(count(get_beamline().detector, md=md_current))
        get_beamline().beam.off()

        self.md["measurement_ID"] += 1
        # reset the num_frame back to 1
        for detector in get_beamline().detector:
            RE(detector.setExposureNumber(1))

        # data collected, link uid to file name
        for detector in cms.detector:
            print("handling the file names")
            self.handle_fileseries(detector, num_frames=num_frames, extra=extra, verbosity=verbosity, **md)

            # if detector.name is 'pilatus2M':
            #     caput('XF:11BMB-ES{Det:PIL2M}:cam1:NumImages', 1)
            # if detector.name is 'pilatus300' :
            #     caput('XF:11BMB-ES{Det:SAXS}:cam1:NumImages', 1)
            # if detector.name is 'pilatus800' :
            #     caput('XF:11BMB-ES{Det:PIL800K}:cam1:NumImages', 1)

    def initialDetector(self):
        # reset the num_frame back to 1
        for detector in get_beamline().detector:
            detector.cam.num_images.put(1)
            # if detector.name is 'pilatus2M':
            #     caput('XF:11BMB-ES{Det:PIL2M}:cam1:NumImages', 1)
            # if detector.name is 'pilatus300' :
            #     caput('XF:11BMB-ES{Det:SAXS}:cam1:NumImages', 1)
            # if detector.name is 'pilatus800' :
            #     caput('XF:11BMB-ES{Det:PIL800K}:cam1:NumImages', 1)

    def _old_handle_fileseries(self, detector, num_frames=None, extra=None, verbosity=3, subdirs=True, **md):
        subdir = ""

        if detector.name == "pilatus300" or detector.name == "pilatus8002":
            # chars = caget('XF:11BMB-ES{Det:SAXS}:TIFF1:FullFileName_RBV')
            # filename = ''.join(chr(char) for char in chars)[:-1]
            # filename_part1 = ''.join(chr(char) for char in chars)[:-13]

            filename = detector.tiff.full_file_name.get()  # RL, 20210831

            print("pilatus300k data handling")
            # Alternate method to get the last filename
            # filename = '{:s}/{:s}.tiff'.format( detector.tiff.file_path.get(), detector.tiff.file_name.get()  )

            # if verbosity>=3:
            #    print('  Data saved to: {}'.format(filename))

            if subdirs:
                subdir = "/maxs/raw/"

            # if md['measure_type'] is not 'snap':
            if True:
                # self.set_attribute('exposure_time', caget('XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime'))
                self.set_attribute("exposure_time", detector.cam.acquire_time.get())  # RL, 20210831
                # Create symlink
                # link_name = '{}/{}{}'.format(RE.md['experiment_alias_directory'], subdir, md['filename'])
                # savename = md['filename'][:-5]

                savename = self.get_savename(savename_extra=extra)
                link_name = "{}/{}{}_{:04d}_maxs.tiff".format(
                    RE.md["experiment_alias_directory"],
                    subdir,
                    savename,
                    RE.md["scan_id"] - 1,
                )
                link_name_part1 = "{}/{}{}_{:04d}".format(
                    RE.md["experiment_alias_directory"],
                    subdir,
                    savename,
                    RE.md["scan_id"] - 1,
                )

                if os.path.isfile(link_name):
                    i = 1
                    while os.path.isfile("{}.{:d}".format(link_name, i)):
                        i += 1
                    os.rename(link_name, "{}.{:d}".format(link_name, i))

                for num_frame in range(num_frames):
                    filename_new = "{}_{:06d}.tiff".format(filename_part1, num_frame)
                    link_name_new = "{}_{:06d}_maxs.tiff".format(link_name_part1, num_frame)
                    os.symlink(filename_new, link_name_new)
                    if verbosity >= 3:
                        if num_frame == 0 or num_frame == np.max(num_frames):
                            print("  Data {} linked as: {}".format(filename_new, link_name_new))

        elif detector.name == "pilatus2M":
            # chars = caget('XF:11BMB-ES{Det:PIL2M}:TIFF1:FullFileName_RBV')
            # filename = ''.join(chr(char) for char in chars)[:-1]
            # filename_part1 = ''.join(chr(char) for char in chars)[:-13]

            filename = detector.tiff.full_file_name.get()  # RL, 20210831
            filename_part1 = detector.tiff.file_path.get() + detector.tiff.file_name.get()

            print("pilatus2M data handling")

            # Alternate method to get the last filename
            # filename = '{:s}/{:s}.tiff'.format( detector.tiff.file_path.get(), detector.tiff.file_name.get()  )
            # filename_part1 = '{:s}/{:s}.tiff'.format( detector.tiff.file_path.get(), detector.tiff.file_name.get()  )

            # if verbosity>=3:
            #    print('  Data saved to: {}'.format(filename))

            if subdirs:
                subdir = "/saxs/raw/"

            # if md['measure_type'] is not 'snap':
            if True:
                # self.set_attribute('exposure_time', caget('XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime'))
                self.set_attribute("exposure_time", detector.cam.acquire_time.get())  # RL, 20210831

                # Create symlink
                # link_name = '{}/{}{}'.format(RE.md['experiment_alias_directory'], subdir, md['filename'])
                # savename = md['filename'][:-5]

                savename = self.get_savename(savename_extra=extra)
                link_name = "{}/{}{}_{:04d}_saxs.tiff".format(
                    RE.md["experiment_alias_directory"],
                    subdir,
                    savename,
                    RE.md["scan_id"] - 1,
                )
                link_name_part1 = "{}/{}{}_{:04d}".format(
                    RE.md["experiment_alias_directory"],
                    subdir,
                    savename,
                    RE.md["scan_id"] - 1,
                )

                if os.path.isfile(link_name):
                    i = 1
                    while os.path.isfile("{}.{:d}".format(link_name, i)):
                        i += 1
                    os.rename(link_name, "{}.{:d}".format(link_name, i))

                for num_frame in range(num_frames):
                    filename_new = "{}_{:06d}.tiff".format(filename_part1, num_frame)
                    link_name_new = "{}_{:06d}_saxs.tiff".format(link_name_part1, num_frame)
                    os.symlink(filename_new, link_name_new)
                    if verbosity >= 3:
                        if num_frame == 0 or num_frame == np.max(num_frames):
                            print("  Data {} linked as: {}".format(filename_new, link_name_new))

        # elif detector.name is  'pilatus800':
        # chars = caget('XF:11BMB-ES{Det:PIL800K}:TIFF1:FullFileName_RBV')
        # filename = ''.join(chr(char) for char in chars)[:-1]
        # filename_part1 = ''.join(chr(char) for char in chars)[:-13]

        elif detector.name == "pilatus800":
            foldername = "/nsls2/xf11bm/"

            # chars = caget('XF:11BMB-ES{Det:PIL800K}:TIFF1:FullFileName_RBV')
            # filename = ''.join(chr(char) for char in chars)[:-1]
            # filename = foldername + filename
            # filename_part1 = foldername + ''.join(chr(char) for char in chars)[:-13]

            filename = pilatus800.tiff.full_file_name.get()  # RL, 20210831
            filename_part1 = pilatus800.tiff.file_path.get() + pilatus800.tiff.file_name.get()

            print("pilatus800 data handling")

            # Alternate method to get the last filename
            # filename = '{:s}/{:s}.tiff'.format( detector.tiff.file_path.get(), detector.tiff.file_name.get()  )

            # if verbosity>=3:
            #    print('  Data saved to: {}'.format(filename))

            if subdirs:
                subdir = "/waxs/raw/"

            # if md['measure_type'] is not 'snap':
            if True:
                # self.set_attribute('exposure_time', caget('XF:11BMB-ES{Det:PIL800K}:cam1:AcquireTime'))
                self.set_attribute("exposure_time", pilatus800.cam.acquire_time.get())  # RL, 20210831

                # Create symlink
                # link_name = '{}/{}{}'.format(RE.md['experiment_alias_directory'], subdir, md['filename'])
                # savename = md['filename'][:-5]

                savename = self.get_savename(savename_extra=extra)
                link_name = "{}/{}{}_{:04d}_waxs.tiff".format(
                    RE.md["experiment_alias_directory"],
                    subdir,
                    savename,
                    RE.md["scan_id"] - 1,
                )
                link_name_part1 = "{}/{}{}_{:04d}".format(
                    RE.md["experiment_alias_directory"],
                    subdir,
                    savename,
                    RE.md["scan_id"] - 1,
                )

                if os.path.isfile(link_name):
                    i = 1
                    while os.path.isfile("{}.{:d}".format(link_name, i)):
                        i += 1
                    os.rename(link_name, "{}.{:d}".format(link_name, i))

                for num_frame in range(num_frames):
                    filename_new = "{}_{:06d}.tiff".format(filename_part1, num_frame)
                    link_name_new = "{}_{:06d}_waxs.tiff".format(link_name_part1, num_frame)
                    os.symlink(filename_new, link_name_new)
                    if verbosity >= 3:
                        if num_frame == 0 or num_frame == np.max(num_frames):
                            print("  Data {} linked as: {}".format(filename_new, link_name_new))

        else:
            if verbosity >= 1:
                print("WARNING: Can't do file handling for detector '{}'.".format(detector.name))
                return

    def handle_fileseries(self, detector, num_frames=None, extra=None, verbosity=3, subdirs=True, **md):
        subdir = ""
        if subdirs:
            if detector.name == "pilatus300" or detector.name == "pilatus8002":
                subdir = "/maxs/raw/"
                detname = "maxs"
                print("{} data handling".format(detector.name))
            elif detector.name == "pilatus2M":
                subdir = "/saxs/raw/"
                detname = "saxs"
                print("pilatus2M data handling")
            elif detector.name == "pilatus800":
                subdir = "/waxs/raw/"
                detname = "waxs"
                print("pilatus800k data handling")
            else:
                if verbosity >= 1:
                    print("WARNING: Can't do file handling for detector '{}'.".format(detector.name))
                    return

        filename = detector.tiff.full_file_name.get()  # RL, 20210831
        filename_part1 = "{:s}/{:s}".format(detector.tiff.file_path.get(), detector.tiff.file_name.get())

        # Alternate method to get the last filename
        # filename = '{:s}/{:s}.tiff'.format( detector.tiff.file_path.get(), detector.tiff.file_name.get()  )

        # if verbosity>=3:
        #    print('  Data saved to: {}'.format(filename))

        # if md['measure_type'] is not 'snap':
        if True:
            # self.set_attribute('exposure_time', caget('XF:11BMB-ES{Det:SAXS}:cam1:AcquireTime'))
            self.set_attribute("exposure_time", detector.cam.acquire_time.get())  # RL, 20210831
            # Create symlink
            # link_name = '{}/{}{}'.format(RE.md['experiment_alias_directory'], subdir, md['filename'])
            # savename = md['filename'][:-5]

            savename = self.get_savename(savename_extra=extra)
            link_name = "{}/{}{}_{:06d}_{}.tiff".format(
                RE.md["experiment_alias_directory"],
                subdir,
                savename,
                RE.md["scan_id"] - 1,
                detname,
            )
            link_name_part1 = "{}/{}{}_{:06d}".format(
                RE.md["experiment_alias_directory"],
                subdir,
                savename,
                RE.md["scan_id"] - 1,
            )
            # link_name = '{}/{}{}_{:06d}_{}.tiff'.format(RE.md['experiment_alias_directory'], subdir, savename, RE.md['scan_id'], detname)
            # link_name_part1 = '{}/{}{}_{:06d}'.format(RE.md['experiment_alias_directory'], subdir, savename, RE.md['scan_id'])

            if os.path.isfile(link_name):
                i = 1
                while os.path.isfile("{}.{:d}".format(link_name, i)):
                    i += 1
                os.rename(link_name, "{}.{:d}".format(link_name, i))

            for num_frame in range(num_frames):
                filename_new = "{}_{:06d}.tiff".format(filename_part1, num_frame)
                if os.path.isfile(filename_new) == False:
                    return print("File number {} does not exist.".format(num_frame))

                link_name_new = "{}_{:06d}_{}.tiff".format(link_name_part1, num_frame, detname)
                os.symlink(filename_new, link_name_new)
                if verbosity >= 3:
                    if num_frame == 0 or num_frame == np.max(num_frames):
                        print("  Data {} linked as: {}".format(filename_new, link_name_new))
            savename = self.get_savename(savename_extra=extra)
            # savename = md['filename']
            # link_name = '{}/{}{}_{:04d}_maxs.tiff'.format(RE.md['experiment_alias_directory'], subdir, savename, RE.md['scan_id']-1)
            link_name = "{}/{}{}_{}.tiff".format(RE.md["experiment_alias_directory"], subdir, savename, detname)

    # Control methods
    ########################################
    def setTemperature(self, temperature, output_channel="1", verbosity=3):
        # if verbosity>=1:
        # print('Temperature functions not implemented in {}'.format(self.__class__.__name__))
        if output_channel == "1":
            if verbosity >= 2:
                print(
                    "  Changing temperature setpoint from {:.3f}°C  to {:.3f}°C".format(
                        caget("XF:11BM-ES{Env:01-Out:1}T-SP") - 273.15, temperature
                    )
                )
            caput("XF:11BM-ES{Env:01-Out:1}T-SP", temperature + 273.15)

        if output_channel == "2":
            if verbosity >= 2:
                print(
                    "  Changing temperature setpoint from {:.3f}°C  to {:.3f}°C".format(
                        caget("XF:11BM-ES{Env:01-Out:2}T-SP") - 273.15, temperature
                    )
                )
            caput("XF:11BM-ES{Env:01-Out:2}T-SP", temperature + 273.15)

        if output_channel == "3":
            if verbosity >= 2:
                print(
                    "  Changing temperature setpoint from {:.3f}°C  to {:.3f}°C".format(
                        caget("XF:11BM-ES{Env:01-Out:3}T-SP") - 273.15, temperature
                    )
                )
            caput("XF:11BM-ES{Env:01-Out:3}T-SP", temperature + 273.15)

        if output_channel == "4":
            if verbosity >= 2:
                print(
                    "  Changing temperature setpoint from {:.3f}°C  to {:.3f}°C".format(
                        caget("XF:11BM-ES{Env:01-Out:4}T-SP") - 273.15, temperature
                    )
                )
            caput("XF:11BM-ES{Env:01-Out:4}T-SP", temperature + 273.15)

    def temperature(self, temperature_probe="A", output_channel="1", RTDchan=2, verbosity=3):
        # if verbosity>=1:
        # print('Temperature functions not implemented in {}'.format(self.__class__.__name__))

        if temperature_probe == "A":
            current_temperature = caget("XF:11BM-ES{Env:01-Chan:A}T:C-I")
            if verbosity >= 3:
                print(
                    "  Temperature = {:.3f}°C (setpoint = {:.3f}°C)".format(
                        current_temperature,
                        self.temperature_setpoint(output_channel=output_channel) - 273.15,
                    )
                )
        if temperature_probe == "B":
            current_temperature = caget("XF:11BM-ES{Env:01-Chan:B}T:C-I")
            if verbosity >= 3:
                print(
                    "  Temperature = {:.3f}°C (setpoint = {:.3f}°C)".format(
                        current_temperature,
                        self.temperature_setpoint(output_channel=output_channel) - 273.15,
                    )
                )
        if temperature_probe == "C":
            current_temperature = caget("XF:11BM-ES{Env:01-Chan:C}T:C-I")
            if verbosity >= 3:
                print(
                    "  Temperature = {:.3f}°C (setpoint = {:.3f}°C)".format(
                        current_temperature,
                        self.temperature_setpoint(output_channel=output_channel) - 273.15,
                    )
                )
        if temperature_probe == "D":
            current_temperature = caget("XF:11BM-ES{Env:01-Chan:D}T:C-I")
            if verbosity >= 3:
                print(
                    "  Temperature = {:.3f}°C (setpoint = {:.3f}°C)".format(
                        current_temperature,
                        self.temperature_setpoint(output_channel=output_channel) - 273.15,
                    )
                )
        if temperature_probe == "E":
            try:
                current_temperature = ioL.read(RTD[RTDchan])
            except TypeError:
                current_temperature = -273.15
        return current_temperature

    def humidity(self, AI_chan=7, temperature=25, verbosity=3):
        return ioL.readRH(AI_chan=AI_chan, temperature=temperature, verbosity=verbosity)

    def transmission_data_output(self, slot_pos):
        """Output the tranmission of direct beam"""
        h = db[-1]
        dtable = h.table()

        # beam.absorber_transmission_list = [1, 0.041, 0.0017425, 0.00007301075, 0.00000287662355, 0.000000122831826, 0.00000000513437]
        scan_id = h.start["scan_id"]
        I_bim5 = h.start["beam_int_bim5"]  # beam intensity from bim5
        I0 = dtable.pilatus2M_stats4_total
        filename = h.start["sample_name"]
        exposure_time = h.start["sample_exposure_time"]
        # I2 = dtable.pilatus2M_stats2_total
        # I3 = 2*dtable.pilatus2M_stats1_total - dtable.pilatus2M_stats2_total
        # In = I3 / beam.absorber_transmission_list[slot_pos] / exposure_time

        current_data = {
            "a_filename": filename,
            "b_scanID": scan_id,
            "c_I0": I0,
            "d_I_bim5": I_bim5,
            "e_absorber_slot": slot_pos,
            #'f_absorber_ratio': beam.absorber_transmission_list[slot_pos],
            "f_absorber_ratio": beam.absorber()[1],
            "g_exposure_seconds": exposure_time,
        }

        return pds.DataFrame(data=current_data)

    def intMeasure(self, output_file, exposure_time):
        """Measure the transmission intensity of the sample by ROI4.
        The intensity will be saved in output_file
        """
        if abs(beam.energy(verbosity=0) - 13.5) < 0.1:
            beam.setAbsorber(4)
        elif abs(beam.energy(verbosity=0) - 17) < 0.1:
            beam.setAbsorber(6)

        print("Absorber is moved to position 4")

        # cms.setAbsorber(4)#armr.move(-31.100167+45) #slot 4 position

        saxs_on()
        bsx.move(bsx.position + 6)
        beam.setTransmission(1)
        cms.setDirectBeamROI(size=[10, 10])

        self.measure(exposure_time)

        temp_data = self.transmission_data_output(4)

        cms.modeMeasurement()
        beam.setAbsorber(0)
        # armr.move(-55) #default position with direct beam thru

        # output_data = output_data.iloc[0:0]

        # create a data file to save the INT data
        INT_FILENAME = "{}/data/{}.csv".format(os.path.dirname(__file__), output_file)

        if os.path.isfile(INT_FILENAME):
            output_data = pds.read_csv(INT_FILENAME, index_col=0)
            output_data = output_data.append(temp_data, ignore_index=True)
            output_data.to_csv(INT_FILENAME)
        else:
            temp_data.to_csv(INT_FILENAME)

    def temperature_setpoint(self, output_channel="1", verbosity=3):
        # if verbosity>=1:
        # print('Temperature functions not implemented in {}'.format(self.__class__.__name__))

        if output_channel == "1":
            setpoint_temperature = caget("XF:11BM-ES{Env:01-Out:1}T-SP")

        if output_channel == "2":
            setpoint_temperature = caget("XF:11BM-ES{Env:01-Out:2}T-SP")

        if output_channel == "3":
            setpoint_temperature = caget("XF:11BM-ES{Env:01-Out:3}T-SP")

        if output_channel == "4":
            setpoint_temperature = caget("XF:11BM-ES{Env:01-Out:4}T-SP")

        return setpoint_temperature

    def monitor_scheme(self, scheme):
        """monitor paramteres in the naming scheme and record it in the fucntion of time.
        scheme = ['x', 'y', 'SAXSx', 'voltage']
        """
        current_data = {}
        current_data["time"] = time.time()
        current_data["dt_time"] = datetime.fromtimestamp(time.time())
        for item in scheme:
            current_data["{}".format(item)] = self.get_attribute(item)
        # print(current_data)
        return pds.DataFrame([current_data])
        # return pds.Series(current_data).to_frame()

    def track_scheme(scheme, time_range, output_file, wait_time=5):
        """monitor paramteres in the naming scheme and record it in the fucntion of time.
        scheme : ['x', 'y', 'SAXSx', 'voltage']
        time_range : recording time, in section
        output_file : saved in /data/ folder.
        wait_time :  interval time for data acquisition
        """
        # create a data file to save the INT data
        data_folder = "{}/data/".format(os.path.dirname(__file__))
        if os.path.exists(RE.md["experiment_alias_directory"]) == False:
            return print("The path is NOT valid. Please create the data folder.")

        INT_FILENAME = "{}/data/{}.csv".format(os.path.dirname(__file__), output_file)
        print(INT_FILENAME)
        time_range = time.time() + time_range
        while time.time() < time_range:
            temp_data = monitor_scheme(scheme=scheme)

            if os.path.isfile(INT_FILENAME):
                output_data = pds.read_csv(INT_FILENAME, index_col=0)
                output_data = output_data.append(temp_data, ignore_index=True)
                output_data.to_csv(INT_FILENAME)
            else:
                temp_data.to_csv(INT_FILENAME)
            if wait_time > 0:
                time.sleep(wait_time)


class Stage(CoordinateSystem):
    pass


class SampleStage(Stage):
    def __init__(self, name="SampleStage", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

    def _set_axes_definitions(self):
        """Internal function which defines the axes for this stage. This is kept
        as a separate function so that it can be over-ridden easily."""

        # The _axes_definitions array holds a list of dicts, each defining an axis
        self._axes_definitions = [
            {
                "name": "x",
                "motor": smx,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": "positive moves stage left/outboard (beam moves right on sample)",
            },
            {
                "name": "y",
                "motor": smy,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": "positive moves stage up (beam moves down on sample)",
            },
            {
                "name": "th",
                "motor": sth,
                "enabled": True,
                "scaling": +1.0,
                "units": "deg",
                "hint": "positive tilts clockwise (positive incident angle)",
            },
        ]


class Holder(Stage):
    """The Holder() classes are used to define bars/stages that hold one or more
    samples. This class can thus help to keep track of coordinate conversions,
    to store the positions of multiple samples, and to automate the measurement
    of multiple samples."""

    # Core methods
    ########################################

    def __init__(self, name="Holder", base=None, **kwargs):
        if base is None:
            base = get_default_stage()

        super().__init__(name=name, base=base, **kwargs)

        self._samples = {}
        self.reset_clock()

    def _set_axes_definitions(self):
        """Internal function which defines the axes for this stage. This is kept
        as a separate function so that it can be over-ridden easily."""

        # The _axes_definitions array holds a list of dicts, each defining an axis
        self._axes_definitions = [
            {
                "name": "x",
                "motor": None,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": "positive moves stage left/outboard (beam moves right on sample)",
            },
            {
                "name": "y",
                "motor": None,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": "positive moves stage up (beam moves down on sample)",
            },
            {
                "name": "th",
                "motor": None,
                "enabled": True,
                "scaling": +1.0,
                "units": "deg",
                "hint": "positive tilts clockwise (positive incident angle)",
            },
        ]

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

    # Sample management
    ########################################

    def addSample(self, sample, sample_number=None):
        """Add a sample to this holder/bar."""

        if sample_number is None:
            if len(self._samples) == 0:
                sample_number = 1
            else:
                ki = [int(key) for key in self._samples.keys()]
                sample_number = np.max(ki) + 1

        if sample_number in self._samples.keys():
            print(
                'Warning: Sample number {} is already defined on holder "{:s}". Use "replaceSample" if you are sure you want to eliminate the existing sample from the holder.'.format(
                    sample_number, self.name
                )
            )

        else:
            self._samples[sample_number] = sample

        self._samples[sample_number] = sample

        sample.set_base_stage(self)
        sample.md["holder_sample_number"] = sample_number

    def removeSample(self, sample_number):
        """Remove a particular sample from this holder/bar."""

        del self._samples[sample_number]

    def removeSamplesAll(self):
        self._samples = {}

    def replaceSample(self, sample, sample_number):
        """Replace a given sample on this holder/bar with a different sample."""

        self.removeSample(sample_number)
        self.addSample(sample, sample_number)

    def getSample(self, sample_number, verbosity=3):
        """Return the requested sample object from this holder/bar.

        One can provide an integer, in which case the corresponding sample
        (from the holder's inventory) is returned. If a string is provided,
        the closest-matching sample (by name) is returned."""

        if type(sample_number) == int:
            if sample_number not in self._samples:
                if verbosity >= 1:
                    print("Error: Sample {} not defined.".format(sample_number))
                return None

            sample_match = self._samples[sample_number]

            if verbosity >= 3:
                print("{}: {:s}".format(sample_number, sample_match.name))

            return sample_match

        elif type(sample_number) == str:
            # First search for an exact name match
            matches = 0
            sample_match = None
            sample_i_match = None
            for sample_i, sample in sorted(self._samples.items()):
                if sample.name == sample_number:
                    matches += 1
                    if sample_match is None:
                        sample_match = sample
                        sample_i_match = sample_i

            if matches == 1:
                if verbosity >= 3:
                    print("{}: {:s}".format(sample_i_match, sample_match.name))
                return sample_match

            elif matches > 1:
                if verbosity >= 2:
                    print(
                        '{:d} exact matches for "{:s}", returning sample {}: {:s}'.format(
                            matches, sample_number, sample_i_match, sample_match.name
                        )
                    )
                return sample_match

            # Try to find a 'start of name' match
            for sample_i, sample in sorted(self._samples.items()):
                if sample.name.startswith(sample_number):
                    matches += 1
                    if sample_match is None:
                        sample_match = sample
                        sample_i_match = sample_i

            if matches == 1:
                if verbosity >= 3:
                    print("Beginning-name match: {}: {:s}".format(sample_i_match, sample_match.name))
                return sample_match

            elif matches > 1:
                if verbosity >= 2:
                    print(
                        '{:d} beginning-name matches for "{:s}", returning sample {}: {:s}'.format(
                            matches, sample_number, sample_i_match, sample_match.name
                        )
                    )
                return sample_match

            # Try to find a substring match
            for sample_i, sample in sorted(self._samples.items()):
                if sample_number in sample.name:
                    matches += 1
                    if sample_match is None:
                        sample_match = sample
                        sample_i_match = sample_i

            if matches == 1:
                if verbosity >= 3:
                    print("Substring match: {}: {:s}".format(sample_i_match, sample_match.name))
                return sample_match

            elif matches > 1:
                if verbosity >= 2:
                    print(
                        '{:d} substring matches for "{:s}", returning sample {}: {:s}'.format(
                            matches, sample_number, sample_i_match, sample_match.name
                        )
                    )
                return sample_match

            if verbosity >= 1:
                print('No sample has a name matching "{:s}"'.format(sample_number))
            return None

        else:
            print('Error: Sample designation "{}" not understood.'.format(sample_number))
            return None

    import string

    def getSamples(self, range=None, verbosity=3):
        """Get the list of samples associated with this holder.

        If the optional range argument is provided (2-tuple), then only sample
        numbers within that range (inclusive) are run. If range is instead a
        string, then all samples with names that match are returned."""

        samples = []

        if range is None:
            for sample_number in sorted(self._samples):
                samples.append(self._samples[sample_number])

        elif type(range) is list:
            if type(range[0]) is int:
                if len(range) == 2:
                    start, stop = range
                    for sample_number in sorted(self._samples):
                        if sample_number >= start and sample_number <= stop:
                            samples.append(self._samples[sample_number])
                else:
                    for sample_number in sorted(self._samples):
                        for ii in range:
                            if sample_number == ii:
                                samples.append(self._samples[sample_number])

            elif type(range[0]) is str:  # For 96 well holder, format: A1, D2 ...
                for sample_number in sorted(self._samples):
                    sample_row = string.ascii_lowercase(sample_number[0])
                    sample_column = int(sample_number[1:])
                    sample_number = sample_row * 12 + sample_column
                    samples.append(self._samples[sample_number])

        elif type(range) is str:
            for sample_number, sample in sorted(self._samples.items()):
                if range in sample.name:
                    samples.append(sample)

        elif type(range) is int:
            samples.append(self._samples[range])

        else:
            if verbosity >= 1:
                print('Range argument "{}" not understood.'.format(range))

        return samples

    def listSamples(self):
        """Print a list of the current samples associated with this holder/
        bar."""

        for sample_number, sample in sorted(self._samples.items()):
            print("{}: {:s}".format(sample_number, sample.name))

    def gotoSample(self, sample_number):
        sample = self.getSample(sample_number, verbosity=0)
        sample.gotoAlignedPosition()

        return sample

    # Control methods
    ########################################
    def setTemperature(self, temperature, output_channel="1", verbosity=3):
        # if verbosity>=1:
        # print('Temperature functions not implemented in {}'.format(self.__class__.__name__))
        if output_channel == "1":
            if verbosity >= 2:
                print(
                    "  Changing temperature setpoint from {:.3f}°C  to {:.3f}°C".format(
                        caget("XF:11BM-ES{Env:01-Out:1}T-SP") - 273.15, temperature
                    )
                )
            caput("XF:11BM-ES{Env:01-Out:1}T-SP", temperature + 273.15)

        if output_channel == "2":
            if verbosity >= 2:
                print(
                    "  Changing temperature setpoint from {:.3f}°C  to {:.3f}°C".format(
                        caget("XF:11BM-ES{Env:01-Out:2}T-SP") - 273.15, temperature
                    )
                )
            caput("XF:11BM-ES{Env:01-Out:2}T-SP", temperature + 273.15)

        if output_channel == "3":
            if verbosity >= 2:
                print(
                    "  Changing temperature setpoint from {:.3f}°C  to {:.3f}°C".format(
                        caget("XF:11BM-ES{Env:01-Out:3}T-SP") - 273.15, temperature
                    )
                )
            caput("XF:11BM-ES{Env:01-Out:3}T-SP", temperature + 273.15)

        if output_channel == "4":
            if verbosity >= 2:
                print(
                    "  Changing temperature setpoint from {:.3f}°C  to {:.3f}°C".format(
                        caget("XF:11BM-ES{Env:01-Out:4}T-SP") - 273.15, temperature
                    )
                )
            caput("XF:11BM-ES{Env:01-Out:4}T-SP", temperature + 273.15)

    def temperature(self, temperature_probe="A", output_channel="1", verbosity=3):
        # if verbosity>=1:
        # print('Temperature functions not implemented in {}'.format(self.__class__.__name__))

        if temperature_probe == "A":
            current_temperature = caget("XF:11BM-ES{Env:01-Chan:A}T:C-I")
            if verbosity >= 3:
                print(
                    "  Temperature = {:.3f}°C (setpoint = {:.3f}°C)".format(
                        current_temperature,
                        self.temperature_setpoint(output_channel=output_channel) - 273.15,
                    )
                )

        if temperature_probe == "B":
            current_temperature = caget("XF:11BM-ES{Env:01-Chan:B}T:C-I")
            if verbosity >= 3:
                print(
                    "  Temperature = {:.3f}°C (setpoint = {:.3f}°C)".format(
                        current_temperature,
                        self.temperature_setpoint(output_channel=output_channel) - 273.15,
                    )
                )

        if temperature_probe == "C":
            current_temperature = caget("XF:11BM-ES{Env:01-Chan:C}T:C-I")
            if verbosity >= 3:
                print(
                    "  Temperature = {:.3f}°C (setpoint = {:.3f}°C)".format(
                        current_temperature,
                        self.temperature_setpoint(output_channel=output_channel) - 273.15,
                    )
                )

        if temperature_probe == "D":
            current_temperature = caget("XF:11BM-ES{Env:01-Chan:D}T:C-I")
            if verbosity >= 3:
                print(
                    "  Temperature = {:.3f}°C (setpoint = {:.3f}°C)".format(
                        current_temperature,
                        self.temperature_setpoint(output_channel=output_channel) - 273.15,
                    )
                )

        return current_temperature

    def temperature_setpoint(self, output_channel="1", verbosity=3):
        # if verbosity>=1:
        # print('Temperature functions not implemented in {}'.format(self.__class__.__name__))

        if output_channel == "1":
            setpoint_temperature = caget("XF:11BM-ES{Env:01-Out:1}T-SP")

        if output_channel == "2":
            setpoint_temperature = caget("XF:11BM-ES{Env:01-Out:2}T-SP")

        if output_channel == "3":
            setpoint_temperature = caget("XF:11BM-ES{Env:01-Out:3}T-SP")

        if output_channel == "4":
            setpoint_temperature = caget("XF:11BM-ES{Env:01-Out:4}T-SP")

        return setpoint_temperature

    # Action (measurement) methods
    ########################################

    def doSamples(self, range=None, verbosity=3, **md):
        """Activate the default action (typically measurement) for all the samples.

        If the optional range argument is provided (2-tuple), then only sample
        numbers within that range (inclusive) are run. If range is instead a
        string, then all samples with names that match are returned."""

        for sample in self.getSamples(range=range):
            if verbosity >= 3:
                print("Doing sample {}...".format(sample.name))
            sample.do(verbosity=verbosity, **md)

    def doTemperature(
        self,
        temperature,
        wait_time=None,
        temperature_probe="A",
        output_channel="1",
        temperature_tolerance=0.4,
        range=None,
        verbosity=3,
        poling_period=2.0,
        **md,
    ):
        # Set new temperature
        self.setTemperature(temperature, output_channel=output_channel, verbosity=verbosity)

        # Wait until we reach the temperature
        # while abs(self.temperature(verbosity=0) - temperature)>temperature_tolerance:
        while (
            abs(self.temperature(temperature_probe=temperature_probe, verbosity=0) - temperature)
            > temperature_tolerance
        ):
            if verbosity >= 3:
                print(
                    "  setpoint = {:.3f}°C, Temperature = {:.3f}°C          \r".format(
                        self.temperature_setpoint() - 273.15,
                        self.temperature(verbosity=0),
                    ),
                    end="",
                )
            time.sleep(poling_period)

        # Allow for additional equilibration at this temperature
        if wait_time is not None:
            time.sleep(wait_time)

        self.doSamples(range=range, verbosity=verbosity, **md)

    def doTemperatures(
        self,
        temperatures,
        wait_time=None,
        temperature_probe="A",
        output_channel="1",
        temperature_tolerance=0.4,
        range=None,
        verbosity=3,
        **md,
    ):
        for temperature in temperatures:
            self.doTemperature(
                temperature,
                wait_time=wait_time,
                temperature_probe=temperature_probe,
                output_channel=output_channel,
                temperature_tolerance=temperature_tolerance,
                range=range,
                verbosity=verbosity,
                **md,
            )


class PositionalHolder(Holder):
    """This class is a sample holder that is one-dimensional. E.g. a bar with a
    set of samples lined up, or a holder with a set number of slots for holding
    samples. This class thus helps to associate each sample with its position
    on the bar."""

    # Core methods
    ########################################

    def __init__(self, name="PositionalHolder", base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        self._positional_axis = "x"
        self.GaragePosition = []
        self.setPosition()

    def setPosition(self):
        # add by RL 060823
        self.position = {}
        for axis in self._positional_axis:
            self.position[axis] = self._axes[axis].origin
        # self.position _axes['x'].origin

    # Sample management
    ########################################

    def slot(self, sample_number):
        """Moves to the selected slot in the holder."""

        getattr(self, self._positional_axis + "abs")(self.get_slot_position(sample_number))

    def get_slot_position(self, slot):
        """Return the motor position for the requested slot number."""
        # This method should be over-ridden in sub-classes, so as to properly
        # implement the positioning appropriate for that holder.

        position = 0.0 + slot * 1.0

        return position

    def addSampleSlot(self, sample, slot, detector_opt="SAXS"):
        """Adds a sample to the specified "slot" (defined/numbered sample
        holding spot on this holder)."""

        self.addSample(sample, sample_number=slot)
        sample.setOrigin([self._positional_axis], [self.get_slot_position(slot)])
        sample.detector = detector_opt

    def addSampleSlotPosition(
        self,
        sample,
        slot,
        position,
        detector_opt="BOTH",
        incident_angles=None,
        transmission=1,
        exposure_time=None,
        exposure_time_WAXS=None,
        tiling=None,
    ):
        """Adds a sample to the specified "slot" (defined/numbered sample
        holding spot on this holder)."""

        self.addSample(sample, sample_number=slot)
        sample.setOrigin([self._positional_axis], [position])
        sample.detector = detector_opt
        sample.incident_angles = incident_angles
        sample.transmission = transmission
        sample.position = position
        # sample.exposure_time = exposure_time
        # sample.exposure_time_WAXS = exposure_time_WAXS
        # sample.exposure_time_MAXS = exposure_time_MAXS
        # sample.tiling = tiling

        # TODO: list sample details including:
        # name, slot, position, detector_opt, incident_angles, transmission, exposure_time_SAXS, exposure_time_WAXS, exposure_time_MAXS, tiling
        # load it in a standard format, such as xls or pandas

    def listSamplesPositions(self):
        """Print a list of the current samples associated with this holder/
        bar."""

        for sample_number, sample in self._samples.items():
            # pos = getattr(sample, self._positional_axis+'pos')(verbosity=0)
            pos = sample.origin(verbosity=0)[self._positional_axis]
            print("%s: %s (%s = %.3f)" % (str(sample_number), sample.name, self._positional_axis, pos))

    def listSamplesDetails(self):
        """Print a list of the current samples associated with this holder/
        bar."""

        for sample_number, sample in self._samples.items():
            # pos = getattr(sample, self._positional_axis+'pos')(verbosity=0)
            pos = sample.origin(verbosity=0)[self._positional_axis]
            print(
                "%s: %s (%s = %.3f) %s"
                % (
                    str(sample_number),
                    sample.name,
                    self._positional_axis,
                    pos,
                    sample.detector,
                )
            )

    def addGaragePosition(self, shelf_num, spot_num):
        """the position in garage"""
        if shelf_num not in range(1, 5) or spot_num not in range(1, 4):
            print("Out of the range in Garage (4 x 3)")

        self.GaragePosition = [shelf_num, spot_num]

    def intMeasure(self, output_file, exposure_time=1):
        for sample in self.getSamples():
            sample.gotoOrigin()
            sample.intMeasure(output_file, exposure_time=1)

    def saveSampleStates(self, output_file=None):
        """Print a list of the current samples associated with this holder/bar.

        It can be saved in the output_file under setup"""

        states = {}
        for sample_number, sample in sorted(self._samples.items()):
            states[sample_number] = sample.save_state()

        cms.samples_states = states
        if output_file is not None:
            with open(output_file, "wb") as handle:
                pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return states

    def restoreSampleStates(self, input_file=None):
        if input_file is not None:
            with open(input_file, "rb") as handle:
                cms.samples_states = pickle.load(handle)

        for sample_number, sample in sorted(self._samples.items()):
            sample.restore_state(cms.samples_states[sample_number])
            print(sample.save_state())

    def checkPositions(self):
        for sample in self.getSamples():
            sample.gotoOrigin()
            time.sleep(1)


stg = SampleStage()


def get_default_stage():
    return stg


if False:
    # For testing:
    # %run -i /opt/ipython_profiles/profile_collection/startup/94-sample.py
    sam = SampleGISAXS_Generic("testing_of_code")
    sam.mark("here")
    # sam.mark('XY_field', 'x', 'y')
    # sam.mark('specified', x=1, th=0.1)
    # sam.naming(['name', 'extra', 'clock', 'th', 'exposure_time', 'id'])
    # sam.thsetOrigin(0.5)
    # sam.marks()

    hol = CapillaryHolder(base=stg)
    hol.addSampleSlot(SampleGISAXS_Generic("test_sample_01"), 1.0)
    hol.addSampleSlot(SampleGISAXS_Generic("test_sample_02"), 3.0)
    hol.addSampleSlot(SampleGISAXS_Generic("test_sample_03"), 5.0)

    sam = hol.getSample(1)
