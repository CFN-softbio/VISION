print(f'Loading {__file__}')

Table_front_to_end = 900.  #in mmm
Table_rear_to_near = 1200.

def _movr_table_pitch(del_deg):
    """Moves the pitch of the mirror support by specified angle in [mrad]"""
    del_rad = del_deg * np.pi/180

    del_mm = 0.5 * Table_front_to_end * del_rad
    yield from bps.movr(TABLEr, -del_mm, TABLEn, -del_mm, TABLEd, del_mm)


def movr_table_pitch(del_deg):
    RE(_movr_table_pitch(del_deg))


def _movr_table_roll(del_deg):
    """Moves the yaw of the mirror support by specified angle in [mrad]"""
    del_rad = del_deg * np.pi/180
    del_mm = 0.5 * Table_rear_to_near * del_rad
    yield from bps.movr(TABLEr, -del_mm, TABLEn, del_mm)

def movr_table_roll(del_deg):
    RE(_movr_table_roll(del_deg))


def _movr_table_y(del_mm):
    """Moves the table support vertically by specified distance in [mm]"""
    yield from bps.movr(TABLEr, del_mm, TABLEn, del_mm, TABLEd, del_mm)

def movr_table_y(del_mm):
    RE(_movr_table_y(del_mm))


def _mov_table_y(del_mm):
    """Moves the table support vertically by specified distance in [mm]"""
    yield from bps.mov(TABLEr, del_mm, TABLEn, del_mm, TABLEd, del_mm)

def mov_table_y(del_mm):
    RE(_mov_table_y(del_mm))

def get_table_angle():
    '''calculate the incident angle of the toroidal mirror in [mrad]'''
    Tr = TABLEr.position()
    Tn = TABLEn.position()
    Td = TABLEd.position()

    Tf = (Tn + Td)/2

    roll_rad = np.arctan((Tn-Tr)/Table_rear_to_near)
    pitch_rad = np.arctan((Td-Tf)/Table_front_to_end)
    print('The modular table')
    print(f"The pitch angle  is {pitch_rad*1e3: .4f} mrad, equal to {pitch_rad/np.pi*180: .4f} deg.")
    print(f"The roll angle  is {roll_rad*1e3: .4f} mrad, equal to {roll_rad/np.pi*180: .4f} deg.")
    return roll_rad, pitch_rad


def wTABLE():
    print("Table near = {}".format(TABLEn.position))
    print("Table rear = {}".format(TABLEr.position))
    print("Table down = {}".format(TABLEd.position))
    get_table_angle()


class ModularTable_Generic(CoordinateSystem):
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

        self.Table_front_to_end = 900.  #in mmm
        self.Table_rear_to_near = 1200.

    def _set_axes_definitions(self):
        """Internal function which defines the axes for this stage. This is kept
        as a separate function so that it can be over-ridden easily."""

        # The _axes_definitions array holds a list of dicts, each defining an axis
        self._axes_definitions = [
            {
                "name": "n",
                "motor": TABLEn,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": None,
            },
            {
                "name": "r",
                "motor": TABLEr,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": None,
            },
            {
                "name": "d",
                "motor": TABLEd,
                "enabled": True,
                "scaling": +1.0,
                "units": "mm",
                "hint": None,
            },
        ]

    def _movr_table_pitch(self, del_deg):
        """Moves the pitch of the mirror support by specified angle in [mrad]"""
        del_rad = del_deg * np.pi/180

        del_mm = 0.5 * Table_front_to_end * del_rad
        yield from bps.movr(TABLEr, -del_mm, TABLEn, -del_mm, TABLEd, del_mm)


    def movr_table_pitch(self, del_deg):
        RE(self._movr_table_pitch(del_deg))


    def _movr_table_roll(self, del_deg):
        """Moves the yaw of the mirror support by specified angle in [mrad]"""
        del_rad = del_deg * np.pi/180
        del_mm = 0.5 * Table_rear_to_near * del_rad
        yield from bps.movr(TABLEr, -del_mm, TABLEn, del_mm)

    def movr_table_roll(self, del_deg):
        RE(self._movr_table_roll(del_deg))


    def _movr_table_y(self, del_mm):
        """Moves the table support vertically by specified distance in [mm]"""
        yield from bps.movr(TABLEr, del_mm, TABLEn, del_mm, TABLEd, del_mm)

    def movr_table_y(self, del_mm):
        RE(self._movr_table_y(del_mm))

    def _gotoOrigin(self):

        yield from _mov_table_y()

    def _mov_table_y(self, del_mm):
        """Moves the table support vertically by specified distance in [mm]"""
        yield from bps.mov(TABLEr, del_mm, TABLEn, del_mm, TABLEd, del_mm)

    def _gotoOrigin(self):
        """Moves the table support vertically by specified distance in [mm]"""
        yield from bps.mov(TABLEr, self._axes['r'].origin, TABLEn, self._axes['n'].origin, TABLEd, self._axes['d'].origin)


    # def mov_table_y(self, del_mm):
    #     RE(self._mov_table_y(del_mm))

    def get_table_angle(self, verbosity=3):
        '''calculate the incident angle of the toroidal mirror in [mrad]'''
        # Tr = TABLEr.position()
        # Tn = TABLEn.position()
        # Td = TABLEd.position()
        Tr = self.rpos()
        Tn = self.npos()
        Td = self.dpos()

        Tf = (Tn + Td)/2

        roll_rad = np.arctan((Tn-Tr)/Table_rear_to_near)
        pitch_rad = np.arctan((Td-Tf)/Table_front_to_end)
        if verbosity>=3:
            print('The angle of the modular table ::: ')
            print(f"The pitch angle  is {pitch_rad*1e3: .4f} mrad, equal to {pitch_rad/np.pi*180: .4f} deg.")
            print(f"The roll angle  is {roll_rad*1e3: .4f} mrad, equal to {roll_rad/np.pi*180: .4f} deg.")
        return roll_rad, pitch_rad

    def get_table_height(self, verbosity=3):
        '''calculate the incident angle of the toroidal mirror in [mrad]'''
        # Tr = TABLEr.position()
        # Tn = TABLEn.position()
        # Td = TABLEd.position()
        Tr = self.rpos()
        Tn = self.npos()
        Td = self.dpos()
        height_ave =(Tr + (Tn + Td)/2)/2
        if verbosity>=3:
            print('The height of the modular table is {}mm.'.format(height_ave))
        return height_ave


class ModularTable(ModularTable_Generic):
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

        super().__init__(name=name, base=base)

        self.nsetOrigin(69.3571)
        self.dsetOrigin(67.7861)
        self.rsetOrigin(71.8701)



tb1 = ModularTable('tb1')


def wTABLE():
    print("Table near = {}".format(TABLEn.position))
    print("Table rear = {}".format(TABLEr.position))
    print("Table down = {}".format(TABLEd.position))
    get_table_angle()