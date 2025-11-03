from nslsii.devices import TwoButtonShutter


class TwoButtonShutterNC(TwoButtonShutter):
    def stop(self, *args):
        ...


class TriState(Device):
    full = Cpt(TwoButtonShutterNC, "V:1}")
    soft = Cpt(TwoButtonShutterNC, "V:1_Soft}")

    def set(self, value):
        if value == "Open":
            return self.full.set("Open")  # & self.soft.set('Open')
        elif value == "Soft":
            return self.soft.set("Open") & self.full.set("Close")
        elif value == "Close":
            return self.full.set("Close") & self.soft.set("Close")
        else:
            raise ValueError("value must be in {'Open', 'Close', 'Soft'}")


# class TriState(Device):
# full = Cpt(TwoButtonShutterNC, 'V:1}')
# soft = Cpt(TwoButtonShutterNC, 'V:1_Soft}')
# def set(self, value):
# if value == 'Open':
# return self.full.set('Open') #& self.soft.set('Open')
# elif value == 'Soft':
# return self.soft.set('Open') & self.full.set('Not Open')
# elif value == 'Not Open':
# return self.full.set('Not Open') & self.soft.set('Not Open')
# else:
# raise ValueError("value must be in {'Open', 'Not Open', 'Soft'}")


def tri_plan(tri, value):
    if value == "Open":
        yield from bps.mov(tri.full, "Open")
    elif value == "Soft":
        yield from bps.mov(tri.full, "Close")
        yield from bps.mov(tri.soft, "Open")
    elif value == "Close":
        yield from bps.mov(tri.full, "Close")
        yield from bps.mov(tri.soft, "Close")


# RE(tri_plan(dev_tri, 'Open'))

# dev_tri = TriState('XF:11BMB-VA{Chm:Smpl-V', name='dev')
# dev_tri.read()
# %mov dev_tri 'Soft'

"""
class Foo:
    @proerty
    def bob(self):
        self._n += 1
        return f'bob {self._n}
class Foo:
    @proerty
    def bob(self):
        self._n += 1
        return f'bob {self._n}'
    def __init__(self):
        self._n = 0


class Foo:
    @property
    def bob(self):
        self._n += 1
        return f'bob {self._n}'
    def __init__(self):
        self._n = 0
foo = Foo()
foo.bob
foo.bob
foo.bob
foo.bob
class Foo:
    @property
    def bob(self):
        self._n += 1
        return f'bob {self._n}'
    @bob.setter
    def bob(self. val):
        self._n = val

    def __init__(self):
        self._n = 0
class Foo:
    @property
    def bob(self):
        self._n += 1
        return f'bob {self._n}'
    @bob.setter
    def bob(self, val):
        self._n = val

    def __init__(self):
        self._n = 0
foo = Foo()
foo.bob
foo.bob
foo.bob = 0
foo.bob
class Foo:
    @property
    def bob(self):
        self._n += 1
        return f'bob {self._n}'
    def __init__(self):
        self._n = 0
foo = Foo()
foo.bob = 5
dev_tri.component_names
dev_tri.read_attrs
dev_tri.full.component_names
??dev_tri.read
bps.mov
? bps.mov
bps.sleep
bps.mv
bps.mov
%mov
%mv
%wa
%wa
class TriState(Device):
    full = Cpt(TwoButtonShutter, 'V:1}')
    soft = Cpt(TwoButtonShutter, 'V:1_Soft}')
    def set(self, value):
        if value == 'Open':
            return self.full.set('Open') & self.soft.set('Open')
        elif value == 'Soft':
            return self.full.set('Close') & self.soft.set('Open')
        elif value == 'Close':
            return self.full.set('Close') & self.soft.set('Close')
        else:
            raise ValueError("value must be in {'Open', 'Close', 'Soft'}")
dev_tri = TriState('XF:11BMB-VA{Chm:Smpl-V', name='dev')
dev_tri.set('Close')
dev_tri.set('Open')
dev_tri.set('Soft')

#dev = TwoButtonShutter('XF:11BMB-VA{Chm:Smpl-VV:1}', name='Smpl_pump')
#dev.set('Close')
#dev.set('Open')
#dev_soft = TwoButtonShutter('XF:11BMB-VA{Chm:Smpl-VV:1_soft}', name='Smpl_pump_soft')
#dev_soft.set('Open')
#dev_soft = TwoButtonShutter('XF:11BMB-VA{Chm:Smpl-VV:1_Soft}', name='Smpl_pump_soft')
#dev_soft.set('Open')
#dev_soft.set('Close')
#dev_soft.set('Open')
#dev.set('Close')
#dev_soft.set('Close')
#dev.set('Open')
#dev_soft.set('Close')
"""
