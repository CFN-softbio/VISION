import time

# import select
import re
from ophyd import Device


# from ophyd.areadetector.plugins import ImagePlugin, TIFFPlugin

# class ImagePluginCustom(ImagePlugin):
# @property
# def image(self):
# return 0.0


# from ophyd.areadetector.filestore_mixins import FileStoreTIFFIterativeWrite
# class TIFFPluginWithFileStore(TIFFPlugin, FileStoreTIFFIterativeWrite):
# pass


class PhotonicSciences_CMS(Device):
    # image = Cpt(ImagePluginCustom, 'image1:')
    # tiff = Cpt(TIFFPluginWithFileStore,
    # suffix='TIFF1:',
    # write_path_template='/nsls2/xf11bm/waxsdet/%Y/%m/%d/')

    def __init__(
        self,
        prefix="",
        *args,
        read_attrs=None,
        configuration_attrs=None,
        name="PhotonicSciences_CMS",
        parent=None,
        **kwargs,
    ):
        super().__init__(
            prefix=prefix,
            *args,
            read_attrs=read_attrs,
            configuration_attrs=configuration_attrs,
            name=name,
            parent=parent,
            **kwargs,
        )

        self.file_path = "/nsls2/xf11bm/waxsdet/images"

        self.temperature_re = re.compile(".+CCD temperature (.+) deg.+")
        self.status_re = re.compile(".+detector status=(.+)\.")
        self.exposure_re = re.compile(".+exposure time set to (.+) second.+")

        self.exposure_time = 1
        self.max_wait_time = 10.0

        self.connect_socket()
        self.detector_binning(2, 2)

    def setExposureTime(self, exposure_time, verbosity=3):
        self.detector_set_exposure_time(exposure_time, verbosity=verbosity)

    # Ophyd methods
    ########################################
    def stage(self, *args, poling_period=0.1, **kwargs):
        # Give detector a chance to get ready
        start_time = time.time()
        while (not self.detector_is_ready(verbosity=0)) and (time.time() - start_time) < (self.max_wait_time):
            time.sleep(poling_period)

        if not self.detector_is_ready():
            print("ERROR: {} is not responding.".format(self.name))

        return super().stage(*args, **kwargs)

    def trigger(self):
        self.detector_measure()

        return super().trigger()

    # def unstage(self):
    # return super().unstage()

    # Essential socket interaction
    ########################################

    def connect_socket(self):
        self.server_address = "10.11.129.11 "
        # self.server_address= '10.11.129.12 ' #The backup computer

        # self.server_IP = '10.11.129.11'
        self.server_port = 27015

        import socket

        # self.sock = socket.socket()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(self.server_address)
        self.sock.connect((self.server_address, self.server_port))

        self.sock.settimeout(0.5)

    def disconnect_socket(self):
        self.sock.close()

    def send_socket(self, msg):
        # self.sock.send(chr(13).encode('ascii', 'ignore')) # Carriage return
        self.sock.send(msg.encode("ascii", "ignore"))
        # self.sock.send(msg.encode('utf-8'))

    def send_get_reply(self, msg, verbosity=3):
        # self.send_socket('\r')
        self.send_socket(msg)

        time.sleep(0.5)

        return self.read_socket(verbosity=verbosity)

    def read_socket(self, timeout_s=3, verbosity=3):
        start_time = time.time()
        terminator = chr(0x18)

        # Look for the response
        amount_received = 0
        amount_cutoff = 5000

        txt = ""
        msg_received = ""

        while terminator not in txt and time.time() - start_time < timeout_s and amount_received < amount_cutoff:
            try:
                data = self.sock.recv(1)
            except:
                break

            amount_received += len(data)
            txt = data.decode("ascii")

            msg_received += txt

        msg_received = msg_received.replace(terminator, "")

        if time.time() - start_time > timeout_s:
            if verbosity >= 1:
                print("Read timeout after {:.1f} s.".format(time.time() - start_time))
            return ""

        else:
            if verbosity >= 2:
                print(msg_received)
            return msg_received

    # Detector commands
    ########################################

    def detector_status(self, verbosity=3):
        reply = self.send_get_reply("ST", verbosity=verbosity)
        m = self.status_re.match(reply)
        if m:
            code = int(m.groups()[0])
            return code
        else:
            return None

    def detector_is_ready(self, verbosity=3):
        return self.detector_status(verbosity=verbosity) == 1

    def detector_abort(self, verbosity=3):
        return self.send_get_reply("AB", verbosity=verbosity)

    def detector_binning(self, binx=2, biny=2, verbosity=3):
        if binx == None or biny == None:
            binx = 2
            biny = 2

        return self.send_get_reply("BN {:d} {:d}".format(binx, biny), verbosity=verbosity)

    def detector_trigger(self, verbosity=3):
        return self.send_get_reply("TR", verbosity=verbosity)

    def detector_temperature(self, verbosity=3):
        reply = self.send_get_reply("T?", verbosity=verbosity)
        m = self.temperature_re.match(reply)
        if m:
            T = float(m.groups()[0])
            return T

        else:
            return reply

    def detector_set_exposure_time(self, exposure_time, verbosity=3):
        self.exposure_time = exposure_time
        reply = self.send_get_reply("EX {:10.2f}".format(exposure_time), verbosity=verbosity)

        return reply

    # def detector_get_exposure_time(self, verbosity=3):

    # reply = self.send_get_reply('EX', verbosity=verbosity)
    # m = self.exposure_re.match(reply)
    # if m:
    # exposure_time = float(m.groups()[0])
    # return exposure_time

    # else:
    # return reply

    def detector_expose(self, verbosity=3):
        return self.send_get_reply("GO", verbosity=verbosity)

    def detector_save(self, filename, verbosity=3):
        reply = self.send_get_reply("SV {:s}".format(filename), verbosity=verbosity)

        return reply

    def detector_measure(self, exposure_time=None, savename="_current", verbosity=3, poling_period=0.1):
        if exposure_time is not None:
            self.detector_set_exposure_time(exposure_time)
            self.exposure_time = exposure_time
        else:
            exposure_time = self.exposure_time

        self.detector_expose()

        if verbosity >= 2:
            start_time = time.time()
            time.sleep(0.2)
            while (not self.detector_is_ready(verbosity=0)) and (time.time() - start_time) < (
                exposure_time + self.max_wait_time
            ):
                percentage = 100 * (time.time() - start_time) / exposure_time
                print(
                    "Exposing {:6.2f} s  ({:3.0f}%)      \r".format((time.time() - start_time), percentage),
                    end="",
                )
                time.sleep(poling_period)
        else:
            time.sleep(exposure_time + 0.2)

        self.file_name = savename
        self.detector_save(savename, verbosity=verbosity)


# psccd = PhotonicSciences_CMS()
# psccd = None

# psccd.detector_is_ready()
# psccd.detector_temperature()
