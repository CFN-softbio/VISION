import shutil
import sys, subprocess, os, time
from PyQt5 import QtCore, QtGui, QtWidgets
from UI import Ui_MainWindow
# from backend import backend

def is_xdotool_installed():
    return shutil.which('xdotool') is not None

class interface:

    def __init__(self, isInterfacing=False):

        self.interfacing = isInterfacing
        self.window_base = "VISION"
        self.window_Op = "IPython"
        self.window_Ana= self.window_base+"-Ana"

        if(isInterfacing):
            #Default numerical values
            self.qramp = 5

            os.system('./open_bsui {}'.format(self.window_Op)) 
            os.system('./open_ana {}'.format(self.window_Ana))


            #Initialize a bluesky placeholder sample to interact with bluesky functions
            # self.sam = Sample("")
            # self.executeCommand(" sam = SampleGISAXS('test')", window_name=self.window_Op)
            # self.executeCommand(" sam.setOrigin(['x','y','th'])", window_name=self.window_Op)
            # self.executeCommand(" detselect([pilatus2M])", window_name=self.window_Op)
            # self.executeCommand(" cms.modeMeasurement()", window_name=self.window_Op)
            # print("Default: sam = Sample('test')")
            # print("Default: detselect([pilatus2M]")
            # #print("Default: cms.modeMeasurement()")
            # #self.executeCommand("$SampleList = []")

    # def getValue(self, key):

    #     if(key == "temperature"):
    #         return self.sam.temperature(temperature_probe='B')
    #     elif(key == "sample"):
    #         return self.sam.name
    #     elif(key == "humidity"):
    #         return 25.00 #self.sam.humidity()
    #     elif(key == "position"):
    #         x = self.sam.xr()
    #         y = self.sam.yr()
    #         pos = "({:.2f}".format(x) + " , " + "{:.2f})".format(y)
    #         return pos

    # def measure(self, exposure_time):
    #     self.sam.measure(exposure_time)
    #     print("\nDone measuring\n")

    # def changeTemperature(self, temperature, nramp=None):
    #     if(nramp == None): #If we do a quick ramp which doesn't have a specificed ramp
    #         caput('XF:11BM-ES{Env:01-Out:1}Val:Ramp-SP', self.qramp)
    #     else: #If we have a designated numerical ramp
            
    #         if(nramp[0] == "min"):
    #             caput('XF:11BM-ES{Env:01-Out:1}Val:Ramp-SP', nramp[1])
    #         else:
    #             caput('XF:11BM-ES{Env:01-Out:1}Val:Ramp-SP', nramp[1] / 60)

    #     self.sam.setTemperature(temperature)


    # def doTemperatureScan(self, temperature, exposure_time, period_sec):
    #     self.sam.setTemperature(temperature)
    #     while self.sam.temperature(temperature_probe='B') < temperature - 1:
    #         self.sam.measure(exposure_time)
    #         print('sleeping..')
    #         self.sam.sleep(period_sec-exposure_time)

    # def setIncidentAngle(self, angle):
    #     self.sam.thabs(angle)

    # def setSampleName(self, sample):
    #     self.sam.name = str(sample)

    def executeCommand(self, command, window_name="TEST", press_enter=True):
        if 1:
            #print("[CLI] ", command, "\n")
            #exec(command[1:], globals())
            #with open("exe_cmd_bsui.txt", "a") as file:
            #    file.write(command+ "\n")
            self.window_inject(text=command, window_name=window_name, press_enter=press_enter)

    def window_inject(self, text='command', window_name="TEST", press_enter=True, remove_indents=False):
        # original_wid = int(subprocess.check_output('xdotool getwindowfocus', shell=True))
        if remove_indents:
            # TODO: Investigate. Using indentations can sometimes lead to problems at beamline.
            text = text.replace("    ", "")

        # Should have been removed on the back-end already
        # text = text.replace("```python", "")
        # text = text.replace("```", "")

        #os.system('./open_window {}'.format(window_name))
        #wid = int(subprocess.check_output('xdotool getwindowfocus', shell=True))
        ### Try using tab
        # try:
        #     wid = int(subprocess.check_output('xdotool search --name "{}"'.format(window_name), shell=True))
        # except:
        #     wid = int(subprocess.check_output('xdotool search --name "{}"'.format(window_name[0:-5]), shell=True))
        #     os.system('xdotool windowactivate {}'.format(wid))
        #     os.system('xdotool key ctrl+Page_Down')

        if not is_xdotool_installed():
            print("Warning: xdotool is not installed. Cannot inject text.")
            print("To install: sudo apt-get install xdotool (Ubuntu/Debian)")
            return False

        try:
            wid = int(subprocess.check_output('xdotool search --name "{}"'.format(window_name), shell=True))
            print("Injection to {} {}".format(window_name, wid))
            os.system('xdotool windowactivate {}'.format(wid))
            os.system('xdotool keyup --window {} a type "{}"'.format(wid, text))
            if press_enter:
                os.system('xdotool key --window {} KP_Enter')
                time.sleep(0.05)
            return True
        except Exception as e:
            print(f"Error during text injection: {e}")
            return False

        # os.system('xdotool windowactivate {}'.format(original_wid))


    def analyze(self, protocol):
        fn = '/nsls2/data3/cms/legacy/xf11bm/data/2024_2/EGomez/waxs/analysis/runXS_WAXS_0.py'
        with open(fn, 'r') as file:
            command = file.read()

        if(interfacing):
            print("Running file {}".format(fn))
            #print("[CLI] ", command, "\n\n")
            exec(command[0:], globals())


#Interface variable determines whether bluesky is linked (True) or not linked(False)
interfacing = 1

#Backend

# print("\n\n LOADING BACKEND \n\n")

# backend = backend(isInterfacing=interfacing)

#Interface

print("\n\n LOADING INTERFACE with interfacing = {} \n\n".format(interfacing))

interfaceManager = interface(isInterfacing=interfacing)

#UI

print("\n\n LOADING UI \n\n")

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow(isInterfacing=interfacing)

#Initializing GUI
# ui.setupUi(MainWindow)
# MainWindow.show()

#print("\n\n SHOWING UI \n\n")

ui.show()

#Connecting everything to each other and interface class
# ui.setBackendModel(backend)
ui.setInterface(interfaceManager)
# backend.setUI(ui)
# backend.setInterface(interfaceManager)

sys.exit(app.exec_())