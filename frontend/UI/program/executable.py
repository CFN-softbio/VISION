import shutil
import sys, subprocess, os, time
from PyQt5 import QtWidgets  #QtCore, QtGui, 
from UI import Ui_MainWindow

def is_xdotool_installed():
    return shutil.which('xdotool') is not None

class interface:

    def __init__(self, isInterfacing=False):

        self.interfacing = isInterfacing
        self.window_base = "VISION"
        self.window_Op = "IPython"
        self.window_Ana= self.window_base+"-Ana"

        if(isInterfacing):
            os.system('./open_bsui {}'.format(self.window_Op)) 
            os.system('./open_ana {}'.format(self.window_Ana))

    def executeCommand(self, command, window_name="TEST", press_enter=True):
        self.window_inject(text=command, window_name=window_name, press_enter=press_enter)

    def window_inject(self, text='command', window_name="TEST", press_enter=True, remove_indents=False):
        # original_wid = int(subprocess.check_output('xdotool getwindowfocus', shell=True))
        if remove_indents:
            # TODO: Investigate. Using indentations can sometimes lead to problems at beamline.
            text = text.replace("    ", "")

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

#### Backend: on HAL

#### Interface

print("\n\n LOADING INTERFACE with interfacing = {} \n\n".format(interfacing))
interfaceManager = interface(isInterfacing=interfacing)

#### UI

print("\n\n LOADING UI \n\n")

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow(isInterfacing=interfacing)
ui.show()
ui.setInterface(interfaceManager)


sys.exit(app.exec_())