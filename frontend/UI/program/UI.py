from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import (
    QMainWindow, QDialog, QPushButton, QLineEdit, QListWidget, QLabel, QTextEdit,
    QComboBox, QCheckBox, QTableWidget, QTableWidgetItem
)
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                             QProgressBar)          # additional Qt widgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QHeaderView
from PyQt5.QtGui     import QMovie

import requests
import json
from datetime import datetime, timezone

import faulthandler, signal, traceback
faulthandler.enable()

#Sound Recording and Transcriptions
# import sounddevice as sd
# import torch
import numpy as np

import sys, os, re
import matplotlib.pyplot as plt 

from scipy.io.wavfile import write
import wave
import pandas as pd
from datetime import datetime
import html
import ast
import numpy as np

# Don't know if this path contains special credentials, if it does we should move them to ENV variables
# rather than using dynamic path changes. For now, you can uncomment this and comment the relative import line after.
# sys.path.insert(0, '/nsls2/data/cms/legacy/xf11bm/data/2024_2/beamline/PTA/test/S3_test')
# from send_to_hal import send_audio, send_audio_file, send_hal, receive

from S3_test.send_to_hal import send_audio, send_audio_file, send_hal, receive, send_hal_async, send_audio_async, send_audio_file_async, cleanup_executor
from S3_test.qt_async_worker import run_qt_async, cleanup_qt_async

# one-way queue object (avoid waiting for reply)
from S3_test.send_to_hal import q as _hal_queue

from auth_dialog import AuthDialog

import time
import sounddevice
import pyaudio as pa

from PyQt5.QtCore import QMetaObject, Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5 import QtWidgets

import sys
import traceback


client_id = _hal_queue.client_id
print(f"[UI] Client ID: {client_id}")


def excepthook_ui(exc_type, exc_value, exc_tb):
    print("\n=== Unhandled Exception (custom excepthook_ui) ===", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    sys.stderr.flush()

sys.excepthook = excepthook_ui


class RecordingThread(QThread):
    stopped = False
    sig_started = pyqtSignal()
    sig_stopped = pyqtSignal()
    sig_transcription = pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self) -> None:
        audio = pa.PyAudio()
        frames = []
        device_index = 8
        channels = 1
        rate = 44100 #48000
        chuck = 1024
        print("=== [PyAudio] device_index selected={}".format(device_index))

        stream = audio.open(
            format = pa.paInt16,
            channels = channels,
            rate = rate,
            input = True,
            input_device_index=device_index,
            frames_per_buffer = chuck
            
        )

        self.stopped = False
        self.sig_started.emit()

        while not self.stopped:
            data = stream.read(chuck, exception_on_overflow=False)
            frames.append(data)

        stream.close()

        if 1: #For debugging audio
            # Convert byte data to numpy array
            audio_bytes = b''.join(frames)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

            # Compute intensity (absolute amplitude)
            intensity = np.abs(audio_np)

            # Create time axis in seconds
            time = np.arange(len(audio_np)) / rate

            # Plot intensity
            plt.figure(figsize=(10, 4))
            plt.plot(time, intensity)
            plt.xlabel('Time [s]')
            plt.ylabel('Intensity (|Amplitude|)')
            plt.title('Audio Intensity Over Time')
            plt.tight_layout()
            # plt.show()
            plt.savefig('audio_plot.png')


        self.sig_stopped.emit()

        exists = True
        i = 1
        while exists:
            if os.path.exists(f"recording{i}.wav"):
                i += 1
            else:
                exists = False

        wf = wave.open("output.wav", 'wb')

        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pa.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        self.sig_transcription.emit()

    @pyqtSlot()
    def stop(self):
        self.stopped = True

# === SimStreamThread helper ===

class SimStreamThread(QThread):
    """
    Background listener that receives simulation-progress messages
    from the backend via the existing S3 bus (send_to_hal.receive).
    Now emits PV name, elapsed time, actual timestamp, and value for each PV event.
    """
    sig_pv        = pyqtSignal(str, float, float, object)   # pvname, Δt, t, value
    sig_error     = pyqtSignal(str)                         # error message
    sig_completed = pyqtSignal(float)                       # duration (s)
    sig_running   = pyqtSignal()                            # status:running

    def __init__(self, sim_id):
        super().__init__()
        self.sim_id = sim_id
        self._running = True
        self.backend_t0 = None          # first PV timestamp

    def stop(self):
        self._running = False

    def run(self):
        """
        Main loop: listens for backend simulation events and emits signals for UI update.
        Handles only modern PV event structure ('p' type, with elapsed_time and delta_time).
        """
        from S3_test.send_to_hal import receive
        while self._running:
            batch = receive()
            if not batch:
                continue

            # --- normalise backend reply ------------------------------------
            import numpy as np

            # 1) single dict → wrap in list
            if isinstance(batch, dict):
                batch = [batch]

            # 2) ndarray whose elements are python objects (dtype=object)
            elif isinstance(batch, np.ndarray):
                if batch.dtype == object:
                    batch = list(batch)
                else:            # numeric array → not part of sim protocol
                    continue

            # 3) list that may itself contain ndarrays of dicts → flatten
            if isinstance(batch, list):
                flat = []
                for item in batch:
                    if isinstance(item, np.ndarray) and item.dtype == object:
                        flat.extend(list(item))
                    else:
                        flat.append(item)
                batch = flat

            if not batch:
                continue

            for msg in batch:
                if not isinstance(msg, dict):
                    continue
                if msg.get("sim_id") != self.sim_id:
                    continue

                mtype = msg.get("type")
                print(f"[UI-SIM-DBG] {mtype.upper()} {msg}")

                # Only handle new PV event structure: type == "p"
                if mtype == "pv":
                    elapsed_time = msg.get("elapsed_time")
                    delta_time = msg.get("delta_time")
                    value = msg.get("value")

                    self.sig_pv.emit(
                        msg["pvname"], delta_time, elapsed_time, value
                    )

                elif mtype == "error":
                    self.sig_error.emit(msg.get("message", ""))

                elif mtype == "status":
                    state = msg.get("state")
                    if state == "running":
                        self.sig_running.emit()
                    elif state == "completed":
                        duration = msg.get("duration")
                        if duration is None and self.backend_t0 is not None and msg.get("timestamp"):
                            duration = msg["timestamp"] - self.backend_t0
                        self.sig_completed.emit(duration or 0)
                        self._running = False
                        break

# === SimStatusDialog: floating visual status window ===

class SimStatusDialog(QtWidgets.QDialog):
    """Floating window that visualises simulation progress."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Simulation status")
        self.resize(520, 420)

        lay = QVBoxLayout(self)

        self.lbl_status = QtWidgets.QLabel("Starting simulator…")
        lay.addWidget(self.lbl_status)

        # Spinner as animated gif (for simulation progress)
        self.spinner = QtWidgets.QLabel()
        self.spinner.setAlignment(Qt.AlignCenter)
        gif_path = os.path.join(os.path.dirname(__file__), "spinner_transp.gif")
        self.movie   = QMovie(gif_path)
        self.spinner.setMovie(self.movie)
        self.movie.start()
        lay.addWidget(self.spinner)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["PV", "Time (s)", "Δt (s)", "Value"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.table)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(0, 300)

        self.txt_error = QtWidgets.QTextEdit()
        self.txt_error.setReadOnly(True)
        self.txt_error.setStyleSheet("color:red;")
        self.txt_error.hide()  # Hide initially, only show when errors occur
        lay.addWidget(self.txt_error)

        # AI evaluation button and text area
        self.btn_evaluate = QtWidgets.QPushButton("Request AI Evaluation")
        self.btn_evaluate.clicked.connect(self._on_evaluate)
        self.btn_evaluate.hide()  # Hide initially, show only when simulation completes
        lay.addWidget(self.btn_evaluate)

        self.txt_evaluation = QtWidgets.QTextEdit()
        self.txt_evaluation.setReadOnly(True)
        self.txt_evaluation.setPlaceholderText("AI evaluation will appear here after clicking 'Request AI Evaluation'")
        self.txt_evaluation.hide()  # Hide initially
        lay.addWidget(self.txt_evaluation)

        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.clicked.connect(self._on_close)
        self.btn_close.setEnabled(False)
        lay.addWidget(self.btn_close)

    # Add PV row to the simulation window.
    # Each timing value in a separate column: elapsed_time and delta_time.
    def add_pv(self, pv, val, elapsed_time=None, delta_time=None):
        """
        Add a row to the PV table with separate columns for time since start and delta to previous
        """
        print(f"add_pv, delta_time={delta_time}, elapsed_time={elapsed_time}")
        r = self.table.rowCount()
        self.table.insertRow(r)

        # Show numbers without " s" and with empty string if None
        time_str = f"{elapsed_time:.2f}" if elapsed_time is not None and elapsed_time != "" else ""
        delta_str = f"{delta_time:.2f}" if delta_time is not None and delta_time != "" else ""

        self.table.setItem(r, 0, QTableWidgetItem(str(pv)))
        self.table.setItem(r, 1, QTableWidgetItem(str(time_str)))
        self.table.setItem(r, 2, QTableWidgetItem(str(delta_str)))
        self.table.setItem(r, 3, QTableWidgetItem(str(val)))
        self.table.scrollToBottom()

    def _on_close(self):
        # notify main window so it can stop the background thread
        if hasattr(self.parent(), "_sim_dialog_closed"):
            self.parent()._sim_dialog_closed()
        self.accept()

    def set_running(self):
        self.lbl_status.setText("Running…")

    def set_error(self, msg):
        self.lbl_status.setText("Error")
        self.txt_error.show()  # Show error box when error occurred
        self.txt_error.insertPlainText(msg + '\n')     # QTextEdit has no appendPlainText
        self.txt_error.moveCursor(QtGui.QTextCursor.End)
        self._finish_spinner()

    def set_completed(self, duration):
        self.lbl_status.setText(f"Completed in {duration:.1f} s")
        self._finish_spinner()
        # Show and enable AI evaluation button when simulation completes
        self.btn_evaluate.show()
        self.btn_evaluate.setEnabled(True)

    # ensure backend-abort is triggered even if the user closes the
    # dialog with the window manager instead of the "Close" push-button
    def closeEvent(self, event):
        """
        Re-route the Qt close event to the same handler that is used when
        the dedicated "Close" button is pressed.
        """
        self._on_close()          # sends abort if still running
        # let Qt actually close the window
        event.accept()

    def _on_evaluate(self):
        """Request AI evaluation of the simulation results"""
        if hasattr(self.parent(), '_request_sim_evaluation'):
            self.btn_evaluate.setEnabled(False)
            self.btn_evaluate.setText("Requesting evaluation...")
            self.parent()._request_sim_evaluation()
        
    def show_evaluation(self, evaluation_text):
        """Display the AI evaluation result"""
        self.txt_evaluation.show()
        self.txt_evaluation.setPlainText(evaluation_text)
        self.btn_evaluate.setText("Request AI Evaluation")
        self.btn_evaluate.setEnabled(True)

    # internal
    def _finish_spinner(self):
        self.movie.stop()
        self.spinner.setText("✓")         # show green check mark
        self.spinner.setStyleSheet("font-size:32px;color:green;")
        self.btn_close.setEnabled(True)


def fetch_pv_values(pv_names, debug=False):
    """
    Fetch current values for a list of PV names from the EPICS archiver.
    
    Args:
        pv_names: List of PV names to fetch
        debug: If True, returns dummy values (5.0) for all PVs
        
    Returns:
        Dictionary mapping PV names to their current values
    """
    print(f"[UI] Fetching values for {len(pv_names)} PVs", pv_names)
    live_pvs_and_values = {}
    
    if debug:
        # Return dummy values for debugging
        live_pvs_and_values = {pv: 5.0 for pv in pv_names}
    else:
        try:
            # Archiver appliance endpoint
            # TODO: Make this configurable or get from data object, shouldn't be hardcoded link specific to CMS
            BASE_URL = "http://epics-services-cms.nsls2.bnl.local:11168"
            ENDPOINT = "/retrieval/data/getDataAtTime"

            # Current UTC time in ISO8601 format
            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            
            # Construct full URL with query parameters
            url = f"{BASE_URL}{ENDPOINT}?at={now_iso}&includeProxies=true"
            
            # Make the POST request
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(pv_names)
            )
            
            # Check for request errors
            response.raise_for_status()
            
            # Parse the returned JSON
            data = response.json()
            
            # Extract each PV value
            for pv in pv_names:
                entry = data.get(pv)
                if entry and "val" in entry:
                    live_pvs_and_values[pv] = entry["val"]
                else:
                    # If no data available, use a default
                    print(f"[UI] No data available for PV: {pv}")
                    live_pvs_and_values[pv] = 0.0
                    
        except Exception as e:
            print(f"[UI] Error fetching PV values: {str(e)}")
            # Fall back to dummy values on error
            live_pvs_and_values = {pv: 0.0 for pv in pv_names}
            
    return live_pvs_and_values


class Ui_MainWindow(QDialog):

    # --- new callback: user closed the simulation-status window ---
    def _on_sim_dialog_close(self):
        """
        Called when the user manually closes the simulation-status window.
        Only sends an abort message while the simulation is still running.
        """
        running = bool(getattr(self, "sim_in_progress", False))

        # stop listener thread (always safe)
        try:
            if getattr(self, "sim_thread", None):
                self.sim_thread.stop()
        except Exception as exc:
            print("[UI] error while stopping sim_thread:", exc)

        if running:
            # tell backend to abort the simulation
            abort_msg = {
                "bl_input_channel": "simulate_abort",
                "sim_id"  : getattr(self.sim_thread, "sim_id", None),
                "beamline": self.BeamlineDropDown_First_Tab.currentText(),
            }
            print("[UI] publish simulate_abort (fire-and-forget):", abort_msg)
            try:
                # publish directly; do NOT wait for a reply → UI stays responsive
                _hal_queue.publish([abort_msg])
            except Exception as exc:
                print("[UI] failed to publish simulate_abort:", exc)

            # reset "Simulate" button to aborted state
            self._finish_sim_button(reset_text="Simulate ✗")

    # ----- Helpers -------------------------------------------------
    def _ensure_dict(self, reply):
        """Make normal dicts out of HAL replies"""
        import numpy as np
        
        # Treat numpy arrays as dicts
        if isinstance(reply, np.ndarray):
            if reply.shape == () and isinstance(reply.item(), dict):
                # Unwrap singleton ndarray with dict
                return reply.item()
            elif reply.dtype == object and reply.shape and all(isinstance(x, dict) for x in reply):
                # Convert ndarray of dicts to list
                reply = list(reply)
        
        # Treating lists, take last item
        if isinstance(reply, list):
            if len(reply) == 1:
                return reply[0]
            elif len(reply) > 1:
                return reply[-1]
            return {}
        
        # If it's a dict, return it
        if isinstance(reply, dict):
            return reply
        
        # Every other type, make it a list
        return {'value': str(reply)}

    # @staticmethod
    # def input_hook(context):
    #     while not context.input_is_ready():
    #         QtWidgets.QApplication.processEvents()

    def __init__(self, isInterfacing=False):

        # Purge S3 queue (delete ONLY ONCE at launch)
        from S3_test.send_to_hal import q as _global_queue
        _global_queue.clear()      # purge once at launch

        voice = True

        #Initializing 
        super(Ui_MainWindow, self).__init__()

        # Determine the base directory (either project root or current directory)
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the relative path to the .ui file
        ui_file_path = os.path.join(base_dir, "UI.ui")

        # Load the .ui file
        uic.loadUi(ui_file_path, self)

        self.setWindowTitle("VISION_v1")
        self.isRecording = False

        self.sim_in_progress = False
        self.sim_start_time  = None

        self.data = {
            # 'datetime': Base().now(),
            'project_path': "",
            'user_id': '',
            'terminate': 0,
            'bl_conf': 0,
            'only_text_input': 0,
            'context': [],
            'errors': [],
            'cog_id_error': [],
            'beamline': "",
            'bl_input_channel': "command",
            'text_input': "",

            'voice_cog_output': "",
            'classifier_cog_output': "",
            'op_cog_output': "",
            'ana_cog_output': "",
            'refinement_cog_output': "",

            'voice_cog_input': "",
            'classifier_cog_input': "",
            'operation_cog_input': "",
            'operation_cog_output': "",
            'analysis_cog_input': "",
            'analysis_cog_output': "",
            'refinement_cog_input': "",

            'classifier_cog_history': "",
            'operator_cog_history': "",
            'classifier_cog_db_history': "",
            'operator_cog_db_history': "",

            'include_context_functions': False,
            'include_verifier': False,
            'status': 'success'
        }

        # Authentication data structures
        self.auth_data = {
            'bl_input_channel': '',
            'user_id': '',
            'password': '',
            'session_id': '',
            'status': 'success',
            'message': ''
        }

        # self.register_data = {
        #     'bl_input_channel': 'register',
        #     'user_id': '',
        #     'email': '',
        #     'password': '',
        #     'status': 'success'
        # }
        #
        # self.logout_data = {
        #     'bl_input_channel': 'logout',
        #     'user_id': '',
        #     'session_id': '',
        #     'status': 'success'
        # }

        # Authentication state
        self.is_authenticated = False
        self.current_user_id = ""
        self.session_id = ""
        self.is_guest_mode = False

        self.context_data = {
            # 'datetime': Base().now(),
            'context_data': "",
            'only_text_input': 1,
            'bl_input_channel': "add_context",

            'project_path': "",
            'user_id': '',
            'terminate': 0,
            'bl_conf': 0,
            'context': [],
            'errors': [],
            'cog_id_error': [],
            'beamline': '11BM',
            'text_input': "",

            'op_cog_output': "",
            'ana_cog_output': "",
            'refinement_cog_output': "",

            'voice_cog_input': "",
            'voice_cog_output': "",
            'classifier_cog_input': "",
            'classifier_cog_output': "",
            'operation_cog_input': "",
            'operation_cog_output': "",
            'analysis_cog_input': "",
            'analysis_cog_output': "",
            'refinement_cog_input': "",

            'include_context_functions': False,
            'include_verifier': False,
            'status': 'success'
        }

        self.chat_data = {
            'bl_input_channel': 'chatbot',
            'user_id': '',
            'text_input': "",
            "history": "",
            'only_text_input': 1,
            'voice_cog_output': ""
        }


        #Voice module
        if voice:
            audio = pa.PyAudio()
            print("=== [PyAudio] Available input devices ===")
            device_count = audio.get_device_count()
            for i in range(device_count):
                info = audio.get_device_info_by_index(i)
                if info.get('maxInputChannels') > 0:
                    print(f"   Device Index: {info.get('index')}, Name: {info.get('name')}, Max Input Channels: {info.get('maxInputChannels')}")

        #Connect everything here to the UI components
        self.user_id_input = self.findChild(QLineEdit, "userIdInput")
        self.Input_Submit = self.findChild(QPushButton, "Input_Submit")

        # Connect logout/login button
        self.logout_button = self.findChild(QPushButton, "logoutButton")
        self.logout_button.clicked.connect(self.handle_logout_login)
        self.logout_button.setEnabled(False)  # Initially disabled until login

        # Show authentication dialog first
        self.show_auth_dialog()

        self.CE_sim = self.findChild(QPushButton, "CE_sim")
        self.CE_sim.clicked.connect(self.simulate_code)
        
        self.useArchiverCheckBox = self.findChild(QCheckBox, "useArchiverCheckBox")

        self.Input_Edit = self.findChild(QTextEdit, "Input_Edit") #QLineEdit
        self.Input_Edit.setLineWrapMode(QTextEdit.WidgetWidth)
        self.projectPathInput = self.findChild(QLineEdit, "projectPathInput")
        self.Input_Submit.clicked.connect(self.inputCommand)

        # self.Log_List = self.findChild(QListWidget, "Log_List")
        self.Log_List = self.findChild(QTextEdit, "Log_List")
        self.Log_Confirm = self.findChild(QPushButton, "Log_Confirm")
        self.Log_Clear = self.findChild(QPushButton, "Log_Clear")

        self.Log_Confirm.clicked.connect(self.submitCommand)
        self.Log_Clear.clicked.connect(self.resetLog)

        # self.CE_List = self.findChild(QListWidget, "CE_List")

        self.CE_List = self.findChild(QTextEdit, "CE_List")
        self.CE_Hist = self.findChild(QTextEdit, "CE_Hist")
        self.CE_List.textChanged.connect(self._on_ce_code_changed)

        self.Sample_Log = self.findChild(QLabel, "Sample_Log")
        self.Temperature_Log = self.findChild(QLabel, "Temperature_Log")
        self.Humidity_Log = self.findChild(QLabel, "Humidity_Log")
        self.Position_Log = self.findChild(QLabel, "Position_Log")

        self.BeamlineTextLabel = self.findChild(QLabel, "label_beamline")
        # self.BeamlineTextBox = self.findChild(QTextEdit, "text_beamlineID")
        self.BeamlineDropDown = self.findChild(QComboBox, "dropdown_beamline")
        self.BeamlineDropDown_First_Tab = self.findChild(QComboBox, "beamlineComboBox")
        self.SelectedCog = self.findChild(QComboBox, "dropdown_cog")
        self.add_context_output = self.findChild(QTextEdit, "text_output")
        self.add_context_result = self.findChild(QTextEdit, "result_message_box")

        self.add_context_input = self.findChild(QLineEdit, "input_text_box")
        self.add_context_submit = self.findChild(QPushButton, "submit_add_context_text_button")
        self.add_context_submit.clicked.connect(self.submit_context)

        if voice:
            self.Input_Voice = self.findChild(QPushButton, "Input_Voice")
            self.recording_thread = RecordingThread()
            self.recording_thread.sig_started.connect(self.recording_started)
            self.recording_thread.sig_stopped.connect(self.recording_stopped)
            self.recording_thread.sig_transcription.connect(self.update_transcription)
            self.Input_Voice.clicked.connect(self.recording_thread.start)

            self.Stop_Voice = self.findChild(QPushButton, "Stop_Voice")
            self.Stop_Voice.clicked.connect(self.recording_thread.stop)
            self.Stop_Voice.setDisabled(True)

            #Recording buttons for add_context
            self.add_context_start = self.findChild(QPushButton, "add_context_start")
            self.recording_thread_context = RecordingThread()
            self.recording_thread_context.sig_started.connect(self.recording_started)
            self.recording_thread_context.sig_stopped.connect(self.recording_stopped)
            self.add_context_start.clicked.connect(self.recording_thread_context.start)

            self.add_context_stop = self.findChild(QPushButton, "add_context_stop")
            self.add_context_stop.clicked.connect(self.recording_thread_context.stop)
            self.recording_thread_context.sig_transcription.connect(self.transcribe_context)

            self.add_context_stop.setDisabled(True)

            self.Confirm_Add_Context = self.findChild(QPushButton, "confirm_save_button")
            self.Confirm_Add_Context.clicked.connect(self.add_context)

            self.check_add_context = self.findChild(QCheckBox, "checkBox")
            # self.check_verifier = self.findChild(QCheckBox, "verifierCheckBox")

            #Recording buttons for chatbot
            self.chatbot_start = self.findChild(QPushButton, "chat_start_record_button")
            self.recording_thread_chatbot = RecordingThread()
            self.recording_thread_chatbot.sig_started.connect(self.recording_started)
            self.recording_thread_chatbot.sig_stopped.connect(self.recording_stopped)
            self.chatbot_start.clicked.connect(self.recording_thread_chatbot.start)

            self.chatbot_stop = self.findChild(QPushButton, "chat_stop_record_button")
            self.chatbot_stop.clicked.connect(self.recording_thread_chatbot.stop)
            self.recording_thread_chatbot.sig_transcription.connect(self.transcribe_chatbot)

            self.chatbot_stop.setDisabled(True)

            # Access UI elements
            self.chat_display = self.findChild(QTextEdit, 'chat_display')
            self.user_input = self.findChild(QLineEdit, 'user_input')
            self.send_button = self.findChild(QPushButton, 'send_button')

            # Make the chat display read-only
            self.chat_display.setReadOnly(True)

            # Connect the button click to the send_message function
            self.send_button.clicked.connect(self.send_chat_message)

            # Data to be sent to HAL

        # self.qr_data_label = self.findChild(QLabel, "qr_label")
        # self.qr_data_label.setScaledContents(True)
        self.ana_text = self.findChild(QTextEdit, "ana_text")
        self.data_label = self.findChild(QLabel, "data_plot") 
        self.data_label.setScaledContents(True)

        # Add table for PV updates (simulation)
        # (now handled by floating SimStatusDialog)

        #UI variables used in interface with other scripts
        self.samples = []
        self.statusTimer = QtCore.QTimer()
        self.dataTimer_qr = QtCore.QTimer()
        self.dataTimer = QtCore.QTimer()
        self.isInterfacing = isInterfacing

        self.dataTimer.timeout.connect(lambda : self.getData())
        self.dataTimer.start(1000) #Milliseconds

        self.currentFile_qr = ""
        self.currentFile = ""

        # self.dataDirectory_qr = "/nsls2/data/cms/legacy/xf11bm/data/2024_3/beamline/ETsai/saxs/analysis/q_image/"
           
        self.dataDirectory = "" #"/nsls2/data/cms/legacy/xf11bm/data/2024_3/beamline/ETsai/saxs/analysis/circular_average/"

        # self.recording_data = []

        self.transcription_str = ""

        # self.worker = Worker(self.fs)
        # self.thread = QThread()
        # self.worker.moveToThread(self.thread)
        # self.worker.transcription_signal.connect(self.update_transcription)

    # Authentication methods
    def show_auth_dialog(self):
        """Show the authentication dialog and handle login/register"""
        print("[AUTH] Showing authentication dialog...")
        auth_dialog = AuthDialog(self)
        result = auth_dialog.exec_()

        print(f"[AUTH] Dialog result: {result}, is_authenticated: {auth_dialog.is_authenticated}")

        if result == QDialog.Accepted and auth_dialog.is_authenticated:
            # Get authentication data
            auth_data = auth_dialog.get_auth_data()
            print(f"[AUTH] Auth data: {auth_data}")

            if auth_data:
                # Check if it's guest mode
                if auth_dialog.is_guest_mode:
                    print("[AUTH] Guest mode selected")
                    # Guest mode - no backend authentication needed
                    self.is_authenticated = True
                    self.current_user_id = auth_dialog.user_id
                    self.session_id = ""
                    self.is_guest_mode = auth_dialog.is_guest_mode

                    # Update data structures
                    self.data['user_id'] = self.current_user_id
                    self.context_data['user_id'] = self.current_user_id
                    self.chat_data['user_id'] = self.current_user_id

                    # Update UI to show user ID
                    self.update_user_display()

                    # Show success message
                    # from PyQt5.QtWidgets import QMessageBox
                    # QMessageBox.information(self, "Success", "Welcome to Guest Mode!")

                else:
                    print("[AUTH] Regular authentication selected")
                    # Regular authentication - Send to HAL backend
                    def auth_success(response):
                        print(f"[AUTH] SUCCESS CALLBACK CALLED with response: {response}")
                        print(f"[AUTH] Callback ID: {id(auth_success)}")

                        # Handle the response
                        if response and isinstance(response, dict):
                            if response.get('status') == 'success':
                                print(f"[AUTH] Processing successful authentication")
                                print(f"[AUTH] Before update - is_authenticated: {self.is_authenticated}, current_user_id: {self.current_user_id}")

                                # Authentication successful
                                self.is_authenticated = True
                                self.current_user_id = auth_dialog.user_id
                                self.session_id = response.get('session_id', '')
                                self.is_guest_mode = auth_dialog.is_guest_mode

                                print(f"[AUTH] After update - is_authenticated: {self.is_authenticated}, current_user_id: {self.current_user_id}")

                                # Update data structures
                                self.data['user_id'] = self.current_user_id
                                self.context_data['user_id'] = self.current_user_id
                                self.chat_data['user_id'] = self.current_user_id

                                self.data['classifier_cog_db_history'] = response.get('classifier_cog_db_history', '')
                                self.data['operator_cog_db_history'] = response.get('operator_cog_db_history', '')

                                # Update UI to show user ID
                                print(f"[AUTH] Calling update_user_display...")
                                self.update_user_display()
                                print(f"[AUTH] update_user_display completed")

                                # Show success message
                                # from PyQt5.QtWidgets import QMessageBox
                                # QMessageBox.information(self, "Success", f"Welcome, {self.current_user_id}!")

                            else:
                                print(f"[AUTH] Authentication failed with status: {response.get('status')}")
                                # Authentication failed
                                error_msg = response.get('error', 'Authentication failed')
                                from PyQt5.QtWidgets import QMessageBox
                                QMessageBox.warning(self, "Authentication Error", error_msg)
                                # Show dialog again
                                self.show_auth_dialog()
                        else:
                            print(f"[AUTH] Invalid response received: {response}")
                            # No response or invalid response
                            from PyQt5.QtWidgets import QMessageBox
                            QMessageBox.warning(self, "Error", "No response from server. Please try again.")
                            self.show_auth_dialog()

                    def auth_error(error_msg):
                        print(f"[AUTH] ERROR CALLBACK CALLED with error: {error_msg}")
                        print(f"[AUTH] Error callback ID: {id(auth_error)}")
                        from PyQt5.QtWidgets import QMessageBox
                        QMessageBox.warning(self, "Error", f"Connection error: {error_msg}")
                        self.show_auth_dialog()

                    print(f"[AUTH] Sending auth data to HAL: {auth_data}")
                    print(f"[AUTH] About to call run_qt_async...")
                    print(f"[AUTH] auth_success callback ID: {id(auth_success)}")
                    print(f"[AUTH] auth_error callback ID: {id(auth_error)}")
                    # Use non-blocking async call
                    run_qt_async(send_hal_async(auth_data), auth_success, auth_error)
                    print(f"[AUTH] run_qt_async called successfully")
            else:
                print("[AUTH] No auth data received from dialog")
        else:
            # User cancelled authentication
            print("[AUTH] User cancelled authentication")
            # Close the application
            self.close()

    def update_user_display(self):
        """Update the UI to show the current user ID"""
        print(f"[UI] update_user_display called - is_authenticated: {self.is_authenticated}, current_user_id: {self.current_user_id}")

        if self.is_authenticated and self.current_user_id:
            print(f"[UI] Updating UI for authenticated user: {self.current_user_id}")

            # Update the user ID input field
            if hasattr(self, 'user_id_input') and self.user_id_input:
                print(f"[UI] Setting user_id_input text to: {self.current_user_id}")
                self.user_id_input.setText(self.current_user_id)
                # Ensure the widget is enabled and properly configured
                self.user_id_input.setEnabled(True)
                self.user_id_input.setReadOnly(True)  # Keep it read-only but ensure it's enabled
                print(f"[UI] user_id_input enabled and configured")
            else:
                print(f"[UI] user_id_input not found or None")

            # Update window title to show user
            if self.is_guest_mode:
                title = f"VISION_v1 - Guest Mode"
                print(f"[UI] Setting window title to: {title}")
                self.setWindowTitle(title)
            else:
                title = f"VISION_v1 - User: {self.current_user_id}"
                print(f"[UI] Setting window title to: {title}")
                self.setWindowTitle(title)

            # Enable logout/login button and set appropriate text
            if hasattr(self, 'logout_button'):
                print(f"[UI] Enabling logout button")
                self.logout_button.setEnabled(True)
                if self.is_guest_mode:
                    button_text = "Login"
                else:
                    button_text = "Logout"
                print(f"[UI] Setting logout button text to: {button_text}")
                self.logout_button.setText(button_text)
                print(f"[UI] logout button enabled and configured")
            else:
                print(f"[UI] logout_button not found")
        else:
            print(f"[UI] Not updating UI - is_authenticated: {self.is_authenticated}, current_user_id: {self.current_user_id}")

    def handle_logout_login(self):
        """Handle logout for authenticated users or show login dialog for guests"""
        if self.is_guest_mode:
            # Guest mode - show login dialog
            self.show_auth_dialog()
        else:
            # Authenticated user - perform logout
            self.logout()

    def logout(self):
        """Handle user logout"""
        if self.is_authenticated and self.current_user_id:
            print(f"[LOGOUT] Starting logout for user: {self.current_user_id}")
            # Create logout data
            logout_data = {
                'bl_input_channel': 'logout',
                'user_id': self.current_user_id,
                'session_id': self.session_id,
                'status': 'success'
            }

            def logout_success(response):
                print(f"[LOGOUT] HAL response: {response}")

                # Clear authentication state
                print(f"[LOGOUT] Clearing authentication state")
                self.is_authenticated = False
                self.current_user_id = ""
                self.session_id = ""
                self.is_guest_mode = False


                # Clear user display
                print(f"[LOGOUT] Clearing user display")
                self.setWindowTitle("VISION_v1")
                if hasattr(self, 'user_id_input') and self.user_id_input:
                    print(f"[LOGOUT] Clearing user_id_input text")
                    self.user_id_input.setText("")
                else:
                    print(f"[LOGOUT] user_id_input not found")

                # Disable logout button
                if hasattr(self, 'logout_button'):
                    print(f"[LOGOUT] Disabling logout button")
                    self.logout_button.setEnabled(False)
                else:
                    print(f"[LOGOUT] logout_button not found")

                # Clear all UI text elements and data
                self.clear_all_ui_data()

                # Show authentication dialog again
                print(f"[LOGOUT] Showing auth dialog after successful logout")
                self.show_auth_dialog()

            def logout_error(error_msg):
                print(f"[LOGOUT] Error: {error_msg}")
                # Even if logout fails, clear the local state
                print(f"[LOGOUT_ERROR] Clearing authentication state")
                self.is_authenticated = False
                self.current_user_id = ""
                self.session_id = ""
                self.is_guest_mode = False

                # Clear user display
                print(f"[LOGOUT_ERROR] Clearing user display")
                self.setWindowTitle("VISION_v1")
                if hasattr(self, 'user_id_input') and self.user_id_input:
                    print(f"[LOGOUT_ERROR] Clearing user_id_input text")
                    self.user_id_input.setText("")
                else:
                    print(f"[LOGOUT_ERROR] user_id_input not found")

                # Disable logout button
                if hasattr(self, 'logout_button'):
                    print(f"[LOGOUT_ERROR] Disabling logout button")
                    self.logout_button.setEnabled(False)
                else:
                    print(f"[LOGOUT_ERROR] logout_button not found")

                # Clear all UI text elements and data
                self.clear_all_ui_data()

                # Show authentication dialog again
                print(f"[LOGOUT_ERROR] Showing auth dialog after logout error")
                self.show_auth_dialog()

            # Use non-blocking async call
            print(f"[LOGOUT] Sending logout request to HAL")
            run_qt_async(send_hal_async(logout_data), logout_success, logout_error)

    def resizeEvent(self, event):
        """Handle window resize events to make widgets scale dynamically."""
        super().resizeEvent(event)
        
        # Get the new size
        new_size = event.size()
        width = new_size.width()
        height = new_size.height()
        
        # Only resize if we have the widgets initialized
        if not hasattr(self, 'Log_List'):
            return
        
        # Define margins and spacing
        margin = 10
        right_margin = 30  # Extra breathing room on the right side
        bottom_margin = 30  # Extra breathing room at the bottom
        button_height = 36
        label_height = 35
        input_height = 61
        voice_button_height = 34
        ana_text_height = 31
        
        # Top section (fixed height) - includes input, voice buttons, and labels
        top_section_height = 195  # Y position where Log starts
        
        # Calculate available height for middle and bottom sections with bottom margin
        available_height = height - top_section_height - margin * 2 - bottom_margin
        
        # Middle section (Log and CE) - 50% of available height
        middle_height = int(available_height * 0.5)
        
        # Bottom section (Ana and CE_Hist) - 50% of available height
        bottom_height = available_height - middle_height
        
        # Calculate widths for left and right columns with extra right margin
        left_width = int((width - 2 * margin - right_margin) * 0.5)
        right_width = width - left_width - 2 * margin - right_margin
        
        # Resize Log_Text label (centered above Log_List)
        if hasattr(self, 'Log_Text') and self.Log_Text:
            log_label_width = 60
            log_label_x = margin + (left_width - log_label_width) // 2
            self.Log_Text.setGeometry(log_label_x, 160, log_label_width, label_height)
        
        # Resize Log_List
        if hasattr(self, 'Log_List') and self.Log_List:
            self.Log_List.setGeometry(
                margin, 
                top_section_height, 
                left_width, 
                middle_height - button_height - margin
            )
        
        # Resize Log buttons
        if hasattr(self, 'Log_Clear') and self.Log_Clear:
            log_buttons_y = top_section_height + middle_height - button_height
            self.Log_Clear.setGeometry(margin, log_buttons_y, 75, button_height)
        
        if hasattr(self, 'Log_Confirm') and self.Log_Confirm:
            log_buttons_y = top_section_height + middle_height - button_height
            self.Log_Confirm.setGeometry(
                margin + 80, 
                log_buttons_y, 
                left_width - 80, 
                button_height
            )
        
        # Resize CE_Text label (centered above CE_List)
        if hasattr(self, 'CE_Text') and self.CE_Text:
            ce_x = margin + left_width + margin
            ce_label_width = 200
            ce_label_x = ce_x + (right_width - ce_label_width) // 2
            self.CE_Text.setGeometry(ce_label_x, 160, ce_label_width, label_height)
        
        # Resize CE_List (Code Equivalent)
        if hasattr(self, 'CE_List') and self.CE_List:
            ce_x = margin + left_width + margin
            self.CE_List.setGeometry(
                ce_x, 
                top_section_height, 
                right_width, 
                middle_height - button_height - margin
            )
        
        # Resize CE buttons
        if hasattr(self, 'CE_sim') and self.CE_sim:
            ce_buttons_y = top_section_height + middle_height - button_height
            self.CE_sim.setGeometry(
                ce_x, 
                ce_buttons_y, 
                right_width - 100,  # More space for Live PV checkbox
                button_height
            )
        
        if hasattr(self, 'useArchiverCheckBox') and self.useArchiverCheckBox:
            ce_buttons_y = top_section_height + middle_height - button_height
            self.useArchiverCheckBox.setGeometry(
                ce_x + right_width - 95,
                ce_buttons_y + 8,
                90,  # Wider checkbox
                21
            )
        
        # Bottom section Y position
        bottom_y = top_section_height + middle_height + margin
        
        # Resize Analysis section (left bottom)
        if hasattr(self, 'Ana_title') and self.Ana_title:
            self.Ana_title.setGeometry(margin, bottom_y, left_width, label_height)
        
        if hasattr(self, 'ana_text') and self.ana_text:
            self.ana_text.setGeometry(
                margin, 
                bottom_y + label_height, 
                left_width, 
                ana_text_height
            )
        
        if hasattr(self, 'data_label') and self.data_label:
            plot_y = bottom_y + label_height + ana_text_height
            plot_height = bottom_height - label_height - ana_text_height - margin * 3
            self.data_label.setGeometry(
                margin, 
                plot_y, 
                left_width, 
                plot_height
            )
        
        # Resize CE_Hist (Code History - right bottom)
        if hasattr(self, 'CE_Text_4') and self.CE_Text_4:
            ce_hist_x = margin + left_width + margin
            self.CE_Text_4.setGeometry(ce_hist_x, bottom_y, right_width, label_height)
        
        if hasattr(self, 'CE_Hist') and self.CE_Hist:
            ce_hist_x = margin + left_width + margin
            ce_hist_height = bottom_height - label_height - margin * 3
            self.CE_Hist.setGeometry(
                ce_hist_x, 
                bottom_y + label_height, 
                right_width, 
                ce_hist_height
            )
        
        # Resize Input_Edit
        if hasattr(self, 'Input_Edit') and self.Input_Edit:
            input_width = width - 256
            self.Input_Edit.setGeometry(10, 50, input_width, input_height)
        
        if hasattr(self, 'Input_Submit') and self.Input_Submit:
            self.Input_Submit.setGeometry(width - 236, 75, 196, 36)
        
        if hasattr(self, 'checkBox') and self.checkBox:
            self.checkBox.setGeometry(width - 236, 50, 196, 25)
        
        # Resize voice buttons
        if hasattr(self, 'Input_Voice') and self.Input_Voice:
            voice_width = int((width - 3 * margin) * 0.5)
            self.Input_Voice.setGeometry(margin, 120, voice_width, voice_button_height)
        
        if hasattr(self, 'Stop_Voice') and self.Stop_Voice:
            voice_x = margin + voice_width + margin
            stop_width = width - voice_x - margin
            self.Stop_Voice.setGeometry(voice_x, 120, stop_width, voice_button_height)
        
        # Resize TabWidget to fill the entire dialog
        if hasattr(self, 'tabWidget') and self.tabWidget:
            self.tabWidget.setGeometry(10, 10, width - 20, height - 20)

    def clear_all_ui_data(self):
        """Clear all text and data in UI elements after logout"""
        print("[LOGOUT] Clearing all UI data...")

        # Clear main input/output areas
        if hasattr(self, 'Input_Edit') and self.Input_Edit is not None:
            self.Input_Edit.clear()

        if hasattr(self, 'Log_List') and self.Log_List is not None:
            self.Log_List.clear()

        if hasattr(self, 'CE_List') and self.CE_List is not None:
            self.CE_List.clear()

        if hasattr(self, 'CE_Hist') and self.CE_Hist is not None:
            self.CE_Hist.clear()

        # Clear project path
        if hasattr(self, 'projectPathInput') and self.projectPathInput is not None:
            self.projectPathInput.clear()

        # Clear context-related elements
        if hasattr(self, 'add_context_input') and self.add_context_input is not None:
            self.add_context_input.clear()

        if hasattr(self, 'add_context_output') and self.add_context_output is not None:
            self.add_context_output.clear()

        if hasattr(self, 'add_context_result') and self.add_context_result is not None:
            self.add_context_result.clear()

        # Clear chatbot elements
        if hasattr(self, 'chat_display') and self.chat_display is not None:
            self.chat_display.clear()

        if hasattr(self, 'user_input') and self.user_input is not None:
            self.user_input.clear()

        # Clear analysis text
        if hasattr(self, 'ana_text') and self.ana_text is not None:
            self.ana_text.clear()

        # Clear status labels
        if hasattr(self, 'Sample_Log') and self.Sample_Log is not None:
            self.Sample_Log.setText("")

        if hasattr(self, 'Temperature_Log') and self.Temperature_Log is not None:
            self.Temperature_Log.setText("")

        if hasattr(self, 'Humidity_Log') and self.Humidity_Log is not None:
            self.Humidity_Log.setText("")

        if hasattr(self, 'Position_Log') and self.Position_Log is not None:
            self.Position_Log.setText("")

        # Clear data structures
        self.data = {
            'project_path': "",
            'user_id': '',
            'terminate': 0,
            'bl_conf': 0,
            'only_text_input': 0,
            'context': [],
            'errors': [],
            'cog_id_error': [],
            'beamline': "",
            'bl_input_channel': "command",
            'text_input': "",
            'voice_cog_output': "",
            'classifier_cog_output': "",
            'op_cog_output': "",
            'ana_cog_output': "",
            'refinement_cog_output': "",
            'voice_cog_input': "",
            'classifier_cog_input': "",
            'operation_cog_input': "",
            'operation_cog_output': "",
            'analysis_cog_input': "",
            'analysis_cog_output': "",
            'refinement_cog_input': "",
            'classifier_cog_history': "",
            'operator_cog_history': "",
            'classifier_cog_db_history': "",
            'operator_cog_db_history': "",
            'include_context_functions': False,
            'include_verifier': False,
            'status': 'success'
        }

        self.context_data = {
            'context_data': "",
            'only_text_input': 1,
            'bl_input_channel': "add_context",
            'project_path': "",
            'user_id': '',
            'terminate': 0,
            'bl_conf': 0,
            'context': [],
            'errors': [],
            'cog_id_error': [],
            'beamline': '11BM',
            'text_input': "",
            'op_cog_output': "",
            'ana_cog_output': "",
            'refinement_cog_output': "",
            'voice_cog_input': "",
            'voice_cog_output': "",
            'classifier_cog_input': "",
            'classifier_cog_output': "",
            'operation_cog_input': "",
            'operation_cog_output': "",
            'analysis_cog_input': "",
            'analysis_cog_output': "",
            'refinement_cog_input': "",
            'include_context_functions': False,
            'include_verifier': False,
            'status': 'success'
        }

        self.chat_data = {
            'bl_input_channel': 'chatbot',
            'user_id': '',
            'text_input': "",
            "history": "",
            'only_text_input': 1,
            'voice_cog_output': ""
        }

        # Reset checkboxes
        if hasattr(self, 'check_add_context') and self.check_add_context is not None:
            self.check_add_context.setChecked(False)

        # if hasattr(self, 'check_verifier') and self.check_verifier is not None:
        #     self.check_verifier.setChecked(False)

        if hasattr(self, 'useArchiverCheckBox') and self.useArchiverCheckBox is not None:
            self.useArchiverCheckBox.setChecked(False)

        # Clear transcription string
        self.transcription_str = ""

        # Reset beamline dropdowns to default
        if hasattr(self, 'BeamlineDropDown') and self.BeamlineDropDown is not None and self.BeamlineDropDown.count() > 0:
            self.BeamlineDropDown.setCurrentIndex(0)

        if hasattr(self, 'BeamlineDropDown_First_Tab') and self.BeamlineDropDown_First_Tab is not None and self.BeamlineDropDown_First_Tab.count() > 0:
            self.BeamlineDropDown_First_Tab.setCurrentIndex(0)

        if hasattr(self, 'SelectedCog') and self.SelectedCog is not None and self.SelectedCog.count() > 0:
            self.SelectedCog.setCurrentIndex(0)

        print("[LOGOUT] All UI data cleared successfully")

    '''
        Display latest analysis result
    '''
    def getData(self, verbose=1):

        firstFile = True
        mostRecentFile = 0
        self.check_create_project_dir()
        self.dataDirectory = self.data['project_path'] #+"/saxs/analysis/" 
        #print(self.dataDirectory)

        png_files = []
        for root, _, files in os.walk(self.dataDirectory):

            for file in files:
                if file.lower().endswith('.png'):
                    png_files.append(os.path.join(root, file))

        if verbose > 0 and png_files:
            print("Number of PNG files: {}".format(len(png_files)))

        if(len(png_files) > 0):
            png_files = sorted(png_files, key=lambda x: os.path.getmtime(x), reverse=True) 
            self.currentFile =  png_files[0]
            if verbose>0: print(self.currentFile)

            self.ana_text.setText(self.currentFile)
            pixmap = QPixmap(self.currentFile)
            self.data_label.setPixmap(pixmap)


    '''
        Sets the Interface instance to interface with in the execution
    '''
    def setInterface(self, interface):
        if(self.isInterfacing):
            self.interface = interface            
            self.statusTimer.timeout.connect(lambda : self.updateStatus())
            self.statusTimer.start(100)

    '''
        Adds to and clears the Log List 
    '''
    def resetLog(self):
        self.Log_List.clear()
        self.CE_List.clear()
        self.CE_Hist.clear()
        self.Input_Edit.clear()

        self.data['classifier_cog_history'] = ""
        self.data['operator_cog_history'] = ""
        self.data['classifier_cog_db_history'] = ""
        self.data['operator_cog_db_history'] = ""


    def send_hal_trans_conf(self):

        self.data['bl_conf'] = 1

        def trans_conf_success(response):
            self.data = response
            print(self.data['op_output'])

        def trans_conf_error(error_msg):
            QtWidgets.QMessageBox.critical(self, "HAL error", error_msg)

        # Use non-blocking async call
        run_qt_async(send_hal_async(self.data), trans_conf_success, trans_conf_error)


    def send_hal_trans_redo(self):

        self.data['bl_conf'] = 0
        self.voiceInput()
        # self.data = send_hal(self.data)


    #This is for the submit button (text) on Add Context Functions tab
    def submit_context(self):
        
        self.context_data['only_text_input'] = 1
        self.context_data['bl_input_channel'] = "add_context"
        self.context_data['text_input'] = self.add_context_input.text()

        self.context_data['beamline_id'] = self.BeamlineDropDown.currentText()
        self.context_data['selected_cog'] = self.SelectedCog.currentText()

        def context_success(response):
            self.context_data = response
            self.add_context_output.setPlainText(self.context_data['refinement_cog_display_output'])
            print(self.context_data)

        def context_error(error_msg):
            QtWidgets.QMessageBox.critical(self, "HAL error", error_msg)

        # Use non-blocking async call
        run_qt_async(send_hal_async(self.context_data), context_success, context_error)

    
    #This is for the confirm and save button on Add Context Functions tab
    def add_context(self):

        current_datetime = datetime.now()
        formatted_date = "["+current_datetime.strftime("%Y-%m-%d %H:%M:%S") + "] "
    
        self.context_data['bl_input_channel'] = "confirm_context"
        self.context_data['beamline_id'] = self.BeamlineDropDown.currentText()
        self.context_data['selected_cog'] = self.SelectedCog.currentText()

    
        print(self.add_context_output.toPlainText())

        bold_start = "<b>"
        bold_end = "</b>"
        
        try:
            dict_check = ast.literal_eval(self.add_context_output.toPlainText())

            if isinstance(dict_check, dict):
                print("Sending to HAL!")
            else:
                self.add_context_result.setHtml(f"{formatted_date}{bold_start}Please enter a valid dictionary with the input, output, and cog fields!{bold_end}")
                return 

        except (ValueError, SyntaxError):
            self.add_context_result.setHtml(f"{formatted_date}{bold_start}Please enter a valid dictionary with the input, output, and cog fields!{bold_end}")
            return 
            


        # try:
        #     parsed_data = json.loads(self.add_context_output.toPlainText())

        #     if isinstance(parsed_data, dict):
        #         print("Sending to HAL")

        # except (json.JSONDecodeError, TypeError):
        #     self.add_context_output.setPlainText("Please enter a valid dictionary with input and output fields!")
        #     return


        self.context_data['add_context_tb_out'] = self.add_context_output.toPlainText()


        self.context_data['bl_tab3_conf'] = 1

        def add_context_success(response):
            self.context_data = response
            print("Added context functions for Beamline: {} and Cog: {}".format(self.context_data['beamline_id'],
            self.context_data['selected_cog']))

            # self.add_context_output.setPlainText(f"Successfully added {self.context_data['refinement_cog_output_dict']['input']} to {self.context_data['selected_cog']} Cog at {self.context_data['beamline_id']} beamline!!")

            # self.add_context_output.setPlainText(self.context_data['append_examples_output'])

            self.add_context_result.setHtml(formatted_date + self.context_data['append_examples_output'])

            self.check_create_project_dir()
            self.save_notebook_tab3()

        def add_context_error(error_msg):
            QtWidgets.QMessageBox.critical(self, "HAL error", error_msg)

        # Use non-blocking async call
        run_qt_async(send_hal_async(self.context_data), add_context_success, add_context_error)



    @pyqtSlot()
    def transcribe_context(self):

        
        
        self.add_context_output.setPlainText("Processing ...")

        self.context_data['only_text_input'] = 0
        # self.context_data['add_context_tb_out'] = self.add_context_output.toPlainText()
        self.context_data['text_input'] = self.add_context_input.text()
        # self.context_data['beamline_id'] = self.BeamlineTextBox.toPlainText()
        self.context_data['beamline_id'] = self.BeamlineDropDown.currentText()
        self.context_data['selected_cog'] = self.SelectedCog.currentText()
        # print(self.context_data)
        def context_transcribe_success(response):
            self.context_data = response
            self.context_data['only_text_input'] = 1
            # self.add_context_output.setPlainText(self.context_data['audio_output'])
            self.add_context_output.setPlainText(self.context_data['refinement_cog_display_output'])
            print(self.context_data)

        def context_transcribe_error(error_msg):
            QtWidgets.QMessageBox.critical(self, "HAL error", error_msg)

        # Use non-blocking async call
        run_qt_async(send_audio_file_async(self.context_data, 'output.wav'), context_transcribe_success, context_transcribe_error)

    @pyqtSlot()
    def transcribe_chatbot(self):
        
        self.chat_data['only_text_input'] = 0
        self.chat_data['text_input'] = self.user_input.text()
        # self.context_data['beamline_id'] = self.BeamlineTextBox.toPlainText()
        # self.context_data['beamline_id'] = self.BeamlineDropDown.currentText()

        # print(self.context_data)
        def chatbot_transcribe_success(response):
            self.chat_data = response
            self.chat_data['only_text_input'] = 1
            self.chat_display.append(f"\nUser: {self.chat_data['prompt']}")
            self.show_chatbot_response()

        def chatbot_transcribe_error(error_msg):
            QtWidgets.QMessageBox.critical(self, "HAL error", error_msg)

        # Use non-blocking async call
        run_qt_async(send_audio_file_async(self.chat_data, 'output.wav'), chatbot_transcribe_success, chatbot_transcribe_error)



    def send_chat_message(self):
        # Get the user's input
        message = self.user_input.text()

        # Check if the message is not empty or whitespace
        if message.strip():
            # Display the user message in the chat display
            self.chat_display.append(f"User: {message}")

            # Clear the input field after sending
            self.user_input.clear()

            # Send the message to HAL and receive the response
            self.chat_data['text_input'] = message

            def chat_success(response):
                self.chat_data = response
                self.show_chatbot_response()

            def chat_error(error_msg):
                QtWidgets.QMessageBox.critical(self, "HAL error", error_msg)

            # Use non-blocking async call
            run_qt_async(send_hal_async(self.chat_data), chat_success, chat_error)

    def show_chatbot_response(self):
        # Display the response from HAL in the chat display
        response = self.chat_data.get('chatbot_response', 'No response from HAL')
        # Escape any HTML special characters and replace newlines with <br> to preserve formatting
        response_html = html.escape(response).replace('\n', '<br>')

        # Display the response in blue color, preserving newlines
        self.chat_display.append(f'<font color="blue">{response_html}</font>')
    
    
    def check_create_project_dir(self):

        self.data['project_path'] = self.projectPathInput.text()
        #print('project_path')

        if self.data['project_path'] == "":
            # self.data['project_path'] = "/nsls2/data3/cms/legacy/xf11bm/data/2024_3/beamline/ETsai/Test/"
            # self.projectPathInput.setText("/nsls2/data3/cms/legacy/xf11bm/data/2024_3/beamline/ETsai/Test/")

            self.data['project_path'] = "./"
            self.projectPathInput.setText("./")
        
        # if self.data['project_path'] is None:
        #     self.data['project_path'] = os.getcwd()
        #     print("Project Path set to current working directory")

        path = self.data['project_path']
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                print(f"Directory created: {path}")
            except Exception as e:
                print(f"Project Path set to current working directory: {path}")

        # else:
        #     print(f"Found directory: {path}")

    def clear_output_fields(self, data):
        for key in list(data.keys()):
            if key.endswith("_output"):
                data[key] = ""
        return data

   
    # # Example usage
    # insert_protocols('runXS.py', [
    #     "Protocols.circular_average(ylog=True, plot_range=[0, 0.12, None, None], label_filename=True)",
    #     "Protocols.linecut_angle(q0=0.01687, dq=0.00455*1.5, show_region=False)"
    # ])
    def insert_protocols(self, input_filename, new_protocols, output_filename=None):
        with open(input_filename, 'r') as f:
            content = f.read()

        # Pattern to locate the protocols block
        pattern = r'protocols\s*=\s*\[(.*?)\](?=\s*#|\s*\n|$)'  # Non-greedy match of content inside brackets

        # Function to append new protocols
        def replacer(match):
            existing = match.group(1).strip()
            if existing:
                # Add comma if necessary
                updated = existing + '\n' + ',\n    '.join(new_protocols) + ',\n    '
            else:
                updated = '\n   ' + '\n    '.join(new_protocols) + ',\n    '
            return f'protocols = [\n    {updated}\n    ]'

        # Replace protocols block
        new_content = re.sub(pattern, replacer, content, flags=re.DOTALL)

        # Determine the output filename
        if output_filename is None:
            base_name, ext = os.path.splitext(input_filename)  # Split into base name and extension
            output_filename = f"{base_name}_llm{ext}"  # Append '_llm' before the extension

        # Write updated content to the new file
        with open(output_filename, 'w') as f:
            f.write(new_content)

        print(f"Inserted {len(new_protocols)} protocol(s) into {output_filename}.")



    def save_runXS(self):
        ana_output = self.data['analysis_cog_output']
        print("ana_output = {}", ana_output)
        #current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        project_path = self.data['project_path']
        self.insert_protocols(project_path + './runXS.py',  [ana_output])

        #self.context_data = self.clear_output_fields(self.context_data)


    def save_notebook_tab3(self):

        text_input = self.context_data['text_input']
        va_output = self.context_data['voice_cog_output']
        cla_output = self.context_data['classifier_cog_output']
        op_output = self.context_data['operation_cog_output']
        ana_output = self.context_data['analysis_cog_output']

        if self.context_data['only_text_input'] == 1:
            refine_output = self.context_data['add_context_tb_out']

        else:
            refine_output = self.context_data['refinement_cog_output']
        

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        notebook_folder_path = self.data['project_path']
        
        csv_file = os.path.join(notebook_folder_path, 'notebook.csv')

        row = {
            'Time':[current_time],
            'Text Input': [text_input],
            'Voice Cog Output': [va_output],
            'Classifier Cog Output': [cla_output],
            'Operation Cog Output': [op_output],
            'Analysis Cog Output': [ana_output],
            'Refinement Cog Output': [refine_output]
        }


        df = pd.DataFrame(row)

        print(df)

        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, mode = 'w', header=True, index = False)

        self.context_data = self.clear_output_fields(self.context_data)
    

    def save_notebook(self):

        text_input = self.data['text_input']
        va_output = self.data['voice_cog_output']
        cla_output = self.data['classifier_cog_output']
        op_output = self.data['op_cog_output']
        ana_output = self.data['ana_cog_output']
        refine_output = self.data['refinement_cog_output']
        ce_output = self.CE_List.toPlainText()

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        notebook_folder_path = self.data['project_path']
        
        csv_file = os.path.join(notebook_folder_path, 'notebook.csv')

        row = {
            'Time':[current_time],
            'Text Input': [text_input],
            'Voice Cog Output': [va_output],
            'Classifier Cog Output': [cla_output],
            'Operation Cog Output': [op_output],
            'Analysis Cog Output': [ana_output],
            'Refinement Cog Output': [refine_output],
            'Code Equivalent Text Box': [ce_output]
        }


        df = pd.DataFrame(row)

        print(df)

        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, mode = 'w', header=True, index = False)

        self.data = self.clear_output_fields(self.data)


    def add_CE_history(self):
        current_datetime = datetime.now()
        formatted_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        self.CE_Hist.append("#=== {} ".format(formatted_date))
        self.CE_Hist.append(self.CE_List.toPlainText())


    @pyqtSlot()
    def update_transcription(self):

        self.data['beamline'] = self.BeamlineDropDown_First_Tab.currentText()

        self.data = self.clear_output_fields(self.data)

        self.data['bl_conf'] = 0
        self.data['bl_input_channel'] = "command"

        self.data['project_path'] = self.projectPathInput.text()

        self.check_create_project_dir()

        if self.check_add_context.isChecked():
            self.data['include_context_functions'] = True

        else:
            self.data['include_context_functions'] = False

        # if self.check_verifier.isChecked():
        #     self.data['include_verifier'] = True

        # else:
        self.data['include_verifier'] = False

        self.data['text_input'] = self.Input_Edit.toPlainText()
        def transcription_success(response):
            self.data = self._ensure_dict(response)
            self._handle_transcription_response()

        def transcription_error(error_msg):
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "HAL error", error_msg)

        # Use non-blocking async call
        run_qt_async(send_audio_file_async(self.data, 'output.wav'), transcription_success, transcription_error)

    def _handle_transcription_response(self):
        """Handle the response from audio transcription."""
        '''
        Add code to save project specific log on beamline here
        '''
        
        # self.save_notebook()

        self.transcription_str = self.data['voice_cog_output']

        # self.Log_List.insertItem(0, str(self.data['classifier_cog_output']))
        beamline = (self.data['beamline'])
        current_datetime = datetime.now()
        formatted_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        role = (self.data['classifier_cog_output'])
        user_prompt = self.data['text_input'] + ";" + self.data['voice_cog_output']
        log_text = "[{} {}] ({}) {}".format(beamline, formatted_date, role, user_prompt)
        # self.Log_List.append(log_text)

        # print(f"Verification Result {self.data['verification_result']['is_response_justified']}")

        if (self.data['include_verifier'] and not self.data['verification_result']['is_response_justified']):
    
            # Create a formatted string of verification issues
            verification_issues = []
            
            # ANSI escape codes for colors
            RED = '\033[91m'
            YELLOW = '\033[93m'
            ORANGE = '\033[38;5;208m'  # Using 256-color code for orange
            RESET = '\033[0m'
            
            if self.data['verification_result']['missing_info']:
                verification_issues.append(f"{RED}Missing Info:{RESET} " + 
                                        ", ".join(self.data['verification_result']['missing_info']))
            
            if self.data['verification_result']['hallucinations']:
                verification_issues.append(f"{YELLOW}Hallucinations:{RESET} " + 
                                        ", ".join(self.data['verification_result']['hallucinations']))
            
            if self.data['verification_result']['concerns']:
                verification_issues.append(f"{ORANGE}Concerns:{RESET} " + 
                                        ", ".join(self.data['verification_result']['concerns']))
            
            # Add verification issues to log text
            log_text += "\nVERIFICATION FLAGS: " + " | ".join(verification_issues)

        self.Log_List.append(log_text)

        if self.data['next_cog'] == "Op":
            self.CE_List.setPlainText(self.data['operator_cog_output'])
            self.add_CE_history()

        elif self.data['next_cog'] == "Ana":
            self.CE_List.setPlainText(self.data['analysis_cog_output'])
            self.add_CE_history()

        if self.data['next_cog'] == "notebook":
            self.CE_List.setPlainText(self.data['voice_cog_output'])

        # print(self.data)

        # self.save_notebook()

    
    
    @pyqtSlot()
    def recording_started(self):
        print("Recording Started")
        self.Input_Voice.setDisabled(True)
        self.Stop_Voice.setDisabled(False)

        self.add_context_start.setDisabled(True)
        self.add_context_stop.setDisabled(False)

        self.chatbot_start.setDisabled(True)
        self.chatbot_stop.setDisabled(False)
    
    @pyqtSlot()
    def recording_stopped(self):
        print("Recording Stopped")
        self.Input_Voice.setDisabled(False)
        self.Stop_Voice.setDisabled(True)

        self.add_context_start.setDisabled(False)
        self.add_context_stop.setDisabled(True)

        self.chatbot_start.setDisabled(False)
        self.chatbot_stop.setDisabled(True)


    def stop_recording(self, filename="output.wav"):
        sample_rate = self.fs
        
        if not self.is_recording:
            print("No recording in progress")
            return

        print("Recording stopped")
        self.is_recording = False

        # recording_data is undefined, so skip its use here
        # print(recording_data)
        # audio_data = np.concatenate(recording_data, axis = 0)
        audio_data = np.array([], dtype=np.float32)

        # with wave.open(filename, 'wb') as wf:
        #     wf.setnchannels(2)
        #     wf.setsampwidth(2)
        #     wf.setframerate(sample_rate)
        #     wf.writeframes(audio_data.tobytes())

        write('output.wav', self.fs, audio_data)

        
        # global recording_data = []

        self.data['bl_input_channel'] = 0

        def recording_success(response):
            self.data = response
            transcription_str = self.data['audio_output']

            print(transcription_str)

            self.Input_Edit.setText(transcription_str)
            self.Log_List.insertItem(0, transcription_str)
            self.Input_Voice.setEnabled(True)

        def recording_error(error_msg):
            QtWidgets.QMessageBox.critical(self, "HAL error", error_msg)
            self.Input_Voice.setEnabled(True)

        # Use non-blocking async call
        run_qt_async(send_audio_file_async(self.data, 'output.wav'), recording_success, recording_error)

    '''
        Inputs the command into the backend Model (on HAL) and clears the input
    '''
    def inputCommand(self):

        self.data['beamline'] = self.BeamlineDropDown_First_Tab.currentText()

        self.data = self.clear_output_fields(self.data)

        if self.check_add_context.isChecked():
            self.data['include_context_functions'] = True

        else:
            self.data['include_context_functions'] = False

        # if self.check_verifier.isChecked():
        #     self.data['include_verifier'] = True

        # else:
        #     
        self.data['include_verifier'] = False
        
        self.data['bl_input_channel'] = "command"
        # self.data['text_input'] = self.Input_Edit.text()
        self.data['text_input'] = self.Input_Edit.toPlainText()

        if self.data['text_input'] == "":
            return 

        self.data['only_text_input'] = 1

        def command_success(response):
            self.data = self._ensure_dict(response)
            self._handle_command_response()

        def command_error(error_msg):
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "HAL error", error_msg)

        # Use non-blocking async call
        run_qt_async(send_hal_async(self.data), command_success, command_error)

    def _handle_command_response(self):
        """Handle the response from a command input."""
        self.transcription_str = self.data['voice_cog_output']


        # self.Log_List.insertItem(0, str(self.data['classifier_cog_output']))
        beamline = (self.data['beamline'])
        current_datetime = datetime.now()
        formatted_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        role = (self.data['classifier_cog_output'])
        user_prompt = self.data['text_input'] + ";" + self.data['voice_cog_output']
        log_text = "[{} {}] ({}) {}".format(beamline, formatted_date, role, user_prompt)
        self.Log_List.append(log_text)
        verification_flags = 0
        # self.Log_List.append("VERIFICATION FLAGS:")

        if self.data['include_verifier']:
            # Create a formatted string of verification issues            
            if self.data['verification_result']['missing_info']:
                # Join all missing info items with line breaks
                verification_flags = 1
                self.Log_List.append("<b>VERIFICATION FLAGS:</b>")
                missing_info_html = '<br>'.join([html.escape(item) for item in self.data['verification_result']['missing_info']])
                self.Log_List.append(f'<font color="red"> Missing info: </font><font color="blue">{missing_info_html}</font>')
            
            if self.data['verification_result']['hallucinations']:
                # Join all hallucinations items with line breaks
                if verification_flags == 0:
                    self.Log_List.append("VERIFICATION FLAGS:")
                hallucinations_html = '<br>'.join([html.escape(item) for item in self.data['verification_result']['hallucinations']])
                self.Log_List.append(f'<font color="red"> Hallucinations: </font> <font color="blue">{hallucinations_html}</font>')
            
            if self.data['verification_result']['concerns']:
                # Join all concerns items with line breaks
                if verification_flags == 0:
                    self.Log_List.append("VERIFICATION FLAGS:")
                concerns_html = '<br>'.join([html.escape(item) for item in self.data['verification_result']['concerns']])
                self.Log_List.append(f'<font color="red"> Concerns: </font> <font color="blue">{concerns_html}</font>')

            
        if self.data['next_cog'] == "Op":
            self.CE_List.setPlainText(self.data['op_cog_output'])
            self.add_CE_history()

        elif self.data['next_cog'] == "Ana":
            self.CE_List.setPlainText(self.data['ana_cog_output'])
            self.add_CE_history()

            print("=== Generating runXS_llm.py ===")
            print(self.data['ana_cog_output'])
            self.save_runXS()

        # print(self.data)

        self.data['project_path'] = self.projectPathInput.text()

        self.check_create_project_dir()
        
        # self.save_notebook()

    '''
        Submits/Confirms the previously inputted command for the backend to interface with the interface class
    '''
    def submitCommand(self):

        self.data['final_output'] = self.CE_List.toPlainText()
        self.data['bl_input_channel'] = "confirm_code"

        def submit_success(response):
            self.data = response
            if self.data['next_cog'] == "Op":
                self.interface.executeCommand(self.CE_List.toPlainText(), window_name=self.interface.window_Op)
            elif self.data['next_cog'] == "Ana":
                self.interface.executeCommand(self.CE_List.toPlainText(), window_name=self.interface.window_Ana, press_enter=True)
                # self.interface.executeCommand(self.data['ana_cog_output'], window_name="VISION-Ana", press_enter=False)

            #self.resetLog()
            self.save_notebook()

        def submit_error(error_msg):
            QtWidgets.QMessageBox.critical(self, "HAL error", error_msg)

        # Use non-blocking async call
        run_qt_async(send_hal_async(self.data), submit_success, submit_error)

    '''
        Clears the code equivalent list  
    '''
    def codeLog(self, output):
        self.CE_List.insertItem(0, output)



    def updateStatus(self):
        """
        Updates the UI data such as temperature and humidity info
        """
        sys.stdout = open(os.devnull, "w")

        sys.stdout = sys.__stdout__

    # ==== Simulation features ======

    @pyqtSlot()
    def simulate_code(self):
        """Send the code in CE_List to the backend for simulation."""
        if self.sim_in_progress:
            return
        code = self.CE_List.toPlainText().strip()
        if not code:
            return

        request = dict(self.data)               # shallow copy
        request["bl_input_channel"] = "simulate"
        request["code_to_simulate"] = code
        request["beamline"] = self.BeamlineDropDown_First_Tab.currentText()
        # tells backend to request current PV status of beamline and use it in simulation when simulation is run
        request["use_archiver_defaults"] = self.useArchiverCheckBox.isChecked()

        # UI feedback: indicate that simulation is starting
        self.sim_in_progress = True
        self.CE_sim.setEnabled(False)
        self.CE_sim.setText("Starting simulator…")
        self.Log_Confirm.setEnabled(False)
        QtWidgets.QApplication.processEvents()     # force repaint

        # Send request and get first reply
        print(f"[HAL-DEBUG/UI.py] simulate_code sending request: {repr(request)}")

        def simulate_success(first_reply):
            print(f"[HAL-DEBUG/UI.py] simulate_code got first_reply: {repr(first_reply)}")
            self._handle_simulation_response(first_reply, request)

        def simulate_error(error_msg):
            print(f"[HAL-DEBUG/UI.py] simulate_code error: {error_msg}")
            QtWidgets.QMessageBox.critical(self, "HAL error", error_msg)
            self._finish_sim_button(reset_text="Simulate ✗")

        # Use non-blocking async call
        run_qt_async(send_hal_async(request), simulate_success, simulate_error)

    def _handle_simulation_response(self, first_reply, request):
        """Handle the simulation response and start the simulation thread."""
        msg = first_reply[0] if isinstance(first_reply, list) else first_reply
        print(f"[HAL-DEBUG/UI.py] simulate_code using msg: {repr(msg)}")

        # Check if we need to handle PV defaults handshake
        needs_pv_defaults = False
        pv_names = []
        
        # Check for direct pv_defaults_request message
        if msg.get("type") == "pv_defaults_request" and request["use_archiver_defaults"]:
            needs_pv_defaults = True
            pv_names = msg.get("pv_names", [])
        # Also check for awaiting_pv_defaults status (alternative format)
        elif msg.get("status") == "awaiting_pv_defaults" and request["use_archiver_defaults"]:
            # Look for the PV defaults request event in the batch
            for item in first_reply if isinstance(first_reply, list) else [first_reply]:
                if item.get("type") == "pv_defaults_request":
                    needs_pv_defaults = True
                    pv_names = item.get("pv_names", [])
                    break
            
        # If we need to provide PV defaults
        if needs_pv_defaults and pv_names:
            print(f"[UI] Received request for {len(pv_names)} PV defaults")
            # Fetch PV values
            pv_defaults = fetch_pv_values(pv_names, debug=True) # TODO: Remove debug
            
            # Add PV defaults to the request and send again
            request["pv_defaults"] = pv_defaults
            print(f"[HAL-DEBUG/UI.py] Sending request with PV defaults")

            def pv_defaults_success(second_reply):
                print(f"[HAL-DEBUG/UI.py] Got reply after sending PV defaults: {repr(second_reply)}")
                msg2 = second_reply[0] if isinstance(second_reply, list) else second_reply
                print(f"[HAL-DEBUG/UI.py] Using msg after PV defaults: {repr(msg2)}")
                self._start_simulation_thread(msg2, request)
            
            def pv_defaults_error(error_msg):
                print(f"[HAL-DEBUG/UI.py] PV defaults error: {error_msg}")
                QtWidgets.QMessageBox.critical(self, "HAL error", error_msg)
                self._finish_sim_button(reset_text="Simulate ✗")

            # Use non-blocking async call for PV defaults
            run_qt_async(send_hal_async(request), pv_defaults_success, pv_defaults_error)
        else:
            # No PV defaults needed, start simulation directly
            self._start_simulation_thread(msg, request)

    def _start_simulation_thread(self, msg, request):
        """Start the simulation thread with the given message."""
        # Now check for sim_id
        sim_id = msg.get("sim_id")
        if not sim_id:
            print(f"[HAL-DEBUG/UI.py] simulate_code missing sim_id in msg: {repr(msg)}")
            self.Log_List.append(
                "<font color='red'>Simulation could not be started.</font>")
            return

        self.sim_start_time = time.time()
        self.sim_thread = SimStreamThread(sim_id)
        self.sim_thread.sig_pv.connect(self._add_pv_row)
        self.sim_thread.sig_error.connect(self._sim_error)
        self.sim_thread.sig_completed.connect(self._sim_completed)
        self.sim_thread.sig_running.connect(self._sim_running)

        self.sim_dialog = SimStatusDialog(self)
        # Store the original code and query for evaluation
        self.sim_original_code = request["code_to_simulate"]
        # Get the current text from the input field, even if not submitted
        current_text = self.Input_Edit.toPlainText().strip()
        self.sim_original_query = current_text if current_text else self.data.get('text_input', '')
        self.sim_dialog.show()
        self.sim_dialog.raise_()
        self.sim_dialog.activateWindow()

        # callback for when dialog is closed
        self._sim_dialog_closed = self._on_sim_dialog_close

        # Update label if status is already "running"
        if msg.get("type") == "status" and msg.get("state") == "running":
            self._sim_running()

        print("[HAL-DEBUG/UI.py] simulate_code: starting SimStreamThread.")
        self.sim_thread.start()

    @pyqtSlot(str, float, float, object)
    def _add_pv_row(self, pv, dt, t, val):
        """
        Handles new PV event from backend. Expects a dict with keys: value, elapsed_time, delta_time.
        Adds debug prints for diagnosis.
        """
        pv_str = str(pv) if pv is not None else ""

        self.sim_dialog.add_pv(
            pv_str, val,
            elapsed_time=t,
            delta_time=dt,
        )

    @pyqtSlot()
    def _sim_running(self):
        self.sim_dialog.set_running()

    @pyqtSlot(str)
    def _sim_error(self, message):
        self.sim_dialog.set_error(message)
        self._finish_sim_button(reset_text="Simulate ⚠")

    @pyqtSlot(float)
    def _sim_completed(self, duration):
        self.sim_dialog.set_completed(duration)
        self._finish_sim_button()

    def _request_sim_evaluation(self):
        """Send request to backend to evaluate the simulation"""
        if not hasattr(self, "sim_thread") or not self.sim_thread.sim_id:
            return
            
        eval_request = {
            "bl_input_channel": "evaluate_simulation",
            "sim_id": self.sim_thread.sim_id,
            "beamline": self.BeamlineDropDown_First_Tab.currentText(),
            "original_query": self.sim_original_query,
            "generated_code": self.sim_original_code
        }
        
        def eval_success(response):
            response = self._ensure_dict(response)
            evaluation = response.get("evaluation", "No evaluation received")
            if hasattr(self, "sim_dialog"):
                self.sim_dialog.show_evaluation(evaluation)
        
        def eval_error(error_msg):
            if hasattr(self, "sim_dialog"):
                self.sim_dialog.show_evaluation(f"Error requesting evaluation: {error_msg}")
        
        # Use non-blocking async call
        run_qt_async(send_hal_async(eval_request), eval_success, eval_error)

    def _finish_sim_button(self, reset_text="Simulation ✓"):
        self.sim_in_progress = False
        self.CE_sim.setEnabled(True)
        self.CE_sim.setText(reset_text)
        self.Log_Confirm.setEnabled(True)
        if hasattr(self, "sim_dialog"):
            self.sim_dialog.btn_close.setEnabled(True)

    def _on_ce_code_changed(self):
        # Called whenever code equivalent box is edited.
        if not self.CE_sim.isEnabled() or self.CE_sim.text() != "Simulate":
            self.CE_sim.setEnabled(True)
            self.CE_sim.setText("Simulate")
        self.sim_in_progress = False

    def closeEvent(self, event):
        """
        Override closeEvent to clean up resources when the application closes.
        """
        try:
            # Clean up the thread pool executor
            cleanup_executor()
            print("[UI] Thread pool executor cleaned up")

            # Clean up the Qt async helper
            cleanup_qt_async()
            print("[UI] Qt async helper cleaned up")
        except Exception as e:
            print(f"[UI] Error during cleanup: {e}")

        # Call the parent closeEvent
        super().closeEvent(event)

