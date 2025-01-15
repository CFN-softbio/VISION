from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QDialog, QPushButton, QLineEdit, QListWidget, QLabel, QTextEdit, QComboBox, QCheckBox
from PyQt5.QtGui import QPixmap

#Sound Recording and Transcriptions
import sounddevice as sd
# import torch
import numpy as np

import sys, os
import matplotlib.pyplot as plt 

from scipy.io.wavfile import write
import wave
import pandas as pd
from datetime import datetime
import html
import ast

# Don't know if this path contains special credentials, if it does we should move them to ENV variables
# rather than using dynamic path changes. For now, you can uncomment this and comment the relative import line after.
# sys.path.insert(0, '/nsls2/data/cms/legacy/xf11bm/data/2024_2/beamline/PTA/test/S3_test')
# from send_to_hal import send_audio, send_audio_file, send_hal, receive

from S3_test.send_to_hal import send_audio, send_audio_file, send_hal, receive

import time
import threading
import pyaudio as pa

from PyQt5.QtCore import QMetaObject, Qt, QObject, QThread, pyqtSignal, pyqtSlot



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
        stream = audio.open(
            format = pa.paInt16,
            channels = 1,
            rate = 48000,
            input = True,
            frames_per_buffer = 1024
            
        )

        self.stopped = False
        self.sig_started.emit()

        while not self.stopped:
            data = stream.read(1024)
            frames.append(data)

        stream.close()

        self.sig_stopped.emit()

        exists= True
        i = 1
        while exists:
            if os.path.exists(f"recording{i}.wav"):
                i+=1
            else:
                exists=False

        # wf = wave.open(f"recording{i}.wav", 'wb')

        wf = wave.open("output.wav", 'wb')

        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pa.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))
        wf.close()

        # self.sig_transcription.emit(f'recording{i}.wav')
        
        self.sig_transcription.emit()

    @pyqtSlot()
    def stop(self):
        self.stopped = True


class Ui_MainWindow(QDialog):

    def __init__(self, isInterfacing=False):

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

        self.data = {

        # 'datetime': Base().now(),
        'project_path': "",
        'terminate': 0,
        'bl_conf': 0,
        'only_text_input':0,
        'context':[],
        'errors': [],
        'cog_id_error': [],
        'beamline': '11BM',
        'bl_input_channel': "command",
        'text_input': "",


        'voice_cog_output': "",
        'classifier_cog_output': "",
        'op_cog_output':"",
        'ana_cog_output': "",
        'refinement_cog_output':"",


        'voice_cog_input': "",
        'voice_cog_output': "",
        'classifier_cog_input': "",
        'classifier_cog_output': "",
        'operation_cog_input':"",
        'operation_cog_output':"",
        'analysis_cog_input': "",
        'analysis_cog_output': "",
        'refinement_cog_input':"",   
        'refinement_cog_output':"",


        'include_context_functions': 0,
        'status': 'success'

        }

        self.context_data = {

        # 'datetime': Base().now(),
        'context_data':"",
        'only_text_input': 1,
        'bl_input_channel': "add_context",
        'text_input': "",
        'voice_cog_output': "",
        'classifier_cog_output': "",
        'op_cog_output':"",
        'ana_cog_output': "",
        'refinement_cog_output':"",
        'status': 'success',


         'project_path': "",
        'terminate': 0,
        'bl_conf': 0,
        'context':[],
        'errors': [],
        'cog_id_error': [],
        'beamline': '11BM',
        'text_input': "",


        'voice_cog_output': "",
        'classifier_cog_output': "",
        'op_cog_output':"",
        'ana_cog_output': "",
        'refinement_cog_output':"",


        'voice_cog_input': "",
        'voice_cog_output': "",
        'classifier_cog_input': "",
        'classifier_cog_output': "",
        'operation_cog_input':"",
        'operation_cog_output':"",
        'analysis_cog_input': "",
        'analysis_cog_output': "",
        'refinement_cog_input':"",   
        'refinement_cog_output':"",


        'include_context_functions': 0,
        'status': 'success'


        }

        self.chat_data = {
            'bl_input_channel': 'chatbot',
            'text_input': "",
            "history":"",
            'only_text_input':1,
            'voice_cog_output': ""}
    

        #Voice module
        if voice:
            print("--Loading Voice Module\n")
            # self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
            # self.model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

            
            
            self.fs = 48000 #16000
            # self.fs = 16000
            self.duration = 5
            sd.default.samplerate = self.fs
            sd.default.channels = 2
            sd.default.device = [0, 4]
            print('===') 
            print(sd.query_devices())   
            print('===')         

        #Connect everything here to the UI components
        self.Input_Submit = self.findChild(QPushButton, "Input_Submit")
        

        self.Input_Edit = self.findChild(QTextEdit, "Input_Edit") #QLineEdit
        self.Input_Edit.setLineWrapMode(QTextEdit.WidgetWidth)
        self.projectPathInput = self.findChild(QLineEdit, "projectPathInput")
        # self.ASR_text = self.findChild(QLabel, "ASR_text")
        self.Input_Submit.clicked.connect(self.inputCommand)

        # self.Log_List = self.findChild(QListWidget, "Log_List")
        self.Log_List = self.findChild(QTextEdit, "Log_List")
        self.Log_Confirm = self.findChild(QPushButton, "Log_Confirm")
        self.Log_Clear = self.findChild(QPushButton, "Log_Clear")

        self.Log_Confirm.clicked.connect(self.submitCommand)
        self.Log_Clear.clicked.connect(self.resetLog)

        # self.CE_List = self.findChild(QListWidget, "CE_List")

        self.CE_List = self.findChild(QTextEdit, "CE_List")

        self.Sample_Log = self.findChild(QLabel, "Sample_Log")
        self.Temperature_Log = self.findChild(QLabel, "Temperature_Log")
        self.Humidity_Log = self.findChild(QLabel, "Humidity_Log")
        self.Position_Log = self.findChild(QLabel, "Position_Log")

        self.BeamlineTextLabel = self.findChild(QLabel, "label_beamline")
        # self.BeamlineTextBox = self.findChild(QTextEdit, "text_beamlineID")
        self.BeamlineDropDown = self.findChild(QComboBox, "dropdown_beamline")
        self.SelectedCog = self.findChild(QComboBox, "dropdown_cog")
        self.add_context_output = self.findChild(QTextEdit, "text_output")
        self.add_context_result = self.findChild(QTextEdit, "result_message_box")

        self.add_context_input = self.findChild(QLineEdit, "input_text_box")
        self.add_context_submit = self.findChild(QPushButton, "submit_add_context_text_button")
        self.add_context_submit.clicked.connect(self.submit_context)


        if voice:
            self.Input_Voice = self.findChild(QPushButton, "Input_Voice")
            # self.Input_Voice.clicked.connect(self.voiceInput)
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

        #UI variables used in interface with other scripts
        self.samples = []
        self.statusTimer = QtCore.QTimer()
        self.dataTimer_qr = QtCore.QTimer()
        self.dataTimer = QtCore.QTimer()
        self.isInterfacing = isInterfacing

        # self.dataTimer_qr.timeout.connect(lambda : self.getDataQR())
        # self.dataTimer_qr.start(1000) #Milliseconds

        self.dataTimer.timeout.connect(lambda : self.getData())
        self.dataTimer.start(1000) #Milliseconds

        self.currentFile_qr = ""
        self.currentFile = ""

        # self.classificationLabel = self.findChild(QLabel, "Classification_Label")

        # self.dataDirectory_qr = "/nsls2/data/cms/legacy/xf11bm/data/2024_3/beamline/ETsai/saxs/analysis/q_image/"
           
        self.dataDirectory = "" #"/nsls2/data/cms/legacy/xf11bm/data/2024_3/beamline/ETsai/saxs/analysis/circular_average/"

        # self.recording_data = []

        self.transcription_str = ""

        # self.worker = Worker(self.fs)
        # self.thread = QThread()

        # self.worker.moveToThread(self.thread)

        

        # self.worker.transcription_signal.connect(self.update_transcription)

    
    # def send_chat_message(self):
    #     # Get the user's input
    #     message = self.user_input.text()

    #     # Check if the message is not empty or whitespace
    #     if message.strip():
    #         # Display the user message in the chat display
    #         self.chat_display.append(f"User: {message}")

    #         # Clear the input field after sending
    #         self.user_input.clear()

    #         # Send the message to HAL and receive the response
    #         self.chat_data['message'] = message
    #         self.chat_data = send_hal(self.chat_data)

    #         # Display the response from HAL in the chat display
    #         response = self.chat_data.get('chatbot_response', 'No response from HAL')
    #         # Escape any HTML special characters and replace newlines with <br> to preserve formatting
    #         response_html = html.escape(response).replace('\n', '<br>')

    #         # Display the response in blue color, preserving newlines
    #         self.chat_display.append(f'<font color="blue">{response_html}</font>')


    '''
        Sets the backend model to interface with in the execution
    '''
    # def setBackendModel(self, backend):
    #     self.backendModel = backend

    '''
        Display latest analysis result
    '''
    def getData(self, verbose=1):

        firstFile = True
        mostRecentFile = 0
        self.check_create_project_dir()
        self.dataDirectory = self.data['project_path'] #+"/saxs/analysis/" 
        print(self.dataDirectory)

        png_files = []
        for root, _, files in os.walk(self.dataDirectory):

            for file in files:
                if file.lower().endswith('.png'):
                    png_files.append(os.path.join(root, file))
            
        if verbose>0: print("Numer of PNG files: {}".format(len(png_files)))

        if(len(png_files) > 0):
            png_files = sorted(png_files, key=lambda x: os.path.getmtime(x), reverse=True) 
            self.currentFile =  png_files[0]
            if verbose>0: print(self.currentFile)

            self.ana_text.setText(self.currentFile)
            pixmap = QPixmap(self.currentFile)
            self.data_label.setPixmap(pixmap)

                # mostRecent = files[0]
                # mostRecentTime = os.path.getmtime(root + "/"+ files[0])

                # for i in range(len(files)):
                #     currentTime = os.path.getmtime(root + "/" + files[i])

                #     if(currentTime > mostRecentTime):
                #         mostRecent = files[i]
                #         mostRecentTime = currentTime

                # if(self.currentFile != mostRecent):

                #     self.currentFile = mostRecent

                #     directory = root + mostRecent
                    
                #     dataX = []
                #     dataY = []

                    # if(mostRecent[len(mostRecent) - 3:len(mostRecent)] == "dat"):
                    #     for line in open(directory):
                    #         if(not "#" in line):
                    #             rawData = line.split(" ")
                    #             dataX.append(float(rawData[0]))
                    #             dataY.append(float(rawData[1]))
                    #     #plt.plot(dataX, dataY)
                    #     #plt.savefig("../graph.png")
                    #     #plt.clf()
                    #     pixmap = QPixmap(self.data['project_path']+"/curve_1d.png")
                    #     self.data_label.setPixmap(pixmap)
                    #     #self.classificationLabel.setText(self.backendModel.classifyData([dataX, dataY]))

                    # if(mostRecent[len(mostRecent) - 3:len(mostRecent)] == "png"):
                    #     pixmap = QPixmap(root + "/" + self.currentFile)
                    #     self.data_label.setPixmap(pixmap)

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
        self.Input_Edit.clear()
        # self.backendModel.clearCommands()

    # def logOutput(self, modelOutput):
    #     #print('logOutput: {}'.format(modelOutput))
    #     self.Log_List.insertItem(0, modelOutput)

    def send_hal_trans_conf(self):

        self.data['bl_conf'] = 1
        self.data = send_hal(self.data)

        print(self.data['op_output'])

    def send_hal_trans_redo(self):

        self.data['bl_conf'] = 0
        self.voiceInput()
        # self.data = send_hal(self.data)

    '''
        Voice Stuff :)
    '''

    # def voiceInput(self):
    #     print(self.worker.recording)
    #     if self.worker.recording:
    #         self.worker.stop_recording()
    #         self.Input_Voice.setStyleSheet("QPushButton"
    #         "{"
    #         "background-color : blue;"
    #         "}"
    #         )
    #         self.Input_Voice.setEnabled(True)

    #     else:
    #         self.worker.recording = True
    #         self.Input_Voice.setStyleSheet("QPushButton"
    #         "{"
    #         "background-color : red;"
    #         "}"
    #         )
            

    #         # self.worker = Worker(self.fs)
    #         # # self.thread = QThread()

    #         # self.worker.moveToThread(self.thread)

    #         # self.worker.transcription_signal.connect(self.update_transcription)
    #         self.worker.sig_started.connect(self.recording_started)
    #         self.worker.sig_stopped.connect(self.recording_stopped)
    #         self.thread.started.connect(self.worker.start_recording)
    #         self.thread.start()


    #This is for the submit button (text) on Add Context Functions tab
    def submit_context(self):
        
        self.context_data['only_text_input'] = 1
        self.context_data['bl_input_channel'] = "add_context"
        self.context_data['text_input'] = self.add_context_input.text()

        self.context_data['beamline_id'] = self.BeamlineDropDown.currentText()
        self.context_data['selected_cog'] = self.SelectedCog.currentText()

        self.context_data = send_hal(self.context_data)


        self.add_context_output.setPlainText(self.context_data['refinement_cog_display_output'])

        print(self.context_data)

    
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

        self.context_data = send_hal(self.context_data)

        print("Added context functions for Beamline: {} and Cog: {}".format(self.context_data['beamline_id'],
        self.context_data['selected_cog']))

        # self.add_context_output.setPlainText(f"Successfully added {self.context_data['refinement_cog_output_dict']['input']} to {self.context_data['selected_cog']} Cog at {self.context_data['beamline_id']} beamline!!")

        # self.add_context_output.setPlainText(self.context_data['append_examples_output'])

        self.add_context_result.setHtml(formatted_date + self.context_data['append_examples_output'])

        self.check_create_project_dir()
        self.save_notebook_tab3()



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
        self.context_data = send_audio_file(self.context_data, 'output.wav')
        self.context_data['only_text_input'] = 1
        # self.add_context_output.setPlainText(self.context_data['audio_output'])
        self.add_context_output.setPlainText(self.context_data['refinement_cog_display_output'])

        

        print(self.context_data)

    @pyqtSlot()
    def transcribe_chatbot(self):
        
        self.chat_data['only_text_input'] = 0
        self.chat_data['text_input'] = self.user_input.text()
        # self.context_data['beamline_id'] = self.BeamlineTextBox.toPlainText()
        # self.context_data['beamline_id'] = self.BeamlineDropDown.currentText()

        # print(self.context_data)
        self.chat_data = send_audio_file(self.chat_data, 'output.wav')
        self.chat_data['only_text_input'] = 1
        self.chat_display.append(f"\nUser: {self.chat_data['prompt']}")

        self.show_chatbot_response()
        


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
            self.chat_data = send_hal(self.chat_data)

            self.show_chatbot_response()

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
                print(f"Directiry created: {path}")
            except Exception as e:
                print(f"Project Path set to current working directory: {path}")

        # else:
        #     print(f"Found directory: {path}")

    def clear_output_fields(self, data):
        for key in list(data.keys()):
            if key.endswith("_output"):
                data[key] = ""
        return data
    
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
    
    @pyqtSlot()
    def update_transcription(self):
        # self.ASR_text.setText(transcription_str)

        self.data = self.clear_output_fields(self.data)

        self.data['bl_conf'] = 0
        self.data['bl_input_channel'] = "command"

        self.data['project_path'] = self.projectPathInput.text()

        self.check_create_project_dir()

        if self.check_add_context.isChecked():
            self.data['include_context_functions'] = 1

        else:
            self.data['include_context_functions'] = 0

        self.data['text_input'] = self.Input_Edit.toPlainText()
        self.data = send_audio_file(self.data, 'output.wav')

        '''
        Add code to save project specific log on beamline here
        '''
        
        # self.save_notebook()

        self.transcription_str = self.data['voice_cog_output']

        #self.ASR_text.setText(self.transcription_str)

        # self.Log_List.insertItem(0, str(self.data['classifier_cog_output']))
        beamline = (self.data['beamline'])
        current_datetime = datetime.now()
        formatted_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        role = (self.data['classifier_cog_output'])
        user_prompt = self.data['text_input'] + ";" + self.data['voice_cog_output']
        log_text = "[{} {}] ({}) {}".format(beamline, formatted_date, role, user_prompt)
        self.Log_List.append(log_text)

        if self.data['next_cog'] == "Op":
            # self.CE_List.insertItem(0, self.data['op_cog_output'])
            self.CE_List.setPlainText(self.data['operator_cog_output'])
        elif self.data['next_cog'] == "Ana":
            # self.CE_List.insertItem(0, self.data['ana_cog_output'])
            self.CE_List.setPlainText(self.data['analysis_cog_output'])
        if self.data['next_cog'] == "notebook":
            self.CE_List.setPlainText(self.data['voice_cog_output'])

        print(self.data)

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

    
    # def voiceInput(self, duration=5, channels=2):

    #     if self.recording:
    #         self.recording = False
    #         print("Stopped Recording")

    #         self.Input_Voice.setStyleSheet("QPushButton"
    #         "{"
    #         "background-color : blue;"
    #         "}"
    #         )

    #         self.ASR_text.setText(self.transcription_str)
    #         self.Log_List.insertItem(0, self.transcription_str)
    #         self.Input_Voice.setEnabled(True)
    #         self.Input_Voice.setStyleSheet("QPushButton"
    #     "{"
    #     "background-color : lightblue;"
    #     "}"
    #     )     
    #         # self.Input_Edit.setText(self.transcription_str)
    #     # self.Input_Edit.insertItem(0, transcription_str)
            
            
            
    #     else:
    #         self.recording=True
    #         print("Start Recording")
    #         self.Input_Voice.setStyleSheet("QPushButton"
    #         "{"
    #         "background-color : red;"
    #         "}"
    #         )
    #         # self.button.config(fg="red")
    #         threading.Thread(target=self.record).start()
    #         # self.record()

        





    # def record(self):
    #     audio = pyaudio.PyAudio()
    #     stream = audio.open(format=pyaudio.paInt16, channels = 1, rate = self.fs,
    #     input = True, frames_per_buffer=1024)

    #     frames = []

    #     start = time.time()

    #     while self.recording:
    #         data = stream.read(1024)
    #         frames.append(data)
    #         passed = time.time() - start

    #         secs = passed % 60
    #         mins = passed // 60
    #         hours = mins // 60

    #         # self.label.config(text = f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}")

        
    #     stream.stop_stream()
    #     stream.close()
    #     audio.terminate()

    #     exists= True
    #     i = 1
    #     while exists:
    #         if os.path.exists(f"recording{i}.wav"):
    #             i+=1
    #         else:
    #             exists=False

    #     sound_file = wave.open(f"output.wav", "w")
    #     sound_file.setnchannels(1)
    #     sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    #     sound_file.setframerate(self.fs)
    #     sound_file.writeframes(b"".join(frames))
    #     sound_file.close()

    #     self.data['bl_input_channel'] = 0

    #     self.data = send_audio_file(self.data, 'output.wav')

    #     self.transcription_str = self.data['audio_output']
        

    #     print(self.transcription_str)

        

        


        

        


        
    
    # def voiceInput(self):
        
    #     self.Input_Voice.setEnabled(False)
    #     verbose = 1

    #     if verbose:
    #         print('Taking voice input..')

        

    

        

        # myrecording = sd.rec(int(self.duration * self.fs), samplerate=self.fs) #, dtype=np.float32)
        # sd.wait()

        # write('output.wav', self.fs, myrecording)

        # self.data['bl_input_channel'] = 0

        # self.data = send_audio_file(self.data, 'output.wav')

        # transcription_str = self.data['audio_output']
        

        # print(transcription_str)





        #Convert data into usable data for transcription
        # audio = np.empty([0])

        

        # for entry in myrecording:
        #     audio = np.insert(audio, len(audio), entry[0])

        # transcription_str = send_audio(audio)

        # input_values = self.processor(audio, return_tensors="pt", padding="longest").input_values

        # # retrieve logits
        # logits = self.model(input_values).logits

        # # take argmax and decode
        # predicted_ids = torch.argmax(logits, dim=-1)
        # transcription = self.processor.batch_decode(predicted_ids)
        # transcription_str = ''.join(transcription) # Convert list to string
        # if verbose:
        #     print('Done taking voice input..')
        #     print(transcription)

        # self.Input_Edit.setText(transcription_str)
        # # self.Input_Edit.insertItem(0, transcription_str)
        # self.ASR_text.setText(transcription_str)
        # self.Log_List.insertItem(0, transcription_str)
        # self.Input_Voice.setEnabled(True)
        # if verbose:
        #     print('Done Voice.')

    
    def stop_recording(self, filename="output.wav"):
        sample_rate = self.fs
        
        if not self.is_recording:
            print("No recording in progress")
            return

        print("Recording stopped")
        self.is_recording = False

        print(recording_data)

        audio_data = np.concatenate(recording_data, axis = 0)

        # with wave.open(filename, 'wb') as wf:
        #     wf.setnchannels(2)
        #     wf.setsampwidth(2)
        #     wf.setframerate(sample_rate)
        #     wf.writeframes(audio_data.tobytes())

        write('output.wav', self.fs, audio_data)

        
        # global recording_data = []

        self.data['bl_input_channel'] = 0

        self.data = send_audio_file(self.data, 'output.wav')

        transcription_str = self.data['audio_output']
        

        print(transcription_str)

        self.Input_Edit.setText(transcription_str)
        # self.Input_Edit.insertItem(0, transcription_str)
        # self.ASR_text.setText(transcription_str)
        self.Log_List.insertItem(0, transcription_str)
        self.Input_Voice.setEnabled(True)

    '''
        Add and Remove functionality for the Sample and Holder Logs and Delete buttons
    '''
    # def addSampleFromUI(self):

    #     sampleName = self.Sample_Name_Text.text()
    #     listOfSamples = [self.Samples_Log.item(i).text() for i in range(self.Samples_Log.count())]
    #     listOfHolders = [self.Holder_Log.item(i).text() for i in range(self.Holder_Log.count())]

    #     if(len(listOfHolders) == 0):
    #         self.logOutput("Please create a holder")
    #     elif(self.Holder_Log.currentRow() == -1):
    #         self.logOutput("Please select a holder")
    #     elif(sampleName == ""):
    #         self.logOutput("Please input valid Sample Name")
    #     elif(sampleName in listOfSamples):
    #         self.logOutput("Please input unique Sample Name")
    #         self.Sample_Name_Text.setText("")
    #     else:
    #         currentHolderIndex = self.Holder_Log.currentRow()
    #         self.Samples_Log.addItem(sampleName)
    #         self.backendModel.addSample(currentHolderIndex, sampleName)
    #         self.Sample_Name_Text.setText("")

    # def deleteSample(self):
    #     if(self.Samples_Log.item(self.Samples_Log.currentRow()) != None and self.Holder_Log.item(self.Holder_Log.currentRow()) != None):

    #         holderIndex = self.Holder_Log.currentRow()
    #         sampleIndex = self.Samples_Log.currentRow()

    #         self.backendModel.deleteSample(holderIndex, sampleIndex)
    #         self.Samples_Log.takeItem(sampleIndex)
    #         self.Samples_SelectedData.clear()
    #     else:
    #         self.logOutput("Please Select a Sample to Remove")

    # #Gets the name of the current Holder
    # def getCurrentHolder(self):

    #     if(self.Holder_Log.currentRow() == -1):
    #         return -1
    #     else:
    #         return self.Holder_Log.item(self.Holder_Log.currentRow()).text()

    # #Gets the index of the current Sample
    # def getCurrentSample(self):

    #     if(self.Samples_Log.currentRow() == -1):
    #         return -1
    #     else:
    #         return self.Samples_Log.item(self.Samples_Log.currentRow()).text()

    # #Updates the sample data on the GUI
    # def getCurrentSampleData(self):
    #     self.Samples_SelectedData.clear()

    #     holderIndex = self.Holder_Log.currentRow()
    #     sampleIndex = self.Samples_Log.currentRow()

    #     sample = self.backendModel.getSampleData(holderIndex, sampleIndex)

    #     self.Samples_SelectedData.addItem("Incident Angles: " + str(sample.angles))
    #     self.Samples_SelectedData.addItem("Exposure time: " + str(sample.time))
    #     self.Samples_SelectedData.addItem("Process: " + str(sample.process))

    # def addHolderFromUI(self):

    #     holderName = self.Holder_Name_Text.text()
    #     listOfHolders = [self.Holder_Log.item(i).text() for i in range(self.Holder_Log.count())]

    #     if(holderName == ""):
    #         self.logOutput("Please input valid Holder Name")
    #     elif(holderName in listOfHolders):
    #         self.logOutput("Please input unique Holder Name")
    #         self.Holder_Name_Text.setText("")
    #     else:
    #         self.Holder_Log.addItem(holderName)
    #         self.backendModel.addHolder(holderName)
    #         self.Holder_Name_Text.setText("")

    # def deleteHolder(self):

    #     if(self.Holder_Log.currentRow() != -1):
    #         holderIndex = self.Holder_Log.currentRow()

    #         self.backendModel.deleteHolder(holderIndex)
    #         self.Holder_Log.takeItem(holderIndex)

    #         self.Samples_Log.clear()
    #         self.Samples_SelectedData.clear()

    # def getCurrentHolderData(self):

    #     self.Samples_Log.clear()

    #     holderIndex = self.Holder_Log.currentRow()

    #     holder = self.backendModel.getHolderData(holderIndex)

    #     for sample in holder.getSampleList():
    #         self.Samples_Log.addItem(sample.name)
        
    '''
        Inputs the command into the backend Model and clears the input
    '''
    def inputCommand(self):
        # inputText = self.Input_Edit.text()
        # self.Input_Edit.clear()
        # if inputText is "":
        #     inputText = self.ASR_text.text()
        #     print("ASR test = {}".format(inputText))
        # self.backendModel.processInput(inputText)

        self.data = self.clear_output_fields(self.data)


        if self.check_add_context.isChecked():
            self.data['include_context_functions'] = 1

        else:
            self.data['include_context_functions'] = 0
        
        self.data['bl_input_channel'] = "command"
        # self.data['text_input'] = self.Input_Edit.text()
        self.data['text_input'] = self.Input_Edit.toPlainText()

        if self.data['text_input'] == "":
            return 

        self.data['only_text_input'] = 1
        self.data = send_hal(self.data)

        self.transcription_str = self.data['voice_cog_output']

        # self.ASR_text.setText(self.transcription_str)

        # self.Log_List.insertItem(0, str(self.data['classifier_cog_output']))
        beamline = (self.data['beamline'])
        current_datetime = datetime.now()
        formatted_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        role = (self.data['classifier_cog_output'])
        user_prompt = self.data['text_input'] + ";" + self.data['voice_cog_output']
        log_text = "[{} {}] ({}) {}".format(beamline, formatted_date, role, user_prompt)
        self.Log_List.append(log_text)


        if self.data['next_cog'] == "Op":
            # self.CE_List.insertItem(0, self.data['op_cog_output'])
            self.CE_List.setPlainText(self.data['op_cog_output'])
        elif self.data['next_cog'] == "Ana":
            # self.CE_List.insertItem(0, self.data['ana_cog_output'])
            self.CE_List.setPlainText(self.data['ana_cog_output'])

        print(self.data)

        self.data['project_path'] = self.projectPathInput.text()

        self.check_create_project_dir()
        
        # self.save_notebook()



    '''
        Submits/Confirms the previously inputted command for the backend to interface with the interface class
    '''
    def submitCommand(self):

        self.data['final_output'] = self.CE_List.toPlainText()
        self.data['bl_input_channel'] = "confirm_code"
        self.data = send_hal(self.data)

        if self.data['next_cog'] == "Op":
            self.interface.executeCommand(self.CE_List.toPlainText(), window_name=self.interface.window_Op)
        elif self.data['next_cog'] == "Ana":
            self.interface.executeCommand(self.CE_List.toPlainText(), window_name=self.interface.window_Ana, press_enter=True)
            # self.interface.executeCommand(self.data['ana_cog_output'], window_name="VISION-Ana", press_enter=False)
        
        
        #self.backendModel.command(True)
        #self.resetLog()

        self.save_notebook()

    '''
        Clears the code equivalent list and then submits the parameter string output 
    '''
    def codeLog(self, output):
        #self.CE_List.clear()
        #self.CE_List.addItem(output)
        self.CE_List.insertItem(0, output)

    '''
        Shows the selected Sample Data
    '''
    # def showData(self):

    #     self.SelectedData_List.clear()

    #     sampleName = self.Samples_Log.currentRow()

    #     for sample in self.samples:
    #         if sample["name"] == sampleName:
    #             output = ""

    #             output += "Sample Name: " + str(sample["name"])
    #             output += "Angles: " + str(sample["angles"])
    #             output += "Exposure Time: " + str(sample["exposure time"])
    #             output += "Process: " + str(samplep["process"])

    '''
        Updates the UI data such as temperature and humidity info
    '''
    def updateStatus(self):
        
        sys.stdout = open(os.devnull, "w")

        #Sample Name
        #if(self.Samples_Log.currentRow() != -1):
        #    self.Sample_Log.setText(str(self.Samples_Log.currentRow()))
        #else:
        # try:
        #    self.Sample_Log.setText(str(self.interface.getValue("sample")))
        # except:
        #    self.Sample_Log.setText("NA")
        # self.Sample_Log.setText(str(self.interface.getValue("sample")))
        
        # # #Temperature and Humidity
        # self.Temperature_Log.setText(str(self.interface.getValue("temperature")))
        # self.Humidity_Log.setText(str(self.interface.getValue("humidity")))

        # #Position
        # self.Position_Log.setText(self.interface.getValue("position"))

        sys.stdout = sys.__stdout__
