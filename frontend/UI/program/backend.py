import numpy as np 
import random 
import pandas as pd 
# import torch 
from tqdm import tqdm 

import sys

from PyQt5 import QtCore, QtGui, QtWidgets
#from UI import Ui_MainWindow
from datastructures import Holder, Sample
#from curveClassifier import curveClassification

# from transformers import AutoTokenizer 
# from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

class backend:

    #Initialization
    def __init__(self, temperature=0, humidity=0, sample = "NA", position = "(0,0)", process = "NA", isInterfacing=False, data_dir="../training/data/"):

        #Stores current command in case user wants to submit command
        self.currentCommands = []

        #Dictionaries to convert from labels to indicies and back
        self.labels_to_ids = {}
        self.ids_to_labels = {}

        self.extractID(data_dir+"/labels_to_ids.txt")

        #max padding length
        self.MAX_LEN = 200 

        #Loads the various components needed
        # print("\nLoading tokenizer\n")
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
        # print("\nLoading Model\n")
        # self.model = AutoModelForTokenClassification.from_pretrained("../training/model/")
        # print("\nLoading Trainer\n")
        # self.trainer = Trainer(model=self.model, tokenizer=self.tokenizer)

        self.modelPredict("")

        #We manage samples and holders through the backend class
        self.samples = []
        self.holders = []

        self.isInterfacing = isInterfacing

        #self.cc = curveClassification('../training/CCWeights/cc')

        print("\n\n")

    '''
        Setter functions for the UI and interface
    '''
    def setUI(self, ui):
        self.ui = ui

    def setInterface(self, interface):
        if self.isInterfacing:
            self.interface = interface
        else:
            self.interface = None

    # '''
    #     Functions to deal with samples
    # '''
    # def deleteSample(self, holderIndex, sampleIndex):
    #     self.holders[holderIndex].removeSample(sampleIndex)
    #     self.interface.executeCommand("$SampleList.pop(" + str(sampleIndex) + ")")
    
    # def addSample(self, holderIndex, sampleName):
    #     newSample = Sample(sampleName, interface=self.interface)
    #     self.holders[holderIndex].addSample(newSample)

    # def getSampleData(self, holderIndex, sampleIndex):
    #     return self.holders[holderIndex].getSampleList()[sampleIndex]

    # '''
    #     Functions to deal with holders
    # '''
    # def addHolder(self, holderName):
    #     newHolder = Holder(holderName)
    #     self.holders.append(newHolder)

    # def getHolderData(self, holderIndex):
    #     return self.holders[holderIndex]

    # def deleteHolder(self, holderIndex):
    #     self.holders[holderIndex].pop()

    '''
        Clears the current commands
    '''
    def clearCommands(self):
        self.currentCommands = []

    '''
        Function that extracts the constructed ID and label dictionaries in the datagenerator class during training for use here
    '''
    def extractID(self, directory):

        with open(directory, "r") as f:
            for line in f:
                array = line.split(":")

                ID = array[1]
                label = array[0]

                if("\n" in ID):
                    self.ids_to_labels[int(ID[:ID.index("\n")])] = label
                    self.labels_to_ids[label] = int(ID[:ID.index("\n")])
                else:
                    self.ids_to_labels[int(ID)] = label
                    self.labels_to_ids[label] = int(ID)

    def classifyData(self, data):
        result =  self.cc.classifyData(data)[0]
        print(result)
        
        result_str = "Diffuse: {0}\nSimple: {1}%\nPeriodic: {2}%\nComplex: {3}%".format(str(abs(result[0]))[2:4],str(abs(result[1]))[2:4],str(abs(result[2]))[2:4],str(abs(result[3]))[2:4])

        return result_str


    '''
        Cleans up and tokenizes input to feed into NER Model to get an output, detokenizing and cleaning up for use
    '''
    def modelPredict(self, input):
        return
    #   tokenized_sentence = []
    #   tokens = []

    #   for raw_word in input.split():
    #         word = raw_word.lower()
    #         # Tokenize the word and count # of subwords the word is broken into
    #         tokenized_word = self.tokenizer.tokenize(word)
    #         token = self.tokenizer.convert_tokens_to_ids(tokenized_word)

    #         # Add the tokenized word to the final tokenized word list
    #         tokenized_sentence.extend(tokenized_word)
    #         tokens.extend(token)

    #   #Inserting [CLS] token at start
    #   tokenized_sentence.insert(0, "[CLS]")
    # #   tokens.insert(0, self.tokenizer.convert_tokens_to_ids("[CLS]"))

    #   #Inserting [SEP] token at the end
    #   tokenized_sentence.append("[SEP]")
    #   tokens.append(self.tokenizer.convert_tokens_to_ids("[SEP]"))

    #   #Adding padding to dataset
    #   if(self.MAX_LEN - len(tokenized_sentence) > 0):
    #         for i in range(0, self.MAX_LEN - len(tokenized_sentence)):
    #               tokenized_sentence.append("[PAD]")
    #               tokens.append(self.tokenizer.convert_tokens_to_ids("[PAD]"))
    #   else: #Truncation
    #         tokenized_sentence = tokenized_sentence[:self.MAX_LEN]
    #         tokens = tokens[:self.MAX_LEN]

    #   #Creating the attention mask to ignore padding
    #   attention_mask = []
    #   for item in tokenized_sentence:
    #         if(item != "[PAD]"):
    #               attention_mask.append(1)
    #         else:
    #               attention_mask.append(0)

    #   item = {}
    #   item["input_ids"] = tokens
    #   item["attention_mask"] = attention_mask

    #   prediction = self.trainer.predict([item])
    #   prediction = np.argmax(prediction.predictions, axis=2)

    #   tokenized_sentence_clean = []
    #   prediction_clean = []

    #   for i in range(0, len(tokenized_sentence)):
    #         if tokenized_sentence[i] != "[CLS]" and tokenized_sentence[i] != "[SEP]" and tokenized_sentence[i] != "[PAD]":

    #               tokenized_sentence_clean.append(tokenized_sentence[i])
    #               prediction_clean.append(prediction[0][i])
      
    #   prediction_output = []
    
    #   print(tokenized_sentence_clean)

    # #   print(prediction_clean)

    #   for i in range(0, len(tokenized_sentence_clean)):

    #         if ("##" in tokenized_sentence_clean[i]):
    #               prediction_output[len(prediction_output) - 1] = [prediction_output[len(prediction_output) - 1][0], (prediction_output[len(prediction_output) - 1][1] + tokenized_sentence_clean[i]).replace("##", "")]
    #         elif("." in tokenized_sentence_clean[i]):
    #               prediction_output[len(prediction_output) - 1] = [prediction_output[len(prediction_output) - 1][0], prediction_output[len(prediction_output) - 1][1] + tokenized_sentence_clean[i]]
    #         elif("." in tokenized_sentence_clean[i - 1]):
    #               prediction_output[len(prediction_output) - 1] = [prediction_output[len(prediction_output) - 1][0], prediction_output[len(prediction_output) - 1][1] + tokenized_sentence_clean[i]]
    #         else:
    #             #   print(i, prediction_clean[i])
    #               prediction_output.append([self.ids_to_labels[prediction_clean[i]], tokenized_sentence_clean[i]])
    #             #   print(prediction_output)
    #   return prediction_output

    ''' 
        Function that goes through the model prediction and seperates into commands with indexes
        Stores command into two arrays of keys (ie TEMPERATURE, ETIME) or values (ie 5, 10, GIWAXS, graphite)
        Instead of returning value, it sets the currentCommands class variable to the command list
    '''
    def processOutput(self, modelOutput):

        #Commands holds the list of commands
        commands = []
        #Each command is an array of two arrays, one for keys and one for values
        currentCommandKeys = []
        currentCommandValues = []

        for word in modelOutput:
            if(word[0] != "O"):

                initialCharacteristic = word[0][0]
                mainCharacteristic = word[0][word[0].index("-") + 1:]

                value = word[1]

                #If the prediction is a start key word with "B-", then we begin a new command to append to the list of commands
                if initialCharacteristic == "B":
                    index = 0

                    if(len(commands) == 0): #If this is the first command being registered
                        currentCommandKeys = [mainCharacteristic]
                        currentCommandValues = [value]

                    else: #If this isnt the first command being registered
                        
                        commands.append([currentCommandKeys, currentCommandValues])
                        currentCommandKeys = [mainCharacteristic]
                        currentCommandValues = [value]

                #If the prediction is an intermediate key word with "I-", then we add along to the command
                elif initialCharacteristic == "I":
                    
                    currentCommandKeys.append(mainCharacteristic)
                    currentCommandValues.append(value)

        if(len(currentCommandKeys) != 0):
            commands.append([currentCommandKeys, currentCommandValues])
        
        self.currentCommands = commands
        print("\nCurrent Commands:\n" + str(self.currentCommands))

    '''
        Actual function that interfaces with the UI, 
        the self.command carries a False to only to output to the log rather than 
        actual create changes to the beamline
    '''
    def processInput(self, inputText):
        print("\n\nInputting Command\n\n")
        if(len(inputText) == 0):
            self.ui.logOutput("Please Input a Command")
        elif(inputText[0] == "$"):
            self.ui.logOutput("Executing Command in Console")
            self.interface.executeCommand(inputText)
        else:
            modelOutput = self.modelPredict(inputText)

            print("\n\Processing Command\n\n")
            self.processOutput(modelOutput)

            print("\nOutputting Commands\n")
            self.command(False)

    '''
        searches the keys in a command to find specificed values, 
        returns a singlular string/float if value occurs once
            otherwise, returns list. Is numeric when possible
    '''
    def searchValue(self, command, key):

        returnValues = []

        for i in range(len(command[0])):
            if(command[0][i] == key):
                element = command[1][i]

                #If the element is outright numeric, then we append it as float
                if(element.isnumeric()):
                    returnValues.append(float(element))

                #If there is a decimal in the element, we check both sides of the decimal point to be
                elif("." in element):

                    decimal = element.index(".")
                    if(element[:decimal].isnumeric() and (element[decimal + 1:] == "" or element[decimal + 1:].isnumeric())):
                        returnValues.append(float(element))
                    else:
                        returnValues.append(element)
                else:
                    returnValues.append(element)

        if(len(returnValues) == 0):
            return "Current"
        elif(len(returnValues) == 1):
            return returnValues[0]
        else:
            return returnValues

    '''
        Processes commands and puts output in Log. If isConfirmed is true, it will carry out the processes
    '''
    def command(self, isConfirmed):

        for command in self.currentCommands:

            keys = command[0]
            values = command[1]

            #Printing out the keys and respective values
            print("Keys: " + str(keys) + "\nValues: " + str(values))
                
            # If there is only a temperature key then we work on that 
            if("TEMPERATURE" in keys):

                temp = self.searchValue(command, "TEMPERATURE")
                
                if("NRAMP_MIN" in keys):
                    nramp = self.searchValue(command, "NRAMP_MIN")

                    if(isConfirmed):
                        self.ui.logOutput("\nTemperature already set to: {} at a rate of: {} degrees per minute".format(temp, nramp))
                        self.interface.changeTemperature(temp, ["min", nramp])
                    else:
                        self.ui.logOutput("\n## Temperature will be set to: {} at a rate of: {} degrees per minute".format(temp, nramp))
                elif("NRAMP_SEC" in keys):
                    nramp = self.searchValue(command, "NRAMP_SEC")

                    if(isConfirmed):
                        self.ui.logOutput("\nTemperature already set to: {} at a rate of: {} degrees per second".format(temp, nramp))
                        self.ui.codeLog("\nsam.setTemperature({})\n".format(temp))
                        
                        self.interface.changeTemperature(temp, ["sec", nramp])
                    else:
                        self.ui.logOutput("\n## Temperature will be set to: {} at a rate of: {} degrees per second".format(temp, nramp))
                else:
                    if(isConfirmed):
                        self.ui.logOutput("\nTemperature already set to: {} quickly".format(temp))
                        self.ui.codeLog("\nsam.setTemperature({})\n".format(temp))
                        self.interface.changeTemperature(temp)
                    else:
                        self.ui.logOutput("\n## Temperature will be set to: {} quickly".format(temp))

            if("TEMPERATURE_CONDITIONAL" in keys) and ("SCAN" in keys):
                
                temp = self.searchValue(command, "TEMPERATURE")
                if ("ETIME" in keys):
                    exposure_time = self.searchValue(command, "ETIME")
                else:
                    exposure_time = 1
                if ("TRATE-SEC" in keys):
                    period_sec = self.searchValue(command, "TRATE-SEC")
                else:
                    period_sec = 60

                if(isConfirmed):
                    self.ui.logOutput("\nTemperature already set to: {} quickly".format(temp) + "\n")
                    self.ui.codeLog("\nsam.setTemperature({})\n".format(temp))
                    self.ui.codeLog("\nwhile self.temperature() < {}-1\n".format(temp))
                    self.ui.codeLog("\n    sam.measure({})\n".format(exposure_time))
                    self.ui.codeLog("\n    sam.sleep({:0.f})\n".format(period_sec - exposure_time))
                    self.interface.doTemperatureScan(temp, exposure_time, period_sec)
                else:
                    self.ui.logOutput("\n## Temperature will be set to {} quickly. Until reach set temperature, sample will be measured every {} s with exposure time {} s".format(temp, period_sec, exposure_time) + "\n")


            if("SAMPLE" in keys):
                sample = self.searchValue(command, "SAMPLE")
                if(isConfirmed):
                    self.ui.logOutput("Sample name already set to: {}".format(sample))
                    self.ui.codeLog("\nsam.name = \'{}\'\n".format(sample))
                    self.interface.setSampleName(sample)
                else:
                    self.ui.logOutput("## Sample name will be set to: {}".format(sample))

            if("ANGLE" in keys):
                #currentHolder = self.ui.getCurrentHolder()
                #currentSample = self.ui.getCurrentSample()
                #angles = self.searchValue(command, "ANGLE")

                #print(currentHolder, currentSample, angles)

                #if(currentHolder == -1):
                #    self.ui.logOutput("\nNo Holder Selected. Can not change angles.")
                #elif(currentSample == -1):
                #    self.ui.logOutput("\nNo Sample Selected. Can not change angles.")
                # else:
                #    for holder in self.holders:
                #     if currentHolder == holder.name:
                #         for sample in holder.getSampleList():
                #             if currentSample == sample.name:
                #                 sample.changeAngles(angles)
                #                 self.ui.getCurrentSampleData() #Updates the data on the GUI
                angle = self.searchValue(command, "ANGLE")

                if(isConfirmed):
                    self.ui.logOutput("Incident angle set to: {} deg.".format(angle))
                    self.ui.codeLog("\nsam.thabs({})\n".format(angle))
                    self.interface.setIncidentAngle(angle)
                else:
                    self.ui.logOutput("## Incident angle will be set to: {} deg.".format(angle))
        
            # For humidity
            if("HUMIDITY" in keys):

                humidity = self.searchValue(command, "HUMIDITY")
                if(isConfirmed):
                    self.ui.logOutput("Humidity set to: {} percent".format(humidity))
                else:
                    self.ui.logOutput("Humidity will be set to: {} percent".format(humidity))

            #If there is an absolute x position to move the motor to, we move it to there
            if("XPOS_AB" in keys):

                xpos = self.searchValue(command, "XPOS_AB")

                if(isConfirmed):
                    self.ui.logOutput("X position of motor set to: {}".format(xpos))
                    self.ui.codeLog("\nsam.xabs({})\n".format(xpos))
                    self.interface.xabs(xpos)
                else:
                    self.ui.logOutput("## X position of motor will be set to: {}".format(xpos))

            #If there is a relative x position to move the motor by
            if("XPOS_REL" in keys):
                xpos = self.searchValue(command, "XPOS_REL")

                if(isConfirmed):
                    self.ui.logOutput("X position of motor moved by: {}".format(xpos))
                    self.ui.codeLog("\nsam.xr({})\n".format(xpos))
                    self.interface.xr(xpos)
                else:
                    self.ui.logOutput("## X position of motor will move by: {}".format(xpos))

            #If there is an absolute y position to move the motor to, we move it to there
            if("YPOS_AB" in keys):

                ypos = self.searchValue(command, "YPOS_AB")

                if(isConfirmed):
                    self.ui.logOutput("X position of motor set to: {}".format(ypos))
                    self.ui.codeLog("\nsam.yabs({})\n".format(ypos))
                    self.interface.yabs(ypos)
                else:
                    self.ui.logOutput("## X position of motor will be set to: {}".format(ypos))

            #If there is a relative y position to move the motor by
            if("YPOS_REL" in keys):
                ypos = self.searchValue(command, "YPOS_REL")

                if(isConfirmed):
                    self.ui.logOutput("Y position of motor moved by: {}".format(ypos))
                    self.ui.codeLog("\nsam.yr({})\n".format(ypos))
                    self.interface.yr(ypos)
                else:
                    self.ui.logOutput("## Y position of motor will move by: {}".format(ypos))

            #Meaure 
            if("SCAN" in keys):
                #This is where scan stuff goes, you can gather the data from the selected holder and selected sample
                if ("ETIME" in keys):
                    exposure_time = self.searchValue(command, "ETIME")
                else:
                    exposure_time = 1

                if(isConfirmed):
                    self.ui.logOutput("Measuring sample with exposure time {} s".format(exposure_time))
                    self.ui.codeLog("\nsam.measure({})\n".format(exposure_time))
                    self.interface.measure(exposure_time)
                else:
                    self.ui.logOutput("## Will measure sample with exposure time {} s".format(exposure_time))
                #break

            #Analysis 
            if("PROTOCOL" in keys): 
                protocol = self.searchValue(command, "PROTOCOL")
                protocol_list = ['qr_image', 'circular_average']
                if protocol not in protocol_list:
                    protocol = "thumbnail"

                self.ui.logOutput("Show data with protocol '{}'".format(protocol))
                if(isConfirmed):
                    #self.ui.logOutput("Show data with protocol '{}'".format(protocol))
                    self.ui.codeLog("\nSee '{}' in runXS_test.py\n".format(protocol))
                    self.interface.analyze(protocol)
                #else:
                    #self.ui.logOutput("Show data with protocol '{}'".format(protocol))
                #break






