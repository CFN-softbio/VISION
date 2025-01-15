import os

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
import tensorflow as tf

from keras import Sequential
from keras.layers import Dense

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

class CCDataGrabber:

    def __init__(self, directory):

        self.directory = directory
        self.dataset = []

        self.getData()

    def getData(self, file):

        for root, dirs, files in os.walk(self.directory):
    
            if(len(files) > 0):
                for i in range(len(files)):
                    data = []
                    for line in open(root + files[i]):
                        if(not "#" in line):
                            rawData = line.split(" ")
                            data.append([float(rawData[0]), float(rawData[2])])
                    self.dataset.append(data)
            else:
                print("No files were found")

class curveClassification:

    def __init__(self, weightDirectory):

        self.weightDirectory = weightDirectory

        self.datalength = 200

        #Construct Model
        self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=self.datalength, activation='relu', input_shape=(self.datalength,)),
        tf.keras.layers.Dense(units=self.datalength, activation='relu'),
        tf.keras.layers.Dense(units=self.datalength / 2, activation='tanh'),
        tf.keras.layers.Dense(units=4, activation='tanh'),
        ])

        #Load trained weights (training occurs in training folder)
        self.model.load_weights(self.weightDirectory)

    def classifyData(self, data):

        originalX = data[0]
        originalY = data[1]

        newX = np.arange(originalX[0], originalX[len(originalX) - 1], (originalX[len(originalX) - 1] - originalX[0]) / 200.0, dtype=float)
        newY = np.interp(newX, originalX, originalY)

        predictions = self.model.predict(tf.stack(newY.reshape(1,200)))

        return predictions