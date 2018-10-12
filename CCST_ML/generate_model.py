"""This Class is responsible for generating a model for RESC ML """
from __future__ import print_function, absolute_import, division

import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd 
import numpy as np 
os.getcwd() 
os.listdir(os.getcwd())

class GENERATE_MODEL:
    def __init__(self):
        self.is_model = True
                
    def __generate__(self, train_data):
        model = keras.Sequential([
                 keras.layers.Dense(128, activation=tf.nn.relu),
                 keras.layers.Dense(128, activation=tf.nn.relu),
                 keras.layers.Dense(1)
                ])
        

        model.compile(loss='mean_squared_logarithmic_error',
                     optimizer='adam',
                     metrics=['mae'])
        return model

   
    def __normalize__(self, train_data, test_data):
        mean = train_data.mean(axis=0)
        std = train_data.std(axis=0)
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std
        data = [train_data, test_data]
        return data

