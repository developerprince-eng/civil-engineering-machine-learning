"""This Class is responsible for generating a model for RESC ML """
from __future__ import print_function, absolute_import, division

import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd 
import numpy as np 
os.getcwd() 
os.listdir(os.getcwd())

class GENERATE_MODEL(object):
    def __init__(self, obj):
        self.obj = obj
    @property
    def obj(self):
        return self._obj
    
    @obj.setter
    def obj(self, obj):
        self._obj = obj

    @classmethod   
    def __generate__(self, train_data):
        model = keras.Sequential([
                 keras.layers.Dense(64, activation=tf.nn.relu, input_shap=(train_data[1],)),
                 keras.layers.Dense(64, activation=tf.nn.relu),
                 keras.layers.Dense(1)
                ])
        optimizer = tf.train.RMSPropOptimizer(0.001)

        model.compile(loss='mse',
                     optimizer=optimizer,
                     metrics=['mae'])
        return model

    @classmethod
    def __normalize__(self, train_data, test_data):
        mean = train_data.mean(axis=0)
        std = train_data.std(axis=0)
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std
        data = [train_data, test_data]
        return data

class PrintStar(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('*', end='')    

