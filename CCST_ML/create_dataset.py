""" This class if for generating a Dataset from the given path """
from __future__ import print_function, absolute_import, division

import tensorflow as tf
import os
import pandas as pd 
import numpy as py
os.getcwd()
os.listdir(os.getcwd())

class RESC_DATA:
    def __init__(self):
        self.is_data = True
        
    def __read_csv__(self, path):
        data_set = pd.read_csv(path, low_memory=False)
        return (data_set)

   
    def __obtain_training_data__(self, path):
        data_set = pd.read_csv(path, low_memory=False)
        df = pd.DataFrame(data_set)

        data_length = len(df.index)
        train_data_len = int(data_length * 0.8)
        train_data = df.iloc[0:train_data_len,:]
        mean = train_data.mean(axis=0)
        std = train_data.std(axis=0)

        train_data = (train_data - mean) / std
        return(train_data)
    
    
    def __obtain_testing_data__(self, path):
        data_set = pd.read_csv(path, low_memory=False)
        df = pd.DataFrame(data_set)

        data_length = len(df.index)
        test_data_len = int(data_length * 0.2)
        test_data = df.iloc[0:test_data_len,:]
        mean = test_data.mean(axis=0)
        std = test_data.std(axis=0)

        test_data = (test_data - mean) / std
        return(test_data)
    
    
    def __obtain_training_label(self, path):
        data_set = pd.read_csv(path, low_memory=False)
        df = pd.DataFrame(data_set)

        data_length = len(df.index)
        train_data_len = int(data_length * 0.8)

        train_label = df.iloc[0:train_data_len,:]
        
        return(train_label)
    
    
    def __obtain_testing_label(self, path):
        data_set = pd.read_csv(path, low_memory=False)
        df = pd.DataFrame(data_set)

        data_length = len(df.index)
        test_data_len = int(data_length * 0.2)

        test_label = df.iloc[0:test_data_len,:]

        return(test_label)

    