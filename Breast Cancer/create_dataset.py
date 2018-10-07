""" This class if for generating a Dataset from the given path """
from __future__ import print_function, absolute_import, division

import tensorflow as tf
import os
import pandas as pd 
import numpy as py
os.getcwd()
os.listdir(os.getcwd())

class RESC_DATA:
    def __init__(self, path): 
        self.path = path

        
    
    def __read_csv__(self):
        data_set = pd.read_csv(self.path, low_memory=False)
        return (data_set)

   
    def __obtain_training_data__(self):
        data_set = pd.read_csv(self.path, low_memory=False)
        df = pd.DataFrame(data_set)

        data_length = len(df.index)
        train_data_len = (data_length * 0.8)

        train_data = df.iloc[0:train_data_len,:]
        
        return(train_data)
    
    
    def __obtain_testing_data__(self):
        data_set = pd.read_csv(self.path, low_memory=False)
        df = pd.DataFrame(data_set)

        data_length = len(df.index)
        test_data_len = (data_length * 0.2)

        test_data = df.iloc[0:test_data_len,:]

        return(test_data)
    
    
    def __obtain_training_label(self):
        data_set = pd.read_csv(self.path, low_memory=False)
        df = pd.DataFrame(data_set)

        data_length = len(df.index)
        train_data_len = (data_length * 0.8)

        train_label = df.iloc[0:train_data_len,:]
        
        return(train_label)
    
    
    def __obtain_testing_label(self):
        data_set = pd.read_csv(self.path, low_memory=False)
        df = pd.DataFrame(data_set)

        data_length = len(df.index)
        test_data_len = (data_length * 0.2)

        test_label = df.iloc[0:test_data_len,:]

        return(test_label)

    