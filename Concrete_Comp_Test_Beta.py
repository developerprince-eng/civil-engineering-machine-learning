from __future__ import print_function

import os
import tensorflow as tf
import pandas as pd 
import numpy as np
os.getcwd()
os.listdir(os.getcwd())

class Data:
    def __init__(self, path):
        self.path = path


    def __input_fn__ (self, label_key, num_epochs, shuffle, batch_size):
        filename = self.path
        data_set = pd.read_csv(filename, low_memory=False)
        df = pd.DataFrame(data_set)
        label = df[label_key]
        ds = tf.data.Dataset.from_tensor_slices((dict(df),label))

        if shuffle:
            ds = ds.shuffle(200)

        ds = ds.batch(batch_size).repeat(num_epochs)

        return ds
    def train_data(self):
        

tf.enable_eager_execution()

