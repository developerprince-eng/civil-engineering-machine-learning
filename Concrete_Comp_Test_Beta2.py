# Developer Prince - Concrete Compressive Strength Regression Machine Learning Demo
""" This is Regression problem which make use of Data set with 8 Features inorder to Determine Concrete Compressve Strength(MPa, Mega Pascals) """

from __future__ import print_function, absolute_import, division

import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
import os
os.getcwd()
os.listdir(os.getcwd())

data = pd.read_csv('sample-data/ccs_data.csv', low_memory=False)
data_df = pd.DataFrame(data)

data_len = len(data_df.index)


train_data_len = int(data_len * 0.8)
test_data_len = int(data_len * 0.2)

train_data =  data_df.iloc[0:train_data_len,:]
test_data = data_df.iloc[0:test_data_len,:]


mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

test_data = (test_data - mean) / std


CCST_model = tf.keras.models.load_model('CCST_predictor.model')

predictions = CCST_model.predict(x=test_data.values)

print(predictions)