# Developer Prince - Concrete Compressive Strength Regression Machine Learning Demo
""" This is Regression problem which make use of Data set with 8 Features inorder to Determine Concrete Compressve Strength(MPa, Mega Pascals) """

from __future__ import print_function, absolute_import, division

import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import os
os.getcwd()
os.listdir(os.getcwd())

data = pd.read_csv('sample-data/ccs_data.csv', low_memory=False)
data_label = pd.read_csv('sample-data/ccs_labels.csv', low_memory=False)
FEATURES = ['Cement', 'Blast Furnace Slag','Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate','Age']

labels = pd.DataFrame(data_label)
data_df = pd.DataFrame(data)


data_len = len(data_df.index)
train_labels_len = int(data_len * 0.8)
test_labels_len = int(data_len * 0.2)
train_data_len = int(data_len * 0.8)
test_data_len = int(data_len * 0.2)


train_data =  data_df.iloc[0:train_data_len,:]
test_data = data_df.iloc[0:test_data_len,:]
train_labels  = labels.iloc[0:train_labels_len,:]
test_labels = labels.iloc[0:test_labels_len,:]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data)

print(train_labels)

print(test_data)

print(test_labels)

model = keras.Sequential([
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

model.compile(loss='mean_squared_logarithmic_error',
                optimizer='adam',
                metrics=['mae'])
model.fit(x=train_data.values, y=train_labels.values, epochs=500)

val_loss, val_acc = model.evaluate(x=test_data.values, y=test_labels.values)

print(val_loss, val_acc)

test_predictions = model.predict(x=test_data.values).flatten()

print(test_predictions)

print(test_labels.values)
""" Save Model """
model.save('CCST_predictor.model')

CCST_model = tf.keras.models.load_model('CCST_predictor.model')

predictions = CCST_model.predict(x=test_data.values)

print(predictions)