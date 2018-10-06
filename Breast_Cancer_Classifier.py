from __future__ import print_function

import pandas as pd 
import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import os 
os.getcwd() 
os.listdir(os.getcwd())

#Print File Paths
#print (os.listdir(os.getcwd()))
data = pd.read_csv('Breast Cancer/data/data.csv', low_memory=False)


data_df = pd.DataFrame(data)
#data_df = data_df.drop(data_df.index[0])

#print Data Frame of Breast Cancer Data
#print(data_df)
data_len = len(data_df.index)
train_data_len = int(data_len * 0.8)
test_data_len = int(data_len * 0.2)

train_data = data_df.iloc[0:train_data_len, 1:10]
test_data = data_df.iloc[0:test_data_len, 1:10 ]
train_labels = data_df.iloc[0:train_data_len, 10:11]
test_labels =  data_df.iloc[0:test_data_len, 10:11]

model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(16,)),
            keras.layers.Dense(32, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
model.fit(x=train_data.values, y=train_labels.values, epochs=500)

val_loss, val_acc = model.evaluate(x=test_data.values, y=test_labels.values)

print(val_loss, val_acc)

test_predictions = model.predict(x=test_data.values).flatten()

print(test_predictions)

print(test_labels)