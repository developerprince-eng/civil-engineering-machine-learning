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

CE = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/CE.txt', sep=" ", header=None))
CP = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/CE.txt', sep=" ", header=None))
EPS1 = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/EPS1.txt', sep=" ", header=None))
FS1 = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/FS1.txt', sep=" ", header=None))
FS2 = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/FS2.txt', sep=" ", header=None))
PS1 = pd.Seies(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/PS1.txt', sep=" ", header=None))
PS2 = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/PS2.txt', sep=" ", header=None))
PS3 = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/PS3.txt', sep=" ", header=None))
PS4 = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/PS4.txt', sep=" ", header=None))
PS5 = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/PS5.txt', sep=" ", header=None))
PS6 = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/PS6.txt', sep=" ", header=None))
SE = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/SE.txt', sep=" ", header=None))
TS1 = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/TS1.txt', sep=" ", header=None))
TS2 = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/TS2.txt', sep=" ", header=None))
TS2 = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/TS2.txt', sep=" ", header=None))
TS3 = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/TS3.txt', sep=" ", header=None))
TS4 = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/TS4.txt', sep=" ", header=None))
VS1 = pd.Series(pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/condition monitoring for hydraulic dataset/VS1.txt', sep=" ", header=None))

labels = pd.DataFrame(VS1)
data_df = pd.DataFrame({'CE':CE, 'CP':CP, 'EPS1':EPS1, 'FS1':FS1, 'FS2':FS2, 'PS1':PS1, 'PS2':PS2, 'PS3':PS3, 'PS4': PS4, 'PS5':PS5, 'PS6':PS6, 'SE':SE, 'TS1':TS1, 'TS2':TS2, 'TS3':TS3, 'TS4':TS4, 'VS1':VS1})

print(data_df)
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


""" Build A Model """
def build_model():
    model = keras.Sequential([
              keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
              keras.layers.Dense(64, activation=tf.nn.relu),
              keras.layers.Dense(1)
            ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model

model = build_model()
model.summary()
""" ********************************************************************************** """

# Display training progress by printing a single dot for each completed epoch
class PrintStar(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('*', end='')

EPOCHS = 500

"""
def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
            label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
            label = 'Val loss')
    plt.legend()
    plt.ylim([0, 5])

"""
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,validation_split=0.2, verbose=0,callbacks=[PrintStar()])


#plot_history(history)

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:7.2f}".format(mae * 1000000))

test_predictions = model.predict(test_data).flatten()

"""
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1 000 000 MPa]')
plt.ylabel('Predictions [1 000 000 MPa]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [ 1000 000 MPa]")
_ = plt.ylabel("Count")
"""