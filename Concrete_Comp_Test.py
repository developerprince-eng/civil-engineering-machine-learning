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

data = pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/ccs_data.csv', low_memory=False)
data_label = pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/ccs_labels.csv', low_memory=False)
column_names = ['Cement (component 1)(kg in a m^3 mixture)', 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)','Fly Ash (component 3)(kg in a m^3 mixture)','Water  (component 4)(kg in a m^3 mixture)','Superplasticizer (component 5)(kg in a m^3 mixture)','Coarse Aggregate  (component 6)(kg in a m^3 mixture)','Fine Aggregate (component 7)(kg in a m^3 mixture)','Age (day)']

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


#Normalize

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

#Convert Data to Lists 

train_data = train_data.iloc[:,0].tolist()
test_data = test_data.iloc[:,0].tolist()

#Convert Lists to DataSets 

train_data = tf.data.Dataset.from_tensor_slices((dict(train_data), train_labels))
test_data = tf.data.Dataset.from_tensor_slices((dict(test_data), test_labels))

print(train_data)

def train_input_fn():
     return (train_data.shuffle(1000).batch(128).repeat().make_one_shot_iterator().get_next())

    
def test_input_fn():
    return (test_data.shuffle(1000).batch(128).make_one_shot_iterator().get_next())

""" Build A Model """

"""
def build_model():
    model = keras.Sequential([
              keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
              keras.layers.Dense(64, activation=tf.nn.relu),
              keras.layers.Dense(1)
            ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae','accuracy'])
    return model

model = build_model()
model.summary()
"""
""" ********************************************************************************** """
feature_columns = ['Cement Kg','Blast Furnace Slag Kg','Fly Ash Kg','Water Kg','Superplasticizer Kg','Coarse Aggregate Kg','Fine Aggregate kg','Age day']

""" Using Estimators """
model = tf.estimator.LinearRegressor(feature_columns=feature_columns)

#Train Model
model.train(input_fn=train_input_fn, steps=1000)

#Test Model
eval_result = model.evaluate(input_fn=test_input_fn)


# The evaluation returns a Python dictionary. The "average_loss" key holds the
# Mean Squared Error (MSE).
average_loss = eval_result["average_loss"]

#Predict 
pred_result = model.predict(input_fn=test_input_fn)
""" ********************************************************************************** """

# Display training progress by printing a single dot for each completed epoch
""" ********************************************************************************** """

"""
class PrintStar(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('*', end='')

EPOCHS = 500
"""

""" ********************************************************************************** """
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
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

""" This section makes use of the manually written model using layers """
# Store training stats
#history = model.fit(train_data, train_labels, epochs=EPOCHS,validation_split=0.2, verbose=0,callbacks=[PrintStar()])


#plot_history(history)

#[loss, mae, accuracy] = model.evaluate(test_data, test_labels, verbose=0)



#test_predictions = model.predict(test_data).flatten()

#print(" \r\n Testing set Mean Abs Error: {:7.2f}".format(mae * 1000000))

#print("\r\n Accuracy is :{:7.2f} ".format(accuracy))
""" ********************************************************************************** """

""" This section makes use of the Estimators """

print("\r\n Evaluation: ")
print("\r\n",eval_result)

print("\r\n Prediction: ")
print("\r\n",pred_result)

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