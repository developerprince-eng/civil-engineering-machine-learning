from __future__ import print_function

import pandas as pd 
import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import os 
os.getcwd() 
os.listdir(os.getcwd())

training_set_size_portion = .8
do_shuffle = True
accuracy_score = 0
hidden_units_spec = [10,20,10]
n_classes_spec = 2
tmp_dir_spec = "tmp/model"
steps_spec = 2000
epochs_spec = 15
features = ['Clump_Thickness','Uniformity_of_Cell_Size','Uniformity_of_Cell_Shape','Marginal_Adhesion','Single_Epithelial_Cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses']
labels = ['Class']

#Print File Paths
#print (os.listdir(os.getcwd()))
data = pd.read_csv('Breast Cancer/data/data.csv', low_memory=False)

if do_shuffle:
    randomized_data =  data.reindex(np.random.permutation(data.index))
else:
    randomized_data = data

#data_df = pd.DataFrame(data)
#data_df = data_df.drop(data_df.index[0])

#print Data Frame of Breast Cancer Data
#print(data_df)
total_records = len(randomized_data)
#data_len = len(data_df.index)
#train_data_len = int(total_records * 0.8)
#test_data_len = int(total_records * 0.2)

training_set_size = int(total_records * training_set_size_portion)
test_set_size = total_records = training_set_size

#train_data = data_df.iloc[0:train_data_len, 1:10]
#test_data = data_df.iloc[0:test_data_len, 1:10 ]
#train_labels = data_df.iloc[0:train_data_len, 10:11]
#test_labels =  data_df.iloc[0:test_data_len, 10:11]

# Build the training features and labels
training_features = randomized_data.head(training_set_size)[features].copy()
training_labels = randomized_data.head(training_set_size)[labels].copy()
training_features = training_features[features].astype('int64')

training_features = training_features.as_matrix()
training_labels = training_labels.as_matrix()
training_labels = training_labels.astype('int64')

print(training_features)
print(training_labels)

# Build the testing features and labels
testing_features = randomized_data.tail(test_set_size)[features].copy()
testing_labels = randomized_data.tail(test_set_size)[labels].copy()

feature_columns = [tf.feature_column.numeric_column(key) for key in features]
print(feature_columns)

#Kera Model

"""
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
"""

#DNN Classifier
classifier = tf.estimator.DNNClassifier(
                feature_columns=feature_columns, 
                hidden_units=hidden_units_spec, 
                n_classes=n_classes_spec, 
                model_dir=tmp_dir_spec)

# Define the training input function using Pandas DataFrame
"""
train_input_fn = tf.estimator.inputs.pandas_input_fn(
                    x=training_features, 
                    y=training_labels, 
                    num_epochs=epochs_spec, 
                    shuffle=True)
"""

# Define the training input function using Numpy Arrays
train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x=training_features, 
                    y=training_labels, 
                    num_epochs=epochs_spec, 
                    shuffle=True)


# Train the model using the classifer.
classifier.train(input_fn=train_input_fn, steps=steps_spec)

# Define the test input function
test_input_fn = tf.estimator.inputs.pandas_input_fn(
                x=testing_features, 
                y=testing_labels, 
                num_epochs=epochs_spec, 
                shuffle=False)
# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("Accuracy = {}".format(accuracy_score))