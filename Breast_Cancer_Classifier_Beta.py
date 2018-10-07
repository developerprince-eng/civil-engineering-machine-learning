import tensorflow as tf
import pandas as pd
import numpy as np
import os 
os.getcwd() 
os.listdir(os.getcwd())

# The data needs to be split into a training set and a test set
# To use 80/20, set the training size to .8
training_set_size_portion = .8
# Set to True to shuffle the data before you split into training and # test sets
do_shuffle = True
# Keep track of the accuracy score
accuracy_score = 0
# The DNN has hidden units, set the spec for them here
hidden_units_spec = [10,20,10]
n_classes_spec = 2
# Define the temp directory for keeping the model and checkpoints
tmp_dir_spec = "tmp/model"
# The number of training steps
steps_spec = 2000
# The number of epochs
epochs_spec = 15
# File Name - be sure to change this if you upload something else
file_name = "data.csv"
# Here's a set of our features. If you look at the CSV, 
# you'll see these are the names of the columns. 
# In this case, we'll just use all of them:
features = ['radius','texture']
# Here's the label that we want to predict -- it's also a column in # the CSV
labels = ['diagnosis_numeric']
# Here's the name we'll give our data
data_name = 'wdbc.csv'
# Here's where we'll load the data from
data_path = 'http://www.laurencemoroney.com/wp-content/uploads/2018/02/wdbc.csv'

file_name = tf.keras.utils.get_file(data_name, data_path)
my_data = pd.read_csv(file_name, delimiter=',')

#The pandas DataFrame allows you to shuffle with the reindex method
# Docs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reindex.html#pandas.DataFrame.reindex
# If the doShuffle property is true, we will shuffle with this
# You really SHOULD shuffle to make sure that trends in data don't affect your learning
# but I make it optional here so you can choose
if do_shuffle:
    randomized_data =  my_data.reindex(np.random.permutation(my_data.index))
else:
    randomized_data = my_data

total_records = len(randomized_data)
training_set_size = int(total_records * training_set_size_portion)
test_set_size = total_records = training_set_size

# Build the training features and labels
training_features = randomized_data.head(training_set_size)[features].copy()
training_labels = randomized_data.head(training_set_size)[labels].copy()
print(training_features.head())
print(training_labels.head())

# Build the testing features and labels
testing_features = randomized_data.tail(test_set_size)[features].copy()
testing_labels = randomized_data.tail(test_set_size)[labels].copy()

feature_columns = [tf.feature_column.numeric_column(key) for key in features]

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns, 
    hidden_units=hidden_units_spec, 
    n_classes=n_classes_spec, 
    model_dir=tmp_dir_spec)


# Define the training input function
train_input_fn = tf.estimator.inputs.pandas_input_fn(
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

# Create a prediction set -- this is a list of input features that you want to classify
prediction_set = pd.DataFrame({'radius':[14, 13], 'texture':[25, 26]})

predict_input_fn = tf.estimator.inputs.pandas_input_fn(
                      x=prediction_set, 
                      num_epochs=1, 
                      shuffle=False)

# Get a list of the predictions
predictions = list(classifier.predict(input_fn=predict_input_fn))

predicted_classes = [p["classes"] for p in predictions] 
results = np.concatenate(predicted_classes) 
print(results)