
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import os
os.getcwd()
os.listdir(os.getcwd())
import sys
import CCST_ML.create_dataset as csdata
import CCST_ML.generate_model as cs

def main():

    DT = csdata.RESC_DATA()

    test_data = DT.__obtain_testing_data__('sample-data/ccs_data.csv')
    train_data = DT.__obtain_training_data__('sample-data/ccs_data.csv')
    MODEL = cs.GENERATE_MODEL()
    model = MODEL.__generate__(train_data)

    test_predictions = model.predict(x=test_data.values).flatten()

    print(test_predictions)

if __name__ == '__main__':
    main()