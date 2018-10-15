# Developer Prince - Concrete Compressive Strength Regression Machine Learning Demo
""" This is Regression problem which make use of Data set with 8 Features inorder to Determine Concrete Compressve Strength(MPa, Mega Pascals) """

from __future__ import print_function, absolute_import, division

import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow import keras
import sys
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)
def main():
    if len(sys.argv) == 10:

        if sys.argv[1] == 'compile':
            test_data = np.array([[int(sys.argv[2]), int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]),int(sys.argv[8]),int(sys.argv[9])]]) 

            CCST_model = tf.keras.models.load_model('CCST_predictor.model')
            
            predictions = CCST_model.predict(x=test_data)
            results = predictions.tolist()
            json_obj = json.dumps({'ccst':results[0][0]})
            print(json_obj)
        elif sys.argv[1] != 'compile':
            json_obj = json.dumps({'error':5})
            print(json_obj)
    elif len(sys.argv) > 10:
        json_obj = json.dumps({'error':1})
        print(json_obj)
    elif len(sys.argv) < 10:
        json_obj = json.dumps({'error':0})
        print(json_obj)
if __name__ == '__main__':
	main()