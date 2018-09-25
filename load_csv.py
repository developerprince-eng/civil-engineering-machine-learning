##The following makes use of pandas to read csv files and display the values on console 

## from __future__import print_function 
##Uncomment the line above if you running your code in the early versions of python like python 2.7 

import tensorflow as tf
import os
os.getcwd()
os.listdir(os.getcwd())

filenames = ["sample-data/recs2009_public.csv"]
record_defaults = [tf.float32] * 940  
dataset = tf.Contrib.data.CsvDataset(filenames, record_defaults)

print(dataset)

