from __future__ import print_function
import tensorflow as tf
import pandas as pd 
import os
os.getcwd()
os.listdir(os.getcwd())

dataset = pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/recs2009_public.csv', low_memory=False)
df = pd.DataFrame(dataset)
print(df)