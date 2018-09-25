import tensorflow as tf
import os
import pandas as pd 
os.getcwd() 

data_set = pd.read_csv('C:/Users/TechVillage Laptop01/Documents/WorkItems/ml-tut/sample-data/recs2009_public.csv', low_memory=False)
pd.set_option('max_info_rows', 11)


dataframe = pd.DataFrame(data_set)
all_data = data_set.iloc[:,:].values
index_data =  dataframe.info(verbose=None, buf=None, max_cols=None, memory_usage=None)

print(index_data)


