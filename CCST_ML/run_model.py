""" This class is for Running the model Generate for RESC """
import tensorflow as tf
import os
import pandas as pd 
import numpy as np 
os.getcwd() 

class RUN_MODEL(object):
    def __init__(self, obj):
        self._obj = obj

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, obj):
        self._obj = obj

    @classmethod
    def __run_model__(self):
        return 0;