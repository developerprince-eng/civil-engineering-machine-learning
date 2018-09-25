""" This Class is used for Testing Data """
import tensorflow as tf
import os
import pandas as pd
import numpy as np 
os.getcwd() 

class TEST_MODEL(object):
    def __init(self, obj):
        self._obj= obj
    @property
    def obj(self):
        return self._obj

    @obj.setter
    def object(self, obj):
        self._obj = obj

    @classmethod
    def __test__(self):
        return 0