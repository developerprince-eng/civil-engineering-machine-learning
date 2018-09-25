""" This Class is used for Training our model based on training Data from the Dataset provide """
import tensorflow as tf 
import os
import numpy as np 
os.getcwd()

class TRAIN_MODEL(object):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name
    @name.setter
    def object(self, name):
        self._name = name 

    @classmethod
    def train(self):
        return 0