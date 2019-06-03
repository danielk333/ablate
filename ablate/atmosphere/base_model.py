#!/usr/bin/env python

'''A basic class structure for implementing atmospheric models.

'''

#
# Basic Python
#
from abc import ABC
from abc import abstractmethod

#
# External packages
#
import numpy as np

class AtmosphereModel(ABC):
    '''Atmospheric model base class. Forces implementation of the method :code:`get` that takes:

    :param numpy.datetime64 npt: Time to evaluate model at.
    :param str variable: String representation of the variable to fetch from the atmospheric model.
    
    and should return a single numerical value or a numpy array.
    '''


    def __init__(self, **kwargs):
        pass


    @abstractmethod
    def get(self, npt, variable):
        pass