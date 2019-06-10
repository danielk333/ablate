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
    '''Atmospheric model base class. 

    Forces implementation of the method :code:`density` that takes:

    :param numpy.datetime64 npt: Time to evaluate model at.
    :param str/list species: String representation(s) of the atmospheric constituent(s) to get data for
    
    and should return a single numerical value or a numpy array. Also forces implementation of :code:`species` property that returns a dictionary of all implemented species.
    '''

    @abstractmethod
    @property
    def species(self):
        pass


    def __init__(self, **kwargs):
        pass


    @abstractmethod
    def density(self, npt, species):
        pass


    def temperature(self, npt, species):
        raise NotImplementedError()
