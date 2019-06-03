#!/usr/bin/env python

'''A class structure for solving ODE's

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
    pass