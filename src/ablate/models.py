#!/usr/bin/env python

'''The basic class structures for implementing a ablation model.

'''

#
# Basic Python
#


#
# External packages
#
import numpy as np

#
# Internal packages
#
from .ode import OrdinaryDifferentialEquation

#
# Internal imports
#
from . import functions
from .ode import ScipyODESolve


__all__ = ['KeroSzasz2008']


class KeroSzasz2008(ScipyODESolve):
    pass