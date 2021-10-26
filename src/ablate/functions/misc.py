#!/usr/bin/env python

'''
Miscellaneous functions
=========================


'''
#
# Basic Python
#
import copy

#
# External packages
#
import numpy as np
from scipy import constants



def curved_earth(s, rp, h_obs, zd, h_start):
    '''Calculates the error of the height? what
    

    
    distance :code:`s` along the trajectory at :code:`h=h_start` when :code:`s=0` at :code:`h=h_obs` given that the zentith distance is :code:`zd` at :code:`h=h_obs`.

    '''

    theta   = np.arctan2(s*np.sin(zd), (rp + h_obs + s*np.cos(zd)))
    h       = (rp + h_obs + s*np.cos(zd))/np.cos(theta) - rp

    err = h - h_start

    return err
