#!/usr/bin/env python

'''
Meteoroid material physics
===========================
'''

# Basic Python
import copy
import logging

logger = logging.getLogger(__name__)


# External packages
import numpy as np
from scipy import constants


material_data = {
    'iron': {
        'rho_m': 7800.0,
        'mu': 56.0*constants.u,
        'm2': 56.0*constants.u,
        'CA': 10.6,
        'CB': 16120.,
        'u0': 4.1,
        'k': 0.35,
        'z2': 26,
        'c': 1200,
        'L': 6.0E6,
    },
    'asteroroidal': {
        'rho_m': 3300.0,
        'mu': 50.0*constants.u,
        'm2': 20.0*constants.u,
        'CA': 10.6,
        'CB': 13500.,
        'u0': 6.4,
        'k': 0.1,
        'z2': 10,
        'c': 1200,
        'L': 6.0E6,
    },
    'cometary': {
        'rho_m': 1000.0,
        'mu': 20.0*constants.u,
        'm2': 12.0*constants.u,
        'CA': 10.6,
        'CB': 13500.,
        'u0': 4.0,
        'k': 0.65,
        'z2': 6,
        'c': 1200,
        'L': 6.0E6,
    },
    'porous': {
        'rho_m': 300.0,
        'mu': 20.0*constants.u,
        'm2': 12.0*constants.u,
        'CA': 10.6,
        'CB': 13500.,
        'u0': 4.0,
        'k': 0.65,
        'z2': 6,
        'c': 1200,
        'L': 6.0E6,
    },
}
'''dict: Meteoroid material data.

# TODO: List origin for all data.
'''

def list_materials():
    '''List currently available meteoroid materials.

    :return: All available materials
    :rtype: list of strings
    '''
    return material_data.keys()


def material_parameters(material):
    '''Returns the physical parameters of the meteoroid based on its material.
    
    :param str material: Meteoroid material, see :func:`~functions.material.material_parameters`.

    :rtype: dict
    :return: Dictionary of all available material parameters (floats)

    **List of properties:**
        * m2: mean atomic mass [1]
        * u0: surface binding energy [eV]
        * k: average atomic number [1]
        * CA, CB: Clausius-Clapeyron coeff [K]
        * mu: mean molecular mass of ablated vapour [kg]
        * k: Botzmann constant [J/K]
        * rho_m: meteoroid density [kg/m3]
        * c: specific heat of meteoroid [J/(K*kg)]
        * L: latent heat of fusion + vapourization [J/kg]


    [Fe SiO2 C C] mean molecular mass per target atoms corresponding to the different meteoroid densities, see Rogers p. 1344 values are from Tielens et al.
    
    Mean molecular mass of ablated vapor assuming that all products liberated from the meteoroid by thermal ablation were released in complete molecular form.

    '''

    _material = material.lower().strip()

    for key in material_data.keys():
        if key[:len(_material)] == _material:
            return material_data[key]
    
    raise ValueError(f'No data exists for material "{_material}"')
