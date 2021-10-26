'''
This is my example script
=========================

Docstring for this example
'''

import ablate.atmosphere as atm
import numpy as np
import matplotlib.pyplot as plt



model = atm.MSISE00()
species = ['O', 'N2', 'O2']
data = model.density(
    npt = np.datetime64('2018-07-28'),
    species = species,
    lat = np.array([69.0, 69.1]),
    lon = np.array([12.0, 12.0]),
    alt = 89e3,
)

print(data)

alt = np.linspace(77e3, 120e3, num=200)
data_2 = model.density(
    npt = np.datetime64('2018-07-28'),
    species = species,
    lat = np.array([69.0, 69.1]),
    lon = np.array([12.0, 12.0]),
    alt = alt,
)

print(data)