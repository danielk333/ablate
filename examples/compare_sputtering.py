'''
Compute sputtering and thermal ablation
========================================

Docstring for this example
'''

import numpy as np
import matplotlib.pyplot as plt
import xarray

import ablate
import ablate.functions as func
import ablate.atmosphere as atm

#data for 2 different meteoroids
model = atm.NRLMSISE00()
data = model.density(
    time = np.datetime64('2018-07-28'),
    lat = np.array([69.0, 69.0]),
    lon = np.array([12.0, 12.0]),
    alt = np.array([90e3, 120e3]),
)
print(data)

material_data = func.material.material_parameters('ast')

dmdt = func.sputtering.sputtering(
    mass = np.array([0.5, 0.2]),
    velocity = np.array([40e3, 40e3]),
    material_data = material_data,
    density = data,
)

dmdt_th = func.ablation.thermal_ablation(
    mass = np.array([0.5, 0.2]),
    temperature = 3700,
    material_data = material_data,
    shape_factor = 1.21,
)


print('thermal dm/dt:')
print(dmdt_th)

print('sputtering dm/dt:')
print(dmdt)