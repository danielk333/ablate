'''
Estimate drag coefficients and heat transfer
=============================================

Docstring for this example
'''

import numpy as np
import matplotlib.pyplot as plt
import xarray
import scipy.constants as constants

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

dmdt_th = func.ablation.thermal_ablation(
    mass = np.array([0.5, 0.2]),
    temperature = 3700,
    material_data = material_data,
    shape_factor = 1.21,
)

atm_mean_mass = np.array([x['A'] for _,x in model.species.items()]).mean() * constants.u #[kg]

Gamma = func.dynamics.drag_coefficient(
    mass = 0.5, 
    velocity = 32e3, 
    temperature = 3700, 
    material_data = material_data, 
    atm_total_density = data['Total'].values.squeeze(),
    atm_mean_mass = atm_mean_mass, 
    res = 100,
)

Lambda = func.dynamics.heat_transfer(
    mass = 0.5, 
    velocity = 32e3, 
    temperature = 3700, 
    material_data = material_data, 
    atm_total_density = data['Total'].values.squeeze(), 
    thermal_ablation = dmdt_th[0], 
    atm_mean_mass = atm_mean_mass, 
    res = 100,
)

print('Alt:')
print([90e3, 120e3])

print('Gamma:')
print(Gamma)

print('Lambda:')
print(Lambda)

