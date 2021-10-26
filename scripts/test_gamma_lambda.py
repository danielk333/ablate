import numpy as np
import matplotlib.pyplot as plt
import xarray
import scipy.constants as constants

import ablate
import ablate.functions as func
import ablate.atmosphere as atm

#data for 2 different meteoroids
model = atm.MSISE00()
data = model.density(
    npt = np.datetime64('2018-07-28'),
    lat = np.array([69.0, 69.1]),
    lon = 12.0,
    alt = 120e3,
)



dmdt_th = func.ablation.thermal_ablation(
    mass = np.array([0.5, 0.2]),
    temperature = 3700,
    material = 'ast',
    A = 1.21,
)

atm_mean_mass = np.array([x['A'] for _,x in model.species.items()]).mean() * constants.u #[kg]

Gamma = func.dynamics.drag_coefficient(
    mass = 0.5, 
    velocity = 32e3, 
    temperature = 3700, 
    material = 'ast', 
    total_density = data['Total'].values.squeeze()[0], 
    atm_mean_mass = atm_mean_mass, 
    res = 100,
)

Lambda = func.dynamics.heat_transfer(
    mass = 0.5, 
    velocity = 32e3, 
    temperature = 3700, 
    material = 'ast', 
    total_density = data['Total'].values.squeeze()[0], 
    thermal_ablation = dmdt_th[0], 
    atm_mean_mass = atm_mean_mass, 
    res = 100,
)

print('Gamma, Lambda:')
print(Gamma, Lambda)

