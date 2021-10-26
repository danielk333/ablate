import numpy as np
import matplotlib.pyplot as plt
import xarray

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

_data = {}
for key in model.species.keys():
    _data[key] = (['met'], data[key].values.squeeze())

density = xarray.Dataset(
    _data,
   coords = {'met': np.arange(2)},
)

print(density)

dmdt = func.sputtering.sputtering(
    mass = np.array([0.5, 0.2]),
    velocity = np.full((2,), 40e3),
    material = 'ast',
    density = density,
)

dmdt_th = func.ablation.thermal_ablation(
    mass = np.array([0.5, 0.2]),
    temperature = 3700,
    material = 'ast',
    shape_factor = 1.21,
)


print('thermal dm/dt:')
print(dmdt_th)

print('sputtering dm/dt:')
print(dmdt)