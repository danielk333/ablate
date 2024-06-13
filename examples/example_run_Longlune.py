'''
Example run of an ablation model
=================================

Docstring for this example
'''

import logging
import sys
import math

import numpy as np
import matplotlib.pyplot as plt

import ablate

from modeleGSI import example
print(example.test(1))


sys.path.append('/Users/jeannelonglune/Desktop/memoire/pyCabaret/src')
import modeleGSI_copy as mod
M_1 = 28.78
reff = 0.48 #HOW TO LINK IT TO link reff and the dx of the boundary layer.
alt = 61
Twall = 270
[heatflux, massblowrate , electron_number_density] = mod.modele_gsi(M_1, alt, reff, Twall)

heat = heatflux * (4 * math.pi *reff*reff) #[W] = [J/s]
#do with shape factor? 
C=710
rho = 2267.0
vol = 4/3 * math.pi * reff^3

dT_dt = heat / (C * rho * vol)
#C =710 specifif heat to get from carbon mat data base.  [J/(kg·K)]
#rho_m: 2267.0 density get from carbon mat data base [kg/m^3]
dm_dt = massblowrate

handler = logging.StreamHandler(sys.stdout)

for name in logging.root.manager.loggerDict:
    if name.startswith('ablate'):
        print(f'logger: {name}')


logger = logging.getLogger('ablate')
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

model = ablate.Longlune2024(
    atmosphere='msise00',
    options = dict(
        sputtering = True,
        minimum_mass = 1e-11,
        max_step_size = 5e-3,
    ),
)

material_data = ablate.functions.material.material_parameters('carbon')

result = model.run(
    velocity0 = 60*1e3, 
    mass0 = 1e-6, 
    altitude0 = 120e3,
    zenith_ang = 35.0,
    azimuth_ang = 0.0,
    material_data = material_data,
    time = np.datetime64('2018-06-28T12:45:33'),
    lat = 69.5866115,
    lon = 19.221555 ,
    alt = 100e3,
)

print(result)

fig = plt.figure(figsize=(15,15))
fig.suptitle('Meteoroid ablation simulation')

ax = fig.add_subplot(221)
ax.plot(result.t, np.log10(result.mass))
ax.set_ylabel('Mass [log$_{10}$(kg)]')
ax.set_xlabel('Time [s]')

ax = fig.add_subplot(222)
ax.plot(result.t, result.velocity*1e-3)
ax.set_ylabel('Velocity [km/s]')
ax.set_xlabel('Time [s]')

ax = fig.add_subplot(223)
ax.plot(result.t, result.position*1e-3)
ax.set_ylabel('Position on trajectory [km]')
ax.set_xlabel('Time [s]')

ax = fig.add_subplot(224)
ax.plot(result.t, result.temperature)
ax.set_ylabel('Meteoroid temperature [K]')
ax.set_xlabel('Time [s]')


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.plot(np.diff(result.mass.values)/np.diff(result.altitude.values), result.altitude.values[:-1]*1e-3)
ax.set_xlabel('Mass loss [kg/m]')
ax.set_ylabel('Altitude [km]')


plt.show()