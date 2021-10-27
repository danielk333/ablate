import logging
import sys

import numpy as np
import matplotlib.pyplot as plt
import xarray
import scipy.constants as constants

import ablate
import ablate.functions as func

for name in logging.root.manager.loggerDict:
    if name.startswith('ablate'):
        print(f'logger: {name}')

logger = logging.getLogger('ablate.models.kero_szasz_2008')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


model = ablate.KeroSzasz2008(
    atmosphere='msise00',
    sputtering = True,
)

material_data = func.material.material_parameters('iron')

result = model.run(
    velocity0 = 60*1E3, 
    mass0 = 1E-6, 
    altitude0 = 500e3,
    zenith_ang = 35.0,
    azimuth_ang = 0.0,
    material_data = material_data,
    time = np.datetime64('2018-06-28T12:45:33'),
    lat = 69.5866115,
    lon = 19.221555 ,
    alt = 100e3,
)

fig = plt.figure(figsize=(15,15))
fig.suptitle('Meteoroid ablation simulation')

ax = fig.add_subplot(221)
ax.plot(result.t, result.y[0,:]*1e-3)
ax.set_ylabel('Velocity [km/s]')
ax.set_xlabel('Time [s]')

ax = fig.add_subplot(222)
ax.plot(result.t, np.log10(result.y[1,:]))
ax.set_ylabel('Mass [log$_{10}$(kg)]')
ax.set_xlabel('Time [s]')

ax = fig.add_subplot(223)
ax.plot(result.t, result.y[2,:]*1e-3)
ax.set_ylabel('Trajectory path [km]')
ax.set_xlabel('Time [s]')

ax = fig.add_subplot(224)
ax.plot(result.t, result.y[3,:])
ax.set_ylabel('Meteoroid temperature [K]')
ax.set_xlabel('Time [s]')

plt.show()