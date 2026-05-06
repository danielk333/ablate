"""
Example run of an ablation model
=================================

Docstring for this example
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

import metablate.models.kero_szasz_2008 as ks
import metablate.material as mat
from spacecoords import frames

logger = logging.getLogger("metablate")
logger.setLevel(logging.DEBUG)

lat = 67 + 50 / 60 + 26.6 / 3600
lon = 20 + 24 / 60 + 40.0 / 3600
alt = 100e3
reference_pos_ecef = frames.geodetic_wgs84_to_ecef(lat, lon, alt, degrees=True)
velocity_dir_ecef = frames.azel_to_ecef(lat, lon, az=0, el=-45, degrees=True)


model = ks.KeroSzasz2008(
    options=ks.KeroSzaszOptions(
        material=mat.asteroidal,
    ),
)

result = model.run(
    times=np.linspace(0, 10, 100),
    parameters=ks.KeroSzaszInitialState(
        epoch=np.datetime64("2018-06-28T12:45:33"),
        position_ecef=reference_pos_ecef,
        velocity_ecef=velocity_dir_ecef * 60e3,
        mass=1e-6,
    ),
)

print(result)

fig = plt.figure(figsize=(15, 15))
fig.suptitle("Meteoroid ablation simulation")

ax = fig.add_subplot(221)
ax.plot(result.t, np.log10(result.mass))
ax.set_ylabel("Mass [log$_{10}$(kg)]")
ax.set_xlabel("Time [s]")

ax = fig.add_subplot(222)
ax.plot(result.t, result.velocity * 1e-3)
ax.set_ylabel("Velocity [km/s]")
ax.set_xlabel("Time [s]")

ax = fig.add_subplot(223)
ax.plot(result.t, result.position * 1e-3)
ax.set_ylabel("Position on trajectory [km]")
ax.set_xlabel("Time [s]")

ax = fig.add_subplot(224)
ax.plot(result.t, result.temperature)
ax.set_ylabel("Meteoroid temperature [K]")
ax.set_xlabel("Time [s]")


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111)
ax.plot(
    np.diff(result.mass.values) / np.diff(result.altitude.values),
    result.altitude.values[:-1] * 1e-3,
)
ax.set_xlabel("Mass loss [kg/m]")
ax.set_ylabel("Altitude [km]")


plt.show()
