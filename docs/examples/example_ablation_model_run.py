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
from metablate import setup_logging
from spacecoords import frames

setup_logging(level=logging.DEBUG)

lat = 67 + 50 / 60 + 26.6 / 3600
lon = 20 + 24 / 60 + 40.0 / 3600
alt = 100e3
reference_pos_ecef = frames.geodetic_wgs84_to_ecef(lat, lon, alt, degrees=True)
velocity_dir_ecef = frames.azel_to_ecef(lat, lon, az=0, el=-45, degrees=True)


model = ks.KeroSzasz2008(
    options=ks.KeroSzaszOptions(
        material=mat.cometary,
        sputtering=False,
        minimum_mass=1e-11,
        max_step_size=1e-3,
        start_altitude=150e3,
    ),
)

result = model.run(
    parameters=ks.KeroSzaszInitialState(
        epoch=np.datetime64("2018-06-28T12:45:33"),
        position_ecef=reference_pos_ecef,
        velocity_ecef=velocity_dir_ecef * 60e3,
        mass=1e-6,
    ),
)
print(f"{result.runtime=} s")
llh = frames.ecef_to_geodetic_wgs84(
    result.position_ecef[0, :],
    result.position_ecef[1, :],
    result.position_ecef[2, :],
)

fig = plt.figure(figsize=(15, 15))
fig.suptitle("Meteoroid ablation simulation")

ax = fig.add_subplot(231)
ax.plot(result.t, np.log10(result.mass))
ax.set_ylabel("Mass [log$_{10}$(kg)]")
ax.set_xlabel("Time [s]")

ax = fig.add_subplot(232)
ax.plot(result.t, result.velocity * 1e-3)
ax.set_ylabel("Velocity [km/s]")
ax.set_xlabel("Time [s]")

ax = fig.add_subplot(233)
ax.plot(result.t, llh[2, :] * 1e-3)
ax.set_ylabel("Altitude [km]")
ax.set_xlabel("Time [s]")

ax = fig.add_subplot(234)
ax.plot(result.t, result.temperature)
ax.set_ylabel("Meteoroid temperature [K]")
ax.set_xlabel("Time [s]")


ax = fig.add_subplot(235)
ax.plot(llh[0, :], llh[1, :])
ax.set_ylabel("Latitude [deg]")
ax.set_xlabel("Longitude [deg]")

ax = fig.add_subplot(236)
ax.plot(
    np.diff(result.mass) / np.diff(llh[2, :]),
    llh[2, :-1] * 1e-3,
)
ax.set_xlabel("Mass loss [kg/m]")
ax.set_ylabel("Altitude [km]")


plt.show()
