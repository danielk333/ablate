"""
Example run of an ablation model
=================================

Docstring for this example
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

import metablate

logger = logging.getLogger("metablate")
logger.setLevel(logging.DEBUG)


model = metablate.KeroSzasz2008(
    atmosphere = metablate.atmosphere.AtmPymsis(),
    config = {
        "options": {
            "temperature0": 290,
            "shape_factor": 1.21,
            "emissivity": 0.9,
            "sputtering": False,
            "Gamma": None,
            "Lambda": None,
        },
        "atmosphere": {
            "version": 2.1,
        },
        "integrate": {
            "minimum_mass_kg": 1e-11,
            "max_step_size_sec": 1e-3,
            "max_time_sec": 100.0,
            "method": "RK45",
        },
    }
)

material_data = metablate.material.get("iron")
print(material_data)

result = model.run(
    velocity0=60 * 1e3,
    mass0=1e-6,
    # mass0=1e-3,
    altitude0=120e3,
    zenith_ang=75.7744,
    azimuth_ang=0.0,
    material_data=material_data,
    time=np.datetime64("2018-06-28T12:45:33"),
    lat=69.5866115,
    lon=19.221555,
    alt=100e3,
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
