"""
Alpha-beta model
================
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

import metablate

logger = logging.getLogger("metablate")
logger.setLevel(logging.DEBUG)

model = metablate.StulovMirskiiVislyi1995(
    atmosphere = metablate.atmosphere.AtmPymsis(),
    config = {
        "atmosphere": {
            "version": 2.1,
        },
        "integrate": {
            "minimum_mass_kg": 1e-7,
            "max_step_size_sec": 5e-4,
            "max_time_sec": 100.0,
            "method": "RK45",
        },
    }
)

material_data = metablate.material.get("asteroidal")
print(material_data)

# assume spherical
mass0 = 1e-1
radius0 = np.cbrt((mass0 / material_data["rho_m"]) * (3 / 4) / np.pi)
area0 = np.pi * radius0 ** 2

result = model.run(
    initial_velocity = 34e3,
    initial_mass = mass0,
    initial_altitude = 120e3,
    initial_radiant_local_elevation = np.radians(30.0),
    epoch=np.datetime64("2018-06-28T12:45:33"),
    lat=69.5866115,
    lon=19.221555,
    enthalpy_of_massloss=material_data["L"]/10,
    initial_cross_sectional_area = area0,
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
ax.plot(result.t, np.degrees(result.trajectory_angle))
ax.set_ylabel("Angle of trajectory [deg]")
ax.set_xlabel("Time [s]")


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111)
ax.plot(
    np.diff(result.mass.values) / np.diff(result.height.values),
    result.height.values[:-1] * 1e-3,
)
ax.set_xlabel("Mass loss [kg/m]")
ax.set_ylabel("Height [km]")


plt.show()

