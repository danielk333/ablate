"""
Fireball ablation
=================
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

import metablate.models.stulov_mirskii_vislyi_1995 as smv
import metablate.material as mat
from metablate import setup_logging
from spacecoords import frames

setup_logging(level=logging.DEBUG)

lat = 67 + 50 / 60 + 26.6 / 3600
lon = 20 + 24 / 60 + 40.0 / 3600
alt = 150e3
vel0 = 30e3
material_data = mat.asteroidal
reference_pos_ecef = frames.geodetic_wgs84_to_ecef(lat, lon, alt, degrees=True)
ref_point_vel_el = [20, 45, 70]
# assume spherical
mass0 = 1e-1
radius0 = np.cbrt((mass0 / material_data.bulk_density) * (3 / 4) / np.pi)
area0 = np.pi * radius0**2


model = smv.StulovMirskiiVislyi1995(
    options=smv.SMV_Options(
        material=material_data,
        minimum_mass=1e-11,
        max_step_size=1e-2,
        start_altitude=alt,
    ),
)

fig, axes = plt.subplots(2, 3, figsize=(15, 15), layout="tight")
axes = axes.flatten()
fig.suptitle("Meteoroid ablation simulation")

for vel_el in ref_point_vel_el:
    velocity_dir_ecef = frames.azel_to_ecef(
        lat,
        lon,
        az=10,
        el=-vel_el,
        degrees=True,
    )
    result = model.run(
        parameters=smv.SMV_InitialState(
            epoch=np.datetime64("2018-06-28T12:45:33"),
            position_ecef=reference_pos_ecef,
            velocity_ecef=velocity_dir_ecef * vel0,
            mass=mass0,
            initial_cross_sectional_area=area0,
        ),
    )
    print(f"{result.runtime=} s")
    llh = frames.ecef_to_geodetic_wgs84(
        result.position_ecef[0, :],
        result.position_ecef[1, :],
        result.position_ecef[2, :],
    )

    axes[0].plot(result.t, np.log10(result.mass))
    axes[1].plot(result.t, result.velocity * 1e-3)
    axes[2].plot(result.t, llh[2, :] * 1e-3)
    axes[3].plot(result.t, np.degrees(result.trajectory_elevation))
    axes[4].plot(llh[0, :], llh[1, :])
    axes[5].plot(
        np.diff(result.mass) / np.diff(llh[2, :]),
        llh[2, :-1] * 1e-3,
    )

axes[0].set_ylabel("Mass [log$_{10}$(kg)]")
axes[0].set_xlabel("Time [s]")

axes[1].set_ylabel("Velocity [km/s]")
axes[1].set_xlabel("Time [s]")

axes[2].set_ylabel("Altitude [km]")
axes[2].set_xlabel("Time [s]")

axes[3].set_ylabel("Trajectory elevation [deg]")
axes[3].set_xlabel("Time [s]")

axes[4].set_ylabel("Latitude [deg]")
axes[4].set_xlabel("Longitude [deg]")

axes[5].set_xlabel("Mass loss [kg/m]")
axes[5].set_ylabel("Altitude [km]")


plt.show()
