"""
Example run of an ablation model
=================================

Docstring for this example
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from spacecoords import frames

import metablate.material as mat
import metablate.models.simple_ablation_2026 as sa
from metablate.physics.spheres import radius_from_mass
from metablate import setup_logging

setup_logging(level=logging.DEBUG)

lat = 67 + 50 / 60 + 26.6 / 3600
lon = 20 + 24 / 60 + 40.0 / 3600
alt = 100e3

mass = 1e-8
density = 1000
radius = radius_from_mass(mass, density)
cross_section = np.pi * radius**2

model = sa.SimpleAblation2026(
    options=sa.SimpleAblationOptions(
        max_step_size=1e-2,
        # linear_trajectory=True,
    ),
)
ballistic_parameter = sa.ballistic_parameter_direct(
    aerodynamic_cd=0.47,
    initial_cross_section=cross_section,
    initial_mass=mass,
)
massloss_parameter = sa.massloss_parameter_direct(
    heat_exchange_coefficient=1,
    enthalpy_of_massloss=mat.cometary.latent_heat_of_fusion_vapourization,
    initial_cross_section=cross_section,
    initial_mass=mass,
)
print(ballistic_parameter, massloss_parameter)
a_vec = 10 ** np.linspace(-3, -1, 5)
b_vec = 10 ** np.linspace(-9, -7, 5)

results = []
for alpha_prim in a_vec:
    result = model.run(
        parameters=sa.SimpleAblationInitialState(
            epoch=np.datetime64("2018-06-28T12:45:33"),
            initial_height=alt,
            initial_velocity=60e3,
            entry_angle=np.radians(70),
            ballistic_parameter=alpha_prim,
            massloss_parameter=b_vec[0],
            shape_change_coefficient=2 / 3,
            latitude=lat,
            longitude=lon,
        ),
    )
    results.append((result, alpha_prim, b_vec[0]))
    print(f"{result.runtime=} s")
for beta_prim in b_vec:
    result = model.run(
        parameters=sa.SimpleAblationInitialState(
            epoch=np.datetime64("2018-06-28T12:45:33"),
            initial_height=alt,
            initial_velocity=60e3,
            entry_angle=np.radians(70),
            ballistic_parameter=a_vec[0],
            massloss_parameter=beta_prim,
            shape_change_coefficient=2 / 3,
            latitude=lat,
            longitude=lon,
        ),
    )
    results.append((result, a_vec[0], beta_prim))
    print(f"{result.runtime=} s")

# We can also make it not ablate and treat it as a ballistic trajectory
model.options.max_step_size = 10.0
result_bal = model.run(
    parameters=sa.SimpleAblationInitialState(
        epoch=np.datetime64("2018-06-28T12:45:33"),
        initial_height=alt,
        initial_velocity=60e3,
        entry_angle=np.radians(70),
        ballistic_parameter=a_vec[0],
        massloss_parameter=0,
        shape_change_coefficient=2 / 3,
        latitude=lat,
        longitude=lon,
    ),
)

fig, axes = plt.subplots(2, 2, figsize=(15, 15), layout="tight")
axes = axes.flatten()

for result, a, b in results:
    axes[0].plot(result.t, np.log10(result.relative_mass), label=f"ballistic={a:.1e} massloss={b:.1e}")
    axes[1].plot(result.t, result.velocity * 1e-3)
    axes[2].plot(result.t, result.height * 1e-3)
    axes[3].plot(result.t, np.degrees(result.trajectory_elevation))

axes[0].legend()
axes[0].set_ylabel("Relative mass [log$_{10}$(kg)]")
axes[0].set_xlabel("Time [s]")
axes[1].set_ylabel("Velocity [km/s]")
axes[1].set_xlabel("Time [s]")
axes[2].set_ylabel("Altitude [km]")
axes[2].set_xlabel("Time [s]")
axes[3].set_ylabel("Trajectory angle [deg]")
axes[3].set_xlabel("Time [s]")
fig.suptitle("Meteoroid ablation simulation")

fig, axes = plt.subplots(2, 2, figsize=(15, 15), layout="tight")
axes = axes.flatten()

result = result_bal
axes[0].plot(result.t, np.log10(result.relative_mass), label=f"ballistic={a:.1e} massloss={b:.1e}")
axes[1].plot(result.t, result.velocity * 1e-3)
axes[2].plot(result.t, result.height * 1e-3)
axes[3].plot(result.t, np.degrees(result.trajectory_elevation))

axes[0].legend()
axes[0].set_ylabel("Relative mass [log$_{10}$(kg)]")
axes[0].set_xlabel("Time [s]")
axes[1].set_ylabel("Velocity [km/s]")
axes[1].set_xlabel("Time [s]")
axes[2].set_ylabel("Altitude [km]")
axes[2].set_xlabel("Time [s]")
axes[3].set_ylabel("Trajectory angle [deg]")
axes[3].set_xlabel("Time [s]")
fig.suptitle("Meteoroid ablation simulation")

plt.show()
