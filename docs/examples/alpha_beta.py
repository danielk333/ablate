"""
Alpha-beta model
================
"""

import numpy as np
import matplotlib.pyplot as plt

import ablate

material = ablate.material.get("asteroidal", as_dict=False)

radius = 2e-2  # m
cross_section = np.pi * radius**2
mass = material.rho_m * 4 / 3 * np.pi * radius**3
ze = 75.7744

print(f"{radius=:1.3e} {cross_section=:1.3e} {mass=:1.3e}")

alpha = ablate.physics.alpha_beta.alpha_direct(
    aerodynamic_cd=0.47,
    sea_level_rho=1.225,
    atmospheric_scale_height=7610.0,  # at T = 260 K
    initial_cross_section=cross_section,
    initial_mass=mass,
    radiant_local_elevation=90 - ze,
    degrees=True,
)
beta = ablate.physics.alpha_beta.beta_direct(
    shape_change_coefficient=2/3,
    heat_exchange_coefficient=1,  # Lambda
    initial_velocity=60e3,
    aerodynamic_cd=0.47,
    enthalpy_of_massloss=material.L*20,  # Probably non-physical value in reality
)
print(f"{alpha=:.2f} {beta=:.2f}")

vel_lims = [
    ablate.physics.alpha_beta.velocity_direct(1 - 1e-6, beta, 60e3, shape_change_coefficient=2/3),
    ablate.physics.alpha_beta.velocity_direct(1e-6, beta, 60e3, shape_change_coefficient=2/3),
]
print(vel_lims)
velocities_vec = np.linspace(vel_lims[0], vel_lims[1], 1000)

masses = ablate.physics.alpha_beta.mass_direct(
    velocity=velocities_vec,
    initial_mass=mass,
    beta=beta,
    shape_change_coefficient=2/3,
    initial_velocity=60e3,
)
heights = ablate.physics.alpha_beta.height_direct(
    velocity=velocities_vec,
    atmospheric_scale_height=7610.0,
    alpha=alpha,
    beta=beta,
    initial_velocity=60e3,
)

mass_loss = np.diff(masses)/np.diff(heights)


fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(velocities_vec*1e-3, masses)
axes[0, 0].set_xlabel("Velocity [km/s]")
axes[0, 0].set_ylabel("Mass [kg]")

axes[1, 0].plot(velocities_vec*1e-3, heights*1e-3)
axes[1, 0].set_xlabel("Velocity [km/s]")
axes[1, 0].set_ylabel("Height [km]")

axes[0, 1].plot(masses, heights*1e-3)
axes[0, 1].set_ylabel("Height [km]")
axes[0, 1].set_xlabel("Remaining mass [kg]")

axes[1, 1].plot(mass_loss, heights[1:]*1e-3)
axes[1, 1].set_xlabel("Mass loss [kg/m]")
axes[1, 1].set_ylabel("Height [km]")

plt.show() 
