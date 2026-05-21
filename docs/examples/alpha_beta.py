"""
Alpha-beta model
================
"""

import numpy as np
import matplotlib.pyplot as plt

import metablate

material = metablate.material.get("asteroidal", as_dict=False)

r_vec = [2e-3, 2e-2, 2e-1]  # m
ze = 75.7744

fig, axes = plt.subplots(2, 2)

for radius in r_vec:
    cross_section = np.pi * radius**2
    mass = material.bulk_density * 4 / 3 * np.pi * radius**3

    print(f"{radius=:1.3e} {cross_section=:1.3e} {mass=:1.3e}")

    alpha = metablate.physics.alpha_beta.alpha_direct(
        aerodynamic_cd=0.47,
        sea_level_rho=1.225,
        atmospheric_scale_height=7610.0,  # at T = 260 K
        initial_cross_section=cross_section,
        initial_mass=mass,
        radiant_local_elevation=90 - ze,
        degrees=True,
    )
    beta = metablate.physics.alpha_beta.beta_direct(
        shape_change_coefficient=2 / 3,
        heat_exchange_coefficient=1,  # Lambda
        initial_velocity=60e3,
        aerodynamic_cd=0.47,
        enthalpy_of_massloss=material.latent_heat_of_fusion_vapourization
        * 20,  # Probably non-physical value in reality?
    )
    print(f"{alpha=:.2f} {beta=:.2f}")

    vel_lims = [
        metablate.physics.alpha_beta.velocity_direct(1 - 1e-6, beta, 60e3, shape_change_coefficient=2 / 3),
        metablate.physics.alpha_beta.velocity_direct(1e-6, beta, 60e3, shape_change_coefficient=2 / 3),
    ]
    print(vel_lims)
    velocities_vec = np.linspace(vel_lims[0], vel_lims[1], 1000)

    masses = metablate.physics.alpha_beta.mass_direct(
        velocity=velocities_vec,
        initial_mass=mass,
        beta=beta,
        shape_change_coefficient=2 / 3,
        initial_velocity=60e3,
    )
    heights = metablate.physics.alpha_beta.height_direct(
        velocity=velocities_vec,
        atmospheric_scale_height=7610.0,
        alpha=alpha,
        beta=beta,
        initial_velocity=60e3,
    )

    mass_loss = np.diff(masses) / np.diff(heights)

    axes[0, 0].semilogy(velocities_vec * 1e-3, masses, label=f"{alpha=:.2f} {beta=:.2f}")
    axes[1, 0].plot(velocities_vec * 1e-3, heights * 1e-3)
    axes[0, 1].plot(masses / mass, heights * 1e-3)
    axes[1, 1].plot(mass_loss / mass, heights[1:] * 1e-3)

axes[0, 0].set_xlabel("Velocity [km/s]")
axes[0, 0].set_ylabel("Mass [kg]")
axes[0, 0].legend()

axes[1, 0].set_xlabel("Velocity [km/s]")
axes[1, 0].set_ylabel("Height [km]")

axes[0, 1].set_ylabel("Height [km]")
axes[0, 1].set_xlabel("Normalized remaining mass [1]")

axes[1, 1].set_xlabel("Normalized mass loss [1/m]")
axes[1, 1].set_ylabel("Height [km]")

plt.show()
