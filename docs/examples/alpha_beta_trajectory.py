"""
Alpha-beta model
================
"""

import numpy as np
import matplotlib.pyplot as plt

import spacecoords.spherical as sph
from metablate.physics import alpha_beta

radius = 2e-2  # m
cross_section = np.pi * radius**2
mu = 2 / 3
bulk_density = 3300.0
mass0 = bulk_density * 4 / 3 * np.pi * radius**3
ze = 75.7744
v0 = 53e3
alpha = 101.0
beta = 10.0

x_s = np.array([0, 0, 100e3])
v_s_hat = sph.sph_to_cart(np.array([0, ze - 90, 1]), degrees=True)

t_step = 5e-3
t_end = 3.36
t_steps = int(t_end / t_step)

pos = np.empty((3, t_steps), dtype=np.float64)
vel = np.empty((3, t_steps), dtype=np.float64)
mass = np.empty((t_steps,), dtype=np.float64)
t_v = np.arange(t_steps) * t_step
pos[:, 0] = x_s
vel[:, 0] = v_s_hat * alpha_beta.velocity_estimate(
    height=pos[2, 0],
    initial_velocity=v0,
    alpha=alpha,
    beta=beta,
    atmospheric_scale_height=7610.0,
)
mass[0] = alpha_beta.mass_direct(
    velocity=np.linalg.norm(vel[:, 0]),
    initial_mass=mass0,
    beta=beta,
    shape_change_coefficient=mu,
    initial_velocity=v0,
)

for ind in range(1, t_steps):
    x_prev = pos[:, ind - 1]
    v_prev = vel[:, ind - 1]
    # ugly "euler" like step method but good enough for this application
    pos[:, ind] = x_prev + v_prev * t_step
    vel[:, ind] = v_s_hat * alpha_beta.velocity_estimate(
        height=pos[2, ind],
        initial_velocity=v0,
        alpha=alpha,
        beta=beta,
        atmospheric_scale_height=7610.0,
    )
    mass[ind] = alpha_beta.mass_direct(
        velocity=np.linalg.norm(vel[:, ind]),
        initial_mass=mass0,
        beta=beta,
        shape_change_coefficient=mu,
        initial_velocity=v0,
    )


print(f"{radius=:1.3e} {cross_section=:1.3e} {mass0=:1.3e}")
print(f"{alpha=:.2f} {beta=:.2f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(t_v, pos[2, :] * 1e-3)
axes[0, 0].set_ylabel("Height [km]")
axes[0, 0].set_xlabel("Time [s]")

axes[0, 1].plot(t_v, np.linalg.norm(vel, axis=0) * 1e-3, label=f"{alpha=:.2f} {beta=:.2f}")
axes[0, 1].set_ylabel("Velocity [km/s]")
axes[0, 1].set_xlabel("Time [s]")
axes[0, 1].legend()

axes[1, 0].plot(t_v, mass)
axes[1, 0].set_ylabel("Mass [kg]")
axes[1, 0].set_xlabel("Time [s]")

axes[1, 1].plot(np.linalg.norm(vel, axis=0) * 1e-3, pos[2, :] * 1e-3)
axes[1, 1].set_xlabel("Velocity [km/s]")
axes[1, 1].set_ylabel("Height [km]")

plt.show()
