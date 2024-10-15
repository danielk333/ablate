"""
Compare analytical and numerical
================================

Docstring for this example
"""

import numpy as np
import matplotlib.pyplot as plt

import ablate

material = ablate.material.get("asteroidal")

radius = 2e-2  # m
cross_section = np.pi * radius**2
mass = material["rho_m"] * 4 / 3 * np.pi * radius**3
ze = 75.7744
initial_velocity = 60e3
cd = 0.47
Lambda = 0.9
mu = 2 / 3
h0 = 7610.0  # at T = 260 K

print(f"{radius=:1.3e} {cross_section=:1.3e} {mass=:1.3e}")

alpha = ablate.physics.alpha_beta.alpha_direct(
    aerodynamic_cd=cd,
    sea_level_rho=1.225,
    atmospheric_scale_height=h0,
    initial_cross_section=cross_section,
    initial_mass=mass,
    radiant_local_elevation=90 - ze,
    degrees=True,
)
beta = ablate.physics.alpha_beta.beta_direct(
    shape_change_coefficient=mu,
    heat_exchange_coefficient=Lambda,  # Lambda
    initial_velocity=initial_velocity,
    aerodynamic_cd=cd,
    enthalpy_of_massloss=material["L"],
)
print(f"{alpha=:.2f} {beta=:.2f}")

vel_lims = [
    ablate.physics.alpha_beta.velocity_direct(
        1 - 1e-6, beta, initial_velocity, shape_change_coefficient=2 / 3
    ),
    ablate.physics.alpha_beta.velocity_direct(
        1e-6, beta, initial_velocity, shape_change_coefficient=2 / 3
    ),
]
print(vel_lims)
velocities_vec = np.linspace(vel_lims[0], vel_lims[1], 1000)

masses = ablate.physics.alpha_beta.mass_direct(
    velocity=velocities_vec,
    initial_mass=mass,
    beta=beta,
    shape_change_coefficient=mu,
    initial_velocity=initial_velocity,
)
heights = ablate.physics.alpha_beta.height_direct(
    velocity=velocities_vec,
    atmospheric_scale_height=h0,
    alpha=alpha,
    beta=beta,
    initial_velocity=initial_velocity,
)


model = ablate.KeroSzasz2008(
    atmosphere=ablate.atmosphere.AtmPymsis(),
    config={
        "options": {
            "temperature0": 290,
            "shape_factor": 1.21,
            "emissivity": 0.9,
            "sputtering": False,
            "Gamma": cd,
            "Lambda": Lambda,
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
    },
)


result = model.run(
    velocity0=initial_velocity,
    mass0=mass,
    altitude0=120e3,
    zenith_ang=ze,
    azimuth_ang=0.0,
    material_data=material,
    time=np.datetime64("2018-06-28T12:45:33"),
    lat=69.5866115,
    lon=19.221555,
    alt=100e3,
)
print(result)

alpha_est, beta_est = ablate.physics.alpha_beta.solve_alpha_beta_versionQ4(
    result.velocity[::10],
    result.altitude[::10],
    initial_velocity=result.velocity.max(),
    atmospheric_scale_height=h0,
    start=None,
    bounds=((0.001, 10000.0), (0.00001, 500.0)),
)
print(f"{alpha_est=:.2f} {beta_est=:.2f}")

norm_massloss_model = np.diff(result.mass.values) / np.diff(result.altitude.values)
norm_massloss_model = norm_massloss_model / norm_massloss_model.max()
inds = norm_massloss_model > 0.9

alpha_est2, beta_est2 = ablate.physics.alpha_beta.solve_alpha_beta_velocity_versionQ5(
    result.velocity[1:][inds],
    result.altitude[1:][inds],
    atmospheric_scale_height=h0,
)
print(f"{alpha_est2=:.2f} {beta_est2=:.2f}")


masses_est = ablate.physics.alpha_beta.mass_direct(
    velocity=result.velocity,
    initial_mass=mass,
    beta=beta_est,
    shape_change_coefficient=mu,
    initial_velocity=result.velocity.max(),
)
heights_est = ablate.physics.alpha_beta.height_direct(
    velocity=result.velocity,
    atmospheric_scale_height=h0,
    alpha=alpha_est,
    beta=beta_est,
    initial_velocity=result.velocity.max(),
)


fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle("Meteoroid ablation simulation vs analytical solution")

ax = axes[0, 0]
ax.plot(np.log10(result.mass), result.altitude * 1e-3, c="b", label="KeroSzasz2008")
ax.plot(
    np.log10(masses), heights * 1e-3, c="g", label=f"alpha-beta direct {alpha=:.2f} {beta=:.2f}"
)
ax.plot(
    np.log10(masses_est),
    heights_est * 1e-3,
    c="r",
    label=f"alpha-beta fit {alpha_est=:.2f} {beta_est=:.2f}",
)
ax.set_xlabel("Mass [log$_{10}$(kg)]")
ax.set_ylabel("Height [km]")
ax.legend()

ax = axes[1, 0]
ax.plot(result.velocity * 1e-3, result.altitude * 1e-3, c="b")
ax.plot(velocities_vec * 1e-3, heights * 1e-3, c="g")
ax.plot(result.velocity * 1e-3, heights_est * 1e-3, c="r")
ax.set_xlabel("Velocity [km/s]")
ax.set_ylabel("Height [km]")

ax = axes[0, 1]
ax.plot(
    np.diff(result.mass.values) / np.diff(result.altitude.values),
    result.altitude.values[:-1] * 1e-3,
    c="b",
)
ax.plot(
    np.diff(masses) / np.diff(heights),
    heights[:-1] * 1e-3,
    c="g",
)
ax.plot(
    np.diff(masses_est) / np.diff(heights_est),
    heights_est[:-1] * 1e-3,
    c="r",
)
ax.set_xlabel("Mass loss [kg/m]")
ax.set_ylabel("Height [km]")

ax = axes[1, 1]
ax.plot(result.temperature, result.altitude * 1e-3)
ax.set_xlabel("Meteoroid temperature [K]")
ax.set_ylabel("Height [km]")

plt.show()
