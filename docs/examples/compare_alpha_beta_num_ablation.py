"""
Compare analytical and numerical
================================

Docstring for this example
"""

import numpy as np
import matplotlib.pyplot as plt

import ablate

material = ablate.material.get("asteroidal")

radius = 8e-4  # m
# radius = 2e-3  # m
# radius = 2e-2  # m
# radius = 2e-1  # m
cross_section = np.pi * radius**2
mass = material["rho_m"] * 4 / 3 * np.pi * radius**3
ze = 75.7744
initial_velocity = 60e3
cd = 0.47
Lambda = 0.9
mu = 2 / 3
shape_factor = 1.21
h0 = 7610.0  # at T = 260 K
sea_level_rho = 1.225

print(f"{radius=:1.3e} {cross_section=:1.3e} {mass=:1.3e}")

alpha = ablate.physics.alpha_beta.alpha_direct(
    # aerodynamic_cd=cd,
    aerodynamic_cd=0.8,
    sea_level_rho=sea_level_rho,
    atmospheric_scale_height=h0,
    initial_cross_section=cross_section,
    initial_mass=mass,
    radiant_local_elevation=90 - ze,
    degrees=True,
)
beta = ablate.physics.alpha_beta.beta_direct(
    shape_change_coefficient=mu,
    # heat_exchange_coefficient=Lambda,
    heat_exchange_coefficient=1.0,
    initial_velocity=initial_velocity,
    aerodynamic_cd=0.8,
    # enthalpy_of_massloss=material["L"],
    enthalpy_of_massloss=material["L"] * 30,
)
print(f"{alpha=:.2e} {beta=:.2e}")

vel_lims = [
    ablate.physics.alpha_beta.velocity_direct(
        1 - 1e-6, beta, initial_velocity, shape_change_coefficient=mu
    ),
    ablate.physics.alpha_beta.velocity_direct(
        1e-6, beta, initial_velocity, shape_change_coefficient=mu
    ),
]
print(vel_lims)
velocities_vec = np.linspace(vel_lims[0], vel_lims[1], 1000)

mass0 = ablate.physics.alpha_beta.initial_mass_direct(
    alpha,
    aerodynamic_cd=cd,
    sea_level_rho=sea_level_rho,
    atmospheric_scale_height=h0,
    radiant_local_elevation=90 - ze,
    bulk_density=material["rho_m"],
    shape_factor=shape_factor,
    degrees=True,
)
masses = ablate.physics.alpha_beta.mass_direct(
    velocity=velocities_vec,
    initial_mass=mass0,
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

atmosphere = ablate.atmosphere.AtmPymsis()

model = ablate.KeroSzasz2008(
    atmosphere=atmosphere,
    config={
        "options": {
            "temperature0": 290,
            "shape_factor": shape_factor,
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
            # "minimum_mass_kg": 1e-6,
            # "max_step_size_sec": 1e-4,
            "max_step_size_sec": 1e-3,
            "max_time_sec": 100.0,
            "method": "RK45",
        },
    },
)

atm_heights = np.linspace(80, 140, 1000) * 1e3
ab_density = ablate.physics.alpha_beta.atmosphere_density(atm_heights, h0, sea_level_rho)
msis_density = atmosphere.density(
    time=np.datetime64("2018-06-28T12:45:33"),
    lat=69.5866115,
    lon=19.221555,
    alt=atm_heights,
)["Total"].values.squeeze()

fig, ax = plt.subplots()
ax.plot(ab_density, heights * 1e-3, "-b", label="Isothermal approximation")
ax.plot(msis_density, heights * 1e-3, "-g", label="MSIS V2.1")
ax.legend()

plt.show()

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
    result.velocity.values[::4],
    result.altitude.values[::4],
    initial_velocity=result.velocity.values.max(),
    atmospheric_scale_height=h0,
    start=None,
    bounds=((0.001, 10000.0), (0.00001, 500.0)),
)
print(f"{alpha_est=:.2e} {beta_est=:.2e}")

norm_massloss_model = np.diff(result.mass.values) / np.diff(result.altitude.values)
norm_massloss_model = norm_massloss_model / norm_massloss_model.max()
inds = np.full(result.velocity.shape, False, dtype=bool)
inds[::4] = True
inds[0] = False
inds[1:] = np.logical_and(inds[1:], norm_massloss_model > 0.5)

vnorm = result.velocity.values[::4] / result.velocity.values[::4].max()
norm_v = np.arange(vnorm.min(), 1, 0.00005)
model_velocities = norm_v * initial_velocity

alpha_partial_est, beta_partial_est, vel0_est = (
    ablate.physics.alpha_beta.solve_alpha_beta_velocity_versionQ5(
        result.velocity.values[inds],
        result.altitude.values[inds],
        atmospheric_scale_height=h0,
        bounds=((0.01, 1000000.0), (0.0001, 300.0), (0, None)),
    )
)
print(f"{alpha_partial_est=:.2e} {beta_partial_est=:.2e} {vel0_est=:.2e}")

vnorm_part = result.velocity.values[inds] / result.velocity.values[inds].max()
norm_part_v = np.arange(vnorm.min(), 1, 0.00005)
model_part_velocities = norm_part_v * vel0_est

mass0_est = ablate.physics.alpha_beta.initial_mass_direct(
    alpha_est,
    aerodynamic_cd=cd,
    sea_level_rho=sea_level_rho,
    atmospheric_scale_height=h0,
    radiant_local_elevation=90 - ze,
    bulk_density=material["rho_m"],
    shape_factor=shape_factor,
    degrees=True,
)
masses_est = ablate.physics.alpha_beta.mass_direct(
    velocity=model_velocities,
    initial_mass=mass0_est,
    beta=beta_est,
    shape_change_coefficient=mu,
    initial_velocity=initial_velocity,
)
heights_est = ablate.physics.alpha_beta.height_direct(
    velocity=model_velocities,
    atmospheric_scale_height=h0,
    alpha=alpha_est,
    beta=beta_est,
    initial_velocity=initial_velocity,
)

mass0_partial_est = ablate.physics.alpha_beta.initial_mass_direct(
    alpha_partial_est,
    aerodynamic_cd=cd,
    sea_level_rho=sea_level_rho,
    atmospheric_scale_height=h0,
    radiant_local_elevation=90 - ze,
    bulk_density=material["rho_m"],
    shape_factor=shape_factor,
    degrees=True,
)
print(f"{mass0_est=:.2e} {mass0_partial_est=:.2e}")
masses_partial_est = ablate.physics.alpha_beta.mass_direct(
    velocity=model_part_velocities,
    initial_mass=mass0_partial_est,
    beta=beta_partial_est,
    shape_change_coefficient=mu,
    initial_velocity=vel0_est,
)
heights_partial_est = ablate.physics.alpha_beta.height_direct(
    velocity=model_part_velocities,
    atmospheric_scale_height=h0,
    alpha=alpha_partial_est,
    beta=beta_partial_est,
    initial_velocity=vel0_est,
)

fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle("Meteoroid ablation simulation")

axes[0, 0].plot(result.t, result.velocity * 1e-3, "-b")
axes[0, 0].plot(result.t[inds], result.velocity[inds] * 1e-3, "xb")
axes[0, 0].set_ylabel("Velocity [km/s]")
axes[0, 0].set_xlabel("Time [s]")

axes[1, 0].plot(result.altitude * 1e-3, result.velocity * 1e-3, "-b")
axes[1, 0].plot(result.altitude[inds] * 1e-3, result.velocity[inds] * 1e-3, "xb")
axes[1, 0].plot(heights_est * 1e-3, model_velocities * 1e-3, "-r")
axes[1, 0].plot(heights_partial_est * 1e-3, model_part_velocities * 1e-3, "--r")
axes[1, 0].set_ylabel("Velocity [km/s]")
axes[1, 0].set_ylabel("Height [km]")

axes[0, 1].plot(result.altitude * 1e-3, result.velocity * 1e-3, "-b")
axes[0, 1].plot(heights_est * 1e-3, model_velocities * 1e-3, "-r")
axes[0, 1].set_ylabel("Velocity [km/s]")
axes[0, 1].set_ylabel("Height [km]")

axes[1, 1].plot(result.altitude[inds] * 1e-3, result.velocity[inds] * 1e-3, "xb")
axes[1, 1].plot(heights_partial_est * 1e-3, model_part_velocities * 1e-3, "--r")
axes[1, 1].set_ylabel("Velocity [km/s]")
axes[1, 1].set_ylabel("Height [km]")

fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle("Meteoroid ablation simulation vs analytical solution")

ax = axes[0, 0]
ax.plot(np.log10(result.mass), result.altitude * 1e-3, c="b", label=f"KeroSzasz2008 {mass=:.2e}")
ax.plot(
    np.log10(result.mass[inds]),
    result.altitude[inds] * 1e-3,
    marker="x",
    ls="none",
    c="b",
    label="KeroSzasz2008 partial",
)
ax.plot(np.log10(masses), heights * 1e-3, c="g", label=f"{alpha=:.2e} {beta=:.2e} {mass0=:.2e}")
ax.plot(
    np.log10(masses_est),
    heights_est * 1e-3,
    c="r",
    label=f"{alpha_est=:.2e} {beta_est=:.2e} {mass0_est=:.2e}",
)
ax.plot(
    np.log10(masses_partial_est),
    heights_partial_est * 1e-3,
    ls="--",
    c="r",
    label=f"{alpha_partial_est=:.2e} {beta_partial_est=:.2e} {mass0_partial_est=:.2e}",
)
ax.set_xlabel("Mass [log$_{10}$(kg)]")
ax.set_ylabel("Height [km]")
ax.legend()

ax = axes[1, 0]
ax.plot(result.velocity * 1e-3, result.altitude * 1e-3, c="b")
ax.plot(result.velocity[inds] * 1e-3, result.altitude[inds] * 1e-3, marker="x", ls="none", c="b")
ax.plot(velocities_vec * 1e-3, heights * 1e-3, c="g")
ax.plot(model_velocities * 1e-3, heights_est * 1e-3, c="r")
ax.plot(model_part_velocities * 1e-3, heights_partial_est * 1e-3, ls="--", c="r")
ax.set_xlabel("Velocity [km/s]")
ax.set_ylabel("Height [km]")

massloss_sim = np.diff(result.mass.values) / np.diff(result.altitude.values)
massloss = np.diff(masses) / np.diff(heights)
massloss_est = np.diff(masses_est) / np.diff(heights_est)
massloss_partial_est = np.diff(masses_partial_est) / np.diff(heights_partial_est)

ax = axes[0, 1]
ax.plot(
    massloss_sim / np.max(massloss_sim),
    result.altitude.values[:-1] * 1e-3,
    c="b",
)
ax.plot(
    massloss_sim[inds[1:]] / np.max(massloss_sim[inds[1:]]),
    result.altitude.values[1:][inds[1:]] * 1e-3,
    marker="x",
    ls="none",
    c="b",
)
ax.plot(
    massloss / np.max(massloss),
    heights[1:] * 1e-3,
    c="g",
)
ax.plot(
    massloss_est / np.max(massloss_est),
    heights_est[1:] * 1e-3,
    c="r",
)
ax.plot(
    massloss_partial_est / np.max(massloss_partial_est),
    heights_partial_est[1:] * 1e-3,
    ls="--",
    c="r",
)
ax.set_xlabel("Normalized mass loss [1]")
ax.set_ylabel("Height [km]")

ax = axes[1, 1]
ax.plot(result.temperature, result.altitude * 1e-3)
ax.set_xlabel("Meteoroid temperature [K]")
ax.set_ylabel("Height [km]")

plt.show()
