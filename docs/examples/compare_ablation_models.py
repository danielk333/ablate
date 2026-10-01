"""
Comparison of ablation models
=============================

Compare three ablation models using the same meteoroid material, mass,
geometry, entry state, and aerodynamic coefficients. Kero--Szasz and Simple
Ablation use their native MSIS atmosphere, while Alpha--Beta necessarily uses
its analytic exponential atmosphere.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from spacecoords import frames

import metablate.material as materials
import metablate.models.alpha_beta_2026 as ab
import metablate.models.kero_szasz_2008 as ks
import metablate.models.simple_ablation_2026 as sa
from metablate.atmosphere import AtmPymsis
from metablate.physics import alpha_beta
from metablate.physics.spheres import radius_from_mass

# One physical setup is converted to each model's input parameters.
material = materials.cometary
epoch = np.datetime64("2018-06-28T12:45:33")
latitude = 67 + 50 / 60 + 26.6 / 3600
longitude = 20 + 24 / 60 + 40.0 / 3600
initial_altitude = 150e3
initial_velocity = 30e3
radiant_azimuth = 10.0
entry_elevation = 45.0
initial_mass = 1e-8
minimum_relative_mass = 1e-4
aerodynamic_cd = 0.47
heat_exchange_coefficient = 1.0
shape_change_coefficient = 2 / 3

ref_altitude_range = np.linspace(80e3, 120e3, 100)

atm = AtmPymsis()
dens = atm.density(
    time=epoch,
    lat=np.array([latitude]),
    lon=np.array([longitude]),
    alt=ref_altitude_range,
)
atm_density = dens["Total"].values.squeeze()
dry_air_R = 287.05
Earth_mass = 5.97219e24
Earth_radius = 6371e3
grav_acceleration = lambda h: constants.G * Earth_mass / (Earth_radius + h) ** 2
scale_heights = dry_air_R * dens["Temperature"].values.squeeze() / grav_acceleration(ref_altitude_range)
atmospheric_scale_height = np.mean(scale_heights)

mid_index = len(ref_altitude_range) // 2
mid_alt = ref_altitude_range[mid_index]
dens_tot_mid = atm_density[mid_index]
sea_level_density = dens_tot_mid * np.exp(mid_alt / atmospheric_scale_height)

radius = radius_from_mass(initial_mass, material.bulk_density)
initial_cross_section = np.pi * radius**2
shape_factor = alpha_beta.shape_factor_direct(
    initial_cross_section,
    initial_mass,
    material.bulk_density,
)
entry_angle = np.radians(entry_elevation)

reference_position_ecef = frames.geodetic_wgs84_to_ecef(
    latitude,
    longitude,
    initial_altitude,
    degrees=True,
)
radiant_ecef = frames.azel_to_ecef(
    latitude,
    longitude,
    az=radiant_azimuth,
    el=entry_elevation,
    degrees=True,
)
initial_velocity_ecef = -radiant_ecef * initial_velocity

kero_model = ks.KeroSzasz2008(
    options=ks.KeroSzaszOptions(
        material=material,
        shape_factor=shape_factor,
        sputtering=False,
        minimum_mass=initial_mass * minimum_relative_mass,
        max_step_size=1e-2,
        max_time=8.0,
        start_altitude=initial_altitude,
    ),
)
kero_result = kero_model.run(
    parameters=ks.KeroSzaszInitialState(
        epoch=epoch,
        position_ecef=reference_position_ecef,
        velocity_ecef=initial_velocity_ecef,
        mass=initial_mass,
        # this Gamma is one half of the conventional C_d.
        drag_coefficient=0.5 * aerodynamic_cd,
        heat_transfer_coefficient=heat_exchange_coefficient,
    ),
)
kero_geodetic = frames.ecef_to_geodetic_wgs84(
    kero_result.position_ecef[0],
    kero_result.position_ecef[1],
    kero_result.position_ecef[2],
)

simple_model = sa.SimpleAblation2026(
    options=sa.SimpleAblationOptions(
        minimum_relative_mass=minimum_relative_mass,
        minimum_altitude=50e3,
        max_step_size=1e-2,
        max_time=8.0,
    ),
)
simple_result = simple_model.run(
    parameters=sa.SimpleAblationInitialState(
        epoch=epoch,
        initial_height=initial_altitude,
        initial_velocity=initial_velocity,
        entry_angle=entry_angle,
        ballistic_parameter=sa.ballistic_parameter_direct(
            aerodynamic_cd,
            initial_cross_section,
            initial_mass,
        ),
        massloss_parameter=sa.massloss_parameter_direct(
            heat_exchange_coefficient,
            material.latent_heat_of_fusion_vapourization,
            initial_cross_section,
            initial_mass,
        ),
        shape_change_coefficient=shape_change_coefficient,
        latitude=latitude,
        longitude=longitude,
    ),
)

alpha = alpha_beta.alpha_direct(
    aerodynamic_cd,
    sea_level_density,
    atmospheric_scale_height,
    initial_cross_section,
    initial_mass,
    entry_angle,
)
beta = alpha_beta.beta_direct(
    shape_change_coefficient,
    heat_exchange_coefficient,
    initial_velocity,
    aerodynamic_cd,
    material.latent_heat_of_fusion_vapourization,
)
alpha_beta_model = ab.AlphaBeta2026(
    options=ab.AlphaBetaOptions(
        minimum_relative_mass=minimum_relative_mass,
        max_step_size=1e-2,
        max_time=8.0,
        atmospheric_scale_height=atmospheric_scale_height,
    ),
)
alpha_beta_result = alpha_beta_model.run(
    parameters=ab.AlphaBetaInitialState(
        epoch=epoch,
        initial_height=initial_altitude,
        entry_velocity=initial_velocity,
        entry_angle=entry_angle,
        alpha=alpha,
        beta=beta,
        shape_change_coefficient=shape_change_coefficient,
    ),
)

results = {
    "KeroSzasz 2008": {
        "time": kero_result.t,
        "height": kero_geodetic[2],
        "velocity": kero_result.velocity,
        "relative_mass": kero_result.mass / initial_mass,
        "relative_massloss": -kero_result.massloss / initial_mass,
    },
    "AlphaBeta 2026": {
        "time": alpha_beta_result.t,
        "height": alpha_beta_result.height,
        "velocity": alpha_beta_result.velocity,
        "relative_mass": alpha_beta_result.relative_mass,
        "relative_massloss": -alpha_beta_result.massloss,
    },
    "SimpleAblation 2026": {
        "time": simple_result.t,
        "height": simple_result.height,
        "velocity": simple_result.velocity,
        "relative_mass": simple_result.relative_mass,
        "relative_massloss": -simple_result.relative_massloss,
    },
}

print(f"Material bulk density: {material.bulk_density:g} kg/m^3")
print(f"Material mass-loss enthalpy: {material.latent_heat_of_fusion_vapourization:g} J/kg")
print(f"Initial cross-section: {initial_cross_section:.6e} m^2")
print(f"Shape factor: {shape_factor:.6f}")

fig, axes = plt.subplots(2, 3, figsize=(12, 9), layout="tight")
for label, result in results.items():
    axes[0, 0].plot(result["time"], result["relative_mass"], label=label)
    axes[0, 1].plot(result["time"], result["velocity"] * 1e-3, label=label)
    axes[1, 0].plot(result["time"], result["height"] * 1e-3, label=label)
    axes[1, 1].plot(result["relative_mass"], result["height"] * 1e-3, label=label)
    axes[0, 2].plot(result["velocity"] * 1e-3, result["height"] * 1e-3, label=label)
    axes[1, 2].plot(result["relative_massloss"], result["height"] * 1e-3, label=label)

axes[0, 0].legend()
axes[0, 0].set_yscale("log")
axes[0, 0].set_ylabel("Relative mass")
axes[0, 0].set_xlabel("Time [s]")

axes[0, 1].set_ylabel("Velocity [km/s]")
axes[0, 1].set_xlabel("Time [s]")

axes[1, 0].set_ylabel("Altitude [km]")
axes[1, 0].set_xlabel("Time [s]")

axes[1, 1].set_xscale("log")
axes[1, 1].set_ylabel("Altitude [km]")
axes[1, 1].set_xlabel("Relative mass")

axes[0, 2].set_xlabel("Velocity [km/s]")
axes[0, 2].set_ylabel("Altitude [km]")

axes[1, 2].set_ylabel("Altitude [km]")
axes[1, 2].set_xlabel("Relative massloss")

fig.suptitle("Ablation models with common material and entry inputs")

plt.show()
