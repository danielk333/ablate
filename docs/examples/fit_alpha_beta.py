"""
Alpha-beta model usage
=======================

"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from tqdm import tqdm

import metablate
from metablate.models.alpha_beta_2026 import AlphaBetaOptions, solve_alpha_beta_posterior
from metablate.physics import alpha_beta

HERE = Path(__file__).parent.absolute()
data_file = HERE / "data" / "20140104-025436.json"

# Load some data
data = {}
with open(data_file, "r") as fh:
    json_data = json.load(fh)
data["vel"] = np.array([x if x is not None else np.nan for x in json_data.pop("velocity")])
data["alt"] = np.array([x if x is not None else np.nan for x in json_data.pop("height")])
data["vel_std"] = np.array([x if x is not None else np.nan for x in json_data.pop("vel_std")])
data["alt_std"] = np.array([x if x is not None else np.nan for x in json_data.pop("alt_std")])
data["t"] = np.array(json_data["t"])
meta = {
    "date": json_data.pop("date"),
    "slope": 90 - json_data.pop("slope"),
}

not_nans = np.logical_not(
    np.logical_or.reduce(
        [
            np.isnan(data["alt"]),
            np.isnan(data["vel"]),
        ]
    )
)
data["t"] = data["t"][not_nans]
data["alt"] = data["alt"][not_nans]
data["vel"] = data["vel"][not_nans]
data["vel_std"] = data["vel_std"][not_nans]
data["alt_std"] = data["alt_std"][not_nans]


vel_obs0 = np.max(data["vel"])

lat = 34.0 + (51.0 + 14.50 / 60.0) / 60.0
lon = 136.0 + (06.0 + 20.24 / 60.0) / 60.0

atm = metablate.atmosphere.AtmPymsis()
dens = atm.density(
    time=np.datetime64("2018-07-28T00:00:00"),
    lat=np.array([lat]),
    lon=np.array([lon]),
    alt=data["alt"],
)
atm_density = dens["Total"].values.squeeze()
dry_air_R = 287.05
Earth_mass = 5.97219e24
Earth_radius = 6371e3
grav_acceleration = lambda h: constants.G * Earth_mass / (Earth_radius + h) ** 2
scale_heights = dry_air_R * dens["Temperature"].values.squeeze() / grav_acceleration(data["alt"])
h0 = np.mean(scale_heights)

mid_alt = (np.max(data["alt"]) + np.min(data["alt"])) / 2
alt = np.linspace(np.min(data["alt"]), np.max(data["alt"]), 1000)
dens = atm.density(
    time=np.datetime64("2018-07-28T00:00:00"),
    lat=np.array([lat]),
    lon=np.array([lon]),
    alt=alt,
)
atm_density = dens["Total"].values.squeeze()
dens_mid = atm.density(
    time=np.datetime64("2018-07-28T00:00:00"),
    lat=np.array([lat]),
    lon=np.array([lon]),
    alt=np.array([mid_alt]),
)
dens_tot_mid = dens_mid["Total"].values.squeeze()
proxy_surface_density = dens_tot_mid * np.exp(mid_alt / h0)

fig, axes = plt.subplots(2, 1)
axes[0].semilogx(atm_density, alt * 1e-3, label="MSIS")
axes[0].semilogx(proxy_surface_density * np.exp(-alt / h0), alt * 1e-3, label="Locally scaled isothermal")
axes[0].set_ylabel("Altitude [km]")
axes[0].set_xlabel("Atmospheric density [kg/m^3]")
axes[0].legend()

axes[1].plot(proxy_surface_density * np.exp(-alt / h0) / atm_density * 100 - 100, alt * 1e-3)
axes[1].set_ylabel("Altitude [km]")
axes[1].set_xlabel("Relative error [%]")

alpha_Q4, beta_Q4 = alpha_beta.solve_alpha_beta_versionQ4(
    data["vel"],
    data["alt"],
    initial_velocity=vel_obs0,
    atmospheric_scale_height=h0,
    bounds=((1e-4, None), (1.0, None)),
)
alpha_Q5, beta_Q5, vel_Q5 = alpha_beta.solve_alpha_beta_velocity_versionQ5(
    data["vel"],
    data["alt"],
    bounds=((1e-2, 1e8), (1.0, 1e3), (0, None)),
    atmospheric_scale_height=h0,
)

pbar = tqdm(desc="Optimization")
res = solve_alpha_beta_posterior(
    data["t"],
    data["alt"],
    data["alt_std"],
    data["vel"],
    data["vel_std"],
    entry_angle=meta["slope"],
    shape_change_coefficient=2 / 3,
    start=[alpha_Q5, beta_Q5, vel_Q5],
    options=AlphaBetaOptions(
        max_step_size=1e-2,
        atmospheric_scale_height=h0,
        target_normalized_velocity=np.min(data["vel"]) / (np.max(data["vel"]) * 1.2),
    ),
    bounds=((1e-2, 1e8), (0.5, 1e3), (0, None)),
    minimize_kwargs={
        "method": "Nelder-Mead",
        "callback": lambda intermediate_result: pbar.update(1),
    },
)
print(res)
alpha_P, beta_P, vel_P = res.x
pbar.close()

print(f"{alpha_Q4=:.3e} {beta_Q4=:.3e}")
print(f"{alpha_Q5=:.3e} {beta_Q5=:.3e} {vel_Q5=:.3e}")
print(f"{alpha_P=:.3e} {beta_P=:.3e} {vel_P=:.3e}")
bulk_density = 3000.0

mass_Q4 = alpha_beta.initial_mass_direct(
    alpha_Q5,
    aerodynamic_cd=2.0,
    sea_level_rho=proxy_surface_density,
    atmospheric_scale_height=h0,
    radiant_local_elevation=meta["slope"],
    bulk_density=bulk_density,
    shape_factor=1.21,
    degrees=True,
)
mass_Q5 = alpha_beta.initial_mass_direct(
    alpha_Q5,
    aerodynamic_cd=2.0,
    sea_level_rho=proxy_surface_density,
    atmospheric_scale_height=h0,
    radiant_local_elevation=meta["slope"],
    bulk_density=bulk_density,
    shape_factor=1.21,
    degrees=True,
)
mass_P = alpha_beta.initial_mass_direct(
    alpha_P,
    aerodynamic_cd=2.0,
    sea_level_rho=proxy_surface_density,
    atmospheric_scale_height=h0,
    radiant_local_elevation=meta["slope"],
    bulk_density=bulk_density,
    shape_factor=1.21,
    degrees=True,
)
diameter_Q4 = alpha_beta.mass_to_radius(mass_Q4, bulk_density) * 2
diameter_Q5 = alpha_beta.mass_to_radius(mass_Q5, bulk_density) * 2
diameter_P = alpha_beta.mass_to_radius(mass_P, bulk_density) * 2
print(f"{mass_Q4=} {diameter_Q4=}")
print(f"{mass_Q5=} {diameter_Q5=}")
print(f"{mass_P=} {diameter_P=}")

norm_v = np.arange(np.nanmin(data["vel"]) / vel_P, 1, 0.00005)

heights_Q4 = alpha_beta.height_direct(
    norm_v,
    h0,
    alpha_Q4,
    beta_Q4,
)
heights_Q5 = alpha_beta.height_direct(
    norm_v,
    h0,
    alpha_Q5,
    beta_Q5,
)
heights_P = alpha_beta.height_direct(
    norm_v,
    h0,
    alpha_P,
    beta_P,
)

fig, ax = plt.subplots()
ax.plot(data["vel"] * 1e-3, data["alt"] * 1e-3, ".b")
ax.plot(
    norm_v * vel_obs0 * 1e-3,
    heights_Q4 * 1e-3,
    "--r",
    label=f"ln($\\alpha$)={np.log(alpha_Q4):.2f} {beta_Q4=:.2f} vel0={vel_obs0 * 1e-3:.1f} km/s",
)
ax.plot(
    norm_v * vel_Q5 * 1e-3,
    heights_Q5 * 1e-3,
    "-r",
    label=f"ln($\\alpha$)={np.log(alpha_Q5):.2f} {beta_Q5=:.2f} vel0={vel_Q5 * 1e-3:.1f} km/s",
)
ax.plot(
    norm_v * vel_P * 1e-3,
    heights_P * 1e-3,
    "-g",
    label=f"ln($\\alpha$)={np.log(alpha_P):.2f} {beta_P=:.2f} vel0={vel_P * 1e-3:.1f} km/s",
)
ax.set_xlabel("Velocity [km/s]")
ax.set_ylabel("Height [km]")
ax.set_title("alpha-beta fitting method comparison")
ax.legend()

plt.show()
