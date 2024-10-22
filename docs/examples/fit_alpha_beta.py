"""
Alpha-beta model usage
=======================

"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

from ablate.physics import alpha_beta
import ablate

HERE = Path(__file__).parent.absolute()
data_file = HERE / "data" / "20140104-025436.json"

h0 = 7160.0

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
    "slope": json_data.pop("slope"),
}

not_nans = np.logical_not(
    np.logical_or.reduce(
        [
            np.isnan(data["alt"]),
            np.isnan(data["vel"]),
        ]
    )
)
data["alt"] = data["alt"][not_nans]
data["vel"] = data["vel"][not_nans]
data["vel_std"] = data["vel_std"][not_nans]
data["alt_std"] = data["alt_std"][not_nans]
data["alt0"] = data["alt"]

vel_obs0 = np.max(data["vel"])

lat = 34.0 + (51.0 + 14.50 / 60.0) / 60.0
lon = 136.0 + (06.0 + 20.24 / 60.0) / 60.0

atm = ablate.atmosphere.AtmPymsis()
dens = atm.density(
    time=np.datetime64("2018-07-28T00:00:00"),
    lat=np.array([lat]),
    lon=np.array([lon]),
    alt=data["alt0"],
)
atm_density = dens["Total"].values.squeeze()
data["alt"] = ablate.physics.alpha_beta.scale_hight_to_exponential_atm(
    atm_total_mass_density=atm_density, 
    atmospheric_scale_height=7610.0,
    sea_level_rho=1.225,
)

alpha_Q4, beta_Q4 = alpha_beta.solve_alpha_beta_versionQ4(
    data["vel"],
    data["alt"],
    initial_velocity=vel_obs0,
    atmospheric_scale_height=h0,
    bounds=((1e-4, None), (1e-4, None)),
)
alpha_Q5, beta_Q5, vel_Q5 = alpha_beta.solve_alpha_beta_velocity_versionQ5(
    data["vel"],
    data["alt"],
    bounds=((1e-2, 1e6), (1e-6, 1e3), (0, None)),
    atmospheric_scale_height=h0,
)
alpha_P, beta_P, vel_P = alpha_beta.solve_alpha_beta_posterior(
    data["vel"],
    data["vel_std"],
    data["alt"],
    atmospheric_scale_height=h0,
    minimize_kwargs={"method": "Nelder-Mead"},
)
print(f"{alpha_Q4=:.3e} {beta_Q4=:.3e}")
print(f"{alpha_Q5=:.3e} {beta_Q5=:.3e} {vel_Q5=:.3e}")
print(f"{alpha_P=:.3e} {beta_P=:.3e} {vel_P=:.3e}")

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
    label=f"{alpha_Q4=:.2f} {beta_Q4=:.2f} vel0={vel_obs0*1e-3:.1f} km/s",
)
ax.plot(
    norm_v * vel_Q5 * 1e-3,
    heights_Q5 * 1e-3,
    "-r",
    label=f"{alpha_Q5=:.2f} {beta_Q5=:.2f} vel0={vel_Q5*1e-3:.1f} km/s",
)
ax.plot(
    norm_v * vel_P * 1e-3,
    heights_P * 1e-3,
    "-g",
    label=f"{alpha_P=:.2f} {beta_P=:.2f} vel0={vel_P*1e-3:.1f} km/s",
)
ax.set_xlabel("Velocity [km/s]")
ax.set_ylabel("Height [km]")
ax.set_title("alpha-beta fitting method comparison")
ax.legend()

plt.show()
