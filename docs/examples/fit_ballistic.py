"""
Ballistic model usage
=======================

"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

from ablate.physics import ballistic
import ablate

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


atm = ablate.atmosphere.AtmPymsis()
msis_density = atm.density(
    time=np.datetime64("2018-06-28T12:45:33"),
    lat=69.5866115,
    lon=19.221555,
    alt=data["alt"],
)["Total"].values.squeeze()

cB_est, v0 = ballistic.fit_velocity(
    data["alt"],
    data["vel"],
    msis_density,
    weights=data["vel_std"],
    radiant_local_elevation=90 - meta["slope"],
    degrees=True,
)
inds = np.argsort(data["alt"])[::-1]
fit_vels = ballistic.velocity(
    data["alt"][inds],
    msis_density[inds],
    cB_est,
    radiant_local_elevation=90 - meta["slope"],
    initial_velocity=v0,
    degrees=True,
)

print(f"{cB_est=} {v0*1e-3} km/s")

fig, ax = plt.subplots()

ax.set_title("Ballistic coefficient fit")
ax.plot(data["vel"] * 1e-3, data["alt"] * 1e-3, ".b", label="Data")
ax.plot(fit_vels * 1e-3, data["alt"][inds] * 1e-3, "-r", label="Fit")
ax.set_xlabel("Velocity [km/s]")
ax.set_ylabel("Height [km]")
ax.legend()

plt.show()
