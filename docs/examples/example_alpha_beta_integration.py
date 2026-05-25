"""
Alpha-beta model
================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl

import metablate.models.alpha_beta_2026 as ab
from metablate import physics
from spacecoords import frames

model = ab.AlphaBeta2026(
    options=ab.AlphaBetaOptions(
        max_step_size=1e-2,
    ),
)

p = ab.AlphaBetaInitialState(
    epoch=None,
    alpha=100.0,
    beta=10.0,
    initial_height=150e3,
    entry_velocity=30e3,
    entry_angle=np.radians(70.0),
    shape_change_coefficient=2 / 3,
)
result = model.run(p)
print(f"{result.runtime=} s")

norm_vel_lims = physics.alpha_beta.norm_velocity_direct(
    np.array([0.99, 0.01]),
    p.beta,
    p.shape_change_coefficient,
)
velocities_vec = np.linspace(norm_vel_lims[0], norm_vel_lims[1], 20)
masses = physics.alpha_beta.norm_mass_direct(
    velocity=velocities_vec,
    beta=p.beta,
    shape_change_coefficient=p.shape_change_coefficient,
)
heights = physics.alpha_beta.norm_height_direct(
    velocity=velocities_vec,
    alpha=p.alpha,
    beta=p.beta,
)
heights *= model.options.atmospheric_scale_height

fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="tight")
fig.suptitle(
    f"alpha={p.alpha}, beta={p.beta}, entry velocity={p.entry_velocity * 1e-3}, "
    f"entry angle={np.degrees(p.entry_angle)} deg"
)
axes[0, 0].plot(result.t, result.height * 1e-3)
axes[0, 0].set_ylabel("Height [km]")
axes[0, 0].set_xlabel("Time [s]")

axes[0, 1].plot(result.t, result.velocity * 1e-3)
# axes[0, 1].plot(result.t[1:], np.diff(result.distance) / np.diff(result.t) * 1e-3)
axes[0, 1].set_ylabel("Velocity [km/s]")
axes[0, 1].set_xlabel("Time [s]")

axes[1, 0].plot(result.relative_mass, result.height * 1e-3)
axes[1, 0].plot(masses, heights * 1e-3, "xr")
axes[1, 0].set_ylabel("Height [km]")
axes[1, 0].set_xlabel("Relative mass [1]")

axes[1, 1].plot(
    result.velocity * 1e-3,
    result.height * 1e-3,
    label="Time dependant solution",
)
axes[1, 1].plot(
    velocities_vec * p.entry_velocity * 1e-3,
    heights * 1e-3,
    "xr",
    label="Phase-space criterion",
)
axes[1, 1].set_ylabel("Height [km]")
axes[1, 1].set_xlabel("Velocity [km/s]")
axes[1, 1].legend()

axes[0, 2].plot(result.massloss, result.height * 1e-3)
axes[0, 2].set_ylabel("Height [km]")
axes[0, 2].set_xlabel("Relative massloss [1/s]")

axes[1, 2].plot(result.t, result.distance * 1e-3)
axes[1, 2].set_ylabel("Distance [km]")
axes[1, 2].set_xlabel("Time [s]")

# We can also easily turn this into a ECEF trajectory with the "distance" parameter

lat = 67 + 50 / 60 + 26.6 / 3600
lon = 20 + 24 / 60 + 40.0 / 3600
origin_ecef = frames.geodetic_wgs84_to_ecef(
    lat,
    lon,
    0,
    degrees=True,
)
reference_pos_ecef = frames.geodetic_wgs84_to_ecef(
    lat,
    lon,
    p.initial_height,
    degrees=True,
)
velocity_dir_ecef = frames.azel_to_ecef(
    lat,
    lon,
    az=10,
    el=-np.degrees(p.entry_angle),
    degrees=True,
)

position_ecef = reference_pos_ecef[:, None] + velocity_dir_ecef[:, None] * result.distance[None, :]
position = frames.ecef_to_enu(lat, lon, position_ecef - origin_ecef[:, None], degrees=True)

segments = np.stack([position.T[:-1], position.T[1:]], axis=1) * 1e-3
c = -result.massloss
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = plt.get_cmap("jet")
colors = cmap(norm(c[:-1]))
lc = Line3DCollection(segments, colors=colors, linewidth=2)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection="3d")
ax.add_collection3d(lc)
ax.plot([0], [0], [0], "or")

mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
plt.colorbar(mappable, ax=ax, label="Relative massloss [1/s]")

plt.show()
