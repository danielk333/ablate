"""
Alpha-beta model
================
"""

import numpy as np
import matplotlib.pyplot as plt

import metablate.models.alpha_beta_2026 as ab


model = ab.AlphaBeta2026(
    options=ab.AlphaBetaOptions(
        max_step_size=1e-1,
    ),
)

p = ab.AlphaBetaInitialState(
    epoch=None,
    alpha=100.0,
    beta=10.0,
    initial_height=150e3,
    entry_velocity=30e3,
    entry_angle=np.radians(45.0),
    shape_change_coefficient=2 / 3,
)
result = model.run(p)
print(f"{result.runtime=} s")

fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="tight")
fig.suptitle(
    f"alpha={p.alpha}, beta={p.beta}, entry velocity={p.entry_velocity * 1e-3}, "
    f"entry angle={np.degrees(p.entry_angle)} deg"
)
axes[0, 0].plot(result.t, result.height * 1e-3)
axes[0, 0].set_ylabel("Height [km]")
axes[0, 0].set_xlabel("Time [s]")

axes[0, 1].plot(result.t, result.velocity * 1e-3)
axes[0, 1].set_ylabel("Velocity [km/s]")
axes[0, 1].set_xlabel("Time [s]")

axes[1, 0].plot(result.t, result.relative_mass)
axes[1, 0].set_ylabel("Relative mass [1]")
axes[1, 0].set_xlabel("Time [s]")

axes[1, 1].plot(result.velocity * 1e-3, result.height * 1e-3)
axes[1, 1].set_ylabel("Height [km]")
axes[1, 1].set_xlabel("Velocity [km/s]")

axes[0, 2].plot(result.massloss, result.height * 1e-3)
axes[0, 2].set_ylabel("Height [km]")
axes[0, 2].set_xlabel("Relative massloss [1/s]")

axes[1, 2].plot(result.t, result.distance * 1e-3)
axes[1, 2].set_ylabel("Distance [km]")
axes[1, 2].set_xlabel("Time [s]")

plt.show()
