"""
Alpha-beta model
================
"""

import numpy as np
import matplotlib.pyplot as plt

import metablate.models.alpha_beta_2026 as ab

ze = 75.7744
v0 = 53e3
alpha = 101.0
beta = 10.0
init_alt = 130e3



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
