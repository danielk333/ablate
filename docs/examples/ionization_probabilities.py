"""
Estimate ionization probability
===============================
"""

import matplotlib.pyplot as plt
import numpy as np

from metablate import material
import metablate.physics.plasma_production as plasma

velocity = np.linspace(10e3, 70e3, 100)
mat = material.get("cometary", as_dict=False)
comp = {
    "Fe": 1.0,
    "Si": 1.0,
    "Cu": 1.0,
    "Mg": 1.0,
    "O": 1.0,
}
beta, _ = plasma.ionization_probability_Jones_1997(velocity, comp)
beta_comet, total_comet = plasma.ionization_probability_Jones_1997(velocity, mat.atomic_composition)
beta_comet_multi, total_comet_multi = plasma.multi_collision_ionization_probability_Jones_1997(
    velocity,
    mat.atomic_composition,
)

fig, ax = plt.subplots(figsize=(8, 6))
for index, key in enumerate(comp):
    (line,) = ax.semilogy(velocity * 1e-3, beta[index, :], label=key, ls="--")

    loc = 40 + 4 * index  # random formula for where to place the arrow
    sgny = -1 if index >= 2 else 1
    sgnx = -1 if 3 >= index >= 2 else 1
    ax.annotate(
        line.get_label(),
        xy=(velocity[loc] * 1e-3, beta[index, loc]),
        xytext=(sgnx * 25, sgny * 50),
        textcoords="offset points",
        color=line.get_color(),
        va="center",
        arrowprops={
            "arrowstyle": "->",
            "color": line.get_color(),
        },
    )
ax.set(
    ylabel="Ionization probability",
    xlabel="Velocity [km/s]",
    ylim=(1e-4, None),
    xlim=(10, 70),
)
ax.legend()

fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharex=True, sharey=True)
for index, key in enumerate(mat.atomic_composition):
    (line1,) = axes[0].loglog(
        velocity * 1e-3,
        beta_comet[index, :],
        label=key,
        ls="--",
    )
    (line2,) = axes[1].loglog(
        velocity * 1e-3,
        beta_comet_multi[index, :],
        label=key,
        ls="--",
        c=line1.get_color(),
    )

    loc = 40 + 4 * index  # random formula for where to place the arrow
    sgny = -1 if key != "Fe" else 1
    sgnx = -1 if index == 2 else 1
    axes[0].annotate(
        line1.get_label(),
        xy=(velocity[loc] * 1e-3, beta_comet[index, loc]),
        xytext=(sgnx * 15, sgny * 30),
        textcoords="offset points",
        color=line1.get_color(),
        va="center",
        arrowprops={
            "arrowstyle": "->",
            "color": line1.get_color(),
        },
    )
    axes[1].annotate(
        line2.get_label(),
        xy=(velocity[loc] * 1e-3, beta_comet_multi[index, loc]),
        xytext=(sgnx * 15, sgny * 30),
        textcoords="offset points",
        color=line2.get_color(),
        va="center",
        arrowprops={
            "arrowstyle": "->",
            "color": line2.get_color(),
        },
    )
axes[0].loglog(velocity * 1e-3, total_comet, label="Total cometary composition")
axes[1].loglog(velocity * 1e-3, total_comet_multi, label="Total cometary composition")
axes[0].set(
    ylabel="Ionization probability",
    xlabel="Velocity [km/s]",
    title="Jones 1997 (beta_0)",
    ylim=(1e-4, None),
    xlim=(10, 70),
)
axes[1].set(
    xlabel="Velocity [km/s]",
    title="Jones 1997 Figure 3 (beta)",
)
ax.legend()

plt.show()
