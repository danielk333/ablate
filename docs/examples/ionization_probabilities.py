"""
Estimate ionization probability
===============================
"""

import matplotlib.pyplot as plt
import numpy as np

from metablate import material, physics

velocity = np.linspace(10e3, 70e3, 100)
mat = material.get("asteroidal", as_dict=False)
comp = {
    "Fe": 1.0,
    "Si": 1.0,
    "Cu": 1.0,
    "Mg": 1.0,
    "O": 1.0,
}
beta = physics.plasma_production.ionization_probability_Jones_1997(velocity, comp)
beta_comet = physics.plasma_production.ionization_probability_Jones_1997(velocity, mat.atomic_composition)

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
ax.plot(velocity * 1e-3, np.sum(beta_comet, axis=0), label="Total cometary composition")
ax.set(
    ylabel="Ionization probability",
    xlabel="Velocity [km/s]",
    ylim=(1e-4, None),
    xlim=(10, 70),
)
ax.legend()

plt.show()
