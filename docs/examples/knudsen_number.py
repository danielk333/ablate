"""
Estimate Knudsen number
=============================================

Docstring for this example
"""

import numpy as np
import matplotlib.pyplot as plt

import metablate.physics as phys
import metablate

alts = np.array([100e3, 80e3, 60e3])

model = metablate.atmosphere.AtmPymsis()
data = model.density(
    time=np.datetime64("2018-07-28"),
    lat=np.array([69.0]),
    lon=np.array([12.0]),
    alt=alts,
)
atm_total_mass_density = data["Total"].values.flatten()
atm_total_number_density = np.zeros_like(atm_total_mass_density)
for symbol in model.species:
    s_dens = data[symbol].values.flatten()
    inds = np.logical_not(np.isnan(s_dens))
    atm_total_number_density[inds] = atm_total_number_density[inds] + s_dens[inds]

material_data = metablate.material.asteroidal

log_masses = np.linspace(-9, 4, 200)
masses = 10.0**log_masses

fig, ax = plt.subplots(figsize=(7, 4), layout="tight")

for ind in range(len(alts)):
    Kn_inf, L = phys.thermal_ablation.Knudsen_number_kero_szasz_2008(
        masses,
        material_data.bulk_density,
        atm_total_number_density[ind],
    )
    ax.loglog(masses, Kn_inf, label=f"Meteoroid at {alts[ind] * 1e-3:.0f} km altitude")
ax.axhline(0.01, c="r", ls="--", label="Continuum flow")
ax.axhline(10.0, c="g", ls="--", label="Free molecular flow")
ax.set_xlabel("Mass [kg]")
ax.set_ylabel("Knudsen number")
ax.legend()

plt.show()
