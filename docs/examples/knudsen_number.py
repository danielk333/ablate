"""
Estimate Knudsen number
=============================================

Docstring for this example
"""

import numpy as np
import matplotlib.pyplot as plt

import metablate.physics as phys
import metablate

# data for 2 different meteoroids
model = metablate.atmosphere.AtmPymsis()
data = model.density(
    time=np.datetime64("2018-07-28"),
    lat=np.array([69.0]),
    lon=np.array([12.0]),
    alt=np.array([80e3]),
)
atm_total_mass_density = data["Total"].values.squeeze()
atm_total_number_density = atm_total_mass_density / model.mean_mass

print(data)

material_data = metablate.material.get("asteroidal")

log_masses = np.linspace(-9, -2, 200)
masses = 10.0**log_masses

Kn_inf, L = phys.thermal_ablation.Knudsen_number_kero_szasz_2008(
    masses, material_data["rho_m"], atm_total_number_density
)

fig, ax = plt.subplots(figsize=(7, 4))

ax.loglog(masses, Kn_inf, c="k")
ax.axhline(0.01, c="r", label="Continuum flow")
ax.axhline(10.0, c="g", label="Free molecular flow")
ax.set_xlabel("Mass [kg]")
ax.set_ylabel("Knudsen number")
ax.legend()

plt.show()
