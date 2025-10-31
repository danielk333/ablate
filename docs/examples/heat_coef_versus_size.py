"""
Estimate drag coefficients and heat transfer
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
velocity = 32.0

temperature = 3700


dmdt_th = phys.thermal_ablation.thermal_ablation_hill_et_al_2005(
    mass=masses,
    temperature=temperature,
    material_data=material_data,
    shape_factor=1.21,
)

Kn_inf, L = phys.thermal_ablation.Knudsen_number_kero_szasz_2008(
    masses, material_data["rho_m"], atm_total_number_density
)

# TODO: debug why lambda calculation does not work properly
Lambda = phys.thermal_ablation.heat_transfer_bronshten_1983(
    mass=masses,
    velocity=velocity,
    temperature=temperature,
    material_data=material_data,
    atm_total_number_density=atm_total_number_density,
    mass_loss_thermal_ablation=-dmdt_th,
    atm_mean_mass=model.mean_mass,
    res=100,
)

fig, axes = plt.subplots(3, 1)
axes[0].loglog(masses, Lambda)
axes[0].set_xlabel("Mass [kg]")
axes[0].set_ylabel("Heat transfer coef")
axes[0].set_title(f"{velocity=} km/s")

axes[1].loglog(masses, Kn_inf, c="k")
axes[1].axhline(0.01, c="r", label="Continuum flow")
axes[1].axhline(10.0, c="g", label="Free molecular flow")
axes[1].set_xlabel("Mass [kg]")
axes[1].set_ylabel("Knudsen number")
axes[1].legend()

axes[2].loglog(masses, -dmdt_th/masses)
axes[2].set_xlabel("Mass [kg]")
axes[2].set_ylabel("Fractional mass loss [1/s]")

plt.show()
