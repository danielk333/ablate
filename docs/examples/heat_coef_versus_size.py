"""
Estimate drag coefficients and heat transfer
=============================================

Docstring for this example
"""

import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt

import ablate.functions as func
import ablate.atmosphere as atm

# data for 2 different meteoroids
model = atm.NRLMSISE00()
data = model.density(
    time=np.datetime64("2018-07-28"),
    lat=np.array([69.0]),
    lon=np.array([12.0]),
    alt=np.array([80e3]),
)
atm_total_density = data["Total"].values.squeeze()

print(data)

material_data = func.material.material_parameters("ast")

atm_mean_mass = (
    np.array([x["A"] for _, x in model.species.items()]).mean() * constants.u
)  # [kg]

log_masses = np.linspace(-9, -2, 200)
masses = 10.0**log_masses
velocity = 32.0

temperature = 3700


dmdt_th = func.ablation.thermal_ablation(
    mass=masses,
    temperature=temperature,
    material_data=material_data,
    shape_factor=1.21,
)


N_rho_tot = atm_total_density / atm_mean_mass
Kn_inf, L = func.dynamics.Knudsen_number(masses, material_data["rho_m"], N_rho_tot)

Lambda = func.dynamics.heat_transfer(
    mass=masses,
    velocity=velocity,
    temperature=temperature,
    material_data=material_data,
    atm_total_density=N_rho_tot,
    thermal_ablation=dmdt_th,
    atm_mean_mass=atm_mean_mass,
    res=100,
)

fig, axes = plt.subplots(2, 1)
axes[0].loglog(masses, Lambda)
axes[0].set_xlabel("Mass [kg]")
axes[0].set_ylabel("Heat transfer coefficient")
axes[0].set_title(f"{velocity=} km/s")

axes[1].loglog(masses, Kn_inf, c="k")
axes[1].axhline(0.01, c="r", label="Continuum flow")
axes[1].axhline(10.0, c="g", label="Free molecular flow")
axes[1].set_xlabel("Mass [kg]")
axes[1].set_ylabel("Knudsen number")
axes[1].legend()

plt.show()
