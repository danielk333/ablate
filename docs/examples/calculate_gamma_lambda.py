"""
Estimate drag coefficients and heat transfer
=============================================

Docstring for this example
"""

import numpy as np
import scipy.constants as constants

import ablate.functions as func
import ablate.atmosphere as atm

# data for 2 different meteoroids
model = atm.NRLMSISE00()
data = model.density(
    time=np.datetime64("2018-07-28"),
    lat=np.array([69.0, 69.0]),
    lon=np.array([12.0, 12.0]),
    alt=np.array([90e3, 120e3]),
)

print(data)
mass = 0.5e-6
velocity = 32e3
temperature = 3700

material_data = func.material.material_parameters("ast")

dmdt_th = func.ablation.thermal_ablation(
    mass=mass,
    temperature=temperature,
    material_data=material_data,
    shape_factor=1.21,
)

print(f"{dmdt_th=})")

atm_mean_mass = (
    np.array([x["A"] for _, x in model.species.items()]).mean() * constants.u
)  # [kg]
atm_total_density = data["Total"].values.squeeze()
N_rho_tot = atm_total_density / atm_mean_mass

print(f"{N_rho_tot=}")

Gamma = func.dynamics.drag_coefficient(
    mass=mass,
    velocity=velocity,
    temperature=temperature,
    material_data=material_data,
    atm_total_density=N_rho_tot,
    atm_mean_mass=atm_mean_mass,
    res=100,
)

Lambda = func.dynamics.heat_transfer(
    mass=mass,
    velocity=velocity,
    temperature=temperature,
    material_data=material_data,
    atm_total_density=N_rho_tot,
    thermal_ablation=dmdt_th,
    atm_mean_mass=atm_mean_mass,
    res=100,
)

print("Alt:")
print([100e3, 120e3])

print("Gamma:")
print(Gamma)

print("Lambda:")
print(Lambda)
