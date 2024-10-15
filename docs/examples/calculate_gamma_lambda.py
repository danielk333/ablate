"""
Estimate drag coefficients and heat transfer
=============================================

Docstring for this example
"""

import numpy as np

import ablate.physics as phys
import ablate

# data for 2 different meteoroids
model = ablate.atmosphere.AtmPymsis()
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

material_data = ablate.material.get("asteroidal")

dmdt_th = phys.thermal_ablation.thermal_ablation_hill_et_al_2005(
    mass=mass,
    temperature=temperature,
    material_data=material_data,
    shape_factor=1.21,
)

print(f"{dmdt_th=})")

atm_total_mass_density = data["Total"].values.squeeze()
atm_total_number_density = atm_total_mass_density / model.mean_mass

print(f"{atm_total_number_density=}")

Gamma = phys.thermal_ablation.drag_coefficient_bronshten_1983(
    mass=mass,
    velocity=velocity,
    temperature=temperature,
    material_data=material_data,
    atm_total_number_density=atm_total_number_density,
    atm_mean_mass=model.mean_mass,
    res=100,
)

Lambda = phys.thermal_ablation.heat_transfer_bronshten_1983(
    mass=mass,
    velocity=velocity,
    temperature=temperature,
    material_data=material_data,
    atm_total_number_density=atm_total_number_density,
    mass_loss_thermal_ablation=-dmdt_th,
    atm_mean_mass=model.mean_mass,
    res=100,
)

print("Alt:")
print([100e3, 120e3])

print("Gamma:")
print(Gamma)

print("Lambda:")
print(Lambda)
