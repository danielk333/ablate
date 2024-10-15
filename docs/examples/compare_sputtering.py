"""
Compute sputtering and thermal ablation
========================================

Docstring for this example
"""

import numpy as np

import ablate.physics as phys
import ablate

# data for 2 different meteoroids
model = ablate.atmosphere.AtmPymsis()
data = model.density(
    time=np.datetime64("2018-07-28T00:00:00"),
    lat=np.array([69.0]),
    lon=np.array([12.0]),
    alt=np.array([90e3, 120e3]),
    mass_densities=False,
)

print(data)

material_data = ablate.material.get("asteroidal")

dmdt = phys.sputtering.sputtering_kero_szasz_2008(
    mass=np.array([0.5, 0.2]),
    velocity=np.array([40e3, 40e3]),
    material_data=material_data,
    density=data,
)

dmdt_th = phys.thermal_ablation.thermal_ablation_hill_et_al_2005(
    mass=np.array([0.5, 0.2]),
    temperature=3700,
    material_data=material_data,
    shape_factor=1.21,
)


print("thermal dm/dt:")
print(dmdt_th)

print("sputtering dm/dt:")
print(dmdt)
