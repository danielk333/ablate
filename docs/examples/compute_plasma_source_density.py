import numpy as np
import matplotlib.pyplot as plt

import metablate.models.alpha_beta_2026 as ab
import metablate.models.dimant_oppenheim_2017 as do
from metablate import physics
from metablate.material import asteroidal as material

model = ab.AlphaBeta2026(
    options=ab.AlphaBetaOptions(
        max_step_size=1e-2,
    ),
)

# assume some parameters
radius = 2e-3
ze = 75.7744
cross_section = np.pi * radius**2
mass = material.bulk_density * 4 / 3 * np.pi * radius**3

alpha = physics.alpha_beta.alpha_direct(
    aerodynamic_cd=0.47,
    sea_level_rho=1.225,
    atmospheric_scale_height=7610.0,  # at T = 260 K
    initial_cross_section=cross_section,
    initial_mass=mass,
    radiant_local_elevation=90 - ze,
    degrees=True,
)

# run time dependant model
p = ab.AlphaBetaInitialState(
    epoch=None,
    alpha=100.0,
    beta=10.0,
    initial_height=150e3,
    entry_velocity=30e3,
    entry_angle=np.radians(70.0),
    shape_change_coefficient=2 / 3,
)
result = model.run(p)

temperatures = np.empty_like(result.massloss)
for ind in range(len(temperatures)):
    T = physics.thermal_ablation.solve_temperature_from_thermal_ablation(
        result.massloss[ind] * mass,
        result.relative_mass[ind] * mass,
        material,
        shape_factor=1.0,  # assume sphere?
    )
    temperatures[ind] = T

Q = -result.massloss * mass / material.mean_atomic_mass

V_T = do.ablated_thermal_speed_bronshten_1983(
    meteoroid_surface_temperature=temperatures,
    meteoroid_molecular_mass=material.mean_atomic_mass,
)
n_0 = Q / (4 * np.pi * radius**2 * V_T) * np.sqrt(np.pi / 2)

fig, axes = plt.subplots(1, 3)
axes[0].plot(result.t, result.massloss * mass)
axes[1].plot(result.t, temperatures)
axes[2].plot(result.t, n_0)
plt.show()
