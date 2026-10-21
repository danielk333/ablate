"""Estimate Dimant source density from an alpha-beta mass-loss history.

Alpha-beta predicts mass loss, not a surface temperature. Here we close that
missing relation by assuming a uniformly evaporating sphere and interpreting
all mass loss as neutral vapor of the material's mean vapor molecular mass.
By default, invert the Hill et al. (2005) vapor-pressure evaporation law to find
an effective emission temperature.

For Dimant & Oppenheim (2017), Part 1, Eq. 18 and the following surface flux,
Q = 4*pi*r(t)**2 * sqrt(2/pi) * n_0 * V_T, with V_T = sqrt(k_B*T/m_vapor).
Thus n_0 is the outward neutral density at the surface, not electron density
or the full equilibrium vapor density. Use the same vapor mass and V_T when
passing this source to the Dimant plasma model.

There are some assumptions in this derivation: constant bulk density, spherical
shrinkage (mu=2/3), evaporation coefficient one and negligible return flux.
Fragmentation or sputtering mass loss must not be interpreted as thermal vapor.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import metablate.models.alpha_beta_2026 as ab
from metablate import physics
from metablate.material import asteroidal as material

parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, help="Prescribe emission temperature [K].")
parser.add_argument("--output-dir", type=Path, help="Save the plot here instead of showing it.")
args = parser.parse_args()

initial_radius = 2e-3
initial_mass = physics.alpha_beta.radius_to_mass(initial_radius, material.bulk_density)
entry_angle = np.radians(70.0)
scale_height = 7610.0
alpha = physics.alpha_beta.alpha_direct(
    aerodynamic_cd=0.47,
    sea_level_rho=1.225,
    atmospheric_scale_height=scale_height,
    initial_cross_section=np.pi * initial_radius**2,
    initial_mass=initial_mass,
    radiant_local_elevation=entry_angle,
)
model = ab.AlphaBeta2026(
    options=ab.AlphaBetaOptions(max_step_size=1e-2, atmospheric_scale_height=scale_height)
)
parameters = ab.AlphaBetaInitialState(
    epoch=None,
    alpha=alpha,
    beta=10.0,
    initial_height=150e3,
    entry_velocity=30e3,
    entry_angle=entry_angle,
    shape_change_coefficient=2 / 3,  # Uniformly shrinking sphere at fixed density.
)
result = model.run(parameters)
mass = result.relative_mass * initial_mass
massloss = result.massloss * initial_mass  # dM/dt, negative for mass loss [kg/s].
temperature, thermal_speed, source_density = physics.plasma_production.source_from_massloss(
    mass, massloss, material, temperature=args.temperature
)
# Supply source_density as plasma_source_density and thermal_speed as
# ablated_thermal_speed to dimant.plasma_distribution; use the current radius
# from mass_to_radius(mass, material.bulk_density), not initial_radius.

fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
axes[0].semilogy(result.t, -massloss)
axes[0].set(ylabel=r"Vapor mass loss [kg s$^{-1}$]")
axes[1].plot(result.t, temperature)
label = "Effective evaporation temperature" if args.temperature is None else "Prescribed temperature"
axes[1].set(ylabel=f"{label} [K]")
axes[2].semilogy(result.t, source_density)
axes[2].set(ylabel=r"Surface neutral density $n_0$ [m$^{-3}$]")
for ax in axes:
    ax.set_xlabel("Time [s]")
if args.output_dir is None:
    plt.show()
else:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "plasma_source_density.png", dpi=200)
    plt.close(fig)

