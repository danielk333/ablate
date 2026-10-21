"""
Compute Radar Cross Section
===========================
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import metablate.atmosphere as atm
import metablate.models.dimant_oppenheim_2017 as do
from metablate import physics

parser = argparse.ArgumentParser()
parser.add_argument(
    "--frequency",
    type=float,
    nargs="+",
    default=[50e6, 220e6, 930e6],
    help="Radar frequencies [Hz].",
)
parser.add_argument("--num", type=int, default=180, help="Number of equal angular wedges.")
parser.add_argument("--r-max", type=float, default=10.0, help="Outer solver bound [lambda_T].")
parser.add_argument("--output-dir", type=Path, help="Save the plot here instead of showing it.")
args = parser.parse_args()

frequencies = np.asarray(args.frequency)

met_vel = 40e3
met_mu = 3.8e-26
ablated_thermal_vel = physics.thermal_ablation.ablated_thermal_speed_sugar_2018(
    meteoroid_surface_temperature=2000, meteoroid_molecular_mass=met_mu
)
met_r = 5e-5
ma = 4.7e-26
n0 = 1e13 / met_r**2
rel_vel = met_vel - ablated_thermal_vel
sigma = physics.plasma_production.collisional_cross_section_bronshten_1983(rel_vel)
beta = physics.plasma_production.ionization_probability_Na_vondrak_2008(rel_vel)

atm_data = atm.AtmPymsis().density(
    time=np.datetime64("2018-07-28"),
    lat=np.array([69.0]),
    lon=np.array([12.0]),
    alt=np.array([100e3]),
    mass_densities=False,
)
na = float(atm_data["Total"].values.item()) / ma
lam = float(
    physics.plasma_production.mean_free_path_dimant_oppenheim_2017(ablated_thermal_vel, na, sigma, met_vel)
)
# n_e = n_star * f / R_lambda; evaluating at f=R_lambda=1 gives n_star.
n_star = do.plasma_distribution(
    total_atmospheric_number_density=na,
    meteoroid_velocity=met_vel,
    meteoroid_radius=met_r,
    plasma_source_density=n0,
    collisional_cross_section=sigma,
    ionization_probability=beta,
    ablated_thermal_speed=ablated_thermal_vel,
    atmospheric_species_mass=ma,
    ablated_species_mass=met_mu,
    base_plasma_distribution_function=1.0,
    R_lambda=1.0,
)
critical_density = np.asarray(physics.plasma_production.critical_plasma_density(frequencies))

delta_theta = 2 * np.pi / args.num
theta = (np.arange(args.num) + 0.5) * delta_theta
radius = np.empty((frequencies.size, theta.size))
for frequency_index, density in enumerate(critical_density):
    for angle_index, angle in tqdm(enumerate(theta), total=len(theta)):
        # The contour must lie between the body surface and the outer bound.
        r_lambda, _, _ = do.solve_r(
            theta=angle,
            target_density=density / n_star,
            r_min=met_r / lam,
            r_max=args.r_max,
            xtol=1e-8,
        )
        radius[frequency_index, angle_index] = r_lambda * lam

# Each midpoint sample represents a sector of area r^2 * delta_theta / 2.
rcs = 0.5 * delta_theta * np.sum(radius**2, axis=1)
for frequency, density, area in zip(frequencies, critical_density, rcs):
    print(f"{frequency / 1e6:g} MHz: n_e = {density:.6e} m^-3, RCS = {area:.6e} m^2")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
for frequency, contour_radius in zip(frequencies, radius):
    x = contour_radius * np.cos(theta)
    y = contour_radius * np.sin(theta)
    axes[0].plot(np.append(x, x[0]), np.append(y, y[0]), label=f"{frequency / 1e6:g} MHz")
axes[0].set(xlabel="x [m]", ylabel="y [m]", title="Critical-density contours", aspect="equal")
axes[0].legend()

order = np.argsort(frequencies)
axes[1].loglog(frequencies[order] / 1e6, rcs[order], "o-")
axes[1].set(xlabel="Radar frequency [MHz]", ylabel="Geometric RCS approximation [m^2]")

if args.output_dir is None:
    plt.show()
else:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "rcs.png", dpi=200)
    plt.close(fig)
