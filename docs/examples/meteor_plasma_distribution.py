import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.constants as consts

import metablate.atmosphere as atm
import metablate.models.dimant_oppenheim_2017 as do

parser = argparse.ArgumentParser()
parser.add_argument("--cache", default=None)
parser.add_argument(
    "--scale",
    choices=[0.0005, 0.005, 0.05, 0.5],
    type=float,
    default=0.05,
)
args = parser.parse_args()

met_vel = 40e3
met_mu = 3.8e-26
ablated_thermal_vel = do.ablated_thermal_speed_bronshten_1983(
    meteoroid_surface_temperature=2000, meteoroid_molecular_mass=met_mu
)
met_r = 5e-5
ma = 4.7e-26
n0 = 10**13 / met_r**2
rel_vel = met_vel - ablated_thermal_vel
sigma = do.collisional_cross_section_bronshten_1983(rel_vel)
beta = do.ionization_probability_Na_vondrak_2008(rel_vel)
radar_freqs = np.array([50e6, 220e6, 500e6, 930e6, 1300e6])
radar_critical_dens = do.critical_plasma_density(radar_freqs)

for f, ne in zip(radar_freqs, radar_critical_dens):
    print(f"{f=:.2e} -> {ne=:.2e}")

model = atm.AtmPymsis()

atm_data = model.density(
    time=np.datetime64("2018-07-28"),
    lat=np.array([69.0]),
    lon=np.array([12.0]),
    alt=np.array([100e3]),
    mass_densities=False,
)
index = dict(lat=0, lon=0, alt=0, time=0)
na = atm_data["Total"][index].values / ma

lam = do.mean_free_path(
    ablated_thermal_vel,
    na,
    sigma,
    met_vel,
)

if args.cache is not None:
    cache = pathlib.Path(args.cache)
    if cache.exists():
        _dat = np.load(cache)
        xlam = _dat["x"]
        ylam = _dat["y"]
        rlam = _dat["r"]
        do_f_dist = _dat["f"]
    else:
        xlam, ylam, rlam, do_f_dist = do.plasma_distribution_morphology(
            x_grid=(np.arange(201) - 60) * args.scale,
            y_grid=(np.arange(201) - 100) * args.scale,
            threads=8,
        )
        np.savez(cache, x=xlam, y=ylam, r=rlam, f=do_f_dist)
else:
    xlam, ylam, rlam, do_f_dist = do.plasma_distribution_morphology(
        x_grid=(np.arange(201) - 60) * args.scale,
        y_grid=(np.arange(201) - 100) * args.scale,
        threads=8,
    )

x = xlam * lam
y = ylam * lam
ne_dist = do.plasma_distribution(
    total_atmospheric_number_density=na,
    meteoroid_velocity=met_vel,
    meteoroid_radius=met_r,
    plasma_source_density=n0,
    collisional_cross_section=sigma,
    ionization_probability=beta,
    ablated_thermal_speed=ablated_thermal_vel,
    atmospheric_species_mass=ma,
    ablated_species_mass=met_mu,
    base_plasma_distribution_function=do_f_dist,
    R_lambda=rlam,
)
f_p = do.plasma_frequency(ne_dist)


def plasma_freq_to_area(dx, dy, f_p, f_p_limit):
    area_unit = dx * dy
    return np.sum(f_p > f_p_limit) * area_unit


fig, ax = plt.subplots()
cmap = "jet"
im = ax.pcolormesh(xlam, ylam, ne_dist, cmap=cmap, norm=LogNorm())
fig.colorbar(im, ax=ax)
ax.set_xlabel("x [lambda_T]")
ax.set_ylabel("y [lambda_T]")


fig, ax = plt.subplots()
cmap = "jet"
im = ax.pcolormesh(x, y, f_p, cmap=cmap, norm=LogNorm())
cs = ax.contour(x, y, f_p, levels=radar_freqs, colors="black")
ax.clabel(cs, cs.levels, fmt="%.2e")
fig.colorbar(im, ax=ax)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

f = np.linspace(10e6, 1e9, 100)
area = np.empty_like(f)
for ind in range(len(f)):
    area[ind] = plasma_freq_to_area(args.scale, args.scale, f_p, f[ind])

fig, ax = plt.subplots()
ax.loglog(f, area)
ax.set_xlabel("Plasma frequency Fp [Hz]")
ax.set_ylabel("xy-area with > Fp frequency [m^2]")


plt.show()
