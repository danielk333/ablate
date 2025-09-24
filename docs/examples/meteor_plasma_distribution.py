import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import ablate.atmosphere as atm
import ablate.models.dimant_oppenheim_2017 as do


met_vel = 40e3
ablated_thermal_vel = 951
met_r = 5e-5
ma = 4.7e-26
n0 = 10**13 / met_r**2
rel_vel = met_vel - ablated_thermal_vel
sigma = do.collisional_cross_section_bronshten_1983(rel_vel)
beta = do.ionization_probability_Na_vondrak_2008(rel_vel)

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
# TODO: pre-compute the F functions that are independant on meteoroid parameters and create a high
# resolution version that can be scaled quickly
if False:
    x, y, ne_dist = do.plasma_distribution(
        total_atmospheric_number_density=na,
        meteoroid_velocity=met_vel,
        meteoroid_radius=met_r,
        plasma_source_density=n0,
        collisional_cross_section=sigma,
        ionization_probability=beta,
        ablated_thermal_speed=ablated_thermal_vel,
        atmospheric_species_mass=ma,
        ablated_species_mass=3.8e-26,
        # x_grid=(np.arange(301) - 100)*0.0003,
        # y_grid=(np.arange(201) - 100)*0.0003,
        x_grid=(np.arange(401) - 100) * 0.005,
        y_grid=(np.arange(201) - 100) * 0.005,
        threads=6,
    )
    np.save("/home/danielk/data/ne_dist_test.npy", ne_dist)
    np.save("/home/danielk/data/x_test.npy", x)
    np.save("/home/danielk/data/y_test.npy", y)
ne_dist = np.load("/home/danielk/data/ne_dist_test.npy")
x = np.load("/home/danielk/data/x_test.npy")
y = np.load("/home/danielk/data/y_test.npy")
f_p = do.plasma_frequency(ne_dist)


def plasma_freq_to_area(dx, dy, f_p, f_p_limit):
    area_unit = dx * dy
    return np.sum(f_p > f_p_limit) * area_unit


fig, ax = plt.subplots()
cmap = "jet"
im = ax.pcolormesh(x / lam, y / lam, ne_dist, cmap=cmap, norm=LogNorm())
fig.colorbar(im, ax=ax)
ax.set_xlabel("x [lambda_T]")
ax.set_ylabel("y [lambda_T]")


fig, ax = plt.subplots()
cmap = "jet"
im = ax.pcolormesh(x, y, f_p, cmap=cmap, norm=LogNorm())
cs = ax.contour(x, y, f_p, levels=[50e6, 220e6, 500e6, 930e6, 1300e6], colors="black")
ax.clabel(cs, cs.levels, fmt="%.2e")
fig.colorbar(im, ax=ax)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

f = np.linspace(10e6, 1e9, 100)
area = np.empty_like(f)
for ind in range(len(f)):
    area[ind] = plasma_freq_to_area(0.01, 0.01, f_p, f[ind])

fig, ax = plt.subplots()
ax.loglog(f, area)


plt.show()
