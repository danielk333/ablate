import warnings
import argparse
import pathlib
import numpy as np
from scipy import constants
from scipy.integrate import IntegrationWarning
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import matplotlib.animation as animation

import metablate.models.kero_szasz_2008 as ks
import metablate.models.dimant_oppenheim_2017 as do
from spacecoords import frames
import metablate

plt.style.use("dark_background")

parser = argparse.ArgumentParser()
parser.add_argument("--cache", default=None)
parser.add_argument("--resolution", type=int, default=201)
args = parser.parse_args()

lat = 67 + 50 / 60 + 26.6 / 3600
lon = 20 + 24 / 60 + 40.0 / 3600
alt = 100e3
reference_pos_ecef = frames.geodetic_wgs84_to_ecef(lat, lon, alt, degrees=True)
velocity_dir_ecef = frames.azel_to_ecef(lat, lon, az=10, el=-45, degrees=True)


material = metablate.material.asteroidal
model = ks.KeroSzasz2008(
    options=ks.KeroSzaszOptions(
        material=material,
        sputtering=False,
        minimum_mass=1e-11,
        max_step_size=1e-3,
        start_altitude=150e3,
    ),
)

result = model.run(
    parameters=ks.KeroSzaszInitialState(
        epoch=np.datetime64("2018-06-28T12:45:33"),
        position_ecef=reference_pos_ecef,
        velocity_ecef=velocity_dir_ecef * 60e3,
        mass=1e-6,
    ),
)
print(f"{result.runtime=} s")
llh = frames.ecef_to_geodetic_wgs84(
    result.position_ecef[0, :],
    result.position_ecef[1, :],
    result.position_ecef[2, :],
)
height = llh[2, :]

scale = 10
x = np.linspace(-0.05 * scale, 0.15 * scale, num=args.resolution)
y = np.linspace(-0.1 * scale, 0.1 * scale, num=args.resolution)

Q = -result.massloss / material.mean_atomic_mass

V_T = do.ablated_thermal_speed_bronshten_1983(
    meteoroid_surface_temperature=result.temperature,
    meteoroid_molecular_mass=material.mean_atomic_mass,
)

radii = np.cbrt(result.mass * 3 / (4 * np.pi * material.bulk_density))

n_0 = Q / (4 * np.pi * radii**2 * V_T) * np.sqrt(np.pi / 2)

collision_sigma = do.collisional_cross_section_bronshten_1983(result.velocity)
ionization_beta = do.ionization_probability_Na_vondrak_2008(result.velocity)
radar_freq = 50e6
radar_critical_dens = do.critical_plasma_density(radar_freq)

atm_model = metablate.atmosphere.AtmPymsis()

atm = atm_model.density(
    time=np.datetime64("2018-07-28"),
    lat=np.array([69.0]),
    lon=np.array([12.0]),
    alt=height,
    mass_densities=False,
)
num_tot = atm["N2"].values.flatten()

mean_free_path = do.mean_free_path(
    V_T,
    num_tot,
    collision_sigma,
    result.velocity,
)

fig, axes = plt.subplots(2, 3, layout="tight")
axes[0, 0].plot(result.velocity * 1e-3, height * 1e-3)
axes[0, 1].plot(collision_sigma, height * 1e-3)
axes[0, 2].plot(num_tot, height * 1e-3)
axes[1, 0].plot(V_T * 1e-3, height * 1e-3)
axes[1, 1].plot(mean_free_path, height * 1e-3)
axes[1, 2].plot(result.temperature, height * 1e-3)
axes[0, 0].set_xlabel("velocity")
axes[0, 1].set_xlabel("collsion sigma")
axes[0, 2].set_xlabel("num tot")
axes[1, 0].set_xlabel("v_t")
axes[1, 1].set_xlabel("mfp")
axes[1, 2].set_xlabel("T")

plt.show()

[xmat, ymat] = np.meshgrid(x, y)
rcs = np.zeros_like(result.t)

ne_dists = np.full((len(y), len(x), len(result.t)), np.nan, dtype=np.float64)
cache = False
if args.cache is not None:
    cache_pth = pathlib.Path(args.cache)
    if cache_pth.exists():
        _dat = np.load(cache_pth)
        ne_dists = _dat["ne"]
        cache = True

# test_index = list(range(0, len(result.t), 120))
test_index = np.argwhere(result.mass < result.mass[0] * 0.95).flatten().tolist()
# for ind in tqdm(range(len(result.t))):
for ind in tqdm(test_index):
    xlam_v = x / mean_free_path[ind]
    ylam_v = y / mean_free_path[ind]
    if not cache:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=IntegrationWarning)

            xlam, ylam, rlam, do_f_dist = do.plasma_distribution_morphology_grid(
                x_grid=xlam_v,
                y_grid=ylam_v,
                threads=8,
                pbar=False,
            )
        ne_dist = do.plasma_distribution(
            total_atmospheric_number_density=num_tot[ind],
            meteoroid_velocity=result.velocity[ind],
            meteoroid_radius=radii[ind],
            plasma_source_density=n_0[ind],
            collisional_cross_section=collision_sigma[ind],
            ionization_probability=ionization_beta[ind],
            ablated_thermal_speed=V_T[ind],
            atmospheric_species_mass=atm_model.species["N2"].A * constants.u,
            ablated_species_mass=material.mean_atomic_mass,
            base_plasma_distribution_function=do_f_dist,
            R_lambda=rlam,
        )
        ne_dists[:, :, ind] = ne_dist
    else:
        ne_dist = ne_dists[:, :, ind]

    f_p = do.plasma_frequency(ne_dist)
    area_unit = (x[1] - x[0]) * (y[1] - y[0])
    rcs[ind] = np.sum(f_p > radar_freq) * area_unit

    # fig, ax = plt.subplots()
    # cmap = "inferno"
    # im = ax.pcolormesh(xmat, ymat, ne_dist, cmap=cmap, norm=LogNorm())
    # cb = fig.colorbar(im, ax=ax)
    # cb.set_label("Electron density [m^-3]")
    # ax.set_xlabel("x [m]")
    # ax.set_ylabel("y [m]")
    # plt.show()


if args.cache is not None and not cache:
    np.savez(args.cache, ne=ne_dists)

fig = plt.figure(figsize=(10, 6), layout="tight")
gs = GridSpec(2, 5, figure=fig)

ax = fig.add_subplot(gs[:, :3])
ax_small_1 = fig.add_subplot(gs[0, 3])
ax_small_2 = fig.add_subplot(gs[1, 3])
ax_small_3 = fig.add_subplot(gs[0, 4])
ax_small_4 = fig.add_subplot(gs[1, 4])
cmap = "inferno"

norm = LogNorm(
    vmax=np.nanmax(ne_dists.flatten()),
    vmin=max(np.nanmin(ne_dists.flatten()), 1e-15),
)
pcm = ax.pcolormesh(
    xmat,
    ymat,
    ne_dists[:, :, 0],
    cmap=cmap,
    shading="auto",
    norm=norm,
)
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label("Electron density [m^-3]")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

ax_small_1.plot(result.t, n_0)
ln1 = ax_small_1.axvline(0, c="r")
ax_small_2.plot(result.t, rcs)
ln2 = ax_small_2.axvline(0, c="r")
ax_small_3.semilogy(result.t, mean_free_path)
ax_small_4.plot(result.t, -result.massloss)

ax_small_1.set_ylabel("Plasma source density [1/m^3]")
ax_small_2.set_ylabel("Radar cross section [m^2]")
ax_small_3.set_ylabel("Mean free path [m]")
ax_small_4.set_ylabel("Massloss [kg/s]")

max_r = np.sqrt(np.nanmax(rcs) / np.pi)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())


def update(frame):
    global pcm
    pcm.remove()
    pcm = ax.pcolormesh(
        xmat,
        ymat,
        ne_dists[:, :, frame],
        cmap=cmap,
        shading="auto",
        norm=norm,
    )
    ln1.set_xdata([result.t[frame]])
    ln2.set_xdata([result.t[frame]])
    return pcm, ln1, ln2


ani = animation.FuncAnimation(fig=fig, func=update, frames=test_index, interval=10)

plt.show()
