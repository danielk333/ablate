"""Visualize the Dimant--Oppenheim meteoric head-plasma distribution in 3D.

The plasma model is cylindrically symmetric about the meteoroid velocity (x)
axis.  It is therefore evaluated in an x-r plane and revolved onto a uniform
3D grid for volume rendering.
"""

import argparse
import pathlib
import warnings

import numpy as np
import pyvista as pv
from scipy.integrate import IntegrationWarning

import metablate.atmosphere as atm
import metablate.models.dimant_oppenheim_2017 as do
from metablate import physics


def unit_interval(value):
    """Parse a floating-point value constrained to the unit interval."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return value


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default=None,
        help="Optional .npz file used to cache the expensive 2D morphology calculation.",
    )
    parser.add_argument(
        "--scale",
        choices=[0.0005, 0.005, 0.05, 0.5],
        type=float,
        default=0.05,
        help="Grid spacing in thermal mean-free-path units.",
    )
    parser.add_argument(
        "--log-density",
        action="store_true",
        help="Map color and opacity to log10 electron density instead of density.",
    )
    parser.add_argument(
        "--max-opacity",
        type=unit_interval,
        default=0.3,
        help="Maximum volume opacity between 0 and 1 (default: 0.3).",
    )
    return parser.parse_args()


def calculate_plasma_distribution(scale=0.05, cache_path=None):
    """Calculate electron density in the model's cylindrically symmetric plane."""
    meteoroid_velocity = 40e3
    meteoroid_mass = 3.8e-26
    ablated_thermal_velocity = physics.thermal_ablation.ablated_thermal_speed_sugar_2018(
        meteoroid_surface_temperature=2000,
        meteoroid_molecular_mass=meteoroid_mass,
    )
    meteoroid_radius = 5e-5
    atmospheric_mass = 4.7e-26
    plasma_source_density = 10**13 / meteoroid_radius**2
    relative_velocity = meteoroid_velocity - ablated_thermal_velocity
    cross_section = physics.plasma_production.collisional_cross_section_bronshten_1983(relative_velocity)
    ionization_probability = physics.plasma_production.ionization_probability_Na_vondrak_2008(relative_velocity)

    atmosphere = atm.AtmPymsis().density(
        time=np.datetime64("2018-07-28"),
        lat=np.array([69.0]),
        lon=np.array([12.0]),
        alt=np.array([100e3]),
        mass_densities=False,
    )
    index = {"lat": 0, "lon": 0, "alt": 0, "time": 0}
    atmospheric_number_density = atmosphere["Total"][index].values / atmospheric_mass

    mean_free_path = physics.plasma_production.mean_free_path_dimant_oppenheim_2017(
        ablated_thermal_velocity,
        atmospheric_number_density,
        cross_section,
        meteoroid_velocity,
    )

    cache = pathlib.Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists():
        cached = np.load(cache)
        x_lambda = cached["x"]
        y_lambda = cached["y"]
        radius_lambda = cached["r"]
        morphology = cached["f"]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=IntegrationWarning)
            x_lambda, y_lambda, radius_lambda, morphology = do.plasma_distribution_morphology_grid(
                x_grid=(np.arange(201) - 60) * scale,
                y_grid=(np.arange(201) - 100) * scale,
                threads=8,
            )
        if cache is not None:
            np.savez(cache, x=x_lambda, y=y_lambda, r=radius_lambda, f=morphology)

    electron_density = do.plasma_distribution(
        total_atmospheric_number_density=atmospheric_number_density,
        meteoroid_velocity=meteoroid_velocity,
        meteoroid_radius=meteoroid_radius,
        plasma_source_density=plasma_source_density,
        collisional_cross_section=cross_section,
        ionization_probability=ionization_probability,
        ablated_thermal_speed=ablated_thermal_velocity,
        atmospheric_species_mass=atmospheric_mass,
        ablated_species_mass=meteoroid_mass,
        base_plasma_distribution_function=morphology,
        R_lambda=radius_lambda,
    )

    return (
        x_lambda[0, :] * mean_free_path,
        y_lambda[:, 0] * mean_free_path,
        electron_density,
    )


def revolve_density(x, radial_axis, density):
    """Revolve an x-r electron-density field about x onto a uniform x-y-z grid."""
    x = np.asarray(x, dtype=float)
    radial_axis = np.asarray(radial_axis, dtype=float)
    density = np.asarray(density, dtype=float)
    if density.shape != (radial_axis.size, x.size):
        raise ValueError("density must have shape (len(radial_axis), len(x))")
    if x.size < 2 or radial_axis.size < 3:
        raise ValueError("x and radial_axis must contain enough points to form a 3D grid")

    nonnegative = radial_axis >= 0
    radius = radial_axis[nonnegative]
    radial_density = density[nonnegative, :]
    if radius.size < 2 or not np.isclose(radius[0], 0.0):
        raise ValueError("radial_axis must include zero and positive radii")

    transverse = np.linspace(-radius[-1], radius[-1], radial_axis.size)
    yy, zz = np.meshgrid(transverse, transverse, indexing="ij")
    distance = np.sqrt(yy**2 + zz**2).ravel()
    volume = np.full((x.size, transverse.size, transverse.size), np.nan, dtype=np.float32)

    for x_index, profile in enumerate(radial_density.T):
        finite = np.isfinite(profile) & (profile > 0)
        if np.count_nonzero(finite) < 2:
            continue
        revolved = np.interp(
            distance,
            radius[finite],
            profile[finite],
            left=profile[finite][0],
            right=np.nan,
        )
        volume[x_index, :, :] = revolved.reshape(transverse.size, transverse.size)

    return x, transverse, volume


def density_image_data(x, transverse, density, log_density=False):
    """Create a PyVista image containing linear or log10 electron density."""
    valid = np.isfinite(density) & (density > 0)
    if not np.any(valid):
        raise ValueError("density does not contain any finite positive values")

    scalar_name = "log10 electron density" if log_density else "electron density"
    scalars = np.full(density.shape, density[valid].min(), dtype=np.float32)
    scalars[valid] = density[valid]
    if log_density:
        scalars = np.log10(scalars)

    grid = pv.ImageData(
        dimensions=density.shape,
        spacing=(x[1] - x[0], transverse[1] - transverse[0], transverse[1] - transverse[0]),
        origin=(x[0], transverse[0], transverse[0]),
    )
    grid.point_data[scalar_name] = scalars.flatten(order="F")
    grid.set_active_scalars(scalar_name)
    return grid


def opacity_for_threshold(scalar_range, threshold, max_opacity=0.3):
    """Create a peak-emphasizing quadratic opacity ramp above a threshold."""
    if not 0.0 <= max_opacity <= 1.0:
        raise ValueError("max_opacity must be between 0 and 1")
    scalar_min, scalar_max = scalar_range
    levels = np.linspace(scalar_min, scalar_max, 256)
    if threshold >= scalar_max:
        opacity = np.zeros(256, dtype=np.uint8)
        opacity[-1] = round(max_opacity * 255)
        return opacity
    opacity = np.clip((levels - threshold) / (scalar_max - threshold), 0.0, 1.0) ** 2
    return (opacity * max_opacity * 255).astype(np.uint8)


def plot_plasma(grid, max_opacity=0.3):
    """Plot the plasma cloud with a minimum-density slider."""
    scalar_range = grid.get_data_range(grid.active_scalars_name)
    initial_threshold = scalar_range[0]
    if grid.active_scalars_name == "log10 electron density":
        # Accumulating even a litle opacity through a volume makes the diffuse
        # end of a logarithmic range appear solid. Keep that tail transparent
        # and map the upper x% with a peak-emphasizing quadratic ramp.
        initial_threshold += 0.35 * (scalar_range[1] - scalar_range[0])
    opacity = opacity_for_threshold(scalar_range, initial_threshold, max_opacity=max_opacity)
    plotter = pv.Plotter()
    volume = plotter.add_volume(
        grid,
        scalars=grid.active_scalars_name,
        cmap="inferno",
        opacity=opacity,
        clim=scalar_range,
        show_scalar_bar=False,
    )
    volume.prop.interpolation_type = "linear"

    lookup_table = volume.mapper.lookup_table

    def set_minimum_density(value):
        """Update volume opacity when the density slider moves."""
        lookup_table.apply_opacity(
            opacity_for_threshold(scalar_range, value, max_opacity=max_opacity)
        )
        volume.prop.apply_lookup_table(lookup_table)
        plotter.render()

    log_density = grid.active_scalars_name == "log10 electron density"
    plotter.add_slider_widget(
        set_minimum_density,
        rng=scalar_range,
        value=initial_threshold,
        title="Minimum log10(ne [m^-3])" if log_density else "Minimum ne [m^-3]",
        pointa=(0.25, 0.92),
        pointb=(0.75, 0.92),
        color="gray",
        title_color="gray",
        fmt="%.2f" if log_density else "%.2e",
        interaction_event="always",
    )
    plotter.background_color = "black"
    plotter.show_grid(
        xtitle="x [m]",
        ytitle="y [m]",
        ztitle="z [m]",
        color="gray",
    )
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.3)
    plotter.enable_trackball_style()
    return plotter.show()


def main():
    """Calculate and display the 3D meteoric plasma distribution."""
    args = parse_args()
    x, radial_axis, density_2d = calculate_plasma_distribution(
        scale=args.scale,
        cache_path=args.cache,
    )
    x, transverse, density_3d = revolve_density(x, radial_axis, density_2d)
    grid = density_image_data(x, transverse, density_3d, log_density=args.log_density)
    plot_plasma(grid, max_opacity=args.max_opacity)


if __name__ == "__main__":
    main()
