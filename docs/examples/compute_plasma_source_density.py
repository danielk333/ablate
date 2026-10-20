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
from scipy.optimize import brentq

import metablate.models.alpha_beta_2026 as ab
from metablate import physics
from metablate.material import asteroidal as material


# TODO: move this into the package
def source_from_massloss(
    mass,
    massloss,
    material_data,
    temperature=None,
    temperature_bounds=(300.0, 6000.0),
):
    """Return temperature [K], V_T [m/s] and Dimant n_0 [m^-3].

    ``mass`` is the remaining body mass [kg]; ``massloss`` is its vapor mass
    derivative [kg/s], negative for loss. With no prescribed temperature,
    solve the material evaporation law on its increasing branch
    within ``temperature_bounds``. At zero mass loss, n_0 is zero and
    an inferred temperature (and V_T) is undefined, represented by NaN.
    """
    mass, massloss = np.broadcast_arrays(np.asarray(mass, float), np.asarray(massloss, float))
    if temperature is not None:
        mass, massloss, temperature = np.broadcast_arrays(mass, massloss, np.asarray(temperature, float))
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("Remaining body mass must be finite and positive.")
    if np.any(~np.isfinite(massloss)) or np.any(massloss > 0):
        raise ValueError("Vapor massloss must be finite and nonpositive [kg/s].")

    if temperature is None:
        lower, upper = temperature_bounds
        # P_v/sqrt(T) increases below T=2*ln(10)*B for P_v=0.1*10**(A-B/T).
        turning_temperature = 2 * np.log(10) * material_data.clausius_clapeyron_coeff_B
        if not (np.isfinite(lower) and np.isfinite(upper) and 0 < lower < upper < turning_temperature):
            raise ValueError("Temperature bounds must be on the increasing evaporation branch.")
        temperature = np.full(mass.shape, np.nan)
        # Hill's area is 4*A*(M/rho)**(2/3); A=1.209... gives a sphere's 4*pi*r**2.
        shape_factor = np.pi * (3 / (4 * np.pi)) ** (2 / 3)
        for index in np.ndindex(mass.shape):
            if massloss[index] == 0:
                continue
            log_target = np.log(-massloss[index])

            def residual(value):
                rate = physics.thermal_ablation.thermal_ablation_hill_et_al_2005(
                    mass[index], value, material_data, shape_factor=shape_factor
                )
                return np.log(-rate) - log_target

            try:
                temperature[index] = brentq(residual, lower, upper, xtol=1e-8)
            except ValueError as error:
                raise ValueError(
                    f"Vapor massloss at {index} cannot be matched between {lower} and {upper} K."
                ) from error
    elif np.any(~np.isfinite(temperature)) or np.any(temperature <= 0):
        raise ValueError("Prescribed temperature must be finite and positive [K].")

    vapor_mass = material_data.ablated_vapour_mean_molecular_mass
    thermal_speed = physics.thermal_ablation.ablated_thermal_speed_sugar_2018(
        meteoroid_surface_temperature=temperature, meteoroid_molecular_mass=vapor_mass
    )
    radius = physics.alpha_beta.mass_to_radius(mass, material_data.bulk_density)
    surface_area = 4 * np.pi * radius**2
    particle_rate = -massloss / vapor_mass
    density = np.zeros_like(mass)
    np.divide(particle_rate, surface_area * thermal_speed, out=density, where=massloss < 0)
    density *= np.sqrt(np.pi / 2)
    return temperature, thermal_speed, density


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    temperature, thermal_speed, source_density = source_from_massloss(
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


if __name__ == "__main__":
    main()
