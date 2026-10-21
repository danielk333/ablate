from __future__ import annotations

import numpy as np
from scipy import constants
from scipy.optimize import brentq

from ..types import ArrayLike, Material, NDArray, Numeric
from .alpha_beta import mass_to_radius
from .thermal_ablation import ablated_thermal_speed_sugar_2018, thermal_ablation_hill_et_al_2005


def _source_estimation_residual(
    value: float,
    mass: float,
    log_target: float,
    shape_factor: float,
    material_data: Material,
) -> float:
    rate = thermal_ablation_hill_et_al_2005(
        mass,
        value,
        material_data,
        shape_factor=shape_factor,
    )
    return float(np.log(-rate) - log_target)


def source_from_massloss(
    mass: ArrayLike,
    massloss: ArrayLike,
    material_data: Material,
    temperature: ArrayLike | None = None,
    temperature_bounds: tuple[float, float] = (300.0, 6000.0),
) -> tuple[NDArray, NDArray, NDArray]:
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

            try:
                temperature[index] = brentq(
                    _source_estimation_residual,
                    lower,
                    upper,
                    args=(
                        mass[index],
                        log_target,
                        shape_factor,
                        material_data,
                    ),
                    xtol=1e-8,
                )
            except ValueError as error:
                raise ValueError(
                    f"Vapor massloss at {index} cannot be matched between {lower} and {upper} K."
                ) from error
    elif np.any(~np.isfinite(temperature)) or np.any(temperature <= 0):
        raise ValueError("Prescribed temperature must be finite and positive [K].")

    vapor_mass = material_data.ablated_vapour_mean_molecular_mass
    thermal_speed = np.asarray(
        ablated_thermal_speed_sugar_2018(
            meteoroid_surface_temperature=temperature,
            meteoroid_molecular_mass=vapor_mass,
        )
    )
    radius = mass_to_radius(mass, material_data.bulk_density)
    surface_area = 4 * np.pi * radius**2
    particle_rate = -massloss / vapor_mass
    density = np.zeros_like(mass)
    np.divide(particle_rate, surface_area * thermal_speed, out=density, where=massloss < 0)
    density *= np.sqrt(np.pi / 2)
    return temperature, thermal_speed, density


def ionization_probability_Jones_1997(
    velocity: float,
    meteoroid_bulk_density: float,
) -> NDArray:
    """Calculate ionization probability according to [^1].

    [^1]: William Jones; Theoretical and observational determinations
    of the ionization coefficient of meteors; MNRAS 288, p. 995-1003 (1997)

    """
    raise NotImplementedError()
    # v_km = v / 1000.0
    """
    # Table 1 from Jones:
    # %   : procentage composition by weight
    # p   : proportion by atom number
    # v0  : minimum velocity at which ionization can take place 
    # (below this velocity, the ionization energy is greater than 
    # the total energy of the colliding atoms in the centre-of-mass frame)
    # c   : empirically derived coeff
    # mu  : ratio of atom mass to a nitrogen molecule
    #                 % p v0 c mu element
    ion_param       = [
        [45, 0.617, 16.7, 4.66E-6, 0.57],  # O
        [15, 0.059, 9.4, 34.5E-6, 2.0],  # Fe
        [9, 0.082, 11.1, 9.29E-6, 0.86],  # Mg
        [31, 0.242, 11.0, 18.5E-6, 1.0],  # Si
    ]
             
    # beta0 = c .* (v - v0).^2 .* v.^0.8 ./ (1 + c .* (v - v0).^2 .* v.^0.8); primary ionization probability, eq. 33 in Jones
    # v is needed in km/s!

    if rho_m == 7800:
        # Then we calculate the ionization probability for pure Fe
        beta0 = ion_param(2,4) .* (v_km - ion_param(2,3)).^2 .* v_km.^0.8 ./ (1 + ion_param(2,4) .* (v_km - ion_param(2,3)).^2 .* v_km.^0.8);
        beta0(find(v_km<ion_param(2,3))) = 0;
    else
        # Otherwise we use Jones' cometary composition (for porous, cometary as well as asteroidal densities)
        # It is faster not to use repmat.m but to generate the necessary matrices with more simple tools!
        c               = [ion_param(1,4)*ones(length(v),1) ion_param(2,4)*ones(length(v),1) ion_param(3,4)*ones(length(v),1) ion_param(4,4)*ones(length(v),1)];
        v0              = [ion_param(1,3)*ones(length(v),1) ion_param(2,3)*ones(length(v),1) ion_param(3,3)*ones(length(v),1) ion_param(4,3)*ones(length(v),1)];
        p               = [ion_param(1,2)*ones(length(v),1) ion_param(2,2)*ones(length(v),1) ion_param(3,2)*ones(length(v),1) ion_param(4,2)*ones(length(v),1)];
        v_km            = [v_km v_km v_km v_km]; 
        beta0_1         = c .* (v_km - v0).^2 .* v_km.^0.8 ./ (1 + c .* (v_km - v0).^2 .* v_km.^0.8);
        beta0_1(find(v_km<v0)) = 0;
        beta0           = sum(beta0_1 .* p , 2);
    """


def plasma_frequency(electron_density: Numeric) -> Numeric:
    """Return electron plasma frequency in Hz [^1].

    [^1]: Sugar, G., et al. (2021), Simulation‐Derived Radar Cross Sections of a New Meteor Head Plasma Distribution Model
    """
    return np.sqrt(electron_density * constants.e**2 / (constants.epsilon_0 * constants.m_e)) / (2 * np.pi)


def critical_plasma_density(critical_plasma_frequency: Numeric) -> Numeric:
    """Return the density associated with an electron plasma frequency [^1].

    [^1]: Sugar, G., et al. (2021), Simulation‐Derived Radar Cross Sections of a New Meteor Head Plasma Distribution Model
    """
    return (
        (critical_plasma_frequency * 2 * np.pi) ** 2 * constants.epsilon_0 * constants.m_e / (constants.e**2)
    )


def mean_free_path_dimant_oppenheim_2017(
    ablated_thermal_speed: Numeric,
    total_atmospheric_number_density: Numeric,
    collisional_cross_section: Numeric,
    meteoroid_velocity: Numeric,
) -> Numeric:
    """Mean free path of an ablated neutral before collision with an atmospheric constituent,
    Equation 1 of [^1].

    [^1]: Sugar, G., Oppenheim, M. M., Dimant, Y. S. & Close, S. (2018),
        Formation of Plasma Around a Small Meteoroid: Simulation and Theory
    """
    return ablated_thermal_speed / (
        total_atmospheric_number_density * collisional_cross_section * meteoroid_velocity
    )


def collisional_cross_section_bronshten_1983(relative_velocity: Numeric) -> Numeric:
    """Return the Bronshten collision cross-section fit [^1].

    [^1]: Sugar, G., Oppenheim, M. M., Dimant, Y. S. & Close, S. (2018),
        Formation of Plasma Around a Small Meteoroid: Simulation and Theory
    """
    vrel = relative_velocity * 1e-3  # requires km/s
    return 5.61e-19 * vrel ** (-0.8)


def ionization_probability_Na_vondrak_2008(relative_velocity: Numeric) -> Numeric:
    """Return the fitted Na collisional ionization probability [^1].

    [^1]: Vondrak, T., Plane, J. M. C., Broadley, S. & Janches, D. (2008), A chemical model of meteoric ablation
    """
    vrel = relative_velocity * 1e-3  # requires km/s
    return 0.933 * (vrel - 8.86) ** 2 * vrel ** (-1.94)
