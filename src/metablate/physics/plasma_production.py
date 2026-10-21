from __future__ import annotations

import numpy as np
from scipy import constants
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ..types import ArrayLike, Composition, Material, NDArray, NDArray_MxN, NDArray_N, Numeric
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


# Element: (threshold [km/s], coefficient c, atom/N2 mass ratio)
_JONES_1997_PARAMETERS = {
    "O": (16.7, 4.66e-6, 0.57),
    "Fe": (9.4, 34.5e-6, 2.0),
    "Mg": (11.1, 9.29e-6, 0.86),
    "Si": (11.0, 18.5e-6, 1.0),
    "Cu": (9.1, 15.3e-6, 2.25),
}


def multi_collision_ionization_probability_Jones_1997(
    velocity: NDArray_N,
    composition: Composition,
    rtol: float = 1e-10,
) -> tuple[NDArray_MxN, NDArray_N]:
    """Return total ionization probability beta from Eq. 25 of [^1].

    [^1]: William Jones; Theoretical and observational determinations
        of the ionization coefficient of meteors; MNRAS 288, 995-1003 (1997),
        Table 1 and Eq. 33. https://doi.org/10.1093/mnras/288.4.995
    """
    speed = velocity * 1e-3
    if np.any(~np.isfinite(speed)) or np.any(speed < 0):
        raise ValueError("Velocity must be finite and nonnegative [m/s].")

    if not composition:
        raise ValueError("Material must specify a nonempty atomic_composition.")

    abundances = np.asarray(list(composition.values()), dtype=float)
    total = abundances.sum()
    if np.any(~np.isfinite(abundances)) or np.any(abundances < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError("Atomic abundances must be finite, nonnegative and have a positive total.")

    probability = np.zeros((abundances.size, speed.size), dtype=np.float64)

    def first_collision(v: Numeric, v0: float, c: float) -> Numeric:
        ratio = c * np.maximum(v - v0, 0.0) ** 2 * v**0.8
        return ratio / (1.0 + ratio)

    # H(v) = integral from threshold to v of beta(u) * u du.
    def derivative(v: float, integral: NDArray, factor: float, v0: float, c: float) -> list[float]:
        beta0 = first_collision(v, v0, c)
        return [v * beta0 + (1.0 - beta0) * factor / v * integral[0]]

    for index, element in enumerate(composition):
        if element not in _JONES_1997_PARAMETERS:
            raise ValueError(
                f"No Jones (1997) ionization parameters for {element!r}; "
                f"supported elements: {', '.join(_JONES_1997_PARAMETERS)}."
            )
        threshold, coefficient, mass_ratio = _JONES_1997_PARAMETERS[element]

        active = speed > threshold
        if not np.any(active):
            continue

        # NOTE: There seems to be something strange with eq 25 of Jones paper
        # in eq 24 the factor is (1+mu)^2/(2*v^2*mu) but in 25 its (1+m)^2/(2*v^2*mu)
        # maybe the u was forgotten in typesetting????
        factor = (1.0 + mass_ratio) ** 2 / (2.0 * mass_ratio)

        solution = solve_ivp(
            derivative,
            (threshold, float(speed.max())),
            [0.0],
            args=(factor, threshold, coefficient),
            rtol=rtol,
            atol=rtol * 1e-3,
            dense_output=True,
        )
        if not solution.success:
            raise RuntimeError(solution.message)

        v = speed[active]
        beta0 = first_collision(v, threshold, coefficient)
        probability[index, active] = beta0 + (1.0 - beta0) * factor / v**2 * solution.sol(v)[0]

    total = np.sum(probability * abundances[:, None] / total, axis=0)
    return probability, total


def ionization_probability_Jones_1997(
    velocity: NDArray_N,
    composition: Composition,
) -> tuple[NDArray_MxN, NDArray_N]:
    """Return first-collision ionization probability beta_0 from Eq. 33 of [^1].

    The result has the same shape (a zero-dimensional array for scalar input) as the number of
    elements ``M`` and the number of velocities ``N`` for the probabilities and also returns the
    total.

    Average the elemental probabilities using ``atomic_composition``,
    whose relative abundances are by atom number, not mass.
    Supported elements are O, Fe, Mg, Si and Cu; abundances need not sum to one.

    [^1]: William Jones; Theoretical and observational determinations
        of the ionization coefficient of meteors; MNRAS 288, 995-1003 (1997),
        Table 1 and Eq. 33. https://doi.org/10.1093/mnras/288.4.995
    """
    speed = velocity * 1e-3
    if np.any(~np.isfinite(speed)) or np.any(speed < 0):
        raise ValueError("Velocity must be finite and nonnegative [m/s].")

    if not composition:
        raise ValueError("Material must specify a nonempty atomic_composition.")

    abundances = np.asarray(list(composition.values()), dtype=float)
    total = abundances.sum()
    if np.any(~np.isfinite(abundances)) or np.any(abundances < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError("Atomic abundances must be finite, nonnegative and have a positive total.")

    probability = np.zeros((abundances.size, speed.size), dtype=np.float64)
    for index, element in enumerate(composition):
        if element not in _JONES_1997_PARAMETERS:
            raise ValueError(
                f"No Jones (1997) ionization parameters for {element!r}; "
                f"supported elements: {', '.join(_JONES_1997_PARAMETERS)}."
            )
        threshold, coefficient, _ = _JONES_1997_PARAMETERS[element]

        # Clamp before squaring so sub-threshold atoms cannot ionize.
        ratio = coefficient * np.maximum(speed - threshold, 0.0) ** 2 * speed**0.8
        probability[index, :] = ratio / (1.0 + ratio)

    total = np.sum(probability * abundances[:, None] / total, axis=0)
    return probability, total


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
