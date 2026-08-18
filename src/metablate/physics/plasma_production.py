from __future__ import annotations

import numpy as np
from scipy import constants

from ..types import NDArray


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


def plasma_frequency(electron_density: NDArray | float) -> NDArray | float:
    """Return electron plasma frequency in Hz [^1].

    [^1]: Sugar, G., et al. (2021), Simulation‐Derived Radar Cross Sections of a New Meteor Head Plasma Distribution Model
    """
    return np.sqrt(electron_density * constants.e**2 / (constants.epsilon_0 * constants.m_e)) / (2 * np.pi)


def critical_plasma_density(critical_plasma_frequency: NDArray | float) -> NDArray | float:
    """Return the density associated with an electron plasma frequency [^1].

    [^1]: Sugar, G., et al. (2021), Simulation‐Derived Radar Cross Sections of a New Meteor Head Plasma Distribution Model
    """
    return (
        (critical_plasma_frequency * 2 * np.pi) ** 2 * constants.epsilon_0 * constants.m_e / (constants.e**2)
    )


def mean_free_path_dimant_oppenheim_2017(
    ablated_thermal_speed: NDArray | float,
    total_atmospheric_number_density: NDArray | float,
    collisional_cross_section: NDArray | float,
    meteoroid_velocity: NDArray | float,
) -> NDArray | float:
    """Mean free path of an ablated neutral before collision with an atmospheric constituent,
    Equation 1 of [^1].

    [^1]: Sugar, G., Oppenheim, M. M., Dimant, Y. S. & Close, S. (2018),
        Formation of Plasma Around a Small Meteoroid: Simulation and Theory
    """
    return ablated_thermal_speed / (
        total_atmospheric_number_density * collisional_cross_section * meteoroid_velocity
    )


def collisional_cross_section_bronshten_1983(relative_velocity: NDArray | float) -> NDArray | float:
    """Return the Bronshten collision cross-section fit [^1].

    [^1]: Sugar, G., Oppenheim, M. M., Dimant, Y. S. & Close, S. (2018),
        Formation of Plasma Around a Small Meteoroid: Simulation and Theory
    """
    vrel = relative_velocity * 1e-3  # requires km/s
    return 5.61e-19 * vrel ** (-0.8)


def ionization_probability_Na_vondrak_2008(relative_velocity: NDArray | float) -> NDArray | float:
    """Return the fitted Na collisional ionization probability [^1].

    [^1]: Vondrak, T., Plane, J. M. C., Broadley, S. & Janches, D. (2008), A chemical model of meteoric ablation
    """
    vrel = relative_velocity * 1e-3  # requires km/s
    return 0.933 * (vrel - 8.86) ** 2 * vrel ** (-1.94)
