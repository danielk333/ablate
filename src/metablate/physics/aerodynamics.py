import numpy as np
from scipy import constants


def speed_of_sound_air_R(temperature, individual_gas_constant, adiabatic_index=1.4):
    """Calculate speed of sound in an ideal gas using the Newton–Laplace equation and the ideal gas
    law [^1].

    [^1]: Ames Research Staff (1953), Equations, Tables, and Charts for Compressible Flow, NACA Report 1135.
    """
    return np.sqrt(individual_gas_constant * adiabatic_index * temperature)


def speed_of_sound_air(temperature, mean_molecular_mass, adiabatic_index=1.4):
    """Calculate speed of sound in an ideal gas using the Newton–Laplace equation and the ideal gas
    law [^1].

    [^1]: Ames Research Staff (1953), Equations, Tables, and Charts for Compressible Flow, NACA Report 1135.
    """
    # Express k_B as R/N_A so this form is numerically consistent with
    # speed_of_sound_air_R even in SciPy releases whose tabulated R is rounded.
    gas_constant = constants.R / (mean_molecular_mass * constants.N_A)
    return np.sqrt(gas_constant * adiabatic_index * temperature)


def mach_number(velocity, wave_velocity):
    return velocity / wave_velocity


def rankine_hugoniot_post_shock_temperature(
    pre_shock_temperature,
    pre_shoch_mach_number,
    adiabatic_index=1.4,
):
    """The post-shock temperature of a normal shocks adiabatic flow of a completly perfect fluid [^1].

    [^1]: Ames Research Staff (1953), Equations, Tables, and Charts for Compressible Flow, NACA Report 1135.
    """
    coef_1_numerator = 2 * adiabatic_index * pre_shoch_mach_number**2 - (adiabatic_index - 1)
    coef_2_numerator = (adiabatic_index - 1) * pre_shoch_mach_number**2 + 2
    coef_denomenator = (adiabatic_index + 1) ** 2 * pre_shoch_mach_number**2
    return pre_shock_temperature * coef_1_numerator * coef_2_numerator / coef_denomenator


def rankine_hugoniot_post_shock_mach_number(
    pre_shoch_mach_number,
    adiabatic_index=1.4,
):
    """The squared downstream Mach number for a normal shock in a calorically perfect fluid [^1].

    [^1]: Ames Research Staff (1953), Equations, Tables, and Charts for Compressible Flow, NACA Report 1135.
    """
    coef_numerator = (adiabatic_index - 1) * pre_shoch_mach_number**2 + 2
    coef_denomenator = 2 * adiabatic_index * pre_shoch_mach_number**2 - (adiabatic_index - 1)
    return coef_numerator / coef_denomenator


def atmospheric_mean_free_path(number_density, collisional_cross_section):
    """Mean free path in a hard-sphere gas [^1].

    [^1]: Hedin, J., et al. (2007), On the efficiency of rocket-borne particle detection in the mesosphere
    """
    return 1 / (np.sqrt(2) * number_density * collisional_cross_section)
