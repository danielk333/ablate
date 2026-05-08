from scipy import constants
import numpy as np


def speed_of_sound_air_R(temperature, individual_gas_constant, adiabatic_index=1.4):
    """Calculate speed of sound in an ideal gas using the Newton–Laplace equation and the ideal gas
    law.
    """
    return np.sqrt(individual_gas_constant * adiabatic_index * temperature)


def speed_of_sound_air(temperature, mean_molecular_mass, adiabatic_index=1.4):
    """Calculate speed of sound in an ideal gas using the Newton–Laplace equation and the ideal gas
    law.
    """
    return np.sqrt(constants.k * adiabatic_index * temperature / mean_molecular_mass)


def mach_number(velocity, wave_velocity):
    return velocity / wave_velocity


def rankine_hugoniot_post_shock_temperature(
    pre_shock_temperature,
    pre_shoch_mach_number,
    adiabatic_index=1.4,
):
    """The post-shock temperature of a normal shocks adiabatic flow of a completly perfect fluid [1].

    [1]: Ames Research Staff, 1953. Equations, tables, and charts for compressible flow (No. NACA-TR-1135). National Advisory Committee for Aeronautics, Washington, D.C.
    """
    coef_1 = 2 * adiabatic_index * pre_shoch_mach_number**2
    coef_2 = (adiabatic_index - 1) * pre_shoch_mach_number**2 + 2
    coef_3 = (adiabatic_index + 1) ** 2 * pre_shoch_mach_number**2
    return pre_shock_temperature * coef_1 * coef_2 / coef_3


def rankine_hugoniot_post_shock_mach_number(
    pre_shoch_mach_number,
    adiabatic_index=1.4,
):
    """The post-shock temperature of a normal shocks adiabatic flow of a completly perfect fluid [1].

    [1]: Ames Research Staff, 1953. Equations, tables, and charts for compressible flow (No. NACA-TR-1135). National Advisory Committee for Aeronautics, Washington, D.C.
    """
    pass


def atmospheric_mean_free_path(number_density, collisional_cross_section):
    return 1 / (np.sqrt(2) * number_density * collisional_cross_section)
