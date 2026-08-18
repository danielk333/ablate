import numpy as np
from scipy import constants

from metablate.physics import aerodynamics


def test_speed_of_sound_forms_are_equivalent_for_air():
    temperature = np.array([220.0, 288.15])
    mean_molecular_mass = 28.9647e-3 / constants.N_A
    gas_constant = constants.R / 28.9647e-3

    from_mass = aerodynamics.speed_of_sound_air(temperature, mean_molecular_mass)
    from_r = aerodynamics.speed_of_sound_air_R(temperature, gas_constant)

    np.testing.assert_allclose(from_mass, from_r, rtol=1e-12)


def test_mach():
    assert aerodynamics.mach_number(600.0, 300.0) == 2.0
