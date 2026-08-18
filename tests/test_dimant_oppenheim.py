import numpy as np
import pytest
from scipy import constants

from metablate.physics import plasma_production as plasma
from metablate.physics import thermal_ablation


def test_plasma_frequency_and_critical_density_are_inverses():
    density = np.array([1.0e10, 1.0e12, 1.0e14])

    frequency = plasma.plasma_frequency(density)
    recovered_density = plasma.critical_plasma_density(frequency)

    np.testing.assert_allclose(recovered_density, density, rtol=1e-12)


def test_sugar_thermal_speed_matches_simulation_parameter():
    # Sugar et al. (2018), Sect. 5.2 and Table 1, state VT = 951 m/s for
    # Na at a typical ablation temperature of 2500 K.
    # https://doi.org/10.1002/2018JA025265
    temperature = 2500.0
    molecular_mass = 22.98976928 * constants.u

    assert thermal_ablation.ablated_thermal_speed_sugar_2018(temperature, molecular_mass) == pytest.approx(
        951.0, abs=1.0
    )


def test_bronshten_evaporated_speed_matches_table_1():
    # Bronshten (1983), Chapter II, Sect. 7, p. 37, Table 1, lists
    # ve = 0.66 km/s at Tw = 1000 K for evaporated molecules with mu_m = 56.
    temperature = 1000.0
    molecular_mass = 56.0 * constants.u

    assert thermal_ablation.ablated_thermal_speed_bronshten_1983(
        temperature, molecular_mass
    ) == pytest.approx(0.66e3, abs=10.0)
