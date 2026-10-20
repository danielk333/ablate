import numpy as np
import pytest
from scipy import constants

from metablate.physics import plasma_production as plasma
from metablate.physics import thermal_ablation
from metablate.models import dimant_oppenheim_2017 as dimant
from scipy.integrate import quad
from scipy.special import erf
import warnings


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


def test_morphology_matches_published_axis_density():
    # Dimant & Oppenheim (2017), Eqs. 43-44, independently written axis
    # expressions.
    radius = np.array([0.01, 0.1, 1.0, 10.0])
    q = radius ** (2 / 3)
    exponential = np.exp(-1.5 * q)
    term_1 = (4 - np.pi) * np.sqrt(2 * np.pi) / (2 * np.sqrt(2 * np.pi + (4 - np.pi) ** 2 * q))
    expected = (1 - term_1) * exponential
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, _, morphology = dimant.plasma_distribution_morphology_grid(
            -radius, np.array([0.0]), pbar=False
        )
    np.testing.assert_allclose(morphology[0], expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("delta", [0.0, 1e-6])
def test_near_axis_density_is_finite_without_invalid_integrands(delta):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, _, morphology = dimant.plasma_distribution_morphology_grid(
            np.array([-1.0, 1.0]),
            np.array([1e-5]),
            upper_limit_delta=delta,
            pbar=False,
        )
    assert np.all(np.isfinite(morphology))
    assert np.all(morphology > 0)


def test_grid_symmetry_parallel_parity_and_undefined_center():
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([-0.5, 0.0, 0.5])
    serial = dimant.plasma_distribution_morphology_grid(x, y, pbar=False)[3]
    parallel = dimant.plasma_distribution_morphology_grid(x, y, threads=2, pbar=False)[3]
    np.testing.assert_allclose(serial, parallel, equal_nan=True)
    np.testing.assert_allclose(serial[0], serial[2])
    assert np.isnan(serial[1, 1])
    assert serial[1, 2] > serial[1, 0]
