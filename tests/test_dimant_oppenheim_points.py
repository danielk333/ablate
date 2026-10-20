"""Paired coordinates and inversion of the normalized electron density."""

import numpy as np
import pytest

from metablate.models import dimant_oppenheim_2017 as dimant


def test_scalar_and_empty_inputs():
    result = dimant.plasma_distribution_morphology_array(1.0, 0.0, pbar=False)
    assert all(
        isinstance(values, np.ndarray) and values.shape == () for values in result
    )
    assert result[3] > 0
    result = dimant.plasma_distribution_morphology_array([], 0.0, pbar=False)
    assert all(values.shape == (0,) for values in result)


@pytest.mark.parametrize("r, theta", [(-1, 0), (np.inf, 0), (1, np.nan)])
def test_invalid_polar_coordinates(r, theta):
    with pytest.raises(ValueError):
        dimant.plasma_distribution_morphology_array(r, theta, pbar=False)


@pytest.mark.parametrize("theta", [0.0, 1e-5, np.pi / 4, np.pi / 2, np.pi - 1e-5])
def test_solver_round_trip_at_other_angles(theta):
    _, _, R, f = dimant.plasma_distribution_morphology_array(0.7, theta, pbar=False)
    recovered, density, residual = dimant.solve_r(theta, float(f / R), xtol=1e-9)
    assert recovered == pytest.approx(0.7, abs=1e-8)
    assert density == pytest.approx(float(f / R), rel=1e-7)
    assert abs(residual) < 1e-7
