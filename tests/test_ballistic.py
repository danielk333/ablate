import numpy as np
import pytest

from metablate.physics import ballistic


def test_fit_velocity_with_fixed_initial_velocity():
    heights = np.linspace(100e3, 40e3, 30)
    density = np.full_like(heights, 1e-5)
    initial_velocity = 30e3
    coefficient = 2.0
    elevation = np.pi / 4
    velocities = ballistic.velocity(
        heights, density, coefficient, elevation, initial_velocity=initial_velocity
    )

    fitted = ballistic.fit_velocity(
        heights,
        velocities,
        density,
        elevation,
        initial_velocity=initial_velocity,
        minimize_kwargs={"method": "Nelder-Mead", "options": {"xatol": 1e-10}},
    )

    assert fitted[0] == pytest.approx(coefficient, rel=1e-6)
