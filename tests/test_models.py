import numpy as np
import pytest
from scipy import constants

from metablate import material
from metablate.models.simple_motion import (
    LinearConstantAcceleration,
    LinearConstantOptions,
    LinearConstantVelocity,
)


class _Values:
    def __init__(self, value):
        self.values = np.asarray(value, dtype=float)


class _ConstantAtmosphere:
    mean_mass = 28.0 * constants.u

    def density(self, **kwargs):
        return {"Total": _Values(1.0e-7), "N2": _Values(1.0e18), "O2": _Values(2.5e17)}


def test_linear_constant_velocity_model_shapes_and_values():
    times = np.array([0.0, 1.0, 2.0])
    options = LinearConstantOptions(evaluation_times=times)
    model = LinearConstantVelocity(options)
    initial = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])

    result = model.run(initial)

    np.testing.assert_allclose(result.position[:, 0], initial[:3])
    np.testing.assert_allclose(result.position[:, -1], initial[:3] + 2.0 * initial[3:6])
    np.testing.assert_allclose(result.velocity, np.repeat(initial[3:6], len(times)).reshape(3, len(times)))


def test_linear_constant_acceleration_model_values():
    times = np.array([0.0, 1.0, 2.0])
    options = LinearConstantOptions(evaluation_times=times)
    model = LinearConstantAcceleration(options)
    initial = np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0, -10.0])

    result = model.run(initial)

    np.testing.assert_allclose(result.position[0], [0.0, 95.0, 180.0])
    np.testing.assert_allclose(result.velocity[0], [100.0, 90.0, 80.0])


@pytest.mark.slow
def test_kero_szasz_short_deterministic_run_with_fixed_coefficients():
    pytest.importorskip("spacecoords")

    from spacecoords import frames
    from metablate.models.kero_szasz_2008 import (
        KeroSzasz2008,
        KeroSzaszInitialState,
        KeroSzaszOptions,
    )

    position_ecef = frames.geodetic_wgs84_to_ecef(67.89, 21.10, 150.0e3, degrees=True)
    velocity_ecef = np.array([0.0, 0.0, -40.0e3])
    options = KeroSzaszOptions(
        atmosphere=_ConstantAtmosphere(),
        max_time=0.02,
        max_step_size=0.01,
        start_altitude=150.0e3,
        sputtering=False,
        material=material.get("cometary", as_dict=False),
    )
    state = KeroSzaszInitialState(
        epoch=np.datetime64("2020-03-20T00:00:00"),
        position_ecef=position_ecef,
        velocity_ecef=velocity_ecef,
        mass=1.0e-6,
        drag_coefficient=0.5,
        heat_transfer_coefficient=0.1,
    )

    result = KeroSzasz2008(options).run(state)

    assert result.t[0] == pytest.approx(0.0)
    assert np.all(np.diff(result.t) > 0)
    assert np.all(np.diff(result.distance) > 0)
    assert np.all(result.mass > 0)
    assert result.mass[-1] <= result.mass[0]
    assert np.all(result.velocity > 0)
    assert np.all(np.isfinite(result.temperature))
    assert result.position_ecef.shape == (3, result.t.size)
    assert result.velocity_ecef.shape == (3, result.t.size)
