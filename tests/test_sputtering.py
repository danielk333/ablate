import numpy as np

from metablate import material
from metablate.physics.sputtering import sputtering_kero_szasz_2008


class _Values:
    def __init__(self, value):
        self.values = np.asarray(value, dtype=float)


class _Density(dict):
    pass


def test_sputtering_returns_negative_finite_massloss_for_fast_meteor():
    density = _Density({"N2": _Values(1.0e20), "O2": _Values(5.0e19)})

    dmdt = sputtering_kero_szasz_2008(
        mass=1.0e-6,
        velocity=60.0e3,
        material_data=material.get("cometary", as_dict=False),
        density=density,
    )

    assert np.isfinite(dmdt)
    assert dmdt < 0
