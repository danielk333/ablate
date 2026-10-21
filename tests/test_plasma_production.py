import numpy as np
import pytest

from metablate import material
from metablate.physics.plasma_production import ionization_probability_Jones_1997


@pytest.mark.parametrize("velocity", [-1.0, np.nan, np.inf, [20000.0, -1.0]])
def test_jones_rejects_invalid_speed(velocity):
    with pytest.raises(ValueError, match="Velocity"):
        ionization_probability_Jones_1997(velocity, material.iron.atomic_composition)


@pytest.mark.parametrize(
    "composition", [None, {}, {"Fe": 0.0}, {"O": 1.0, "Fe": -0.1}, {"Fe": np.nan}, {"Fe": np.inf}]
)
def test_jones_rejects_missing_or_invalid_composition(composition):
    with pytest.raises(ValueError, match="composition|abundances"):
        ionization_probability_Jones_1997(20000.0, composition)


def test_jones_rejects_elements_without_parameters():
    composition = {"Na": 1.0}
    with pytest.raises(ValueError, match="No Jones.*Na"):
        ionization_probability_Jones_1997(20000.0, composition)
