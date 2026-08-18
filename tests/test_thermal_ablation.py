import pytest
from scipy import constants

from metablate import material
from metablate.physics import thermal_ablation


def test_thermal_accommodation_matches_published_iron_value():
    # Silber et al. (2018), Sect. 2.1, state an accommodation coefficient
    # of 0.75 for iron. https://doi.org/10.1016/j.asr.2018.05.010
    atm_mean_mass = 28.0 * constants.u
    met_mean_mass = 56.0 * constants.u

    assert thermal_ablation.thermal_accommodation_bronshten_1983(
        atm_mean_mass, met_mean_mass
    ) == pytest.approx(0.75, abs=0.05)


def test_solve_temperature_roundtrips_thermal_ablation():
    mat = material.get("cometary", as_dict=False)
    mass = 1.0e-6
    shape_factor = 1.21
    temperature = 2400.0
    massloss = thermal_ablation.thermal_ablation_hill_et_al_2005(mass, temperature, mat, shape_factor)

    solved = thermal_ablation.solve_temperature_from_thermal_ablation(
        massloss,
        mass,
        mat,
        shape_factor,
        minimize_kwargs={"method": "bounded", "options": {"xatol": 1e-7}},
    )

    assert solved == pytest.approx(temperature, rel=1e-4)
