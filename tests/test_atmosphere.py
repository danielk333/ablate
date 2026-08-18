import numpy as np
import pytest


def test_atm_pymsis_fixed_density_call_has_expected_variables():
    pytest.importorskip("xarray")
    pytest.importorskip("pymsis")

    from metablate.atmosphere import AtmPymsis

    model = AtmPymsis()
    result = model.density(
        time=np.datetime64("2020-03-20T00:00:00"),
        lat=np.array([67.89]),
        lon=np.array([21.10]),
        alt=np.array([100.0e3]),
        f107=70.0,
        f107s=70.0,
        aps=[[4, 4, 4, 4, 4, 4, 4]],
    )

    for key in ["Total", "N2", "O2", "O", "He", "H", "Ar", "N", "Temperature"]:
        assert key in result
        assert np.all(np.isfinite(result[key].values))
    assert float(result["Total"].values.squeeze()) > 0
    assert float(result["Temperature"].values.squeeze()) > 0


def test_atm_msise00_initializes_when_dependency_is_installed():
    pytest.importorskip("xarray")
    pytest.importorskip("msise00")

    from metablate.atmosphere import AtmMSISE00

    model = AtmMSISE00()

    assert set(model.species) == {"He", "O", "N2", "O2", "Ar", "H", "N"}
    assert model.mean_mass > 0
