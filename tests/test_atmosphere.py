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


@pytest.mark.parametrize("version", [0, 2.0, 2.1])
@pytest.mark.parametrize("mass_densities", [False, True])
@pytest.mark.parametrize("alt", [100e3, np.array([80e3, 100e3, 150e3])])
def test_atm_pymsis_variables_match_dataset(version, mass_densities, alt):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("pymsis")

    from metablate.atmosphere import AtmPymsis

    model = AtmPymsis()
    kwargs = dict(
        time=np.datetime64("2020-03-20T00:00:00"),
        lat=67.89,
        lon=21.10,
        alt=alt,
        f107=70.0,
        f107s=70.0,
        aps=[[4, 4, 4, 4, 4, 4, 4]],
        version=version,
        mass_densities=mass_densities,
        geomagnetic_activity=-1,
    )
    expected = model.density(**kwargs)
    actual = model.density(**kwargs, as_dataset=False)

    assert isinstance(expected, xr.Dataset)
    assert isinstance(actual, dict)
    assert actual.keys() == expected.data_vars.keys()
    for key, value in actual.items():
        assert isinstance(value, xr.Variable)
        assert value.dims == expected[key].dims
        np.testing.assert_array_equal(value.values, expected[key].values)
