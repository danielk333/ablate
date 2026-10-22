"""Compare the direct backend with the upstream Python wrapper as an oracle."""

import numpy as np
import pytest
from pymsis import msis

from metablate.atmosphere import AtmPymsis

FIXED = dict(f107=70.0, f107s=80.0, aps=[[4, 5, 6, 7, 8, 9, 10]])
POINT = dict(time=np.datetime64("2020-03-20T12:34:56.123456"), lat=67.89, lon=21.1, alt=100e3)
NAMES = ["Total", "N2", "O2", "O", "He", "H", "Ar", "N", "AnomalousO", "NO", "Temperature"]


def reference(time, lat, lon, alt, f107=None, f107s=None, aps=None, **kwargs):
    data = msis.run(
        dates=time,
        lons=lon,
        lats=lat,
        alts=np.asarray(alt) * 1e-3,
        f107s=f107,
        f107as=f107s,
        aps=aps,
        **kwargs,
    )
    if data.ndim == 2:
        data = data.reshape(data.shape[0], 1, 1, 1, 11)
    return data


@pytest.mark.parametrize("version", [0, 2.0, 2.1])
@pytest.mark.parametrize("mode", ["scalar", "grid", "flythrough"])
@pytest.mark.parametrize("historical", [False, True])
def test_direct_matches_original_all_variables(version, mode, historical):
    kwargs = dict(POINT, version=version, geomagnetic_activity=-1)
    if not historical:
        kwargs.update(FIXED)
    if mode != "scalar":
        kwargs["time"] = np.array([POINT["time"], POINT["time"] + np.timedelta64(1, "h")])
        kwargs["alt"] = np.array([80e3, 100e3, 120e3])
    if mode == "flythrough":
        kwargs.update(lat=[67.0, 68.0], lon=[20.0, 21.0], alt=[90e3, 100e3])
    if not historical and mode != "scalar":
        kwargs.update(f107=[70.0, 71.0], f107s=[80.0, 81.0], aps=np.tile(FIXED["aps"], (2, 1)))
    expected = reference(**kwargs)
    model = AtmPymsis()
    np.testing.assert_array_equal(model.total_density(**kwargs), expected[..., 0])
    actual = model.density(**kwargs, as_dataset=False)
    for index, name in enumerate(NAMES):
        np.testing.assert_array_equal(actual[name].values, expected[..., index])


@pytest.mark.parametrize(
    "update",
    [
        {"alt": np.nan},
        {"lat": np.inf},
        {"time": np.datetime64("NaT")},
        {"time": np.array([np.datetime64("NaT"), POINT["time"]])},
        {"options": [1.0]},
        {"version": 3},
        {"f107": [70, 80]},
    ],
)
def test_invalid_inputs_are_rejected_before_fortran(update):
    with pytest.raises(ValueError):
        AtmPymsis().total_density(**{**POINT, **FIXED, **update})

