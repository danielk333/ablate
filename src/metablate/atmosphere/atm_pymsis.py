"""pymsis"""

from __future__ import annotations

from threading import Lock
from types import ModuleType
from typing import Any

import numpy as np
import scipy.constants
import xarray as xr

from ..types import ArrayLike, AtmosphereData, NDArray
from .atmosphere import Atmosphere

msis: ModuleType | None
try:
    from pymsis import msis
except ImportError:
    msis = None

_input_lock = Lock()


class AtmPymsis(Atmosphere):
    """MSIS atmosphere evaluated through the Fortran extensions shipped by PyMSIS.

    ``density`` returns labeled species densities and temperature;
    ``total_density`` returns only a NumPy mass-density array. Both use
    (time, longitude, latitude, altitude) grid axes and MSIS 2.1 by default.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_species=["He", "O", "N2", "O2", "Ar", "H", "N"],
        )
        self._input = np.empty((1, 14), dtype=np.float32)
        self._input_time = np.datetime64("NaT")
        self._weather_time = np.datetime64("NaT")
        self._weather: tuple[NDArray, NDArray, NDArray] | None = None
        self._default_options = msis.create_options() if msis is not None else []

    def _evaluate(
        self,
        time: ArrayLike,
        lat: ArrayLike,
        lon: ArrayLike,
        alt: ArrayLike,
        f107: ArrayLike | None = None,
        f107s: ArrayLike | None = None,
        aps: ArrayLike | None = None,
        version: float | str = 2.1,
        options: list[float] | None = None,
        interpolate_indices: bool = False,
        **kwargs: Any,
    ) -> NDArray:
        """Call the PyMSIS Fortran extension; prepare scalar inputs in place."""
        if msis is None:
            raise ImportError("pymsis is not installed")
        libraries = {
            "0": msis.msis00f,
            "00": msis.msis00f,
            "2.0": msis.msis20f,
            "2": msis.msis21f,
            "2.1": msis.msis21f,
        }
        if str(version) not in libraries:
            raise ValueError(f"Unsupported MSIS version: {version}")
        library = libraries[str(version)]
        if options is None:
            options = msis.create_options(**kwargs) if kwargs else self._default_options
        elif len(options) != 25:
            raise ValueError("MSIS options must contain 25 values.")
        options = list(options)

        time, lat, lon, alt = (np.atleast_1d(value) for value in (time, lat, lon, alt))
        # The scalar buffer and weather cache belong to this instance. The
        # extension lock below is also shared with ordinary PyMSIS calls.
        with _input_lock:
            if all(value.size == 1 for value in (time, lat, lon, alt)) and not interpolate_indices:
                date = np.datetime64(time.flat[0], "s")
                if np.isnat(date):
                    raise ValueError("MSIS time must not be NaT.")
                data = self._input
                if date != self._input_time:
                    day = date.astype("datetime64[D]")
                    year = date.astype("datetime64[Y]")
                    data[0, 0] = (day - year).astype("timedelta64[D]").astype(float) + 1
                    data[0, 1] = (date - day).astype("timedelta64[s]").astype(float)
                    self._input_time = date
                data[0, 2:5] = lon.flat[0], lat.flat[0], alt.flat[0] * 1e-3
                if f107 is None or f107s is None or aps is None:
                    # Historical F10.7 is daily and Ap is three-hourly.
                    weather_time = date.astype("datetime64[3h]")
                    if self._weather is None or weather_time != self._weather_time:
                        self._weather = msis.get_f107_ap(np.array([date]))
                        self._weather_time = weather_time
                    solar, average, activity = self._weather
                    if f107 is None:
                        f107 = solar
                    if f107s is None:
                        f107s = average
                    if aps is None:
                        aps = activity
                solar = np.atleast_1d(f107)
                average = np.atleast_1d(f107s)
                activity = np.atleast_1d(aps)
                if solar.size != 1 or average.size != 1 or len(activity) != 1:
                    raise ValueError("Space-weather inputs must match the number of dates.")
                data[0, 5] = solar.item()
                data[0, 6] = average.item()
                data[0, 7:] = activity[0]
                shape: tuple[int, ...] = (1,)
            else:
                # Reuse upstream gridding and optional index interpolation.
                if np.any(np.isnat(np.asarray(time, dtype="datetime64[s]"))):
                    raise ValueError("MSIS time must not be NaT.")
                input_kwargs = {"interpolate_indices": True} if interpolate_indices else {}
                shape, data = msis.create_input(
                    time, lon, lat, alt * 1e-3, f107, f107s, aps, **input_kwargs
                )
            if not np.all(np.isfinite(data)):
                raise ValueError("MSIS inputs must be finite.")
            # PyMSIS 0.11/0.12 use per-extension locks; 0.13 uses one global lock.
            lock = getattr(msis, "_lock", None)
            if lock is None:
                lock = library._lock
            with lock:
                if library._last_used_options != options:
                    library.pyinitswitch(options, parmpath=msis._MSIS_PARAMETER_PATH)
                    library._last_used_options = options.copy()
                output = library.pymsiscalc(
                    data[:, 0],
                    data[:, 1],
                    data[:, 2],
                    data[:, 3],
                    data[:, 4],
                    data[:, 5],
                    data[:, 6],
                    data[:, 7:],
                )
        # Older Fortran builds encode missing species as values near 9.9e-38.
        output[(output >= 9.9e-38) & (output < 1e-37)] = np.nan
        if len(shape) == 1:
            shape = (shape[0], 1, 1, 1)
        return output.reshape(*shape, 11)

    def total_density(
        self, time: ArrayLike, lat: ArrayLike, lon: ArrayLike, alt: ArrayLike, **kwargs: Any
    ) -> NDArray:
        """Return total mass density [kg/m^3], with the full method's array shape.

        Accept the same inputs as ``density``. Output-format and species-unit
        flags do not affect total mass density. No xarray objects are created.
        """
        kwargs.pop("as_dataset", None)
        kwargs.pop("mass_densities", None)
        return self._evaluate(time, lat, lon, alt, **kwargs)[..., 0]

    def density(
        self,
        time: ArrayLike,
        lat: ArrayLike,
        lon: ArrayLike,
        alt: ArrayLike,
        f107: ArrayLike | None = None,
        f107s: ArrayLike | None = None,
        aps: ArrayLike | None = None,
        mass_densities: bool = False,
        version: float | str = 2.1,
        as_dataset: bool = True,
        **kwargs: Any,
    ) -> AtmosphereData:
        """Return species number densities [m^-3], or mass densities [kg/m^3].

        Set ``as_dataset=False`` in an ablation model's ``atmosphere_kwargs``
        to skip dataset and coordinate construction at each solver step.
        The returned variables still support ``.values``; total mass density
        and temperature retain their units of kg/m^3 and K, respectively.
        """
        if msis is None:
            raise ImportError("pymsis is not installed")

        time = np.array(time) if not isinstance(time, np.ndarray) else time
        lon = np.array(lon) if not isinstance(lon, np.ndarray) else lon
        lat = np.array(lat) if not isinstance(lat, np.ndarray) else lat
        alt = np.array(alt) if not isinstance(alt, np.ndarray) else alt

        for inp in [time, lon, lat, alt]:
            if len(inp.shape) == 0:
                inp.shape = (1,)

        data = self._evaluate(
            time=time,
            lat=lat,
            lon=lon,
            alt=alt,
            f107=f107,
            f107s=f107s,
            aps=aps,
            version=version,
            **kwargs,
        )
        variables = {
            "Total": (["time", "lon", "lat", "alt"], data[:, :, :, :, 0]),
            "N2": (["time", "lon", "lat", "alt"], data[:, :, :, :, 1]),
            "O2": (["time", "lon", "lat", "alt"], data[:, :, :, :, 2]),
            "O": (["time", "lon", "lat", "alt"], data[:, :, :, :, 3]),
            "He": (["time", "lon", "lat", "alt"], data[:, :, :, :, 4]),
            "H": (["time", "lon", "lat", "alt"], data[:, :, :, :, 5]),
            "Ar": (["time", "lon", "lat", "alt"], data[:, :, :, :, 6]),
            "N": (["time", "lon", "lat", "alt"], data[:, :, :, :, 7]),
            "AnomalousO": (["time", "lon", "lat", "alt"], data[:, :, :, :, 8]),
            "NO": (["time", "lon", "lat", "alt"], data[:, :, :, :, 9]),
            "Temperature": (["time", "lon", "lat", "alt"], data[:, :, :, :, 10]),
        }
        result: AtmosphereData
        if as_dataset:
            result = xr.Dataset(
                variables,
                coords={
                    "time": time,
                    "lon": lon,
                    "lat": lat,
                    "alt": alt,
                },
                attrs=dict(
                    f107=f107,
                    f107s=f107s,
                    aps=aps,
                    mass_densities=mass_densities,
                    version=version,
                ),
            )
        else:
            result = {key: xr.Variable(*value) for key, value in variables.items()}

        if mass_densities:
            for symbol, s in self.species.items():
                weight = scipy.constants.u * s.A
                result[symbol] = result[symbol] * weight

        return result
