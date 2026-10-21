"""pymsis

"""
from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import scipy.constants
import xarray as xr

from ..types import ArrayLike, AtmosphereData
from .atmosphere import Atmosphere

msis: ModuleType | None
try:
    from pymsis import msis
except ImportError:
    msis = None


class AtmPymsis(Atmosphere):
    """
    # TODO: docstring

    ndarray (ndates, nlons, nlats, nalts, 11) or (ndates, 11) | 
    The data calculated at each grid point: 
    | [Total mass density (kg/m3) | N2 # density (m-3), | O2 # density (m-3), | 
    O # density (m-3), | He # density (m-3), | H # density (m-3), | 
    Ar # density (m-3), | N # density (m-3), | 
    Anomalous oxygen # density (m-3), | NO # density (m-3), 
    | Temperature (K)]
    """
    def __init__(self) -> None:
        super().__init__(
            supported_species=["He", "O", "N2", "O2", "Ar", "H", "N"],
        )

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

        data = msis.run(
            dates=time,
            lons=lon,
            lats=lat,
            alts=alt * 1e-3,
            f107s=f107,
            f107as=f107s,
            aps=aps,
            version=version,
            **kwargs,
        )
        if len(data.shape) == 2:
            data = data.reshape(data.shape[0], 1, 1, 1, data.shape[1])

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
