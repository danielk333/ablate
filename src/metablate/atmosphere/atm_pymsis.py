#!/usr/bin/env python

"""pymsis

"""
from typing import Union
import scipy.constants
from .atmosphere import Atmosphere
import numpy as np
import numpy.typing as npt
import xarray as xr

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
    def __init__(self):
        super().__init__(
            supported_species=["He", "O", "N2", "O2", "Ar", "H", "N"],
        )

    def density(
        self,
        time: npt.ArrayLike,
        lat: npt.ArrayLike,
        lon: npt.ArrayLike,
        alt: npt.ArrayLike,
        f107: npt.ArrayLike | None = None,
        f107s: npt.ArrayLike | None = None,
        aps: npt.ArrayLike | None = None,
        mass_densities: bool = False,
        version: Union[float, str] = 2.1,
        as_dataset: bool = True,
        **kwargs
    ) -> xr.Dataset | dict[str, xr.Variable]:
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

        result = msis.run(
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
        if len(result.shape) == 2:
            result = result.reshape(result.shape[0], 1, 1, 1, result.shape[1])

        variables = {
            "Total": (["time", "lon", "lat", "alt"], result[:, :, :, :, 0]),
            "N2": (["time", "lon", "lat", "alt"], result[:, :, :, :, 1]),
            "O2": (["time", "lon", "lat", "alt"], result[:, :, :, :, 2]),
            "O": (["time", "lon", "lat", "alt"], result[:, :, :, :, 3]),
            "He": (["time", "lon", "lat", "alt"], result[:, :, :, :, 4]),
            "H": (["time", "lon", "lat", "alt"], result[:, :, :, :, 5]),
            "Ar": (["time", "lon", "lat", "alt"], result[:, :, :, :, 6]),
            "N": (["time", "lon", "lat", "alt"], result[:, :, :, :, 7]),
            "AnomalousO": (["time", "lon", "lat", "alt"], result[:, :, :, :, 8]),
            "NO": (["time", "lon", "lat", "alt"], result[:, :, :, :, 9]),
            "Temperature": (["time", "lon", "lat", "alt"], result[:, :, :, :, 10]),
        }
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
