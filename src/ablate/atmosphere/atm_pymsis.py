#!/usr/bin/env python

"""pymsis

"""
from typing import Union
import scipy.constants
from .atmosphere import Atmosphere
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
        f107: npt.ArrayLike = None,
        f107s: npt.ArrayLike = None,
        Ap: npt.ArrayLike = None,
        mass_densities: bool = True,
        version: Union[float, str] = 2.1,
        **kwargs
    ):
        """TODO: Write docstring

        returns density in [m^-3]

        """
        if msis is None:
            raise ImportError("pymsis is not installed")

        result = msis.run(
            dates=time,
            lons=lon,
            lats=lat,
            alts=alt * 1e-3,
            f107s=f107,
            f107as=f107s,
            aps=Ap,
            version=version,
            **kwargs,
        )
        if len(result.shape) == 2:
            result = result.reshape(result.shape[0], 1, 1, result.shape[1])

        result = xr.Dataset(
            {
                "Total": (["time", "lon", "lat", "alt"], result[:, :, :, 0]),
                "N2": (["time", "lon", "lat", "alt"], result[:, :, :, 1]),
                "O2": (["time", "lon", "lat", "alt"], result[:, :, :, 2]),
                "O": (["time", "lon", "lat", "alt"], result[:, :, :, 3]),
                "He": (["time", "lon", "lat", "alt"], result[:, :, :, 4]),
                "H": (["time", "lon", "lat", "alt"], result[:, :, :, 5]),
                "Ar": (["time", "lon", "lat", "alt"], result[:, :, :, 6]),
                "N": (["time", "lon", "lat", "alt"], result[:, :, :, 7]),
                "Anomalous_O": (["time", "lon", "lat", "alt"], result[:, :, :, 8]),
                "NO": (["time", "lon", "lat", "alt"], result[:, :, :, 9]),
                "Temperature": (["time", "lon", "lat", "alt"], result[:, :, :, 10]),
            },
            coords={
                "lon": lon,
                "lat": lat,
                "alt": alt,
                "time": time,
            },
            attrs=dict(
                f107 = f107,
                f107s = f107s,
                Ap = Ap,
                mass_densities = mass_densities,
                version = version,
            )
        )

        if mass_densities:
            for symbol, s in self.species.items():
                weight = scipy.constants.u * s.A
                result[symbol] = result[symbol] * weight

        return result
