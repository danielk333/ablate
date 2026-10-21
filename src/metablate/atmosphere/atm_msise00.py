"""msis00

"""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np
import scipy.constants
import xarray as xr

from ..types import ArrayLike
from .atmosphere import Atmosphere

msise00: ModuleType | None
try:
    msise00 = import_module("msise00")
except ImportError:
    msise00 = None


class AtmMSISE00(Atmosphere):

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
        f107: float = 80.0,
        f107s: float = 80.0,
        Ap: float = 4.0,
        mass_densities: bool = False,
        **kwargs: Any,
    ) -> xr.Dataset:
        """TODO: Write docstring

        returns density in [m^-3]

        """
        if msise00 is None:
            raise ImportError("msis00 is not installed")

        result = msise00.run(
            time=time,
            altkm=np.asarray(alt) * 1e-3,
            glat=lat,
            glon=lon,
            indices=dict(
                f107=f107,
                f107s=f107s,
                Ap=Ap,
            ),
            **kwargs,
        )
        result["alt_km"] = result["alt_km"]*1e3
        result = result.rename({
            "alt_km": "alt",
            "Tn": "Temperature",
        })
        result = result.transpose("time", "lon", "lat", "alt")
        result.attrs["mass_densities"] = mass_densities

        if mass_densities:
            for symbol, s in self.species.items():
                weight = scipy.constants.u * s.A
                result[symbol] = result[symbol] * weight

        return result
