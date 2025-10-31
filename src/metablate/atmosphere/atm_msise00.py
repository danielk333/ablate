#!/usr/bin/env python

"""msis00

"""
import scipy.constants
from .atmosphere import Atmosphere
import numpy.typing as npt

try:
    import msise00
except ImportError:
    msise00 = None


class AtmMSISE00(Atmosphere):
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
        f107: float = 80.0,
        f107s: float = 80.0,
        Ap: float = 4.0,
        mass_densities: bool = False,
        **kwargs
    ):
        """TODO: Write docstring

        returns density in [m^-3]

        """
        if msise00 is None:
            raise ImportError("msis00 is not installed")

        result = msise00.run(
            time=time,
            altkm=alt * 1e-3,
            glat=lat,
            glon=lon,
            indices=dict(
                f107=f107,
                f107s=f107s,
                Ap=Ap,
            ),
            **kwargs
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
