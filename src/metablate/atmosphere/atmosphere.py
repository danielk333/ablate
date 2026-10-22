from __future__ import annotations

import logging
from abc import abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import constants

from ..types import ArrayLike, AtmosphereData, NDArray

logger = logging.getLogger(__name__)


@dataclass
class Species:
    symbol: str
    A: float
    Z: float


SPECIES = {
    "He": Species(
        symbol="He",
        A=4.002602,
        Z=2,
    ),
    "O": Species(
        symbol="O",
        A=15.999,
        Z=8,
    ),
    "N2": Species(
        symbol="N2",
        A=14.007 * 2,
        Z=14,
    ),
    "O2": Species(
        symbol="O2",
        A=15.999 * 2,
        Z=16,
    ),
    "Ar": Species(
        symbol="Ar",
        A=39.948,
        Z=18,
    ),
    "H": Species(
        symbol="H",
        A=1.0079,
        Z=1,
    ),
    "N": Species(
        symbol="N",
        A=14.007,
        Z=7,
    ),
}


# TODO: get a a hold of other atm models to figure out how to generalize interface better
class Atmosphere:

    def __init__(self, supported_species: list[Species] | list[str]) -> None:
        self.species = {}
        for s in supported_species:
            if isinstance(s, Species):
                self.species[s.symbol] = s
            else:
                self.species[s] = copy(SPECIES[s])

        self.mean_mass = np.array([x.A for _, x in self.species.items()]).mean() * constants.u  # [kg]

    @abstractmethod
    def density(
        self, time: ArrayLike, lat: ArrayLike, lon: ArrayLike, alt: ArrayLike, **kwargs: Any
    ) -> AtmosphereData:
        pass

    def total_density(
        self, time: ArrayLike, lat: ArrayLike, lon: ArrayLike, alt: ArrayLike, **kwargs: Any
    ) -> NDArray:
        """Return total mass density [kg/m^3] as an array, without species data.

        The array has the same shape as ``density(...)["Total"].values``.
        Subclasses can override this fallback to avoid building labeled data.
        """
        kwargs.pop("as_dataset", None)
        return np.asarray(
            self.density(time=time, lat=lat, lon=lon, alt=alt, **kwargs)["Total"].values
        )
