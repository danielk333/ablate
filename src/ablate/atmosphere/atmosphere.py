from abc import abstractmethod
from dataclasses import dataclass
import logging
from copy import copy

import numpy.typing as npt

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


class Atmosphere:
    def __init__(self, supported_species):
        self.species = {}
        for s in supported_species:
            if isinstance(s, Species):
                self.species[s.symbol] = s
            else:
                self.species[s] = copy(SPECIES[s])

    @abstractmethod
    def density(
        self,
        time: npt.ArrayLike,
        lat: npt.ArrayLike,
        lon: npt.ArrayLike,
        alt: npt.ArrayLike,
        **kwargs
    ):
        pass
