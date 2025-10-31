from abc import abstractmethod
from dataclasses import dataclass
import logging
from copy import copy

import scipy.constants as constants
import numpy as np
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


# TODO: get a a hold of other atm models to figure out how to generalize interface better
class Atmosphere:
    def __init__(self, supported_species):
        self.species = {}
        for s in supported_species:
            if isinstance(s, Species):
                self.species[s.symbol] = s
            else:
                self.species[s] = copy(SPECIES[s])

        self.mean_mass = np.array([x.A for _, x in self.species.items()]).mean() * constants.u  # [kg]

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
