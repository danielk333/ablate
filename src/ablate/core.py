#!/usr/bin/env python

'''The basic class structures for implementing a ablation model.

'''

# Basic Python
from abc import ABC
from abc import abstractmethod

# External packages
import numpy as np


# Internal packages
from .atmosphere import AtmosphereModel



class AblationModel(ABC):
    def __init__(self, atmosphere, **kwargs):
        super().__init__()
        if not isinstance(atmosphere, AtmosphereModel):
            raise ValueError(f'"atmosphere" is not a AtmosphereModel instance but "{atmosphere!r}"')
        self.atmosphere = atmosphere

    @abstractmethod
    def run(self, state, dt, **kwargs):
        pass
