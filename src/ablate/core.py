#!/usr/bin/env python

'''The basic class structures for implementing a ablation model.

'''

# Basic Python
from abc import ABC
from abc import abstractmethod
import logging

logger = logging.getLogger(__name__)

# External packages
import numpy as np


class AblationModel(ABC):

    DEFAULT_OPTIONS = {}
    ATMOSPHERES = {}

    def __init__(self, atmosphere, options = None, **kwargs):
        super().__init__()

        if atmosphere not in self.ATMOSPHERES:
            raise ValueError(f'"{atmosphere}" is not a supported by "{self.__class__}"')
        self.atmosphere = atmosphere
        
        self.results = None

        self.options = {}
        self.options.update(self.DEFAULT_OPTIONS)
        if options is not None:
            self.options.update(options)


    def _register_atmosphere(self, atmosphere, data_getter, meta):
        self.ATMOSPHERES[atmosphere] = (data_getter, meta)


    def _unregister_atmosphere(self, atmosphere, data_getter, meta):
        del self.ATMOSPHERES[atmosphere]


    @abstractmethod
    def run(self, *args, **kwargs):
        pass
