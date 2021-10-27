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
    ATMOSPHERES = None

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

    def get_atmosphere(self, *args, **kwargs):
        logger.debug(f'{self.__class__} getting atmosphere {self.atmosphere} data')
        _getter = self.ATMOSPHERES[self.atmosphere][0]
        data = _getter(*args, **kwargs)
        return data


    def get_atmosphere_meta(self, *args, **kwargs):
        logger.debug(f'{self.__class__} getting atmosphere {self.atmosphere} meta')
        _getter = self.ATMOSPHERES[self.atmosphere][1]
        meta = _getter(*args, **kwargs)
        return meta


    @classmethod
    def _register_atmosphere(cls, atmosphere, data_getter, meta):
        cls.ATMOSPHERES[atmosphere] = (data_getter, meta)

    @classmethod
    def _unregister_atmosphere(cls, atmosphere):
        del cls.ATMOSPHERES[atmosphere]


    @abstractmethod
    def run(self, *args, **kwargs):
        pass
