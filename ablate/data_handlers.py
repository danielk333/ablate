#!/usr/bin/env python

'''Different data handler implementations

'''

#
# Basic Python
#
import pathlib
from abc import ABC
from abc import abstractmethod


#
# External packages
#
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

#
# Internal packages
#
from .base_models import Parameters
from .base_models import Persistent


class hdf5_handler(Persistent):

    def __init__(self, parameters, *kwargs):
        super().__init(parameters)


    @staticmethod
    def _check_path(path):
        if isinstance(path, str):
            path = pathlib.Path(path)
        elif not isinstance(path, pathlib.Path):
            raise ValueError(f'Path type "{type(path)}" not supported')

        return path


    def load(self, path):
        path = hdf5_handler._check_path(path)


    def save(self, path):
        path = hdf5_handler._check_path(path)

        parameters = self.parameters

        with h5py.File(path,'w') as hf:
            parameters.data = None
            parameters._constants = {}
            parameters._dependant = {}

            parameters.variables = []
            parameters.constants = []
            parameters.dependant = []
            
            parameters._v_dtype = []
            parameters._c_dtype = []
            parameters._d_dtype = []
