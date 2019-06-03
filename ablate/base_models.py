#!/usr/bin/env python

'''The basic class structures for implementing a ablation model.

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


#
# Internal packages
#
from .atmosphere import AtmosphereModel


class Parameters:

    _default_dtype = 'float64'

    def __init__(self):
        self.clear()


    @property
    def names(self):
        return self.variables + self.constants + self.dependant


    def clear(self):
        self.data = None
        self._constants = {}
        self._dependant = {}

        self.variables = []
        self.constants = []
        self.dependant = []
        
        self._v_dtype = []
        self._c_dtype = []
        self._d_dtype = []


    def add_parameters(self, **kwargs):

        for key, vals in kwargs.items():
            if key in self.names:
                raise ValueError(f'Parameter "{key}" already a parameter')
                #TODO add delete parameter method and update parameter here instead of raise

            dtype = vals.get('dtype', Parameters._default_dtype)
            _type = vals.get('type', 'variable').lower()
            if _type == 'variable':
                self._v_dtype.append((key, dtype))
                self.variables.append(key)
            elif _type == 'constant':
                self._c_dtype.append((key, dtype))
                self.constants.append(key)
            elif _type == 'dependence':
                self.dependant.append(key)
                self._d_dtype.append((key, dtype))
                self._dependant[key] = np.empty((0,), dtype=dtype)
            else:
                raise TypeError(f'Parameter must be variable or constant, not {_type}')

        for var, dt in _c_dtype:
            _tmp = np.empty((1,), dtype=dt)
            _tmp[0] = kwargs[var]['value']
            self._constants[var] = _tmp[0]

        self._allocate()

    def _allocate(self):
        shape = [len(x) for _, x in self.dependant.items()]
        self.data = np.empty(shape, dtype=self._v_dtype)


    def __setitem__(self, key, val):
        if isinstance(key, str):
            if key in self.constants:
                self._constants[key] = val
            elif key in self.variables:
                self.data[key] = val
            elif key in self.dependant:
                if not isinstance(val, np.ndarray):
                    val = np.array(val)
                self._dependant[key] = val
                self._allocate()
            else:
                raise KeyError(f'Key "{key}" is not a parameter')
        else:
            self.data[key] = val


    def __getitem__(self, key):
        if key in self.constants:
            return self._constants[key]
        elif key in self.variables:
            return self.data[key]
        elif key in self.dependant:
            return self._dependant[key]
        else:
            raise KeyError(f'Key "{key}" is not a parameter')


class AblationModel(Parameters, ABC):
    def __init__(self, atmosphere, **kwargs):
        super().__init__()
        if not isinstance(atmosphere, AtmosphereModel):
            raise ValueError(f'"atmosphere" is not a AtmosphereModel instance but "{atmosphere!r}"')


    @abstractmethod
    def run(self, **kwargs):
        pass



class Persistent(ABC):
    '''A abstract handler for saving and loading ablation results.
    '''

    def __init__(self, parameters):
        if not isinstance(parameters, Parameters):
            raise ValueError(f'"parameters" is not a Parameters instance but "{parameters!r}"')

        self._parameters = parameters #save pointer


    @abstractmethod
    def load(self, **kwargs):
        pass


    @abstractmethod
    def save(self, **kwargs):
        pass

