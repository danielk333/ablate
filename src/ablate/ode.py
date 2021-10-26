#!/usr/bin/env python

'''A class structure for solving ODE's

'''


# Basic Python
from abc import ABC
from abc import abstractmethod
import copy
import logging

logger = logging.getLogger(__name__)


# External packages
import scipy.integrate
from scipy.integrate import solve_ivp
import numpy as np

# Internal imports
from .core import AblationModel

class ScipyODESolve(AblationModel):

    DEFAULT_OPTIONS = copy.deepcopy(AblationModel.DEFAULT_OPTIONS)
    DEFAULT_OPTIONS.update(dict(
        minimum_mass = 1e-11, #kg
        max_step_size = 1e-3, #s
        max_time = 100.0, #s
    ))

    def __init__(self,
                *args,
                method='RK45',
                method_options={},
                **kwargs
            ):
        super().__init__(self, *args, **kwargs)
        self.method = method.upper()
        self.method_options = method_options


    def integrate(self, state, *args, **kwargs):

        def _low_mass(t, y):
            logger.info(f'Stopping @ {t:<1.4e} s: {np.log10(y[1]):1.4e} log10(kg)')
            return np.log10(y[1]) - np.log10(self.options['minimum_mass'])

        _low_mass.terminal = True
        _low_mass.direction = -1

        events = [_low_mass]

        self._ivp_result = solve_ivp(
            fun=lambda t, y: self.rhs(t, y, *args, **kwargs),
            t_span2 = (0, self.options['max_time']),
            y0 = state,
            method = self.method,
            max_step = self.options['max_step_size'],
            dense_output = True,
            events = events,
            options = self.method_options,
        )

        return self._ivp_result


    @abstractmethod
    def rhs(self, t, y, *args, **kwargs):
        pass