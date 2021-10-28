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

    ATMOSPHERES = None
    DEFAULT_OPTIONS = copy.deepcopy(AblationModel.DEFAULT_OPTIONS)
    DEFAULT_OPTIONS.update(dict(
        minimum_mass = 1e-11, #kg
        max_step_size = 1e-1, #s
        max_time = 100.0, #s
    ))

    def __init__(self,
                *args,
                method='RK45',
                method_options=None,
                **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.method = method.upper()
        self.method_options = method_options

    def integrate(self, state, *args, **kwargs):

        def _low_mass(t, y):
            res = y[0]/self.options['minimum_mass'] - 1
            #logger.debug(f'Stopping @ {t:<1.4e} s = {res}: {np.log10(y[0]):1.4e} log10(kg) | {y[0]:1.4e} kg')
            return res

        _low_mass.terminal = True
        _low_mass.direction = -1

        events = [_low_mass]

        logger.debug(f'{self.__class__} integrating IVP:\n- method: {self.method}\n- method-options: {self.method_options}')
        logger.debug('Options:\n' + '\n-- '.join([f'{key}: {val}' for key, val in self.options.items()]))

        ivp_kw = {}
        if not (self.method_options is None or len(self.method_options) == 0):
            ivp_kw['options'] = self.method_options

        self._ivp_result = solve_ivp(
            fun=lambda t, y: self.rhs(t, y[0], y[1:], *args, **kwargs),
            t_span = (0, self.options['max_time']),
            y0 = state,
            method = self.method,
            max_step = self.options['max_step_size'],
            dense_output = False,
            events = events,
            **ivp_kw
        )

        logger.debug(f'{self.__class__} IVP integration complete')

        return self._ivp_result


    @abstractmethod
    def rhs(self, t, m, y, *args, **kwargs):
        pass