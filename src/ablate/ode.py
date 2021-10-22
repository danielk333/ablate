#!/usr/bin/env python

'''A class structure for solving ODE's

'''

#
# Basic Python
#
from abc import ABC
from abc import abstractmethod

#
# External packages
#
import scipy.integrate
from scipy.integrate import solve_ivp
import numpy as np

# Internal imports
from .core import AblationModel

class ScipyODESolve(AblationModel):

    def __init__(self, 
                atmosphere,
                method='RK45',
                options={},
                **kwargs
            ):
        super().__init__(self, atmosphere, **kwargs)
        self.method = method.upper()
        self.options = options


    def integrate(self, y0, dt, **kwargs):

        #todo; fix this
        self.result = solve_ivp(
            fun=lambda t, y: self.rhs(t, y),
            t_span = (t0, t1),
            y0 = y0,
            method = self.method,
            dense_output = True,
            t_eval = t,
            options = self.options,
            **kwargs
        )

        return self.result.y

    def run(self, state, dt, **kwargs):
        return self.integrate(state, dt, **kwargs)

    @abstractmethod
    def rhs(self, t, y):
        pass