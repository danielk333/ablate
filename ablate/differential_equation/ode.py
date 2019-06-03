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



class OrdinaryDifferentialEquation(ABC):

    def __init__(self, 
                method='RK45',
                options={},
                **kwargs
            ):
        self.method = method.upper()
        self.options = options


    def integrate(self, y0, t, **kwargs):

        t_sort = np.argsort(t)
        t_restore = np.argsort(t_sort)
        t = t[t_sort]

        t0 = t[0]
        t1 = t[-1]

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

        return self.result.t[t_restore], self.result.y[:, t_restore]


    @abstractmethod
    def rhs(self, t, y):
        pass