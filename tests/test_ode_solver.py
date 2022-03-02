#!/usr/bin/env python

# Basic Python
import unittest

# External packages
import numpy as np
import numpy.testing as nt


# Internal packages
import ablate


class TestScipyODESolve(unittest.TestCase):

    def test_subclass_ABC(self):
        with self.assertRaises(TypeError):
            class TestODE(ablate.ScipyODESolve):
                pass
            inst = TestODE()

    def test_class_ABC(self):
        with self.assertRaises(TypeError):
            inst = ablate.ScipyODESolve()

    def test_subclass_init(self):
        class TestODE(ablate.ScipyODESolve):
            ATMOSPHERES = {}
            
            def run(self, y0):
                return self.integrate(y0)
        
            def rhs(self, t, m, y):
                return -m
        TestODE._register_atmosphere('my_atm', lambda x: None, {'my_meta':None})

        ode = TestODE('my_atm')
