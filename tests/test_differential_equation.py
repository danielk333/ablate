#!/usr/bin/env python

#
# Basic Python
#
import unittest

#
# External packages
#
import numpy as np
import numpy.testing as nt

#
# Internal packages
#
from ablate.differential_equation import OrdinaryDifferentialEquation


class TestIntegrate(unittest.TestCase):

    def test_child_class(self):
        class TestODE(OrdinaryDifferentialEquation):
            def rhs(self, t, y):
                pass

    def test_child_class_ABC(self):
        with self.assertRaises(TypeError):
            class TestODE(OrdinaryDifferentialEquation):
                pass
            inst = TestODE()

    def test_class_init_ABC(self):
        with self.assertRaises(TypeError):
            inst = OrdinaryDifferentialEquation()


    def test_child_class_init(self):
        class TestODE(OrdinaryDifferentialEquation):
            def rhs(self, t, y):
                '''exponential decay'''
                return -0.5 * y

        ode = TestODE()

    def test_child_class_integrate(self):
        class TestODE(OrdinaryDifferentialEquation):
            def rhs(self, t, y):
                '''exponential decay'''
                return -0.5 * y

        ode = TestODE()

        t = np.linspace(0.0, 10.0, num=100)
        y = ode.integrate(
            y0 = np.array([2.0]), 
            t = t,
        )

        assert y.shape[1] == len(t)
        assert y.shape[0] == 1