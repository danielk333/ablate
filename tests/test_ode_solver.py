#!/usr/bin/env python

# Basic Python
import unittest

# External packages
import numpy as np
import numpy.testing as nt


# Internal packages
import metablate


class TestScipyODESolve(unittest.TestCase):

    def test_subclass_ABC(self):
        with self.assertRaises(TypeError):

            class TestODE(ablate.ScipyODESolve):
                pass

            inst = TestODE()

    def test_class_ABC(self):
        with self.assertRaises(TypeError):
            inst = ablate.ScipyODESolve()
