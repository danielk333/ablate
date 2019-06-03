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
import ablate


class TestParameters(unittest.TestCase):

    def test_init(self):

        p = ablate.Parameters()

        print(p)
        assert False