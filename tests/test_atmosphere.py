#!/usr/bin/env python

#
# Basic Python
#
import unittest

# External packages
import numpy as np
import numpy.testing as nt

# Internal packages
import metablate.atmosphere as atm


class TestNRLMSISE00(unittest.TestCase):

    def test_msise00_package(self):
        import msise00

    def test_init(self):
        model = atm.NRLMSISE00()

