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


    def test_add_params(self):
        p = ablate.Parameters()
        p.add_parameters(
            x = {
                'dtype': 'float64',
                'type': 'variable',
            },
            c = {
                'value': 3.2,
                'dtype': 'float64',
                'type': 'constant',
            },
            k = {
                'type': 'constant',
            },
            t = {
                'type': 'dependence',
            },
        )


    def test_fill_params(self):
        p = ablate.Parameters()
        p.add_parameters(
            x = {
                'dtype': 'float64',
                'type': 'variable',
            },
            c = {
                'value': 3.2,
                'dtype': 'float64',
                'type': 'constant',
            },
            k = {
                'type': 'constant',
            },
            t = {
                'type': 'dependence',
            },
        )

        print('set k')
        print(p['k'])
        p['k'] = 11.4
        print(p['k'])

        print('set t')
        print(p['t'])
        p['t'] = np.linspace(0,1,num=10)
        print(p['t'])

        print('set x')
        print(p['x'])
        p['x'] = p['t']**2
        print(p['x'])

        print(p)

    def test_fill_params_many(self):
        p = ablate.Parameters()
        p.add_parameters(
            x = {'type': 'variable'},
            y = {'type': 'variable'},
            c = {
                'value': 3.2,
                'type': 'constant',
            },
            t = {'type': 'dependence'},
            m = {'type': 'dependence'},
        )

        print(p)
        print(p.data)

        p['t'] = np.linspace(0,1,num=4)
        p['m'] = np.linspace(10,55,num=4)

        print(p)
        
        print('set x')
        p['x'] = np.random.randn(4,4)

        print('x')
        print(p.data['x'])

        print('y')
        print(p.data['y'])
