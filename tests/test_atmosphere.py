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
import ablate.atmosphere as atm


class TestAtmosphericModel(unittest.TestCase):
    
    def test_base_class_raise(self):
        with self.assertRaises(TypeError):
            model = atm.AtmosphereModel()


    def test_subclass_raises(self):
        with self.assertRaises(TypeError):
            class TestSubClass(atm.AtmosphereModel):
                pass
            model = TestSubClass()


    def test_subclass(self):

        class TestSubClass(atm.AtmosphereModel):    
            @property
            def species(self):
                pass

            def density(self, npt, species, **kwargs):
                pass

        model = TestSubClass()




class TestNRLMSISE00(unittest.TestCase):

    def test_init(self):
        model = atm.NRLMSISE00()
    

    def test_species(self):
        model = atm.NRLMSISE00()
        species = model.species

        for key in ['O', 'N2', 'O2', 'He', 'Ar', 'H', 'N']:
            assert key in species


    def test_density(self):
        model = atm.NRLMSISE00()
        species = ['O', 'N2', 'O2']
        data = model.density(
            npt = np.datetime64('2018-07-28', 's'),
            species = species,
            lat = 69,
            lon = 12,
            alt = 89e3,
        )

        self.assertSetEqual(set(species + ['date']), set(data.dtype.names))

        assert data.size == 1