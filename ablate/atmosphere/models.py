#!/usr/bin/env python

'''A collection of implemented atmospheric models.

'''

#
# Basic Python
#
import pathlib

#
# External packages
#
import numpy as np

#
# Internal imports
#
from .base_model import AtmosphereModel


class NRLMSISE00(AtmosphereModel):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        path = kwargs['path']
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        if not path.exists():
            raise FileNotFoundError(str(path))

        if path.is_dir():
            #do glob?
            pass
        else:
            paths = [path]


        #here we load data into RAM

    @property
    def species(self):
        return {
            'He':{
                'A': 4.002602,
                'Z': 2,
            },
            'O':{
                'A': 15.999,
                'Z': 8,
            },
            'N2':{
                'A': 14.007*2,
                'Z': 14,
            },
            'O2':{
                'A': 15.999*2,
                'Z': 16,
            },
            'Ar':{
                'A': 39.948,
                'Z': 18,
            },
            'H':{
                'A': 1.0079,
                'Z': 1,
            },
            'N':{
                'A': 14.007,
                'Z': 7,
            },
        }

    
    def density(self, npt, species):
        raise NotImplementedError()


    def temperature(self, npt, species):
        raise NotImplementedError()
