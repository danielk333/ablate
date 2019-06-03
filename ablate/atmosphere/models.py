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

    def get(self, npt, variable):
        #this will implement interplaltion and return results
        return 0.0
