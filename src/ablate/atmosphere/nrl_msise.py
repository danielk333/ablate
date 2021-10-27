#!/usr/bin/env python

'''A collection of implemented atmospheric models.

'''


# Basic Python
import pathlib
from datetime import datetime

# External packages
import xarray
import scipy.constants

try:
    import msise00
except ImportError:
    msise00 = None



class NRLMSISE00:
    
    def __init__(self):
        self.species = {
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
    
    def density(self, time, lat, lon, alt, f107=80.0, f107s=80.0, Ap=4.0, mass_densities=True):
        ''' TODO: Write docstring

        returns density in [m^-3]

        '''
        result = msise00.run(
            time=time,
            altkm=alt*1e-3,
            glat=lat,
            glon=lon,
            indices=dict(
                f107=f107,
                f107s=f107s,
                Ap=Ap,
            ),
        )

        if mass_densities:
            for s in self.species:
                weight = scipy.constants.u*self.species[s]['A']
                result[s] = result[s]*weight

        return result

