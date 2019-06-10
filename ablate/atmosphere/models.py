#!/usr/bin/env python

'''A collection of implemented atmospheric models.

'''

#
# Basic Python
#
import pathlib
from datetime import datetime

#
# External packages
#
import numpy as np
try:
    import msise00
except ImportError:
    msise00 = None


#
# Internal imports
#
from .base_model import AtmosphereModel
from . import util


class NRLMSISE00(AtmosphereModel):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    
    def density(self, npt, species, **kwargs):
        ''' TODO: Write docstring

        returns density in [m^-3]

        '''

        glat = float(kwargs['lat'])
        glon = float(kwargs['lon'])
        alt = float(kwargs['alt'])
        
        if not isinstance(npt, np.ndarray):
            _npt = np.array([npt], dtype=np.datetime64)
        else:
            _npt = npt
        
        data_all = []
        for _time in _npt:
            data_all += [msise00.run(
                    time=_time,
                    altkm=alt*1e-3,
                    glat=glat,
                    glon=glon,
                )]

        _dtype = [('date', _npt.dtype)]
        for key in species:
            _dtype += [(key, 'float64')]

        data = np.empty((len(data_all),), dtype=_dtype)
        for ind, raw in enumerate(data_all):
            data['date'][ind] = _npt[ind]
            for key in species:
                data[ind][key] = raw[key].data.flatten()[0]

        return data

