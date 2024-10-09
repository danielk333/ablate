#!/usr/bin/env python

'''A collection of utility functions.

'''

import numpy as np


# Scale height
HT_NORM_CONST = 7160.0


def rescale_hight(lat, lon, ht_data, jd):
    """ Given observed heights, rescale them from the real NRLMSISE model to the a simplified exponential
        atmosphere model used by the Alpha-Beta procedure.
    
    Arguments:
        lat: [ndarray] Latitude in radians.
        lon: [ndarray] Longitude in radians.
        ht_data: [ndarray] Height in meters.
        jd: [float] Julian date.

    Return:
        rescaled_ht_data
    """

    def _expAtmosphere(ht_data, rho_atm_0=1.0):
        """ Compute the atmosphere mass density using a simple exponential model and a scale height. 
    
        Arguments:
            ht_data: [ndarray] Height in meters.

        Keyword arguments: 
            rho_atm_0: [float] Sea-level atmospheric air density in kg/m^3.

        Return:
            [float] Atmospheric mass density in kg/m^3.
        """

        return rho_atm_0*(1/np.e**(ht_data/HT_NORM_CONST))

    def _expAtmosphereHeight(air_density, rho_atm_0=1.225):
        """ Compute the height given the air density and exponential atmosphere assumption. 

        Arguments:
            air_density: [float] Air density in kg/m^3.

        Keyword arguments: 
            rho_atm_0: [float] Sea-level atmospheric air density in kg/m^3.

        Return:
            [float] Height in meters.
        """

        return HT_NORM_CONST*np.log(rho_atm_0/air_density)


    # Get the atmosphere mass density from the NRLMSISE model for the observed heights
    atm_dens = getAtmDensity_vect(lat, lon, ht_data, jd)

    # Get the equivalent heights using the exponential atmosphre model
    ht_rescaled = _expAtmosphereHeight(atm_dens)

    # # Compare the models
    # plt.semilogy(ht_data/1000, atm_dens, label='NRLMSISE')
    # plt.semilogy(ht_data/1000, _expAtmosphere(ht_data), label='Exp')
    # plt.xlabel("Height (km)")
    # plt.ylabel("log air density kg/m3")
    # plt.legend()
    # plt.show()

    # # Compare the heights before and after rescaling
    # plt.scatter(ht_data/1000, ht_data - ht_rescaled)
    # plt.xlabel("Height (km)")
    # plt.ylabel("Height difference (m)")
    # plt.show()
    # sys.exit()

    return ht_rescaled
