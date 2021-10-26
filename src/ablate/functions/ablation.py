#!/usr/bin/env python

'''
Thermal ablation physics
=========================


Examples
------------

Calculate thermal ablation:

.. code-block:: python

    import numpy as np

    import ablate.functions as func

    dmdt_th = func.ablation.thermal_ablation(
        mass = np.array([0.5, 0.2]),
        temperature = 3700,
        material = 'ast',
        A = 1.21,
    )

    print(f'Thermal ablation {dmdt_th} kg/s')


'''
# Basic Python
import copy
import logging

logger = logging.getLogger(__name__)

# External packages
import numpy as np
from scipy import constants


# Internal packages
from .material import material_parameters




def luminosity(velocity, thermal_ablation):
    '''Luminosity during thermal ablation.
    
    :param float/numpy.ndarray velocity: Meteoroid velocity [m/s]
    :param float/numpy.ndarray thermal_ablation: Mass loss due to thermal ablation [kg/s]

    :rtype: float/numpy.ndarray
    :return: Luminosity [W]
    

    Luminosity occurs in meteors as a result of decay of excited atomic (and
    a few molecular) states following collisions between ablated meteor atoms
    and atmospheric constituents.

    **References:**

        * Friichtenicht and Becker: Determination of meteor paramters using laboratory simulations techniques, 'Evolutionary and physical properties of meteoroids', 
        * National Astronautics and Space Administration, Chapter 6, p. 53-81 (1973)
        * Hill et al.: High geocentric velocity meteor ablation, A&A 444 p. 615-624 (2005)


    **Symbol definitions:**

        * I = [W] light intensity of a meteoroid, i.e. luminous intensity; radiant intensity
        * tau_I = luminous efficiency factor
        * mu = [kg] mean molecular mass of ablated material
        * v = [m/s] meteoroid velocity
        * epsilon = emissivity
        * zeta = excitation coeff
        * rho_m = [kg/m3] meteoroid density
        * abla = mass loss due to thermal ablation

    '''

    #-- Universal constants
    u = constants.u


    #-- Variables
    v = velocity #[m/s] velocity
    vkm = v/1E3 #[km/s] velocity
    abla = thermal_ablation

    #-- for visible meteors, the energy is in lines, it is believed that they are composed of iron (Cepleca p. 355, table 3)
    epsilon_mu = 7.668E6 #[J/kg] mean excitation energy = epsilon/mu; mu = mean molecular mass


    #-- The excitation coeff for different velocity intervals:
    #-- See Hill et al. and references therein
    zeta = np.zeros(v.shape, dtype=v.dtype)
    I = v < 20000

    zeta[I] = -2.1887E-9*v[I]**2 + 4.2903E-13*v[I]**3 - 1.2447E-17*v[I]**4

    I = np.logical_and(v >= 20000, v < 60000)
    zeta[I] = 0.01333*vkm[I]**1.25

    I = np.logical_and(v >= 60000, v < 100000)
    zeta[I] = -12.835 + 6.7672E-4*v[I] - 1.163076E-8*v[I]**2 + 9.191681E-14*v[I]**3 - 2.7465805E-19*v[I]**4

    I = v >= 100000
    zeta[I] = 1.615 + 1.3725E-5*v[I] 


    #-- Luminous efficiency factor
    tau_I = 2*epsilon_mu*zeta/v**2;


    #-- The normal lumonous equation
    I = -0.5*tau_I*abla*v**2

    return I


def temperature_rate(mass, velocity, temperature, material, shape_factor, atm_total_density, thermal_ablation, Lambda, atm_temperature = 280, emissivity = 0.9):
    '''Calculates the rate of change of temperature of the meteoroid. A homogeneous metoroid experiencing an isotropic heat flux is assumed as well as the meteoroid undergoing isothermal heating. (isotherma heating: here: dTdS = 0 i.e. same spatial temperature)
    
    
    :param float/numpy.ndarray mass: Meteoroid mass [kg]
    :param float/numpy.ndarray velocity: Meteoroid velocity [m/s]
    :param float/numpy.ndarray temperature: Meteoroid temperature [K]
    :param str material: Meteoroid material, see :mod:`~functions.material.material_data`.
    :param float/numpy.ndarray shape_factor: Shape factor [1]
    :param float/numpy.ndarray atm_total_density: Total atmospheric number density [1/m^3]
    :param float/numpy.ndarray thermal_ablation: Mass loss due to thermal ablation [kg/s]
    :param float/numpy.ndarray Lambda: Heat transfer coefficient [1]
    :param float/numpy.ndarray atm_temperature: Effective atmospheric temperature [K]
    :param float/numpy.ndarray emissivity: Electromagnetic emissivity of meteoroid [1]

    :rtype: float/numpy.ndarray
    :return: Rate of change of temperature [K/s]
        

    **Default values:**

        * atm_temperature: ?
        * emissivity: 0.9 from Hill et al.; Love & Brownlee: 1; 0.2 is characteristic for a metal, oxides are between 0.4 and 0.8


    **References:**

        * Hill et al.: High geocentric velocity meteor ablation, A&A 444 p. 615-624 (2005)


    **Change log:**

        * Changes post-python port: See git-log
        * Changes pre-python port: See `matlab-version <https://gitlab.irf.se/kero/ablation_matlab>`_


    **Heat balance equation per cross-sectional area:**

        * A = shape factor
        * c = specific heat of meteoroid
        * mass = meteoroid mass
        * rho_m = meteoroid density
        * Lambda = heat transfer coefficient
        * rho_tot= total atmospheric mass density
        * vel = meteoroid velocity
        * kb = Stefan-Bolzmann constant
        * T = meteoroid temperature
        * Ta = effective atmospheric temperature
        * L = latent heat of fusion + vapourization
        * thermal_ablation = mass loss due to thermal ablation


    '''

    #-- variables
    kB = constants.value(u'Boltzmann constant') #[J/K]

    epsilon = emissivity

    Ta = atm_temperature #[K] effective atmospheric temperature
    
    vel = velocity
    T = temperature
    A = shape_factor
    rho_tot = atm_total_density

    mat_data = material_parameters(material)

    rho_m = mat_data['rho_m']
    c = mat_data['c']
    L = mat_data['L']

    dTdt = A/(c*mass**(1.0/3.0)*rho_m**(2.0/3.0))*(0.5*Lambda*rho_tot*vel**3 - 4*kB*epsilon*(T**4 - Ta**4) + L/A*(rho_m/mass)**(2.0/3.0)*thermal_ablation)
    return dTdt



def thermal_ablation(mass, temperature, material, shape_factor):
    '''Calculates the mass loss for meteoroids due to thermal ablation.

    :param float/numpy.ndarray mass: Meteoroid mass [kg]
    :param float/numpy.ndarray temperature: Meteoroid temperature [K]
    :param str material: Meteoroid material, see :mod:`~functions.material.material_data`.
    :param float/numpy.ndarray shape_factor: Shape factor [1]


    :rtype: float/numpy.ndarray
    :return: Mass loss due to thermal ablation [kg/s]


    **Reference:**

        * Rogers et al.: Mass loss due to  sputtering and thermal processes in meteoroid ablation, Planetary and Space Science 53 p. 1341-1354 (2005)
        * Hill et al.: High geocentric velocity meteor ablation, A&A 444 p. 615-624 (2005)


    **Change log:**

        * Changes post-python port: See git-log
        * Changes pre-python port: See `matlab-version <https://gitlab.irf.se/kero/ablation_matlab>`_


    **Symbol definintions:**

        * Pv: vapour pressure of meteoroid [N/m^2]
        * CA, CB: Clausius-Clapeyron coeff [K]
        * mu: mean molecular mass of ablated vapour [kg]
        * rho_m: meteoroid density [kg/m3]


    '''

    mat_data = material_parameters(material)

    CA = mat_data['CA']
    CB = mat_data['CB']
    mu = mat_data['mu']
    rho_m = mat_data['rho_m']

    # 2007-03-21 Ed suggests that the vapor pressure should be lowered by a factor of 0.8 or 0.7
    # due to the large ram pressure forcing the evaporated atoms and molecules back on to the surface
    # thus leading to evaporation at a pressure that is lower than the equilibrium vapor pressure.

    Pv = 10.0**(CA - CB/temperature) # in [d/cm2]...; d=dyne=10-5 Newton: the force required to accelearte a mass of one g at a rate of one cm per s^2
    Pv = Pv*1E-5/1E-4 # Convert to [N/m2]

    dmdt = -4.0*shape_factor*(mass/rho_m)**(2.0/3.0)*Pv*np.sqrt(mu/(2.0*np.pi*constants.k*temperature)) #[kg/s]

    return dmdt
