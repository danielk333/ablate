#!/usr/bin/env python

'''
General dynamics
=========================


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


def drag_coefficient(mass, velocity, temperature, material_data, atm_total_density, atm_mean_mass, res = 100):
    '''Calculates the drag coefficient Gamma.

    :param float/numpy.ndarray mass: Meteoroid mass [kg]
    :param float/numpy.ndarray velocity: Meteoroid velocity [m/s]
    :param float/numpy.ndarray temperature: Meteoroid temperature [K]
    :param dict material_data: Meteoroid material data, see :mod:`~functions.material.material_data`.
    :param float/numpy.ndarray atm_total_density: Total atmospheric number density [1/m^3]
    :param float/numpy.ndarray atm_mean_mass: Mean mass of atmospheric constituents [kg]
    :param int res: Resolution used for the numerical integration.
    
    :rtype: float/numpy.ndarray
    :return: Drag coefficient


    **References:**
        * V. A. Bronshten; Physics of meteoric phenomena (1983)
        * A. Westman et al.: Meteor head echo altitude distributions and the height cutoff effect sudied with the EISCAT HPLA UHF and VHF radars; Annales Geophysicae 22: 1575-1584 (2004)
        * Tielens et al.: The physics of grain-grain collisions and gas-grain sputtering in interstellar shocks, The Astrophisicsl Journal 431, p. 321-340 (1994)
        * Rogers et al.: Mass loss due to  sputtering and thermal processes in meteoroid ablation, Planetary and Space Science 53 p. 1341-1354 (2005)
        * Salby: Fundamentals of atmospheric physics, Academic Press (1996)


    **Change log:**
    
        * Changes post-python port: See git-log
        * Changes pre-python port: See `matlab-version <https://gitlab.irf.se/kero/ablation_matlab>`_



    '''
    #Universal constants
    kB = constants.value(u'Boltzmann constant') #[J/K]

    #Meteoroid constituents
    met_mean_mass = material_data['m2']
    met_density = material_data['rho_m']

    #angle between the normal to the surface and the molecular flow, theta0 is a row vector
    theta0 = np.linspace(0, np.pi/2, num=res, dtype=np.float64) #[radians] 


    #parameter... Bronshten p. 69 eq. 10.7
    h_star = atm_mean_mass*velocity**2/(2.0*kB*temperature) #[dimensionless]

    a_e = thermal_accommodation(atm_mean_mass, met_mean_mass)
    u_s = nomalized_evaporated_velocity(velocity, temperature, atm_mean_mass)
    Kn_inf, _ = Knudsen_number(mass, met_density, atm_total_density)


    ######## Gamma ##########
    epsilon_p = 0.86 #[dimensionless] Bronshten p. 69 eq. 10.7 and p. 71

    #integrating p0, Bronshten p. 71
    _F = np.sin(theta0)*np.cos(theta0)*(1.0 - u_s)

    _Fp = np.sin(theta0)*4/np.sqrt(2*np.pi)*np.sqrt(1 - 2*u_s*np.cos(theta0) + u_s**2)*np.cos(theta0)
    _Fp *= np.cos(theta0)*(1 - u_s)/2\
        - 1.0/24.0*(1 + np.cos(theta0))**3\
        + u_s/24.0*(1 + np.cos(theta0))**2*(4 - 2*np.cos(theta0) + np.cos(theta0)**2)

    P0 = np.trapz(theta0, _F)

    #integrating p_prim over theta0 from 0 - pi/2, Bronshten p. 71
    P_prim = np.trapz(theta0, _Fp)     
    
    # the momentum flux shielding coeff, Bronshten p. 71 same as a_Gamma but with the integrated vectors
    A_Gamma_r = 1 - epsilon_p*np.sqrt(h_star)/Kn_inf*P_prim/P0 #[dimensionless]
    Gamma = A_Gamma_r*a_e #[dimensionless] drag coeff

    # We don't want Gamma or Lambda to be > 1, searching for those and setting them = 1
    
    if isinstance(Gamma, np.ndarray):
        Gamma[Gamma > 1] = 1.0
    else:
        if Gamma > 1:
            Gamma = 1.0

    return Gamma



def heat_transfer(mass, velocity, temperature, material_data, atm_total_density, thermal_ablation, atm_mean_mass, res = 100):
    '''Calculates the heat transfer coefficient Lambda.

    :param float/numpy.ndarray mass: Meteoroid mass [kg]
    :param float/numpy.ndarray velocity: Meteoroid velocity [m/s]
    :param float/numpy.ndarray temperature: Meteoroid temperature [K]
    :param dict material_data: Meteoroid material data, see :mod:`~functions.material.material_data`.
    :param float/numpy.ndarray atm_total_density: Total atmospheric number density [1/m^3]
    :param float/numpy.ndarray thermal_ablation: Mass loss due to thermal ablation [kg/s]
    :param float/numpy.ndarray atm_mean_mass: Mean mass of atmospheric constituents [kg]
    :param int res: Resolution used for the numerical integration.

    :rtype: float/numpy.ndarray
    :return: Heat transfer coefficient


    **References:**
        * V. A. Bronshten; Physics of meteoric phenomena (1983)
        * A. Westman et al.: Meteor head echo altitude distributions and the height cutoff effect sudied with the EISCAT HPLA UHF and VHF radars; Annales Geophysicae 22: 1575-1584 (2004)
        * Tielens et al.: The physics of grain-grain collisions and gas-grain sputtering in interstellar shocks, The Astrophisicsl Journal 431, p. 321-340 (1994)
        * Rogers et al.: Mass loss due to  sputtering and thermal processes in meteoroid ablation, Planetary and Space Science 53 p. 1341-1354 (2005)
        * Salby: Fundamentals of atmospheric physics, Academic Press (1996)


    **Change log:**
    
        * Changes post-python port: See git-log
        * Changes pre-python port: See `matlab-version <https://gitlab.irf.se/kero/ablation_matlab>`_



    '''

    #Universal constants
    kB = constants.value(u'Boltzmann constant') #[J/K]

    #Meteoroid constituents
    met_mean_mass = material_data['m2']
    met_density = material_data['rho_m']


    #angle between the normal to the surface and the molecular flow, theta0 is a row vector
    theta0 = np.linspace(0, np.pi/2, num=res, dtype=np.float64) #[radians] 

    a_e = thermal_accommodation(atm_mean_mass, met_mean_mass)
    u_s = nomalized_evaporated_velocity(velocity, temperature, atm_mean_mass)
    Kn_inf, L = Knudsen_number(mass, met_density, atm_total_density)

    #parameter... Bronshten p. 69 eq. 10.7
    h_star = atm_mean_mass*velocity**2/(2.0*kB*temperature) #[dimensionless]


    ######## Lambda ##########


    # Shielding effects of a meteoroid surface due to reflected molecules:
    epsilon_q = 1.6 #[dimensionless]

    _G = np.sin(theta0)*np.cos(theta0)*(1 - u_s**2)

    _Gp = np.sin(theta0)*4/np.sqrt(2*np.pi)*np.sqrt(1 - 2*u_s*np.cos(theta0) + u_s**2)*np.cos(theta0)
    _Gp *= np.cos(theta0)*(1 - u_s**2)/2 - (1.0/48.0*(1 + np.cos(theta0))**3*(3 - np.cos(theta0)) - u_s/48.0*(1 + np.cos(theta0))**2*(8 - 7*np.cos(theta0) + 2*np.cos(theta0)**2 - np.cos(theta0)**3))


    Q0 = np.trapz(theta0, _G) 

    #integrating q_prim over theta0 from 0 - pi/2; Bronshten p. 69, eq. 10.10 and Bronshten I eq. 11
    Q_prim = np.trapz(theta0, _Gp)

    # the energy flux shielding coeff, Bronshten p. 71, same as a_Lambda but with the integrated vectors; Bronshten p. 69 eq. 10.8
    A_Lambda_r  = 1 - epsilon_q*np.sqrt(h_star)/Kn_inf*Q_prim/Q0 #[dimensionless]
    
    
    # Shielding effects of a meteoroid surface due to evaporated molecules:
    
    #scattering cross section for meteor atoms and ions on N2 and O2 molecules; Bronshten p. 84 eq. 11.29, Bronshten II. eq. 40; the actual equation is for v in [cm/s] and gives the cross section in cm^2 -> sigmaD = 5.6E-11 [cm^2] *(v [cm/s])^-0.8 = 5.6E-11*(1E-2[m])^2*(v [m/s]*1E2)^-0.8 = 5.6E-11*1E-4*(1E2)^-0.8*v^-0.8 = 5.6E-15*10^-1.6*v^-0.8
    sigmaD = 5.6E-15 * 10**(-1.6) * velocity**(-0.8) #[m^2]

    #the velocity of molecules evaporated from the meteor body, Bronshten eq. 7.2 p. 37 
    v_e = np.sqrt(8*kB*temperature/(np.pi*met_mean_mass)) #[m/s]

    #mean free path of the evaporated molecules, Bronshten p. 79 eq. 11.8; Bronshten II. eq. 2
    mfp_e = v_e/(atm_total_density*velocity*sigmaD) #[m] 

    #the toatal fraction of evaporated molecules from the entire front surface of the meteoroid, Bronshten II. eq. 35
    eta = (L/mfp_e)**2/(1 + 2*np.sqrt(L/mfp_e)) #[dimensionless] 
    
    #the total number of molecules evaporated from a certain area of cross section and which takes part in shielding, Bronshten II. eq. 42 and eq. 1
    N_i = eta/met_mean_mass*(-1.0*thermal_ablation) #[number] 

    #cross section area of meteoroid (L = characteristic length, but here it is equal to the meteoroid radius)
    S_m = np.pi*L**2 #[m^2] 

    #the number of molecules advancing on the area S, Bronshten II. p. 135
    N_a = atm_total_density*velocity*S_m #[number]
    
    #average velocity of evaporated molecules from the meteoroid surface, see v_s which is the same thing but for reflected molecules
    v_s_e = np.sqrt(3*constants.value(u'molar gas constant')*temperature/(met_mean_mass*constants.value(u'Avogadro constant'))) #[m/s]
    
    #nomalized velocity of evaporated molecules, compare to u_s
    u_e = v_s_e/velocity #[dimensionless] 

    _H = np.sin(theta0)*1.0/48.0*(1 + np.cos(theta0))**3*(3 - np.cos(theta0)) - u_e/48.0*(1 + np.cos(theta0))**2*(8 - 7*np.cos(theta0) + 2*np.cos(theta0)**2 - np.cos(theta0)**3)

    #same as Q_star but for evaporated molecules, i.e. using u_e instead of u_s
    Q_star_e = np.trapz(theta0, _H)
    #same as Q0 but for evaporated molecues, i.e. using u_e instead of u_s
    Q0_e = np.trapz(theta0, np.sin(theta0)*np.cos(theta0)*(1 - u_e**2)) 
    
    #determines how many of the reflected molecules that are thrown back, Bronshten I. eq. 17
    zeta = 1 - 2*Q_star_e/Q0_e

    #the energy flux shielding by evaporation coeff for a sphere, Bronshten II. eq. 43
    A_Lambda_e = 1 - zeta*N_i/N_a #[dimensionless]
    
    # How to add the shielding effects by reflection and evaporation? A_Lambda_r and A_Lambda_e 
    # is defined as the probability that an incoing molecule reaches the surface of the body and 
    # transfers its momentum or energy respectively to it. Thus 1 - A_Lambda_r and 1 - A_Lambda_e
    # is the probability that an incoming molecule does not reach the surface of the body, i.e., 
    # is shielded. So, to add how many of the incoming molecules contributes to the shielding both
    # by reflection and evaporation, we need to add the probability that the incoming molecules 
    # does not reach the surface: 1 - ( (1 - A_Lambda_r) + (1 - A_Lambda_e) ) = 
    # = 1 - 1 + A_Lambda_r - 1 + A_Lambda_e = A_lambda_r + A_lambda_e -1
    
    Lambda = (A_Lambda_r + A_Lambda_e - 1)*a_e #[dimensionless] heat transfer coeff
    
    # We don't want Gamma or Lambda to be > 1, searching for those and setting them = 1
    if isinstance(Lambda, np.ndarray):
        Lambda[Lambda > 1] = 1.0
    else:
        if Lambda > 1:
            Lambda = 1.0

    return Lambda

def thermal_accommodation(atm_mean_mass, met_mean_mass):
    '''Calculate thermal accommodation coefficient, Bronshten p. 40 eq. 7.11

    :param float/numpy.ndarray met_mean_mass: Mean mass of meteoroid constituents [kg]
    :param float/numpy.ndarray atm_mean_mass: Mean mass of atmospheric constituents [kg]
    
    :rtype: float/numpy.ndarray
    :return: Thermal accommodation coefficient

    '''


    #the relative masses of the molecules of the air and of the meteoroid, Bronshten p. 40
    mu_star = atm_mean_mass / met_mean_mass #[dimensionless] 

    # thermal accommodation coefficient, Bronshten p. 40 eq. 7.11
    a_e = (3.0 + mu_star)*mu_star/(1.0 + mu_star)**2 #[dimensionless]

    return a_e


def nomalized_evaporated_velocity(velocity, temperature, atm_mean_mass):
    '''Calculate the nomalized reflected and evaporated velocity, Bronshten p. 69

    :param float/numpy.ndarray velocity: Meteoroid velocity [m/s]
    :param float/numpy.ndarray temperature: Meteoroid temperature [K]
    :param float/numpy.ndarray atm_mean_mass: Mean mass of atmospheric constituents [kg]

    :rtype: float/numpy.ndarray
    :return: Nomalized reflected and evaporated velocity

    '''

    #Universal constants
    R = constants.value(u'molar gas constant') #[J/(mol K)]
    NA = constants.value(u'Avogadro constant') #[molecules/mol]

    #average velocity of reflected molecules near the meteoroid surface, Bronshten p. 37 eq. 7.1 (and 7.2)
    v_s = np.sqrt(3*R*temperature/(atm_mean_mass*NA)) #[m/s] 

    #nomalized reflected and evaporated velocity, Bronshten p. 69 but uncorrect in the book...
    u_s = v_s/velocity #[dimensionless]

    return u_s


def Knudsen_number(met_mass, met_density, atm_total_density):
    '''Calculate the Knudsen number.

    :param float/numpy.ndarray met_mass: Meteoroid mass [kg]
    :param float/numpy.ndarray met_density: Meteoroid density [kg/m^3]
    :param float/numpy.ndarray atm_total_density: Total atmospheric number density [1/m^3]

    :rtype: tuple(float/numpy.ndarray, float/numpy.ndarray)
    :return: Knudsen number, Body characteristic dimension


    '''

    # Atmospheric mean free path, source: Westman et al.
    #Physics Handbook p. 186 This is the one used!
    mfp_inf = 1.0/(np.pi*(3.62E-10)**2*atm_total_density) # [m]

    # The Knudsen number
    # Calculating the radius of the meteoroid at different altutude to get L
    #characteristic dimension of the body (the meteoroid); in particular, for a sphere L = the radius; Bronshten p. 31
    L = ((met_mass/met_density)/(4*np.pi/3))**(1.0/3.0) #[m] 
    
    #[dimensionless] the Knudsen no; calculated from 'mfp_inf'
    Kn_inf = mfp_inf/L 

    return Kn_inf, L

