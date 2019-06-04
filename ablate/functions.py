#!/usr/bin/env python

'''Collection of functions.

'''
#
# Basic Python
#


#
# External packages
#
import numpy as np
from scipy import constants



def thermal_ablation(mass, T, material, A):
    '''Calculates the mass loss for meteoroids due to thermal ablation.

    :param float/numpy.ndarray mass: Meteoroid mass [kg]
    :param float/numpy.ndarray T: Meteoroid temperature [K]
    :param str material: Meteoroid material
    :param float/numpy.ndarray A: Shape factor [1]
    
    :return: Mass loss [kg/s]
    :rtype: float/numpy.ndarray

    **Reference: **
     Rogers et al.: Mass loss due to  sputtering and thermal processes 
     in meteoroid ablation, Planetary and Space Science 53 p. 1341-1354 (2005)
     Hill et al.: High geocentric velocity meteor ablation, A&A 444 p. 615-624 (2005)
    

    **Change log:**
    
    2019-06-04 Daniel Kastinen
       Ported to Python.

    2007-03-21 Johan Kero
       Ed Murad suggests that the vapor pressure should be lowered by a factor of 0.8 or 0.7 due to the large ram pressure forcing the evaporated atoms and molecules back on to the surface thus leading to evaporation at a pressure that is lower than the equilibrium vapor pressure.

    2006-12-12 Csilla Szasz
       This functions is now able to handle vectors.


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

    #-- 2007-03-21 Ed suggests that the vapor pressure should be lowered by a factor of 0.8 or 0.7
    #-- due to the large ram pressure forcing the evaporated atoms and molecules back on to the surface
    #-- thus leading to evaporation at a pressure that is lower than the equilibrium vapor pressure.

    Pv = 10.0**(CA - CB/T) # in [d/cm2]...; d=dyne=10-5 Newton: the force required to accelearte a mass of one g at a rate of one cm per s^2
    Pv = Pv*1E-5/1E-4 # Convert to [N/m2]

    dmdt = -4.0*A*(mass/rho_m)**(2.0/3.0)*Pv*np.sqrt(mu/(2.0*np.pi*constants.k*T)) #[kg/s]

    return dmdt



_material_data = {
    'iron': {
        'rho_m': 7800.0,
        'mu': 56.0*constants.u,
        'm2': 56.0*constants.u,
        'CA': 10.6,
        'CB': 16120.,
        'u0': 4.1,                            
        'k': 0.35,
        'z2': 26,
    },
    'asteroroidal': {
        'rho_m': 3300.0,
        'mu': 50.0*constants.u,
        'm2': 20.0*constants.u,
        'CA': 10.6,
        'CB': 13500.,
        'u0': 6.4,                            
        'k': 0.1,
        'z2': 10,
    },
    'cometary': {
        'rho_m': 1000.0,
        'mu': 20.0*constants.u,
        'm2': 12.0*constants.u,
        'CA': 10.6,
        'CB': 13500.,
        'u0': 4.0,                            
        'k': 0.65,
        'z2': 6,
    },
    'porous': {
        'rho_m': 300.0,
        'mu': 20.0*constants.u,
        'm2': 12.0*constants.u,
        'CA': 10.6,
        'CB': 13500.,
        'u0': 4.0,                            
        'k': 0.65,
        'z2': 6,
    },
}


def material_parameters(material):
    '''Returns the physical parameters of the meteoroid based on its material.
    

    **List of properties:**
        * m2: mean atomic mass [1]
        * u0: surface binding energy [eV]
        * k: average atomic number [1]
        * CA, CB: Clausius-Clapeyron coeff [K]
        * mu: mean molecular mass of ablated vapour [kg]
        * k: Botzmann constant [J/K]
        * rho_m: meteoroid density [kg/m3]


    [Fe SiO2 C C] mean molecular mass per target atoms corresponding to the different meteoroid densities, see Rogers p. 1344 values are from Tielens et al.
    
    Mean molecular mass of ablated vapur assuming that all products liberated from the meteoroid by thermal abaltion were released in complete molecular form.

    '''

    _material = material.lower().strip()

    if _material in _material_data:
        return _material_data[_material]
    else:
        raise ValueError(f'No data exists for material "{_material}"')



def sputtering7(mass, velocity, material, density):
    '''calculates the meteoroid mass loss due to sputtering.
    
    :param :

    **References:**

        * Rogers et al.: Mass loss due to  sputtering and thermal processes in meteoroid ablation, Planetary and Space Science 53 p. 1341-1354 (2005)

        * Tielens et al.: The physics of grain-grain collisions and gas-grain sputtering in interstellar shocks, The Astrophisicsl Journal 431, p. 321-340 (1994)


    **Change log:**
    
    2019-06-04 Daniel Kastinen
       Ported to Python.

    2007-05-13 Csilla Szasz
       Spline creates ripples when used to fit the msis densities: changing to 'pchip' instead. The only function needed to run this model where spline is used is 'interpolate_atm_dens.m' which is changed to 'interpolate_atm_dens2.m'. The named functions is only called in  'run_atm6.m'. We have also changed 'spline' to 'pchip' in all comments in all files... NOTE: Spline fits may give overshoots. Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) do not overshoot, but as a draw back may produce less smooth functions. It is anyway wiser to use pchip rather than spline to interpolate the data as the spline may create artifacts. 

    2007-04-10 Csilla Szasz
       The atmospheric no densities are given as input instead of being calculated here to eliminate the number of times 'ppval' is used because it is very time consuming.

    2007-02-22 Csilla Szasz
       The switch 'sab' is now a part of param. Implementing a new switch, 'glb', also a part of param.

    2007-01-30 Csilla Szasz
       The greatest Tolarable Local truncation error TL (the step size grows with TL, needed in the Runge-Kutta function file: rkf5_?.m) is now a part of the input parameter param. The reason for this is to be able to vary TL externally. If no TL is given, i.e., TL = NaN, TL is set to 1E-6 m/s in rkf5_5.m.  See the input for 2006-12-06 date of change.

    2006-12-07 Csilla Szasz and Johan Kero
       A fourth set of meteoroid surface values are added: for iron. The new meteoroid density that can be handled is 7 g/cm3.

    2006-12-06 Csilla Szasz
       When the atmospheric density is calculate, not only the hour, but also the day of experiment is taken into account -> input variables are messed around with.
    
    2006-11-20 Csilla Szasz
       In this version the MSIS density is calculated for each meteoroid's time of arrival. The MSIS valus are fore every full hour, so it is taken as the central time. So, if a meteoroid's time of detection is between h-1:30 and h:30 -> the atmoshperic density used is the one for hour h. M2, U0, Z2, K are implemented as functions (by DD Meisel) instead of having a table of surface material.

    2006-11-15 Csilla Szasz: sputtering4.m
       The change from sputtering3 is that this version does not use repmat because it calculates the yeild on one surface type at a time, i.e., for one meteoroid density at a time. This (sputtering4) is the version used to verify the results in Rogers et al. To switch between the paper's model and the MSIS one: in sputtering4.m row 112ff. density       = ppval(fit_coef, h/1E3);  h is needed in km; from the spline fit coefficients, calculating the atmospheric density for all atmospheric consituents for the height 'h' we are interested in. density(4)    = [];                      the msis table for confirming Rogers et al. contains the total densit at position 4, which is not needed here. density       = rogers_atm(h)';         atmosfären i Rogers et al, som är felaktig...

    2006-11 Csilla Szasz
       This function calculates the yield all meteoroid denities at the same time, i.e., as many masslosses as there are different meteoroid densities are calculated. However, the MSIS density still need to take the coefficients for the hour in question... because it takes the first hour for all. This function uses matrix multiplications to calculate the yield from all atmospheric and surface atoms at the same time. It has been compared to a the function sputtering_for_loops.m which does the same thing but using for statements. Both functions got the same result, but tis one is faster (tic, toc gave time 0 and time 0.01 s respectively for calculating the yield for seven velocities)

    2006-11-07 Csilla Szasz
       The change from sputtering1 is that this functions calculates one sputtering for each given meteoroid density with one given surface molecule.

    
    **Variables explanation:**
        * E_th  = threshold energy: the minimum projectile kinetic energy needed for a given projectile and target to induce sputtering
        * a     = screening length
        * beta  = maximum fractional energy transfer in head-on elastic collision
        * alpha = dimensionless function of the mass ratio
        * Rp    = mean projected range
        * R     = mean penetrated path
        * Y     = sputtering yield at normal incidence: the ratio of the mean number of sputtered particles per projectile
        * U0    = surface binding energy (eV), use the sublimation energy of the material
        * M1    = projectile mass
        * M2    = mean molecular mass per atom of the target
        * Z1    = projectile atomic number
        * Z2    = target atomic number
        * sn    = universal function
        * E     = incident projectile energy
        * Gamma = reduced energy
        * M2    = mean molecular mass per atom of the target
        * A     = shape factor, sphere = 1.21
        * v     = meteoroid velocity
        * m     = meteoroid mass
        * rho_m = meteoroid density


    -----------------------------------------------------------------------
    -- All units need to be given in the cgs system to remain consistent --
    -- with the units in which the sputtering yield equation was derived --
    -- i.e., variables given in SI system and converted into cgs in this --
    -- function when needed.                                             --
    -----------------------------------------------------------------------

    '''
    v = velocity
    m = mass

    #Universal constants
    u = constants.u
    elem = constants.value(u'elementary charge')
    a0 = constants.value(u'Bohr radius')*1E+2 #Bohr radius in cm (cgs system)

    #Atmospheric molecules
    m1 = np.array([15.9994, 14.007, 15.999, 4.0026, 39.948, 1.0079, 14.007])*u
    z1 = np.array([8, 7, 8, 2, 18, 1, 7])

    #Meteoroid constituents
    mat_data = material_parameters(material)

    m2 = mat_data['m2']
    u0 = mat_data['u0']
    k = mat_data['k']
    z2 = mat_data['z2']
    rho_m = mat_data['rho_m']

    beta = 4*m1*m2/(m1 + m2)**2

    
    m1_m2 = m1/m2
    E_th = 8*u0*m1_m2**(1.0/3.0)

    # when M1 ./ M2 > 0.3; E_th is in eV
    less03 = m1_m2 <= 0.3
    E_th[less03] = u0/(beta[less03]*(1 - beta[less03]))

    alpha = 0.3*(m2/m1)**(2.0/3.0)

    less05 = (m2/m1) < 0.5
    alpha[less05] = 0.2;

    #balances 'alpha' if m1/m2 grows too big; 
    #'alpha' should be between 1/2 and 2/3 see page 324 in Tielens
    Rp_R = (k*m2/m1 + 1.0)**(-1.0)
    a = 0.885*a0/np.sqrt(z1**(2.0/3.0) + z2**(2.0/3.0))


    #-------------------------------------
    #-- Stepping through the atmosphere --
    #-------------------------------------

    #-------------------------------------------
    #-- Sputtering yield at normal incidence: --
    #-------------------------------------------
    E = (m1*v**2.0/2.0)/elem #joule -> eV by dividing by 'elem' 
    #-- Sputtering only occurs if E > E_th
    yes = E > E_th

    Ey = E[yes]
    E_th_y = E_th[yes]
    m1y = m1[yes]*1E3 #[g] projectile mean molecular mass
    m2y = m2*1E3 #[g] target mean molecular mass
    m = m * 1E3 #[g] meteoroid mass
    rho_m = rho_m / 1E3 #[g/cm3] meteoroid density
    Rp_R_y = Rp_R[yes]
    z1y = z1[yes]
    ay = a[yes]
    alpha_y = alpha[yes]
    density_y = density[yes]

    #-- Calculating the yield for all atmospheric constituents for the given surface material (meteoroid dencity)
    #taking the energy in ergs so the unit of sn is ergs cm2; elementary charge in cgs: in esu
    Gamma = m2y/(m1y + m2y)*ay/(z1y*z2*(elem/3.33564E-10)**2.0)*Ey*elem*1E7

    sn_1 = 3.441*np.sqrt(Gamma)*np.log(Gamma + 2.781)
    sn_2 = (1.0 + 6.35*np.sqrt(Gamma)+ Gamma*(-1.708 + 6.882*np.sqrt(Gamma)))
    sn = sn_1/sn_2# [ergs cm2]


    #-- Valid for E > E_th:
    Y = 3.56/u0*m1y/(m1y + m2y)*z1y*z2/np.sqrt(z1y**(2.0/3.0) + z2**(2.0/3.0))
    Y *= alpha_y*Rp_R_y*sn*(1.0 - (E_th_y/Ey)**(2.0/3.0))*(1.0 - E_th_y/Ey)**2.0 
    #-- we have tried to put in E_th in eV and E in cgs-units (ergs) but the results were convincingly wrong...

    #the total yield is the sum of all individual yields x the atmospheric density
    Y_tot = np.sum(density_y*Y)

    #----------------------------------
    #-- Mass loss due to sputtering: --
    #----------------------------------
    A = 1.21 #Sphere

    dmdt          = -2.0*m2y*A*v*1E2*(m/rho_m)**(2.0/3.0)*Y_tot.T #[g/s]
    dmdt          = dm_dt / 1E3 #[kg/s]

    return dmdt
