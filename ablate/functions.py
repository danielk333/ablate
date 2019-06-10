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
    :param str material: Meteoroid material, see :mod:`~functions.material_data`.
    :param float/numpy.ndarray A: Shape factor [1]
    
    :return: Mass loss [kg/s]
    :rtype: float/numpy.ndarray

    **Reference: **
     Rogers et al.: Mass loss due to  sputtering and thermal processes 
     in meteoroid ablation, Planetary and Space Science 53 p. 1341-1354 (2005)
     Hill et al.: High geocentric velocity meteor ablation, A&A 444 p. 615-624 (2005)
    

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

    #-- 2007-03-21 Ed suggests that the vapor pressure should be lowered by a factor of 0.8 or 0.7
    #-- due to the large ram pressure forcing the evaporated atoms and molecules back on to the surface
    #-- thus leading to evaporation at a pressure that is lower than the equilibrium vapor pressure.

    Pv = 10.0**(CA - CB/T) # in [d/cm2]...; d=dyne=10-5 Newton: the force required to accelearte a mass of one g at a rate of one cm per s^2
    Pv = Pv*1E-5/1E-4 # Convert to [N/m2]

    dmdt = -4.0*A*(mass/rho_m)**(2.0/3.0)*Pv*np.sqrt(mu/(2.0*np.pi*constants.k*T)) #[kg/s]

    return dmdt



material_data = {
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
'''dict: Meteoroid material data.

# TODO: List origin for all data.
'''

def material_parameters(material):
    '''Returns the physical parameters of the meteoroid based on its material.
    
    :param str material: Meteoroid material, see :mod:`~functions.material_data`.

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

    if _material in material_data:
        return material_data[_material]
    else:
        raise ValueError(f'No data exists for material "{_material}"')

#    :param AtmosphereModel atmosphere: Atmospheric model instance.
#    :param numpy.datetime64 npt: Time to evaluate Atmospheric model at. 

def sputtering7(mass, velocity, material, density):
    '''calculates the meteoroid mass loss due to sputtering.
    
    :param float/numpy.ndarray mass: Meteoroid mass in [kg]
    :param float/numpy.ndarray velocity: Meteoroid velocity in [m/s]
    :param str material: Meteoroid material, see :mod:`~functions.material_data`.
    :param numpy.ndarray density: Structured numpy array of atmospheric constituent densities.

    All units need to be given in the cgs system to remain consistent with the units in which the sputtering yield equation was derived i.e., variables given in SI system and converted into cgs in this function when needed.


    # TODO: Make m and v able to be vectors

    **References:**

        * Rogers et al.: Mass loss due to  sputtering and thermal processes in meteoroid ablation, Planetary and Space Science 53 p. 1341-1354 (2005)
        * Tielens et al.: The physics of grain-grain collisions and gas-grain sputtering in interstellar shocks, The Astrophisicsl Journal 431, p. 321-340 (1994)


    **Change log:**
    
        * Changes post-python port: See git-log
        * Changes pre-python port: See `matlab-version <https://gitlab.irf.se/kero/ablation_matlab>`_


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

    '''
    v = velocity
    m = mass

    #Universal constants
    u = constants.u
    elem = constants.value(u'elementary charge')
    a0 = constants.value(u'Bohr radius')*1E+2 #Bohr radius in cm (cgs system)

    #Atmospheric molecules
    #M1 = projectile mass in kg [O N2 O2 He Ar H N], for molecules: average of its constitutents (see table 3 on p. 333 in Tielens et al)
    ##Z for a molecule: average of its consitutents (see table 3 on p. 333 in Tielens et al, not given in the table, but I use the same method)
    avalible_species = ['O', 'N2', 'O2', 'He', 'Ar', 'H', 'N']
    m1 = np.array([15.9994, 14.007, 15.999, 4.0026, 39.948, 1.0079, 14.007]*u)
    z1 = np.array([8, 7, 8, 2, 18, 1, 7])

    use_species = np.full(m1.shape, False, dtype=np.bool)
    _density = []
    for key in density.dtype.names:
        if key in avalible_species:
            use_species[ind] = True
            _density += [density[key][0]]

    m1 = m1[use_species]
    z1 = z1[use_species]
    _density = np.array(_density, dtype=np.float64)

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

    #If no energies above, no sputtering
    if np.sum(yes) == 0:
        return 0.0

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
    density_y = _density[yes]

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

    dmdt = -2.0*m2y*A*v*1E2*(m/rho_m)**(2.0/3.0)*Y_tot.T #[g/s]
    dmdt = dm_dt / 1E3 #[kg/s]

    return dmdt


def curved_earth(s, xdata, ydata):
    '''Calculates the distance :code:`s` along the trajectory at h=h_start (500km) when s=0 at h=h_obs (96km) given that the zentith distance is zd at h=h_obs (96km)
    


    '''

    rp      = xdata[0]
    h_obs   = xdata[1]
    zd      = ydata[0]
    h_start = ydata[1]

    theta   = np.arctan2(s*np.sin(zd), (rp + h_obs + s*np.cos(zd)))
    h       = (rp + h_obs + s*np.cos(zd))/np.cos(theta) - rp

    err = h - h_start

    return err


