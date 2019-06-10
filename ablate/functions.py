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

    # 2007-03-21 Ed suggests that the vapor pressure should be lowered by a factor of 0.8 or 0.7
    # due to the large ram pressure forcing the evaporated atoms and molecules back on to the surface
    # thus leading to evaporation at a pressure that is lower than the equilibrium vapor pressure.

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


    #---------------------------------
    # Stepping through the atmosphere --
    #---------------------------------

    #---------------------------------------
    # Sputtering yield at normal incidence: --
    #---------------------------------------
    E = (m1*v**2.0/2.0)/elem #joule -> eV by dividing by 'elem' 
    # Sputtering only occurs if E > E_th
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

    # Calculating the yield for all atmospheric constituents for the given surface material (meteoroid dencity)
    #taking the energy in ergs so the unit of sn is ergs cm2; elementary charge in cgs: in esu
    Gamma = m2y/(m1y + m2y)*ay/(z1y*z2*(elem/3.33564E-10)**2.0)*Ey*elem*1E7

    sn_1 = 3.441*np.sqrt(Gamma)*np.log(Gamma + 2.781)
    sn_2 = (1.0 + 6.35*np.sqrt(Gamma)+ Gamma*(-1.708 + 6.882*np.sqrt(Gamma)))
    sn = sn_1/sn_2# [ergs cm2]


    # Valid for E > E_th:
    Y = 3.56/u0*m1y/(m1y + m2y)*z1y*z2/np.sqrt(z1y**(2.0/3.0) + z2**(2.0/3.0))
    Y *= alpha_y*Rp_R_y*sn*(1.0 - (E_th_y/Ey)**(2.0/3.0))*(1.0 - E_th_y/Ey)**2.0 
    # we have tried to put in E_th in eV and E in cgs-units (ergs) but the results were convincingly wrong...

    #the total yield is the sum of all individual yields x the atmospheric density
    Y_tot = np.sum(density_y*Y)

    #------------------------------
    # Mass loss due to sputtering: --
    #------------------------------
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


def dp5(t, yvar, fit_coef, time, param):
    '''Gives out a vector of derivatives for meteoroid ablation.


    This function is originally a tranlation of a Mathematica function written by DD Meisel and is a part of the program ThermalR8.nb.

    This function gives out a vector of derivatives: x = [dvdt dmdt dhdt dTdt dmdt_a dmdt_s I] =
    [Velocity, Tot_mass_loss, Height, Temp, Mass_loss_due_to_ablation, Mass_loss_due_to_sputtering, Luminosity]


    **Function called:**
        * gam_lam3
        * sputtering7.m
        * ablation.m
        * temperature2.m


    **Change log:**
    
        * Changes post-python port: See git-log
        * Changes pre-python port: See `matlab-version <https://gitlab.irf.se/kero/ablation_matlab>`_


    '''

    '''
    # variables
    deg     = pi/180;                                                               #conversion from degrees to radians

    # yvar = [velocity mass height meteoroid_temperature (mass_loss_rate_due_to_ablation mass_loss_rate_due_to_sputtering)]; 
    vel     = abs(yvar(1));
    mass    = yvar(2);
    s       = yvar(3);

    # param = [rho_m za hour h_obs rp h]
    rho_m   = param(1);
    za      = param(2);
    day     = param(3);
    hour    = param(4);
    h_obs   = param(5);
    rp      = param(6);
    A       = param(8);
    G       = param(10);
    M       = param(11);
    sab     = param(16);
    glbn    = param(17);

    # Taking the curved earth -> curved atmosphere into account instead of the plane paralel one used earlier
    # (See figure in notes written in Geneseo, NY, page 15 (by Csilla).)
    # Theta and height has to be calculated for each time dp4 is called by rkf5_4, the final height is calculated
    # in run_atm3 and saved in a variable
    theta   = atan2( s.*sin(za) , ( rp + h_obs + s.*cos(za) ) );                    #angle at earth center between the meteor when observed and when at 500 km
    height  = (rp + h_obs + s.*cos(za)) ./ cos(theta) - rp;                         #meteoroid height above earth surface when at 500 km distance from detection

    # Calculating the total atmoshperic mass denisty from the pchip fit coefficients
    no      = find(time(:,3) == day & time(:,4) == hour);                           #the order of the pchip fit coeffs are the same as the order of the hours in the time matrix; hour is in column no 4 in time matrix
    rho_a   = ppval(fit_coef(no), height / 1E3) #the msis tables contain height in km
    rho_tot = rho_a(4) * 1E3 #total density is in the fourth row and convert it to kg/m3
    rho_a(4)= [] #[cm-3] the msis table contains the total density at position 4, which is treated separately

    # Calculating the total no denisty of the atmosphere:
    # adding the no denisty of each atmospheric constituent tabulated by MSIS to get
    # a total number density. This is done by the function tot_no_dens2.m but I have
    # moved its contents here to make gam_lam2.m faster by eliminating 'ppval' iside 
    # it (or inside tot_no_dens.m)
    rho_a_m             = rho_a * 1E6;                                              #[m-3] convert the no densities from /cm3 to /m3
    # I am interested in the total number density(!) at 'height' -> sum the no densities for each atmospheric component for each height, i.e., sum the columns of 'rho_a_m'
    totnodens           = sum(rho_a_m,1);                                           #[m-3] sum each column of 'density' (same as sum(density))

    # checking whether Gamma is given or is to be calculated (Gamma is needed before abla is 
    # calculated and abla is necessary for the Lambda calculations -> calculating them separatly)
    if glbn == 3 | glbn == 1
        gamlam          = gam_lam3(yvar, param, totnodens, [], []);
        Gamma           = gamlam;
        param(9)        = Gamma;
    else
        Gamma           = param(9);
    end

    # Differential equation for the velocity to solve
    dvdt_d              = - Gamma * A * rho_tot * vel^2 / ( mass^(1/3) * rho_m^(2/3) ); #[m/s2] momentekvationen; drag equation (because of conservation of linear momentum): decelearation=Drag_coeff*shape_factor*atm_dens*vel/(mass^1/2 * meteoroid_dens^2/3)
    dvdt_g              = mass * G * M / (height * rp);                                 #[m/s2] acceleration due to earth gravitaion
    dvdt                = dvdt_d + dvdt_g;

    # Differential equation for the height to solve
    dsdt                = -vel;                                                         #range from the common volume along the meteoroid trajectory

    # sab = 1 -> ablation ; sab = 2 -> ablation ; sab = 3 -> both ablation and sputtering
    if sab == 1 | sab ==  3 
        # P.S. This doesn't work properly... if sab==1 the temperature is not calculated and 
        # rkf5_7 will become sad... The easy solution is to put dTdt=0 (as I did). The
        # proper solution is to fiddle with everything so that every quantity is calculated
        # regardless of sab but sab decides what goes into the output x. Well, well, who cares?
        # We never used this variable anymore anyway. Found this when writing about the
        # ablation model for J-s thesis and wanted to comparde sputtering to ablation. The old
        # comp_abla_sput that uses early versions of the model still works though. D.S.
        # The mass loss equation to solve
        dmdt_s          = sputtering7(yvar, param, rho_a);                              #dmdt due to sputtering by Rogers et al.
        if isempty(dmdt_s); dmdt_s = 0; end                                             #Checking whether the equations give empty matrices -> putting in zero instead
        x               = [dvdt dmdt_s dsdt 0];                                           #Putting the calculated variables into a matrix
    end
    if sab == 2 | sab == 3
        # The mass loss and temperature equation to solve
        temp            = yvar(4);
        dmdt_a          = ablation(mass, temp, rho_m, A);                               #dmdt due to thermal ablation by Rogers et al. and Hill et al.
        # Calculating the heat transfer coeff Lambda by checking whether it's given or not (Gamma is needed before 
        # abla is calculated and abla is necessary for the Lambda calculations -> calculating them separatly)
        if glbn == 3 | glbn == 2
            gamlam      = gam_lam3(yvar, param, totnodens, dmdt_a, []);
            param(13)   = gamlam;
        end
        # Calculating the temperature
        dTdt            = temperature2(yvar, param, dmdt_a, rho_tot);
        if isempty(dmdt_a); dmdt_a = 0; end
        if isempty(dTdt);   dTdt   = 0; end
        x               = [dvdt dmdt_a dsdt dTdt];                                      #output
    end
    if sab == 3
        dmdt            = dmdt_a + dmdt_s;                                              #total mass loss
        x               = [dvdt dmdt dsdt dTdt];                                        #output
    end
    '''


def gam_lam3(mass, velocity, temperature, material, glbn, totnodens, abla, outvars):
    '''Calculates the drag coefficient Gamma and the heat transfer coefficient Lambda.


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

    v = velocity
    m = mass
    T = temperature

    rho_m 


    #Universal constants
    u = constants.u #[kg] atomic mass in kg
    kB = constants.value(u'Boltzmann constant') #[J/K]
    R = constants.value(u'molar gas constant') #[J/(mol K)]
    NA = constants.value(u'Avogadro constant') #[molecules/mol]


    '''
    #atmospheric mean molecular mass: projectile mass in kg [O N2 O2 He Ar H N], (see table 3 on p. 333 in Tielens et al)
    m_a = (15.9994 + 14.007*2 + 15.999*2 + 4.0026 + 39.948 + 1.0079 + 14.007) / 7 * u; #[kg]


    gamlam = []

    # Atmospheric mean free path, source: Westman et al.
    #Physics Handbook p. 186 This is the one used!
    mfp_inf = 1 ./ (pi * (3.62E-10)^2 * totnodens); # [m]

    mat_data = material_parameters(material)

    m2 = mat_data['m2']

    # The Knudsen number
    # Calculating the radius of the meteoroid at different altutude to get L
    #characteristic dimension of the body (the meteoroid); in particular, for a sphere L = the radius; Bronshten p. 31
    L               = ( (m / rho_m) / (4*pi/3) ).^(1/3) #[m] 
    Kn_inf          = mfp_inf ./ L;                                     #[dimensionless] the Knudsen no; calculated from 'mfp_inf'


    # Heat transfer coeff (Lambda) and drag coeff (Gamma) #

    #the relative masses of the molecules of the air and of the meteoroid, Bronshten p. 40
    mu_star = m_a / m2; #[dimensionless] 

    # thermal accommodation coefficient, Bronshten p. 40 eq. 7.11
    a_e = (3 + mu_star) * mu_star / (1 + mu_star)^2 #[dimensionless]

    #parameter... Bronshten p. 69 eq. 10.7
    h_star = m_a .* v.^2 ./ (2 * kB * T) #[dimensionless]

    #average velocity of reflected molecules near the meteoroid surface, Bronshten p. 37 eq. 7.1 (and 7.2)
    v_s = sqrt(3 * R * T / (m_a * NA)) #[m/s] 

    #nomalized reflected and evaporated velocity, Bronshten p. 69 but uncorrect in the book...
    u_s             = v_s ./ v;#[dimensionless] 

    #angle between the normal to the surface and the molecular flow, theta0 is a row vector
    theta0          = 0:pi/200:pi/2 #[radians] 
    theta0          = theta0(ones(1,length(u_s)),:)
    u_s             = u_s(:, ones(1, size(theta0,2)))


    # Gamma
    if isempty(abla) & ( glbn == 1 | glbn == 3 )
        epsilon_p   = 0.86;                                             #[dimensionless] Bronshten p. 69 eq. 10.7 and p. 71
        P0          = trapz(theta0(1,:), sin(theta0) .* cos(theta0) .* (1 - u_s), 2);                                   #integrating p0, Bronshten p. 71
        P_prim      = trapz(theta0(1,:), sin(theta0) .* 4 / sqrt(2*pi) .* sqrt(1 - 2 .* u_s .* cos(theta0) + u_s.^2) .* cos(theta0) .* (cos(theta0) .* (1 - u_s) / 2 - (1/24 .* (1+cos(theta0)).^3 - u_s./24.*(1+cos(theta0)).^2 .* (4-2*cos(theta0)+cos(theta0).^2))), 2);     #integrating p_prim over theta0 from 0 - pi/2, Bronshten p. 71
        
        
        
        
        A_Gamma_r   = 1 - epsilon_p .* sqrt(h_star) ./ Kn_inf .* P_prim ./ P0;                                          #[dimensionless] the momentum flux shielding coeff, Bronshten p. 71 same as a_Gamma but with the integrated vectors
        Gamma       = A_Gamma_r * a_e;                                  #[dimensionless] drag coeff

        # We don't want Gamma or Lambda to be > 1, searching for those and setting them = 1
        Gamma(find(Gamma > 1)) = 1;

        gamlam      = Gamma;                                            #since Gamma is calculated first, 'gamlam' is empty for surede
    end

    #  if outvars is given, abla is taken from it, but only after the if
    # statement checking wheterh to calculate Gamma or not
    if ~isempty(outvars)
        abla        = outvars(:,6);
    end


    #--------#
    # Lambda #
    #--------#
    if ( glbn == 2 | glbn == 3 ) & ~isempty(abla)
        # Shielding effects of a meteoroid surface due to reflected molecules:
        epsilon_q   = 1.6;                                              #[dimensionless]
        Q0          = trapz(theta0(1,:), sin(theta0) .* cos(theta0) .* (1 - u_s.^2), 2);                                #integrating q0
        Q_prim      = trapz(theta0(1,:), sin(theta0) .* 4 / sqrt(2*pi) .* sqrt(1 - 2 .* u_s .* cos(theta0) + u_s.^2) .* cos(theta0) .* (cos(theta0) .* (1 - u_s.^2) /2 - (1/48 .* (1+cos(theta0)).^3 .* (3-cos(theta0)) - u_s/48 .* (1+cos(theta0)).^2 .* (8-7.*cos(theta0)+2.*cos(theta0).^2-cos(theta0).^3))), 2);     #integrating q_prim over theta0 from 0 - pi/2; Bronshten p. 69, eq. 10.10 and Bronshten I eq. 11
        A_Lambda_r  = 1 - epsilon_q .* sqrt(h_star) ./ Kn_inf .* Q_prim ./ Q0;                                          #[dimensionless] the energy flux shielding coeff, Bronshten p. 71, same as a_Lambda but with the integrated vectors; Bronshten p. 69 eq. 10.8
        
        
        # Shielding effects of a meteoroid surface due to evaporated molecules:
        
        sigmaD      = 5.6E-15 * 10^(-1.6) * v.^(-0.8);                  #[m^2] scattering cross section for meteor atoms and ions on N2 and O2 molecules; Bronshten p. 84 eq. 11.29, Bronshten II. eq. 40; the actual equation is for v in [cm/s] and gives the cross section in cm^2 -> sigmaD = 5.6E-11 [cm^2] *(v [cm/s])^-0.8 = 5.6E-11*(1E-2[m])^2*(v [m/s]*1E2)^-0.8 = 5.6E-11*1E-4*(1E2)^-0.8*v^-0.8 = 5.6E-15*10^-1.6*v^-0.8
        v_e         = sqrt( 8 * kB * T / (pi * m2) );                   #[m/s] the velocity of molecules evaporated from the meteor body, Bronshten eq. 7.2 p. 37
        mfp_e       = v_e ./ (totnodens .* v .* sigmaD);                #[m] mean free path of the evaporated molecules, Bronshten p. 79 eq. 11.8; Bronshten II. eq. 2
        eta         = (L ./ mfp_e).^2 ./ (1 + 2 * sqrt(L ./ mfp_e));    #[dimensionless] the toatal fraction of evaporated molecules from the entire front surface of the meteoroid, Bronshten II. eq. 35
        N_i         = eta / m2 .* -abla;                                #[number] the total number of molecules evaporated from a certain area of cross section and which takes part in shielding, Bronshten II. eq. 42 and eq. 1
        S_m         = pi * L.^2;                                        #[m^2] cross section area of meteoroid (L = characteristic length, but here it is equal to the meteoroid radius)
        N_a         = totnodens .* v .* S_m;                            #[number] the number of molecules advancing on the area S, Bronshten II. p. 135
        
        v_s_e       = sqrt(3 * R * T / (m2 * NA));                      #[m/s] average velocity of evaporated molecules from the meteoroid surface, see v_s which is the same thing but for reflected molecules 
        u_e         = v_s_e ./ v;                                       #[dimensionless] nomalized velocity of evaporated molecules, compare to u_s
        u_e         = u_e(:, ones(1, size(theta0,2)));                  #instead of repmat(u_e, 1, size(theta0,2)); u_e was a column vector
        Q_star_e    = trapz(theta0(1,:), sin(theta0) .* 1/48 .* (1+cos(theta0)).^3 .* (3-cos(theta0)) - u_e/48 .* (1+cos(theta0)).^2 .* (8-7.*cos(theta0)+2.*cos(theta0).^2-cos(theta0).^3), 2); #same as Q_star but for evaporated molecules, i.e. using u_e instead of u_s
        Q0_e        = trapz(theta0(1,:), sin(theta0) .* cos(theta0) .* (1 - u_e.^2), 2);                                #same as Q0 but for evaporated molecues, i.e. using u_e instead of u_s
        
        zeta        = 1 - 2 * Q_star_e ./ Q0_e;                         #determines how many of the reflected molecules that are thrown back, Bronshten I. eq. 17
        A_Lambda_e  = 1 - zeta .* N_i ./ N_a;                           #[dimensionless] the energy flux shielding by evaporation coeff for a sphere, Bronshten II. eq. 43
        
        # How to add the shielding effects by reflection and evaporation? A_Lambda_r and A_Lambda_e 
        # is defined as the probability that an incoing molecule reaches the surface of the body and 
        # transfers its momentum or energy respectively to it. Thus 1 - A_Lambda_r and 1 - A_Lambda_e
        # is the probability that an incoming molecule does not reach the surface of the body, i.e., 
        # is shielded. So, to add how many of the incoming molecules contributes to the shielding both
        # by reflection and evaporation, we need to add the probability that the incoming molecules 
        # does not reach the surface: 1 - ( (1 - A_Lambda_r) + (1 - A_Lambda_e) ) = 
        # = 1 - 1 + A_Lambda_r - 1 + A_Lambda_e = A_lambda_r + A_lambda_e -1
        Lambda      = (A_Lambda_r + A_Lambda_e -1) * a_e;               #[dimensionless] värmeöverföringscoeff; heat transfer coeff
        
        # We don't want Gamma or Lambda to be > 1, searching for those and setting them = 1
        Lambda(find(Lambda > 1)) = 1;

        gamlam      = [gamlam Lambda];                                  #if glbn = 3, gamlam already contains Gamma, otherwhise, gamlam will only contain Lambda
    end
    '''

def luminosity2(zq, outvars):
    '''
    # Luminosity occurs in meteors as a result of decay of excited atomic (and
    # a few molecular) states following collisions between ablated meteor atoms
    # and atmospheric consituents.

    # References:
    # Friichtenicht and Becker: Determination of meteor paramters using laboratory
    # simulations techniques, 'Evolutionary and physical properties of meteoroids', 
    # National Astronautics and Space Administration, Chapter 6, p. 53-81 (1973)
    # Hill et al.: High geocentric velocity meteor ablation, A&A 444 p. 615-624 (2005)

    #---------------------------------------------------------------------------#
    # 2007-05-30 Csilla Szasz and Johan Kero: luminosity2.m
    # More bugs in Hill et al... this time in the calculation of zeta for different velocity intervals:
    # some intevals need the velocity in km/s and some in m/s... Inconsistent? Definitely. See also comments
    # in luminosity_testHILL.m

    # 2007-05-28 Csilla Szasz and Johan Kero: luminosity2.m
    # The original calculations are wrong because Hill et al.'s table is wrong: epsilon is the emissivity in the
    # table but should be the excitation energy. Thus our previous calculations have to be redun.
    # Param is no longer needed as input.

    # 2006-12-11 Csilla Szasz and Johan Kero: luminosity.m
    # Inparameters: zq       = the coulumns contain: [v m s T abla sput radius]
    #                        v       = [m/s] meteoroid velocity
    #                        m       = [kg] meteoroid mass
    #                        s       = [m] range 
    #                        T       = [K] meteoroid temperature
    #               param    = parameters that doesn't change for the one and the same meteoroid: [rho_m za hour h_obs rp h]
    #                         1. rho_m   = [kg/m3] meteoroid density 
    #                         2. za      = [degrees] zenith_angle at top of atmosphere
    #                         3. day     = day of observation
    #                         4. hour    = hour when the meteoroid came into the atmosphere centered at full hour (e.g. 4.34 -> 5, 5.25 -> 5)
    #                         5. h_obs   = [m] observational altitude 
    #                         6. rp      = [m] earth radius 
    #                         7. h       = starting value for time variable (not used, needed in Runge-Kutta funcion)
    #                         8. A       = shape factor
    #                         9. Gamma   = drag coefficient
    #                        10. G       = [Nm2/kg2 = m3/(s2*kg)] universal gravitational constant
    #                        11. M       = [kg] mass of earth
    #                        12. epsilon = emissivity
    #                        13. Lambda  = heat transfer coeff
    #                        14. rr_state = 0=off or 1=on; whether each step is printed on the screen as a running number or not
    #                        15. TL       = [dimensionless] TL=greatest Tolarable Local truncation error; if TL=NaN, default value of 1E-6 is used; the step size grows with TL 
    #                        16. sab      = 1=sputtering, 2=ablation, or 3=both: what to take into account when integrating through the atmospere
    #                        17. glbn     = 1=Gamma, 2=Lambda, 3=both, 4=none: to calculate either Gamma or Lamba (or both) or take the input value. It is set automatically in 'meteor_thru_atm2' depending on the values of Gamma and Lambda
    #                outvars = [step theta height t radius I abla]
    #                        step   = [seconds] step size
    #                        theta  = [radians] angle at earth center between the meteor when observed and when at 500 km
    #                        height = [m] meteoroid height above earth surface when at 500 km distance from detection
    #                        t      = [seconds] time
    #                        radius = [m] meteoroid radius
    #                        abla   = [kg/s] mass loss due to thermal ablation

    # Outparameter: I        = [W] luminous intensity (as a vector)

    #---------------------------------------------------------------------------#


    #-- Symbol definitions:
    #-- I       = [W] light intensity of a meteoroid, i.e. luminous intensity; radiant intensity
    #-- tau_I   = luminous efficiency factor
    #-- mu      = [kg] mean molecular mass of ablated material
    #-- v       = [m/s] meteoroid velocity
    #-- epsilon = emissivity
    #-- zeta    = excitation coeff
    #-- rho_m   = [kg/m3] meteoroid density
    #-- abla    = mass loss due to thermal ablation

    '''

    '''
    #-- Universal constants
    u         = 1.6605402E-27;                      #atomic mass in kg


    #-- Variables
    v           = zq(:,1);                          #[m/s] velocity
    vkm         = v/1E3;                            #[km/s] velocity
    abla        = outvars(:,6);
    #-- for visible meteors, the energy is in lines, it is believed that they are composed of iron (Cepleca p. 355, table 3)
    epsilon_mu  = 7.668E6;                             #[J/kg] mean excitation energy = epsilon/mu; mu = mean molecular mass


    #-- The excitation coeff for different velocity intervals:
    #-- See Hill et al. and references therein
    zeta              = zeros(1,length(v));
    v2                = find(v < 20000);
    if ~isempty(v2);      
        zeta(v2)      = -2.1887E-9 * v(v2).^2 + 4.2903E-13 * v(v2).^3 - 1.2447E-17 * v(v2).^4; 
    end

    v2_60             = find(v >= 20000 & v < 60000);
    if ~isempty(v2_60);   
        zeta(v2_60)   = 0.01333*vkm(v2_60).^1.25; 
    end

    v60_100           = find(v >= 60000 & v < 100000 );
    if ~isempty(v60_100); 
        zeta(v60_100) = -12.835 + 6.7672E-4*v(v60_100) - 1.163076E-8*v(v60_100).^2 + ...
                         9.191681E-14*v(v60_100).^3 - 2.7465805E-19*v(v60_100).^4; 
    end

    v100              = find(v >= 100000);
    if ~isempty(v100);    
        zeta(v100)    = 1.615 + 1.3725E-5*v(v100); 
    end


    #-- Luminous efficiency factor
    tau_I    = 2 * epsilon_mu .* zeta.T ./ v.^2;


    #-- The normal lumonous equation
    I = - 1/2 * tau_I .* abla .* v.^2; 

    '''