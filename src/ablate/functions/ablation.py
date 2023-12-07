#!/usr/bin/env python

"""
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


"""
import logging
import numpy as np
from scipy import constants
import scipy.optimize as sco
import scipy.special as scs


logger = logging.getLogger(__name__)


def alpha_beta_entry_mass(alpha, beta, slope, aerodynamic_cd, density, shape_coef, sea_level_rho=1.29):
    """ TODO: validate, docstring and make sure no coefficients are left over """
    sin_gamma = np.sin(slope)
    return (0.5*aerodynamic_cd*sea_level_rho*7160*shape_coef/(alpha*sin_gamma*density**(2/3.0)))**3


def alpha_beta_final_mass(entry_mass, beta, shape_change_coef, final_norm_vel):
    """ TODO: validate, docstring and make sure no coefficients are left over """
    return entry_mass*np.exp(-beta/(1 - shape_change_coef)*(1 - final_norm_vel**2))


def alpha_beta_velocity_Q4_min(velocities, heights, h0, v0):
    """TODO docstring

    # TODO: proper default and customizable limits
    """
    Yvalues = heights/h0  # normalisation of heights here
    b0 = 0.01
    a0 = np.exp(Yvalues[-1]) / (2.0 * b0)
    x0 = [a0, b0, v0 / 1000]

    # /1000 is a hack to make velocities small so minimisation doesnt use stupid steps
    xmin = [0.01, 0.0001, v0 * 0.7 / 1000]
    xmax = [1000000.0, 200.0, v0 * 1.3 / 1000]

    bnds = ((xmin[0], xmax[0]), (xmin[1], xmax[1]), (xmin[2], xmax[2]))

    res = sco.minimize(alpha_beta_velocity_min_fun, x0, args=(velocities, Yvalues), bounds=bnds)
    out = res.x
    out[2] *= 1000.0  # fix velocities for return
    return out


def alpha_beta_velocity_min_fun(x, velocities, yvals):
    """TODO docstring

    # TODO: this can be vectorized for speed in the future

    """
    if len(x.shape) == 1:
        x.shape = (x.size, 1)
    size = x.shape[1]

    res = np.zeros((size,))
    for i in range(len(velocities)):
        vval = velocities[i] / (x[2, ...] * 1000.0)
        r0 = 2*x[0, ...]*np.exp(-yvals[i])
        r0 -= (scs.expi(x[1, ...]) - scs.expi(x[1, ...] * vval**2))*np.exp(-x[1, ...])
        inds = np.logical_not(np.isnan(r0))
        res[inds] += r0[inds]**2
    if res.size == 1:
        return res[0]
    else:
        return res


def alpha_beta_Q4_min(Vvalues, Yvalues):
    """Solve for alpha and beta using Q4 minimization.
    TODO: validate, docstring and make sure no coefficients are left over
    Reference: Gritsevich 2007 ( https://doi.org/10.1134/S1028335808020110 )
    """

    b0 = 1.0
    a0 = np.exp(Yvalues[-1])/(2.0*b0)
    x0 = [a0, b0]
    xmin = [0.001, 0.00001]
    xmax = [10000.0, 500.0]

    bnds = ((xmin[0], xmax[0]), (xmin[1], xmax[1]))

    res = sco.minimize(
        alpha_beta_min_fun, x0, args=(Vvalues, Yvalues), bounds=bnds
    )
    return res.x


def alpha_beta_min_fun(x, vvals, yvals):
    """alpha and beta Q4 minimization function.
    TODO: validate, docstring and make sure no coefficients are left over
    Reference: Gritsevich 2007 ( https://doi.org/10.1134/S1028335808020110 )
    """
    res = 0.0
    for i in range(len(vvals)):
        r0 = 2*x[0, ...]*np.exp(-yvals[i])
        r0 -= (scs.expi(x[1, ...]) - scs.expi(x[1, ...]*vvals[i]**2))*np.exp(-x[1, ...])
        res += pow(r0, 2)
    return res


def luminosity(velocity, thermal_ablation):
    """Luminosity during thermal ablation.

    :param float/numpy.ndarray velocity: Meteoroid velocity [m/s]
    :param float/numpy.ndarray thermal_ablation: Mass loss due to thermal ablation [kg/s]

    :rtype: float/numpy.ndarray
    :return: Luminosity [W]


    Luminosity occurs in meteors as a result of decay of excited atomic (and
    a few molecular) states following collisions between ablated meteor atoms
    and atmospheric constituents.

    **References:**

       *Friichtenicht and Becker: Determination of meteor paramters using laboratory
            simulations techniques, 'Evolutionary and physical properties of meteoroids',
       *National Astronautics and Space Administration, Chapter 6, p. 53-81 (1973)
       *Hill et al.: High geocentric velocity meteor ablation, A&A 444 p. 615-624 (2005)


    **Symbol definitions:**

       *I = [W] light intensity of a meteoroid, i.e. luminous intensity; radiant intensity
       *tau_I = luminous efficiency factor
       *mu = [kg] mean molecular mass of ablated material
       *v = [m/s] meteoroid velocity
       *epsilon = emissivity
       *zeta = excitation coeff
       *rho_m = [kg/m3] meteoroid density
       *abla = mass loss due to thermal ablation

    """

    # -- Universal constants

    # -- Variables
    v = velocity  # [m/s] velocity
    vkm = v/1e3  # [km/s] velocity
    abla = thermal_ablation

    # -- for visible meteors, the energy is in lines, it is believed that they
    # are composed of iron (Cepleca p. 355, table 3)
    epsilon_mu = (
        7.668e6  # [J/kg] mean excitation energy = epsilon/mu; mu = mean molecular mass
    )

    # -- The excitation coeff for different velocity intervals:
    # -- See Hill et al. and references therein
    zeta = np.zeros(v.shape, dtype=v.dtype)
    ind = v < 20000

    zeta[ind] = -2.1887e-9*v[ind]**2 + 4.2903e-13*v[ind]**3 - 1.2447e-17*v[ind]**4

    ind = np.logical_and(v >= 20000, v < 60000)
    zeta[ind] = 0.01333*vkm[ind]**1.25

    ind = np.logical_and(v >= 60000, v < 100000)
    zeta[ind] = -12.835 + 6.7672e-4*v[ind] - 1.163076e-8*v[ind]**2
    zeta[ind] += 9.191681e-14*v[ind]**3 - 2.7465805e-19*v[ind]**4

    ind = v >= 100000
    zeta[ind] = 1.615 + 1.3725e-5*v[ind]

    # -- Luminous efficiency factor
    tau_I = 2*epsilon_mu*zeta/v**2

    # -- The normal lumonous equation
    intensity = -0.5*tau_I*abla*v**2

    return intensity


def temperature_rate(
    mass,
    velocity,
    temperature,
    material_data,
    shape_factor,
    atm_total_density,
    thermal_ablation,
    Lambda,
    atm_temperature=280,
    emissivity=0.9,
):
    """Calculates the rate of change of temperature of the meteoroid.
    A homogeneous metoroid experiencing an isotropic heat flux is assumed as
    well as the meteoroid undergoing isothermal heating. (isothermal heating:
    here: dTdS = 0 i.e. same spatial temperature)


    :param float/numpy.ndarray mass: Meteoroid mass [kg]
    :param float/numpy.ndarray velocity: Meteoroid velocity [m/s]
    :param float/numpy.ndarray temperature: Meteoroid temperature [K]
    :param dict material_data: Meteoroid material data, see :mod:`~functions.material.material_data`.
    :param float/numpy.ndarray shape_factor: Shape factor [1]
    :param float/numpy.ndarray atm_total_density: Total atmospheric number density [1/m^3]
    :param float/numpy.ndarray thermal_ablation: Mass loss due to thermal ablation [kg/s]
    :param float/numpy.ndarray Lambda: Heat transfer coefficient [1]
    :param float/numpy.ndarray atm_temperature: Effective atmospheric temperature [K]
    :param float/numpy.ndarray emissivity: Electromagnetic emissivity of meteoroid [1]

    :rtype: float/numpy.ndarray
    :return: Rate of change of temperature [K/s]


    **Default values:**

       *atm_temperature: ?
       *emissivity: 0.9 from Hill et al.; Love & Brownlee: 1; 0.2 is
            characteristic for a metal, oxides are between 0.4 and 0.8


    **References:**

       *Hill et al.: High geocentric velocity meteor ablation, A&A 444 p. 615-624 (2005)


    **Change log:**

       *Changes post-python port: See git-log
       *Changes pre-python port: See `matlab-version <https://gitlab.irf.se/kero/ablation_matlab>`_


    **Heat balance equation per cross-sectional area:**

       *A = shape factor
       *c = specific heat of meteoroid
       *mass = meteoroid mass
       *rho_m = meteoroid density
       *Lambda = heat transfer coefficient
       *rho_tot= total atmospheric mass density
       *vel = meteoroid velocity
       *kb = Stefan-Bolzmann constant
       *T = meteoroid temperature
       *Ta = effective atmospheric temperature
       *L = latent heat of fusion + vapourization
       *thermal_ablation = mass loss due to thermal ablation


    """

    # -- variables
    kB = constants.value("Boltzmann constant")  # [J/K]

    epsilon = emissivity

    Ta = atm_temperature  # [K] effective atmospheric temperature

    vel = velocity
    T = temperature
    A = shape_factor
    rho_tot = atm_total_density

    rho_m = material_data["rho_m"]
    c = material_data["c"]
    L = material_data["L"]

    coef0 = 0.5*Lambda*rho_tot*vel**3 - 4*kB*epsilon*(T**4 - Ta**4)
    coef0 += L/A*(rho_m/mass)**(2.0/3.0)*thermal_ablation

    coef1 = c*mass**(1.0/3.0)*rho_m**(2.0/3.0)

    dTdt = (A/coef1)*coef0
    return dTdt


def thermal_ablation(mass, temperature, material_data, shape_factor):
    """Calculates the mass loss for meteoroids due to thermal ablation.

    :param float/numpy.ndarray mass: Meteoroid mass [kg]
    :param float/numpy.ndarray temperature: Meteoroid temperature [K]
    :param dict material_data: Meteoroid material data, see :mod:`~functions.material.material_data`.
    :param float/numpy.ndarray shape_factor: Shape factor [1]


    :rtype: float/numpy.ndarray
    :return: Mass loss due to thermal ablation [kg/s]


    **Reference:**

       *Rogers et al.: Mass loss due to  sputtering and thermal processes in
            meteoroid ablation, Planetary and Space Science 53 p. 1341-1354 (2005)
       *Hill et al.: High geocentric velocity meteor ablation, A&A 444 p. 615-624 (2005)


    **Change log:**

       *Changes post-python port: See git-log
       *Changes pre-python port: See `matlab-version <https://gitlab.irf.se/kero/ablation_matlab>`_


    **Symbol definintions:**

       *Pv: vapour pressure of meteoroid [N/m^2]
       *CA, CB: Clausius-Clapeyron coeff [K]
       *mu: mean molecular mass of ablated vapour [kg]
       *rho_m: meteoroid density [kg/m3]


    """

    CA = material_data["CA"]
    CB = material_data["CB"]
    mu = material_data["mu"]
    rho_m = material_data["rho_m"]

    # 2007-03-21 Ed suggests that the vapor pressure should be lowered by a factor of 0.8 or 0.7
    # due to the large ram pressure forcing the evaporated atoms and molecules back on to the surface
    # thus leading to evaporation at a pressure that is lower than the equilibrium vapor pressure.

    Pv = 10.0**(CA - CB/temperature)
    # in [d/cm2]...; d=dyne=10-5 Newton: the force required to accelearte a
    # mass of one g at a rate of one cm per s^2
    Pv = Pv*1e-5/1e-4  # Convert to [N/m2]

    coef0 = np.sqrt(mu/(2.0*np.pi*constants.k*temperature))
    dmdt = -4.0*shape_factor*(mass/rho_m)**(2.0/3.0)*Pv*coef0  # [kg/s]

    return dmdt
