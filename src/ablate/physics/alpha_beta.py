import numpy as np
import scipy.optimize as sco
import scipy.special as scs


def alpha_direct(
    aerodynamic_cd,
    sea_level_rho,
    atmospheric_scale_height,
    initial_cross_section,
    initial_mass,
    radiant_local_elevation,
    degrees=False,
):
    """Ballistic coefficient used to describe the analytic solutions to the
    ablation equations described in [^1].

    $$
        \\alpha = \\frac{1}{2} c_d \\frac{\\rho_0 h_0 S_e}{M_e \\sin(\\gamma)}
    $$

    [^1]: Gritsevich MI (2009) Determination of parameters of meteor bodies based
        on flight observational data. Advances in Space Research 44:323-334.
        https://doi.org/10.1016/j.asr.2009.03.030

    Parameters
    ----------
    aerodynamic_cd : float or numpy.ndarray
        Aerodynamic drag coefficient [1]
    sea_level_rho : float or numpy.ndarray
        Atmospheric density of homogenous atmosphere at sea-level [kg/m^3].
    atmospheric_scale_height : float or numpy.ndarray
        Characteristic scale height of the homogenous atmosphere [m]
    initial_cross_section : float or numpy.ndarray
        Initial meteoroid cross-sectional area [m^2].
    initial_mass : float or numpy.ndarray
        Initial mass of the meteoroid [kg].
    radiant_local_elevation : float or numpy.ndarray
        Elevation angle of local radiant at start of ablation [deg].

    Returns
    -------
    float or numpy.ndarray
        Ballistic coefficient [1]

    """
    el = np.radians(radiant_local_elevation) if degrees else radiant_local_elevation
    return (
        0.5
        * aerodynamic_cd
        * sea_level_rho
        * atmospheric_scale_height
        * initial_cross_section
        / (initial_mass * np.sin(el))
    )


def beta_direct(
    shape_change_coefficient,
    heat_exchange_coefficient,
    initial_velocity,
    aerodynamic_cd,
    enthalpy_of_massloss,
):
    """Mass loss parameter used to describe the analytic solutions to the
    ablation equations described in [^1].

    $$
        \\beta = (1 - \\mu) \\frac{c_h V_e^2}{2c_d H^*}
    $$

    [^1]: Gritsevich MI (2009) Determination of parameters of meteor bodies based
        on flight observational data. Advances in Space Research 44:323-334.
        https://doi.org/10.1016/j.asr.2009.03.030

    Parameters
    ----------
    shape_change_coefficient : float or numpy.ndarray
        Coefficient $\\mu$ describing the relation between the *dimensionless*
        mass of the body and the *dimensionless* cross-sectional (i.e. projected)
        area by $s = m^{\\mu}$. Since $s,m$ are the dimensionless quantities
        normalized by the initial state, $\\mu$ also characterizes the possible
        role of rotation during the flight. It is assumed as constant during ablation.
    heat_exchange_coefficient : float or numpy.ndarray
        The amount of the heat flux from the ambient gas that is spent
        for ablating the object [1].
    initial_velocity : float or numpy.ndarray
        Initial meteoroid entry-velocity before interaction with the Earth's atmosphere [m/s].
    aerodynamic_cd : float or numpy.ndarray
        Aerodynamic drag coefficient [1]
    enthalpy_of_massloss : float or numpy.ndarray
        Initial mass of the meteoroid [J/kg].

    Returns
    -------
    float or numpy.ndarray
        Mass loss parameter [1]

    """
    return (
        (1 - shape_change_coefficient)
        * heat_exchange_coefficient
        * initial_velocity**2
        / (2 * aerodynamic_cd * enthalpy_of_massloss)
    )


def ablation_coefficient(beta, shape_change_coefficient, initial_velocity):
    """Ablation coefficient calculated from the parameters of the
    analytic solutions to the ablation equations described in [^1].

    $$
        \\sigma = \\frac{2\\beta}{(1 - \\mu) V_e^2}
    $$


    [^1]: Gritsevich MI (2009) Determination of parameters of meteor bodies based
        on flight observational data. Advances in Space Research 44:323-334.
        https://doi.org/10.1016/j.asr.2009.03.030

    Parameters
    ----------
    beta : float or numpy.ndarray
        Mass loss parameter used to describe the analytic solutions to the
        ablation equations.
    shape_change_coefficient : float or numpy.ndarray
        Coefficient $\\mu$ describing the relation between the *dimensionless*
        mass of the body and the *dimensionless* cross-sectional (i.e. projected)
        area by $s = m^{\\mu}$. Since $s,m$ are the dimensionless quantities
        normalized by the initial state, $\\mu$ also characterizes the possible
        role of rotation during the flight. It is assumed as constant during ablation.
    initial_velocity : float or numpy.ndarray
        Initial meteoroid entry-velocity before interaction with the Earth's atmosphere [m/s].

    Returns
    -------
    float or numpy.ndarray
        Ablation coefficient [1]

    """
    return 2 * beta / ((1 - shape_change_coefficient) * initial_velocity**2)


def area_to_mass_ratio(
    alpha,
    aerodynamic_cd,
    sea_level_rho,
    atmospheric_scale_height,
    radiant_local_elevation,
    degrees=False,
):
    """Area to mass ratio from alpha."""
    el = np.radians(radiant_local_elevation) if degrees else radiant_local_elevation
    sin_gamma = np.sin(el)
    return 2 * alpha * sin_gamma / (aerodynamic_cd * sea_level_rho * atmospheric_scale_height)


def ballistic_coefficient(bulk_density, aerodynamic_cd, characteristic_length):
    return bulk_density * characteristic_length / aerodynamic_cd


def shape_factor_direct(initial_cross_section, initial_mass, bulk_density):
    """
    Shape factor $ A_e $ is defined as the ratio between the body cross-sectional
    area $ S_e $ and its volume $ W_e^{\\frac{2}{3}} $ to the 2/3'd power [^1]. To not
    have to work with volume, the formula is re-written using the bulk density to

    $$
        A_e = \\frac{S_e}{W_e^{\\frac{2}{3}}} = S_e\\frac{\\rho^{\\frac{2}{3}}}{M_e^{\\frac{2}{3}}}
    $$

    [^1]: Gritsevich MI (2009) Determination of parameters of meteor bodies based
        on flight observational data. Advances in Space Research 44:323-334.
        https://doi.org/10.1016/j.asr.2009.03.030

    """
    return initial_cross_section * (bulk_density / initial_mass) ** (2.0 / 3.0)


def initial_mass_direct(
    alpha,
    aerodynamic_cd,
    sea_level_rho,
    atmospheric_scale_height,
    radiant_local_elevation,
    bulk_density,
    shape_factor,
    degrees=False,
):
    """Directly compute initial mass from analytic solution coefficients"""

    el = np.radians(radiant_local_elevation) if degrees else radiant_local_elevation
    sin_gamma = np.sin(el)
    return (
        0.5
        * aerodynamic_cd
        * sea_level_rho
        * atmospheric_scale_height
        * shape_factor
        / (alpha * sin_gamma * bulk_density ** (2.0 / 3.0))
    ) ** 3


def mass_direct(velocity, initial_velocity, initial_mass, beta, shape_change_coefficient):
    """The solution for mass from the analytic solutions of the ablation eqations [^1].

    $$
        M = M_e e^{- \\frac{\\beta}{1 - \\mu} \\left ( 1 - \\left ( \\frac{V}{V_e} \\right )^2 \\right ) }
    $$

    [^1]: Sansom EK, Gritsevich M, Devillepoix HAR, et al (2019)
        Determining Fireball Fates Using the α–β Criterion. ApJ 885:115.
        https://doi.org/10.3847/1538-4357/ab4516
    """
    return initial_mass * np.exp(
        -beta / (1 - shape_change_coefficient) * (1 - (velocity / initial_velocity) ** 2)
    )


def final_mass_direct(
    final_velocity, initial_velocity, initial_mass, beta, shape_change_coefficient
):
    """Final mass based on the analytic solutions in
    [`mass_direct`][`ablate.ablation.alpha_beta.mass_direct`].
    """
    return mass_direct(
        final_velocity, initial_velocity, initial_mass, beta, shape_change_coefficient
    )


def solve_alpha_beta_velocity_versionQ5(
    velocities,
    heights,
    initial_height=None,
    start=None,
    bounds = ((0.01, 1000000.0), (0.0001, 200.0), (0, None)),
):
    """Solve for alpha and beta using minimization of the least squares of the
    observables input into the preserved analytical relation.

    [^1]: In preparation.
    """

    def Q5(x, velocities, yvals):
        if len(x.shape) == 1:
            x.shape = (x.size, 1)
        size = x.shape[1]

        res = np.zeros((size,))
        for i in range(len(velocities)):
            vval = velocities[i] / (x[2, ...] * 1000.0)
            r0 = 2 * x[0, ...] * np.exp(-yvals[i])
            r0 -= (scs.expi(x[1, ...]) - scs.expi(x[1, ...] * vval**2)) * np.exp(-x[1, ...])
            inds = np.logical_not(np.isnan(r0))
            res[inds] += r0[inds] ** 2
        if res.size == 1:
            return res[0]
        else:
            return res

    if initial_height is None:
        initial_height = heights[0]

    Yvalues = heights / initial_height

    b0 = 0.01
    a0 = np.exp(Yvalues[-1]) / (2.0 * b0)
    # /1000 is a hack to make velocities small so minimisation doesnt use stupid steps
    v0 = velocities[0] / 1000
    if start is None:
        start = [a0, b0, v0]
    else:
        if start[1] is None:
            start[1] = b0
        if start[0] is None:
            start[0] = np.exp(Yvalues[-1]) / (2.0 * start[1])
        if start[2] is None:
            start[2] = v0

    res = sco.minimize(Q5, start, args=(velocities, Yvalues), bounds=bounds)
    out = res.x
    out[2] *= 1000.0  # fix velocities for return
    return out


def solve_alpha_beta_versionQ4(
    velocities,
    heights,
    initial_velocity=None,
    initial_height=None,
    start=None,
    bounds=((0.001, 10000.0), (0.00001, 500.0)),
):
    """Solve for alpha and beta using minimization of the least squares of the
    observables input into the preserved analytical relation [^1].

    [^1]: Gritsevich MI (2008) Identification of fireball dynamic parameters.
        Moscow University Mechanics Bulletin 63:1-5.
        https://doi.org/10.1007/s11971-008-1001-5
    """

    def Q4(x, vvals, yvals):
        res = 0.0
        for i in range(len(vvals)):
            r0 = 2 * x[0, ...] * np.exp(-yvals[i])
            r0 -= (scs.expi(x[1, ...]) - scs.expi(x[1, ...] * vvals[i] ** 2)) * np.exp(-x[1, ...])
            res += pow(r0, 2)
        return res

    if initial_velocity is None:
        initial_velocity = velocities[0]

    if initial_height is None:
        initial_height = heights[0]

    Vvalues = velocities / initial_velocity
    Yvalues = heights / initial_height

    b0 = 1.0
    a0 = np.exp(Yvalues[-1]) / (2.0 * b0)
    if start is None:
        start = [a0, b0]
    else:
        if start[1] is None:
            start[1] = b0
        if start[0] is None:
            start[0] = np.exp(Yvalues[-1]) / (2.0 * start[1])

    res = sco.minimize(Q4, start, args=(Vvalues, Yvalues), bounds=bounds)
    return res.x
