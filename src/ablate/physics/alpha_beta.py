import numpy as np
import scipy.optimize as sco
import scipy.special as scs
import scipy.interpolate as sci


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


def mass_direct(velocity, initial_mass, beta, shape_change_coefficient, initial_velocity=None):
    """Shorthand for `norm_mass_direct` scaled to physical units"""
    return initial_mass * norm_mass_direct(
        velocity, beta, shape_change_coefficient, initial_velocity=initial_velocity
    )


def norm_mass_direct(velocity, beta, shape_change_coefficient, initial_velocity=None):
    """The solution for mass from the analytic solutions of the ablation eqations [^1].

    $$
        M = M_e e^{- \\frac{\\beta}{1 - \\mu} \\left ( 1 - \\left ( \\frac{V}{V_e} \\right )^2 \\right ) }
    $$

    [^1]: Sansom EK, Gritsevich M, Devillepoix HAR, et al (2019)
        Determining Fireball Fates Using the α–β Criterion. ApJ 885:115.
        https://doi.org/10.3847/1538-4357/ab4516
    """
    v = velocity
    if initial_velocity is not None:
        v = v / initial_velocity
    return np.exp(-beta / (1 - shape_change_coefficient) * (1 - v**2))


def height_direct(velocity, atmospheric_scale_height, alpha, beta, initial_velocity=None):
    """Shorthand for `norm_mass_direct` scaled to physical units"""
    return atmospheric_scale_height * norm_height_direct(
        velocity, alpha, beta, initial_velocity=initial_velocity
    )


def norm_height_direct(velocity, alpha, beta, initial_velocity=None):
    """The solution for height as a function of velocity from the analytic
    solutions of the ablation eqations.
    """
    v = velocity
    if initial_velocity is not None:
        v = v / initial_velocity
    return np.log(alpha) + beta - np.log((scs.expi(beta) - scs.expi(beta * v**2)) * 0.5)


def velocity_direct(mass, beta, initial_velocity, shape_change_coefficient, initial_mass=None):
    """Shorthand..."""
    return initial_velocity * norm_velocity_direct(
        mass, beta, shape_change_coefficient, initial_mass=initial_mass
    )


def norm_velocity_direct(mass, beta, shape_change_coefficient, initial_mass=None):
    """Inverse of `norm_mass_direct`"""
    m = mass
    if initial_mass is not None:
        m = m / initial_mass
    return np.sqrt(np.abs(1 + np.log(m) * (1 - shape_change_coefficient) / beta))


def velocity_estimate(height, initial_velocity, alpha, beta, atmospheric_scale_height=None):
    """Shorthand for `norm_mass_direct` scaled to physical units"""
    return initial_velocity * norm_velocity_estimate(
        height, alpha, beta, atmospheric_scale_height=atmospheric_scale_height
    )


def norm_velocity_estimate(
    height,
    alpha,
    beta,
    atmospheric_scale_height=None,
    resolution=1000,
    edge_delta=1e-10,
    sampling=None,
):
    """The solution for height as a function of velocity from the analytic
    solutions of the ablation eqations.
    """
    y = height
    if atmospheric_scale_height is not None:
        y = y / atmospheric_scale_height

    if sampling is None:
        # Sample according to a rescaled inverse exponential line to match the
        # tangent space of the interpolated function
        v_grid = 1 - np.exp(np.linspace(np.log(1 - edge_delta), np.log(edge_delta), resolution))
    else:
        v_grid = sampling
    h_grid = norm_height_direct(v_grid, alpha, beta)
    interp = sci.interp1d(h_grid, v_grid, kind="linear", fill_value=np.nan, bounds_error=False)
    return interp(y)


def final_mass_direct(
    final_velocity,
    initial_mass,
    beta,
    shape_change_coefficient,
    initial_velocity=None,
):
    """Final mass based on the analytic solutions in
    [`mass_direct`][`ablate.ablation.alpha_beta.mass_direct`].
    """
    return mass_direct(
        final_velocity,
        initial_mass,
        beta,
        shape_change_coefficient,
        initial_velocity=initial_velocity,
    )


def atmosphere_density(height, atmospheric_scale_height, sea_level_rho):
    return sea_level_rho * np.exp(-height / atmospheric_scale_height)


def Q5(x, velocities, yvals):
    """ """
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


def solve_alpha_beta_velocity_versionQ5(
    velocities,
    heights,
    atmospheric_scale_height=None,
    start=None,
    bounds=((0.01, 1000000.0), (0.0001, 200.0), (0, None)),
):
    """Solve for alpha and beta using minimization of the least squares of the
    observables input into the preserved analytical relation.

    [^1]: In preparation.
    """
    Yvalues = heights
    if atmospheric_scale_height is not None:
        Yvalues = Yvalues / atmospheric_scale_height

    b0 = 1.0
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


def Q4(x, vvals, yvals):
    """ """
    res = 0.0
    for i in range(len(vvals)):
        r0 = 2 * x[0, ...] * np.exp(-yvals[i])
        r0 -= (scs.expi(x[1, ...]) - scs.expi(x[1, ...] * vvals[i] ** 2)) * np.exp(-x[1, ...])
        res += pow(r0, 2)
    return res


def solve_alpha_beta_versionQ4(
    velocities,
    heights,
    initial_velocity=None,
    atmospheric_scale_height=None,
    start=None,
    bounds=((0.001, 10000.0), (0.00001, 500.0)),
):
    """Solve for alpha and beta using minimization of the least squares of the
    observables input into the preserved analytical relation [^1].

    [^1]: Gritsevich MI (2008) Identification of fireball dynamic parameters.
        Moscow University Mechanics Bulletin 63:1-5.
        https://doi.org/10.1007/s11971-008-1001-5
    """

    Yvalues = heights
    if atmospheric_scale_height is not None:
        Yvalues = Yvalues / atmospheric_scale_height

    Vvalues = velocities
    if initial_velocity is not None:
        Vvalues = Vvalues / initial_velocity

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
