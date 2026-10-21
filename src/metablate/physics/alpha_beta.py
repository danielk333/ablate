from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable

import numpy as np
import scipy.interpolate as sci
import scipy.optimize as sco
import scipy.special as scs

from ..types import (
    ArrayLike,
    AtmosphereData,
    Bounds,
    Kwargs,
    NDArray,
    NDArray_N,
    Numeric,
)

if TYPE_CHECKING:
    from ..atmosphere import Atmosphere


def alpha_direct(
    aerodynamic_cd: Numeric,
    sea_level_rho: Numeric,
    atmospheric_scale_height: Numeric,
    initial_cross_section: Numeric,
    initial_mass: Numeric,
    radiant_local_elevation: Numeric,
    degrees: bool = False,
) -> Numeric:
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
    shape_change_coefficient: Numeric,
    heat_exchange_coefficient: Numeric,
    initial_velocity: Numeric,
    aerodynamic_cd: Numeric,
    enthalpy_of_massloss: Numeric,
) -> Numeric:
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


def ablation_coefficient(
    beta: Numeric, shape_change_coefficient: Numeric, initial_velocity: Numeric
) -> Numeric:
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
    alpha: Numeric,
    aerodynamic_cd: Numeric,
    sea_level_rho: Numeric,
    atmospheric_scale_height: Numeric,
    radiant_local_elevation: Numeric,
    degrees: bool = False,
) -> Numeric:
    """Area to mass ratio from alpha."""
    el = np.radians(radiant_local_elevation) if degrees else radiant_local_elevation
    sin_gamma = np.sin(el)
    return 2 * alpha * sin_gamma / (aerodynamic_cd * sea_level_rho * atmospheric_scale_height)


def alpha_from_ballistic_coefficient(
    ballistic_coefficient: Numeric,
    sea_level_rho: Numeric,
    atmospheric_scale_height: Numeric,
    radiant_local_elevation: Numeric,
    degrees: bool = False,
) -> Numeric:
    el = np.radians(radiant_local_elevation) if degrees else radiant_local_elevation
    return sea_level_rho * atmospheric_scale_height / (2 * ballistic_coefficient * np.sin(el))


def ballistic_coefficient_from_alpha(
    alpha: Numeric,
    sea_level_rho: Numeric,
    atmospheric_scale_height: Numeric,
    radiant_local_elevation: Numeric,
    degrees: bool = False,
) -> Numeric:
    el = np.radians(radiant_local_elevation) if degrees else radiant_local_elevation
    return sea_level_rho * atmospheric_scale_height / (2 * alpha * np.sin(el))


def mass_to_radius(mass: Numeric, bulk_density: Numeric) -> Numeric:
    return np.cbrt(mass / bulk_density * 3 / (4 * np.pi))


def radius_to_mass(radius: Numeric, bulk_density: Numeric) -> Numeric:
    return 4 * bulk_density * np.pi * radius**3 / 3


def shape_factor_direct(
    initial_cross_section: Numeric, initial_mass: Numeric, bulk_density: Numeric
) -> Numeric:
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
    alpha: Numeric,
    aerodynamic_cd: Numeric,
    sea_level_rho: Numeric,
    atmospheric_scale_height: Numeric,
    radiant_local_elevation: Numeric,
    bulk_density: Numeric,
    shape_factor: Numeric,
    degrees: bool = False,
) -> Numeric:
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


def mass_direct(
    velocity: Numeric,
    initial_mass: Numeric,
    beta: Numeric,
    shape_change_coefficient: Numeric,
    initial_velocity: Numeric | None = None,
) -> Numeric:
    """Shorthand for `norm_mass_direct` scaled to physical units"""
    return initial_mass * norm_mass_direct(
        velocity, beta, shape_change_coefficient, initial_velocity=initial_velocity
    )


def norm_mass_direct(
    velocity: Numeric,
    beta: Numeric,
    shape_change_coefficient: Numeric,
    initial_velocity: Numeric | None = None,
) -> Numeric:
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


def height_direct(
    velocity: Numeric,
    atmospheric_scale_height: Numeric,
    alpha: Numeric,
    beta: Numeric,
    initial_velocity: Numeric | None = None,
) -> Numeric:
    """Shorthand for `norm_height_direct` scaled to physical units"""
    return atmospheric_scale_height * norm_height_direct(
        velocity, alpha, beta, initial_velocity=initial_velocity
    )


def approx_norm_velocity(
    height: Numeric,
    alpha: float,
    beta: float,
    atmospheric_scale_height: Numeric | None = None,
    start: float = 1 - 1e-12,
    root_find_kwargs: Kwargs = {"tol": 1e-12, "maxiter": 200},
) -> Numeric:
    """Approximate solution for velocity for large beta as a function of height from the analytic
    solutions of the ablation eqations.
    """
    Yvalues = height
    if atmospheric_scale_height is not None:
        Yvalues = Yvalues / atmospheric_scale_height

    if not isinstance(Yvalues, np.ndarray):
        Yvalues = np.array([Yvalues])

    def func_exact(v: Numeric, y: Numeric) -> Numeric:
        return np.log(alpha) + 0.83 * beta * (1 - v) - np.log(-np.log(v)) - y

    def abs_func_exact(v: Numeric, y: Numeric) -> Numeric:
        return np.abs(np.log(alpha) + 0.83 * beta * (1 - v) - np.log(-np.log(v)) - y)

    def dabs_func_exact(v: Numeric, y: Numeric) -> Numeric:
        return (-0.83 * beta - 1 / (v * np.log(v))) * np.sign(func_exact(v, y))

    Vvalues = np.zeros_like(Yvalues)
    for ind in range(len(Yvalues)):
        try:
            Vvalues[ind] = sco.newton(
                abs_func_exact,
                start,
                fprime=dabs_func_exact,
                args=(Yvalues[ind],),
                **root_find_kwargs,
            )
        except RuntimeError:
            Vvalues[ind] = np.nan

    if not isinstance(height, np.ndarray):
        Vvalues = Vvalues[0]
    return Vvalues


def approx_norm_height_direct(
    velocity: Numeric, alpha: Numeric, beta: Numeric, initial_velocity: Numeric | None = None
) -> Numeric:
    """Approximate solution for height for large beta as a function of velocity from the analytic
    solutions of the ablation eqations.
    """
    v = velocity
    if initial_velocity is not None:
        v = v / initial_velocity
    return np.log(alpha) + 0.83 * beta * (1 - v) - np.log(-np.log(v))


def norm_height_direct(
    velocity: Numeric, alpha: Numeric, beta: Numeric, initial_velocity: Numeric | None = None
) -> Numeric:
    """The solution for height as a function of velocity from the analytic
    solutions of the ablation eqations.
    """
    v = velocity
    if initial_velocity is not None:
        v = v / initial_velocity
    return np.log(alpha) + beta - np.log((scs.expi(beta) - scs.expi(beta * v**2)) * 0.5)


def velocity_direct(
    mass: Numeric,
    beta: Numeric,
    initial_velocity: Numeric,
    shape_change_coefficient: Numeric,
    initial_mass: Numeric | None = None,
) -> Numeric:
    """Shorthand..."""
    return initial_velocity * norm_velocity_direct(
        mass, beta, shape_change_coefficient, initial_mass=initial_mass
    )


def norm_velocity_direct(
    mass: Numeric,
    beta: Numeric,
    shape_change_coefficient: Numeric,
    initial_mass: Numeric | None = None,
) -> Numeric:
    """Inverse of `norm_mass_direct`"""
    m = mass if isinstance(mass, np.ndarray) else np.array([mass])
    if initial_mass is not None:
        m = m / initial_mass
    low_lim = np.exp(-beta / (1 - shape_change_coefficient))
    inds = m >= low_lim
    if np.any(np.logical_not(inds)):
        warnings.warn(f"Input masses below theoretical limit\n{m[np.logical_not(inds)]} < {low_lim}")
    vel = np.full_like(m, np.nan)
    vel[inds] = np.sqrt(1 + np.log(m[inds]) * (1 - shape_change_coefficient) / beta)
    vel = vel if isinstance(mass, np.ndarray) else vel[0]
    return vel


def velocity_estimate(
    height: Numeric,
    initial_velocity: Numeric,
    alpha: float,
    beta: float,
    atmospheric_scale_height: Numeric | None = None,
) -> Numeric:
    """Shorthand for `norm_velocity_estimate` scaled to physical units"""
    return initial_velocity * norm_velocity_estimate(
        height, alpha, beta, atmospheric_scale_height=atmospheric_scale_height
    )


def norm_velocity_estimate_brute(
    height: Numeric,
    alpha: Numeric,
    beta: Numeric,
    atmospheric_scale_height: Numeric | None = None,
    resolution: int = 1000,
    edge_delta: Numeric = 1e-10,
    sampling: NDArray | None = None,
) -> Numeric:
    """Brute force approach to estimating velocity as a function of height from the analytic
    solutions of the ablation equations.
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


def norm_velocity_estimate(
    height: Numeric,
    alpha: float,
    beta: float,
    atmospheric_scale_height: Numeric | None = None,
    start: float = 1 - 1e-12,
    minimize_kwargs: Kwargs = {
        "method": "Nelder-Mead",
        "bounds": [(0, 1)],
        "options": {"xatol": 1e-12, "fatol": 1e-12, "maxiter": 1000},
    },
) -> Numeric:
    """Root-solver estimation of velocity as a function of height from the analytic
    solutions of the ablation equations.
    """
    Yvalues = height
    if atmospheric_scale_height is not None:
        Yvalues = Yvalues / atmospheric_scale_height

    if not isinstance(Yvalues, np.ndarray):
        Yvalues = np.array([Yvalues])

    def func_exact(v: Numeric, y: Numeric) -> Numeric:
        return np.abs(np.log(alpha) + beta - np.log((scs.expi(beta) - scs.expi(beta * v**2)) * 0.5) - y)

    # If beta is larger than 500, the expi estimation algorithms start having trouble
    # as they use float64, we need scs.expi(beta) to work to compute the difference value
    if beta > 600:
        if isinstance(height, np.ndarray):
            return np.full_like(height, np.nan)
        else:
            return np.nan

    Vvalues = np.zeros_like(Yvalues)
    for ind in range(len(Yvalues)):
        res = sco.minimize(func_exact, start, args=(Yvalues[ind],), **minimize_kwargs)
        Vvalues[ind] = res.x[0]

    if not isinstance(height, np.ndarray):
        Vvalues = Vvalues[0]
    return Vvalues


def final_mass_direct(
    final_velocity: Numeric,
    initial_mass: Numeric,
    beta: Numeric,
    shape_change_coefficient: Numeric,
    initial_velocity: Numeric | None = None,
) -> Numeric:
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


def density_to_height_exponential_atm(
    atm_total_mass_density: Numeric, atmospheric_scale_height: Numeric, sea_level_rho: Numeric
) -> Numeric:
    """Calculate height at a given atmospheric mass density given an exponential atmosphere."""
    return -np.log(atm_total_mass_density / sea_level_rho) * atmospheric_scale_height


def scale_hight_to_exponential_atm(
    height: Numeric,
    atmosphere: Atmosphere,
    sea_level_rho: Numeric,
    atmospheric_scale_height: Numeric,
    density_args: tuple[ArrayLike, ...] = (),
    density_kwargs: Kwargs = {},
    lims: tuple[float, float] | None = None,
    root_find_kwargs: Kwargs = {},
) -> Numeric:
    """Scale heights using an arbitrary but strictly decreeing atmospheric
    density model as a function of the original height, so that the new heights gives the
    same density using a simple exponential atmospheric density model.
    """
    density: Callable[..., AtmosphereData] = atmosphere.density
    d = density(*density_args, alt=height, **density_kwargs)
    atm_dens = d["Total"].values.squeeze()

    scaled_height = density_to_height_exponential_atm(
        atm_dens,
        atmospheric_scale_height,
        sea_level_rho,
    )
    return scaled_height


def scale_hight_to_model_atm(
    height: NDArray_N,
    atmosphere: Atmosphere,
    sea_level_rho: Numeric,
    density_args: tuple[ArrayLike, ...] = (),
    density_kwargs: Kwargs = {},
    atmospheric_scale_height: Numeric | None = None,
    lims: tuple[float, float] | None = None,
    root_find_kwargs: Kwargs = {},
) -> NDArray_N:
    """Scale heights from an exponential model to an arbitrary but strictly decreeing atmospheric
    density model as a function of the new height, so that the new heights in the atmospheric
    model gives the same density as the simple exponential model.
    """
    Yvalues = height
    if atmospheric_scale_height is not None:
        Yvalues = Yvalues / atmospheric_scale_height

    density: Callable[..., AtmosphereData] = atmosphere.density

    def get_density(h: Numeric) -> Numeric:
        d = density(*density_args, alt=h, **density_kwargs)
        return d["Total"].values.squeeze()

    def func_exact(h: Numeric, y: Numeric) -> Numeric:
        return sea_level_rho * np.exp(-y) - get_density(h)

    scaled_height = np.zeros_like(height)
    for ind in range(len(height)):
        _lim = lims if lims is not None else [height[ind] * 0.7, height[ind] * 1.3]
        scaled_height[ind] = sco.bisect(
            func_exact, _lim[0], _lim[1], args=(Yvalues[ind],), **root_find_kwargs
        )
    return scaled_height


def atmosphere_density(
    height: Numeric, atmospheric_scale_height: Numeric, sea_level_rho: Numeric
) -> Numeric:
    """The atmospheric density model used by alpha-beta"""
    return sea_level_rho * np.exp(-height / atmospheric_scale_height)


def Q5(x: NDArray, velocities: NDArray, yvals: NDArray, fill_value: Numeric = 10.0) -> Numeric:
    if len(x.shape) == 1:
        x.shape = (x.size, 1)
    size = x.shape[1]

    res = np.zeros((size,))
    for i in range(len(velocities)):
        vval = velocities[i] / (x[2, ...] * 1000.0)
        r0 = 2 * x[0, ...] * np.exp(-yvals[i]) - (
            (scs.expi(x[1, ...]) - scs.expi(x[1, ...] * vval**2)) * np.exp(-x[1, ...])
        )
        r0[np.isnan(r0)] = fill_value
        res += r0**2
    if res.size == 1:
        return res[0]
    else:
        return res


def solve_alpha_beta_velocity_versionQ5(
    velocities: NDArray_N,
    heights: NDArray_N,
    atmospheric_scale_height: Numeric | None = None,
    start: NDArray | list[float | None] | None = None,
    bounds: Bounds = ((1e-2, 1e6), (5e-1, 1e3), (0, None)),
    minimize_kwargs: Kwargs = {},
) -> NDArray:
    """Solve for alpha and beta using minimization of the least squares of the
    observables input into the preserved analytical relation [^1].

    Adapted from [^2] and [^3].

    [^1]: In preparation.
    [^2]: https://github.com/desertfireballnetwork/alpha_beta_modules
    [^3]: https://doi.org/10.1016/j.asr.2009.03.030
    """
    Yvalues = heights
    if atmospheric_scale_height is not None:
        Yvalues = Yvalues / atmospheric_scale_height

    b0 = 1.0
    a0 = np.exp(Yvalues[-1]) / (2.0 * b0)
    if np.log(a0) > 8: 
        # If initial estimate is that ln(a)>8, its probably not a fireball
        # but a smaller meteoroid and we need to start on the correct side of the
        # a-hump of Q5
        a0 /= 10
    # /1000 is a hack to make velocities small so minimisation doesnt use stupid steps
    # and add a litle bit to velocity as we usually always have some deceleration before observation
    v0 = np.nanmean(velocities[:10]) * 1.01 / 1000
    if start is None:
        start = [a0, b0, v0]
    else:
        beta_start = start[1]
        if beta_start is None:
            beta_start = b0
            start[1] = beta_start
        if start[0] is None:
            start[0] = np.exp(Yvalues[-1]) / (2.0 * beta_start)

        if start[2] is None:
            start[2] = v0
        else:
            start[2] *= 1e-3

    res = sco.minimize(Q5, start, args=(velocities, Yvalues), bounds=bounds, **minimize_kwargs)
    out = res.x
    out[2] *= 1000.0  # fix velocities for return
    return out


def Q4(x: NDArray, vvals: NDArray, yvals: NDArray) -> Numeric:
    res = 0.0
    for i in range(len(vvals)):
        r0 = 2 * x[0, ...] * np.exp(-yvals[i])
        r0 -= (scs.expi(x[1, ...]) - scs.expi(x[1, ...] * vvals[i] ** 2)) * np.exp(-x[1, ...])
        res += pow(r0, 2)
    return res


def solve_alpha_beta_versionQ4(
    velocities: NDArray_N,
    heights: NDArray_N,
    initial_velocity: Numeric | None = None,
    atmospheric_scale_height: Numeric | None = None,
    start: NDArray | list[float | None] | None = None,
    bounds: Bounds = ((1e-3, 1e4), (1.0, 1e3)),
    minimize_kwargs: Kwargs = {},
) -> NDArray:
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
    # TODO: automatic better initial guess for beta
    b0 = 1.0
    a0 = np.exp(Yvalues[-1]) / (2.0 * b0)
    if np.log(a0) > 8: 
        # If initial estimate is that ln(a)>8, its probably not a fireball
        # but a smaller meteoroid and we need to start on the correct side of the
        # a-hump of Q4
        a0 /= 10
    if start is None:
        start = [a0, b0]
    else:
        beta_start = start[1]
        if beta_start is None:
            beta_start = b0
            start[1] = beta_start
        if start[0] is None:
            start[0] = np.exp(Yvalues[-1]) / (2.0 * beta_start)

    res = sco.minimize(Q4, start, args=(Vvalues, Yvalues), bounds=bounds, **minimize_kwargs)
    return res.x


def velocity_smoother_dfn(velocity: NDArray_N, iters: int = 2) -> NDArray_N:
    """Velocity smoother based on consistency of neighboring data points,
    adapted from [^1]. Removes the first and last point `iters` times.

    [^1]: https://github.com/desertfireballnetwork/alpha_beta_modules
    """
    vel_smooth = velocity.copy()
    for ind in range(iters):
        vel_smooth = (vel_smooth[:-2] + vel_smooth[1:-1] + vel_smooth[2:]) / 3.0
    return vel_smooth


def velocity_smoother_mov_avg(velocity: NDArray_N, window: int = 7) -> NDArray_N:
    movavg = np.cumsum(velocity)
    movavg[window:] = movavg[window:] - movavg[:-window]
    return movavg[(window - 1) :] / window
