import numpy as np
import scipy.integrate as sci
import scipy.optimize as sco


def velocity(
    heights,
    atm_mass_density,
    ballistic_coefficient,
    radiant_local_elevation,
    initial_velocity=None,
    degrees=False,
):
    """

    Notes
    -----
    Assumes atmospheric mass densities are sorted according to time
    """
    el = np.radians(radiant_local_elevation) if degrees else radiant_local_elevation
    alpha = 0.5 / (ballistic_coefficient * np.sin(el))

    int_atm_density = sci.cumulative_trapezoid(atm_mass_density, heights)
    v = np.exp(alpha * int_atm_density)
    v = np.insert(v, 0, 1.0)

    if initial_velocity is not None:
        v *= initial_velocity
    return v


def _LSQ_fit_func(cB, heights, velocities, initial_velocity, weights, atm_dens, el, degrees):
    pred_v = velocity(
        heights,
        atm_dens,
        cB,
        el,
        initial_velocity=initial_velocity,
        degrees=degrees,
    )
    w = weights[1:] if weights.size - pred_v.size == 1 else weights
    lsq = np.sum(((pred_v - velocities)/w) ** 2)
    return lsq


def _LSQ_fit_func_two_vars(x, heights, velocities, weights, atm_dens, el, degrees):
    cB, initial_velocity = x
    pred_v = velocity(
        heights,
        atm_dens,
        cB,
        el,
        initial_velocity=initial_velocity,
        degrees=degrees,
    )
    w = weights[1:] if weights.size - pred_v.size == 1 else weights
    lsq = np.sum(((pred_v - velocities)/w) ** 2)
    return lsq


def fit_velocity(
    heights,
    velocities,
    atm_mass_density,
    radiant_local_elevation,
    initial_velocity=None,
    start=None,
    weights=None,
    degrees=False,
    minimize_kwargs={},
):
    if weights is None:
        weights = np.ones_like(velocities)
    if initial_velocity is None:
        x0 = (1.0, velocities[0])
        _obj_func = _LSQ_fit_func_two_vars
    else:
        x0 = (1.0, )
        _obj_func = _LSQ_fit_func

    if start is None:
        start = x0

    result = sco.minimize(
        _obj_func,
        x0=start,
        args=(
            heights,
            velocities,
            weights,
            atm_mass_density,
            radiant_local_elevation,
            degrees,
        ),
        **minimize_kwargs
    )
    return result.x
