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
    el = np.radians(radiant_local_elevation) if degrees else radiant_local_elevation
    alpha = 0.5 / (ballistic_coefficient * np.sin(el))

    int_atm_density = sci.cumulative_trapezoid(atm_mass_density, heights)
    v = np.exp(alpha * int_atm_density)

    if initial_velocity is not None:
        v *= initial_velocity
    return v


def _LSQ_fit_func(cB, heights, velocities, weights, atm_dens, el, degrees):
    pred_v = velocity(
        heights,
        atm_dens,
        cB,
        el,
        initial_velocity=velocities[0],
        degrees=degrees,
    )
    vels = velocities[1:] if velocities.size - pred_v.size == 1 else velocities
    w = weights[1:] if weights.size - pred_v.size == 1 else weights
    lsq = np.sum(((pred_v - vels)/w) ** 2)
    return lsq


def fit_velocity(
    heights,
    velocities,
    atm_mass_density,
    radiant_local_elevation,
    weights=None,
    degrees=False,
    minimize_kwargs={},
):
    if weights is None:
        weights = np.ones_like(velocities)
    result = sco.minimize(
        _LSQ_fit_func,
        x0=(1.0,),
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
    return result.x[0]
