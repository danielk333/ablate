from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.integrate as sci
import scipy.optimize as sco

from ..types import Kwargs, NDArray, NDArray_N, Numeric


def velocity(
    heights: NDArray_N,
    atm_mass_density: NDArray_N,
    ballistic_coefficient: Numeric,
    radiant_local_elevation: float,
    initial_velocity: float | None = None,
    degrees: bool = False,
) -> NDArray_N:
    """Velocity function for ballistic coefficient solution of trajectory

    TODO: make version with free molecular flow

    $$
        c_b = \\frac{M_e}{c_d S_e}
    $$

    where $M_e$ is entry mass, $S_e$ is cross-sectional area and $c_d$ is
    aerodynamic drag coefficient.

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


def _LSQ_fit_func(
    cB: Numeric,
    heights: NDArray_N,
    velocities: NDArray_N,
    initial_velocity: float,
    weights: NDArray_N,
    atm_dens: NDArray_N,
    el: float,
    degrees: bool,
) -> float:
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


def _LSQ_fit_func_two_vars(
    x: NDArray_N,
    heights: NDArray_N,
    velocities: NDArray_N,
    weights: NDArray_N,
    atm_dens: NDArray_N,
    el: float,
    degrees: bool,
) -> float:
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
    heights: NDArray_N,
    velocities: NDArray_N,
    atm_mass_density: NDArray_N,
    radiant_local_elevation: float,
    initial_velocity: float | None = None,
    start: NDArray | tuple[float, ...] | None = None,
    weights: NDArray_N | None = None,
    degrees: bool = False,
    minimize_kwargs: Kwargs = {},
) -> NDArray_N:
    if weights is None:
        weights = np.ones_like(velocities)
    x0: tuple[float, ...]
    _obj_func: Callable[..., float]
    fit_args: tuple[object, ...] = (
        heights,
        velocities,
        weights,
        atm_mass_density,
        radiant_local_elevation,
        degrees,
    )
    if initial_velocity is None:
        x0 = (1.0, velocities[0])
        _obj_func = _LSQ_fit_func_two_vars
    else:
        x0 = (1.0, )
        _obj_func = _LSQ_fit_func
        fit_args = (
            heights,
            velocities,
            initial_velocity,
            weights,
            atm_mass_density,
            radiant_local_elevation,
            degrees,
        )

    if start is None:
        start = x0

    result = sco.minimize(_obj_func, x0=start, args=fit_args, **minimize_kwargs)
    return result.x
