"""Model described in: to be published"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.interpolate as sci
import scipy.optimize as sco
import scipy.special as scs
from scipy.integrate import solve_ivp

from .. import physics
from ..atmosphere import Atmosphere
from ..types import (
    NDArray_2xN,
    NDArray_N,
    Options,
    Results,
)
from .model import MeteorModel

logger = logging.getLogger(__name__)


@dataclass
class AlphaBetaOptions(Options):
    minimum_relative_mass: float = 1e-7  # [kg]
    max_step_size: float = 5e-3  # [s]
    first_step: float | None = None  # [s]
    max_time: float = 100.0  # [s]
    method: str = "RK45"
    method_kwargs: dict[str, Any] = field(default_factory=lambda: dict())
    atmosphere: Atmosphere | None = None
    atmosphere_kwargs: dict[str, Any] = field(default_factory=lambda: dict())
    atmospheric_scale_height: float = 7610.0


@dataclass
class AlphaBetaInitialState:
    epoch: np.datetime64 | None
    initial_height: float  # m
    entry_velocity: float  # m/s
    entry_angle: float  # rad
    alpha: float
    beta: float
    shape_change_coefficient: float


@dataclass
class AlphaBetaResults(Results):
    t: NDArray_N
    distance: NDArray_N
    velocity: NDArray_N
    height: NDArray_N
    relative_mass: NDArray_N
    massloss: NDArray_N


def dmdt(
    v: NDArray_N | float,
    a: NDArray_N | float,
    beta: NDArray_N | float,
    mu: NDArray_N | float,
) -> NDArray_N | float:
    return 2 * beta * v * a / (1 - mu) * np.exp(-beta * (1 - v**2) / (1 - mu))


def diff_eq_rhs(
    t: float,
    params: list[NDArray_N | float],
    alpha: float,
    beta: float,
    gamma: float,
    entry_velocity: float,
    atmospheric_scale_height: float,
) -> NDArray_N | NDArray_2xN:
    s, v = params
    delta = scs.expi(beta) - scs.expi(beta * v**2)
    coef = entry_velocity / atmospheric_scale_height
    return np.array(
        [
            v,
            -(v**2) * coef * np.sin(gamma) * np.exp(-beta * v**2) * delta * 0.5,
        ]
    )


class AlphaBeta2026(MeteorModel[AlphaBetaInitialState, AlphaBetaOptions, AlphaBetaResults]):
    def __init__(self, options: AlphaBetaOptions = AlphaBetaOptions()):
        self.options = options
        if options.atmosphere is not None:
            raise NotImplementedError("still have not fixed rescaling")

    def run(self, parameters: AlphaBetaInitialState) -> AlphaBetaResults:
        """Ablation model"""
        t0 = time.perf_counter()
        if parameters.shape_change_coefficient < 0 or parameters.shape_change_coefficient > 2 / 3:
            logger.warning(
                "Only values in [0, 2/3] make physical sense for the `shape_change_coefficient`"
                f", got '{parameters.shape_change_coefficient}'"
            )
        v0_norm = physics.alpha_beta.norm_velocity_estimate(
            parameters.initial_height,
            parameters.alpha,
            parameters.beta,
            atmospheric_scale_height=self.options.atmospheric_scale_height,
        )
        v_final_norm = physics.alpha_beta.norm_velocity_direct(
            self.options.minimum_relative_mass,
            parameters.beta,
            parameters.shape_change_coefficient,
        )
        s0 = 0
        y0 = np.array([s0, v0_norm])

        def _low_velocity(t: float, y: NDArray_N, *args: tuple[Any]) -> float:
            res = y[1] / v_final_norm - 1
            return res

        _low_velocity.terminal = True  # type: ignore
        _low_velocity.direction = -1  # type: ignore

        events = [_low_velocity]

        ivp_result = solve_ivp(
            fun=diff_eq_rhs,
            t_span=(0, self.options.max_time),
            y0=y0,
            args=(
                parameters.alpha,
                parameters.beta,
                parameters.entry_angle,
                parameters.entry_velocity,
                self.options.atmospheric_scale_height,
            ),
            method=self.options.method,
            max_step=self.options.max_step_size,
            first_step=self.options.first_step,
            dense_output=False,
            events=events,
            **self.options.method_kwargs,
        )
        logger.debug(f"{self.__class__} IVP integration complete")

        s_vals = ivp_result.y[0, :] * parameters.entry_velocity
        v_norm = ivp_result.y[1, :]
        diffs = diff_eq_rhs(
            0,
            [ivp_result.y[0, :], v_norm],
            parameters.alpha,
            parameters.beta,
            parameters.entry_angle,
            parameters.entry_velocity,
            self.options.atmospheric_scale_height,
        )

        return AlphaBetaResults(
            runtime=time.perf_counter() - t0,
            t=ivp_result.t,
            distance=s_vals,
            velocity=v_norm * parameters.entry_velocity,
            height=physics.alpha_beta.height_direct(
                v_norm,
                self.options.atmospheric_scale_height,
                parameters.alpha,
                parameters.beta,
            ),
            relative_mass=physics.alpha_beta.norm_mass_direct(
                v_norm,
                parameters.beta,
                parameters.shape_change_coefficient,
            ),
            massloss=dmdt(
                v_norm,
                diffs[1, :],
                parameters.beta,
                parameters.shape_change_coefficient,
            ),
        )


def logposterior_alpha_beta(
    alpha,
    beta,
    heights,
    heights_std,
    velocities,
    velocities_std,
    t_eval,
    initial_height,
    entry_velocity,
    entry_angle,
    shape_change_coefficient,
    model,
):
    p = AlphaBetaInitialState(
        epoch=None,
        alpha=alpha,
        beta=beta,
        initial_height=initial_height,
        entry_velocity=entry_velocity,
        entry_angle=entry_angle,
        shape_change_coefficient=shape_change_coefficient,
    )
    result = model.run(p)
    model_h = sci.interp1d(result.t, result.height, kind="linear", fill_value=np.nan, bounds_error=False)
    model_v = sci.interp1d(result.t, result.velocity, kind="linear", fill_value=np.nan, bounds_error=False)
    resids_sq = np.nansum(((heights - model_h(t_eval)) / heights_std) ** 2)
    resids_sq += np.nansum(((velocities - model_v(t_eval)) / velocities_std) ** 2)

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2)
    # axes[0].plot(t_eval, model_h(t_eval))
    # axes[0].plot(t_eval, heights)
    # axes[1].plot(t_eval, model_v(t_eval))
    # axes[1].plot(t_eval, velocities)
    # fig.suptitle(f"{resids_sq}")
    # plt.show()

    return -0.5 * resids_sq


def solve_alpha_beta_posterior(
    t,
    heights,
    heights_std,
    velocities,
    velocities_std,
    entry_angle,
    shape_change_coefficient,
    atmospheric_scale_height,
    entry_velocity=None,
    options=AlphaBetaOptions(
        max_step_size=1e-2,
    ),
    start=None,
    bounds=((0.01, 1000000.0), (0.0001, 200.0), (0.0, None)),
    minimize_kwargs={},
):
    """Solve for alpha and beta (and optionally velocity) using minimization of the posterior [^1].

    [^1]: Kastinen et al (2026) in preparation
    """
    Yvalues = heights / atmospheric_scale_height

    b0 = 1.0
    a0 = np.exp(Yvalues[-1]) / (2.0 * b0)
    v0 = velocities[0]
    if start is None:
        start = [a0, b0, v0]
    else:
        if start[1] is None:
            start[1] = b0
        if start[0] is None:
            start[0] = np.exp(Yvalues[-1]) / (2.0 * start[1])
        if start[2] is None:
            start[2] = v0

    model = AlphaBeta2026(
        options=options,
    )
    if entry_velocity is not None:
        start = start[:2]
        bounds = (bounds[0], bounds[1])

        def wrap_func(x):
            return -logposterior_alpha_beta(
                x[0],
                x[1],
                heights,
                heights_std,
                velocities,
                velocities_std,
                t - t[0],
                heights[0],
                entry_velocity,
                entry_angle,
                shape_change_coefficient,
                model,
            )
    else:

        def wrap_func(x):
            return -logposterior_alpha_beta(
                x[0],
                x[1],
                heights,
                heights_std,
                velocities,
                velocities_std,
                t - t[0],
                heights[0],
                x[2],
                entry_angle,
                shape_change_coefficient,
                model,
            )

    res = sco.minimize(wrap_func, start, bounds=bounds, **minimize_kwargs)
    out = res.x
    print(res)
    return out
