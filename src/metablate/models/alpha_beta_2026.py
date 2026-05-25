#!/usr/bin/env python

"""Model described in: to be published"""

import time
import logging
from typing import Any
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp
import scipy.special as scs

from .. import physics
from .model import MeteorModel
from ..atmosphere import Atmosphere
from ..types import (
    Options,
    Results,
    NDArray_2xN,
    NDArray_N,
)

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
) -> NDArray_N | NDArray_2xN:
    s, v = params
    delta = scs.expi(beta) - scs.expi(beta * v**2)
    return np.array(
        [
            v,
            -(v**2) * np.sin(gamma) * np.exp(-beta * v**2) * delta * 0.5,
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
        v0 = physics.alpha_beta.velocity_estimate(
            parameters.initial_height,
            parameters.entry_velocity,
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
        ds0 = v0 / parameters.entry_velocity
        y0 = np.array([s0, ds0])

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
            ),
            method=self.options.method,
            max_step=self.options.max_step_size,
            first_step=self.options.first_step,
            dense_output=False,
            events=events,
            **self.options.method_kwargs,
        )
        logger.debug(f"{self.__class__} IVP integration complete")

        v_norm = ivp_result.y[1, :]
        diffs = diff_eq_rhs(
            0,
            [ivp_result.y[0, :], v_norm],
            parameters.alpha,
            parameters.beta,
            parameters.entry_angle,
        )

        return AlphaBetaResults(
            runtime=time.perf_counter() - t0,
            t=ivp_result.t,
            distance=ivp_result.y[0, :] * self.options.atmospheric_scale_height,
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
