"""Model described in: to be published"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.optimize as sco
from scipy import constants
from scipy.integrate import solve_ivp

from ..atmosphere import Atmosphere, AtmPymsis
from ..types import (
    Bounds,
    Kwargs,
    NDArray,
    NDArray_2xN,
    NDArray_N,
    Numeric,
    Options,
    Results,
    StateVector,
)
from .model import MeteorModel

logger = logging.getLogger(__name__)

WGS84_GM = 3986004.418e8
WGS84_M_EARTH = WGS84_GM / constants.G
EARTH_RADIUS = 6378136.3  # TODO: reference


def ballistic_parameter_direct(
    aerodynamic_cd: float,
    initial_cross_section: float,
    initial_mass: float,
) -> float:
    return 0.5 * aerodynamic_cd * initial_cross_section / initial_mass


def massloss_parameter_direct(
    heat_exchange_coefficient: float,
    enthalpy_of_massloss: float,
    initial_cross_section: float,
    initial_mass: float,
) -> float:
    return 0.5 * heat_exchange_coefficient * initial_cross_section / (enthalpy_of_massloss * initial_mass)


def ballistic_parameter_to_area_to_mass(
    ballistic_parameter: float,
    aerodynamic_cd: float,
) -> float:
    return 2 * ballistic_parameter / aerodynamic_cd


@dataclass
class SimpleAblationOptions(Options):
    minimum_relative_mass: float = 1e-4  # [1]
    minimum_altitude: float = 0  # [m]
    max_step_size: float = 5e-3  # [s]
    first_step: float | None = None  # [s]
    max_time: float = 100.0  # [s]
    linear_trajectory: bool = False
    method: str = "RK45"
    method_kwargs: Kwargs = field(default_factory=dict)
    atmosphere: Atmosphere = field(default_factory=AtmPymsis)
    atmosphere_kwargs: Kwargs = field(default_factory=dict)


@dataclass
class SimpleAblationInitialState:
    epoch: np.datetime64
    initial_height: float  # m
    initial_velocity: float  # m/s
    entry_angle: float  # rad
    ballistic_parameter: float
    massloss_parameter: float
    shape_change_coefficient: float
    latitude: float
    longitude: float


@dataclass
class SimpleAblationResults(Results):
    t: NDArray_N
    distance: NDArray_N
    velocity: NDArray_N
    height: NDArray_N
    trajectory_elevation: NDArray_N
    relative_mass: NDArray_N
    relative_massloss: NDArray_N


def diff_eq_rhs(
    t: float,
    params: StateVector,
    ballistic_parameter: float,
    massloss_parameter: float,
    shape_change_coefficient: float,
    lat: float,
    lon: float,
    epoch: np.datetime64,
    initial_gamma: float,
    options: SimpleAblationOptions,
) -> NDArray_N | NDArray_2xN:
    gamma: Numeric
    if options.linear_trajectory:
        _, h, v, m = params
        gamma = initial_gamma
    else:
        _, h, v, m, gamma = params
    grav = constants.G * WGS84_M_EARTH / (EARTH_RADIUS + h) ** 2

    # TODO: should try to do this in innertial frame some time, to include velocity of Atmosphere
    # relative to gravitational trajecotry - needed for space debris for example
    # figure out how to do it somehow?

    # TODO: hack for now
    atm = options.atmosphere.density(
        time=epoch + np.timedelta64(int(t * 1e6), "us"),
        lat=lat,
        lon=lon,
        alt=h,
        mass_densities=False,
        **options.atmosphere_kwargs,
    )
    m = max(m, options.minimum_relative_mass)
    rho_tot = atm["Total"].values.squeeze()
    if options.linear_trajectory:
        ret = np.array(
            [
                v,
                -v * np.sin(gamma),
                -ballistic_parameter * m ** (shape_change_coefficient - 1) * rho_tot * v**2
                + grav * np.sin(gamma),
                -massloss_parameter * m**shape_change_coefficient * rho_tot * v**3,
            ]
        )
    else:
        ret = np.array(
            [
                v,
                -v * np.sin(gamma),
                -ballistic_parameter * m ** (shape_change_coefficient - 1) * rho_tot * v**2
                + grav * np.sin(gamma),
                -massloss_parameter * m**shape_change_coefficient * rho_tot * v**3,
                (grav / v - v / (EARTH_RADIUS + h)) * np.cos(gamma),
            ]
        )
    return ret


class SimpleAblation2026(
    MeteorModel[SimpleAblationInitialState, SimpleAblationOptions, SimpleAblationResults]
):

    def __init__(self, options: SimpleAblationOptions = SimpleAblationOptions()) -> None:
        self.options = options

    def run(self, parameters: SimpleAblationInitialState) -> SimpleAblationResults:
        """Ablation model"""
        t0 = time.perf_counter()
        if parameters.shape_change_coefficient < 0 or parameters.shape_change_coefficient > 2 / 3:
            logger.warning(
                "Only values in [0, 2/3] make physical sense for the `shape_change_coefficient`"
                f", got '{parameters.shape_change_coefficient}'"
            )
        if self.options.linear_trajectory:
            y0 = np.array(
                [
                    0,
                    parameters.initial_height,
                    parameters.initial_velocity,
                    1.0,
                ],
                dtype=np.float64,
            )
        else:
            y0 = np.array(
                [
                    0,
                    parameters.initial_height,
                    parameters.initial_velocity,
                    1.0,
                    parameters.entry_angle,
                ],
                dtype=np.float64,
            )

        def _low_mass(t: float, y: NDArray_N, *args: object) -> float:
            res = y[3] / self.options.minimum_relative_mass - 1
            return res

        def _low_alt(t: float, y: NDArray_N, *args: object) -> float:
            res = y[1] / self.options.minimum_altitude - 1
            return res

        _low_mass.terminal = True  # type: ignore[attr-defined]
        _low_mass.direction = -1  # type: ignore[attr-defined]
        _low_alt.terminal = True  # type: ignore[attr-defined]
        _low_alt.direction = -1  # type: ignore[attr-defined]

        events = [_low_mass, _low_alt]

        ivp_result = solve_ivp(
            fun=diff_eq_rhs,
            t_span=(0, self.options.max_time),
            y0=y0,
            args=(
                parameters.ballistic_parameter,
                parameters.massloss_parameter,
                parameters.shape_change_coefficient,
                parameters.latitude,
                parameters.longitude,
                parameters.epoch,
                parameters.entry_angle,
                self.options,
            ),
            method=self.options.method,
            max_step=self.options.max_step_size,
            first_step=self.options.first_step,
            dense_output=False,
            events=events,
            **self.options.method_kwargs,
        )
        logger.debug(f"{self.__class__} IVP integration complete")

        if self.options.linear_trajectory:
            gamma = np.full_like(ivp_result.t, parameters.entry_angle)
        else:
            gamma = ivp_result.y[4, :]

        # TODO: this is wasting a lot of computations, we could just vectorize this and run it for
        # the massloss but im lazy for now
        relative_massloss = np.asarray(
            [
                diff_eq_rhs(
                    t,
                    state,
                    parameters.ballistic_parameter,
                    parameters.massloss_parameter,
                    parameters.shape_change_coefficient,
                    parameters.latitude,
                    parameters.longitude,
                    parameters.epoch,
                    parameters.entry_angle,
                    self.options,
                )[3]
                for t, state in zip(ivp_result.t, ivp_result.y.T)
            ]
        )

        return SimpleAblationResults(
            runtime=time.perf_counter() - t0,
            t=ivp_result.t,
            distance=ivp_result.y[0, :],
            height=ivp_result.y[1, :],
            velocity=ivp_result.y[2, :],
            trajectory_elevation=gamma,
            relative_mass=ivp_result.y[3, :],
            relative_massloss=relative_massloss,
        )


def logposterior_alpha_beta(
    alpha: float,
    beta: float,
    heights: NDArray_N,
    heights_std: Numeric,
    velocities: NDArray_N,
    velocities_std: Numeric,
    t_eval: NDArray_N,
    initial_height: float,
    entry_velocity: float,
    entry_angle: float,
    shape_change_coefficient: float,
    model: SimpleAblation2026,
) -> float:
    """Placeholder: alpha-beta parameters are not simple-ablation state fields."""
    raise NotImplementedError("Simple-ablation posterior fitting is not implemented.")


def solve_simple_posterior(
    t: NDArray_N,
    heights: NDArray_N,
    heights_std: Numeric,
    velocities: NDArray_N,
    velocities_std: Numeric,
    entry_angle: float,
    shape_change_coefficient: float,
    entry_velocity: float | None = None,
    options: SimpleAblationOptions | None = None,
    start: NDArray | list[float | None] | None = None,
    bounds: Bounds = ((0.01, 1e8), (0.5e-1, 200.0), (0.0, None)),
    minimize_kwargs: Kwargs = {},
) -> sco.OptimizeResult:
    """Placeholder for posterior fitting with the simple-ablation model."""
    raise NotImplementedError("Simple-ablation posterior fitting is not implemented.")
