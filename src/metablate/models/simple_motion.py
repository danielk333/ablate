from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from spacecoords import frames
from spacecoords.constants import WGS84

from ..atmosphere import Atmosphere, AtmPymsis
from ..types import NDArray_3, NDArray_3xN, NDArray_M, NDArray_N, Options, Results
from .model import MeteorModel


@dataclass
class LinearConstantOptions(Options):
    evaluation_times: NDArray_N

LCV_InitState = NDArray_M
"""(m,) shaped ndarray with index:
    0-2: initial ECEF position
    3-5: initial ECEF velocity vector
"""


@dataclass
class PosVelResults(Results):
    position: NDArray_3xN
    velocity: NDArray_3xN


@dataclass
class BallisticOptions(Options):
    evaluation_times: NDArray_N
    atmosphere: Atmosphere = field(default_factory=AtmPymsis)
    atmosphere_kwargs: dict[str, Any] = field(default_factory=dict)
    minimum_altitude: float = 0.0  # m above the WGS84 ellipsoid
    gravitational_parameter: float = WGS84.GM  # m^3 / s^2
    max_step_size: float = np.inf  # s
    first_step: float | None = None  # s
    method: str = "RK45"
    method_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class BallisticInitialState:
    """Initial ECEF state and ballistic coefficient."""

    epoch: np.datetime64
    position_ecef: NDArray_3  # m
    velocity_ecef: NDArray_3  # m/s
    ballistic_coefficient: float  # kg/m^2


@dataclass
class BallisticResults(Results):
    t: NDArray_N
    position_ecef: NDArray_3xN
    velocity_ecef: NDArray_3xN


LCA_InitState = NDArray_M
"""(m,) shaped ndarray with index:
    0-2: initial ECEF position
    3-5: initial ECEF velocity vector
    6: initial acceleration
"""


class LinearConstantVelocity(MeteorModel[LCV_InitState, LinearConstantOptions, PosVelResults]):
    def run(self, parameters: LCV_InitState) -> PosVelResults:
        t0 = time.perf_counter()
        times = self.options.evaluation_times
        position = parameters[0:3, None] + parameters[3:6, None] * times[None, :]
        velocity = np.repeat(parameters[3:6], len(times)).reshape(3, len(times))
        return PosVelResults(
            runtime=time.perf_counter() - t0,
            position=position,
            velocity=velocity,
        )


class LinearConstantAcceleration(MeteorModel[LCA_InitState, LinearConstantOptions, PosVelResults]):
    def run(self, parameters: LCA_InitState) -> PosVelResults:
        t0 = time.perf_counter()
        times = self.options.evaluation_times
        v0 = np.linalg.norm(parameters[3:6])
        v_hat = parameters[3:6] / v0
        disp = v0 * times + 0.5 * parameters[6] * times**2
        position = parameters[0:3, None] + v_hat[:, None] * disp[None, :]
        velocity = parameters[3:6, None] + v_hat[:, None] * parameters[6] * times[None, :]
        return PosVelResults(
            runtime=time.perf_counter() - t0,
            position=position,
            velocity=velocity,
        )


class Ballistic(MeteorModel[BallisticInitialState, BallisticOptions, BallisticResults]):
    """Integrate a ballistic ECEF trajectory with atmospheric drag and gravity.

    The drag uses the same ballistic-coefficient convention as
    `metablate.physics.ballistic.velocity`, i.e.
    `c_B = mass / (drag_coefficient * area)`.

    Earth rotation and lift are not included.
    """

    def run(self, parameters: BallisticInitialState) -> BallisticResults:
        t0 = time.perf_counter()
        times = np.asarray(self.options.evaluation_times, dtype=float)
        position0 = parameters.position_ecef
        velocity0 = parameters.velocity_ecef
        self._validate_inputs(times, position0, velocity0, parameters.ballistic_coefficient)

        initial_altitude = self._geodetic(position0)[2]
        if initial_altitude <= self.options.minimum_altitude:
            raise ValueError("initial position must be above minimum_altitude")

        state0 = np.concatenate((position0, velocity0))

        def rhs(t: float, state: NDArray_N) -> NDArray_N:
            position = state[:3]
            velocity = state[3:]
            radius = np.linalg.norm(position)
            speed = np.linalg.norm(velocity)
            lat, lon, altitude = self._geodetic(position)
            atmosphere = self.options.atmosphere.density(
                time=parameters.epoch + np.timedelta64(int(round(t * 1e6)), "us"),
                lat=lat,
                lon=lon,
                alt=altitude,
                mass_densities=False,
                **self.options.atmosphere_kwargs,
            )
            density = float(np.asarray(atmosphere["Total"].values).squeeze())
            if not np.isfinite(density) or density < 0.0:
                raise ValueError("atmospheric mass density must be finite and nonnegative")

            gravity = -self.options.gravitational_parameter * position / radius**3
            drag = -0.5 * density * speed * velocity / parameters.ballistic_coefficient
            return np.concatenate((velocity, gravity + drag))

        def minimum_altitude_event(t: float, state: NDArray_N) -> float:
            return self._geodetic(state[:3])[2] - self.options.minimum_altitude

        minimum_altitude_event.terminal = True  # type: ignore[attr-defined]
        minimum_altitude_event.direction = -1  # type: ignore[attr-defined]

        ivp_result = solve_ivp(
            rhs,
            (0.0, times[-1]),
            state0,
            t_eval=times,
            events=minimum_altitude_event,
            method=self.options.method,
            max_step=self.options.max_step_size,
            first_step=self.options.first_step,
            **self.options.method_kwargs,
        )
        if not ivp_result.success:
            raise RuntimeError(f"ballistic integration failed: {ivp_result.message}")

        result_times = ivp_result.t
        states = ivp_result.y
        if ivp_result.t_events[0].size:
            event_time = ivp_result.t_events[0][0]
            event_state = ivp_result.y_events[0][0]
            if result_times.size == 0 or not np.isclose(result_times[-1], event_time):
                result_times = np.append(result_times, event_time)
                states = np.column_stack((states, event_state))

        return BallisticResults(
            runtime=time.perf_counter() - t0,
            t=result_times,
            position_ecef=states[:3, :],
            velocity_ecef=states[3:, :],
        )

    @staticmethod
    def _geodetic(position_ecef: NDArray_3) -> NDArray_3:
        return np.asarray(
            frames.ecef_to_geodetic_wgs84(*position_ecef, degrees=True),
            dtype=float,
        )

    @staticmethod
    def _validate_inputs(
        times: NDArray_N,
        position: NDArray_3,
        velocity: NDArray_3,
        ballistic_coefficient: float,
    ) -> None:
        if times.ndim != 1 or times.size < 2:
            raise ValueError(
                "evaluation_times must be a one-dimensional array with at least two entries"
            )
        if not np.all(np.isfinite(times)) or times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError(
                "evaluation_times must be finite, strictly increasing, and start at zero"
            )
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_ecef must contain three finite values")
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            raise ValueError("velocity_ecef must contain three finite values")
        if not np.isfinite(ballistic_coefficient) or ballistic_coefficient <= 0.0:
            raise ValueError("ballistic_coefficient must be finite and positive")
