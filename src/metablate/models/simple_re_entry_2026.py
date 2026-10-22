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


@dataclass
class SimpleReEntryOptions(Options):
    minimum_relative_mass: float = 1e-4  # [1]
    minimum_altitude: float = 0  # [m]
    max_step_size: float = 5e-3  # [s]
    first_step: float | None = None  # [s]
    max_time: float = 100.0  # [s]
    method: str = "RK45"
    method_kwargs: Kwargs = field(default_factory=dict)
    atmosphere: Atmosphere = field(default_factory=AtmPymsis)
    atmosphere_kwargs: Kwargs = field(default_factory=dict)


@dataclass
class SimpleReEntryInitialState:
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
class SimpleReEntryResults(Results):
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
    options: SimpleReEntryOptions,
) -> NDArray_N | NDArray_2xN:
    return None

# TODO: should try to do this in innertial frame some time, to include velocity of Atmosphere
# relative to gravitational trajecotry - needed for space debris for example
# figure out how to do it somehow?

class SimpleReEntry2026(MeteorModel[SimpleReEntryInitialState, SimpleReEntryOptions, SimpleReEntryResults]):
    def __init__(self, options: SimpleReEntryOptions | None = None) -> None:
        if options is None:
            options = SimpleReEntryOptions()
        self.options = options

    def run(self, parameters: SimpleReEntryInitialState) -> SimpleReEntryResults:
        """Ablation model"""
        raise NotImplementedError()
