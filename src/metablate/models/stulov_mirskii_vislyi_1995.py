#!/usr/bin/env python

"""Model described in:

Stulov, V. P., V. N. Mirskii, and A. I. Vislyi.
"Aerodynamics of bolides."
Moscow: Science. Fizmatlit (1995).

and later in english in:

Gritsevich, M. I.
"Validity of the Photometric Formula for Estimating the Mass of a Fireball Projectile."
Doklady Physics 53, no. 2 (2008): 97–102. https://doi.org/10.1134/S1028335808020110.
"""

import time
import logging
from typing import Any
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import numpy as np

from spacecoords import frames, spherical
from spacecoords.constants import WGS84, R_earth
from .model import MeteorModel
from ..atmosphere import Atmosphere, AtmPymsis
from ..types import (
    Options,
    NDArray_3,
    Results,
    NDArray_3xN,
    NDArray_N,
    Material,
)
from .. import material

logger = logging.getLogger(__name__)


@dataclass
class SMV_Options(Options):
    minimum_mass: float = 1e-9  # [kg]
    max_step_size: float = 1e-1  # [s]
    first_step: float | None = None  # [s]
    max_time: float = 100.0  # [s]
    method: str = "RK45"
    method_kwargs: dict[str, Any] = field(default_factory=lambda: dict())
    atmosphere: Atmosphere = AtmPymsis()
    atmosphere_kwargs: dict[str, Any] = field(default_factory=lambda: dict())
    start_altitude: float | None = 150e3  # [m]
    material: Material = field(default_factory=lambda: material.get("cometary"))


@dataclass
class SMV_InitialState:
    epoch: np.datetime64
    position_ecef: NDArray_3  # m
    velocity_ecef: NDArray_3  # m/s
    mass: float  # kg
    initial_cross_sectional_area: float
    aerodynamic_cd: float = 0.47
    aerodynamic_cl: float = 0
    heat_exchange_coefficient: float = 1
    shape_change_coefficient: float = 2 / 3


@dataclass
class SMV_Results(Results):
    t: NDArray_N
    distance: NDArray_N
    velocity: NDArray_N
    mass: NDArray_N
    trajectory_elevation: NDArray_N
    position_ecef: NDArray_3xN
    velocity_ecef: NDArray_3xN


class StulovMirskiiVislyi1995(MeteorModel[SMV_InitialState, SMV_Options, SMV_Results]):
    def __init__(self, options: SMV_Options = SMV_Options()):
        self.options = options

    def run(self, parameters: SMV_InitialState) -> SMV_Results:
        """Runs the ablation model."""
        t0 = time.perf_counter()

        velocity0 = np.linalg.norm(parameters.velocity_ecef)
        v_dir = parameters.velocity_ecef / velocity0

        def s_to_geo(s: float) -> NDArray_N:
            point_ecef = parameters.position_ecef + v_dir * s
            geo = frames.ecef_to_geodetic_wgs84(point_ecef[0], point_ecef[1], point_ecef[2], degrees=False)
            return geo

        init_displacement = minimize_scalar(lambda s: np.abs(s_to_geo(s)[2] - self.options.start_altitude)).x
        init_geo = s_to_geo(init_displacement)
        local_v_dir = frames.ecef_to_enu(init_geo[0], init_geo[1], v_dir, degrees=False)
        local_radiant = spherical.cart_to_sph(-local_v_dir, degrees=False)

        y0 = np.array(
            [
                velocity0,
                parameters.mass,
                init_displacement,
                local_radiant[1],
            ],
            dtype=np.float64,
        )

        def _low_mass(t: float, y: NDArray_N, *args: tuple[Any]) -> float:
            res = y[1] / self.options.minimum_mass - 1
            return res

        _low_mass.terminal = True  # type: ignore
        _low_mass.direction = -1  # type: ignore

        events = [_low_mass]

        ivp_result = solve_ivp(
            fun=diff_eq_rhs,
            t_span=(0, self.options.max_time),
            y0=y0,
            args=(
                parameters.epoch,
                parameters.position_ecef,
                v_dir,
                parameters.mass,
                parameters.initial_cross_sectional_area,
                parameters.aerodynamic_cd,
                parameters.aerodynamic_cl,
                parameters.heat_exchange_coefficient,
                parameters.shape_change_coefficient,
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

        position_ecef = parameters.position_ecef[:, None] + v_dir[:, None] * ivp_result.y[2, :][None, :]
        velocity_ecef = v_dir[:, None] * ivp_result.y[0, :][None, :]

        return SMV_Results(
            runtime=time.perf_counter() - t0,
            t=ivp_result.t,
            distance=ivp_result.y[2, :],
            velocity=ivp_result.y[0, :],
            mass=ivp_result.y[1, :],
            trajectory_elevation=ivp_result.y[3, :],
            position_ecef=position_ecef,
            velocity_ecef=velocity_ecef,
        )


def diff_eq_rhs(
    t,
    y,
    epoch: np.datetime64,
    reference_ecef: NDArray_3,
    v_dir_ecef: NDArray_3,
    initial_mass: float,
    initial_cross_sectional_area: float,
    aerodynamic_cd: float,
    aerodynamic_cl: float,
    heat_exchange_coefficient: float,
    shape_change_coefficient: float,
    options: SMV_Options,
) -> NDArray_N:
    """The right hand side of the differential equation to be integrated, i.e:

    The numpy vector is structured as follows (numbers indicating index):

        0. dmdt
        1. dvdt
        2. dsdt
        3. dhdt
        4. dgammadt

    """
    vel, mass, s, gamma = y
    mass_norm = mass / initial_mass
    enthalpy_of_massloss = options.material.latent_heat_of_fusion_vapourization

    # meteoroid height above earth surface
    point_ecef = reference_ecef + v_dir_ecef * s
    lat, lon, alt = frames.ecef_to_geodetic_wgs84(point_ecef[0], point_ecef[1], point_ecef[2], degrees=True)
    r = np.linalg.norm(point_ecef)

    atm = options.atmosphere.density(
        time=epoch + np.timedelta64(int(t * 1e6), "us"),
        lat=lat,
        lon=lon,
        alt=alt,
        mass_densities=False,
        **options.atmosphere_kwargs,
    )

    rho_atm = atm["Total"].values.squeeze()
    
    if mass_norm > 0:
        area = initial_cross_sectional_area * (mass_norm**shape_change_coefficient)
    else:
        area = 0
    drag_f = 0.5 * aerodynamic_cd * rho_atm * vel**2 * area
    lift_f = 0.5 * aerodynamic_cl * rho_atm * vel**2 * area
    mass_loss_r = 0.5 * heat_exchange_coefficient * rho_atm * vel**3 * area
    grav_a = WGS84.GM / r**2

    #  dMdt
    #  dVdt
    #  dsdt
    #  dgammadt
    # -- Differential equations to solve
    dmdt = -mass_loss_r / enthalpy_of_massloss
    dvdt = -drag_f / mass + grav_a * np.sin(gamma)
    dsdt = vel
    dgammadt = (grav_a / vel - vel / R_earth) * np.cos(gamma) - lift_f / (mass * vel)

    # logger.debug(f"{(dvdt, dmdt, dsdt, dgammadt)}")
    # logger.debug(f"{y=}")
    ret = np.array([dvdt, dmdt, dsdt, dgammadt], dtype=np.float64)

    return ret
