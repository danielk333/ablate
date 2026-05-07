#!/usr/bin/env python

"""Model implemented in:

Kero, J., Szasz, C., Pellinen-Wannberg, A., Wannberg, G., Westman, A., and Meisel, D. D. (2008).
Determination of meteoroid physical properties from tristatic radar observations.
Annales Geophysicae

# TODO FIX THIS TEXT
Parameters:

- temperature0 = 290 [K]: Meteoroid temperature at starting height
- shape_factor = 1.21 [1]: Shape is assumed to be a sphere.
- emissivity = 0.9 [1]: Hill et al.; Love & Brownlee: 1; (metal oxides)
- sputtering = True [bool]: If sputtering is used in mass loss calculation.
- Gamma = None [1]: Drag coefficient, if :code:`None` it is dynamically
    calculated assuming a transition from (and including) free molecular
    flow to a (and not including) shock regime. Otherwise assumed a
    constant with the given value.
- Lambda = None [1]: Heat transfer coefficient, if :code:`None` it is
    dynamically calculated assuming a transition from (and including)
    free molecular flow to a (and not including) shock regime.
    Otherwise assumed a constant with the given value.

"""

import time
import logging
from typing import Any
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from spacecoords import frames
from spacecoords.constants import WGS84
from .. import physics
from .model import MeteorModel
from ..atmosphere import Atmosphere, AtmPymsis
from ..types import (
    Options,
    NDArray_3,
    Results,
    NDArray_3xN,
    NDArray_N,
    NDArray_M,
    Material,
)
from .. import material

logger = logging.getLogger(__name__)


@dataclass
class KeroSzaszOptions(Options):
    integral_resolution: int = 100
    minimum_mass: float = 1e-11  # [kg]
    max_step_size: float = 5e-2  # [s]
    max_time: float = 100.0  # [s]
    method: str = "RK45"
    method_kwargs: dict[str, Any] = field(default_factory=lambda: dict())
    sputtering: bool = False
    atmosphere: Atmosphere = AtmPymsis()
    atmosphere_kwargs: dict[str, Any] = field(default_factory=lambda: dict())
    effective_atmospheric_temperature: float = 280  # [K]
    start_altitude: float | None = 150e3  # [m]
    material: Material = field(default_factory=lambda: material.get("cometary"))
    shape_factor: float = 1.21  # [1]
    emissivity: float = 0.9  # [1]


@dataclass
class KeroSzaszInitialState:
    epoch: np.datetime64
    position_ecef: NDArray_3  # m
    velocity_ecef: NDArray_3  # m/s
    mass: float  # kg
    # drag_coefficient (Gamma) = half the standard aerodynamic drag coefficient (0.5 * C_d)
    drag_coefficient: float | None = None  # [1]
    heat_transfer_coefficient: float | None = None  # [1]


@dataclass
class KeroSzaszResults(Results):
    t: NDArray_N
    distance: NDArray_N
    velocity: NDArray_N
    mass: NDArray_N
    temperature: NDArray_N
    position_ecef: NDArray_3xN
    velocity_ecef: NDArray_3xN


class KeroSzasz2008(
    MeteorModel[KeroSzaszInitialState, KeroSzaszOptions, KeroSzaszResults]
):
    def __init__(self, options: KeroSzaszOptions = KeroSzaszOptions()):
        self.options = options

    def run(
        self, times: NDArray_N, parameters: KeroSzaszInitialState
    ) -> KeroSzaszResults:
        """Runs the ablation model.

        TODO: update this docstring

        Parameters
        ----------
        velocity0 : float or numpy.ndarray
            Meteoroid initial velocity [m/s]
        mass0 : float or numpy.ndarray
            Meteoroid initial mass [kg]
        altitude0 : float or numpy.ndarray
            Meteoroid initial altitude [m]
        zenith_ang : float or numpy.ndarray
            Zenith angle of the trajectory (-velocity vector) w.r.t reference point [deg]
        azimuth_ang : float or numpy.ndarray
            Azimuthal angle east of north of the trajectory (-velocity vector) w.r.t reference point [deg]
        material_data : dict
            Meteoroid material data, see [`material_data`][ablate.functions.material.material_data].
        lat : float or numpy.ndarray
            Geographic latitude in degrees of reference point on the meteoroid trajectory
        lon : float or numpy.ndarray
            Geographic longitude in degrees of reference point on the meteoroid trajectory
        alt : float or numpy.ndarray
            Altitude above geoid in meters of reference point on the meteoroid trajectory

        Returns
        -------
        xarray.Dataset
            mass, velocity, position, temperature, ecef_x, ecef_y, ecef_z, altitude as variables.

        Notes
        -----
        This function is based on `calc_sput.m` which was used to verify the
        sputtering described in Rogers et al.: Mass loss due to sputtering and
        thermal processes in meteoroid ablation,
        Planetary and Space Science 53 p. 1341-1354 (2005).

        TODO: Add additional dynamical parameters to data structure e.g.
        lambda and gamma if they are not constant

        """
        t0 = time.perf_counter()
        logger.debug(f"Running {self.__class__} model")

        velocity0 = np.linalg.norm(parameters.velocity_ecef)
        v_dir = parameters.velocity_ecef / velocity0

        def s_to_geo(s):
            point_ecef = parameters.position_ecef + v_dir * s
            geo = frames.ecef_to_geodetic_wgs84(
                point_ecef[0], point_ecef[1], point_ecef[2], degrees=False
            )
            return geo

        init_displacement = minimize_scalar(
            lambda s: np.abs(s_to_geo(s)[2] - self.options.start_altitude)
        ).x
        y0 = np.array(
            [
                velocity0,
                parameters.mass,
                init_displacement,
                self.options.effective_atmospheric_temperature,
            ],
            dtype=np.float64,
        )

        def _low_mass(t, y, *args):
            res = y[0] / self.options.minimum_mass - 1
            # logger.debug(
            #     f"Stopping @ {t:<1.4e} s = {res}: {np.log10(y[0]):1.4e} log10(kg) | {y[0]:1.4e} kg"
            # )
            return res

        _low_mass.terminal = True  # type: ignore
        _low_mass.direction = -1  # type: ignore

        events = [_low_mass]

        ivp_result = solve_ivp(
            fun=diff_eq_rhs,
            t_span=(0, self.options.max_time),
            y0=y0,
            args=(
                parameters.drag_coefficient,
                parameters.heat_transfer_coefficient,
                parameters.epoch,
                parameters.position_ecef,
                v_dir,
                self.options,
            ),
            method=self.options.method,
            max_step=self.options.max_step_size,
            first_step=self.options.max_step_size,
            dense_output=False,
            events=events,
            **self.options.method_kwargs,
        )
        logger.debug(f"{self.__class__} IVP integration complete")

        position_ecef = (
            parameters.position_ecef[:, None]
            + v_dir[:, None] * ivp_result.y[2, :][None, :]
        )
        velocity_ecef = v_dir[:, None] * ivp_result.y[0, :][None, :]

        return KeroSzaszResults(
            runtime=time.perf_counter() - t0,
            t=ivp_result.t,
            distance=ivp_result.y[2, :],
            velocity=ivp_result.y[0, :],
            mass=ivp_result.y[1, :],
            temperature=ivp_result.y[3, :],
            position_ecef=position_ecef,
            velocity_ecef=velocity_ecef,
        )


def diff_eq_rhs(
    t: float,
    y: NDArray_M,
    drag_coefficient: float | None,
    heat_transfer_coefficient: float | None,
    epoch: np.datetime64,
    reference_ecef: NDArray_3,
    v_dir_ecef: NDArray_3,
    options: KeroSzaszOptions,
):
    """The right hand side of the differential equation to be integrated, i.e:

    .. math::

        \\frac{\\mathrm{d}v}{\\mathrm{d}t} = \\\\
        \\\\
        \\frac{\\mathrm{d}m}{\\mathrm{d}t} = \\\\
        \\\\
        \\frac{\\mathrm{d}s}{\\mathrm{d}t} = \\\\
        \\\\
        \\frac{\\mathrm{d}T}{\\mathrm{d}t} = 


    The numpy vector is structured as follows (numbers indicating index):

        0. dvdt
        1. dmdt
        2. dsdt
        3. dTdt

    """
    vel, mass, s, temp = y

    # meteoroid height above earth surface
    point_ecef = reference_ecef + v_dir_ecef * s
    lat, lon, alt = frames.ecef_to_geodetic_wgs84(
        point_ecef[0], point_ecef[1], point_ecef[2], degrees=True
    )
    r = np.linalg.norm(point_ecef)

    atm = options.atmosphere.density(
        time=epoch + np.timedelta64(int(t * 1e6), "us"),
        lat=lat,
        lon=lon,
        alt=alt,
        mass_densities=False,
        **options.atmosphere_kwargs,
    )

    rho_tot = atm["Total"].values.squeeze()
    N_rho_tot = rho_tot / options.atmosphere.mean_mass

    if options.sputtering:
        dmdt_s = physics.sputtering.sputtering_kero_szasz_2008(
            mass=mass,
            velocity=vel,
            material_data=options.material,
            density=atm,
        )
    else:
        dmdt_s = 0.0

    dmdt_a = physics.thermal_ablation.thermal_ablation_hill_et_al_2005(
        mass=mass,
        temperature=temp,
        material_data=options.material,
        shape_factor=options.shape_factor,
    )

    if heat_transfer_coefficient is None:
        heat_transfer_coefficient = (
            physics.thermal_ablation.heat_transfer_bronshten_1983(
                mass=mass,
                velocity=vel,
                temperature=temp,
                material_data=options.material,
                atm_total_number_density=N_rho_tot,
                mass_loss_thermal_ablation=dmdt_a,
                atm_mean_mass=options.atmosphere.mean_mass,
                res=options.integral_resolution,
            )
        )

    if drag_coefficient is None:
        drag_coefficient = physics.thermal_ablation.drag_coefficient_bronshten_1983(
            mass=mass,
            velocity=vel,
            temperature=temp,
            material_data=options.material,
            atm_total_number_density=N_rho_tot,
            atm_mean_mass=options.atmosphere.mean_mass,
            res=options.integral_resolution,
        )

    # -- Differential equation for the velocity to solve
    dvdt_d = (
        -drag_coefficient
        * options.shape_factor
        * rho_tot
        * vel**2
        / (mass ** (1.0 / 3.0) * options.material.bulk_density ** (2.0 / 3.0))
    )  # [m/s2] drag equation (because of conservation of linear momentum):

    # TODO: add bending of trajectory due to gravity - easy via angle equations in stulov
    # decelearation=Drag_coeff*shape_factor*atm_dens*vel/(mass^1/2 * meteoroid_dens^2/3)
    dvdt_g = WGS84.GM / (r**2)  # [m/s2] acceleration due to earth gravitaion
    dvdt = dvdt_d + dvdt_g

    # -- Differential equation for the displacement to solve
    dsdt = vel  # range from the common volume along the meteoroid trajectory

    dTdt = physics.thermal_ablation.temperature_rate_hill_et_al_2005(
        mass=mass,
        velocity=vel,
        temperature=temp,
        material_data=options.material,
        shape_factor=options.shape_factor,
        atm_total_mass_density=rho_tot,
        mass_loss_thermal_ablation=-dmdt_a,
        Lambda=heat_transfer_coefficient,
        atm_temperature=options.effective_atmospheric_temperature,
        emissivity=options.emissivity,
    )

    dmdt = dmdt_a + dmdt_s  # total mass loss

    logger.debug(
        f"""
            {mass=}
            {vel=}
            {temp=}
            {options.material=}
            {options.shape_factor=}
            {rho_tot=}
            {dmdt_a=}
            {heat_transfer_coefficient=}
            {options.effective_atmospheric_temperature=}
            {options.emissivity=}
        """
    )
    logger.debug(
        f"DERIVS: vel = {dvdt * 1e-3} km/s^2, traj-s = {dsdt * 1e-3}"
        f"km/s, mass = {dmdt} kg/s, temp = {dTdt} K/s"
    )
    if np.isnan(dTdt):
        logger.debug(f"dmdt sputtering: {dmdt_s} kg/s")
        logger.debug(f"dmdt ablation  : {dmdt_a} kg/s")
        logger.debug(f"Position: {point_ecef * 1e-3} km")
        logger.debug(f"lat = {lat}, lon = {lon}, {alt * 1e-3} km")
        logger.debug(f"t0 + {t} s: ")
        logger.debug(
            f"vel = {vel * 1e-3} km/s, traj-s = {s * 1e-3} km, mass = {mass} kg, temp = {temp} K"
        )
        logger.debug(f"altitude = {alt * 1e-3} km")
        breakpoint()

    ret = np.array([dvdt, dmdt, dsdt, dTdt], dtype=np.float64)

    # TODO: make this return aux variables such as massloss and then wrap it in a iterator and return
    # only derivs for diff eq solving - that makes it easy to extract the aux variables later

    return ret
