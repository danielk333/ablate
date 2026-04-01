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

import logging
from typing import Union
import numpy as np
from scipy import constants
import xarray

from ..ode import ScipyODESolve
from ..atmosphere import AtmPymsis, AtmMSISE00

logger = logging.getLogger(__name__)


class StulovMirskiiVislyi1995(ScipyODESolve):
    """Ablation model

    Keyword arguments:


    """

    DEFAULT_CONFIG = {
        "options": {},
        "atmosphere": {
            "f107": 80.0,
            "f107s": 80.0,
            "version": 2.1,
        },
        "integrate": {
            "minimum_mass_kg": 1e-9,
            "max_step_size_sec": 1e-1,
            "max_time_sec": 100.0,
            "method": "RK45",
        },
        "method_options": {},
    }

    def __init__(self, atmosphere: Union[AtmPymsis, AtmMSISE00], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._G = constants.G
        self._M = 5.9742e24  # [kg] mass of earth
        self._R_e = 6371.0088e3  # [m] radius of earth (IUGG)
        self.atmosphere = atmosphere

    def to_trajectory(self, gamma, vel, reference_azimith, reference_ecef):
        pass

    def rhs(
        self,
        t,
        mass,
        y,
        epoch: np.datetime64,
        lat: float,
        lon: float,
        enthalpy_of_massloss: float,
        initial_cross_sectional_area: float,
        aerodynamic_cd: float,
        aerodynamic_cl: float,
        heat_exchange_coefficient: float,
        shape_change_coefficient: float,
    ):
        """The right hand side of the differential equation to be integrated, i.e:

        .. math::

            \\frac{\\mathrm{d}m}{\\mathrm{d}t} = \\\\
            \\\\
            \\frac{\\mathrm{d}v}{\\mathrm{d}t} = \\\\
            \\\\
            \\frac{\\mathrm{d}h}{\\mathrm{d}t} = \\\\
            \\\\
            \\frac{\\mathrm{d}s}{\\mathrm{d}t} = \\\\
            \\\\
            \\frac{\\mathrm{d}\\gamma}{\\mathrm{d}t} = 


        The numpy vector is structured as follows (numbers indicating index):

            0. dmdt
            1. dvdt
            2. dsdt
            3. dhdt
            4. dgammadt

        """
        vel, dist, height, gamma = y

        f107 = self.config.getfloat("atmosphere", "f107")
        f107s = self.config.getfloat("atmosphere", "f107s")

        atm = self.atmosphere.density(
            time=epoch + np.timedelta64(int(t * 1e6), "us"),
            lat=lat,
            lon=lon,
            alt=height,
            f107=f107,
            f107s=f107s,
            mass_densities=False,
            version=self.config.getfloat("atmosphere", "version"),
        )

        rho_atm = atm["Total"].values.squeeze()

        area = initial_cross_sectional_area * (mass**shape_change_coefficient)
        drag_f = 0.5 * aerodynamic_cd * rho_atm * vel**2 * area
        lift_f = 0.5 * aerodynamic_cl * rho_atm * vel**2 * area
        mass_loss_r = 0.5 * heat_exchange_coefficient * rho_atm * vel**3 * area
        grav_a = self._G * self._M / (self._R_e + height) ** 2

        #  dMdt
        #  dVdt
        #  dsdt
        #  dgammadt
        # -- Differential equations to solve
        dmdt = -mass_loss_r / enthalpy_of_massloss
        dvdt = -drag_f / mass + grav_a * np.sin(gamma)
        dsdt = vel
        dhdt = -vel * np.sin(gamma)
        dgammadt = (grav_a / vel - vel / self._R_e) * np.cos(gamma) - lift_f / (mass * vel)

        ret = np.array([dmdt, dvdt, dsdt, dhdt, dgammadt], dtype=np.float64)

        # logging.debug(
        #     f"DERIVS: vel = {dvdt*1e-3} km/s^2, traj-s = {dsdt*1e-3}"
        #     f"km/s, mass = {dmdt} kg/s, gamma = {dgammadt} rad/s"
        # )
        # logger.debug(f"t0 + {t} s: ")
        # logger.debug(f"vel = {vel*1e-3} km/s, traj dist = {dist*1e-3} km")
        # logger.debug(f"mass = {mass} kg, ang = {np.degrees(gamma)} deg")
        # logger.debug(f"height = {height*1e-3} km")

        return ret

    def run(
        self,
        initial_velocity: float,
        initial_mass: float,
        initial_altitude: float,
        initial_radiant_local_elevation: float,
        epoch: np.datetime64,
        lat: float,
        lon: float,
        enthalpy_of_massloss: float,
        initial_cross_sectional_area: float,
        aerodynamic_cd: float = 0.47,
        aerodynamic_cl: float = 0,
        heat_exchange_coefficient: float = 1,
        shape_change_coefficient: float = 2 / 3,
    ):
        """Runs the ablation model."""
        logger.debug(f"Running {self.__class__} model")
        initial_distance = 0.0
        y0 = np.array(
            [
                initial_mass,
                initial_velocity,
                initial_distance,
                initial_altitude,
                initial_radiant_local_elevation,
            ],
            dtype=np.float64,
        )

        ivp_result = self.integrate(
            y0,
            epoch,
            lat,
            lon,
            enthalpy_of_massloss,
            initial_cross_sectional_area,
            aerodynamic_cd,
            aerodynamic_cl,
            heat_exchange_coefficient,
            shape_change_coefficient,
        )
        t = ivp_result.t
        _data = {}
        variables = [
            "mass",
            "velocity",
            "position",
            "height",
            "trajectory_angle",
        ]
        for key in variables:
            _data[key] = (["t"], np.empty(t.shape, dtype=np.float64))

        results = xarray.Dataset(
            _data,
            coords={"t": t},
            attrs={key: val for key, val in self.config.items("options")},
        )

        results["mass"][:] = ivp_result.y[0, :]
        results["velocity"][:] = ivp_result.y[1, :]
        results["position"][:] = ivp_result.y[2, :]
        results["height"][:] = ivp_result.y[3, :]
        results["trajectory_angle"][:] = ivp_result.y[4, :]

        return results
