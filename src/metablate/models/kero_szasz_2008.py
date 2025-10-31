#!/usr/bin/env python

"""Model implemented in:

Kero, J., Szasz, C., Pellinen-Wannberg, A., Wannberg, G., Westman, A., and Meisel, D. D. (2008). 
Determination of meteoroid physical properties from tristatic radar observations.
Annales Geophysicae
"""

import logging
from typing import Union
import numpy as np
import scipy
from scipy import constants
import xarray

from ..ode import ScipyODESolve
from .. import physics
from .. import coordinates
from ..atmosphere import AtmPymsis, AtmMSISE00

logger = logging.getLogger(__name__)


class KeroSzasz2008(ScipyODESolve):
    """Ablation model

    Keyword arguments:

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

    DEFAULT_CONFIG = {
        "options": {
            "temperature0": 290,
            "shape_factor": 1.21,
            "emissivity": 0.9,
            "sputtering": False,
            "Gamma": None,
            "Lambda": None,
            "integral_resolution": 100,
        },
        "atmosphere": {
            "f107": None,
            "f107s": None,
            "version": 2.1,
        },
        "integrate": {
            "minimum_mass_kg": 1e-11,
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
        self.atmosphere = atmosphere

    def rhs(
        self,
        t,
        mass,
        y,
        material_data,
        Lambda,
        Gamma,
        epoch,
        reference_ecef,
        v_dir_ecef,
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
        vel, s, T = y

        # meteoroid height above earth surface
        traj = reference_ecef - v_dir_ecef * s
        lat, lon, alt = coordinates.ecef2geodetic(*traj.tolist())

        ecef = coordinates.geodetic2ecef(lat, lon, alt)
        r = np.linalg.norm(ecef)

        # logger.debug(f'Position: {ecef*1e-3} km')
        # logger.debug(f'lat = {lat}, lon = {lon}, {alt*1e-3} km')

        # logger.debug(f't0 + {t} s: ')
        # logger.debug(f'vel = {vel*1e-3} km/s, traj-s = {s*1e-3} km, mass = {mass} kg, temp = {T} K')
        # logger.debug(f'altitude = {alt*1e-3} km')

        f107 = self.config.get("atmosphere", "f107")
        if f107 is not None:
            f107 = float(f107)
        f107s = self.config.get("atmosphere", "f107s")
        if f107s is not None:
            f107s = float(f107s)

        atm = self.atmosphere.density(
            time=epoch + np.timedelta64(int(t * 1e6), "us"),
            lat=lat,
            lon=lon,
            alt=alt,
            f107=f107,
            f107s=f107s,
            mass_densities=False,
            version=self.config.getfloat("atmosphere", "version"),
        )

        rho_tot = atm["Total"].values.squeeze()
        N_rho_tot = rho_tot / self.atmosphere.mean_mass

        if self.config.getboolean("options", "sputtering"):
            dmdt_s = physics.sputtering.sputtering(
                mass=mass,
                velocity=vel,
                material_data=material_data,
                density=atm,
            )
        else:
            dmdt_s = 0.0

        dmdt_a = physics.thermal_ablation.thermal_ablation_hill_et_al_2005(
            mass=mass,
            temperature=T,
            material_data=material_data,
            shape_factor=self.config.getfloat("options", "shape_factor"),
        )

        # logger.debug(f'dmdt sputtering: {dmdt_s} kg/s')
        # logger.debug(f'dmdt ablation  : {dmdt_a} kg/s')

        if Lambda is None:
            Lambda = physics.thermal_ablation.heat_transfer_bronshten_1983(
                mass=mass,
                velocity=vel,
                temperature=T,
                material_data=material_data,
                atm_total_number_density=N_rho_tot,
                mass_loss_thermal_ablation=dmdt_a,
                atm_mean_mass=self.atmosphere.mean_mass,
                res=self.config.getint("options", "integral_resolution"),
            )

        if Gamma is None:
            Gamma = physics.thermal_ablation.drag_coefficient_bronshten_1983(
                mass=mass,
                velocity=vel,
                temperature=T,
                material_data=material_data,
                atm_total_number_density=N_rho_tot,
                atm_mean_mass=self.atmosphere.mean_mass,
                res=self.config.getint("options", "integral_resolution"),
            )

        # -- Differential equation for the velocity to solve
        dvdt_d = (
            -Gamma
            * self.config.getfloat("options", "shape_factor")
            * rho_tot
            * vel**2
            / (mass ** (1.0 / 3.0) * material_data["rho_m"] ** (2.0 / 3.0))
        )  # [m/s2] drag equation (because of conservation of linear momentum):
        # decelearation=Drag_coeff*shape_factor*atm_dens*vel/(mass^1/2 * meteoroid_dens^2/3)
        dvdt_g = self._G * self._M / (r**2)  # [m/s2] acceleration due to earth gravitaion
        dvdt = dvdt_d + dvdt_g

        # -- Differential equation for the height to solve
        dsdt = -vel  # range from the common volume along the meteoroid trajectory

        dTdt = physics.thermal_ablation.temperature_rate_hill_et_al_2005(
            mass=mass,
            velocity=vel,
            temperature=T,
            material_data=material_data,
            shape_factor=self.config.getfloat("options", "shape_factor"),
            atm_total_mass_density=rho_tot,
            mass_loss_thermal_ablation=-dmdt_a,
            Lambda=Lambda,
            atm_temperature=self.config.getfloat("options", "temperature0"),
            emissivity=self.config.getfloat("options", "emissivity"),
        )

        dmdt = dmdt_a + dmdt_s  # total mass loss

        ret = np.array([dmdt, dvdt, dsdt, dTdt], dtype=np.float64)

        # logging.debug(
        #   f"DERIVS: vel = {dvdt*1e-3} km/s^2, traj-s = {dsdt*1e-3}"
        #   f"km/s, mass = {dmdt} kg/s, temp = {dTdt} K/s"
        # )

        return ret

    def run(
        self,
        velocity0,
        mass0,
        altitude0,
        zenith_ang,
        azimuth_ang,
        material_data,
        time,
        lat,
        lon,
        alt,
    ):
        """Runs the ablation model.

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
        This function is based on calc_sput.m which was used to verify the
        sputtering described in Rogers et al.: Mass loss due to sputtering and
        thermal processes in meteoroid ablation,
        Planetary and Space Science 53 p. 1341-1354 (2005).

        TODO: Add additional dynamical parameters to data structure e.g.
        lambda and gamma if they are not constant

        """
        logger.debug(f"Running {self.__class__} model")

        reference_ecef = coordinates.geodetic2ecef(lat, lon, alt)
        v_dir = -1.0 * coordinates.azel_to_cart(azimuth_ang, zenith_ang, 1.0)
        v_dir_ecef = coordinates.enu2ecef(lat, lon, alt, *v_dir.tolist())

        def s_to_geo(s):
            traj = reference_ecef - v_dir_ecef * s
            geo = coordinates.ecef2geodetic(*traj.tolist())
            return geo

        s0 = scipy.optimize.minimize_scalar(lambda s: np.abs(s_to_geo(s)[2] - altitude0)).x
        t0 = self.config.getfloat("options", "temperature0")
        y0 = np.array(
            [mass0, velocity0, s0, t0],
            dtype=np.float64,
        )

        Lambda = self.config.get("options", "Lambda")
        if Lambda is not None:
            Lambda = float(Lambda)
        Gamma = self.config.get("options", "Gamma")
        if Gamma is not None:
            Gamma = float(Gamma)

        ivp_result = self.integrate(
            y0,
            material_data,
            Lambda,
            Gamma,
            time,
            reference_ecef,
            v_dir_ecef,
        )
        t = ivp_result.t
        _data = {}
        variables = [
            "mass",
            "velocity",
            "position",
            "temperature",
            "ecef_x",
            "ecef_y",
            "ecef_z",
            "altitude",
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
        results["temperature"][:] = ivp_result.y[3, :]

        ecef = ivp_result.y[2, :]
        ecef = reference_ecef[:, None] - v_dir_ecef[:, None] * ecef[None, :]
        alts = np.array([s_to_geo(_s)[2] for _s in ivp_result.y[2, :]])

        results["ecef_x"][:] = ecef[0, :]
        results["ecef_y"][:] = ecef[1, :]
        results["ecef_z"][:] = ecef[2, :]
        results["altitude"][:] = alts

        return results
