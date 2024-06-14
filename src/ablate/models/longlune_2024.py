#!/usr/bin/env python

"""The basic class structures for implementing a ablation model.

"""

# Basic Python
import logging
import copy

# External packages
import numpy as np
import scipy
from scipy import constants
import xarray


# Internal packages
from ..ode import ScipyODESolve
from .. import functions
from .. import atmosphere as atm

import sys

sys.path.append("/home/danielk/git/pyCabaret/src")
import modeleGSI_copy as mod

logger = logging.getLogger(__name__)


class Longlune2024(ScipyODESolve):
    """Ablation model"""

    ATMOSPHERES = {}
    DEFAULT_OPTIONS = copy.deepcopy(ScipyODESolve.DEFAULT_OPTIONS)
    DEFAULT_OPTIONS.update(
        dict(
            temperature0=290,
            shape_factor=1.21,
            emissivity=0.9,
            Gamma=None,
            integral_resolution=100,
        )
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug(f"{self.__class__} instance initialized")
        self._G = constants.G
        self._M = 5.9742e24  # [kg] mass of earth

    def rhs(self, t, mass, y, material_data, Gamma, epoch):
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
        vel, s, Twall = y

        rho_m = material_data["rho_m"]
        C = material_data["c"]

        lat, lon, alt = self.s_to_geo(s)  # meteoroid height above earth surface

        ecef = functions.coordinate.geodetic2ecef(lat, lon, alt)
        r = np.linalg.norm(ecef)

        # logger.debug(f'Position: {ecef*1e-3} km')
        # logger.debug(f'lat = {lat}, lon = {lon}, {alt*1e-3} km')

        # logger.debug(f't0 + {t} s: ')
        # logger.debug(f'vel = {vel*1e-3} km/s, traj-s = {s*1e-3} km, mass = {mass} kg, temp = {T} K')
        # logger.debug(f'altitude = {alt*1e-3} km')

        atm = self.get_atmosphere(
            time=epoch + np.timedelta64(int(t * 1e6), "us"),
            lat=lat,
            lon=lon,
            alt=alt,
        )

        rho_tot = atm["Total"].values.squeeze()

        radius = np.cbrt(mass * 3 / (4 * rho_m * np.pi))

        kB = constants.value(u'Boltzmann constant')  # [J/K]
        # reff = 0.48  # HOW TO LINK IT TO link reff and the dx of the boundary layer.
        [heatflux, dmdt, electron_number_density] = mod.modele_gsi(
            vel, alt*1e-3, radius, Twall
        )
        # rad_heatflux = 4*kB*self.options['emissivity']*(Twall**4 - self.options['temperature0']**4)
        rad_heatflux = 0
        heat = (heatflux - rad_heatflux) * (4 * np.pi * radius ** 2)  # [W] = [J/s]
        # do with shape factor?
        vol = 4 / 3 * np.pi * radius ** 3
        dTdt = heat * self.options['shape_factor'] / (C * rho_m * vol)
        # C =710 specifif heat to get from carbon mat data base.  [J/(kg·K)]
        # rho_m: 2267.0 density get from carbon mat data base [kg/m^3]

        # logger.debug(f'dmdt sputtering: {dmdt_s} kg/s')
        # logger.debug(f'dmdt ablation  : {dmdt_a} kg/s')

        # -- Differential equation for the velocity to solve
        dvdt_d = (
            -Gamma
            * self.options["shape_factor"]
            * rho_tot
            * vel**2
            / (mass ** (1.0 / 3.0) * rho_m ** (2.0 / 3.0))
        )  # [m/s2] drag equation (because of conservation of linear momentum):
        # decelearation=Drag_coeff*shape_factor*atm_dens*vel/(mass^1/2 * meteoroid_dens^2/3)
        dvdt_g = (
            self._G * self._M / (r**2)
        )  # [m/s2] acceleration due to earth gravitaion
        dvdt = dvdt_d + dvdt_g
        # -- Differential equation for the height to solve
        dsdt = -vel  # range from the common volume along the meteoroid trajectory

        if Twall > 3800:
            dTdt = 0

        ret = np.array([dmdt, dvdt, dsdt, dTdt], dtype=np.float64)

        print(
            f"y   : alt = {alt*1e-3} km, vel = {vel*1e-3} km/s, traj-s = {s*1e-3} km, "
            f"mass = {mass} kg, temp = {Twall} K\n"
            f"dydt: vel = {dvdt*1e-3} km/s^2, traj-s = {dsdt*1e-3} km/s, "
            f"mass = {dmdt} kg/s, temp = {dTdt} K/s"
        )

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
        """This function is based on calc_sput.m which was used to verify the
        sputtering described in Rogers et al.: Mass loss due to sputtering and
        thermal processes in meteoroid ablation,
        Planetary and Space Science 53 p. 1341-1354 (2005).

        :param float/numpy.ndarray velocity0: Meteoroid initial velocity [m/s]
        :param float/numpy.ndarray mass0: Meteoroid initial mass [kg]
        :param float/numpy.ndarray altitude0: Meteoroid initial altitude [m]
        :param float/numpy.ndarray zenith_ang:
            Zenith angle of the trajectory (-velocity vector) w.r.t reference point [deg]
        :param float/numpy.ndarray azimuth_ang:
            Azimuthal angle east of north of the trajectory (-velocity vector) w.r.t reference point [deg]
        :param dict material_data:
            Meteoroid material data, see :mod:`~functions.material.material_data`.
        :param float/numpy.ndarray lat:
            Geographic latitude in degrees of reference point on the meteoroid trajectory
        :param float/numpy.ndarray lon:
            Geographic longitude in degrees of reference point on the meteoroid trajectory
        :param float/numpy.ndarray alt:
            Altitude above geoid in meters of reference point on the meteoroid trajectory


        #TODO: Add additional dynamical parameters to data structure e.g.
        #   lambda and gamma if they are not constant


        **Keyword arguments:**

            * temperature0 = 290 [K]: Meteoroid temperature at starting height
            * shape_factor = 1.21 [1]: Shape is assumed to be a sphere.
            * emissivity = 0.9 [1]: Hill et al.; Love & Brownlee: 1; (metal oxides)
            * sputtering = True [bool]: If sputtering is used in mass loss calculation.
            * Gamma = None [1]: Drag coefficient, if :code:`None` it is dynamically
                calculated assuming a transition from (and including) free molecular
                flow to a (and not including) shock regime. Otherwise assumed a
                constant with the given value.
            * Lambda = None [1]: Heat transfer coefficient, if :code:`None` it is
                dynamically calculated assuming a transition from (and including)
                free molecular flow to a (and not including) shock regime.
                Otherwise assumed a constant with the given value.

        """
        logger.debug(f"Running {self.__class__} model")

        meta = self.get_atmosphere_meta()
        self.atm_mean_mass = (
            np.array([x["A"] for _, x in meta.items()]).mean() * constants.u
        )

        reference_ecef = functions.coordinate.geodetic2ecef(lat, lon, alt)
        v_dir = -1.0 * functions.coordinate.azel_to_cart(azimuth_ang, zenith_ang, 1.0)
        v_dir_ecef = functions.coordinate.enu2ecef(lat, lon, alt, *v_dir.tolist())

        def s_to_geo(s):
            traj = reference_ecef - v_dir_ecef * s
            geo = functions.coordinate.ecef2geodetic(*traj.tolist())
            return geo

        self.s_to_geo = s_to_geo

        s0 = scipy.optimize.minimize_scalar(
            lambda s: np.abs(s_to_geo(s)[2] - altitude0)
        ).x

        y0 = np.array(
            [mass0, velocity0, s0, self.options["temperature0"]], dtype=np.float64
        )

        self.integrate(
            y0,
            material_data,
            self.options["Gamma"],
            time,
        )

        self._allocate(self._ivp_result.t)
        self.results["mass"][:] = self._ivp_result.y[0, :]
        self.results["velocity"][:] = self._ivp_result.y[1, :]
        self.results["position"][:] = self._ivp_result.y[2, :]
        self.results["temperature"][:] = self._ivp_result.y[3, :]

        ecef = self._ivp_result.y[2, :]
        ecef = reference_ecef[:, None] - v_dir_ecef[:, None] * ecef[None, :]
        alts = np.array([self.s_to_geo(_s)[2] for _s in self._ivp_result.y[2, :]])

        self.results["ecef_x"][:] = ecef[0, :]
        self.results["ecef_y"][:] = ecef[1, :]
        self.results["ecef_z"][:] = ecef[2, :]
        self.results["altitude"][:] = alts

        return self.results

    def _allocate(self, t):

        _data = {}
        for key in [
            "mass",
            "velocity",
            "position",
            "ecef_x",
            "ecef_y",
            "ecef_z",
            "altitude",
        ]:
            _data[key] = (["t"], np.empty(t.shape, dtype=np.float64))

        self.results = xarray.Dataset(
            _data,
            coords={"t": t},
            attrs={key: val for key, val in self.options.items()},
        )


try:
    import msise00 as msise_test

    msise_test = True
except ImportError:
    msise_test = False

if msise_test:
    msise00 = atm.NRLMSISE00()

    def _meta_getter():
        return msise00.species

    Longlune2024._register_atmosphere("msise00", msise00.density, _meta_getter)
else:

    def _msise_getter(*args, **kwargs):
        raise ImportError(
            'msise00 import error: cannot use "msise00" as atmosphere. Plase confirm it has been installed.'
        )

    Longlune2024._register_atmosphere("msise00", _msise_getter, _msise_getter)
