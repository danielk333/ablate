#!/usr/bin/env python

'''The basic class structures for implementing a ablation model.

'''

# Basic Python
import logging
import copy

logger = logging.getLogger(__name__)

# External packages
import numpy as np
import scipy
from scipy import constants
import xarray


# Internal packages
from ..ode import ScipyODESolve
from .. import functions
from .. import atmosphere as atm


class KeroSzasz2008(ScipyODESolve):
    '''Ablation model

    '''

    ATMOSPHERES = {}
    DEFAULT_OPTIONS = copy.deepcopy(ScipyODESolve.DEFAULT_OPTIONS)
    DEFAULT_OPTIONS.update(dict(
        temperature0 = 290,
        shape_factor = 1.21,
        emissivity = 0.9,
        sputtering = True,
        Gamma = None,
        Lambda = None,
        integral_resolution = 100,
    ))

    def __init__(self,
                *args,
                **kwargs
            ):
        super().__init__(*args, **kwargs)
        logger.debug(f'{self.__class__} instance initialized')
        self._G = constants.G
        self._M = 5.9742E24 #[kg] mass of earth



    def rhs(self, t, mass, y, material_data, Lambda, Gamma, epoch):
        '''The right hand side of the differential equation to be integrated, i.e:

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

        '''
        vel, s, T = y

        rho_m = material_data['rho_m']

        lat, lon, alt = self.s_to_geo(s) #meteoroid height above earth surface

        ecef = functions.coordinate.geodetic2ecef(lat, lon, alt)
        r = np.linalg.norm(ecef)

        #logger.debug(f'Position: {ecef*1e-3} km')
        #logger.debug(f'lat = {lat}, lon = {lon}, {alt*1e-3} km')

        #logger.debug(f't0 + {t} s: ')
        #logger.debug(f'vel = {vel*1e-3} km/s, traj-s = {s*1e-3} km, mass = {mass} kg, temp = {T} K')
        #logger.debug(f'altitude = {alt*1e-3} km')

        atm = self.get_atmosphere(
            time = epoch + np.timedelta64(int(t*1e6), 'us'),
            lat = lat,
            lon = lon,
            alt = alt,
        )

        N_rho_tot = atm["Total"].values.squeeze() / self.atm_mean_mass

        if self.options['sputtering']:
            dmdt_s = functions.sputtering.sputtering(
                mass = mass,
                velocity = vel,
                material_data = material_data,
                density = atm,
            )
        else:
            dmdt_s = 0.0
        
        dmdt_a = functions.ablation.thermal_ablation(
            mass = mass,
            temperature = T,
            material_data = material_data,
            shape_factor = self.options['shape_factor'],
        )

        #logger.debug(f'dmdt sputtering: {dmdt_s} kg/s')
        #logger.debug(f'dmdt ablation  : {dmdt_a} kg/s')

        if Lambda is None:
            Lambda = functions.dynamics.heat_transfer(
                mass = mass,
                velocity = vel,
                temperature = T,
                material_data = material_data,
                thermal_ablation = dmdt_a,
                atm_mean_mass = self.atm_mean_mass,
                res = self.options['integral_resolution'],
                atm_total_density=N_rho_tot,
            )

        if Gamma is None:
            Gamma = functions.dynamics.drag_coefficient(
                mass = mass,
                velocity = vel,
                temperature = T,
                material_data = material_data,
                atm_mean_mass = self.atm_mean_mass,
                res = self.options['integral_resolution'],
                atm_total_density=N_rho_tot,
            )

        #-- Differential equation for the velocity to solve
        dvdt_d = -Gamma*self.options['shape_factor']*rho_tot*vel**2/(mass**(1.0/3.0)*rho_m**(2.0/3.0)) #[m/s2] drag equation (because of conservation of linear momentum): decelearation=Drag_coeff*shape_factor*atm_dens*vel/(mass^1/2 * meteoroid_dens^2/3)
        dvdt_g = self._G*self._M/(r**2) #[m/s2] acceleration due to earth gravitaion
        dvdt = dvdt_d + dvdt_g

        #-- Differential equation for the height to solve
        dsdt = -vel #range from the common volume along the meteoroid trajectory

        dTdt = functions.ablation.temperature_rate(
            mass = mass,
            velocity = vel,
            temperature = T,
            material_data = material_data,
            shape_factor = self.options['shape_factor'],
            thermal_ablation = dmdt_a,
            Lambda = Lambda,
            atm_temperature = self.options['temperature0'],
            emissivity = self.options['emissivity'],
            atm_total_density=N_rho_tot,
        )

        dmdt = dmdt_a + dmdt_s #total mass loss

        ret = np.array([dmdt, dvdt, dsdt, dTdt], dtype=np.float64)

        #logging.debug(f'DERIVS: vel = {dvdt*1e-3} km/s^2, traj-s = {dsdt*1e-3} km/s, mass = {dmdt} kg/s, temp = {dTdt} K/s')

        return ret


    def run(self,
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
        '''This function is based on calc_sput.m which was used to verify the sputtering described in Rogers et al.: Mass loss due to  sputtering and thermal processes in meteoroid ablation, Planetary and Space Science 53 p. 1341-1354 (2005).
        
        :param float/numpy.ndarray velocity0: Meteoroid initial velocity [m/s]
        :param float/numpy.ndarray mass0: Meteoroid initial mass [kg]
        :param float/numpy.ndarray altitude0: Meteoroid initial altitude [m]
        :param float/numpy.ndarray zenith_ang: Zenith angle of the trajectory (-velocity vector) w.r.t reference point [deg]
        :param float/numpy.ndarray azimuth_ang: Azimuthal angle east of north of the trajectory (-velocity vector) w.r.t reference point [deg]
        :param dict material_data: Meteoroid material data, see :mod:`~functions.material.material_data`.
        :param float/numpy.ndarray lat: Geographic latitude in degrees of reference point on the meteoroid trajectory
        :param float/numpy.ndarray lon: Geographic longitude in degrees of reference point on the meteoroid trajectory 
        :param float/numpy.ndarray alt: Altitude above geoid in meters of reference point on the meteoroid trajectory
        
        
        #TODO: Add additional dynamical parameters to data structure e.g. lambda and gamma if they are not constant


        **Keyword arguments:**

            * temperature0 = 290 [K]: Meteoroid temperature at starting height
            * shape_factor = 1.21 [1]: Shape is assumed to be a sphere.
            * emissivity = 0.9 [1]: Hill et al.; Love & Brownlee: 1; (metal oxides)
            * sputtering = True [bool]: If sputtering is used in mass loss calculation.
            * Gamma = None [1]: Drag coefficient, if :code:`None` it is dynamically calculated assuming a transition from (and including) free molecular flow to a (and not including) shock regime. Otherwise assumed a constant with the given value.
            * Lambda = None [1]: Heat transfer coefficient, if :code:`None` it is dynamically calculated assuming a transition from (and including) free molecular flow to a (and not including) shock regime. Otherwise assumed a constant with the given value.

        '''
        logger.debug(f'Running {self.__class__} model')

        meta = self.get_atmosphere_meta()
        self.atm_mean_mass = np.array([x['A'] for _,x in meta.items()]).mean() * constants.u

        reference_ecef = functions.coordinate.geodetic2ecef(lat, lon, alt)
        v_dir = -1.0*functions.coordinate.azel_to_cart(azimuth_ang, zenith_ang, 1.0)
        v_dir_ecef = functions.coordinate.enu2ecef(lat, lon, alt, *v_dir.tolist())

        def s_to_geo(s):
            traj = reference_ecef - v_dir_ecef*s
            geo = functions.coordinate.ecef2geodetic(*traj.tolist())
            return geo

        self.s_to_geo = s_to_geo

        s0 = scipy.optimize.minimize_scalar(lambda s: np.abs(s_to_geo(s)[2] - altitude0)).x
        
        y0 = np.array([mass0, velocity0, s0, self.options['temperature0']], dtype=np.float64)

        self.integrate(
            y0, 
            material_data, 
            self.options['Lambda'], 
            self.options['Gamma'],
            time,
        )

        self._allocate(self._ivp_result.t)
        self.results['mass'][:] = self._ivp_result.y[0,:]
        self.results['velocity'][:] = self._ivp_result.y[1,:]
        self.results['position'][:] = self._ivp_result.y[2,:]
        self.results['temperature'][:] = self._ivp_result.y[3,:]
        
        ecef = self._ivp_result.y[2,:]
        ecef = reference_ecef[:,None] - v_dir_ecef[:,None]*ecef[None,:]
        alts = np.array([self.s_to_geo(_s)[2] for _s in self._ivp_result.y[2,:]])

        self.results['ecef_x'][:] = ecef[0,:]
        self.results['ecef_y'][:] = ecef[1,:]
        self.results['ecef_z'][:] = ecef[2,:]
        self.results['altitude'][:] = alts

        return self.results


    def _allocate(self, t):

        _data = {}
        for key in ['mass','velocity','position','temperature', 'ecef_x', 'ecef_y', 'ecef_z', 'altitude']:
            _data[key] = (['t'], np.empty(t.shape, dtype=np.float64))

        self.results = xarray.Dataset(
            _data,
            coords = {'t': t},
            attrs = {key:val for key, val in self.options.items()}
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

    KeroSzasz2008._register_atmosphere('msise00', msise00.density, _meta_getter)
else:
    def _msise_getter(*args, **kwargs):
        raise ImportError('msise00 import error: cannot use "msise00" as atmosphere. Plase confirm it has been installed.')
    KeroSzasz2008._register_atmosphere('msise00', _msise_getter, _msise_getter)