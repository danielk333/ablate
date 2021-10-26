#!/usr/bin/env python

'''The basic class structures for implementing a ablation model.

'''

#
# Basic Python
#


#
# External packages
#
import numpy as np
import scipy
from scipy import constants
import xarray

#
# Internal packages
#
from .ode import OrdinaryDifferentialEquation
from . import functions
from .base_models import AblationModel


__all__ = ['KeroSzasz2008']


class KeroSzasz2008(AblationModel, OrdinaryDifferentialEquation):
    '''Ablation model

    '''


    def __init__(self,
                atmosphere,
                npt,
                **kwargs
            ):
        '''Setup the ablation model.
        
        :param AtmosphericModel atmosphere: Atmospheric model used for ablation.
        :param numpy.datetime64/numpy.ndarray npt: Date and time to evaluate atmospheric model at.

        '''
        AblationModel.__init__(self, atmosphere)
        


        self._attrs = {
            'npt': npt,
            'atm_mean_mass': np.array([x['A'] for _,x in atmosphere.species.items()]).mean() * constants.u,
        }

        ode_kw = {}
        if 'method' in kwargs:
            ode_kw['method'] = kwargs['method']
        if 'options' in kwargs:
            ode_kw['options'] = kwargs['options']

        OrdinaryDifferentialEquation.__init__(self, **ode_kw)


    def _allocate(self, t):

        _data = {}
        for key in ['v','m','s','T']:
            _data[key] = (['t'], np.empty(t.shape, dtype=np.float64))

        #here we set what data is to be filled by the model.
        self.data = xarray.Dataset(
            _data,
            coords = {'t': t},
            attrs = {
                'zenith_ang': None, 
                'shape_factor': None, 
                'emissivity': None, 
                'sputtering': None, 
            }
        )


    def rhs(self, t, y):
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
        vel, mass, s, T = y


        rho_m = self.material_data['rho_m']
        G = constants.G
        M = 5.9742E24 #[kg] mass of earth

        lat, lon, alt = self.s_to_geo(s) #meteoroid height above earth surface

        ecef = functions.coordinate.geodetic2ecef(lat, lon, alt)
        r = np.linalg.norm(ecef)

        DEBUG = True

        if DEBUG:
            print('---------------')
            print(f'Position: {ecef*1e-3} km')
            print(f'lat = {lat}, lon = {lon}, {alt*1e-3} km')

            print(f't0 + {t} s: ')
            print(f'vel = {vel*1e-3} km/s, traj-s = {s*1e-3} km, mass = {mass} kg, temp = {T} K')

        atm = self.atmosphere.density(
            npt = self._attrs['npt'] + np.timedelta64(int(t*1e6), 'us'),
            lat = lat,
            lon = lon,
            alt = alt,
        )

        rho_tot = atm['Total'].values.squeeze()

        _data = {}
        for key in self.atmosphere.species.keys():
            _data[key] = (['met'], [atm[key].values.squeeze()])

        density = xarray.Dataset(
            _data,
           coords = {'met': np.arange(1)},
        )

        if self.sputtering:
            dmdt_s = functions.sputtering.sputtering(
                mass = mass,
                velocity = vel,
                material = self._attrs['material'],
                density = density,
            )
        else:
            dmdt_s = 0.0
        
        dmdt_a = functions.ablation.thermal_ablation(
            mass = mass,
            temperature = T,
            material = self._attrs['material'],
            shape_factor = self._attrs['A'],
        )

        if DEBUG:
            print(f'dmdt sputtering: {dmdt_s} kg/s')
            print(f'dmdt ablation  : {dmdt_a} kg/s')

        if self._attrs['Lambda'] is None:
            Lambda = functions.dynamics.heat_transfer(
                mass = mass,
                velocity = vel,
                temperature = T,
                material = self._attrs['material'],
                atm_total_density = rho_tot,
                thermal_ablation = dmdt_a,
                atm_mean_mass = self._attrs['atm_mean_mass'],
                res = 100,
            )
        else:
            Lambda = self._attrs['Lambda']

        if self._attrs['Gamma'] is None:
            Gamma = functions.dynamics.drag_coefficient(
                mass = mass,
                velocity = vel,
                temperature = T,
                material = self._attrs['material'],
                atm_total_density = rho_tot,
                atm_mean_mass = self._attrs['atm_mean_mass'],
                res = 100,
            )
        else:
            Gamma = self._attrs['Gamma']
        #-- Differential equation for the velocity to solve
        dvdt_d = -Gamma*self._attrs['A']*rho_tot*vel**2/(mass**(1.0/3.0)*rho_m**(2.0/3.0)) #[m/s2] drag equation (because of conservation of linear momentum): decelearation=Drag_coeff*shape_factor*atm_dens*vel/(mass^1/2 * meteoroid_dens^2/3)
        dvdt_g = G*M/(r**2) #[m/s2] acceleration due to earth gravitaion
        dvdt = dvdt_d + dvdt_g

        #-- Differential equation for the height to solve
        dsdt = -vel #range from the common volume along the meteoroid trajectory

        dTdt =  functions.ablation.temperature_rate(
            mass = mass,
            velocity = vel,
            temperature = T,
            material = self._attrs['material'],
            shape_factor = self._attrs['A'],
            atm_total_density = rho_tot,
            thermal_ablation = dmdt_a,
            Lambda = Lambda,
            atm_temperature = 280,
            emissivity = 0.9,
        )

        dmdt = dmdt_a + dmdt_s; #total mass loss

        ret = np.array([dvdt, dmdt, dsdt, dTdt], dtype=np.float64)#output

        if DEBUG:
            print(f'DERIVS: vel = {dvdt*1e-3} km/s^2, traj-s = {dsdt*1e-3} km/s, mass = {dmdt} kg/s, temp = {dTdt} K/s')

        return ret


    def run(self,
                velocity0, 
                mass0, 
                altitude0,
                zenith_ang,
                azimuth_ang,
                material,
                lat,
                lon,
                alt,
                **kwargs
            ):
        '''This function is based on calc_sput.m which was used to verify the sputtering described in Rogers et al.: Mass loss due to  sputtering and thermal processes in meteoroid ablation, Planetary and Space Science 53 p. 1341-1354 (2005).
        
        :param float/numpy.ndarray velocity0: Meteoroid initial velocity [m/s]
        :param float/numpy.ndarray mass0: Meteoroid initial mass [kg]
        :param float/numpy.ndarray altitude0: Meteoroid initial altitude [m]
        :param float/numpy.ndarray zenith_ang: Zenith angle of the trajectory (-velocity vector) w.r.t reference point [deg]
        :param float/numpy.ndarray azimuth_ang: Azimuthal angle east of north of the trajectory (-velocity vector) w.r.t reference point [deg]
        :param str material: Meteoroid material, see :mod:`~functions.material.material_data`.
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
            * max_t = 60 [s]: Maximum integration time.
            * num = 1000 [s]: Number of steps between :code:`t=0` and :code:`t=max_t` to evaluate solution at.
            * t_eval = None [s]: numpy vector of times to evaluate the model on, overrides the :code:`num` and :code:`max_t` parameters.
            * mass_stop_fraction = 1e-4 [1]: Fraction of the starting mass at witch the integration should break.

        '''

        self._attrs['reference'] = (lat, lon, alt)

        #meteoroid temperature at top of atmosphere
        temperature0 = kwargs.get('temperature0', 290) #[K] 
        
        self._attrs['A'] = kwargs.get('shape_factor', 1.21) #shape factor
        
        #emissivity 0.9 from Hill et al.; Love & Brownlee: 1; 0.2 is characteristic for a metal, oxides are between 0.4 and 0.8
        epsilon = kwargs.get('emissivity', .9) #shape factor

        self.sputtering = kwargs.get('sputtering', True)

        self._attrs['Gamma'] = kwargs.get('Gamma', None)
        self._attrs['Lambda'] = kwargs.get('Lambda', None)

        max_t = kwargs.get('max_t', 60.0)
        num = kwargs.get('num', 1000)
        if 't_eval' in kwargs:
            t_vec = kwargs['t_eval']
        else:
            t_vec = np.linspace(0, max_t, num=num)

        self._allocate(t_vec)

        lat, lon, alt = self._attrs['reference']

        reference_ecef = functions.coordinate.geodetic2ecef(lat, lon, alt)
        v_dir = -1.0*functions.coordinate.azel_to_cart(azimuth_ang, zenith_ang, 1.0)
        v_dir_ecef = functions.coordinate.enu2ecef(lat, lon, alt, *v_dir.tolist())

        def s_to_geo(s):
            traj = reference_ecef - v_dir_ecef*s

            geo = functions.coordinate.ecef2geodetic(*traj.tolist())

            return geo

        self.s_to_geo = s_to_geo

        s0 = scipy.optimize.minimize_scalar(lambda s: np.abs(s_to_geo(s)[2] - altitude0)).x
        
        y0 = np.array([velocity0, mass0, s0, temperature0], dtype=np.float64)

        mass_stop_fraction = kwargs.get('mass_stop_fraction', 1e-4)
        def _zero_mass(t, y):
            print(f'STOPPING @ {t:<6.3f} s: {np.log10(y[1]) - np.log10(mass0*mass_stop_fraction)} log10(kg)')
            return np.log10(y[1]) - np.log10(mass0*mass_stop_fraction)

        _zero_mass.terminal = True
        _zero_mass.direction = -1

        self._attrs['material'] = material
        self.material_data = functions.material.material_parameters(material)

        events = [_zero_mass]

        self.data['t'] = t_vec

        max_step = kwargs.get('max_step', 0.01) #s

        self.integrate(y0, t_vec, events = events, max_step = max_step)

        return self.result



