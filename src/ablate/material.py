'''
# Meteoroid material physics

List of properties:
 - `m2`: mean atomic mass [1]
 - `u0`: surface binding energy [eV]
 - `k`: average atomic number [1]
 - `CA`, CB: Clausius-Clapeyron coeff [K]
 - `mu`: mean molecular mass of ablated vapour [kg]
 - `k`: Botzmann constant [J/K]
 - `rho_m`: meteoroid density [kg/m3]
 - `c`: specific heat of meteoroid [J/(K*kg)]
 - `L`: latent heat of fusion + vapourization [J/kg]

# TODO: list references for material data

[Fe SiO2 C C] mean molecular mass per target atoms corresponding
to the different meteoroid densities, see Rogers p. 1344 values
are from Tielens et al.

Mean molecular mass of ablated vapor assuming that all products
liberated from the meteoroid by thermal ablation were released
in complete molecular form.

'''
from dataclasses import dataclass, asdict
import logging
from copy import copy
from scipy import constants

logger = logging.getLogger(__name__)


@dataclass
class Material:
    """Container for material properties

    TODO: proper names for all vars

List of properties:
 - `m2`: mean atomic mass [1]
 - `u0`: surface binding energy [eV]
 - `k`: average atomic number [1]
 - `CA`, CB: Clausius-Clapeyron coeff [K]
 - `mu`: mean molecular mass of ablated vapour [kg]
 - `k`: Botzmann constant [J/K]
 - `rho_m`: meteoroid density [kg/m3]
 - `c`: specific heat of meteoroid [J/(K*kg)]
 - `L`: latent heat of fusion + vapourization [J/kg]
    """
    rho_m: float
    mu: float
    m2: float
    CA: float
    CB: float
    u0: float
    k: float
    z2: float
    c: float
    L: float


MATERIALS = {}

iron = Material(
    rho_m = 7800.0,
    mu = 56.0*constants.u,
    m2 = 56.0*constants.u,
    CA = 10.6,
    CB = 16120.,
    u0 = 4.1,
    k = 0.35,
    z2 = 26,
    c = 1200,
    L = 6.0E6,
)
MATERIALS["iron"] = iron

asteroidal = Material(
    rho_m = 3300.0,
    mu = 50.0*constants.u,
    m2 = 20.0*constants.u,
    CA = 10.6,
    CB = 13500.,
    u0 = 6.4,
    k = 0.1,
    z2 = 10,
    c = 1200,
    L = 6.0E6,
)
MATERIALS["asteroidal"] = asteroidal

cometary = Material(
    rho_m = 1000.0,
    mu = 20.0*constants.u,
    m2 = 12.0*constants.u,
    CA = 10.6,
    CB = 13500.,
    u0 = 4.0,
    k = 0.65,
    z2 = 6,
    c = 1200,
    L = 6.0E6,
)
MATERIALS["cometary"] = cometary

porous = Material(
    rho_m = 300.0,
    mu = 20.0*constants.u,
    m2 = 12.0*constants.u,
    CA = 10.6,
    CB = 13500.,
    u0 = 4.0,
    k = 0.65,
    z2 = 6,
    c = 1200,
    L = 6.0E6,
)
MATERIALS["porous"] = porous


def available():
    '''List currently available meteoroid materials.
    '''
    return list(MATERIALS.keys())


def get(material, as_dict=True):
    '''Returns the physical parameters of the meteoroid based on its material.

    # TODO: figure out if we keep as dict or keep as material???

    Parameters
    ----------
    material : str
        name of material
    as_dict : bool
        Return data as dict instead of as a Material instance

    Returns
    -------
    Material or dict
        Dictionary of all available material parameters (float or None)
    '''

    _material = material.lower().strip()
    if _material not in MATERIALS:
        raise ValueError(
            f'No data exists for material "{_material}"\n'
            f'Avalible materials are: {available()}'
        )
    if as_dict:
        return asdict(MATERIALS[_material])
    else:
        return copy(MATERIALS[_material])
