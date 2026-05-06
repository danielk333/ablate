"""
# Meteoroid material physics

[Fe SiO2 C C] mean molecular mass per target atoms corresponding
to the different meteoroid densities, see Rogers p. 1344 values
are from Tielens et al.

Mean molecular mass of ablated vapor assuming that all products
liberated from the meteoroid by thermal ablation were released
in complete molecular form.

"""

from dataclasses import asdict
import logging
from copy import copy
from scipy import constants
from .types import Material

logger = logging.getLogger(__name__)

MATERIALS = {}

iron = Material(
    bulk_density=7800.0,
    mean_atomic_mass=56.0 * constants.u,
    latent_heat_of_fusion_vapourization=6.0e6,
    specific_heat=1200,
    average_atomic_number=26,
    surface_binding_energy=4.1,
    ablated_vapour_mean_molecular_mass=56.0 * constants.u,
    penetration_correction_parameter_K=0.35,
    clausius_clapeyron_coeff_A=10.6,
    clausius_clapeyron_coeff_B=16120.0,
)
MATERIALS["iron"] = iron

asteroidal = Material(
    bulk_density=3300.0,
    mean_atomic_mass=20.0 * constants.u,
    latent_heat_of_fusion_vapourization=6.0e6,
    specific_heat=1200,
    average_atomic_number=10,
    surface_binding_energy=6.4,
    ablated_vapour_mean_molecular_mass=50.0 * constants.u,
    penetration_correction_parameter_K=0.1,
    clausius_clapeyron_coeff_A=10.6,
    clausius_clapeyron_coeff_B=13500.0,
)
MATERIALS["asteroidal"] = asteroidal

cometary = Material(
    bulk_density=1000.0,
    mean_atomic_mass=12.0 * constants.u,
    latent_heat_of_fusion_vapourization=6.0e6,
    specific_heat=1200,
    average_atomic_number=6,
    surface_binding_energy=4.0,
    ablated_vapour_mean_molecular_mass=20.0 * constants.u,
    penetration_correction_parameter_K=0.65,
    clausius_clapeyron_coeff_A=10.6,
    clausius_clapeyron_coeff_B=13500.0,
)
MATERIALS["cometary"] = cometary

porous = Material(
    bulk_density=300.0,
    mean_atomic_mass=12.0 * constants.u,
    latent_heat_of_fusion_vapourization=6.0e6,
    specific_heat=1200,
    average_atomic_number=6,
    surface_binding_energy=4.0,
    ablated_vapour_mean_molecular_mass=20.0 * constants.u,
    penetration_correction_parameter_K=0.65,
    clausius_clapeyron_coeff_A=10.6,
    clausius_clapeyron_coeff_B=13500.0,
)
MATERIALS["porous"] = porous


def available():
    """List currently available meteoroid materials."""
    return list(MATERIALS.keys())


def get(material, as_dict=True):
    """Returns the physical parameters of the meteoroid based on its material.

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
    """

    _material = material.lower().strip()
    if _material not in MATERIALS:
        raise ValueError(
            f'No data exists for material "{_material}"\n' f"Avalible materials are: {available()}"
        )
    if as_dict:
        return asdict(MATERIALS[_material])
    else:
        return copy(MATERIALS[_material])
