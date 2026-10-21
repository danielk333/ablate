"""
# Meteoroid material physics

[Fe SiO2 C C] mean molecular mass per target atoms corresponding
to the different meteoroid densities, see Rogers p. 1344 values
are from Tielens et al.

Mean molecular mass of ablated vapor assuming that all products
liberated from the meteoroid by thermal ablation were released
in complete molecular form.

"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import asdict
from typing import Literal, overload

from scipy import constants

from .types import Material, MaterialDict

logger = logging.getLogger(__name__)

MATERIALS: dict[str, Material] = {}

# Jones (1997), Table 1: abundances by atom number. As in the original
# ionization model
# TODO: this should probably not be used as default for all materials, but i dont know what to put
# as default for now
_JONES_COMPOSITION = {"O": 0.617, "Fe": 0.059, "Mg": 0.082, "Si": 0.242}

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
    atomic_composition={"Fe": 1.0},
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
    atomic_composition=_JONES_COMPOSITION.copy(),
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
    atomic_composition=_JONES_COMPOSITION.copy(),
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
    atomic_composition=_JONES_COMPOSITION.copy(),
)
MATERIALS["porous"] = porous


def available() -> list[str]:
    """List currently available meteoroid materials."""
    return list(MATERIALS.keys())


@overload
def get(material: str, as_dict: Literal[True] = True) -> MaterialDict: ...


@overload
def get(material: str, as_dict: Literal[False]) -> Material: ...


@overload
def get(material: str, as_dict: bool) -> Material | MaterialDict: ...


def get(material: str, as_dict: bool = True) -> Material | MaterialDict:
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
        Material properties, including the optional atomic composition.
    """

    _material = material.lower().strip()
    if _material not in MATERIALS:
        raise ValueError(
            f'No data exists for material "{_material}"\n' f"Avalible materials are: {available()}"
        )
    if as_dict:
        return asdict(MATERIALS[_material])
    else:
        return deepcopy(MATERIALS[_material])
