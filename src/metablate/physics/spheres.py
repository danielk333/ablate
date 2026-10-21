import numpy as np

from ..types import Numeric


def radius_from_mass(mass: Numeric, density: Numeric) -> Numeric:
    """Mass calculated from an mass-to-area ratio assuming a sphere"""
    return np.cbrt(3 * mass / (4 * np.pi * density))


def mass_from_mass_to_area_ratio(mass_to_area: Numeric, density: Numeric) -> Numeric:
    """Mass calculated from an mass-to-area ratio assuming a sphere"""
    return 9 * np.pi * mass_to_area**3 / (16 * density**2)
