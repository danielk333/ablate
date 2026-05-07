"""
This module contains convenient type information so that typing can be precise
but not too verbose in the code itself.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import TypeVar
import numpy.typing as npt

NDArray_N = npt.NDArray
"(n,) shaped ndarray"

NDArray_M = npt.NDArray
"(m,) shaped ndarray"

NDArray_2 = npt.NDArray
"(2,) shaped ndarray"

NDArray_3 = npt.NDArray
"(3,) shaped ndarray"

NDArray_2xN = npt.NDArray
"(2,n) shaped ndarray"

NDArray_Mx2 = npt.NDArray
"(m,2) shaped ndarray"

NDArray_3xN = npt.NDArray
"(3,n) shaped ndarray"

NDArray_3xNxM = npt.NDArray
"(3,n,m) shaped ndarray"

NDArray_Mx2xN = npt.NDArray
"(m,2,n) shaped ndarray"

NDArray_MxM = npt.NDArray
"(m,m) shaped ndarray"

NDArray_MxN = npt.NDArray
"(m,n) shaped ndarray"

P = TypeVar("P")
R = TypeVar("R")
Op = TypeVar("Op", bound="Options")

@dataclass
class Options:
    pass

@dataclass
class Results:
    runtime: float | None

@dataclass
class Material:
    """Container for material properties"""

    bulk_density: float  # [kg/m3]
    mean_atomic_mass: float  # [kg]
    latent_heat_of_fusion_vapourization: float  # [J/kg]
    specific_heat: float  # [J/(K*kg)]
    average_atomic_number: float  # [1]
    surface_binding_energy: float  # [eV]
    ablated_vapour_mean_molecular_mass: float  # [kg]
    penetration_correction_parameter_K: float  # [1]
    clausius_clapeyron_coeff_A: float  # [K]
    clausius_clapeyron_coeff_B: float  # [K]
