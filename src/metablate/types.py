"""
This module contains convenient type information so that typing can be precise
but not too verbose in the code itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

import numpy.typing as npt
import xarray as xr

NDArray = npt.NDArray
"Any ndarray"

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

Numeric = float | NDArray
"A real scalar or a NumPy array; operations follow NumPy broadcasting rules."

ArrayLike = npt.ArrayLike
"Input accepted by numpy.asarray, before conversion to an array."

Composition = dict[str, float]
"Relative abundances by atom number, keyed by chemical element symbol."

MaterialDict = dict[str, float | Composition | None]
"Material properties returned by material.get(as_dict=True)."

StateVector = NDArray | Sequence[Numeric]
"An ODE state, or paired arrays of state components for vectorized evaluation."

Kwargs = dict[str, Any]
"Keyword arguments forwarded to a solver or atmospheric backend."

Bounds = Sequence[tuple[float | None, float | None]]
"Lower and upper optimizer bounds; None denotes an unbounded endpoint."

AtmosphereData = xr.Dataset | dict[str, xr.Variable]
"Atmospheric variables with or without dataset coordinates and metadata."

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
    atomic_composition: Composition | None = None
