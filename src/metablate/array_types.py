"""
This module contains convenient type information so that typing can be precise
but not too verbose in the code itself.
"""

import numpy.typing as npt

NDArray_N = npt.NDArray
"(n,) shaped ndarray"

NDArray_M = npt.NDArray
"(n,) shaped ndarray"

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


