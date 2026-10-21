from __future__ import annotations

import ctypes
import multiprocessing as mp
from multiprocessing.sharedctypes import SynchronizedArray

import numpy as np
import scipy.integrate as sci
import scipy.special as scs
from tqdm import tqdm

from ..types import NDArray_M, NDArray_N, Numeric, NDArray, ArrayLike


def _worker(
    inds: NDArray_N,
    tind: int,
    Func3_1: SynchronizedArray,
    Func3_2: SynchronizedArray,
    low_lim: NDArray_N,
    up_lim: NDArray_N,
    Rlam23: NDArray_N,
    costh: NDArray_N,
    sinth: NDArray_N,
    pbar: bool,
    quad_abs_tol: float,
) -> None:
    if pbar:
        tqdm_bar = tqdm(
            desc=f"computing electron density process {tind}",
            total=len(inds),
            position=tind,
        )
    f31 = np.empty(inds.shape, dtype=costh.dtype)
    f32 = np.empty(inds.shape, dtype=costh.dtype)
    for lind, ind in enumerate(inds):
        f31[lind] = do_int_1_quad(low_lim[ind], up_lim[ind], Rlam23[ind], costh[ind], quad_abs_tol)
        f32[lind] = do_int_2_quad(
            low_lim[ind], up_lim[ind], Rlam23[ind], costh[ind], sinth[ind], quad_abs_tol
        )
        if pbar:
            tqdm_bar.update(1)
    with Func3_1.get_lock():
        shared_f31 = np.ndarray(costh.shape, dtype=np.float64, buffer=Func3_1.get_obj())
        shared_f31[inds] = f31
    with Func3_2.get_lock():
        shared_f32 = np.ndarray(costh.shape, dtype=np.float64, buffer=Func3_2.get_obj())
        shared_f32[inds] = f32
    if pbar:
        tqdm_bar.close()


def _do_integral_1(ksi: float, rlam23: float, cth: float) -> float:
    """First integral of equation 5 in Sugar 2018 "Formation of Plasma Around a Small Meteoroid:
    Simulation and Theory"
    """
    return (
        np.sqrt(1 + 2 * rlam23 * ksi ** (2 / 3) / np.pi)
        * np.exp(-3 * rlam23 * ksi ** (2 / 3) / 2)
        * np.sqrt((ksi**2 - cth**2) / (1 - ksi**2))
    )


def _do_integral_2(ksi: float, rlam23: float, cth: float, sth: float) -> float:
    """Second integral of equation 5 in Sugar 2018 "Formation of Plasma Around a Small Meteoroid:
    Simulation and Theory"
    """
    return (
        np.sqrt(1 + 2 * rlam23 * ksi ** (2 / 3) / np.pi)
        * np.exp(-3 * rlam23 * ksi ** (2 / 3) / 2)
        * np.arcsin(np.sqrt(1 - ksi**2) * np.abs(cth) / (ksi * np.abs(sth)))
    )


def do_int_1_quad(a: float, b: float, rlam23: float, cth: float, quad_abs_tol: float) -> float:
    """Numerically evaluate integral 1 in `_do_integral_1`"""
    if not np.isfinite(a):
        return np.nan
    if a == b:
        return 0.0
    int1 = sci.quad(
        _do_integral_1,
        a,
        b,
        args=(rlam23, cth),
        epsrel=0,
        epsabs=quad_abs_tol,
    )
    return int1[0]


def do_int_2_quad(
    a: float, b: float, rlam23: float, cth: float, sth: float, quad_abs_tol: float
) -> float:
    """Numerically evaluate integral 2 in `_do_integral_2`"""
    if not np.isfinite(a):
        return np.nan
    if a == b or cth == 0:
        return 0.0
    int2 = sci.quad(
        _do_integral_2,
        a,
        b,
        args=(rlam23, cth, sth),
        epsrel=0,
        epsabs=quad_abs_tol,
    )
    return int2[0]


def plasma_distribution_morphology_grid(
    x_grid: NDArray_N,
    y_grid: NDArray_M,
    quad_abs_tol: float = 1e-7,
    upper_limit_delta: float = 0.0,
    threads: int | None = None,
    pbar: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """The plasma distribution morphology component of the Dimant and Oppenheim model, that is
    scaled by mean free path and independent of meteor parameters.

    Note
    ----
        Grid is in units of mean free path

        The default includes the integrable endpoint at xi=1. A positive
        upper_limit_delta truncates that endpoint for legacy calculations.
        The density is undefined at the meteoroid center (R=0).
    """
    Xlam, Ylam = np.meshgrid(x_grid, y_grid)
    result = _plasma_distribution_morphology(
        Xlam, Ylam, quad_abs_tol, upper_limit_delta, threads, pbar
    )
    if threads is not None:
        print("\n" * threads)
    return result


def plasma_distribution_morphology_array(
    r: ArrayLike,
    theta: ArrayLike,
    quad_abs_tol: float = 1e-7,
    upper_limit_delta: float = 0.0,
    threads: int | None = None,
    pbar: bool = True,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Evaluate morphology at paired polar coordinates, using NumPy broadcasting.

    Parameters
    ----------
    r : array_like
        Nonnegative radius R/lambda_T. The morphology is undefined at zero.
    theta : array_like
        Angle from the positive x axis, in radians. Must broadcast with ``r``.
        For all radius-angle combinations use ``r[None, :]`` and ``theta[:, None]``.
    quad_abs_tol, upper_limit_delta, threads, pbar
        As in :func:`plasma_distribution_morphology_grid`.

    Returns
    -------
    x, y, R, f : numpy.ndarray
        Coordinates, radius, and morphology with the broadcast input shape.
        Scalar inputs return zero-dimensional arrays. The normalized electron
        density is ``f / R`` (ne/n_star), not ``f`` itself.
    """
    r, theta = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(theta, dtype=float))
    if np.any(~np.isfinite(r)) or np.any(r < 0):
        raise ValueError("r must contain finite, nonnegative radii.")
    if np.any(~np.isfinite(theta)):
        raise ValueError("theta must contain finite angles in radians.")
    Xlam = r * np.cos(theta)
    Ylam = r * np.sin(theta)

    return _plasma_distribution_morphology(
        Xlam, Ylam, quad_abs_tol, upper_limit_delta, threads, pbar
    )


def _plasma_distribution_morphology(
    Xlam: NDArray,
    Ylam: NDArray,
    quad_abs_tol: float,
    upper_limit_delta: float,
    threads: int | None,
    pbar: bool,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Evaluate morphology at Cartesian points without constructing a grid."""
    Rlam = np.sqrt(Xlam**2 + Ylam**2)
    Rlam = np.where(Rlam == 0, np.nan, Rlam)

    costh = Xlam / Rlam
    sinth = Ylam / Rlam
    Rlam23 = Rlam ** (2 / 3)

    Func1 = np.sqrt(2 * np.pi / 3) / Rlam * scs.erf(
        np.sqrt(3 / 2) * Rlam ** (1 / 3) * np.abs(costh) ** (1 / 3)
    ) - np.exp(-3 / 2 * Rlam23 * np.abs(costh) ** (2 / 3)) * (
        (4 - np.pi)
        * np.abs(costh)
        / (2 * np.sqrt(1 + (4 - np.pi) ** 2 * Rlam23 * np.abs(costh) ** (2 / 3) / (2 * np.pi)))
        + 2 * np.abs(costh) ** (1 / 3) / Rlam23
    )

    Func2 = np.sqrt(2 * np.pi / 3) / Rlam * scs.erf(np.sqrt(3 * Rlam23 / 2)) - (1 + 2 / Rlam23) * np.exp(
        -3 * Rlam23 / 2
    )

    orig_shape = costh.shape
    new_shape = (costh.size,)
    costh = costh.reshape(new_shape)
    sinth = sinth.reshape(new_shape)
    Rlam23 = Rlam23.reshape(new_shape)
    low_lim = np.abs(costh)
    # Never move the lower bound below |cos(theta)|: that makes the
    # integrands undefined near the symmetry axis. On the axis both
    # integrals have zero width and vanish analytically.
    up_lim = np.maximum(np.full_like(costh, 1 - upper_limit_delta), low_lim)

    if threads is None:
        if pbar:
            tqdm_bar = tqdm(desc="computing electron density", total=len(costh))
        Func3_1 = np.empty_like(costh)
        Func3_2 = np.empty_like(costh)
        for ind in range(len(costh)):
            Func3_1[ind] = do_int_1_quad(
                low_lim[ind],
                up_lim[ind],
                Rlam23[ind],
                costh[ind],
                quad_abs_tol,
            )
            Func3_2[ind] = do_int_2_quad(
                low_lim[ind],
                up_lim[ind],
                Rlam23[ind],
                costh[ind],
                sinth[ind],
                quad_abs_tol,
            )
            if pbar:
                tqdm_bar.update(1)
        if pbar:
            tqdm_bar.close()
    else:
        th = []
        chunks = np.array_split(np.arange(len(costh)), threads)

        f31 = mp.Array(ctypes.c_double, costh.size)
        f32 = mp.Array(ctypes.c_double, costh.size)
        for tind, chunk in enumerate(chunks):
            t = mp.Process(
                target=_worker,
                args=(
                    chunk,
                    tind,
                    f31,
                    f32,
                    low_lim,
                    up_lim,
                    Rlam23,
                    costh,
                    sinth,
                    pbar,
                    quad_abs_tol,
                ),
            )
            th.append(t)
            t.start()

        for t in th:
            t.join()
        Func3_1 = np.ndarray(costh.shape, dtype=np.float64, buffer=f31.get_obj())
        Func3_2 = np.ndarray(costh.shape, dtype=np.float64, buffer=f32.get_obj())
    Func3_1 = Func3_1.reshape(orig_shape)
    Func3_2 = Func3_2.reshape(orig_shape)

    costh = costh.reshape(orig_shape)
    sinth = sinth.reshape(orig_shape)

    t1 = np.abs(costh) * Func1
    t2 = costh * Func2
    t3 = np.abs(costh) * Func3_2 + Func3_1
    t3 = np.where(np.isnan(t3), 0, t3)

    ne_morph = np.abs(t1 + t2 + t3)

    return np.asarray(Xlam), np.asarray(Ylam), np.asarray(Rlam), np.asarray(ne_morph)


def solve_r(
    theta: float,
    target_density: float,
    r_min: float = 0.001,
    r_max: float = 10.0,
    quad_abs_tol: float = 1e-7,
    xtol: float = 1e-6,
) -> tuple[float, float, float]:
    """Find R/lambda_T for a normalized electron density ne/n_star at fixed theta.

    ``theta`` is a finite scalar angle in radians. ``target_density`` must be
    finite and positive. For a physical electron density, divide by n_star
    before calling this function. This inverts ``f / R``, not the morphology ``f``.

    The finite bounds must satisfy ``0 < r_min < r_max`` and bracket a root.
    No automatic bracket expansion is performed. ``quad_abs_tol`` controls
    morphology quadrature; ``xtol`` is the absolute radius tolerance of Brent's
    method, not a density tolerance.

    Returns ``(radius, density, residual)``, where residual is the evaluated
    normalized density minus the target. Raises ValueError for invalid inputs
    or an unbracketed root, and RuntimeError if the solver does not converge.
    """
    from scipy.optimize import brentq

    if not np.isfinite(theta):
        raise ValueError("theta must be a finite scalar angle in radians.")
    if not np.isfinite(target_density) or target_density <= 0:
        raise ValueError("target_density must be finite and positive (ne/n_star).")

    def objective(radius: float) -> float:
        _, _, R, f = plasma_distribution_morphology_array(
            radius, theta, quad_abs_tol=quad_abs_tol, pbar=False
        )
        residual = float(f / R) - target_density
        if not np.isfinite(residual):
            raise ValueError("The density is not finite within the supplied radius bracket.")
        return residual

    lower, upper = objective(r_min), objective(r_max)
    if lower == 0:
        radius = r_min
    elif upper == 0:
        radius = r_max
    elif np.signbit(lower) == np.signbit(upper):
        raise ValueError("Target density is not bracketed; adjust r_min and r_max.")
    else:
        radius = brentq(objective, r_min, r_max, xtol=xtol)

    residual = objective(radius)
    return float(radius), target_density + residual, residual


def plasma_distribution(
    total_atmospheric_number_density: Numeric,
    meteoroid_velocity: Numeric,
    meteoroid_radius: Numeric,
    plasma_source_density: Numeric,
    collisional_cross_section: Numeric,
    ionization_probability: Numeric,
    base_plasma_distribution_function: Numeric,
    R_lambda: Numeric,
    ablated_thermal_speed: Numeric = 951,
    atmospheric_species_mass: Numeric = 4.7e-26,
    ablated_species_mass: Numeric = 3.8e-26,
) -> Numeric:
    """Calculate meteor head plasma distribution using model from Dimant and
    Oppenheim 2017

    Originally developed by Tarnecki, Liane et al. 20 June 2020
    Code available at https://zenodo.org/records/4723667
    Paper available at https://doi.org/10.1029/2021JA029525
    Ported to Python by Daniel Kastinen 2025

    Note
    ----
        There was a bug in line 79 in ``YD_to_q.m`` that is fixed in this implementation
    """

    # mean free path
    lamT = ablated_thermal_speed / (
        total_atmospheric_number_density * collisional_cross_section * meteoroid_velocity
    )

    # Calculate the ionizing cross section Gion(meteoroid_velocity)
    Gcoll = collisional_cross_section / (4 * np.pi)  # net differential cross section
    Gion = Gcoll * ionization_probability

    coeff_const = (
        8
        * np.pi
        * meteoroid_radius**2
        * plasma_source_density
        * total_atmospheric_number_density
        / np.sqrt(3)
        * (1 + ablated_species_mass / atmospheric_species_mass)
        * Gion
    )  # Density coefficient

    coeff = coeff_const / (R_lambda * lamT)

    ne2D = coeff * base_plasma_distribution_function
    return ne2D
