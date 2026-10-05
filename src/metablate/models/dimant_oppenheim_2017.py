from __future__ import annotations

import ctypes
import multiprocessing as mp

import numpy as np
import scipy.integrate as sci
import scipy.special as scs
from tqdm import tqdm

from ..types import NDArray_M, NDArray_MxN, NDArray_N


def _worker(
    inds, tind, Func3_1, Func3_2, low_lim, up_lim, Rlam23, costh, sinth, pbar, quad_abs_tol
):
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


def _do_integral_1(ksi, rlam23, cth):
    """First integral of equation 5 in Sugar 2018 "Formation of Plasma Around a Small Meteoroid:
    Simulation and Theory"
    """
    return (
        np.sqrt(1 + 2 * rlam23 * ksi ** (2 / 3) / np.pi)
        * np.exp(-3 * rlam23 * ksi ** (2 / 3) / 2)
        * np.sqrt(np.maximum(ksi**2 - cth**2, 0) / (1 - ksi**2))
    )


def _do_integral_2(ksi, rlam23, cth, sth):
    """Second integral of equation 5 in Sugar 2018 "Formation of Plasma Around a Small Meteoroid:
    Simulation and Theory"
    """
    return (
        np.sqrt(1 + 2 * rlam23 * ksi ** (2 / 3) / np.pi)
        * np.exp(-3 * rlam23 * ksi ** (2 / 3) / 2)
        * np.arcsin(np.clip(np.sqrt(1 - ksi**2) * np.abs(cth) / (ksi * np.abs(sth)), 0, 1))
    )


def do_int_1_quad(a, b, rlam23, cth, quad_abs_tol):
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


def do_int_2_quad(a, b, rlam23, cth, sth, quad_abs_tol):
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
) -> tuple[NDArray_MxN, NDArray_MxN, NDArray_MxN, NDArray_MxN]:
    """The plasma distribution morphology component of the Dimant and Oppenheim model, that is
    scaled by mean free path and independent of meteor parameters.

    Note
    ----
        Grid is in units of mean free path

        The default includes the integrable endpoint at xi=1. A positive
        upper_limit_delta truncates that endpoint for legacy calculations.
        The density is undefined at the meteoroid center (R=0).
    """
    # Generate density distribution
    # Solve for a 2D plane
    [Xlam, Ylam] = np.meshgrid(x_grid, y_grid)

    Rlam = np.sqrt(Xlam**2 + Ylam**2)
    Rlam[Rlam == 0] = np.nan

    costh = Xlam / Rlam
    sinth = Ylam / Rlam
    Rlam23 = Rlam ** (2 / 3)

    Func1 = np.sqrt(2 * np.pi / 3) / Rlam * scs.erf(
        np.sqrt(3 / 2) * Rlam ** (1 / 3) * np.abs(costh) ** (1 / 3)
    ) - np.exp(-3 / 2 * Rlam23 * np.abs(costh) ** (2 / 3)) * (
        (4 - np.pi) * np.abs(costh)
        # Dimant & Oppenheim (2017), Eq. 46: only the second term
        # inside the square root is divided by 2*pi.
        / (2 * np.sqrt(1 + (4 - np.pi) ** 2 * Rlam23 * np.abs(costh) ** (2 / 3) / (2 * np.pi)))
        + 2 * np.abs(costh) ** (1 / 3) / Rlam23
    )

    Func2 = np.sqrt(2 * np.pi / 3) / Rlam * scs.erf(np.sqrt(3 * Rlam23 / 2)) - (
        1 + 2 / Rlam23
    ) * np.exp(-3 * Rlam23 / 2)

    orig_shape = costh.shape
    new_shape = (costh.size,)
    costh = costh.reshape(new_shape)
    sinth = sinth.reshape(new_shape)
    Rlam23 = Rlam23.reshape(new_shape)
    low_lim = np.abs(costh)  # Lower limit of integration |cosθ|
    if not 0 <= upper_limit_delta < 1:
        raise ValueError("upper_limit_delta must be in [0, 1)")
    # Never move the lower bound below |cos(theta)|: that makes the
    # integrands undefined near the symmetry axis. On the axis both
    # integrals have zero width and vanish analytically.
    up_lim = np.maximum(np.full_like(costh, 1 - upper_limit_delta), low_lim)

    # TODO: skip problematic points in the calc
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
            if t.exitcode != 0:
                raise RuntimeError(f"Density worker exited with code {t.exitcode}")
        Func3_1 = np.ndarray(costh.shape, dtype=np.float64, buffer=f31.get_obj())
        Func3_2 = np.ndarray(costh.shape, dtype=np.float64, buffer=f32.get_obj())
    Func3_1 = Func3_1.reshape(orig_shape)
    Func3_2 = Func3_2.reshape(orig_shape)

    costh = costh.reshape(orig_shape)
    sinth = sinth.reshape(orig_shape)

    t1 = np.abs(costh) * Func1
    t2 = costh * Func2
    t3 = np.abs(costh) * Func3_2 + Func3_1

    ne_morph = np.abs(t1 + t2 + t3)

    # todo: this is to fix some numerical shenanigans, can probably be solved better
    # zero_point = np.argmin(np.abs(y_grid))
    # if y_grid[zero_point] < 1e-3:
    #     ne_morph[zero_point, :] = (ne_morph[zero_point - 1, :] + ne_morph[zero_point + 1, :]) / 2

    return Xlam, Ylam, Rlam, ne_morph


def plasma_distribution(
    total_atmospheric_number_density,
    meteoroid_velocity,
    meteoroid_radius,
    plasma_source_density,
    collisional_cross_section,
    ionization_probability,
    base_plasma_distribution_function,
    R_lambda,
    ablated_thermal_speed=951,
    atmospheric_species_mass=4.7e-26,
    ablated_species_mass=3.8e-26,
):
    """Calculate meteor head plasma distribution using model from Dimant and
    Oppenheim 2017

    Originally developed by Tarnecki, Liane et al. 20 June 2020
    Code available at https://zenodo.org/records/4723667
    Paper available at https://doi.org/10.1029/2021JA029525
    Ported to Python by Daniel Kastinen 2025
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
