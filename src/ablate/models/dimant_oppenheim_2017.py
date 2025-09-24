import ctypes
import numpy as np
import scipy.special as scs
import scipy.integrate as sci
import scipy.constants as consts
from tqdm import tqdm
import multiprocessing as mp


def plasma_frequency(electron_density):
    return np.sqrt(electron_density * consts.e**2 / (consts.epsilon_0 * consts.m_e)) / (2 * np.pi)


def mean_free_path(
    ablated_thermal_speed,
    total_atmospheric_number_density,
    collisional_cross_section,
    meteoroid_velocity,
):
    return ablated_thermal_speed / (
        total_atmospheric_number_density * collisional_cross_section * meteoroid_velocity
    )


def collisional_cross_section_bronshten_1983(relative_velocity):
    vrel = relative_velocity * 1e-3  # requires km/s
    return 5.61e-19 * vrel ** (-0.8)


def ionization_probability_Na_vondrak_2008(relative_velocity):
    vrel = relative_velocity * 1e-3  # requires km/s
    return 0.933 * (vrel - 8.86) ** 2 * vrel ** (-1.94)


def plasma_distribution(
    total_atmospheric_number_density,
    meteoroid_velocity,
    meteoroid_radius,
    plasma_source_density,
    collisional_cross_section,
    ionization_probability,
    x_grid,
    y_grid,
    ablated_thermal_speed=951,
    atmospheric_species_mass=4.7e-26,
    ablated_species_mass=3.8e-26,
    quad_abs_tol=1e-7,
    upper_limit_delta=1e-6,
    threads=None,
    pbar=True,
):
    """Calculate meteor head plasma distribution using model from Dimant and
    Oppenheim 2017

    Originally developed by Tarnecki, Liane et al. 20 June 2020
    Code available at https://zenodo.org/records/4723667
    Paper available at https://doi.org/10.1029/2021JA029525
    Ported to Python by Daniel Kastinen 2025

    ablated_thermal_speed = 951  #thermal speed of ablated particles, m/s
    # Set meteoroid and atmospheric characteristics
    # atmospheric_species_mass = 4.7e-26  #atmospheric species mass (N2), kg
    # ablated_species_mass = 3.8e-26  #ablated species mass (Na), kg
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

    # Generate density distribution
    # Solve for a 2D plane
    [X2, Y2] = np.meshgrid(x_grid, y_grid)

    R = np.sqrt(X2**2 + Y2**2)
    R[R == 0] = np.nan

    costh = X2 / R
    # costh[costh == 0] == np.nan
    sinth = Y2 / R
    Rlam = R / lamT
    Rlam23 = Rlam ** (2 / 3)

    coeff = coeff_const / R

    Func1 = np.sqrt(2 * np.pi / 3) / Rlam * scs.erf(
        np.sqrt(3 / 2) * Rlam ** (1 / 3) * np.abs(costh) ** (1 / 3)
    ) - np.exp(-3 / 2 * Rlam23 * np.abs(costh) ** (2 / 3)) * (
        (4 - np.pi)
        * np.abs(costh)
        / (2 * np.sqrt((1 + (4 - np.pi) ** 2 * Rlam23 * np.abs(costh) ** (2 / 3)) / (2 * np.pi)))
        + 2 * np.abs(costh) ** (1 / 3) / Rlam23
    )

    Func2 = np.sqrt(2 * np.pi / 3) / Rlam * scs.erf(np.sqrt(3 * Rlam23 / 2)) - (
        1 + 2 / Rlam23
    ) * np.exp(-3 * Rlam23 / 2)

    def _do_integral_1(ksi, rlam23, cth):
        return (
            np.sqrt(1 + 2 * rlam23 * ksi ** (2 / 3) / np.pi)
            * np.exp(-3 * rlam23 * ksi ** (2 / 3) / 2)
            * np.sqrt((ksi**2 - cth**2) / (1 - ksi**2))
        )

    def _do_integral_2(ksi, rlam23, cth, sth):
        return (
            np.sqrt(1 + 2 * rlam23 * ksi ** (2 / 3) / np.pi)
            * np.exp(-3 * rlam23 * ksi ** (2 / 3) / 2)
            * np.arcsin(np.sqrt(1 - ksi**2) * np.abs(cth) / (ksi * np.abs(sth)))
        )

    def do_int_1_quad(a, b, rlam23, cth):
        int1 = sci.quad(
            _do_integral_1,
            a,
            b,
            args=(rlam23, cth),
            epsrel=0,
            epsabs=quad_abs_tol,
        )
        return int1[0]

    def do_int_2_quad(a, b, rlam23, cth, sth):
        int2 = sci.quad(
            _do_integral_2,
            a,
            b,
            args=(rlam23, cth, sth),
            epsrel=0,
            epsabs=quad_abs_tol,
        )
        return int2[0]

    orig_shape = costh.shape
    new_shape = (costh.size,)
    costh = costh.reshape(new_shape)
    sinth = sinth.reshape(new_shape)
    Rlam23 = Rlam23.reshape(new_shape)
    low_lim = np.abs(costh)  # Lower limit of integration |cosθ|
    up_lim = np.full_like(
        costh, 1 - upper_limit_delta
    )  # Upper limit of integration 1 (avoid singularity at exactly 1)

    # todo: hack to make sure integration limits are sane
    low_lim[low_lim > up_lim] = up_lim[low_lim > up_lim] - upper_limit_delta

    # TODO: skip problematic points in the calc

    if threads is None:
        if pbar:
            tqdm_bar = tqdm(desc="computing electron density", total=len(costh))
        Func3_1 = np.empty_like(costh)
        Func3_2 = np.empty_like(costh)
        for ind in range(len(costh)):
            Func3_1[ind] = do_int_1_quad(low_lim[ind], up_lim[ind], Rlam23[ind], costh[ind])
            Func3_2[ind] = do_int_2_quad(
                low_lim[ind], up_lim[ind], Rlam23[ind], costh[ind], sinth[ind]
            )
            if pbar:
                tqdm_bar.update(1)
        if pbar:
            tqdm_bar.close()
    else:
        th = []
        chunks = np.array_split(np.arange(len(costh)), threads)

        def worker(inds, tind, Func3_1, Func3_2, low_lim, up_lim, Rlam23, costh, sinth, pbar):
            if pbar:
                tqdm_bar = tqdm(
                    desc=f"computing electron density process {tind}",
                    total=len(inds),
                    position=tind,
                )
            f31 = np.empty(inds.shape, dtype=costh.dtype)
            f32 = np.empty(inds.shape, dtype=costh.dtype)
            for lind, ind in enumerate(inds):
                f31[lind] = do_int_1_quad(low_lim[ind], up_lim[ind], Rlam23[ind], costh[ind])
                f32[lind] = do_int_2_quad(
                    low_lim[ind], up_lim[ind], Rlam23[ind], costh[ind], sinth[ind]
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

        f31 = mp.Array(ctypes.c_double, costh.size)
        f32 = mp.Array(ctypes.c_double, costh.size)
        for tind, chunk in enumerate(chunks):
            t = mp.Process(
                target=worker,
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
                ),
            )
            th.append(t)
            t.start()

        for t in th:
            t.join()
        print("\n" * threads)
        Func3_1 = np.ndarray(costh.shape, dtype=np.float64, buffer=f31.get_obj())
        Func3_2 = np.ndarray(costh.shape, dtype=np.float64, buffer=f32.get_obj())
    Func3_1 = Func3_1.reshape(orig_shape)
    Func3_2 = Func3_2.reshape(orig_shape)

    costh = costh.reshape(orig_shape)
    sinth = sinth.reshape(orig_shape)

    t1 = np.abs(costh) * Func1
    t2 = costh * Func2
    t3 = np.abs(costh) * Func3_2 + Func3_1
    t3[np.isnan(t3)] = 0
    ne2D = coeff * np.abs(t1 + t2 + t3)
    # breakpoint()

    # some magic to prevent numerical artifact? todo figure out why
    # zero_point = np.argmin(np.abs(y_grid))
    # if y_grid[zero_point] < 1e-3:
    #     ne2D[:, zero_point] = (ne2D[:, zero_point - 1] + ne2D[:, zero_point + 1]) / 2

    return X2, Y2, ne2D
