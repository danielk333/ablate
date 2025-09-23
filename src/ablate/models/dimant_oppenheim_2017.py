import numpy as np
import scipy.special as scs
import scipy.integrate as sci
from tqdm import tqdm


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
    grid_size=(301, 301),
    grid_step=(0.05, 0.05),
    ablated_thermal_speed=951,
    atmospheric_species_mass=4.7e-26,
    ablated_species_mass=3.8e-26,
    quad_abs_tol=1e-8,
):
    """Calculate meteor head plasma distribution using model from Dimant and
    Oppenheim 2017

    Originally developed by Tarnecki, Liane et al. 20 June 2020
    Code available at https://zenodo.org/records/4723667
    Paper available at https://doi.org/10.1029/2021JA029525
    Ported to Python by Daniel Kastinen 2025

    Inputs:
           in: structure containing DO parameters
                meteoroid_velocity: meteor velocity, m/s
                altitude: meteor ablation altitude, m
                rM: meteoroid radius, m
                n0: surface density, m**-3

    Outputs:
           q: line density, # e-/m

    ablated_thermal_speed = 951  #thermal speed of ablated particles, m/s

    """

    assert len(grid_size) == len(grid_step), "grid size and step must equal"
    if len(grid_size) == 3:
        Nx, Ny, Nz = grid_size
        dx, dy, dz = grid_step
    elif len(grid_size) == 2:
        Nx, Ny = grid_size
        dx, dy = grid_step
    else:
        raise NotImplementedError("unrecogized grid size")

    # todo: remove singularity at middle?
    x = -(Nx - 1) / 2 * dx + dx * np.arange(Nx - 1)
    y = -(Ny - 1) / 2 * dy + dy * np.arange(Ny - 1)

    # mean free path
    lamT = ablated_thermal_speed / (
        total_atmospheric_number_density * collisional_cross_section * meteoroid_velocity
    )

    # Calculate the ionizing cross section Gion(meteoroid_velocity)
    Gcoll = collisional_cross_section / (
        4 * np.pi
    )  # G(meteoroid_velocity), net differential cross section
    Gion = Gcoll * ionization_probability  # Gion(meteoroid_velocity)

    # Set meteoroid and atmospheric characteristics
    # atmospheric_species_mass = 4.7e-26  #atmospheric species mass (N2), kg
    # ablated_species_mass = 3.8e-26  #ablated species mass (Na), kg

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
    [X2, Y2] = np.meshgrid(x, y)

    R = np.sqrt(X2**2 + Y2**2)
    theta = np.atan2(Y2, X2)

    # Define some common factors: cos(θ), sin(θ), R/λΤ and (R/λT)**(2/3)
    costh = np.cos(theta)
    sinth = np.sin(theta)
    Rlam = R / lamT
    Rlam23 = Rlam ** (2 / 3)

    coeff = coeff_const / R

    Func1 = np.sqrt(2 * np.pi / 3) / Rlam * scs.erf(
        np.sqrt(3 / 2) * Rlam ** (1 / 3) * (np.abs(costh)) ** (1 / 3)
    ) - np.exp(-3 / 2 * Rlam23 * (np.abs(costh)) ** (2 / 3)) * (
        (4 - np.pi)
        * np.abs(costh)
        / (2 * np.sqrt((1 + (4 - np.pi) ** 2 * Rlam23 * (np.abs(costh)) ** (2 / 3)) / (2 * np.pi)))
        + 2 * (np.abs(costh)) ** (1 / 3) / Rlam23
    )

    Func2 = np.sqrt(2 * np.pi / 3) / Rlam * scs.erf(np.sqrt(3 * Rlam23 / 2)) - (
        1 + 2 / Rlam23
    ) * np.exp(-3 * Rlam23 / 2)

    alpha = 1e-6

    # ksi == ξ in Dimant and Oppenheim's notation
    def _do_integral_1(ksi, rlam, cth):
        return (
            np.sqrt(1 + 2 * rlam * ksi ** (2 / 3) / np.pi)
            * np.exp(-3 * rlam * ksi ** (2 / 3) / 2)
            * np.sqrt((ksi**2 - cth**2) / (1 - ksi**2))
        )

    def _do_integral_2(ksi, rlam, cth, sth):
        return (
            np.sqrt(1 + 2 * rlam * ksi ** (2 / 3) / np.pi)
            * np.exp(-3 * rlam * ksi ** (2 / 3) / 2)
            * np.arcsin(np.sqrt(1 - ksi**2) * np.abs(cth) / (ksi * np.abs(sth)))
        )

    def do_int_1_quad(a, b, rlam, cth):
        return sci.quad(_do_integral_1, a, b, args=(rlam, cth), epsrel=0, epsabs=quad_abs_tol)[0]

    def do_int_2_quad(a, b, rlam, cth, sth):
        return sci.quad(_do_integral_2, a, b, args=(rlam, cth, sth), epsrel=0, epsabs=quad_abs_tol)[0]

    orig_shape = costh.shape
    costh = costh.flatten()
    sinth = sinth.flatten()
    Rlam23 = Rlam23.flatten()
    low_lim = np.abs(costh)  # Lower limit of integration |cosθ|
    up_lim = np.full_like(
        costh, 1 - alpha
    )  # Upper limit of integration 1 (avoid singularity at exactly 1)

    Func3_1 = np.empty_like(costh)
    Func3_2 = np.empty_like(costh)
    for ind in tqdm(range(len(costh)), total=len(costh)):
        Func3_1[ind] = do_int_1_quad(low_lim[ind], up_lim[ind], Rlam23[ind], costh[ind])
        Func3_2[ind] = do_int_2_quad(low_lim[ind], up_lim[ind], Rlam23[ind], costh[ind], sinth[ind])

    # do_int_1_quad_vec = np.vectorize(do_int_1_quad)
    # do_int_2_quad_vec = np.vectorize(do_int_2_quad)
    # Func3_1 = do_int_1_quad_vec(low_lim, up_lim, Rlam23, costh)
    Func3_1 = Func3_1.reshape(orig_shape)

    # Func3_2 = do_int_2_quad_vec(low_lim, up_lim, Rlam23, costh, sinth)
    Func3_2 = Func3_2.reshape(orig_shape)

    costh = costh.reshape(orig_shape)
    sinth = sinth.reshape(orig_shape)

    t1 = coeff * np.abs(costh) * Func1
    t2 = coeff * costh * Func2
    t3 = coeff * (np.abs(costh) * Func3_2 + Func3_1)
    t3[np.isnan(t3)] = 0
    ne2D = np.abs(t1 + t2 + t3)
    # ne2D[:, (Ny + 1) / 2] = (ne2D[:, (Ny - 1) / 2 - 1] + ne2D[:, (Ny + 3) / 2 - 1]) / 2

    return ne2D
