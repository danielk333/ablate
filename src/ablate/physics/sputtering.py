"""
# Sputtering physics
"""

import copy
import logging

import numpy as np
from scipy import constants


logger = logging.getLogger(__name__)


def sputtering_kero_szasz_2008(mass, velocity, material_data, density):
    """Calculates the meteoroid mass loss due to sputtering [^1][^2].

    All units need to be given in the cgs system to remain 
    consistent with the units in which the sputtering yield
    equation was derived i.e., variables given in SI system
    are converted into cgs in this function when needed.

    Changes pre-python port: See [matlab-version](https://gitlab.irf.se/kero/ablation_matlab)

    [^1]: Rogers et al.: Mass loss due to sputtering and 
        thermal processes in meteoroid ablation. 
        Planetary and Space Science 53 p. 1341-1354 (2005)
    [^2]: Tielens et al.: The physics of grain-grain 
        collisions and gas-grain sputtering in interstellar shocks.
        The Astrophisicsl Journal 431, p. 321-340 (1994)


    Parameters
    ----------
    mass : float/numpy.ndarray
        Meteoroid mass in [kg]
    velocity : float/numpy.ndarray
        Meteoroid velocity in [m/s]
    material_data : dict
        Meteoroid material data, see [`material_data`][ablate.functions.material.material_data].
    density : xarray.Dataset
        Dataset of atmospheric constituent densities. Variables should be 
        constituents and coordinates, and then indexed on the mass(es) and
        velocity(ies) of the input meteoroid(s).
        See [`Atmosphere`][ablate.atmosphere.atmosphere.Atmosphere].

    Returns
    -------
    float or numpy.ndarray
        Mass loss due to sputtering for each input meteoroid

    Notes
    -----
    Variables explanation:

     - `E_th`: threshold energy: the minimum projectile kinetic energy needed for
         a given projectile and target to induce sputtering
     - `a`: screening length
     - `beta`: maximum fractional energy transfer in head-on elastic collision
     - `alpha`: dimensionless function of the mass ratio
     - `Rp`: mean projected range
     - `R`: mean penetrated path
     - `Y`: sputtering yield at normal incidence: the ratio of the mean number 
        of sputtered particles per projectile
     - `U0`: surface binding energy (eV), use the sublimation energy of the material
     - `M1`: projectile mass
     - `M2`: mean molecular mass per atom of the target
     - `Z1`: projectile atomic number
     - `Z2`: target atomic number
     - `sn`: universal function
     - `E`: incident projectile energy
     - `Gamma`: reduced energy
     - `M2`: mean molecular mass per atom of the target
     - `A`: shape factor, sphere = 1.21
     - `v`: meteoroid velocity
     - `m`: meteoroid mass
     - `rho_m`: meteoroid density

    """
    v = copy.copy(velocity)
    m = copy.copy(mass)

    shape = None
    if isinstance(v, np.ndarray):
        if isinstance(m, np.ndarray):
            assert v.shape == m.shape
        shape = v.shape
    elif isinstance(m, np.ndarray):
        shape = m.shape

    # Universal constants
    u = constants.u
    elem = constants.value("elementary charge")
    a0 = constants.value("Bohr radius") * 1e2  # Bohr radius in cm (cgs system)

    # Atmospheric molecules
    # M1 = projectile mass in kg [O N2 O2 He Ar H N], for molecules: 
    #  average of its constitutents (see table 3 on p. 333 in Tielens et al)
    # Z for a molecule: average of its consitutents 
    # (see table 3 on p. 333 in Tielens et al, not given in the table, but I use the same method)
    avalible_species = ["O", "N2", "O2", "He", "Ar", "H", "N"]
    m1 = np.array([15.9994, 14.007, 15.999, 4.0026, 39.948, 1.0079, 14.007]) * u
    z1 = np.array([8, 7, 8, 2, 18, 1, 7])
    use_species = np.full(m1.shape, False, dtype=bool)

    if shape is not None:
        # This needs to happen first
        v = np.outer(np.ones(m1.shape), v)
        m = np.outer(np.ones(m1.shape), m)
        # then this
        m1 = np.outer(m1, np.ones(shape))
        z1 = np.outer(z1, np.ones(shape))

    _density = np.empty(m1.shape, dtype=np.float64)
    for ind, key in enumerate(avalible_species):
        if key in density:
            use_species[ind] = True
            _density[ind, ...] = density[key].values.squeeze()

    m1 = m1[use_species, ...]
    z1 = z1[use_species, ...]
    _density = _density[use_species, ...]

    if shape is not None:
        m = m[use_species, ...]
        v = v[use_species, ...]

    m2 = material_data["m2"]
    u0 = material_data["u0"]
    k = material_data["k"]
    z2 = material_data["z2"]
    rho_m = material_data["rho_m"]

    beta = 4 * m1 * m2 / (m1 + m2) ** 2

    m1_m2 = m1 / m2
    E_th = 8 * u0 * m1_m2 ** (1.0 / 3.0)

    # when M1 ./ M2 > 0.3; E_th is in eV
    less03 = m1_m2 <= 0.3
    E_th[less03] = u0 / (beta[less03] * (1 - beta[less03]))

    alpha = 0.3 * (m2 / m1) ** (2.0 / 3.0)

    less05 = (m2 / m1) < 0.5
    alpha[less05] = 0.2

    # balances 'alpha' if m1/m2 grows too big;
    # 'alpha' should be between 1/2 and 2/3 see page 324 in Tielens
    Rp_R = (k * m2 / m1 + 1.0) ** (-1.0)
    a = 0.885 * a0 / np.sqrt(z1 ** (2.0 / 3.0) + z2 ** (2.0 / 3.0))

    # ---------------------------------
    # Stepping through the atmosphere --
    # ---------------------------------

    # ---------------------------------------
    # Sputtering yield at normal incidence: --
    # ---------------------------------------
    E = (m1 * v**2.0 / 2.0) / elem  # joule -> eV by dividing by 'elem'
    # Sputtering only occurs if E > E_th
    yes = E > E_th

    Y = np.zeros(yes.shape, dtype=np.float64)
    Gamma = np.zeros(yes.shape, dtype=np.float64)

    m1 = m1 * 1e3  # [g] projectile mean molecular mass
    m2 = m2 * 1e3  # [g] target mean molecular mass
    m = m * 1e3  # [g] meteoroid mass
    rho_m = rho_m / 1e3  # [g/cm3] meteoroid density

    # Calculating the yield for all atmospheric constituents for the given 
    # surface material (meteoroid density)
    # taking the energy in ergs so the unit of sn is ergs cm2; elementary charge in cgs: in esu
    Gamma[yes] = (
        m2
        / (m1[yes] + m2)
        * a[yes]
        / (z1[yes] * z2 * (elem / 3.33564e-10) ** 2.0)
        * E[yes]
        * elem
        * 1e7
    )

    sn_1 = 3.441 * np.sqrt(Gamma) * np.log(Gamma + 2.781)
    sn_2 = 1.0 + 6.35 * np.sqrt(Gamma) + Gamma * (-1.708 + 6.882 * np.sqrt(Gamma))
    sn = sn_1 / sn_2  # [ergs cm2]

    # Valid for E > E_th:
    Y[yes] = (
        3.56
        / u0
        * m1[yes]
        / (m1[yes] + m2)
        * z1[yes]
        * z2
        / np.sqrt(z1[yes] ** (2.0 / 3.0) + z2 ** (2.0 / 3.0))
    )
    Y[yes] *= (
        alpha[yes]
        * Rp_R[yes]
        * sn[yes]
        * (1.0 - (E_th[yes] / E[yes]) ** (2.0 / 3.0))
        * (1.0 - E_th[yes] / E[yes]) ** 2.0
    )
    # we have tried to put in E_th in eV and E in cgs-units (ergs) but the results were convincingly wrong...

    # the total yield is the sum of all individual yields x the atmospheric density
    Y_tot = np.sum(_density * 1e-6 * Y, axis=0)  # density in cm^-3?

    # ------------------------------
    # Mass loss due to sputtering: --
    # ------------------------------
    A = 1.21  # Sphere

    dmdt = -2.0 * m2 * A * velocity * 1e2 * (mass / rho_m) ** (2.0 / 3.0) * Y_tot  # [g/s]
    dmdt = dmdt / 1e3  # [kg/s]

    return dmdt
