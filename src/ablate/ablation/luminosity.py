import numpy as np


def hill_2005_luminosity(velocity, thermal_ablation):
    """Luminosity during thermal ablation.

    Luminosity occurs in meteors as a result of decay of excited atomic (and
    a few molecular) states following collisions between ablated meteor atoms
    and atmospheric constituents. This specific model is described in [^1] and [^2].

    Parameters
    ----------
    velocity : float or numpy.ndarray
        Meteoroid velocity [m/s]
    thermal_ablation : float or numpy.ndarray
        Mass loss due to thermal ablation [kg/s]

    Returns
    -------
    float or numpy.ndarray
        Luminosity [W]

    Notes
    -----
    Symbol definitions:

     - `I` = [W] light intensity of a meteoroid, i.e. luminous intensity; radiant intensity
     - `tau_I` = luminous efficiency factor
     - `mu` = [kg] mean molecular mass of ablated material
     - `v` = [m/s] meteoroid velocity
     - `epsilon` = emissivity
     - `zeta` = excitation coeff
     - `rho_m` = [kg/m3] meteoroid density
     - `abla` = mass loss due to thermal ablation


    [^1]: Friichtenicht and Becker: Determination of meteor paramters using laboratory
        simulations techniques, 'Evolutionary and physical properties of meteoroids',
        National Astronautics and Space Administration, Chapter 6, p. 53-81 (1973)

    [^2]: Hill et al.: High geocentric velocity meteor ablation, A&A 444 p. 615-624 (2005)

    """

    # -- Universal constants

    # -- Variables
    v = velocity  # [m/s] velocity
    vkm = v / 1e3  # [km/s] velocity
    abla = thermal_ablation

    # -- for visible meteors, the energy is in lines, it is believed that they
    # are composed of iron (Cepleca p. 355, table 3)
    epsilon_mu = 7.668e6  # [J/kg] mean excitation energy = epsilon/mu; mu = mean molecular mass

    # -- The excitation coeff for different velocity intervals:
    # -- See Hill et al. and references therein
    zeta = np.zeros(v.shape, dtype=v.dtype)
    ind = v < 20000

    zeta[ind] = -2.1887e-9 * v[ind] ** 2 + 4.2903e-13 * v[ind] ** 3 - 1.2447e-17 * v[ind] ** 4

    ind = np.logical_and(v >= 20000, v < 60000)
    zeta[ind] = 0.01333 * vkm[ind] ** 1.25

    ind = np.logical_and(v >= 60000, v < 100000)
    zeta[ind] = -12.835 + 6.7672e-4 * v[ind] - 1.163076e-8 * v[ind] ** 2
    zeta[ind] += 9.191681e-14 * v[ind] ** 3 - 2.7465805e-19 * v[ind] ** 4

    ind = v >= 100000
    zeta[ind] = 1.615 + 1.3725e-5 * v[ind]

    # -- Luminous efficiency factor
    tau_I = 2 * epsilon_mu * zeta / v**2

    # -- The normal lumonous equation
    intensity = -0.5 * tau_I * abla * v**2

    return intensity
