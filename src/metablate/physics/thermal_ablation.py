"""
# Thermal ablation physics

## Examples

Calculate thermal ablation:

```python
import numpy as np
import ablate.functions as func

dmdt_th = func.ablation.thermal_ablation(
    mass = np.array([0.5, 0.2]),
    temperature = 3700,
    material = 'ast',
    A = 1.21,
)

print(f'Thermal ablation {dmdt_th} kg/s')
```
"""

import logging
from scipy import constants
import numpy as np

logger = logging.getLogger(__name__)

Kn_LIM = 5.0


def temperature_rate_hill_et_al_2005(
    mass,
    velocity,
    temperature,
    material_data,
    shape_factor,
    atm_total_mass_density,
    mass_loss_thermal_ablation,
    Lambda,
    atm_temperature=280,
    emissivity=0.9,
):
    """Calculates the rate of change of temperature of the meteoroid.
    A homogeneous metoroid experiencing an isotropic heat flux is assumed as
    well as the meteoroid undergoing isothermal heating. (isothermal heating:
    here: dTdS = 0 i.e. same spatial temperature).

    Parameters
    ----------
    mass : float or numpy.ndarray
        Meteoroid mass [kg]
    velocity : float or numpy.ndarray
        Meteoroid velocity [m/s]
    temperature : float or numpy.ndarray
        Meteoroid temperature [K]
    material_data : dict
        Meteoroid material data, see [`material_data`][ablate.functions.material.material_data].
    shape_factor : float or numpy.ndarray
        Shape factor [1]
    atm_total_mass_density : float or numpy.ndarray
        Total atmospheric mass density [kg/m^3]
    mass_loss_thermal_ablation : float or numpy.ndarray
        Mass loss due to thermal ablation, i.e. -dm_dt [kg/s]
    Lambda : float or numpy.ndarray
        Heat transfer coefficient [1]
    atm_temperature : float or numpy.ndarray
        Effective atmospheric temperature [K]. Default = 280 K
    emissivity : float or numpy.ndarray
        Electromagnetic emissivity of meteoroid [1]. Default = 0.9

    Returns
    -------
    float or numpy.ndarray
        Rate of change of temperature [K/s]

    Notes
    -----
    The default emissivity 0.9 is taken from [^1], [^2] report 1.0 (todo: check), 0.2 is
    characteristic for a metal, oxides are between 0.4 and 0.8.

    Changes pre-python port: See [matlab-version](https://gitlab.irf.se/kero/ablation_matlab)

    Heat balance equation per cross-sectional area symbols:

    - `A` = shape factor
    - `c` = specific heat of meteoroid
    - `mass` = meteoroid mass
    - `rho_m` = meteoroid density
    - `Lambda` = heat transfer coefficient
    - `rho_tot`= total atmospheric mass density
    - `vel` = meteoroid velocity
    - `kb` = Stefan-Bolzmann constant
    - `T` = meteoroid temperature
    - `Ta` = effective atmospheric temperature
    - `L` = latent heat of fusion + vapourization
    - `thermal_ablation` = mass loss due to thermal ablation

    [1^]: Hill et al.: High geocentric velocity meteor ablation, A&A 444 p. 615-624 (2005)
    [2^]: Love & Brownlee (? todo)

    """

    # -- variables
    kB = constants.value("Boltzmann constant")  # [J/K]

    rho_m = material_data["rho_m"]
    c = material_data["c"]
    L = material_data["L"]

    coef0 = 0.5 * Lambda * atm_total_mass_density * velocity**3
    coef0 -= 4 * kB * emissivity * (temperature**4 - atm_temperature**4)
    coef0 += L / shape_factor * (rho_m / mass) ** (2.0 / 3.0) * (-mass_loss_thermal_ablation)

    coef1 = c * mass ** (1.0 / 3.0) * rho_m ** (2.0 / 3.0)

    dTdt = (shape_factor / coef1) * coef0
    return dTdt


def heat_transfer_bronshten_1983(
    mass,
    velocity,
    temperature,
    material_data,
    atm_total_number_density,
    mass_loss_thermal_ablation,
    atm_mean_mass,
    res=100,
):
    """Calculates the heat transfer coefficient Lambda. Information taken from [^1] [^2] [^3] [^4] [^5].

    Changes pre-python port: See [matlab-version](https://gitlab.irf.se/kero/ablation_matlab)

    [^1]: V. A. Bronshten; Physics of meteoric phenomena (1983)
    [^2]: A. Westman et al.: Meteor head echo altitude distributions and the
        height cutoff effect sudied with the EISCAT HPLA UHF and VHF radars;
        Annales Geophysicae 22: 1575-1584 (2004)
    [^3]: Tielens et al.: The physics of grain-grain collisions and gas-grain
        sputtering in interstellar shocks, The Astrophisicsl Journal 431, p. 321-340 (1994)
    [^4]: Rogers et al.: Mass loss due to  sputtering and thermal processes in
        meteoroid ablation, Planetary and Space Science 53 p. 1341-1354 (2005)
    [^5]: Salby: Fundamentals of atmospheric physics, Academic Press (1996)

    Parameters
    ----------
    mass : float or numpy.ndarray
        Meteoroid mass [kg]
    velocity : float or numpy.ndarray
        Meteoroid velocity [m/s]
    temperature : float or numpy.ndarray
        Meteoroid temperature [K]
    material_data : dict
        Meteoroid material data, see [`material_data`][ablate.functions.material.material_data].
    atm_total_number_density : float or numpy.ndarray
        Total atmospheric number density [1/m^3]
    mass_loss_thermal_ablation : float or numpy.ndarray
        Mass loss due to thermal ablation, i.e. -dm_dt [kg/s]
    atm_mean_mass : float or numpy.ndarray
        Mean mass of atmospheric constituents [kg]
    res : int
        Resolution used for the numerical integration.

    Returns
    -------
    float or numpy.ndarray
        Heat transfer coefficient

    """

    # Universal constants
    kB = constants.value("Boltzmann constant")  # [J/K]

    # Meteoroid constituents
    met_mean_mass = material_data["m2"]
    met_density = material_data["rho_m"]

    # angle between the normal to the surface and the molecular flow, theta0 is a row vector
    theta0 = np.linspace(0, np.pi / 2, num=res, dtype=np.float64)  # [radians]

    a_e = thermal_accommodation_bronshten_1983(atm_mean_mass, met_mean_mass)
    u_s = nomalized_evaporated_velocity_bronshten_1983(velocity, temperature, atm_mean_mass)
    Kn_inf, L = Knudsen_number_kero_szasz_2008(mass, met_density, atm_total_number_density)

    # parameter... Bronshten p. 69 eq. 10.7
    h_star = atm_mean_mass * velocity**2 / (2.0 * kB * temperature)  # [dimensionless]

    # ####### Lambda ##########

    # Shielding effects of a meteoroid surface due to reflected molecules:
    epsilon_q = 1.6  # [dimensionless]

    _G = np.sin(theta0) * np.cos(theta0) * (1 - u_s**2)

    _Gp = (
        np.sin(theta0)
        * 4
        / np.sqrt(2 * np.pi)
        * np.sqrt(1 - 2 * u_s * np.cos(theta0) + u_s**2)
        * np.cos(theta0)
    )
    _Gp *= np.cos(theta0) * (1 - u_s**2) / 2 - (
        1.0 / 48.0 * (1 + np.cos(theta0)) ** 3 * (3 - np.cos(theta0))
        - u_s
        / 48.0
        * (1 + np.cos(theta0)) ** 2
        * (8 - 7 * np.cos(theta0) + 2 * np.cos(theta0) ** 2 - np.cos(theta0) ** 3)
    )

    Q0 = np.trapz(theta0, _G)

    # integrating q_prim over theta0 from 0 - pi/2; Bronshten p. 69, eq. 10.10 and Bronshten I eq. 11
    Q_prim = np.trapz(theta0, _Gp)

    # the energy flux shielding coeff, Bronshten p. 71, same as a_Lambda but with
    # the integrated vectors; Bronshten p. 69 eq. 10.8
    A_Lambda_r = 1 - epsilon_q * np.sqrt(h_star) / Kn_inf * Q_prim / Q0  # [dimensionless]

    # Shielding effects of a meteoroid surface due to evaporated molecules:

    # scattering cross section for meteor atoms and ions on N2 and O2 molecules;
    # Bronshten p. 84 eq. 11.29, Bronshten II. eq. 40; the actual equation is for
    # v in [cm/s] and gives the cross section in cm^2 ->
    # sigmaD = 5.6E-11 [cm^2] *(v [cm/s])^-0.8 = 5.6E-11*(1E-2[m])^2*(v [m/s]*1E2)^-0.8 =
    # 5.6E-11*1E-4*(1E2)^-0.8*v^-0.8 = 5.6E-15*10^-1.6*v^-0.8
    sigmaD = 5.6e-15 * 10 ** (-1.6) * velocity ** (-0.8)  # [m^2]

    # the velocity of molecules evaporated from the meteor body, Bronshten eq. 7.2 p. 37
    v_e = np.sqrt(8 * kB * temperature / (np.pi * met_mean_mass))  # [m/s]

    # mean free path of the evaporated molecules, Bronshten p. 79 eq. 11.8; Bronshten II. eq. 2
    mfp_e = v_e / (atm_total_number_density * velocity * sigmaD)  # [m]

    # the toatal fraction of evaporated molecules from the entire front surface
    # of the meteoroid, Bronshten II. eq. 35
    eta = (L / mfp_e) ** 2 / (1 + 2 * np.sqrt(L / mfp_e))  # [dimensionless]

    # the total number of molecules evaporated from a certain area of cross section
    # and which takes part in shielding, Bronshten II. eq. 42 and eq. 1
    N_i = eta / met_mean_mass * mass_loss_thermal_ablation  # [number]

    # cross section area of meteoroid (L = characteristic length, but here it is
    # equal to the meteoroid radius)
    S_m = np.pi * L**2  # [m^2]

    # the number of molecules advancing on the area S, Bronshten II. p. 135
    N_a = atm_total_number_density * velocity * S_m  # [number]

    # average velocity of evaporated molecules from the meteoroid surface, see
    # v_s which is the same thing but for reflected molecules
    v_s_e = np.sqrt(
        3
        * constants.value("molar gas constant")
        * temperature
        / (met_mean_mass * constants.value("Avogadro constant"))
    )  # [m/s]

    # nomalized velocity of evaporated molecules, compare to u_s
    u_e = v_s_e / velocity  # [dimensionless]

    _H = np.sin(theta0) * 1.0 / 48.0 * (1 + np.cos(theta0)) ** 3 * (
        3 - np.cos(theta0)
    ) - u_e / 48.0 * (1 + np.cos(theta0)) ** 2 * (
        8 - 7 * np.cos(theta0) + 2 * np.cos(theta0) ** 2 - np.cos(theta0) ** 3
    )

    # same as Q_star but for evaporated molecules, i.e. using u_e instead of u_s
    Q_star_e = np.trapz(theta0, _H)
    # same as Q0 but for evaporated molecues, i.e. using u_e instead of u_s
    Q0_e = np.trapz(theta0, np.sin(theta0) * np.cos(theta0) * (1 - u_e**2))

    # determines how many of the reflected molecules that are thrown back, Bronshten I. eq. 17
    zeta = 1 - 2 * Q_star_e / Q0_e

    # the energy flux shielding by evaporation coeff for a sphere, Bronshten II. eq. 43
    A_Lambda_e = 1 - zeta * N_i / N_a  # [dimensionless]

    # How to add the shielding effects by reflection and evaporation? A_Lambda_r and A_Lambda_e
    # is defined as the probability that an incoing molecule reaches the surface of the body and
    # transfers its momentum or energy respectively to it. Thus 1 - A_Lambda_r and 1 - A_Lambda_e
    # is the probability that an incoming molecule does not reach the surface of the body, i.e.,
    # is shielded. So, to add how many of the incoming molecules contributes to the shielding both
    # by reflection and evaporation, we need to add the probability that the incoming molecules
    # does not reach the surface: 1 - ( (1 - A_Lambda_r) + (1 - A_Lambda_e) ) =
    # = 1 - 1 + A_Lambda_r - 1 + A_Lambda_e = A_lambda_r + A_lambda_e -1

    Lambda = (A_Lambda_r + A_Lambda_e - 1) * a_e  # [dimensionless] heat transfer coeff

    # We don't want Gamma or Lambda to be > 1, searching for those and setting them = 1
    if isinstance(Lambda, np.ndarray):
        Lambda[Lambda > 1] = 1.0
    else:
        if Lambda > 1:
            Lambda = 1.0

    return Lambda


def thermal_accommodation_bronshten_1983(atm_mean_mass, met_mean_mass):
    """Calculate thermal accommodation coefficient [^1].

    [^1]: Bronshten p. 40 eq. 7.11

    Parameters
    ----------
    met_mean_mass : float or numpy.ndarray
        Mean mass of meteoroid constituents [kg]
    atm_mean_mass : float or numpy.ndarray
        Mean mass of atmospheric constituents [kg]

    Returns
    -------
    float or numpy.ndarray
        Thermal accommodation coefficient

    """

    # the relative masses of the molecules of the air and of the meteoroid, Bronshten p. 40
    mu_star = atm_mean_mass / met_mean_mass  # [dimensionless]

    # thermal accommodation coefficient, Bronshten p. 40 eq. 7.11
    a_e = (3.0 + mu_star) * mu_star / (1.0 + mu_star) ** 2  # [dimensionless]

    return a_e


def nomalized_evaporated_velocity_bronshten_1983(velocity, temperature, atm_mean_mass):
    """Calculate the nomalized reflected and evaporated velocity [^1].

    [^1]: Bronshten p. 69

    Parameters
    ----------
    velocity : float or numpy.ndarray
        Meteoroid velocity [m/s]
    temperature : float or numpy.ndarray
        Meteoroid temperature [K]
    atm_mean_mass : float or numpy.ndarray
        Mean mass of atmospheric constituents [kg]

    Returns
    -------
    float or numpy.ndarray
        Nomalized reflected and evaporated velocity

    """

    # Universal constants
    R = constants.value("molar gas constant")  # [J/(mol K)]
    NA = constants.value("Avogadro constant")  # [molecules/mol]

    # average velocity of reflected molecules near the meteoroid surface, Bronshten p. 37 eq. 7.1 (and 7.2)
    v_s = np.sqrt(3 * R * temperature / (atm_mean_mass * NA))  # [m/s]

    # nomalized reflected and evaporated velocity, Bronshten p. 69 but uncorrect in the book...
    u_s = v_s / velocity  # [dimensionless]

    return u_s


def thermal_ablation_hill_et_al_2005(mass, temperature, material_data, shape_factor):
    """Calculates the mass loss for meteoroids due to thermal ablation [^1][^2].

    Changes pre-python port: See [matlab-version](https://gitlab.irf.se/kero/ablation_matlab)

    [^1]: Rogers et al.: Mass loss due to  sputtering and thermal processes in
        meteoroid ablation, Planetary and Space Science 53 p. 1341-1354 (2005)
    [^2]: Hill et al.: High geocentric velocity meteor ablation, A&A 444 p. 615-624 (2005)

    Parameters
    ----------
    mass : float or numpy.ndarray
        Meteoroid mass [kg]
    temperature : float or numpy.ndarray
        Meteoroid temperature [K]
    material_data : dict
        Meteoroid material data, see [`material_data`][ablate.functions.material.material_data].
    shape_factor : float or numpy.ndarray
        Shape factor [1]


    Returns
    -------
    float or numpy.ndarray
        Mass loss due to thermal ablation [kg/s]

    Notes
    -----
    Symbol definitions:

     - `Pv`: vapour pressure of meteoroid [N/m^2]
     - `CA`, CB: Clausius-Clapeyron coeff [K]
     - `mu`: mean molecular mass of ablated vapour [kg]
     - `rho_m`: meteoroid density [kg/m3]


    """

    CA = material_data["CA"]
    CB = material_data["CB"]
    mu = material_data["mu"]
    rho_m = material_data["rho_m"]

    # 2007-03-21 Ed suggests that the vapor pressure should be lowered by a factor of 0.8 or 0.7
    # due to the large ram pressure forcing the evaporated atoms and molecules back on to the surface
    # thus leading to evaporation at a pressure that is lower than the equilibrium vapor pressure.

    Pv = 10.0 ** (CA - CB / temperature)
    # in [d/cm2]...; d=dyne=10-5 Newton: the force required to accelearte a
    # mass of one g at a rate of one cm per s^2
    Pv = Pv * 1e-5 / 1e-4  # Convert to [N/m2]

    coef0 = np.sqrt(mu / (2.0 * np.pi * constants.k * temperature))
    dmdt = -4.0 * shape_factor * (mass / rho_m) ** (2.0 / 3.0) * Pv * coef0  # [kg/s]

    return dmdt


def drag_coefficient_bronshten_1983(
    mass,
    velocity,
    temperature,
    material_data,
    atm_total_number_density,
    atm_mean_mass,
    res=100,
):
    """Calculates the drag coefficient Gamma[^1] [^2] [^3] [^4] [^5].

    Changes pre-python port: See [matlab-version](https://gitlab.irf.se/kero/ablation_matlab)


    [^1]: V. A. Bronshten; Physics of meteoric phenomena (1983)
    [^2]: A. Westman et al.: Meteor head echo altitude distributions and the
        height cutoff effect sudied with the EISCAT HPLA UHF and VHF radars;
        Annales Geophysicae 22: 1575-1584 (2004)
    [^3]: Tielens et al.: The physics of grain-grain collisions and gas-grain
        sputtering in interstellar shocks, The Astrophisicsl Journal 431, p. 321-340 (1994)
    [^4]: Rogers et al.: Mass loss due to  sputtering and thermal processes in
        meteoroid ablation, Planetary and Space Science 53 p. 1341-1354 (2005)
    [^5]: Salby: Fundamentals of atmospheric physics, Academic Press (1996)

    Parameters
    ----------
    mass : float or numpy.ndarray
        Meteoroid mass [kg]
    velocity : float or numpy.ndarray
        Meteoroid velocity [m/s]
    temperature : float or numpy.ndarray
        Meteoroid temperature [K]
    material_data : dict
        Meteoroid material data, see [`material_data`][ablate.functions.material.material_data].
    atm_total_number_density : float or numpy.ndarray
        Total atmospheric number density [1/m^3]
    atm_mean_mass : float or numpy.ndarray
        Mean mass of atmospheric constituents [kg]
    res : int
        Resolution used for the numerical integration.

    Returns
    -------
    float or numpy.ndarray
        Drag coefficient

    """
    # Universal constants
    kB = constants.value("Boltzmann constant")  # [J/K]

    # Meteoroid constituents
    met_mean_mass = material_data["m2"]
    met_density = material_data["rho_m"]

    # angle between the normal to the surface and the molecular flow, theta0 is a row vector
    theta0 = np.linspace(0, np.pi / 2, num=res, dtype=np.float64)  # [radians]

    # parameter... Bronshten p. 69 eq. 10.7
    h_star = atm_mean_mass * velocity**2 / (2.0 * kB * temperature)  # [dimensionless]

    a_e = thermal_accommodation_bronshten_1983(atm_mean_mass, met_mean_mass)
    u_s = nomalized_evaporated_velocity_bronshten_1983(velocity, temperature, atm_mean_mass)
    Kn_inf, _ = Knudsen_number_kero_szasz_2008(mass, met_density, atm_total_number_density)

    # ####### Gamma ##########
    epsilon_p = 0.86  # [dimensionless] Bronshten p. 69 eq. 10.7 and p. 71

    # integrating p0, Bronshten p. 71
    _F = np.sin(theta0) * np.cos(theta0) * (1.0 - u_s)

    _Fp = (
        np.sin(theta0)
        * 4
        / np.sqrt(2 * np.pi)
        * np.sqrt(1 - 2 * u_s * np.cos(theta0) + u_s**2)
        * np.cos(theta0)
    )
    _Fp *= (
        np.cos(theta0) * (1 - u_s) / 2
        - 1.0 / 24.0 * (1 + np.cos(theta0)) ** 3
        + u_s / 24.0 * (1 + np.cos(theta0)) ** 2 * (4 - 2 * np.cos(theta0) + np.cos(theta0) ** 2)
    )

    P0 = np.trapz(theta0, _F)

    # integrating p_prim over theta0 from 0 - pi/2, Bronshten p. 71
    P_prim = np.trapz(theta0, _Fp)

    # the momentum flux shielding coeff, Bronshten p. 71 same as a_Gamma but with the integrated vectors
    if isinstance(Kn_inf, np.ndarray):
        A_Gamma_r = np.zeros_like(Kn_inf)
        inds = Kn_inf > Kn_LIM
        A_Gamma_r[inds] = (
            1 - epsilon_p * np.sqrt(h_star) / Kn_inf[inds] * P_prim / P0
        )  # [dimensionless]
        Gamma = A_Gamma_r * a_e  # [dimensionless] drag coeff

        # TODO: replace this with actual cont. flow dynamics during ablation
        Gamma[np.logical_not(inds)] = (
            0.47 * 0.5
        )  # [dimensionless] drag coeff of sphere in cont. flow
    else:
        if Kn_inf > Kn_LIM:
            A_Gamma_r = 1 - epsilon_p * np.sqrt(h_star) / Kn_inf * P_prim / P0  # [dimensionless]
            Gamma = A_Gamma_r * a_e  # [dimensionless] drag coeff
        else:
            # TODO: replace this with actual cont. flow dynamics during ablation
            Gamma = 0.47 * 0.5  # [dimensionless] drag coeff of sphere in cont. flow

    # We don't want Gamma or Lambda to be > 1, searching for those and setting them = 1
    if isinstance(Gamma, np.ndarray):
        Gamma[Gamma > 1] = 1.0
    else:
        if Gamma > 1:
            Gamma = 1.0

    return Gamma


def Knudsen_number_kero_szasz_2008(met_mass, met_density, atm_total_number_density):
    """Calculate the Knudsen number.

    Parameters
    ----------
    met_mass : float or numpy.ndarray
        Meteoroid mass [kg]
    met_density : float or numpy.ndarray
        Meteoroid density [kg/m^3]
    atm_total_number_density : float or numpy.ndarray
        Total atmospheric number density [1/m^3]

    Returns
    -------
    tuple(float/numpy.ndarray, float/numpy.ndarray)
        Knudsen number, Body characteristic dimension

    """

    # Atmospheric mean free path, source: Westman et al.
    # Physics Handbook p. 186 This is the one used!
    mfp_inf = 1.0 / (np.pi * (3.62e-10) ** 2 * atm_total_number_density)  # [m]

    # The Knudsen number
    # Calculating the radius of the meteoroid at different altutude to get L
    # characteristic dimension of the body (the meteoroid); in particular, for a
    # sphere L = the radius; Bronshten p. 31
    L = ((met_mass / met_density) / (4 * np.pi / 3)) ** (1.0 / 3.0)  # [m]

    # [dimensionless] the Knudsen no; calculated from 'mfp_inf'
    Kn_inf = mfp_inf / L

    return Kn_inf, L


def ionization_probability_Jones_1997(velocity, meteoroid_bulk_density):
    """Calculate ionization probability according to [^1].

    [^1]: William Jones; Theoretical and observational determinations
    of the ionization coefficient of meteors; MNRAS 288, p. 995-1003 (1997)

    """
    raise NotImplementedError()
    # v_km = v / 1000.0
    """
    # Table 1 from Jones:
    # %   : procentage composition by weight
    # p   : proportion by atom number
    # v0  : minimum velocity at which ionization can take place 
    # (below this velocity, the ionization energy is greater than 
    # the total energy of the colliding atoms in the centre-of-mass frame)
    # c   : empirically derived coeff
    # mu  : ratio of atom mass to a nitrogen molecule
    #                 % p v0 c mu element
    ion_param       = [
        [45, 0.617, 16.7, 4.66E-6, 0.57],  # O
        [15, 0.059, 9.4, 34.5E-6, 2.0],  # Fe
        [9, 0.082, 11.1, 9.29E-6, 0.86],  # Mg
        [31, 0.242, 11.0, 18.5E-6, 1.0],  # Si
    ]
             
    # beta0 = c .* (v - v0).^2 .* v.^0.8 ./ (1 + c .* (v - v0).^2 .* v.^0.8); primary ionization probability, eq. 33 in Jones
    # v is needed in km/s!

    if rho_m == 7800:
        # Then we calculate the ionization probability for pure Fe
        beta0 = ion_param(2,4) .* (v_km - ion_param(2,3)).^2 .* v_km.^0.8 ./ (1 + ion_param(2,4) .* (v_km - ion_param(2,3)).^2 .* v_km.^0.8);
        beta0(find(v_km<ion_param(2,3))) = 0;
    else
        # Otherwise we use Jones' cometary composition (for porous, cometary as well as asteroidal densities)
        # It is faster not to use repmat.m but to generate the necessary matrices with more simple tools!
        c               = [ion_param(1,4)*ones(length(v),1) ion_param(2,4)*ones(length(v),1) ion_param(3,4)*ones(length(v),1) ion_param(4,4)*ones(length(v),1)];
        v0              = [ion_param(1,3)*ones(length(v),1) ion_param(2,3)*ones(length(v),1) ion_param(3,3)*ones(length(v),1) ion_param(4,3)*ones(length(v),1)];
        p               = [ion_param(1,2)*ones(length(v),1) ion_param(2,2)*ones(length(v),1) ion_param(3,2)*ones(length(v),1) ion_param(4,2)*ones(length(v),1)];
        v_km            = [v_km v_km v_km v_km]; 
        beta0_1         = c .* (v_km - v0).^2 .* v_km.^0.8 ./ (1 + c .* (v_km - v0).^2 .* v_km.^0.8);
        beta0_1(find(v_km<v0)) = 0;
        beta0           = sum(beta0_1 .* p , 2);
    """
