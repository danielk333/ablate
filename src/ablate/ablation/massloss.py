
def thermal_ablation(mass, temperature, material_data, shape_factor):
    """Calculates the mass loss for meteoroids due to thermal ablation.

    :param float/numpy.ndarray mass: Meteoroid mass [kg]
    :param float/numpy.ndarray temperature: Meteoroid temperature [K]
    :param dict material_data: Meteoroid material data, see :mod:`~functions.material.material_data`.
    :param float/numpy.ndarray shape_factor: Shape factor [1]


    :rtype: float/numpy.ndarray
    :return: Mass loss due to thermal ablation [kg/s]


    **Reference:**

       *Rogers et al.: Mass loss due to  sputtering and thermal processes in
            meteoroid ablation, Planetary and Space Science 53 p. 1341-1354 (2005)
       *Hill et al.: High geocentric velocity meteor ablation, A&A 444 p. 615-624 (2005)


    **Change log:**

       *Changes post-python port: See git-log
       *Changes pre-python port: See `matlab-version <https://gitlab.irf.se/kero/ablation_matlab>`_


    **Symbol definintions:**

       *Pv: vapour pressure of meteoroid [N/m^2]
       *CA, CB: Clausius-Clapeyron coeff [K]
       *mu: mean molecular mass of ablated vapour [kg]
       *rho_m: meteoroid density [kg/m3]


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
