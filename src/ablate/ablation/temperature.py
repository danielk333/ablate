#!/usr/bin/env python

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
import numpy as np
from scipy import constants
import scipy.optimize as sco
import scipy.special as scs


logger = logging.getLogger(__name__)



def temperature_rate(
    mass,
    velocity,
    temperature,
    material_data,
    shape_factor,
    atm_total_mass_density,
    thermal_ablation,
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
    thermal_ablation : float or numpy.ndarray
        Mass loss due to thermal ablation [kg/s]
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
    The default emissivity 0.9 is taken from [1]. Other like [2] report 1.0, 0.2 is
    characteristic for a metal, oxides are between 0.4 and 0.8.

    Changes pre-python port: See [matlab-version](https://gitlab.irf.se/kero/ablation_matlab)

    Heat balance equation per cross-sectional area:

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

    See Also
    --------
    [1] Hill et al.: High geocentric velocity meteor ablation, A&A 444 p. 615-624 (2005)
    [2] Love & Brownlee (?)

    """

    # -- variables
    kB = constants.value("Boltzmann constant")  # [J/K]

    rho_m = material_data["rho_m"]
    c = material_data["c"]
    L = material_data["L"]

    coef0 = 0.5 * Lambda * atm_total_mass_density * velocity**3
    coef0 -= 4 * kB * emissivity * (temperature**4 - atm_temperature**4)
    coef0 += L / shape_factor * (rho_m / mass) ** (2.0 / 3.0) * thermal_ablation

    coef1 = c * mass ** (1.0 / 3.0) * rho_m ** (2.0 / 3.0)

    dTdt = (shape_factor / coef1) * coef0
    return dTdt

