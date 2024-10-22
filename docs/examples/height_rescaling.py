import matplotlib.pyplot as plt
import numpy as np

import ablate

atm = ablate.atmosphere.AtmPymsis()
heights = np.linspace(70e3, 120e3, 200)

data = atm.density(
    time=np.datetime64("2018-07-28T00:00:00"),
    lat=np.array([69.0]),
    lon=np.array([12.0]),
    alt=heights,
)
atm_density = data["Total"].values.squeeze()
atm_density_exp = ablate.physics.alpha_beta.atmosphere_density(
    height=heights,
    atmospheric_scale_height=7610.0,  # at T = 260 K,
    sea_level_rho=1.225,
)
print(data)

exp_scaled_heights = ablate.physics.alpha_beta.scale_hight_to_exponential_atm(
    atm_total_mass_density=atm_density, 
    atmospheric_scale_height=7610.0,
    sea_level_rho=1.225,
)

atm_scale_heights = ablate.physics.alpha_beta.scale_hight_to_model_atm(
    heights,
    atm,
    sea_level_rho=1.225,
    density_kwargs=dict(
        time=np.datetime64("2018-07-28T00:00:00"),
        lat=np.array([69.0]),
        lon=np.array([12.0]),
    ),
    atmospheric_scale_height=7610.0,
)


fig, axes = plt.subplots(1, 2)

axes[0].plot(heights * 1e-3, exp_scaled_heights * 1e-3, label="Scale to exponential")
axes[0].plot(heights * 1e-3, atm_scale_heights * 1e-3, label="Scale to MSIS")
axes[0].set_xlabel("Initial Heights [km]")
axes[0].set_ylabel("Scaled heights [km]")
axes[0].grid("on")
axes[0].legend()

axes[1].semilogx(atm_density, heights * 1e-3, "-b", label="MSIS")
axes[1].semilogx(atm_density, exp_scaled_heights * 1e-3, "xb", label="Rescaled MSIS")
axes[1].semilogx(atm_density_exp, heights * 1e-3, "-r", label="Exponential")
axes[1].semilogx(atm_density_exp, atm_scale_heights * 1e-3, "xr", label="Rescaled exponential")
axes[1].set_xlabel("Heights [km]")
axes[1].set_xlabel("Atmospheric mass density [kg/m^3]")
axes[1].legend()

plt.show()
