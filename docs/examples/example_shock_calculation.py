import numpy as np
import matplotlib.pyplot as plt

from metablate.physics import aerodynamics
from metablate.atmosphere import AtmPymsis

model = AtmPymsis()

print("NRL MSISE00 species:")
for name, species_data in model.species.items():
    print(f"{name}:{species_data}")

select_species = ["N2", "O2"]
alts = np.linspace(30e3, 80e3, num=100)
v0 = 6e3

atm = model.density(
    time=np.datetime64("2018-07-28"),
    lat=np.array([69.0]),
    lon=np.array([12.0]),
    alt=alts,
    mass_densities=False,
)
temp = atm["Temperature"].values.flatten()
num_tot = np.zeros_like(temp)
mean_mass = np.zeros_like(temp)
for symbol in select_species:
    num_tot += atm[symbol].values.flatten()
mean_mass = atm["Total"].values.flatten() / num_tot

sound_speeds = aerodynamics.speed_of_sound_air(temp, mean_mass)
mach_numbers = v0 / sound_speeds
pre_shock_speeds = mach_numbers * sound_speeds
post_shock_temps = aerodynamics.rankine_hugoniot_post_shock_temperature(temp, mach_numbers)
post_shock_mach_nums = aerodynamics.rankine_hugoniot_post_shock_mach_number(mach_numbers)
post_shock_speeds = post_shock_mach_nums * sound_speeds
eff_area = np.pi * (3.7e-10 / 2) ** 2
mfp = aerodynamics.atmospheric_mean_free_path(num_tot, eff_area)
Kn = mfp / 1.0

fig, axes = plt.subplots(1, 4, layout="tight")

axes[0].plot(mach_numbers, alts * 1e-3)
axes[0].set_xlabel("Relative flow speed [Mach]")
axes[0].set_ylabel("Altitude [km]")

axes[1].plot(post_shock_temps, alts * 1e-3)
axes[1].set_xlabel("Post-shock temperature [K]")
axes[1].set_ylabel("Altitude [km]")

axes[2].semilogx(Kn, alts * 1e-3)
axes[2].set_xlabel("Knutsen number [1]")
axes[2].set_ylabel("Altitude [km]")

for symbol in select_species:
    axes[3].semilogx(atm[symbol].values.flatten(), alts * 1e-3, label=symbol)
axes[3].semilogx(num_tot, alts * 1e-3, label="Total")
axes[3].legend()
axes[3].set_xlabel("Number density [1/km^3]")
axes[3].set_ylabel("Altitude [km]")


plt.show()
