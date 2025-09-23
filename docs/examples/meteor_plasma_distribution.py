import numpy as np
import matplotlib.pyplot as plt
import ablate.atmosphere as atm
import ablate.models.dimant_oppenheim_2017 as do

met_vel = 40e3
ablated_thermal_vel = 951
met_r = 5e-5
ma=4.7e-26
n0 = 10**13 / met_r**2
rel_vel = met_vel - ablated_thermal_vel
sigma = do.collisional_cross_section_bronshten_1983(rel_vel)
beta = do.ionization_probability_Na_vondrak_2008(rel_vel)

model = atm.AtmPymsis()

atm_data = model.density(
    time=np.datetime64("2018-07-28"),
    lat=np.array([69.0]),
    lon=np.array([12.0]),
    alt=np.array([100e3]),
    mass_densities=False,
)
index = dict(lat=0, lon=0, alt=0, time=0)
na = atm_data["Total"][index].values / ma

ne_dist = do.plasma_distribution(
    total_atmospheric_number_density=na,
    meteoroid_velocity=met_vel,
    meteoroid_radius=met_r,
    plasma_source_density=n0,
    collisional_cross_section=sigma,
    ionization_probability=beta,
    ablated_thermal_speed=ablated_thermal_vel,
    atmospheric_species_mass=ma,
    ablated_species_mass=3.8e-26,
    grid_size=(201, 201),
    grid_step=(3.9e-4, 3.9e-4)
)

fig, ax = plt.subplots()
ax.pcolormesh(np.log10(ne_dist))

plt.show()
