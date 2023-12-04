"""
Example run of fitting ablation models
========================================
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as scio

import ablate


f = "/home/danielk/data/MU/test_tabledata.txt"
vel_col = "velocity"
h_col = "height"
data = np.genfromtxt(
    f, skip_header=1, delimiter=",", dtype="f8,f8", names=["height", "velocity"],
)
zenith_angle = 75.7744

# remove any nan values
times = np.arange(data.size)
keep = np.logical_not(np.isnan(data[vel_col]))
data = data[keep]
times = times[keep]
vel = data[vel_col]
alt = data[h_col]
vel0 = vel[0]

model = ablate.KeroSzasz2008(
    atmosphere="msise00",
    options=dict(
        sputtering=True,
        minimum_mass=1e-11,
        max_step_size=5e-3,
    ),
)

material_data = ablate.functions.material.material_parameters("iron")

mass = 10**np.linspace(-8, -3, 100)
velocity = np.linspace(vel0*0.8, vel0*1.2, 20)

mmat, vmat = np.meshgrid(mass, velocity)
fit_vals = np.zeros_like(mmat)

for i in range(len(mass)):
    for j in range(len(velocity)):

        result = model.run(
            velocity0=velocity[j],
            mass0=mass[i],
            altitude0=120e3,
            zenith_ang=zenith_angle,
            azimuth_ang=0.0,
            material_data=material_data,
            time=np.datetime64("2018-06-28T12:45:33"),
            lat=69.5866115,
            lon=19.221555,
            alt=100e3,
        )
        vfun = scio.interp1d(result.altitude.values, result.velocity.values)

        fig, ax = plt.subplots()
        ax.plot(alt, vfun(alt))
        ax.plot(alt, vel, '.')
        plt.show()

        fit_vals[j, i] = np.sum((vel - vfun(alt))**2)
        print(fit_vals[j, i])


fig, ax = plt.subplots()
ax.pcolormesh(mmat, vmat, fit_vals)

# fig = plt.figure(figsize=(15, 15))
# fig.suptitle("Meteoroid ablation simulation")

# ax = fig.add_subplot(221)
# ax.plot(result.t, np.log10(result.mass))
# ax.set_ylabel("Mass [log$_{10}$(kg)]")
# ax.set_xlabel("Time [s]")

# ax = fig.add_subplot(222)
# ax.plot(result.t, result.velocity * 1e-3)
# ax.set_ylabel("Velocity [km/s]")
# ax.set_xlabel("Time [s]")

# ax = fig.add_subplot(223)
# ax.plot(result.t, result.position * 1e-3)
# ax.set_ylabel("Position on trajectory [km]")
# ax.set_xlabel("Time [s]")

# ax = fig.add_subplot(224)
# ax.plot(result.t, result.temperature)
# ax.set_ylabel("Meteoroid temperature [K]")
# ax.set_xlabel("Time [s]")


# fig = plt.figure(figsize=(15, 15))
# ax = fig.add_subplot(111)
# ax.plot(
#     np.diff(result.mass.values) / np.diff(result.altitude.values),
#     result.altitude.values[:-1] * 1e-3,
# )
# ax.set_xlabel("Mass loss [kg/m]")
# ax.set_ylabel("Altitude [km]")


plt.show()
