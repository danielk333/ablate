"""
Example run of an ablation model
=================================

Docstring for this example
"""

import logging
import sys

import numpy as np
import matplotlib.pyplot as plt

import ablate

handler = logging.StreamHandler(sys.stdout)

for name in logging.root.manager.loggerDict:
    if name.startswith("ablate"):
        print(f"logger: {name}")


logger = logging.getLogger("ablate")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

model = ablate.KeroSzasz2008(
    atmosphere="msise00",
    options=dict(
        sputtering=True,
        minimum_mass=1e-11,
        max_step_size=5e-3,
    ),
)

material_data = ablate.functions.material.material_parameters("iron")

result = model.run(
    velocity0=60 * 1e3,
    mass0=1e-6,
    # mass0=1e-3,
    altitude0=120e3,
    zenith_ang=75.7744,
    azimuth_ang=0.0,
    material_data=material_data,
    time=np.datetime64("2018-06-28T12:45:33"),
    lat=69.5866115,
    lon=19.221555,
    alt=100e3,
)

print(result)

fig = plt.figure(figsize=(15, 15))
fig.suptitle("Meteoroid ablation simulation")

ax = fig.add_subplot(221)
ax.plot(result.t, np.log10(result.mass))
ax.set_ylabel("Mass [log$_{10}$(kg)]")
ax.set_xlabel("Time [s]")

ax = fig.add_subplot(222)
ax.plot(result.t, result.velocity * 1e-3)
ax.set_ylabel("Velocity [km/s]")
ax.set_xlabel("Time [s]")

ax = fig.add_subplot(223)
ax.plot(result.t, result.position * 1e-3)
ax.set_ylabel("Position on trajectory [km]")
ax.set_xlabel("Time [s]")

ax = fig.add_subplot(224)
ax.plot(result.t, result.temperature)
ax.set_ylabel("Meteoroid temperature [K]")
ax.set_xlabel("Time [s]")


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111)
ax.plot(
    np.diff(result.mass.values) / np.diff(result.altitude.values),
    result.altitude.values[:-1] * 1e-3,
)
ax.set_xlabel("Mass loss [kg/m]")
ax.set_ylabel("Altitude [km]")


plt.show()


"""
Alpha-beta model usage
=======================

Example adapted from:
 - https://github.com/desertfireballnetwork/alpha_beta_modules
 - https://doi.org/10.1016/j.asr.2009.03.030

# TODO: clean up

"""
import scipy
import numpy as np
import matplotlib.pyplot as plt

from ablate.functions import ablation

f = "/home/danielk/data/test_data/DN150417.csv"
slope = 15.17
vel_col = "D_DT_geo"
h_col = "height"
data = np.genfromtxt(
    f, delimiter=",", dtype="f8,f8,f8", names=["height", "D_DT_geo", "D_DT_fitted"]
)

# f = "/home/danielk/data/MU/test_tabledata.txt"
# vel_col = "velocity"
# h_col = "height"
# data = np.genfromtxt(
#     f, skip_header=1, delimiter=",", dtype="f8,f8", names=["height", "velocity"],
# )
# slope = 14.2256


slope = np.deg2rad(slope)
# remove any nan values
data = data[np.logical_not(np.isnan(data[vel_col]))]
vel = data[vel_col]
alt = data[h_col]

# Initial velocity
v0 = vel[0]

# dimensionless parameter for velocity
Vvalues = vel/v0


# normalise height - if statement accounts for km vs. metres data values.
h0 = 7160.0  # metres
Yvalues = alt/h0

mat_shape = (300, 200)
alpha_mat, beta_mat = np.meshgrid(
    np.linspace(-3, 5, mat_shape[1]),
    np.linspace(-5, 3, mat_shape[0]),
)

xvec = np.stack([10.0**alpha_mat.flatten(), 10.0**beta_mat.flatten()])
fit_func = ablation.alpha_beta_min_fun(xvec, Vvalues, Yvalues)

fit_func = fit_func.reshape(mat_shape)

fig, ax = plt.subplots()
ax.pcolormesh(alpha_mat, beta_mat, np.exp(-fit_func))

plt.show()


Gparams = ablation.alpha_beta_Q4_min(Vvalues, Yvalues)

alpha = Gparams[0]
beta = Gparams[1]

print(f"alpha = {alpha}, beta = {beta}")

norm_v = np.arange(0.1, 1, 0.00005)


def norm_hight(norm_velocity):
    y = np.log(alpha) + beta
    y -= np.log((scipy.special.expi(beta) - scipy.special.expi(beta*norm_velocity**2))/2)
    return y


norm_h = norm_hight(norm_v)

fig, ax = plt.subplots()

ax.scatter(Vvalues, Yvalues, marker="x", label=None)
plt.plot(norm_v, norm_h, color="r")

ax.set_xlabel("normalised velocity")
ax.set_ylabel("normalised height")

# define x values
ln_alpha_singamma = np.arange(0, 10, 0.00005)


# The parameter mu represents rotation during the flight
# If mu = 0, no rotation; if mu = 2/3, the ablation becomes uniform
# over the surface (shape factor A does not change).


def ln_beta_paramerized(ln_alpha_singamma, shape_change_coef, log_mass_fraction):
    # function for mu = 0, 50 g possible meteorite:
    beta = (shape_change_coef - 1)*(log_mass_fraction + 3*ln_alpha_singamma)
    return np.log(beta)


log_mass_fraction_50g = -13.2

ln_beta_0 = ln_beta_paramerized(ln_alpha_singamma, 0, log_mass_fraction_50g)
ln_beta_23 = ln_beta_paramerized(ln_alpha_singamma, 2.0/3.0, log_mass_fraction_50g)


fig, ax = plt.subplots()

# plot mu0, mu2/3 lines and your point:
ax.plot(ln_alpha_singamma, ln_beta_0, color="black")
ax.plot(ln_alpha_singamma, ln_beta_23, color="grey")
ax.scatter([np.log(alpha*np.sin(slope))], [np.log(beta)], color="r")

# defite plot parameters
ax.set_xlim((-1, 7))
ax.set_ylim((-3, 4))
ax.set_xlabel("ln(alpha x sin(slope))")
ax.set_ylabel("ln(beta)")
ax.set_aspect("equal")

# Assumeing values:
# AERODYNAMIC drag coefficient (not Gamma)
cd = 1.3
# Possible shape coefficients
A = [1.21, 1.3, 1.55]
# possible meteoroid densities
m_rho = [2700, 3500, 7000]
# shape change coefficient
mu = 2./3.


me_sphere = [ablation.alpha_beta_entry_mass(alpha, beta, slope, cd, rho, A[0]) for rho in m_rho]
me_round_brick = [ablation.alpha_beta_entry_mass(alpha, beta, slope, cd, rho, A[1]) for rho in m_rho]
me_brick = [ablation.alpha_beta_entry_mass(alpha, beta, slope, cd, rho, A[2]) for rho in m_rho]

mf_sphere = [ablation.alpha_beta_final_mass(me, beta, mu, Vvalues[-1]) for me in me_sphere]
mf_round_brick = [ablation.alpha_beta_final_mass(me, beta, mu, Vvalues[-1]) for me in me_round_brick]
mf_brick = [ablation.alpha_beta_final_mass(me, beta, mu, Vvalues[-1]) for me in me_brick]

print("Masses in [g]")
print("Density in [kg/m^3]")
print(f"Entry mass of spherical body with {m_rho} density =\n", np.array(me_sphere)*1e3)
print(f"Final mass of spherical body with {m_rho} density =\n", np.array(mf_sphere)*1e3)
print("\n")
print(f"Entry mass of typical shape with {m_rho} density =\n", np.array(me_round_brick)*1e3)
print(f"Final mass of typical shape with {m_rho} density =\n", np.array(mf_round_brick)*1e3)
print("\n")
print(f"Entry mass of brick shape with {m_rho} density =\n", np.array(me_brick)*1e3)
print(f"Final mass of brick shape with {m_rho} density =\n", np.array(mf_brick)*1e3)
print("\n")


plt.show()
