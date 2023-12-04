"""
Alpha-beta model usage
=======================

Example adapted from:
 - https://github.com/desertfireballnetwork/alpha_beta_modules
 - https://doi.org/10.1016/j.asr.2009.03.030

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


