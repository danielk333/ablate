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

# f = "/home/danielk/data/test_data/DN150417.csv"
# slope = 15.17
# vel_col = "D_DT_geo"
# h_col = "height"
# data = np.genfromtxt(
#     f, delimiter=",", dtype="f8,f8,f8", names=["height", "D_DT_geo", "D_DT_fitted"]
# )

f = "/home/danielk/data/MU/test_tabledata.txt"
vel_col = "velocity"
h_col = "height"
data = np.genfromtxt(
    f, skip_header=1, delimiter=",", dtype="f8,f8", names=["height", "velocity"],
)
slope = 14.2256


slope = np.deg2rad(slope)
# remove any nan values
data = data[np.logical_not(np.isnan(data[vel_col]))]
vel = data[vel_col]
alt = data[h_col]

# normalise height - if statement accounts for km vs. metres data values.
h0 = 7160.0  # metres
Yvalues = alt/h0
# start_vel = np.mean(vel[0:10])
start_vel = vel[0]

Gparams = ablation.alpha_beta_velocity_Q4_min(vel, alt, h0, start_vel)

alpha = Gparams[0]
beta = Gparams[1]
velocity0 = Gparams[2]

print(f"observed start velocity={start_vel}")
print(f"alpha = {alpha}, beta = {beta}, velocity0={velocity0}")

Vvalues = vel/velocity0

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

plt.show()
