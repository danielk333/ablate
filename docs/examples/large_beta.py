import numpy as np
import matplotlib.pyplot as plt

from metablate.physics import alpha_beta

alpha = 100
beta = np.linspace(1e1, 1e3, 200)

norm_vel = 0.99
norm_h = 12.0

h = alpha_beta.norm_height_direct(norm_vel, alpha, beta)
h_approx = alpha_beta.approx_norm_height_direct(norm_vel, alpha, beta)

# these are not vectorized in alpha and beta yet
v = np.array([
    alpha_beta.norm_velocity_estimate(norm_h, alpha, b)
    for b in beta
])
v_approx = np.array([
    alpha_beta.approx_norm_velocity(norm_h, alpha, b)
    for b in beta
])

fig, axes = plt.subplots(2, 2, sharex="all")
ax = axes[0, 0]
ax.loglog(beta, h, label="Exact formula")
ax.loglog(beta, h_approx, label="Approximate formula")
ax.legend()

ax = axes[1, 0]
ax.plot(beta, h - h_approx)

ax = axes[0, 1]
ax.loglog(beta, v, label="Exact formula")
ax.loglog(beta, v_approx, label="Approximate formula")
ax.legend()

ax = axes[1, 1]
ax.plot(beta, v - v_approx)


plt.show()
# 
