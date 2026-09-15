"""
Estimate covariance
===================

Estimate the linearized covariance of alpha, beta, and entry velocity from a
single station's azimuth and elevation observations.
"""

import matplotlib.pyplot as plt
import numpy as np
from spacecoords import frames, spherical

import metablate.models.alpha_beta_2026 as ab

latitude = 67 + 50 / 60 + 26.6 / 3600
longitude = 20 + 24 / 60 + 40.0 / 3600
initial_height = 100e3
radiant_elevation = 45.0
radiant_azimuth = 10.0
shape_change_coefficient = 2 / 3
observation_times = np.arange(0, 1.7, 0.02)
angular_error = 0.1

station_ecef = frames.geodetic_wgs84_to_ecef(latitude, longitude, 0, degrees=True)
initial_position_ecef = frames.geodetic_wgs84_to_ecef(
    latitude + 0.01,
    longitude,
    initial_height,
    degrees=True,
)
# The radiant points back toward the source; the velocity points away from it.
radiant_ecef = frames.azel_to_ecef(
    latitude,
    longitude,
    az=radiant_azimuth,
    el=radiant_elevation,
    degrees=True,
)
velocity_direction_ecef = -radiant_ecef
model = ab.AlphaBeta2026(
    options=ab.AlphaBetaOptions(
        max_step_size=1e-2,
        max_time=5.0,
    ),
)


def generate_observations(parameters):
    """Return stacked azimuth and elevation observations in radians."""
    alpha, beta, entry_velocity = parameters
    result = model.run(
        ab.AlphaBetaInitialState(
            epoch=None,
            alpha=alpha,
            beta=beta,
            initial_height=initial_height,
            entry_velocity=entry_velocity,
            entry_angle=np.radians(radiant_elevation),
            shape_change_coefficient=shape_change_coefficient,
        )
    )
    if result.t[-1] < observation_times[-1]:
        raise ValueError(
            f"trajectory ended before the final observation time {result.t[-1]} < {observation_times[-1]}"
        )
    distance = np.interp(observation_times, result.t, result.distance)
    position_ecef = initial_position_ecef[:, None] + velocity_direction_ecef[:, None] * distance
    position_enu = frames.ecef_to_enu(
        latitude,
        longitude,
        position_ecef - station_ecef[:, None],
        degrees=True,
    )
    azelr = spherical.cart_to_sph(position_enu, degrees=True)
    return np.concatenate([azelr[0, :], azelr[1, :]]), result


parameter_names = ["alpha", "beta", "entry velocity [m/s]"]
nominal_parameters = np.array([100.0, 10.0, 30e3])
finite_difference_steps = np.array([1e-2, 1e-3, 1e-1])
nominal_observations, result = generate_observations(nominal_parameters)

jacobian = np.empty((len(observation_times) * 2, len(nominal_parameters)), dtype=np.float64)
for ind, dx in enumerate(finite_difference_steps):
    perturbed_parameters = nominal_parameters.copy()
    perturbed_parameters[ind] += dx
    perturbed_observations, _ = generate_observations(perturbed_parameters)
    jacobian[:, ind] = (perturbed_observations - nominal_observations) / dx

inv_observation_covariance = np.eye(nominal_observations.size) / angular_error**2

parameter_covariance = np.linalg.inv(jacobian.T @ inv_observation_covariance @ jacobian)
parameter_std = np.sqrt(np.diag(parameter_covariance))


np.set_printoptions(precision=6, suppress=False)
print(f"Assumed independent azimuth/elevation 1-sigma error: {angular_error} deg")
print("Parameter order:", parameter_names)
print("Linearized parameter covariance:\n", parameter_covariance)
print("Linearized parameter standard deviation:\n", parameter_std)

count = len(observation_times)

fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="tight")
fig.suptitle(
    f"alpha={nominal_parameters[0]}, beta={nominal_parameters[1]}, entry"
    f"velocity={nominal_parameters[2] * 1e-3}, "
    f"entry angle={radiant_elevation} deg"
)
axes[0, 0].plot(result.t, result.height * 1e-3)
axes[0, 0].set_ylabel("Height [km]")
axes[0, 0].set_xlabel("Time [s]")

axes[0, 1].plot(result.t, result.velocity * 1e-3)
axes[0, 1].set_ylabel("Velocity [km/s]")
axes[0, 1].set_xlabel("Time [s]")

axes[1, 0].plot(observation_times, nominal_observations[count:])
axes[1, 0].set_xlabel("Time [s]")
axes[1, 0].set_ylabel("Elevation [deg]")

axes[1, 1].plot(
    result.velocity * 1e-3,
    result.height * 1e-3,
    label="Time dependant solution",
)
axes[1, 1].set_ylabel("Height [km]")
axes[1, 1].set_xlabel("Velocity [km/s]")
axes[1, 1].legend()

axes[0, 2].plot(result.massloss, result.height * 1e-3)
axes[0, 2].set_ylabel("Height [km]")
axes[0, 2].set_xlabel("Relative massloss [1/s]")

axes[1, 2].plot(observation_times, nominal_observations[:count])
axes[1, 2].set_ylabel("Azimuth [deg]")
axes[1, 2].set_xlabel("Time [s]")

plt.show()
