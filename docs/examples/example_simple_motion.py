"""
Simple meteor motion models
===========================
"""

import matplotlib.pyplot as plt
import numpy as np
from spacecoords.constants import R_earth

from metablate.models import simple_motion

times = np.linspace(0, 3, num=100)
opts = simple_motion.LinearConstantOptions(
    evaluation_times=times,
)
lin_model = simple_motion.LinearConstantVelocity(opts)
lin_model_accel = simple_motion.LinearConstantAcceleration(opts)
ballistic_model = simple_motion.Ballistic(
    simple_motion.BallisticOptions(
        evaluation_times=times,
        max_step_size=0.1,
    )
)

initial_position = np.array([R_earth + 100e3, 0, 0])
initial_velocity = np.array([-30e3, 20e3, 0])
init_state_lin = np.concatenate((initial_position, initial_velocity))
init_state_acc = np.concatenate((initial_position, initial_velocity, [-3e3]))
init_state_ballistic = simple_motion.BallisticInitialState(
    epoch=np.datetime64("2026-03-20T00:00:00"),
    position_ecef=initial_position,
    velocity_ecef=initial_velocity,
    ballistic_coefficient=100.0,
)

output_lin = lin_model.run(init_state_lin)
output_acc = lin_model_accel.run(init_state_acc)
output_ballistic = ballistic_model.run(init_state_ballistic)


print(f"{output_lin.runtime=} s")
print(f"{output_acc.runtime=} s")
print(f"{output_ballistic.runtime=} s")

outputs = [
    (
        "Linear constant velocity",
        times,
        output_lin.position,
        output_lin.velocity,
    ),
    (
        "Linear along-track acceleration",
        times,
        output_acc.position,
        output_acc.velocity,
    ),
    (
        "Ballistic drag and gravity",
        output_ballistic.t,
        output_ballistic.position_ecef,
        output_ballistic.velocity_ecef,
    ),
]

fig, axes = plt.subplots(2, 1, layout="tight")
for label, output_times, position, velocity in outputs:
    altitude = np.linalg.norm(position, axis=0) - R_earth
    axes[0].plot(output_times, altitude * 1e-3, label=label)
    axes[1].plot(output_times, np.linalg.norm(velocity, axis=0) * 1e-3, label=label)

axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Altitude [km]")
axes[0].legend()
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Speed [km/s]")

plt.show()
