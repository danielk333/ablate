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

initial_position = np.array([R_earth + 100e3, 0, 0])
initial_velocity = np.array([-30e3, 20e3, 0])
init_state_lin = np.concatenate((initial_position, initial_velocity))
init_state_acc = np.concatenate((initial_position, initial_velocity, [-3e3]))

output_lin = lin_model.run(init_state_lin)
output_acc = lin_model_accel.run(init_state_acc)


print(f"{output_lin.runtime=} s")
print(f"{output_acc.runtime=} s")

outputs = [
    ("Linear constant velocity", output_lin),
    ("Linear along-track acceleration", output_acc),
]

fig, axes = plt.subplots(2, 1, layout="tight")
for label, output in outputs:
    altitude = np.linalg.norm(output.position, axis=0) - R_earth
    axes[0].plot(times, altitude * 1e-3, label=label)
    axes[1].plot(times, np.linalg.norm(output.velocity, axis=0) * 1e-3, label=label)

axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Altitude [km]")
axes[0].legend()
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Speed [km/s]")

plt.show()
