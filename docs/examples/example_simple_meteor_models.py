import numpy as np
import matplotlib.pyplot as plt
from metablate.models import simple

opts = simple.LinearConstantOptions(
    evaluation_times=np.linspace(0, 1, num=100),
)
lin_model = simple.LinearConstantVelocity(opts)
lin_model_accel = simple.LinearConstantAcceleration(opts)

init_state_lin = np.array([0, 0, 100e3, 0, 20e3, -30e3])
init_state_acc = np.array([0, 0, 100e3, 0, 20e3, -30e3, -3e3])

output_lin = lin_model.run(init_state_lin)
output_acc = lin_model_accel.run(init_state_acc)

print(f"{output_lin.runtime=} s")
print(f"{output_acc.runtime=} s")

fig, axes = plt.subplots(2, 1, layout="tight")
axes[0].plot(
    opts.evaluation_times,
    output_lin.position[2, :] * 1e-3,
    label="Linear constant velocity",
)
axes[0].plot(
    opts.evaluation_times,
    output_acc.position[2, :] * 1e-3,
    label="Linear constant acceleration",
)
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Height [km]")
axes[0].legend()
axes[1].plot(
    opts.evaluation_times,
    np.linalg.norm(output_lin.velocity, axis=0) * 1e-3,
)
axes[1].plot(
    opts.evaluation_times,
    np.linalg.norm(output_acc.velocity, axis=0) * 1e-3,
)
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Velocity [km]")

plt.show()
