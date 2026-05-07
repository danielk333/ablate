import numpy as np
import matplotlib.pyplot as plt
import metablate

lin_model = metablate.models.simple.LinearConstantVelocity()
lin_model_accel = metablate.models.simple.LinearConstantAcceleration()

times = np.linspace(0, 1, num=100)
init_state_lin = np.array([0, 0, 100e3, 0, 20e3, -30e3])
init_state_acc = np.array([0, 0, 100e3, 0, 20e3, -30e3, -3e3])

output_lin = lin_model.run(times, init_state_lin)
output_acc = lin_model_accel.run(times, init_state_acc)

print(f"{output_lin.runtime=} s")
print(f"{output_acc.runtime=} s")

fig, axes = plt.subplots(2, 1)
axes[0].plot(
    times,
    output_lin.position[2, :],
)
axes[0].plot(
    times,
    output_acc.position[2, :],
)
axes[1].plot(
    times,
    np.linalg.norm(output_lin.velocity, axis=0),
)
axes[1].plot(
    times,
    np.linalg.norm(output_acc.velocity, axis=0),
)

plt.show()
