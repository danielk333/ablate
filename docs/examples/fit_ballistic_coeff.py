"""
Compare analytical and numerical
================================

Docstring for this example
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sci

from ablate.physics import ballistic
import ablate


def analyse_scenario(
    axes,
    radius,
    trajectory_sampling_edges,
    sample_delta_sec,
    initial_velocity,
    material_name,
    integrate_options,
):

    material = ablate.material.get(material_name)

    cross_section = np.pi * radius**2
    mass = material["rho_m"] * 4 / 3 * np.pi * radius**3
    ze = 75.7744
    cd = 0.47
    Lambda = 0.9
    shape_factor = 1.21
    cB = mass / (cd * cross_section)

    print(f"{radius=:1.3e} {cross_section=:1.3e} {mass=:1.3e}")
    atmosphere = ablate.atmosphere.AtmPymsis()

    model = ablate.KeroSzasz2008(
        atmosphere=atmosphere,
        config={
            "options": {
                "temperature0": 290,
                "shape_factor": shape_factor,
                "emissivity": 0.9,
                "sputtering": False,
                "Gamma": cd,
                "Lambda": Lambda,
            },
            "atmosphere": {
                "version": 2.1,
            },
            "integrate": dict(max_time_sec=100.0, method="RK45", **integrate_options),
        },
    )

    result = model.run(
        velocity0=initial_velocity,
        mass0=mass,
        altitude0=110e3,
        zenith_ang=ze,
        azimuth_ang=0.0,
        material_data=material,
        time=np.datetime64("2018-06-28T12:45:33"),
        lat=69.5866115,
        lon=19.221555,
        alt=100e3,
    )
    print(result)

    t = result.t.values
    t_tot = t.max() - t.min()
    tn = (t - t.min()) / t_tot
    sim_heights = result.altitude.values
    sim_vels = result.velocity.values

    height_generator = sci.interp1d(tn, sim_heights, kind="linear")
    velocity_generator = sci.interp1d(tn, sim_vels, kind="linear")

    for ind, edges in enumerate(trajectory_sampling_edges):
        tn_samp = np.arange(edges[0], edges[1], sample_delta_sec / t_tot)
        N_data = len(tn_samp)
        h_ms = height_generator(tn_samp)
        v_ms = velocity_generator(tn_samp)
        print(f"{ind=}: {edges=}")

        msis_density = atmosphere.density(
            time=np.datetime64("2018-06-28T12:45:33"),
            lat=69.5866115,
            lon=19.221555,
            alt=h_ms,
        )["Total"].values.squeeze()

        cB_est = ballistic.fit_velocity(
            h_ms,
            v_ms,
            msis_density,
            radiant_local_elevation=90 - ze,
            degrees=True,
        )
        fit_vels = ballistic.velocity(
            h_ms,
            msis_density,
            cB_est,
            radiant_local_elevation=90 - ze,
            initial_velocity=v_ms[0],
            degrees=True,
        )

        print(f"{cB=}")
        print(f"{cB_est=}")

        ax = axes[ind][0]
        if ind == 0:
            labels = [f"KeroSzasz2008 {cB=:.2f}", "Partial data", f"Fit {cB_est=:.2f}"]
        else:
            labels = [None, None, f"Fit {cB_est=:.2f}"]

        ax.set_title(f"{edges=} {N_data=}")
        ax.plot(sim_vels * 1e-3, sim_heights * 1e-3, "-b", alpha=0.5, label=labels[0])
        ax.plot(v_ms * 1e-3, h_ms * 1e-3, "xb", label=labels[1])
        ax.plot(fit_vels * 1e-3, h_ms[1:] * 1e-3, "--r", label=labels[2])
        ax.set_xlabel("Velocity [km/s]")
        ax.set_ylabel("Height [km]")
        ax.legend()

        ax = axes[ind][1]

        ax.set_title(f"{edges=} {N_data=}")
        ax.plot(t, sim_vels * 1e-3, "-b", alpha=0.5, label=labels[0])
        ax.plot(tn_samp * t_tot, v_ms * 1e-3, "xb", label=labels[1])
        ax.plot(tn_samp[1:] * t_tot, fit_vels * 1e-3, "--r", label=labels[2])
        ax.set_ylabel("Velocity [km/s]")
        ax.set_xlabel("Time [s]")
        ax.legend()

    return


fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))


analyse_scenario(
    list(zip(axes1.flatten(), axes2.flatten())),
    radius=8e-4,
    trajectory_sampling_edges=[
        (0, 1),
        (0.3, 0.6),
        (0.6, 0.7),
        (0.9, 1.0),
    ],
    sample_delta_sec=0.005,
    initial_velocity=60e3,
    material_name="asteroidal",
    integrate_options={
        "minimum_mass_kg": 1e-11,
        "max_step_size_sec": 1e-3,
    },
)

plt.show()
