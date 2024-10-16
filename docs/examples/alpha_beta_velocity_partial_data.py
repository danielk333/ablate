import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sci

from ablate.physics import alpha_beta
import ablate


def analyse_scenario(
    axes,
    trajectory_sampling_edges,
    sample_delta_sec,
    radius,
    initial_velocity,
    material_name,
    integrate_options,
    rescale=False,
):

    material = ablate.material.get(material_name)

    cross_section = np.pi * radius**2
    mass = material["rho_m"] * 4 / 3 * np.pi * radius**3
    ze = 75.7744
    cd = 0.47
    Lambda = 0.9
    # mu = 2 / 3
    shape_factor = 1.21
    h0 = 7610.0  # at T = 260 K
    sea_level_rho = 1.225

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
    t_tot = (t.max() - t.min())
    tn = (t - t.min())/t_tot

    height_generator = sci.interp1d(tn, result.altitude.values, kind="linear")
    velocity_generator = sci.interp1d(tn, result.velocity.values, kind="linear")
    # mass_generator = sci.interp1d(tn, result.mass.values, kind="linear")

    sim_vels = result.velocity.values

    for ind, edges in enumerate(trajectory_sampling_edges):
        tn_samp = np.arange(edges[0], edges[1], sample_delta_sec/t_tot)
        N_data = len(tn_samp)
        h_ms_raw = height_generator(tn_samp)
        v_ms = velocity_generator(tn_samp)
        # m_ms = mass_generator(tn_samp)
        print(f"{ind=}: {edges=}")

        if rescale:
            msis_density = atmosphere.density(
                time=np.datetime64("2018-06-28T12:45:33"),
                lat=69.5866115,
                lon=19.221555,
                alt=h_ms_raw,
            )["Total"].values.squeeze()
            h_ms = alpha_beta.rescale_hight(msis_density, h0, sea_level_rho)
        else:
            h_ms = h_ms_raw

        alpha, beta, vel0 = alpha_beta.solve_alpha_beta_velocity_versionQ5(
            v_ms,
            h_ms,
            atmospheric_scale_height=h0,
            bounds=((0.01, 1000000.0), (10.0, 300.0), (0, None)),
            start=[None, 50.0, None],
            minimize_kwargs=dict(
                method="Nelder-Mead",
            )
        )
        print(f"{alpha=:.2e} {beta=:.2e} {vel0=:.2e}")

        mass0 = alpha_beta.initial_mass_direct(
            alpha, cd, sea_level_rho, h0, 90 - ze, material["rho_m"], shape_factor,
            degrees=True,
        )
        print(f"{mass0=:.2e}")
        # masses = alpha_beta.mass_direct(
        #     v_ms, mass0, beta, mu,
        #     initial_velocity=vel0,
        # )
        h_ab = alpha_beta.height_direct(
            v_ms, h0, alpha, beta,
            initial_velocity=vel0,
        )
        dh = h_ms - h_ab

        norm_v = np.arange(sim_vels.min()/vel0, 1, 0.00005)
        sample_vels = norm_v * vel0
        heights = alpha_beta.height_direct(
            sample_vels, h0, alpha, beta,
            initial_velocity=vel0,
        )
        ax = axes[ind]
        if ind == 0:
            labels = [
                "KeroSzasz2008",
                "Partial data",
                f"fit:\n{alpha=:.2f}\n{beta=:3f}\n{vel0=:.1e}\n{mass0=:.1e}",
            ]
        else:
            labels = [
                None,
                None,
                f"fit:\n{alpha=:.2f}\n{beta=:3f}\n{vel0=:.1e}\n{mass0=:.1e}",
            ]
        if rescale:
            msis_density = atmosphere.density(
                time=np.datetime64("2018-06-28T12:45:33"),
                lat=69.5866115,
                lon=19.221555,
                alt=result.altitude.values,
            )["Total"].values.squeeze()
            new_res_alt = alpha_beta.rescale_hight(msis_density, h0, sea_level_rho)
            ax.plot(result.velocity * 1e-3, new_res_alt * 1e-3, "--b", alpha=0.5, label=labels[0])
        else:
            ax.plot(result.velocity * 1e-3, result.altitude * 1e-3, "--b", alpha=0.5, label=labels[0])
        ax.plot(v_ms * 1e-3, h_ms * 1e-3, "xb", label=labels[1])
        ax.plot(sample_vels * 1e-3, heights * 1e-3, "-r", label=labels[2])
        ax.set_xlabel("Velocity [km/s]")
        ax.set_ylabel("Height [km]")
        ax.set_title(f"{edges=} {N_data=} {mass=:.1e} (Height {rescale=})")

        ax2 = ax.twinx()
        ax2.set_ylabel("Residual [km]", color="g")
        ax2.plot(v_ms * 1e-3, dh * 1e-3, c="g")
        ax2.tick_params(axis='y', labelcolor="g")

        ax.legend()


for rs in [True, False]:
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle("Partial data alpha-beta fitting")

    analyse_scenario(
        axes.flatten(),
        trajectory_sampling_edges=[
            (0, 1),
            (0.3, 1.0),
            (0.6, 0.8),
            (0.8, 1.0),
        ],
        sample_delta_sec=0.005,
        radius=8e-4,
        initial_velocity=60e3,
        material_name="asteroidal",
        integrate_options={
            "minimum_mass_kg": 1e-11,
            "max_step_size_sec": 1e-3,
        },
        rescale=rs,
    )

plt.show()
