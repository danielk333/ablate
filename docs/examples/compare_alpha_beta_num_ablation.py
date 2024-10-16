"""
Compare analytical and numerical
================================

Docstring for this example
"""

import numpy as np
import matplotlib.pyplot as plt

from ablate.physics import alpha_beta
import ablate


def analyse_scenario(radius, initial_velocity, material_name, integrate_options):

    material = ablate.material.get(material_name)

    cross_section = np.pi * radius**2
    mass = material["rho_m"] * 4 / 3 * np.pi * radius**3
    ze = 75.7744
    cd = 0.47
    Lambda = 0.9
    mu = 2 / 3
    shape_factor = 1.21
    h0 = 7610.0  # at T = 260 K
    sea_level_rho = 1.225

    print(f"{radius=:1.3e} {cross_section=:1.3e} {mass=:1.3e}")

    alpha = alpha_beta.alpha_direct(
        aerodynamic_cd=cd,
        sea_level_rho=sea_level_rho,
        atmospheric_scale_height=h0,
        initial_cross_section=cross_section,
        initial_mass=mass,
        radiant_local_elevation=90 - ze,
        degrees=True,
    )
    beta = alpha_beta.beta_direct(
        shape_change_coefficient=mu,
        heat_exchange_coefficient=Lambda,
        initial_velocity=initial_velocity,
        aerodynamic_cd=cd,
        enthalpy_of_massloss=material["L"],
    )
    print(f"direct {alpha=:.2e} {beta=:.2e} calculation")

    vel_lims = [
        alpha_beta.velocity_direct(1e-6, beta, initial_velocity, shape_change_coefficient=mu),
        alpha_beta.velocity_direct(1 - 1e-6, beta, initial_velocity, shape_change_coefficient=mu),
    ]
    print(f"{vel_lims=}")
    velocities_vec = np.linspace(vel_lims[0], vel_lims[1], 1000)

    mass0 = alpha_beta.initial_mass_direct(
        alpha,
        aerodynamic_cd=cd,
        sea_level_rho=sea_level_rho,
        atmospheric_scale_height=h0,
        radiant_local_elevation=90 - ze,
        bulk_density=material["rho_m"],
        shape_factor=shape_factor,
        degrees=True,
    )
    masses = alpha_beta.mass_direct(
        velocity=velocities_vec,
        initial_mass=mass0,
        beta=beta,
        shape_change_coefficient=mu,
        initial_velocity=initial_velocity,
    )
    heights = alpha_beta.height_direct(
        velocity=velocities_vec,
        atmospheric_scale_height=h0,
        alpha=alpha,
        beta=beta,
        initial_velocity=initial_velocity,
    )

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

    sim_heights = result.altitude.values
    sim_vels = result.velocity.values

    ab_density = alpha_beta.atmosphere_density(sim_heights, h0, sea_level_rho)
    msis_density = atmosphere.density(
        time=np.datetime64("2018-06-28T12:45:33"),
        lat=69.5866115,
        lon=19.221555,
        alt=sim_heights,
    )["Total"].values.squeeze()

    rescaled_sim_heights = alpha_beta.rescale_hight(msis_density, h0, sea_level_rho)

    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Atmospheric model differences - rescaling")
    axes[0].plot(sim_heights * 1e-3, rescaled_sim_heights * 1e-3)
    axes[0].set_xlabel("Heights [km]")
    axes[0].set_ylabel("Rescaled heights [km]")
    axes[0].grid("on")

    axes[1].semilogx(ab_density, sim_heights * 1e-3, "-b", label="Isothermal approximation")
    axes[1].semilogx(msis_density, sim_heights * 1e-3, "-g", label="MSIS V2.1")
    axes[1].set_ylabel("Heights [km]")
    axes[1].set_xlabel("Atmospheric mass density [kg/m^3]")
    axes[1].legend()

    sim_vels_samples = sim_vels[::4]
    sim_h_samples = sim_heights[::4]
    sim_rescaled_h_samples = rescaled_sim_heights[::4]

    alpha_est, beta_est = alpha_beta.solve_alpha_beta_versionQ4(
        sim_vels_samples,
        sim_h_samples,
        initial_velocity=sim_vels.max(),
        atmospheric_scale_height=h0,
        start=None,
        bounds=((0.001, 10000.0), (0.00001, 500.0)),
    )
    alpha_scaled_est, beta_scaled_est = alpha_beta.solve_alpha_beta_versionQ4(
        sim_vels_samples,
        sim_rescaled_h_samples,
        initial_velocity=sim_vels.max(),
        atmospheric_scale_height=h0,
        start=None,
        bounds=((0.001, 10000.0), (0.00001, 500.0)),
    )
    print(f"{alpha_est=:.2e} {beta_est=:.2e}")
    print(f"{alpha_scaled_est=:.2e} {beta_scaled_est=:.2e}")

    # vnorm = sim_vels_samples / sim_vels_samples.max()
    # norm_v = np.arange(vnorm.min(), 1, 0.00005)
    # alpha_beta_est_velocities = norm_v * initial_velocity
    alpha_beta_est_velocities = sim_vels_samples

    mass0_vec = []
    masses_vec = []
    heights_vec = []

    for alpha_v, beta_v in zip(
        [alpha_est, alpha_scaled_est],
        [beta_est, beta_scaled_est],
    ):
        _mass0 = alpha_beta.initial_mass_direct(
            alpha_v,
            aerodynamic_cd=cd,
            sea_level_rho=sea_level_rho,
            atmospheric_scale_height=h0,
            radiant_local_elevation=90 - ze,
            bulk_density=material["rho_m"],
            shape_factor=shape_factor,
            degrees=True,
        )
        _masses = alpha_beta.mass_direct(
            velocity=alpha_beta_est_velocities,
            initial_mass=_mass0,
            beta=beta_v,
            shape_change_coefficient=mu,
            initial_velocity=initial_velocity,
        )
        _heights = alpha_beta.height_direct(
            velocity=alpha_beta_est_velocities,
            atmospheric_scale_height=h0,
            alpha=alpha_v,
            beta=beta_v,
            initial_velocity=initial_velocity,
        )
        mass0_vec.append(_mass0)
        masses_vec.append(_masses)
        heights_vec.append(_heights)

    mass0_est, mass0_scaled_est = mass0_vec
    masses_est, masses_scaled_est = masses_vec
    heights_est, heights_scaled_est = heights_vec

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f"'{material_name} {radius=}' Simulation vs analytical solution - direct & fit")

    ax = axes[0, 0]
    ax.plot(np.log10(result.mass), sim_heights * 1e-3, c="b", label=f"KeroSzasz2008 {mass=:.2e}")
    ax.plot(
        np.log10(masses),
        heights * 1e-3,
        c="g",
        label=f"direct (alpha, beta, mass)=({alpha:.2f}, {beta:.2f}, {mass0:.2e})",
    )
    ax.plot(
        np.log10(masses_est),
        heights_est * 1e-3,
        c="r",
        label=f"estimate (alpha, beta, mass)=({alpha_est:.2f}, {beta_est:.2f}, {mass0_est:.2e})",
    )
    ax.set_xlabel("Mass [log$_{10}$(kg)]")
    ax.set_ylabel("Height [km]")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(result.velocity * 1e-3, sim_heights * 1e-3, c="b")
    ax.plot(velocities_vec * 1e-3, heights * 1e-3, c="g")
    ax.plot(alpha_beta_est_velocities * 1e-3, heights_est * 1e-3, c="r")
    ax.set_xlabel("Velocity [km/s]")
    ax.set_ylabel("Height [km]")

    massloss_sim = np.diff(result.mass.values) / np.diff(sim_heights)
    massloss = np.diff(masses) / np.diff(heights)
    massloss_est = np.diff(masses_est) / np.diff(heights_est)
    massloss_scaled_est = np.diff(masses_scaled_est) / np.diff(heights_scaled_est)

    ax = axes[0, 1]
    ax.plot(
        massloss_sim / np.nanmax(massloss_sim),
        sim_heights[:-1] * 1e-3,
        c="b",
    )
    ax.plot(
        massloss / np.nanmax(massloss),
        heights[1:] * 1e-3,
        c="g",
    )
    ax.plot(
        massloss_est / np.nanmax(massloss_est),
        heights_est[1:] * 1e-3,
        c="r",
    )
    ax.set_xlabel("Normalized mass loss [1]")
    ax.set_ylabel("Height [km]")

    ax = axes[1, 1]
    ax.plot(result.temperature, result.altitude * 1e-3)
    ax.set_xlabel("Meteoroid temperature [K]")
    ax.set_ylabel("Height [km]")

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f"'{material_name} {radius=}' Simulation vs analytical solution - scaled fit")

    ax = axes[0, 0]
    ax.plot(np.log10(result.mass), sim_heights * 1e-3, c="b", label=f"KeroSzasz2008 {mass=:.2e}")
    ax.plot(
        np.log10(result.mass),
        rescaled_sim_heights * 1e-3,
        ls="--",
        c="b",
        label="rescaled KeroSzasz2008",
    )
    ax.plot(
        np.log10(masses_scaled_est),
        heights_scaled_est * 1e-3,
        ls="--",
        c="r",
        label=f"scaled estimate (alpha, beta, mass)=({alpha_est:.2f}, {beta_est:.2f}, {mass0_est:.2e})",
    )
    ax.set_xlabel("Mass [log$_{10}$(kg)]")
    ax.set_ylabel("Height [km]")
    ax.legend()

    dh = heights_scaled_est - sim_rescaled_h_samples
    h_resid = np.std(dh[np.logical_not(np.logical_or(np.isnan(dh), np.isinf(dh)))])

    ax = axes[1, 0]
    ax.plot(result.velocity * 1e-3, sim_heights * 1e-3, c="b")
    ax.plot(result.velocity * 1e-3, rescaled_sim_heights * 1e-3, ls="--", c="b")
    ax.plot(
        alpha_beta_est_velocities * 1e-3,
        heights_scaled_est * 1e-3,
        ls="--",
        c="r",
        label=f"std-err H={h_resid * 1e-3:.4f} km",
    )
    ax.set_xlabel("Velocity [km/s]")
    ax.set_ylabel("Height [km]")
    ax.legend()

    ax2 = ax.twinx()
    ax2.set_ylabel("Residual [km]", color="g")
    ax2.plot(alpha_beta_est_velocities * 1e-3, dh * 1e-3, c="g")
    ax2.tick_params(axis='y', labelcolor="g")

    massloss_scaled_sim = np.diff(result.mass.values) / np.diff(rescaled_sim_heights)
    massloss_scaled_est = np.diff(masses_scaled_est) / np.diff(heights_scaled_est)

    ax = axes[0, 1]
    ax.plot(
        massloss_sim / np.nanmax(massloss_sim),
        sim_heights[:-1] * 1e-3,
        c="b",
    )
    ax.plot(
        massloss_scaled_sim / np.nanmax(massloss_scaled_sim),
        rescaled_sim_heights[:-1] * 1e-3,
        ls="--",
        c="b",
    )
    ax.plot(
        massloss_scaled_est / np.nanmax(massloss_scaled_est),
        heights_scaled_est[1:] * 1e-3,
        ls="--",
        c="r",
    )
    ax.set_xlabel("Normalized mass loss [1]")
    ax.set_ylabel("Height [km]")

    ax = axes[1, 1]
    ax.plot(result.temperature, sim_heights * 1e-3, c="b")
    ax.plot(result.temperature, rescaled_sim_heights * 1e-3, ls="--", c="b")
    ax.set_xlabel("Meteoroid temperature [K]")
    ax.set_ylabel("Height [km]")

    return


# analyse_scenario(
#     radius = 8e-4,
#     initial_velocity = 20e3,
#     material_name="cometary",
#     integrate_options={
#         "minimum_mass_kg": 1e-11,
#         "max_step_size_sec": 1e-3,
#     }
# )
# analyse_scenario(
#     radius = 8e-4,
#     initial_velocity = 60e3,
#     material_name="cometary",
#     integrate_options={
#         "minimum_mass_kg": 1e-11,
#         "max_step_size_sec": 1e-3,
#     }
# )

analyse_scenario(
    radius=8e-4,
    initial_velocity=20e3,
    material_name="asteroidal",
    integrate_options={
        "minimum_mass_kg": 1e-11,
        "max_step_size_sec": 1e-3,
    },
)
analyse_scenario(
    radius=8e-4,
    initial_velocity=60e3,
    material_name="asteroidal",
    integrate_options={
        "minimum_mass_kg": 1e-11,
        "max_step_size_sec": 1e-3,
    },
)

# analyse_scenario(
#     radius = 8e-4,
#     initial_velocity = 60e3,
#     material_name="iron",
#     integrate_options={
#         "minimum_mass_kg": 1e-11,
#         "max_step_size_sec": 1e-3,
#     }
# )
# analyse_scenario(
#     radius = 2e-2,
#     initial_velocity = 60e3,
#     material_name="asteroidal",
#     integrate_options={
#         "minimum_mass_kg": 1e-6,
#         "max_step_size_sec": 1e-3,
#     }
# )


plt.show()
