"""
Kero-Szasz ablation sweep
=========================

Bare-bones reproduction of a 2x2 ablation summary figure using the built-in
pymsis atmosphere interface.
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm

from spacecoords import frames
import metablate
import metablate.models.kero_szasz_2008 as ks


TIME = np.datetime64("2018-06-28T12:45:33", "ns")
LAT = 69.30
LON = 16.04
REFERENCE_ALTITUDE_M = 100e3
START_ALTITUDE_M = 170e3
AZIMUTH_DEG = 0.0
MASS_KG = 1e-8
VELOCITIES_KM_S = np.array([32.0, 53.0, 72.0])
ENTRY_ELEVATION_ANGLES_DEG = np.array([70.0, 45.0, 20.0])


def run_case(model, velocity_km_s, entry_elevation_angle_deg):
    reference_pos_ecef = frames.geodetic_wgs84_to_ecef(
        LAT,
        LON,
        REFERENCE_ALTITUDE_M,
        degrees=True,
    )
    velocity_ecef = -frames.azel_to_ecef(
        LAT,
        LON,
        az=AZIMUTH_DEG,
        el=entry_elevation_angle_deg,
        degrees=True,
    )
    velocity_ecef *= velocity_km_s * 1e3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = model.run(
            parameters=ks.KeroSzaszInitialState(
                epoch=TIME,
                position_ecef=reference_pos_ecef,
                velocity_ecef=velocity_ecef,
                mass=MASS_KG,
                drag_coefficient=1,
                heat_transfer_coefficient=1,
            ),
        )

    llh = frames.ecef_to_geodetic_wgs84(
        result.position_ecef[0, :],
        result.position_ecef[1, :],
        result.position_ecef[2, :],
    )
    altitude_km = llh[2, :] * 1e-3
    order = np.argsort(altitude_km)
    mass_loss_rate = np.abs(np.gradient(result.mass, result.t, edge_order=1))
    mass_loss_rate[~np.isfinite(mass_loss_rate)] = np.nan
    mass_loss_line_density = mass_loss_rate / result.velocity
    mass_loss_line_density[~np.isfinite(mass_loss_line_density)] = np.nan

    return {
        "altitude_km": altitude_km[order],
        "velocity_km_s": result.velocity[order] * 1e-3,
        "mass_loss_line_density_kg_m": mass_loss_line_density[order],
        "temperature_k": result.temperature[order],
        "mass_kg": result.mass[order],
    }


def make_figure():
    model = ks.KeroSzasz2008(
        options=ks.KeroSzaszOptions(
            material=metablate.material.cometary,
            sputtering=False,
            minimum_mass=1e-11,
            max_step_size=5e-3,
            max_time=5.0,
            integral_resolution=40,
            effective_atmospheric_temperature=280,
            start_altitude=START_ALTITUDE_M,
        ),
    )
    results = []

    pbar = tqdm(
        total=len(ENTRY_ELEVATION_ANGLES_DEG) * len(VELOCITIES_KM_S),
        desc="Iterating ablation model",
    )
    for entry_elevation_angle_deg in ENTRY_ELEVATION_ANGLES_DEG:
        for velocity_km_s in VELOCITIES_KM_S:
            results.append(
                {
                    "entry_elevation_angle_deg": entry_elevation_angle_deg,
                    "velocity_km_s": velocity_km_s,
                    "data": run_case(
                        model,
                        velocity_km_s,
                        entry_elevation_angle_deg,
                    ),
                }
            )
            pbar.update(1)
    pbar.close()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    angle_colors = {
        angle: colors[index % len(colors)] for index, angle in enumerate(ENTRY_ELEVATION_ANGLES_DEG)
    }
    velocity_linestyles = {32.0: "-", 53.0: "--", 72.0: ":"}

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharey=True, constrained_layout=True)
    axes = axes.ravel()

    plot_specs = [
        (0, "velocity_km_s", "Velocity [km s$^{-1}$]", "(a) Velocity", False),
        (
            1,
            "mass_loss_line_density_kg_m",
            r"Mass-loss line density, $|dm/dt|/v$ [kg m$^{-1}$]",
            "(b) Mass-loss line density",
            True,
        ),
        (2, "temperature_k", "Temperature [K]", "(c) Temperature", False),
        (3, "mass_kg", "Mass [kg]", "(d) Mass", True),
    ]

    for ax_index, key, xlabel, title, log_x in plot_specs:
        ax = axes[ax_index]
        for item in results:
            angle = float(item["entry_elevation_angle_deg"])
            velocity = float(item["velocity_km_s"])
            data = item["data"]
            ax.plot(
                data[key],
                data["altitude_km"],
                color=angle_colors[angle],
                linestyle=velocity_linestyles[velocity],
                lw=2.0,
            )
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_ylim(70, START_ALTITUDE_M * 1e-3)
        ax.grid(True, which="both", alpha=0.25)

    axes[1].set_xlim(1e-22, None)  # Go above floating point limit of diff
    axes[0].set_ylabel("Altitude [km]")
    axes[2].set_ylabel("Altitude [km]")

    angle_handles = [
        Line2D([0], [0], color=color, lw=2.0, label=rf"$\alpha={angle:.0f}^\circ$")
        for angle, color in angle_colors.items()
    ]
    velocity_handles = [
        Line2D(
            [0],
            [0],
            color="0.2",
            linestyle=linestyle,
            lw=2.0,
            label=f"{velocity:.0f} km s$^{{-1}}$",
        )
        for velocity, linestyle in velocity_linestyles.items()
    ]
    axes[0].legend(handles=angle_handles, frameon=False, loc="lower right", title="Entry elevation")
    axes[1].legend(handles=velocity_handles, frameon=False, loc="lower left", title="Velocity")

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        help="Save the figure to this path instead of opening a window.",
    )
    args = parser.parse_args()

    fig = make_figure()
    if args.output:
        fig.savefig(args.output, dpi=180)
    else:
        plt.show()


if __name__ == "__main__":
    main()
