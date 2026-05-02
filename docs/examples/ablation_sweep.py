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

import metablate


TIME = np.datetime64("2018-06-28T12:45:33", "ns")
LAT = 69.30
LON = 16.04
REFERENCE_ALTITUDE_M = 100e3
START_ALTITUDE_M = 130e3
AZIMUTH_DEG = 0.0
MASS_KG = 1e-8
VELOCITIES_KM_S = np.array([32.0, 53.0, 72.0])
ENTRY_ELEVATION_ANGLES_DEG = np.array([70.0, 45.0, 20.0])


def build_model():
    return metablate.KeroSzasz2008(
        atmosphere=metablate.atmosphere.AtmPymsis(),
        config={
            "options": {
                "temperature0": 290,
                "shape_factor": 1.21,
                "emissivity": 0.9,
                "sputtering": False,
                "Gamma": 1.0,
                "Lambda": 1.0,
                "integral_resolution": 40,
            },
            "atmosphere": {
                "version": 2.1,
            },
            "integrate": {
                "minimum_mass_kg": MASS_KG * 1e-5,
                "max_step_size_sec": 5e-2,
                "max_time_sec": 5.0,
                "method": "RK45",
            },
        },
    )


def run_case(model, material_data, velocity_km_s, entry_elevation_angle_deg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = model.run(
            velocity0=velocity_km_s * 1e3,
            mass0=MASS_KG,
            altitude0=START_ALTITUDE_M,
            # The model parameter is named zenith_ang, but this implementation
            # passes it to the coordinate helper as elevation above the horizon.
            zenith_ang=entry_elevation_angle_deg,
            azimuth_ang=AZIMUTH_DEG,
            material_data=material_data,
            time=TIME,
            lat=LAT,
            lon=LON,
            alt=REFERENCE_ALTITUDE_M,
        )

    altitude_km = result.altitude.values * 1e-3
    order = np.argsort(altitude_km)
    mass_loss_rate = np.abs(np.gradient(result.mass.values, result.t, edge_order=1))
    mass_loss_rate[~np.isfinite(mass_loss_rate)] = np.nan
    mass_loss_line_density = mass_loss_rate / result.velocity.values
    mass_loss_line_density[~np.isfinite(mass_loss_line_density)] = np.nan

    return {
        "altitude_km": altitude_km[order],
        "velocity_km_s": result.velocity.values[order] * 1e-3,
        "mass_loss_line_density_kg_m": mass_loss_line_density[order],
        "temperature_k": result.temperature.values[order],
        "mass_kg": result.mass.values[order],
    }


def make_figure():
    model = build_model()
    material_data = metablate.material.get("cometary")
    results = []

    for entry_elevation_angle_deg in ENTRY_ELEVATION_ANGLES_DEG:
        for velocity_km_s in VELOCITIES_KM_S:
            results.append(
                {
                    "entry_elevation_angle_deg": entry_elevation_angle_deg,
                    "velocity_km_s": velocity_km_s,
                    "data": run_case(
                        model,
                        material_data,
                        velocity_km_s,
                        entry_elevation_angle_deg,
                    ),
                }
            )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    angle_colors = {
        angle: colors[index % len(colors)]
        for index, angle in enumerate(ENTRY_ELEVATION_ANGLES_DEG)
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
        ax.set_ylim(70, 130)
        ax.grid(True, which="both", alpha=0.25)

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
    axes[0].legend(handles=angle_handles, frameon=False, loc="upper left", title="Entry elevation")
    axes[1].legend(handles=velocity_handles, frameon=False, loc="lower right", title="Velocity")

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
