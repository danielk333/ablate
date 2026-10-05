"""Plot the normalized Dimant-Oppenheim electron density; archive arrays in HDF5.

Run: conda run -n base python docs/examples/plot_dimant_oppenheim_density.py --output-dir PATH
Requires h5py in addition to the core MetAblate dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import scipy

from metablate.models import dimant_oppenheim_2017 as dimant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=pathlib.Path("dimant_oppenheim_output")
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hide-provenance", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pathlib.Path(__file__).resolve()
    repo = source.parents[2]
    model = pathlib.Path(dimant.__file__).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    patch = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "HEAD", "--", str(model)], text=True
    )

    # Cell centers avoid R=0. Reflect the positive half-plane using axial symmetry.
    dx = 0.05
    x = np.arange(-4 + dx / 2, 12, dx)
    y_positive = np.arange(dx / 2, 6, dx)
    _, _, radius, morphology = dimant.plasma_distribution_morphology_grid(
        x,
        y_positive,
        threads=args.threads,
        pbar=False,
    )
    normalized_positive = morphology / radius
    y = np.concatenate((-y_positive[::-1], y_positive))
    density = np.concatenate((normalized_positive[::-1], normalized_positive), axis=0)
    assert np.all(np.isfinite(density)) and np.all(density > 0)
    np.testing.assert_allclose(density, density[::-1], rtol=0, atol=0)

    r = np.geomspace(0.01, 100, 180)
    profiles = {}
    for name, gx, gy in [
        ("behind", r, np.array([0.0])),
        ("ahead", -r, np.array([0.0])),
        ("transverse", np.array([0.0]), r),
    ]:
        _, _, rr, f = dimant.plasma_distribution_morphology_grid(
            gx, gy, quad_abs_tol=1e-10, pbar=False
        )
        profiles[name] = (f / rr).ravel()
    near = r < 0.05
    far = r > 30
    near_slope = np.polyfit(np.log(r[near]), np.log(profiles["behind"][near]), 1)[0]
    far_slope = np.polyfit(np.log(r[far]), np.log(profiles["behind"][far]), 1)[0]
    assert abs(near_slope + 1) < 0.04
    assert abs(far_slope + 2) < 0.01

    # Compare selected off-axis points to a stricter independent run of the quadrature.
    validation = []
    for i, j in [(4, 65), (27, 110), (60, 210)]:
        _, _, rr, f = dimant.plasma_distribution_morphology_grid(
            x[j : j + 1],
            y_positive[i : i + 1],
            quad_abs_tol=1e-10,
            pbar=False,
        )
        reference = float((f / rr)[0, 0])
        relative_error = abs(normalized_positive[i, j] / reference - 1)
        assert relative_error < 1e-5
        validation.append([x[j], y_positive[i], reference, relative_error])

    stem = args.output_dir / "dimant_oppenheim_density"
    with h5py.File(stem.with_suffix(".h5"), "w") as h5:
        h5.attrs.update(
            {
                "model": "Dimant and Oppenheim (2017), Eqs. 45-48",
                "reference": "https://doi.org/10.1002/2017JA023963",
                "source_url": "https://arxiv.org/pdf/1609.00256",
                "normalization": "n_star=8*pi*r_M^2*n0*nA*(1+m/mA)*Gion/(sqrt(3)*lambda_T); ne/n_star=f/R_lambda",
                "coordinate_convention": "meteor velocity is along -x; x>0 is behind",
                "approximation": "quasineutral ne approximately ni; single-collision head plasma",
                "script": str(source),
                "git_revision": revision,
                "model_local_diff": patch,
                "script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                "quad_abs_tol": 1e-7,
                "upper_limit_delta": 0.0,
                "near_behind_log_slope": near_slope,
                "far_behind_log_slope": far_slope,
            }
        )
        for name, array in [
            ("x_over_lambda_T", x),
            ("y_over_lambda_T", y),
            ("electron_density_over_n_star", density),
            ("profile_R_over_lambda_T", r),
        ]:
            ds = h5.create_dataset(name, data=array, compression="gzip")
            ds.attrs["units"] = "dimensionless"
        group = h5.create_group("radial_profiles")
        for name, array in profiles.items():
            group.create_dataset(name, data=array)
        h5.create_dataset("quadrature_validation", data=validation).attrs["columns"] = (
            "x_lambda,y_lambda,reference_ne_over_nstar,relative_error"
        )

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )
    fig = plt.figure(figsize=(7.2, 8.8))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1, 0.80],
        width_ratios=[1, 0.04],
        left=0.13,
        right=0.84,
        top=0.89,
        bottom=0.24,
        hspace=0.34,
        wspace=0.08,
    )
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    image = ax.pcolormesh(
        x, y, density, norm=LogNorm(1e-4, 1e2), cmap="inferno", shading="nearest", rasterized=True
    )
    fig.colorbar(image, cax=cax, label=r"Electron density $n_e/n_*$")
    ax.contour(x, y, density, levels=[0.01, 0.1, 1], colors="white", linewidths=0.5, alpha=0.6)
    ax.plot(0, 0, "o", ms=4, mec="black", mfc="white")
    ax.annotate(
        "Meteor motion",
        xy=(-3.2, 4.6),
        xytext=(0.8, 4.6),
        color="white",
        arrowprops={"arrowstyle": "->", "color": "white"},
        ha="center",
        va="center",
    )
    ax.text(-3.6, -5.2, "Ahead", color="white")
    ax.text(8.0, -5.2, "Behind", color="white")
    ax.set(
        xlabel=r"Along-track position $x/\lambda_T$",
        ylabel=r"Transverse position $y/\lambda_T$",
        title="(a) Cross-section through the symmetry axis",
        xlim=(-4, 12),
        ylim=(-6, 6),
    )
    ax.set_aspect("equal")
    ax = fig.add_subplot(grid[1, 0])
    for name, color in [("behind", "#175b9b"), ("transverse", "#687075"), ("ahead", "#bb3e35")]:
        ax.loglog(r, profiles[name], color=color, lw=1.8, label=name.capitalize())
    guide_r = np.array([0.01, 0.15])
    ax.loglog(guide_r, profiles["behind"][0] * r[0] / guide_r, "k--", lw=1)
    ax.text(0.055, 85, r"$R^{-1}$")
    guide_r = np.array([8.0, 100.0])
    ax.loglog(guide_r, profiles["behind"][-1] * r[-1] ** 2 / guide_r**2, "k--", lw=1)
    ax.text(12, 0.075, r"$R^{-2}$")
    ax.set(
        xlabel=r"Distance from meteor $R/\lambda_T$",
        ylabel=r"Electron density $n_e/n_*$",
        title="(b) Radial density profiles",
        xlim=(0.01, 100),
        ylim=(1e-5, 1e3),
    )
    ax.legend(loc="lower left", frameon=False)
    ax.grid(which="major", alpha=0.15)
    fig.suptitle("Dimant-Oppenheim meteor electron density", y=0.97, fontsize=14)
    fig.text(
        0.5,
        0.925,
        r"Normalized head-plasma model: $n_e\simeq n_i$, distances in mean free paths",
        ha="center",
    )
    fig.text(
        0.13,
        0.133,
        r"$n_* = 8\pi r_M^2 n_0 n_A (1+m/m_A)G_{\rm ion}/(\sqrt{3}\,\lambda_T)$",
        fontsize=11,
    )
    fig.text(0.13, 0.093, "Dimant & Oppenheim (2017), doi:10.1002/2017JA023963", fontsize=11)
    if not args.hide_provenance:
        fig.text(0.13, 0.035, f"Script: {source.name}\nMetAblate: {revision[:12]}", fontsize=11)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"), dpi=220)
    plt.close(fig)
    print(
        f"Saved {stem}.[png,pdf,h5]; behind-axis slopes: near={near_slope:.4f}, far={far_slope:.4f}"
    )


if __name__ == "__main__":
    main()
