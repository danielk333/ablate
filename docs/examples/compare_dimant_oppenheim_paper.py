"""Reproduce Figures 3 and 4 of Dimant & Oppenheim (2017)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from metablate.models import dimant_oppenheim_2017 as dimant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Save PNGs here instead of showing plots.")
    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    radius = np.linspace(0.01, 5, 100)
    theta = np.linspace(0, np.pi, 9)
    _, _, R, f = dimant.plasma_distribution_morphology_array(radius[None, :], theta[:, None], pbar=False)
    profiles = np.pi * f / R  # Figure 3's M normalization.

    z = np.linspace(-5, 20, 101)
    y = np.linspace(-10, 10, 81)
    _, _, R, f = dimant.plasma_distribution_morphology_grid(z, y, pbar=False)
    density = np.where(R <= 20, f / R, np.nan)  # ne/n_star, Eqs. 45-48.

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 12), layout="constrained")
    for i, profile in enumerate(profiles):
        axes[0].semilogy(radius, profile, "r--" if i in (0, theta.size - 1) else "k-", linewidth=1)
    axes[0].set(
        xlim=(0, 5),
        ylim=(0.005, 1000),
        xlabel=r"$R/\lambda_T$",
        ylabel=r"$M = \pi n_e/n_*$",
        title="MetAblate: Figure 3",
    )

    # Chosen display scale; the article specifies only relative units.
    im = axes[1].pcolormesh(
        z,
        y,
        8 * np.pi * density,
        cmap="turbo",
        norm=LogNorm(1e-3, 1e4),
        shading="auto",
        rasterized=True,
    )
    fig.colorbar(im, ax=axes[1], label=r"$8\pi n_e/n_*$ (relative units)")
    axes[1].axhline(0, color="skyblue", linestyle="--", linewidth=1)
    axes[1].plot(0, 0, "wo", markersize=4)
    axes[1].set(
        xlabel=r"$z/\lambda_T$",
        ylabel=r"$r/\lambda_T$",
        title="MetAblate: Figure 4",
        aspect="equal",
    )

    if args.output_dir is None:
        plt.show()
    else:
        fig.savefig(args.output_dir / "reproduced_do_figure3_4.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
