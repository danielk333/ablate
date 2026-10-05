"""Compare MetAblate with Figures 3 and 4 of Dimant & Oppenheim (2017).

Run from any directory: python compare_dimant_oppenheim_paper.py
Requires h5py for HDF5 output. Source images are bundled under docs/assets.
"""

from pathlib import Path
import argparse

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
import numpy as np

from metablate.models import dimant_oppenheim_2017 as dimant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dimant_oppenheim_comparison"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    reference = Path(__file__).resolve().parents[1] / "assets" / "dimant_oppenheim"
    radius, theta = np.linspace(.01, 5, 100), np.linspace(0, np.pi, 9)
    profiles = np.empty((len(theta), len(radius)))
    for i, angle in enumerate(theta):
        for j, r in enumerate(radius):
            _, _, R, f = dimant.plasma_distribution_morphology_grid(
                np.array([r * np.cos(angle)]), np.array([r * np.sin(angle)]), pbar=False,
            )
            profiles[i, j] = (np.pi * f / R)[0, 0]  # Figure 3's M normalization.

    z, y = np.linspace(-5, 20, 101), np.linspace(-10, 10, 81)
    _, _, R, f = dimant.plasma_distribution_morphology_grid(z, y, pbar=False)
    density = f / R  # ne/n_star from Eqs. 45-48; not the morphology f alone.
    density[R > 20] = np.nan
    with h5py.File(out / "paper_comparison.h5", "w") as data:
        for name, values in [("R_over_lambda_T", radius), ("theta", theta),
                             ("figure3_normalized_density", profiles), ("z", z),
                             ("y", y), ("ne_over_n_star", density)]:
            data[name] = values
        data.attrs["script"] = "docs/examples/compare_dimant_oppenheim_paper.py"
        data.attrs["model"] = "metablate.models.dimant_oppenheim_2017"
        data.attrs["reference"] = "https://doi.org/10.1002/2017JA023963"
        data.attrs["figure3_scale"] = np.pi
        data.attrs["figure4_display_scale"] = 8 * np.pi

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(7.2, 6))
    for i, row in enumerate(profiles):
        ax.semilogy(radius, row, "r--" if i in (0, 8) else "k-", linewidth=1)
    ax.set(xlim=(0, 5), ylim=(.005, 1000), xlabel=r"$R/\lambda_T$", ylabel="Figure 3 normalized density", title="MetAblate: Figure 3")
    fig.subplots_adjust(bottom=.22)
    fig.text(.12, .06, "Script: compare_dimant_oppenheim_paper.py", fontsize=11)
    fig.savefig(out / "figure3.png", dpi=200)
    plt.close(fig)

    image = plt.imread(reference / "article_figure4.png")
    cmap = ListedColormap(image[143:1297, 2040, :3][::-1])
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.pcolormesh(z, y, 8 * np.pi * density, cmap=cmap, norm=LogNorm(1e-3, 1e4), rasterized=True)
    fig.colorbar(im, ax=ax, label=r"$8\pi n_e/n_*$ (relative units)")
    ax.axhline(0, color="skyblue", linestyle="--", linewidth=1)
    ax.plot(0, 0, "wo", markersize=4)
    ax.set(xlabel=r"$z/\lambda_T$", ylabel=r"$r/\lambda_T$", title="MetAblate: Figure 4", aspect="equal")
    fig.subplots_adjust(left=.12, right=.90, bottom=.22)
    fig.text(.12, .06, "Script: compare_dimant_oppenheim_paper.py", fontsize=11)
    fig.savefig(out / "figure4.png", dpi=200)
    plt.close(fig)

    for number in [3, 4]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, path, title in zip(axes, [reference / f"article_figure{number}.png", out / f"figure{number}.png"], [f"Article, Figure {number}", "MetAblate"]):
            ax.imshow(plt.imread(path))
            ax.set_title(title)
            ax.axis("off")
        fig.subplots_adjust(left=.02, right=.98, bottom=.12, top=.86)
        if number == 4:
            fig.text(.5, .025, "Figure 4: 8*pi*ne/n_star is our chosen display scale; the article specifies only relative units.", ha="center", fontsize=11)
        fig.savefig(out / f"comparison_figure{number}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
