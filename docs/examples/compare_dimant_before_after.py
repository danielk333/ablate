"""Plot the exact upstream and PR #3 density functions on identical grids.

Run from the repository: conda run -n base python docs/examples/compare_dimant_before_after.py
Requires h5py and Git history containing upstream commit c31ef4e.
The upstream source is loaded unchanged; the after source comes from this checkout.
"""
from pathlib import Path
import hashlib
import subprocess
import types
import warnings

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

repo = Path(__file__).resolve().parents[2]
out = Path("dimant_oppenheim_before_after")
out.mkdir(parents=True, exist_ok=True)
radius = np.linspace(.01, 5, 100)
theta = np.linspace(0, np.pi, 9)
z, y = np.linspace(-5, 20, 101), np.linspace(-10, 10, 81)
profiles, maps = [], []
commits = ["c31ef4e67c58d477391ec499ca8dd73976be5e3d", subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()]

with h5py.File(out / "before_after.h5", "w") as data:
    data.attrs["script"] = Path(__file__).name
    data.attrs["reference"] = "https://doi.org/10.1002/2017JA023963"
    for name, values in [("R_over_lambda_T", radius), ("theta", theta), ("z", z), ("y", y)]:
        data[name] = values
    for label, commit in zip(["before", "after"], commits):
        path = "src/metablate/models/dimant_oppenheim_2017.py"
        source = (subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{path}"])
                  if label == "before" else (repo / path).read_bytes())
        model = types.ModuleType(f"metablate.models.{label}")
        model.__package__ = "metablate.models"
        exec(compile(source, f"{label}_dimant_oppenheim.py", "exec"), model.__dict__)
        rows = np.empty((9, len(radius)))
        # Keep the original behavior unchanged; record its numerical warnings.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for i, angle in enumerate(theta):
                for j, r in enumerate(radius):
                    _, _, R, f = model.plasma_distribution_morphology_grid(
                        np.array([r * np.cos(angle)]), np.array([r * np.sin(angle)]), pbar=False)
                    rows[i, j] = (np.pi * f / R)[0, 0]
            _, _, R, f = model.plasma_distribution_morphology_grid(z, y, pbar=False)
            density = f / R
        density[R > 20] = np.nan
        profiles.append(rows)
        maps.append(density)
        group = data.create_group(label)
        group["figure3_normalized_density"] = rows
        group["ne_over_n_star"] = density
        group.attrs["commit"] = commit
        group.attrs["model_sha256"] = hashlib.sha256(source).hexdigest()
        group.attrs["warning_count"] = len(caught)
        group.attrs["warning_types"] = ", ".join(sorted({w.category.__name__ for w in caught}))
        print(f"{label}: {len(caught)} numerical warnings recorded")

plt.rcParams.update({"font.size": 12})
colors = plt.cm.viridis(np.linspace(0, 1, 9))
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
for ax, rows, label, commit in zip(axes[:2], profiles, ["Before", "After"], commits):
    for row, color in zip(rows, colors):
        ax.semilogy(radius, row, color=color)
    ax.set(xlim=(0, 5), ylim=(.005, 1000), xlabel=r"$R/\lambda_T$",
           ylabel=r"Figure 3: $\pi n_e/n_*$", title=f"{label} ({commit[:7]})")
for i, color in enumerate(colors):
    axes[2].plot(radius, 100 * (profiles[1][i] - profiles[0][i]) / profiles[1][i],
                 color=color, label=rf"$\theta={i}\pi/8$")
axes[2].set(xlim=(0, 5), xlabel=r"$R/\lambda_T$", ylabel="100 × (after − before) / after (%)",
            title="Change relative to corrected density")
axes[2].legend(fontsize=9, ncol=2)
fig.tight_layout(rect=(0, .07, 1, 1))
fig.text(.5, .025, f"Script: {Path(__file__).name}; nine angles from 0 to π.", ha="center")
fig.savefig(out / "figure3_before_after.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
for ax, density, label in zip(axes[:2], maps, ["Before", "After"]):
    im = ax.pcolormesh(z, y, density, norm=LogNorm(1e-4, 1e3), cmap="inferno", rasterized=True)
    fig.colorbar(im, ax=ax, label=r"$n_e/n_*$")
    ax.set_title(label)
change = 100 * (maps[1] - maps[0]) / maps[1]
im = axes[2].pcolormesh(z, y, change, cmap="coolwarm", vmin=-100, vmax=100, rasterized=True)
fig.colorbar(im, ax=axes[2], label="100 × (after − before) / after (%)")
axes[2].set_title("Relative change")
for ax in axes:
    ax.set(xlabel=r"$z/\lambda_T$", ylabel=r"$r/\lambda_T$", aspect="equal")
    ax.axhline(0, color="skyblue", linestyle="--", linewidth=.7)
fig.tight_layout(rect=(0, .07, 1, 1))
fig.text(.5, .025, f"Script: {Path(__file__).name}; identical density scale before and after.", ha="center")
fig.savefig(out / "figure4_before_after.png", dpi=180)
plt.close(fig)
