"""Figures: PSD fit, force-chain networks, equilibrium history."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle

from .psd import SieveCurve, grading_of


def plot_psd(curve: SieveCurve, d_mm: np.ndarray, d_min: float, d_max: float, path) -> None:
    sizes = np.geomspace(0.1, 1.3, 200)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogx(curve.size_mm, curve.passing, "o-", color="#1f4e79", label="ASTM C778 graded sand (mid-band)")
    ax.semilogx(sizes, grading_of(d_mm, sizes), color="#c55a11", lw=2, label=f"seed sample (n={len(d_mm)})")
    ax.axvspan(d_min, d_max, color="0.92", zorder=0, label=f"modelled range {d_min}-{d_max} mm ({d_max/d_min:.2f}x)")
    ax.set_xlabel("particle size (mm)")
    ax.set_ylabel("percent passing by mass (%)")
    ax.set_ylim(0, 102)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_force_chains(a, path, title: str, tiles: tuple[int, int] | None = None,
                      seed_size: tuple[float, float] | None = None, show_discs: bool = True) -> None:
    a.compute_forces(dt=0.0)
    c = a.contacts
    scale = 1e3  # m -> mm
    seg = np.stack([a.x[c.i], a.x[c.j]], axis=1) * scale
    f = np.hypot(c.fn, c.ft)
    lw = 0.2 + 2.2 * f / f.max()
    w = (a.box[1] - a.box[0]) * scale
    h = (a.box[3] - a.box[2]) * scale
    fig, ax = plt.subplots(figsize=(7, 7 * h / w))
    if show_discs:
        patches = [Circle(p * scale, rr * scale) for p, rr in zip(a.x, a.r)]
        ax.add_collection(PatchCollection(patches, facecolor="#e8e2d6", edgecolor="#b9ae9a", linewidth=0.15))
    ax.add_collection(LineCollection(seg, linewidths=lw, colors="#8b1a1a"))
    xl, xr, yb, yt = a.box * scale
    ax.plot([xl, xr, xr, xl, xl], [yb, yb, yt, yt, yb], color="k", lw=1)
    if tiles and seed_size:
        for k in range(1, tiles[0]):
            ax.axvline(xl + k * seed_size[0] * scale, color="#1f77b4", lw=0.8, ls="--")
        for k in range(1, tiles[1]):
            ax.axhline(yb + k * seed_size[1] * scale, color="#1f77b4", lw=0.8, ls="--")
    ax.set_aspect("equal")
    ax.set_xlim(xl - 0.02 * w, xr + 0.02 * w)
    ax.set_ylim(yb - 0.02 * h, yt + 0.02 * h)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ufr_history(seed_hist: list[dict], big_hist: list[dict], target: float, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    s = [h["step"] for h in seed_hist]
    axes[0].semilogy(s, [h["ufr_mean"] for h in seed_hist], color="#1f4e79")
    axes[0].axhline(target, color="#c55a11", ls="--", lw=1, label="target 1e-4 (0.01%)")
    axes[0].set_title("Seed (625 discs): generation", fontsize=10)
    axes[0].set_xlabel("timestep")
    axes[0].set_ylabel("unbalanced force ratio (mean)")
    axes[0].legend(frameon=False, fontsize=8)
    b = [h["step"] for h in big_hist]
    axes[1].semilogy(b, [h["ufr_mean"] for h in big_hist], "o-", ms=3, color="#1f4e79")
    axes[1].axhline(target, color="#c55a11", ls="--", lw=1)
    axes[1].set_title("Tiled specimen (10,000 discs): free relaxation", fontsize=10)
    axes[1].set_xlabel("timestep after tiling")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
