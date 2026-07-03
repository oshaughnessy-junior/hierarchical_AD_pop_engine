#!/usr/bin/env python3
r"""Overlay two Stage-1 posteriors (e.g. realistic-measurement vs exact-injection
delta) on one corner, truth marked. Reads the npz that posterior_stage1_gwkokab.py
saves (stage1_post_svi<tag>.npz). Pure plotting; no re-fit.

    python plot_stage1_overlay_corner.py A.npz B.npz --labels "measurement,exact injection" \
        --out figures/stage1_corner_overlay
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "mu_natal": r"$\mu_{\rm natal}\,[M_\odot]$",
    "sigma_natal": r"$\sigma_{\rm natal}\,[M_\odot]$",
    "natal_spin_loc": r"$\mu_{\chi,\rm natal}$",
    "natal_spin_scale": r"$\sigma_{\chi,\rm natal}$",
    "kernel_b": r"$b$ (kernel)",
    "log_rate": r"$\log R$",
}
COLORS = ["#1f5fbf", "#d1495b"]  # A (measurement) blue, B (delta) red


def _load(path):
    d = np.load(path, allow_pickle=True)
    names = [str(x) for x in d["param_names"]]
    truth = {n: float(t) for n, t in zip(names, d["theta_star"])}
    samp = {n: np.asarray(d[n]).ravel() for n in names}
    fin = np.all([np.isfinite(samp[n]) for n in names], axis=0)
    samp = {n: samp[n][fin] for n in names}
    return names, truth, samp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", help="1 or 2 stage1_post_svi*.npz")
    ap.add_argument("--labels", default="measurement,exact injection")
    ap.add_argument("--out", default="stage1_corner_overlay")
    args = ap.parse_args()

    labels = [s.strip() for s in args.labels.split(",")]
    runs = [_load(p) for p in args.npz]
    names, truth = runs[0][0], runs[0][1]
    P = names
    n = len(P)

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(n, n, figsize=(1.9 * n, 1.9 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                for r, (_, _, samp) in enumerate(runs):
                    ax.hist(samp[P[i]], bins=32, color=COLORS[r % 2], alpha=0.45,
                            density=True, label=labels[r] if r < len(labels) else None)
                ax.axvline(truth[P[i]], color="k", ls="--", lw=1.1)
                ax.set_yticks([])
            elif j < i:
                for r, (_, _, samp) in enumerate(runs):
                    ax.plot(samp[P[j]], samp[P[i]], ".", ms=1.3, alpha=0.20,
                            color=COLORS[r % 2])
                ax.axvline(truth[P[j]], color="k", ls="--", lw=0.7)
                ax.axhline(truth[P[i]], color="k", ls="--", lw=0.7)
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(LABELS.get(P[j], P[j]), fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(LABELS.get(P[i], P[i]), fontsize=8)
            else:
                if not (i == j):
                    ax.set_yticklabels([])
    # legend in the empty upper-right
    handles = [plt.Line2D([0], [0], color=COLORS[r % 2], lw=6, alpha=0.5)
               for r in range(len(runs))]
    handles.append(plt.Line2D([0], [0], color="k", ls="--", lw=1.1))
    leglabels = labels[:len(runs)] + ["truth"]
    axes[0, n - 1].legend(handles, leglabels, loc="center", fontsize=9, frameon=False)
    fig.suptitle("Single-pulse isotropic recovery (JAX MC): posterior overlay", y=0.995)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=150)
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
