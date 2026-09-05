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
from matplotlib.ticker import MaxNLocator

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
    ap.add_argument("--figsize", default="7.0,5.8",
                    help="W,H in INCHES -- draw at the final printed size "
                         "so the LaTeX include needs no scaling (revtex4-2 "
                         "\\textwidth = 7.06 in, \\columnwidth = 3.40 in)")
    args = ap.parse_args()

    labels = [s.strip() for s in args.labels.split(",")]
    runs = [_load(p) for p in args.npz]
    names, truth = runs[0][0], runs[0][1]
    P = names
    n = len(P)

    # PRINT SIZING (ROS 2026-09-05: "infinitesimal rendered font size as
    # rendered, unreadable ... lots of whitespace on the page wasted").  The
    # old figure was 11.4 in wide and went into the page at 0.62\textwidth
    # (4.4 in), a 0.39x reduction that turned 8 pt labels into 3 pt.  The fix
    # is to draw AT the final printed size -- default 7.0 in = revtex4-2
    # \textwidth, so the include is width=\textwidth and the scale factor is
    # 1.0 -- and to spend the recovered area on the panels rather than on
    # inter-panel padding: shared axes per column/row, three ticks per axis,
    # no suptitle (the caption carries it).
    W, H = (float(v) for v in args.figsize.split(","))
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })
    fig, axes = plt.subplots(n, n, figsize=(W, H), sharex="col")
    # y is shared ACROSS a row only among the scatter panels; the diagonal
    # carries a density and must keep its own scale.
    for i in range(1, n):
        for j in range(1, i):
            axes[i, j].sharey(axes[i, 0])
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                for r, (_, _, samp) in enumerate(runs):
                    ax.hist(samp[P[i]], bins=32, color=COLORS[r % 2], alpha=0.45,
                            density=True, label=labels[r] if r < len(labels) else None)
                ax.axvline(truth[P[i]], color="k", ls="--", lw=1.0)
                ax.set_yticks([])
            elif j < i:
                for r, (_, _, samp) in enumerate(runs):
                    ax.plot(samp[P[j]], samp[P[i]], ".", ms=1.1, alpha=0.18,
                            color=COLORS[r % 2], rasterized=True)
                ax.axvline(truth[P[j]], color="k", ls="--", lw=0.7)
                ax.axhline(truth[P[i]], color="k", ls="--", lw=0.7)
            else:
                ax.axis("off")
                continue
            ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
            if j < i:
                ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
            if i == n - 1:
                ax.set_xlabel(LABELS.get(P[j], P[j]))
                for t in ax.get_xticklabels():
                    t.set_rotation(45)
                    t.set_horizontalalignment("right")
            if j == 0 and i > 0:
                ax.set_ylabel(LABELS.get(P[i], P[i]))
            elif j > 0 and j < i:
                ax.tick_params(labelleft=False)
    # legend in the empty upper-right triangle
    handles = [plt.Line2D([0], [0], color=COLORS[r % 2], lw=6, alpha=0.5)
               for r in range(len(runs))]
    handles.append(plt.Line2D([0], [0], color="k", ls="--", lw=1.0))
    leglabels = labels[:len(runs)] + ["truth"]
    # figure-level legend so the entries are not clipped by the single
    # (narrow) upper-right cell they used to be drawn inside
    fig.legend(handles, leglabels, loc="upper right",
               bbox_to_anchor=(0.995, 0.985), frameon=False,
               handlelength=1.6, borderaxespad=0.0)
    fig.align_ylabels(axes[:, 0])
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.105, top=0.995,
                        wspace=0.09, hspace=0.09)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)
    print(f"wrote {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
