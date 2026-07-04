#!/usr/bin/env python3
r"""Publication-quality re-plot of the forward-modelling-observables figure.

``forward_modeling_observables.py`` renders an 8-panel *diagnostic* strip that
is fine on screen but far too wide for a PRD column.  This script reads the
``cells_<res>.npz`` + ``metadata.json`` that run already writes and lays the
same content out as a compact **tiled grid** sized for a PRD full-width figure
(``\\figure*``), plus (optionally) the two 2-D joints as a companion figure.

Nothing here re-runs the engine or the MC — it is pure plotting off saved
arrays, so iterating on layout is instantaneous.  Run e.g.::

    python demo/plot_fmobs_publication.py \
        demo/artifacts_fmobs_gpu128_isotropic/cells_24x128.npz

Outputs ``fmobs_publication.{pdf,png}`` (tiled grid) next to the npz.

The engine curve is the implicit Picard solution; the MC curve is the
particle-cascade k-NN-balloon KDE.  For the *fractal* spin marginal (dN/dχ,
dN/dχ_eff) the two are expected to track structure, not to be smooth — the
resolution just sets how much of the self-similar tail is resolved.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

ENGINE_C = "#1f5fbf"   # implicit Picard engine — solid blue
MC_C = "#1a9850"       # particle MC k-NN KDE — dashed green
MC_RAW_C = "#9ecae1"   # MC raw histogram — faint

# Mass-axis range/ticks to match the GWTC-5 population paper Fig. 1
# (arXiv:2605.27226): log mass axis over ~3-100 Msun with plain decade labels.
MASS_LIM = (3.0, 100.0)
MASS_TICKS = (3, 5, 10, 30, 100)


def _set_log_mass_axis(ax, which="x"):
    """Apply the shared log mass-axis styling (range + plain decade ticks)."""
    set_scale = ax.set_xscale if which == "x" else ax.set_yscale
    set_lim = ax.set_xlim if which == "x" else ax.set_ylim
    axis = ax.xaxis if which == "x" else ax.yaxis
    set_scale("log")
    set_lim(*MASS_LIM)
    axis.set_major_locator(FixedLocator(MASS_TICKS))
    axis.set_minor_locator(NullLocator())
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    axis.set_major_formatter(fmt)


def _jensen_shannon(x_a, p_a, x_b, p_b, n=512):
    """JS divergence (nats) between two densities given on their own grids.

    Interpolates both onto a shared linear grid over the overlap, renormalises,
    and returns the symmetric JS.  Robust to the engine/MC grids differing.
    """
    lo = max(np.min(x_a), np.min(x_b))
    hi = min(np.max(x_a), np.max(x_b))
    if not (hi > lo):
        return float("nan")
    xs = np.linspace(lo, hi, n)
    pa = np.interp(xs, x_a, p_a, left=0.0, right=0.0).clip(min=0)
    pb = np.interp(xs, x_b, p_b, left=0.0, right=0.0).clip(min=0)
    if pa.sum() <= 0 or pb.sum() <= 0:
        return float("nan")
    pa /= np.trapezoid(pa, xs)
    pb /= np.trapezoid(pb, xs)
    m = 0.5 * (pa + pb)

    def _kl(p, q):
        mask = p > 0
        return np.trapezoid(np.where(mask, p * np.log(np.where(mask, p / q, 1.0)), 0.0), xs)

    return float(0.5 * _kl(pa, m) + 0.5 * _kl(pb, m))


def _line_panel(ax, d, key, *, xlabel, ylabel, title, logx, js_store=None):
    ex, ey = d[f"{key}_x"], d[f"{key}_eng"]
    kx, ky = d[f"{key}_mc_kde_x"], d[f"{key}_mc_kde"]
    rx, ry = d[f"{key}_mc_raw_x"], d[f"{key}_mc_raw"]
    ax.plot(rx, ry, drawstyle="steps-mid", color=MC_RAW_C, lw=0.8, alpha=0.7, zorder=1)
    ax.plot(kx, ky, color=MC_C, lw=1.6, ls="--", label="particle MC", zorder=3)
    ax.plot(ex, ey, color=ENGINE_C, lw=1.8, label="engine (implicit)", zorder=4)
    ax.set_yscale("log")
    if logx:
        # logx is only ever set for the mass panels (src_m, m1); style them to
        # match the GWTC-5 population paper Fig. 1 (log mass, 3-100 Msun).
        _set_log_mass_axis(ax, which="x")
    # Clip the log-y range to ~6 decades below the engine peak.  The fractal
    # spin marginals dip to numerical zero between self-similar spikes and the
    # MC raw-histogram has empty bins; without a floor those crush the visible
    # structure into an unreadable 10^-17 range.
    y_top = float(np.nanmax(ey[ey > 0])) if np.any(ey > 0) else 1.0
    ax.set_ylim(y_top * 1e-6, y_top * 4.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.4)
    js = _jensen_shannon(ex, ey, kx, ky)
    if np.isfinite(js):
        ax.text(0.03, 0.05, f"JS={js:.3f}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8))
    if js_store is not None:
        js_store[key] = js


def _joint_panel(ax, d, key, *, xlabel, ylabel, title):
    xe = d[f"{key}_edges_x"] if f"{key}_edges_x" in d else None
    # map key -> stored edge/center names
    x_c, y_c = d[f"{key}_x"], d[f"{key}_y"]
    eng, mc = d[f"{key}_eng"], d[f"{key}_mc"]
    x_edges = d["m1_edges"]
    y_edges = d["q_edges"] if key == "m1_q" else d["chi_eff_edges"]
    vmax = eng.max() if eng.max() > 0 else 1.0
    vmin = vmax * 1e-4   # fixed 4-decade dynamic range (numerical-zero cells clip to floor)
    disp = np.where(eng > vmin, eng, vmin)
    pcm = ax.pcolormesh(x_edges, y_edges, disp.T, norm=LogNorm(vmin=vmin, vmax=vmax),
                        cmap="viridis", shading="flat", rasterized=True)
    # MC as white contours on shared centres
    if np.any(mc > 0):
        levels = np.quantile(mc[mc > 0], [0.5, 0.75, 0.9, 0.97])
        levels = np.unique(levels)
        if levels.size >= 1:
            ax.contour(x_c, y_c, mc.T, levels=levels, colors="white",
                       linewidths=0.7, alpha=0.85)
    # x is always m1 [Msun] for the two joint panels this is called on; style
    # to match the GWTC-5 population paper Fig. 1 (log mass, 3-100 Msun).
    # y (q or chi_eff) is not a mass axis and is left untouched.
    _set_log_mass_axis(ax, which="x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    return pcm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="path to a cells_<res>.npz produced by fmobs")
    ap.add_argument("--out", default=None, help="output basename (default: alongside npz)")
    args = ap.parse_args()

    d = dict(np.load(args.npz, allow_pickle=True))
    meta_path = os.path.join(os.path.dirname(args.npz), "metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    pr = (meta.get("per_resolution") or [{}])[0]
    n_m = pr.get("n_m", "?")
    n_chi = pr.get("n_chi", "?")
    n_quad = meta.get("orientation_n_quad", "?")
    n_mc = meta.get("n_mc", "?")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.5))
    js = {}
    _line_panel(axes[0, 0], d, "src_m", xlabel=r"$m\ [M_\odot]$",
                ylabel=r"$dN/dm$", title="source-frame mass", logx=True, js_store=js)
    _line_panel(axes[0, 1], d, "src_chi", xlabel=r"$\chi$",
                ylabel=r"$dN/d\chi$", title="source-frame spin (fractal)", logx=False, js_store=js)
    _line_panel(axes[0, 2], d, "m1", xlabel=r"$m_1\ [M_\odot]$",
                ylabel=r"$dN/dm_1$", title="detector-frame $m_1$", logx=True, js_store=js)
    _line_panel(axes[1, 0], d, "chi_eff", xlabel=r"$\chi_{\rm eff}$",
                ylabel=r"$dN/d\chi_{\rm eff}$", title=r"effective spin", logx=False, js_store=js)
    p1 = _joint_panel(axes[1, 1], d, "m1_q", xlabel=r"$m_1\ [M_\odot]$",
                      ylabel=r"$q=m_2/m_1$", title=r"$dN/(dm_1\,dq)$")
    p2 = _joint_panel(axes[1, 2], d, "m1_chi_eff", xlabel=r"$m_1\ [M_\odot]$",
                      ylabel=r"$\chi_{\rm eff}$", title=r"$dN/(dm_1\,d\chi_{\rm eff})$")
    fig.colorbar(p1, ax=axes[1, 1], fraction=0.046, pad=0.04)
    fig.colorbar(p2, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # single shared legend from the line panels
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(
        rf"Forward-modelled observables: implicit engine vs particle MC "
        rf"(${n_m}\times{n_chi}$ grid, {n_quad}-pt isotropic quadrature, "
        rf"$N_{{\rm MC}}={n_mc}$)", y=1.05, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = args.out or os.path.join(os.path.dirname(args.npz), "fmobs_publication")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"wrote {out}.pdf / .png")
    print("JS(engine || MC) [nats]:")
    for k, v in js.items():
        print(f"  {k:10s} {v:.4f}")


if __name__ == "__main__":
    main()
