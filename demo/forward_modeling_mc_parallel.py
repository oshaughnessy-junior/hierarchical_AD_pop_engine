r"""Parallel plot demo — numpy vs JAX-B.1 vs JAX-B.2, side by side.

Three cascade implementations of the SAME physics at the SAME θ*,
n_seed, depletion, overlaid as raw histograms (no KDE — deep-cascade
peaks would be smoothed away):

  * numpy ``particle_mc``            — discrete K-rejection (oracle)
  * JAX ``particle_mc_jax``          — B.1 Bernoulli K-gate (round 3)
  * JAX ``particle_mc_jax_soft``     — B.2 deterministic soft occupancy gate

The eye is the instrument here.  If the three dN/dm and dN/dχ histograms
lie on top of each other (and the residual panels are flat noise), the
two JAX branches reproduce the discrete-K-rejection dynamics and B.2's
k_ratio-free + straight-through-free design costs nothing in fidelity.

Reuses the run helpers from ``forward_modeling_mc_vs_mc.py`` so the
cascade config + weighting conventions stay identical.

Run::

    python demo/forward_modeling_mc_parallel.py                # n_seed=20000
    python demo/forward_modeling_mc_parallel.py --n-seed 5000  # faster
    # writes demo/figures_<hash>/mc_parallel_{dN_dm,dN_dchi,2d_resid}.png
"""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse the validated runners (numpy + JAX with gate dispatch).
from forward_modeling_mc_vs_mc import (  # noqa: E402
    THETA_STAR, _run_numpy, _run_jax,
)

_SERIES = [
    # (label, color, linestyle)
    ("numpy discrete (oracle)", "C0", "-"),
    ("JAX B.1 Bernoulli", "C2", "--"),
    ("JAX B.2 soft", "C3", ":"),
]


def _hist_dN_dm(run, m_edges):
    h, _ = np.histogram(run["m_active"], bins=m_edges, weights=run["w_active"])
    return h / np.diff(m_edges)


def _hist_dN_dchi(run, chi_edges):
    h, _ = np.histogram(run["chi_active"], bins=chi_edges,
                        weights=run["w_active"])
    return h / np.diff(chi_edges)[0]


def _ks_vs_oracle(run, oracle, rng):
    """Weighted-resample KS of run.m_active vs the oracle's."""
    from scipy.stats import ks_2samp
    v, w = run["m_active"], run["w_active"]
    if v.size == 0 or w.sum() <= 0:
        return float("nan")
    n = min(v.size, oracle["m_active"].size, 5000)
    idx = rng.choice(v.size, size=n, p=w / w.sum(), replace=True)
    return ks_2samp(v[idx], oracle["m_active"][:n]).pvalue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-hash", default=None)
    parser.add_argument("--n-seed", type=int, default=20_000)
    parser.add_argument("--depletion-fraction", type=float, default=0.10)
    parser.add_argument("--jax-key-seed", type=int, default=2026)
    parser.add_argument("--np-rng-seed", type=int, default=20260518)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    rh = args.run_hash or secrets.token_hex(3)
    out_dir = Path(__file__).parent / f"figures_{rh}"
    out_dir.mkdir(exist_ok=True)
    print(f"Run hash: {rh}  →  demo/figures_{rh}/")
    print(f"cfg: n_seed={args.n_seed} depletion={args.depletion_fraction}")

    print("\nRunning numpy discrete K-rejection (oracle)...")
    runs = [_run_numpy(args.n_seed, args.depletion_fraction, args.np_rng_seed)]
    print("Running JAX B.1 (Bernoulli K-gate)...")
    runs.append(_run_jax(args.n_seed, args.depletion_fraction,
                         args.jax_key_seed, gate="bernoulli"))
    print("Running JAX B.2 (soft occupancy gate)...")
    runs.append(_run_jax(args.n_seed, args.depletion_fraction,
                         args.jax_key_seed, gate="soft"))

    rng = np.random.default_rng(2026)
    for (label, _, _), run in zip(_SERIES, runs):
        ks = _ks_vs_oracle(run, runs[0], rng) if run is not runs[0] else 1.0
        print(f"  {label:24s}: n_active={run['m_active'].size:6d}  "
              f"Σw={run['w_active'].sum():8.1f}  KS_p(vs oracle)={ks:.3f}")

    m_min, m_max = THETA_STAR["m_min"], THETA_STAR["m_max"]
    m_edges = np.geomspace(m_min, m_max * 2, 41)
    m_ctr = np.sqrt(m_edges[:-1] * m_edges[1:])
    chi_edges = np.linspace(0, 1, 51)
    chi_ctr = 0.5 * (chi_edges[:-1] + chi_edges[1:])

    # ---- Figure: 2 columns (dN/dm, dN/dχ), 2 rows (overlay, residual) ----
    fig, axes = plt.subplots(2, 2, figsize=(15, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    h_m = [_hist_dN_dm(r, m_edges) for r in runs]
    h_c = [_hist_dN_dchi(r, chi_edges) for r in runs]

    # dN/dm overlay
    ax = axes[0, 0]
    for (label, c, ls), h in zip(_SERIES, h_m):
        ax.loglog(m_ctr, np.maximum(h, 1e-30), drawstyle="steps-mid",
                  color=c, ls=ls, lw=2.0, label=label)
    ax.set_ylabel(r"$dN/dm$ (per $M_\odot$)")
    ax.set_title("Surviving pool $dN/dm$ — raw histograms")
    ax.set_ylim(bottom=max(hh.max() for hh in h_m) * 1e-5)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize="small")
    # residual vs oracle
    ax = axes[1, 0]
    for (label, c, ls), h in zip(_SERIES[1:], h_m[1:]):
        resid = (h - h_m[0]) / np.maximum(h_m[0], 1e-30)
        ax.semilogx(m_ctr, resid, drawstyle="steps-mid", color=c, ls=ls,
                    lw=1.6, label=f"({label} − oracle)/oracle")
    ax.axhline(0, color="C0", lw=1.0)
    ax.set_xlabel(r"$m\;[M_\odot]$")
    ax.set_ylabel("frac. resid.")
    ax.set_ylim(-1, 1)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize="x-small")

    # dN/dχ overlay
    ax = axes[0, 1]
    for (label, c, ls), h in zip(_SERIES, h_c):
        ax.semilogy(chi_ctr, np.maximum(h, 1e-30), drawstyle="steps-mid",
                    color=c, ls=ls, lw=2.0, label=label)
    ax.axvline(0.6865, color="0.4", lw=0.8, ls=":",
               label=r"$\chi_{\rm rem}(1,0)$")
    ax.set_ylabel(r"$dN/d\chi$")
    ax.set_title(r"Surviving pool $dN/d\chi$ — raw histograms")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=max(hh.max() for hh in h_c) * 1e-4)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize="small")
    # residual vs oracle
    ax = axes[1, 1]
    for (label, c, ls), h in zip(_SERIES[1:], h_c[1:]):
        resid = (h - h_c[0]) / np.maximum(h_c[0], 1e-30)
        ax.plot(chi_ctr, resid, drawstyle="steps-mid", color=c, ls=ls,
                lw=1.6, label=f"({label} − oracle)/oracle")
    ax.axhline(0, color="C0", lw=1.0)
    ax.set_xlabel(r"single-BH spin $\chi$")
    ax.set_ylabel("frac. resid.")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize="x-small")

    fig.suptitle(
        f"Parallel MC: numpy vs B.1 vs B.2 (n_seed={args.n_seed}, "
        f"depletion={args.depletion_fraction}) — overlay + residual vs oracle"
    )
    fig.tight_layout()
    out = out_dir / "mc_parallel_marginals.png"
    fig.savefig(out, dpi=140)
    print(f"\n  wrote {out}")

    # ---- 2-D residual maps (B.1 − oracle, B.2 − oracle) ----
    log_m_edges = np.linspace(np.log(m_min), np.log(m_max), 41)
    m2_ctr = np.exp(0.5 * (log_m_edges[:-1] + log_m_edges[1:]))
    chi2_ctr = 0.5 * (chi_edges[:-1] + chi_edges[1:])

    def _h2(run):
        H, _, _ = np.histogram2d(
            np.log(np.clip(run["m_active"], m_min, m_max)),
            run["chi_active"], bins=[log_m_edges, chi_edges],
            weights=run["w_active"],
        )
        return H

    H = [_h2(r) for r in runs]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, Hk, (label, _, _) in zip(axes, H[1:], _SERIES[1:]):
        cmax = max(np.abs(Hk - H[0]).max(), 1e-3)
        im = ax.pcolormesh(m2_ctr, chi2_ctr, (Hk - H[0]).T,
                           shading="auto", cmap="seismic",
                           vmin=-cmax, vmax=cmax)
        ax.set_xscale("log")
        ax.set_xlim(m_min, m_max)
        ax.set_xlabel(r"$m\;[M_\odot]$")
        ax.set_title(f"{label} − numpy oracle  (signed counts)")
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    axes[0].set_ylabel(r"single-BH spin $\chi$")
    fig.suptitle("Parallel MC: 2-D (m, χ) residual vs numpy oracle "
                 "(flat ≈ noise ⇒ faithful)")
    fig.tight_layout()
    out = out_dir / "mc_parallel_2d_resid.png"
    fig.savefig(out, dpi=140)
    print(f"  wrote {out}")
    print("Done.")


if __name__ == "__main__":
    main()
