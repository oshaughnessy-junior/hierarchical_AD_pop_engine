r"""Stage-1 single-pulse TRUTH in the (m1,q) and (m1,chi_eff) planes.

Companion to the injection-recovery figure (fig:single_pulse_recovery): instead
of a noisy N-event injection scatter, this renders the *underlying* true model —
the population the Stage-1 fit is trying to recover — as two 2-D observable
densities, so a talk audience can read off the expected structure directly.

It rebuilds the EXACT recovery truth model (same code path as
``posterior_stage1_gwkokab._setup``: ``build_truth_model_and_calibrate_rate`` →
``dist_fn(depletion_fraction=0.10, orientation=isotropic, solver=mc_jax)`` at
``theta_star`` with ``natal_spin_loc=0.05``), then draws a large pool of binary
events straight from the engine's own particle pool
(``fit.sample_pairs_from_particles`` — grid-free, so the discrete high-generation
remnant spins are represented exactly) and histograms them in:

    * ``dN/(dm_1 dq)``            q = m_2/m_1        (m_1 >= m_2, so q in (0,1])
    * ``dN/(dm_1 dchi_eff)``      chi_eff = (m_1 chi_1 + m_2 chi_2)/(m_1+m_2)

The chi_eff convention matches the engine observable (``forward_modeling_
observables.py``): mass-weighted mean of the [0,1] spin magnitudes, so chi_eff in
[0,1]. Pure forward model — no fitting, no MC overlay.

Run (from the repo, env.sh sourced):

    taskset -c 8-19 env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
        python3 demo/plot_stage1_truth_2d.py --n-seed 4000 --n-pairs 300000

Outputs ``demo/figures/stage1_truth_2d_panels.{png,pdf}``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_DEMO = Path(__file__).resolve().parent
_REPO = _DEMO.parent
for p in (str(_DEMO), str(_REPO), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import demo_gwkokab_fit_stage1 as demo  # noqa: E402
import fit_stage1_gwkokab as fit  # noqa: E402
from hierarchical_ad_pop_engine.grids import BinaryGrid  # noqa: E402
from hierarchical_ad_pop_engine.spin_orientation import Isotropic  # noqa: E402
from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (  # noqa: E402
    coagulation_poisson_mean_estimator,
    make_coagulation_dist_fn,
)

# Match the recovery's chi_eff / mass conventions (engine observable defn).
ENGINE_CMAP = "viridis"


def build_truth(args):
    """Reconstruct the Stage-1 recovery truth model verbatim, return its
    particle-resampled binary pairs (N,4) = [m1, m2, chi1, chi2]."""
    fixed = demo.FIXED
    n_m = args.n_m_fit or demo.N_M_FIT
    n_chi = args.n_chi_fit or demo.N_CHI_FIT
    fit_grid = BinaryGrid.default(m_min=fixed["m_min"], m_max=fixed["m_max"],
                                  n_m=n_m, n_chi=n_chi)
    # sets demo.THETA_STAR + a rate calibrated to n_obs (irrelevant to the
    # SHAPE we plot; we only use theta_star's physical params).
    demo.build_truth_model_and_calibrate_rate(args.n_obs, grid=fit_grid)
    demo.THETA_STAR["natal_spin_loc"] = 0.05
    theta_star = dict(demo.THETA_STAR)

    orientation = Isotropic.default(n_theta=args.n_theta, n_dphi=args.n_dphi)
    key = jax.random.PRNGKey(args.seed)
    k_mc, k_probe, k_pairs = jax.random.split(key, 3)
    dist_fn = make_coagulation_dist_fn(
        v_esc=fixed["v_esc"], m_min=fixed["m_min"], m_max=fixed["m_max"],
        kernel_a=fixed["kernel_a"], k_max=args.k_max, n_seed=args.n_seed,
        depletion_fraction=0.10, gate="soft", grid=fit_grid, mc_jax_key=k_mc,
        solver="mc_jax", orientation=orientation)

    # self-consistent log_rate (matches the recovery; only affects overall N).
    probe = dist_fn(**{**theta_star, "log_rate": 0.0})
    mu_unit, _ = coagulation_poisson_mean_estimator(probe, T_obs=1.0, n_quad=args.n_quad)
    theta_star["log_rate"] = float(jnp.log(args.n_obs / mu_unit))

    truth_model = dist_fn(**theta_star)
    pairs = fit.sample_pairs_from_particles(truth_model, args.n_pairs, k_pairs)
    return np.asarray(pairs), theta_star, fixed


def observables(pairs):
    m1, m2, c1, c2 = pairs[:, 0], pairs[:, 1], pairs[:, 2], pairs[:, 3]
    q = m2 / m1                                   # m1 >= m2 already ordered
    chi_eff = (m1 * c1 + m2 * c2) / (m1 + m2)     # engine convention, in [0,1]
    return m1, q, chi_eff


def _joint_panel(ax, x, y, x_edges, y_edges, *, xlabel, ylabel, title):
    H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    dens = H / (H.sum() * np.outer(np.diff(x_edges), np.diff(y_edges)))
    pos = dens[dens > 0]
    vmin = pos.min() if pos.size else 1e-12
    vmax = dens.max() if pos.size else 1.0
    pcm = ax.pcolormesh(x_edges, y_edges, dens.T,
                        norm=LogNorm(vmin=max(vmin, vmax * 1e-4), vmax=vmax),
                        cmap=ENGINE_CMAP, shading="flat", rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return pcm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obs", type=int, default=300,
                    help="rate-calibration target (shape-invariant; matches recovery)")
    ap.add_argument("--n-seed", type=int, default=4000,
                    help="engine MC particle count (higher => smoother truth)")
    ap.add_argument("--n-pairs", type=int, default=300000,
                    help="binary events resampled from the particle pool")
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--n-quad", type=int, default=32)
    ap.add_argument("--n-theta", type=int, default=2)
    ap.add_argument("--n-dphi", type=int, default=4)
    ap.add_argument("--n-m-fit", type=int, default=None)
    ap.add_argument("--n-chi-fit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--out-tag", type=str, default="")
    args = ap.parse_args()

    print(f"[cfg] n_seed={args.n_seed} n_pairs={args.n_pairs} k_max={args.k_max}")
    pairs, theta_star, fixed = build_truth(args)
    m1, q, chi_eff = observables(pairs)
    print(f"[obs] pairs={len(pairs)}  m1[{m1.min():.1f},{m1.max():.1f}]  "
          f"q[{q.min():.3f},{q.max():.3f}]  chi_eff[{chi_eff.min():.3f},{chi_eff.max():.3f}]")
    print(f"[obs] chi1 range [{pairs[:,2].min():.3f},{pairs[:,2].max():.3f}] "
          f"(sanity: magnitude convention => >= 0)")

    m_hi = 0.7 * fixed["m_max"] * 2.5
    m_edges = np.geomspace(fixed["m_min"], min(m1.max() * 1.02, m_hi), 49)
    q_edges = np.linspace(0.0, 1.0, 41)
    chi_edges = np.linspace(0.0, 1.0, 41)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    p1 = _joint_panel(axes[0], m1, q, m_edges, q_edges,
                      xlabel=r"$m_1\ [M_\odot]$", ylabel=r"$q = m_2/m_1$",
                      title=r"$dN/(dm_1\,dq)$")
    p2 = _joint_panel(axes[1], m1, chi_eff, m_edges, chi_edges,
                      xlabel=r"$m_1\ [M_\odot]$", ylabel=r"$\chi_{\rm eff}$",
                      title=r"$dN/(dm_1\,d\chi_{\rm eff})$")
    for pcm, ax in ((p1, axes[0]), (p2, axes[1])):
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        r"Stage-1 single-pulse TRUTH ($\mu_{\rm nat}=8\,M_\odot$, "
        r"$f_{\rm dep}=0.10$, isotropic): the model being recovered",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    figdir = _DEMO / "figures"
    figdir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        out = figdir / f"stage1_truth_2d_panels{args.out_tag}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[fig] {out}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
