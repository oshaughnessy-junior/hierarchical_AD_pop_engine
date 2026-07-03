r"""Stage-1 observables recovery — posterior-predictive checks (PPC/PPD).

Hyperparameter recovery (``posterior_stage1_gwkokab.py``) shows the *fit*
works; this shows the *observables* are reproduced.  For a set of posterior
draws we **forward-model the population** (run the mc_jax cascade at each draw —
the "MC for every posterior point" the user flagged) and build:

1. an **event scatter** ``(m₁, m₂)`` overlaid on the model's predicted
   population density (a forward-sampled 2-D histogram at the posterior
   median); and
2. **posterior-predictive distributions (PPDs)** in component mass and in
   component spin — the median + 90 % predictive band over posterior draws,
   overlaid on the observed catalogue.

``--detection vt`` weights the predicted catalogue by the ``M_chirp^{15/6}``
selection (the "detection-weighted" view); default ``none`` is the intrinsic
PPC consistent with the flat-selection fit (the band should straddle the data —
the actual goodness-of-fit check).

Forward sampling uses ``fit_stage1_gwkokab.sample_pairs_from_particles`` —
**grid-free**: it resamples the engine's own MC particle pool into binary pairs
``∝ K·w_i·w_j·VT`` (the discrete high-χ 2G remnant spins are represented
exactly; no density grid or KDE).

Run (full stack; in the Cowork VM use the RIFT stub — HANDOFF_PORTABLE §4.2):

    python3 demo/ppc_stage1_gwkokab.py --n-draws 80 --n-pred 2000 \
        --svi-steps 2000 --n-obs 40 --n-seed 300 --k-max 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_DEMO = Path(__file__).resolve().parent
_REPO = _DEMO.parent
for p in (str(_DEMO), str(_REPO), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import demo_gwkokab_fit_stage1 as demo  # noqa: E402
import fit_stage1_gwkokab as fit  # noqa: E402
from mock_catalog import scatter_to_posteriors  # noqa: E402

from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (  # noqa: E402
    coagulation_poisson_mean_estimator,
    make_coagulation_dist_fn,
)
from hierarchical_ad_pop_engine.inference.selection import vt_powerlaw  # noqa: E402
from hierarchical_ad_pop_engine.inference.stage1_wiring import (  # noqa: E402
    build_coagulation_likelihood,
    build_stage1_priors,
    build_stage1_variables,
)
from hierarchical_ad_pop_engine.grids import BinaryGrid  # noqa: E402
from hierarchical_ad_pop_engine.spin_orientation import Isotropic, Fisher  # noqa: E402

# COAG_LEAN_GAIN patch (only matters for solver="implicit"; harmless for mc_jax)
import os as _os  # noqa: E402
if _os.environ.get("COAG_LEAN_GAIN"):
    import hierarchical_ad_pop_engine.orientation_precompute as _op  # noqa: E402
    import hierarchical_ad_pop_engine.response.implicit as _imp  # noqa: E402
    _op.R_CHI_OUT_DEFAULT = None
    _imp.R_CHI_OUT_DEFAULT = None

PARAM_NAMES = fit.PARAM_NAMES


def _build_orientation(spec, n_theta=None, n_dphi=None):
    extra = {}
    if n_theta is not None:
        extra["n_theta"] = int(n_theta)
    if n_dphi is not None:
        extra["n_dphi"] = int(n_dphi)
    s = str(spec).strip().lower()
    if s in ("aligned", "a"):
        return None
    if s in ("isotropic", "iso", "i"):
        return Isotropic.default(**extra)
    if s.startswith("fisher:"):
        return Fisher.default(kappa=float(s.split(":", 1)[1]), **extra)
    raise ValueError(f"bad --orientation {spec!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obs", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-seed", type=int, default=300)
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--n-quad", type=int, default=32)
    ap.add_argument("--svi-steps", type=int, default=2000)
    ap.add_argument("--svi-lr", type=float, default=0.01)
    ap.add_argument("--n-draws", type=int, default=80,
                    help="posterior draws forward-modelled for the PPD band")
    ap.add_argument("--n-pred", type=int, default=2000,
                    help="predicted binary pairs resampled per posterior draw")
    ap.add_argument("--detection", choices=["none", "vt"], default="none")
    ap.add_argument("--gate", choices=["soft", "bernoulli"], default="soft",
                    help="cascade K-gate; bernoulli (B.1) is more SVI-robust")
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--orientation", type=str, default="aligned",
                    help="aligned | isotropic | fisher:<kappa>")
    ap.add_argument("--solver", type=str, default="mc_jax",
                    choices=("mc_jax", "implicit"))
    ap.add_argument("--n-chi-fit", type=int, default=None)
    ap.add_argument("--n-m-fit", type=int, default=None)
    ap.add_argument("--n-theta-fit", type=int, default=2)
    ap.add_argument("--n-dphi-fit", type=int, default=4)
    ap.add_argument("--out-tag", type=str, default="")
    ap.add_argument("--load-posterior", type=str, default=None,
                    help="npz of saved posterior draws (stage1_post_svi*.npz from "
                         "posterior_stage1_gwkokab.py) to forward-model; skips the "
                         "PPC's own SVI so the PPD uses the SAME (validated) "
                         "posterior as the corner")
    args = ap.parse_args()
    print(f"[cfg] n_obs={args.n_obs} n_seed={args.n_seed} k_max={args.k_max} "
          f"n_draws={args.n_draws} n_pred={args.n_pred} det={args.detection}")

    fixed = demo.FIXED
    m_hi = fixed["m_max"] * 2.5
    vt_fn = vt_powerlaw() if args.detection == "vt" else None

    # --- setup: dist_fn, calibrated θ*, self-consistent observed catalogue ---
    if args.n_chi_fit is None and args.n_m_fit is None:
        fit_grid = demo._grid_for_fit()
    else:
        fit_grid = BinaryGrid.default(
            m_min=fixed["m_min"], m_max=fixed["m_max"],
            n_m=args.n_m_fit or demo.N_M_FIT, n_chi=args.n_chi_fit or demo.N_CHI_FIT)
    demo.build_truth_model_and_calibrate_rate(args.n_obs, grid=fit_grid)
    demo.THETA_STAR["natal_spin_loc"] = 0.05
    theta_star = dict(demo.THETA_STAR)
    orientation = _build_orientation(args.orientation, args.n_theta_fit, args.n_dphi_fit)

    key = jax.random.PRNGKey(args.seed)
    k_mc, k_cat, k_svi, k_post, k_pred = jax.random.split(key, 5)
    dist_fn = make_coagulation_dist_fn(
        v_esc=fixed["v_esc"], m_min=fixed["m_min"], m_max=fixed["m_max"],
        kernel_a=fixed["kernel_a"], k_max=args.k_max, n_seed=args.n_seed,
        depletion_fraction=0.10, gate=args.gate, grid=fit_grid, mc_jax_key=k_mc,
        solver=args.solver, orientation=orientation)

    probe = dist_fn(**{**theta_star, "log_rate": 0.0})
    mu_unit, _ = coagulation_poisson_mean_estimator(probe, T_obs=1.0, n_quad=args.n_quad)
    theta_star["log_rate"] = float(jnp.log(args.n_obs / mu_unit))

    truth_model = dist_fn(**theta_star)
    k_true, k_scat = jax.random.split(k_cat, 2)
    obs_true = fit.sample_pairs_from_particles(truth_model, args.n_obs, k_true,
                                               vt_fn=vt_fn)
    events = scatter_to_posteriors(obs_true, args.n_samples, k_scat, m_hi=m_hi)
    ln_offsets = jnp.zeros((args.n_obs, args.n_samples))
    # plot the INJECTIONS (obs_true) — the PPD predicts the astrophysical
    # distribution; the per-event PE mean would distort it (regression to mean).
    obs_pts = np.asarray(obs_true)
    print(f"[cat] events {tuple(events.shape)}; log_rate*={theta_star['log_rate']:.3f}")

    # --- posterior: reuse the corner's validated draws, or fit our own -------
    if args.load_posterior:
        zz = np.load(args.load_posterior, allow_pickle=True)
        stack = np.stack([np.asarray(zz[p]).ravel() for p in PARAM_NAMES], axis=1)
        stack = stack[np.all(np.isfinite(stack), axis=1)]
        print(f"[post] loaded {len(stack)} posterior draws from {args.load_posterior}")
    else:
        import optax
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal
        from numpyro.optim import optax_to_numpyro

        variables, variables_index = build_stage1_variables()
        priors = build_stage1_priors(variables)
        log_lik_fn = build_coagulation_likelihood(
            dist_fn=dist_fn, priors=priors, variables=variables,
            variables_index=variables_index,
            poisson_mean_estimator=coagulation_poisson_mean_estimator)
        pmean = {"T_obs": 1.0, "n_quad": args.n_quad}

        def model():
            log_lik_fn(events, ln_offsets, pmean)

        print(f"[SVI] {args.svi_steps} steps")
        guide = AutoNormal(model, init_scale=0.1)
        optim = optax_to_numpyro(optax.chain(optax.clip_by_global_norm(10.0),
                                             optax.adam(args.svi_lr)))
        svi = SVI(model, guide, optim, Trace_ELBO())
        svi_res = svi.run(k_svi, args.svi_steps, progress_bar=False)
        post = guide.sample_posterior(k_post, svi_res.params,
                                      sample_shape=(max(args.n_draws * 4, 400),))
        stack = np.stack([np.asarray(post[p]).ravel() for p in PARAM_NAMES], axis=1)
        stack = stack[np.all(np.isfinite(stack), axis=1)]
        if len(stack) == 0:
            raise SystemExit(
                "[abort] SVI produced no finite draws — the soft-gate cascade "
                "NaN'd in the backward pass at this config. Retry with a different "
                "--seed, lower --svi-lr, or gate=bernoulli (n_seed=300 is the "
                "known-stable point).")
    idx = np.linspace(0, len(stack) - 1, args.n_draws).astype(int)
    draws = stack[idx]
    median_theta = {p: float(np.median(stack[:, i])) for i, p in enumerate(PARAM_NAMES)}
    print(f"[post] {len(stack)} finite draws; forward-modelling {len(draws)}")
    print("[post] recovered median:",
          {p: round(median_theta[p], 3) for p in PARAM_NAMES},
          "| truth:", {p: round(float(theta_star[p]), 3) for p in PARAM_NAMES})

    # --- forward-model each posterior draw → predicted component m, χ --------
    m_bins = np.geomspace(fixed["m_min"], 0.7 * m_hi, 41)  # log-spaced for log-x
    chi_bins = np.linspace(0.0, 1.0, 31)
    m_cent = 0.5 * (m_bins[:-1] + m_bins[1:])
    chi_cent = 0.5 * (chi_bins[:-1] + chi_bins[1:])
    m_edges = jnp.asarray(m_bins)
    chi_edges = jnp.asarray(chi_bins)

    def _hist_density(x, edges):
        idx = jnp.clip(jnp.searchsorted(edges, x) - 1, 0, edges.size - 2)
        counts = jnp.bincount(idx, length=edges.size - 1).astype(jnp.float32)
        widths = jnp.diff(edges)
        return counts / (jnp.sum(counts) * widths)

    # keep a fixed subsample of each draw's predicted events so the scatter
    # density is the FULL posterior-predictive (pooled over draws), not a single
    # median-model realization — individual draws reach lower masses than the
    # median, so the median density spuriously excludes low-m2 injections.
    n_cloud_keep = min(args.n_pred, 600)

    @jax.jit
    def forward_one(theta_vec, pk):
        th = {p: theta_vec[i] for i, p in enumerate(PARAM_NAMES)}
        mdl = dist_fn(**th)
        pred = fit.sample_pairs_from_particles(mdl, args.n_pred, pk, vt_fn=vt_fn)
        comp_m = jnp.concatenate([pred[:, 0], pred[:, 1]])
        comp_chi = jnp.concatenate([pred[:, 2], pred[:, 3]])
        return (_hist_density(comp_m, m_edges), _hist_density(comp_chi, chi_edges),
                pred[:n_cloud_keep])

    # jit once → bounded memory + fast across all draws (per-draw EAGER builds
    # otherwise accumulate device buffers and OOM the 3.9 GB VM).
    m_dens = np.zeros((len(draws), m_cent.size))
    chi_dens = np.zeros((len(draws), chi_cent.size))
    cloud_parts = []
    pred_keys = jax.random.split(k_pred, len(draws))
    for d, (row, pk) in enumerate(zip(draws, pred_keys)):
        mh, ch, psub = forward_one(jnp.asarray(row), pk)
        m_dens[d] = np.asarray(mh)
        chi_dens[d] = np.asarray(ch)
        cloud_parts.append(np.asarray(psub))
        if d % 10 == 0:
            print(f"   draw {d}/{len(draws)}", flush=True)

    # Central line = MEAN over draws = the pooled / mixture posterior-predictive
    # density (equal N per draw, so mean-of-densities == pooled density). The
    # per-bin MEDIAN we used before spuriously SUPPRESSES the peak: when the
    # pulse centroid varies across posterior draws, at the peak bin most draws
    # are slightly off-peak, so the median sits far below the true peak even
    # though every single draw — and the median model — is a sharp pulse. The
    # 5/95 band still reports the per-draw envelope.
    def band(a):
        return np.percentile(a, [5, 95], axis=0)

    m_lo_, m_hi_ = band(m_dens)
    c_lo_, c_hi_ = band(chi_dens)
    # Central line = the density of the MEDIAN model at large N (the recovered
    # density itself), pooled over a few realizations for smoothness. This is
    # unambiguously "the recovered density" — not the per-bin median over draws
    # (which suppresses a location-varying peak) nor the mixture (which is
    # broadened by posterior spread).
    med_vec = jnp.asarray([median_theta[p] for p in PARAM_NAMES])
    _mk = jax.random.split(jax.random.fold_in(k_pred, 424242), 6)
    m_med = np.mean([np.asarray(forward_one(med_vec, kk)[0]) for kk in _mk], axis=0)
    c_med = np.mean([np.asarray(forward_one(med_vec, kk)[1]) for kk in _mk], axis=0)
    obs_m = np.concatenate([obs_pts[:, 0], obs_pts[:, 1]])
    obs_chi = np.concatenate([obs_pts[:, 2], obs_pts[:, 3]])

    # --- posterior-predictive 2-D density (pooled over ALL draws) for scatter -
    cloud = np.concatenate(cloud_parts, axis=0)

    # persist the computed bands so figures survive a killed/again run
    np.savez(_DEMO / "figures" / f"stage1_ppc_data_{args.detection}.npz",
             m_cent=m_cent, m_lo=m_lo_, m_med=m_med, m_hi=m_hi_, obs_m=obs_m,
             chi_cent=chi_cent, c_lo=c_lo_, c_med=c_med, c_hi=c_hi_,
             obs_chi=obs_chi, obs_pts=obs_pts, cloud=cloud)
    _figures(args, m_cent, m_lo_, m_med, m_hi_, obs_m, m_bins,
             chi_cent, c_lo_, c_med, c_hi_, obs_chi, chi_bins,
             obs_pts, cloud, fixed)
    print("\n=== DONE ===")


def _ppd_panel(ax, cent, lo, med, hi, obs, bins, xlabel, *,
               logx=False, logy=False):
    if logy:
        floor = max(float(np.max(med)), 1e-30) * 1e-4
        lo = np.clip(lo, floor, None)
        med = np.clip(med, floor, None)
        hi = np.clip(hi, floor, None)
    ax.fill_between(cent, lo, hi, color="C0", alpha=0.3, label="90% band")
    ax.plot(cent, med, color="C0", lw=2, label="posterior median")
    ax.hist(obs, bins=bins, density=True, histtype="step", color="C3", lw=2,
            label="injections")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$p$ (log)" if logy else r"$p$")
    ax.grid(True, which="both", ls=":", alpha=0.4)


def _figures(args, m_cent, m_lo_, m_med, m_hi_, obs_m, m_bins,
             chi_cent, c_lo_, c_med, c_hi_, obs_chi, chi_bins,
             obs_pts, cloud, fixed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = args.detection
    figdir = _DEMO / "figures"
    figdir.mkdir(exist_ok=True)
    m_xlabel = r"component mass $m\ [M_\odot]$"
    chi_xlabel = r"component spin $\chi$"

    # PPD: rows = {linear-y, log-p y}; cols = {mass (log-x), spin}.
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    _ppd_panel(ax[0, 0], m_cent, m_lo_, m_med, m_hi_, obs_m, m_bins,
               m_xlabel, logx=True)
    _ppd_panel(ax[0, 1], chi_cent, c_lo_, c_med, c_hi_, obs_chi, chi_bins,
               chi_xlabel)
    _ppd_panel(ax[1, 0], m_cent, m_lo_, m_med, m_hi_, obs_m, m_bins,
               m_xlabel, logx=True, logy=True)
    _ppd_panel(ax[1, 1], chi_cent, c_lo_, c_med, c_hi_, obs_chi, chi_bins,
               chi_xlabel, logy=True)
    ax[0, 0].set_title("Mass PPD — linear")
    ax[0, 1].set_title("Spin PPD — linear")
    ax[1, 0].set_title(r"Mass PPD — $\log p$ (dynamic range)")
    ax[1, 1].set_title(r"Spin PPD — $\log p$ (the 2G high-$\chi$ bump)")
    ax[0, 0].legend(fontsize=9)
    fig.suptitle(f"Stage-1 posterior-predictive distributions "
                 f"({tag} selection; N={args.n_obs}, {args.n_draws} draws × "
                 f"{args.n_pred} pred pairs; particle-resampled)")
    fig.tight_layout()
    out1 = figdir / f"stage1_ppd_mass_spin_{tag}.png"
    fig.savefig(out1, dpi=140)
    print(f"[fig] {out1}")

    # Event scatter over predicted 2-D density (log-log mass plane)
    fig2, axs = plt.subplots(figsize=(6.8, 6.2))
    hi_m = 0.7 * fixed["m_max"] * 2.5
    edges = np.geomspace(fixed["m_min"], hi_m, 60)
    h, xe, ye = np.histogram2d(cloud[:, 0], cloud[:, 1], bins=[edges, edges])
    axs.pcolormesh(xe, ye, np.log10(h.T + 1.0), cmap="Blues", shading="auto")
    axs.scatter(obs_pts[:, 0], obs_pts[:, 1], s=24, facecolor="none",
                edgecolor="C3", lw=1.3, label="injections")
    axs.set_xscale("log")
    axs.set_yscale("log")
    axs.set_xlabel(r"$m_1\ [M_\odot]$")
    axs.set_ylabel(r"$m_2\ [M_\odot]$")
    axs.set_title(f"Events over predicted density — $\\log_{{10}}$ counts ({tag})")
    axs.legend(fontsize=9, loc="upper left")
    fig2.tight_layout()
    out2 = figdir / f"stage1_ppc_scatter_{tag}.png"
    fig2.savefig(out2, dpi=140)
    print(f"[fig] {out2}")


if __name__ == "__main__":
    main()
