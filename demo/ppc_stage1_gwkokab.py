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

Forward sampling reuses ``fit_stage1_gwkokab.sample_from_model`` (inverse-CDF on
the model's own pair intensity, optionally ``vt``-weighted), so it is
solver-agnostic.

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

PARAM_NAMES = fit.PARAM_NAMES


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
                    help="predicted events sampled per posterior draw")
    ap.add_argument("--detection", choices=["none", "vt"], default="none")
    ap.add_argument("--gate", choices=["soft", "bernoulli"], default="soft",
                    help="cascade K-gate; bernoulli (B.1) is more SVI-robust")
    ap.add_argument("--seed", type=int, default=20260530)
    args = ap.parse_args()
    print(f"[cfg] n_obs={args.n_obs} n_seed={args.n_seed} k_max={args.k_max} "
          f"n_draws={args.n_draws} n_pred={args.n_pred} det={args.detection}")

    fixed = demo.FIXED
    m_hi = fixed["m_max"] * 2.5
    vt_fn = vt_powerlaw() if args.detection == "vt" else None

    # --- setup: dist_fn, calibrated θ*, self-consistent observed catalogue ---
    fit_grid = demo._grid_for_fit()
    demo.build_truth_model_and_calibrate_rate(args.n_obs, grid=fit_grid)
    demo.THETA_STAR["natal_spin_loc"] = 0.05
    theta_star = dict(demo.THETA_STAR)

    key = jax.random.PRNGKey(args.seed)
    k_mc, k_cat, k_svi, k_post, k_pred = jax.random.split(key, 5)
    dist_fn = make_coagulation_dist_fn(
        v_esc=fixed["v_esc"], m_min=fixed["m_min"], m_max=fixed["m_max"],
        kernel_a=fixed["kernel_a"], k_max=args.k_max, n_seed=args.n_seed,
        depletion_fraction=0.10, gate=args.gate, grid=fit_grid, mc_jax_key=k_mc)

    probe = dist_fn(**{**theta_star, "log_rate": 0.0})
    mu_unit, _ = coagulation_poisson_mean_estimator(probe, T_obs=1.0, n_quad=args.n_quad)
    theta_star["log_rate"] = float(jnp.log(args.n_obs / mu_unit))

    truth_model = dist_fn(**theta_star)
    k_true, k_scat = jax.random.split(k_cat, 2)
    obs_true = fit.sample_from_model(truth_model, args.n_obs, k_true,
                                     m_lo=fixed["m_min"], m_hi=m_hi, vt_fn=vt_fn)
    events = scatter_to_posteriors(obs_true, args.n_samples, k_scat, m_hi=m_hi)
    ln_offsets = jnp.zeros((args.n_obs, args.n_samples))
    obs_pts = np.asarray(jnp.mean(events, axis=1))  # per-event PE mean → (N,4)
    print(f"[cat] events {tuple(events.shape)}; log_rate*={theta_star['log_rate']:.3f}")

    # --- posterior via robust SVI -------------------------------------------
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
    # thin to n_draws, keeping finite draws only
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

    # --- forward-model each posterior draw → predicted component m, χ --------
    m_bins = np.linspace(fixed["m_min"], 0.7 * m_hi, 41)
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

    @jax.jit
    def forward_one(theta_vec, pk):
        th = {p: theta_vec[i] for i, p in enumerate(PARAM_NAMES)}
        mdl = dist_fn(**th)
        pred = fit.sample_from_model(mdl, args.n_pred, pk, m_lo=fixed["m_min"],
                                     m_hi=m_hi, vt_fn=vt_fn, n_sm=16, n_sc=6)
        comp_m = jnp.concatenate([pred[:, 0], pred[:, 1]])
        comp_chi = jnp.concatenate([pred[:, 2], pred[:, 3]])
        return _hist_density(comp_m, m_edges), _hist_density(comp_chi, chi_edges)

    # jit once → bounded memory + fast across all draws (per-draw EAGER builds
    # otherwise accumulate device buffers and OOM the 3.9 GB VM).
    m_dens = np.zeros((len(draws), m_cent.size))
    chi_dens = np.zeros((len(draws), chi_cent.size))
    pred_keys = jax.random.split(k_pred, len(draws))
    for d, (row, pk) in enumerate(zip(draws, pred_keys)):
        mh, ch = forward_one(jnp.asarray(row), pk)
        m_dens[d] = np.asarray(mh)
        chi_dens[d] = np.asarray(ch)
        if d % 10 == 0:
            print(f"   draw {d}/{len(draws)}", flush=True)

    def band(a):
        return np.percentile(a, [5, 50, 95], axis=0)

    m_lo_, m_med, m_hi_ = band(m_dens)
    c_lo_, c_med, c_hi_ = band(chi_dens)
    obs_m = np.concatenate([obs_pts[:, 0], obs_pts[:, 1]])
    obs_chi = np.concatenate([obs_pts[:, 2], obs_pts[:, 3]])

    # --- median-model predicted 2-D density (big cloud) for the scatter ------
    med_model = dist_fn(**median_theta)
    cloud = np.asarray(fit.sample_from_model(
        med_model, 15000, jax.random.PRNGKey(args.seed + 99),
        m_lo=fixed["m_min"], m_hi=m_hi, vt_fn=vt_fn, n_sm=16, n_sc=6))

    # persist the computed bands so figures survive a killed/again run
    np.savez(_DEMO / "figures" / f"stage1_ppc_data_{args.detection}.npz",
             m_cent=m_cent, m_lo=m_lo_, m_med=m_med, m_hi=m_hi_, obs_m=obs_m,
             chi_cent=chi_cent, c_lo=c_lo_, c_med=c_med, c_hi=c_hi_,
             obs_chi=obs_chi, obs_pts=obs_pts, cloud=cloud)
    _figures(args, m_cent, m_lo_, m_med, m_hi_, obs_m, m_bins,
             chi_cent, c_lo_, c_med, c_hi_, obs_chi, chi_bins,
             obs_pts, cloud, fixed)
    print("\n=== DONE ===")


def _figures(args, m_cent, m_lo_, m_med, m_hi_, obs_m, m_bins,
             chi_cent, c_lo_, c_med, c_hi_, obs_chi, chi_bins,
             obs_pts, cloud, fixed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = args.detection
    figdir = _DEMO / "figures"
    figdir.mkdir(exist_ok=True)

    # PPD in mass + spin
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].fill_between(m_cent, m_lo_, m_hi_, color="C0", alpha=0.3,
                       label="90% predictive band")
    ax[0].plot(m_cent, m_med, color="C0", lw=2, label="posterior median")
    ax[0].hist(obs_m, bins=m_bins, density=True, histtype="step",
               color="C3", lw=2, label="observed (component)")
    ax[0].set_xlabel(r"component mass $m\ [M_\odot]$")
    ax[0].set_ylabel(r"$p(m)$ (normalised)")
    ax[0].set_title(f"Mass PPD ({tag} selection)")
    ax[0].legend(fontsize=9)
    ax[0].grid(True, ls=":", alpha=0.4)

    ax[1].fill_between(chi_cent, c_lo_, c_hi_, color="C0", alpha=0.3,
                       label="90% predictive band")
    ax[1].plot(chi_cent, c_med, color="C0", lw=2, label="posterior median")
    ax[1].hist(obs_chi, bins=chi_bins, density=True, histtype="step",
               color="C3", lw=2, label="observed (component)")
    ax[1].set_xlabel(r"component spin $\chi$")
    ax[1].set_ylabel(r"$p(\chi)$ (normalised)")
    ax[1].set_title(f"Spin PPD ({tag} selection)")
    ax[1].legend(fontsize=9)
    ax[1].grid(True, ls=":", alpha=0.4)
    fig.suptitle("Stage-1 posterior-predictive distributions "
                 f"(N={args.n_obs}, {args.n_draws} draws × {args.n_pred} pred)")
    fig.tight_layout()
    out1 = figdir / f"stage1_ppd_mass_spin_{tag}.png"
    fig.savefig(out1, dpi=140)
    print(f"[fig] {out1}")

    # Event scatter over predicted 2-D density
    fig2, axs = plt.subplots(figsize=(6.5, 6))
    h, xe, ye = np.histogram2d(cloud[:, 0], cloud[:, 1], bins=60,
                               range=[[fixed["m_min"], 0.7 * fixed["m_max"] * 2.5],
                                      [fixed["m_min"], 0.7 * fixed["m_max"] * 2.5]])
    axs.pcolormesh(xe, ye, h.T, cmap="Blues", shading="auto")
    axs.scatter(obs_pts[:, 0], obs_pts[:, 1], s=22, facecolor="none",
                edgecolor="C3", lw=1.2, label="observed events")
    axs.set_xlabel(r"$m_1\ [M_\odot]$")
    axs.set_ylabel(r"$m_2\ [M_\odot]$")
    axs.set_title(f"Events over predicted population density ({tag})")
    axs.legend(fontsize=9, loc="upper left")
    fig2.tight_layout()
    out2 = figdir / f"stage1_ppc_scatter_{tag}.png"
    fig2.savefig(out2, dpi=140)
    print(f"[fig] {out2}")


if __name__ == "__main__":
    main()
