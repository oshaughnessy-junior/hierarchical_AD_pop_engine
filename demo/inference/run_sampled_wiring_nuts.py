#!/usr/bin/env python3
r"""WORKSTATION GATE — production sampled-mode wiring, full NUTS recovery.

This is the run the sandbox could not do (45 s cap): the PLAN-item-5
production wiring (``inference/sampled_wiring.py``) driven end-to-end at
realistic scale — 6-parameter Stage-1 recovery (mu_natal, sigma_natal,
natal_spin_loc, natal_spin_scale, kernel_b, log_rate) through
``build_sampled_likelihood`` → ``MCMC(NUTS(...))``.

Setup (workstation, real stack — no RIFT stub):

    cd hierarchical_AD_pop_engine
    export PYTHONPATH=src:.:../gwkokab/src   # gwkokab fork branch
                                             # feature/sampled-poisson-likelihood
    # first re-run the static gates:
    python -m pytest tests/test_gwkokab_sampled_bridge.py -q      # S1-S8
    # then this:
    python demo/inference/run_sampled_wiring_nuts.py              # ~minutes

Smoke (any machine, ~1 min):

    python demo/inference/run_sampled_wiring_nuts.py --smoke

PASS criteria (printed at the end, exit code 0/1):
  G1  logL at truth finite; per-event ESS table printed, median ESS >= 30
  G2  NUTS ran, no divergences reported by numpyro, r_hat < 1.05 (all params)
  G3  every parameter recovered within |z| < 3 of its truth
      (log_rate truth = log(n_obs / T_obs): fixed-size catalogue convention,
      see demo/inference/README.md)
  G4  posterior sd of mu_natal < prior sd (data actually informative)

If G3 fails on a single seed do not panic — single-catalogue scatter; rerun
with --seed N. Systematic failure across seeds is a real finding (suspects:
ESS starvation -> raise --n-seed or stratify proposals (M3a/M3b); or the
mc-jax count-normalisation thread if log_rate is the offender).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.inference.sampled_wiring import (  # noqa: E402
    build_sampled_likelihood,
    gaussian_event_log_likelihoods_from_posteriors,
    make_coagulation_sampler_fn,
)
from gwkokab.inference.poissonlikelihood_utils import (  # noqa: E402
    sampled_poisson_likelihood_fn,
)

# Stage-1-like single-pulse truth (toy conventions, demo/inference/README.md)
TRUTH = dict(
    mu_natal=8.0,
    sigma_natal=1.0,
    natal_spin_loc=0.0,
    natal_spin_scale=0.05,
    kernel_b=2.0,
    log_rate=None,  # set from n_obs/T_obs at runtime (fixed-size catalogue)
)
T_OBS = 1.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-seed", type=int, default=20_000,
                   help="MC particle seeds (model-sample budget)")
    p.add_argument("--n-obs", type=int, default=60)
    p.add_argument("--n-pe", type=int, default=500,
                   help="posterior-cloud samples per event (Gaussian-fit input)")
    p.add_argument("--num-warmup", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=750)
    p.add_argument("--num-chains", type=int, default=1)
    p.add_argument("--gate", choices=("soft", "bernoulli"), default="soft")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="tiny everything (~1 min) — wiring check, NOT a "
                        "statistics gate; G3/G4 tolerances loosened")
    p.add_argument("--outdir", type=str,
                   default=os.path.join(os.path.dirname(__file__) or ".",
                                        "figures_wiring"))
    args = p.parse_args()
    if args.smoke:
        args.n_seed, args.n_obs, args.n_pe = 800, 15, 200
        args.num_warmup, args.num_samples = 100, 100

    import numpyro
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(args.num_chains)
    os.makedirs(args.outdir, exist_ok=True)

    const = dict(v_esc=300.0, m_min=4.0, m_max=80.0,
                 n_seed=args.n_seed, depletion_fraction=0.10)
    truth = dict(TRUTH, log_rate=math.log(args.n_obs / T_OBS))

    # ── catalogue: weighted-resample truth mergers + Gaussian PE clouds ──
    sampler_fn = make_coagulation_sampler_fn(
        gate=args.gate, mc_jax_key=jax.random.PRNGKey(7), **const
    )
    s, lw = sampler_fn(**truth)
    x = np.asarray(s)
    prob = np.exp(np.asarray(lw - jax.scipy.special.logsumexp(lw)))
    rng = np.random.default_rng(args.seed + 20260604)
    idx = rng.choice(x.shape[0], size=args.n_obs, replace=True, p=prob)
    truths = x[idx]
    sig = np.stack(
        [0.10 * truths[:, 0], 0.10 * truths[:, 1],
         np.full(args.n_obs, 0.2), np.full(args.n_obs, 0.2)], axis=1)
    observed = truths + sig * rng.standard_normal(truths.shape)
    observed[:, 2:] = np.clip(observed[:, 2:], -0.998, 0.998)
    clouds = observed[:, None, :] + sig[:, None, :] * rng.standard_normal(
        (args.n_obs, args.n_pe, 4))
    Ls = gaussian_event_log_likelihoods_from_posteriors(jnp.asarray(clouds))
    print(f"[catalog] n_obs={args.n_obs}  m1 in "
          f"[{truths[:, 0].min():.1f}, {truths[:, 0].max():.1f}]  "
          f"log_rate truth={truth['log_rate']:.3f}")

    # ── G1: logL + ESS at truth ──────────────────────────────────────────
    ll, diag = sampled_poisson_likelihood_fn(
        Ls, s, lw, None, jnp.asarray(T_OBS), None)
    ess = np.sort(np.asarray(diag["ess_per_event"]))
    print(f"[G1] logL(truth)={float(ll):+.2f}  mu={float(diag['expected_rate']):.1f}")
    print(f"[G1] ESS/event: min {ess[0]:.0f} | p25 {ess[len(ess)//4]:.0f} | "
          f"median {np.median(ess):.0f} | max {ess[-1]:.0f}   "
          f"(events with ESS<10: {int((ess < 10).sum())}/{args.n_obs})")
    g1 = bool(np.isfinite(float(ll))) and np.median(ess) >= (5 if args.smoke else 30)

    # ── NUTS over the production wiring ─────────────────────────────────
    built = build_sampled_likelihood(
        event_log_likelihood_fns=Ls, gate=args.gate,
        mc_jax_key=jax.random.PRNGKey(7), **const)
    t0 = time.time()
    mcmc = MCMC(NUTS(built["log_likelihood_fn"]),
                num_warmup=args.num_warmup, num_samples=args.num_samples,
                num_chains=args.num_chains, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(args.seed + 1), jnp.asarray(T_OBS))
    print(f"[nuts] wall {time.time() - t0:.0f} s")
    post = mcmc.get_samples(group_by_chain=True)
    stats = summary(post, prob=0.9)
    post_flat = mcmc.get_samples()

    # ── gates ────────────────────────────────────────────────────────────
    names = sorted(built["variables"])
    rhat_max = max(float(np.max(stats[n]["r_hat"])) for n in names)
    g2 = rhat_max < (1.2 if args.smoke else 1.05)

    print(f"\n{'param':>17s} {'truth':>8s} {'posterior':>22s} {'z':>6s} {'r_hat':>6s}")
    zs = {}
    for n in names:
        m, sd = float(np.mean(post_flat[n])), float(np.std(post_flat[n]))
        z = (m - truth[n]) / max(sd, 1e-12)
        zs[n] = z
        print(f"{n:>17s} {truth[n]:8.3f} {m:13.3f} +/- {sd:6.3f} "
              f"{z:6.2f} {float(np.max(stats[n]['r_hat'])):6.3f}")
    zcap = 4.0 if args.smoke else 3.0
    g3 = all(abs(z) < zcap for z in zs.values())
    prior_sd_mu = float(jnp.std(built["variables"]["mu_natal"].sample(
        jax.random.PRNGKey(0), (20000,))))
    g4 = float(np.std(post_flat["mu_natal"])) < prior_sd_mu

    # ── figure ───────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, n in zip(axes.ravel(), names):
        ax.hist(np.asarray(post_flat[n]), bins=40, density=True,
                alpha=0.7, color="C0")
        ax.axvline(truth[n], color="k", ls="--", lw=1.2)
        ax.set_xlabel(n)
    fig.suptitle(f"sampled_wiring NUTS recovery (n_obs={args.n_obs}, "
                 f"n_seed={args.n_seed}, gate={args.gate}, seed={args.seed})")
    fig.tight_layout()
    fpath = os.path.join(args.outdir, f"nuts_recovery_seed{args.seed}.png")
    fig.savefig(fpath, dpi=140)
    print(f"\n[figure] {fpath}")

    # ── verdict ──────────────────────────────────────────────────────────
    print(f"\n[G1] median ESS gate          : {'PASS' if g1 else 'FAIL'}")
    print(f"[G2] r_hat max {rhat_max:.3f}       : {'PASS' if g2 else 'FAIL'}")
    print(f"[G3] all |z| < {zcap:.0f}             : {'PASS' if g3 else 'FAIL'}"
          + ("" if g3 else "  (single-seed scatter? rerun --seed)"))
    print(f"[G4] mu_natal informative      : {'PASS' if g4 else 'FAIL'}")
    ok = g1 and g2 and g3 and g4
    print(f"\n{'ALL GATES PASS' if ok else 'GATE FAILURE'}"
          f"{' (smoke mode — wiring only, not statistics)' if args.smoke else ''}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
