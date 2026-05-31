#!/usr/bin/env python3
r"""Diagnose the two-pulse λ/σ₂/σ₁ recovery bias (worsens with n_obs).

Three questions, each a small profile so it runs on a memory-tight box
(reuses the fit's dist_fn + estimator; pass --n-seed low and it stays small):

  D1  WHICH TERM carries the bias?  Split the gwkokab log-likelihood into
      the SHAPE term  Σ_n log_intensity(event_n)  and the POISSON term  −μ
      (= −total_rate), and 1-D-profile EACH over {sigma_natal, sigma_natal_2,
      lambda_natal}.  If the SHAPE term already peaks off-truth → the bias is
      in the density (KDE contamination / degeneracy), not the −μ metric.
      If only the SUM is off → the total_rate metric is dragging it.

  D2  HOW MUCH gen≥2 contamination sits under pulse 2?  At θ*, decompose the
      density along m (at χ≈0) into the EXACT analytic-1G part vs the gen≥2
      KDE part, and report the KDE fraction in [25,35] M⊙ (pulse 2) vs
      [8,12] M⊙ (pulse 1).  A large KDE fraction under pulse 2 is the
      mechanism: the fit can explain those events as natal-pulse-2 OR as
      cascade remnants → (μ₂,σ₂,λ) degeneracy.

  D3  DOES IT RESOLVE OUT?  Re-profile the SHAPE term over lambda_natal at
      a few n_seed values.  If the peak walks toward truth as n_seed grows,
      the bias is n_seed-resolution-limited (the under-resolved gen≥2 KDE),
      pointing the fix at the 2G+ estimator — NOT the engine physics.

Run (workstation / full stack):
    python3 demo/diagnose_two_pulse_bias.py --n-obs 60 --n-seed 600 --k-max 3
    python3 demo/diagnose_two_pulse_bias.py --n-obs 200 --n-seed 600   # bias↑
    python3 demo/diagnose_two_pulse_bias.py --d3 --n-seed-list 400 800 1600
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
from fit_stage1_gwkokab import sample_from_model  # noqa: E402
from mock_catalog import scatter_to_posteriors  # noqa: E402
import fit_two_pulse_gwkokab as F  # noqa: E402  (reuse PARAM_NAMES, THETA_STAR_2P, helpers)

from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (  # noqa: E402
    coagulation_poisson_mean_estimator,
    make_coagulation_dist_fn,
)


def _make(args, key):
    fit_grid = demo._grid_for_fit()
    fixed = demo.FIXED
    dist_fn = make_coagulation_dist_fn(
        v_esc=fixed["v_esc"], m_min=fixed["m_min"], m_max=fixed["m_max"],
        kernel_a=fixed["kernel_a"], k_max=args.k_max, n_seed=args.n_seed,
        depletion_fraction=0.10, gate="soft", grid=fit_grid, mc_jax_key=key,
        natal_family="two_gaussian", float_kernel_a=True,
        kde_adaptive=getattr(args, "kde_adaptive", False))
    return dist_fn, coagulation_poisson_mean_estimator, fixed


def _calib_and_catalogue(args, dist_fn, estimator, fixed, key):
    theta_star = dict(F.THETA_STAR_2P)
    probe = dist_fn(**{**theta_star, "log_rate": 0.0})
    mu_unit, _ = estimator(probe, T_obs=1.0, n_quad=args.n_quad)
    theta_star["log_rate"] = float(jnp.log(args.n_obs / mu_unit))
    m_hi = fixed["m_max"] * 2.5
    truth = dist_fn(**theta_star)
    k_true, k_scat = jax.random.split(key, 2)
    pts = sample_from_model(truth, args.n_obs, k_true,
                            m_lo=fixed["m_min"], m_hi=m_hi)
    events = scatter_to_posteriors(pts, args.n_samples, k_scat, m_hi=m_hi)
    return theta_star, events


def _shape_and_mu(dist_fn, estimator, events, theta_dict, n_quad):
    """Return (shape_term, mu) at a θ dict.  shape = Σ_n logsumexp_k log_prob."""
    model = dist_fn(**theta_dict)
    # events: (n_obs, n_samples, 4); log_prob broadcasts over leading axes.
    lp = model.log_prob(events)                       # (n_obs, n_samples)
    shape = float(jnp.sum(jax.scipy.special.logsumexp(lp, axis=1)
                          - jnp.log(events.shape[1])))
    mu, _ = estimator(model, T_obs=1.0, n_quad=n_quad)
    return shape, float(mu)


def d1_term_split(args, dist_fn, estimator, events, theta_star):
    print("\n=== D1  shape-term vs −μ-term, per parameter ===")
    print("    (peak of SHAPE alone tells you if the density is the culprit)")
    grids = {
        "sigma_natal": np.linspace(0.5, 3.0, 11),
        "sigma_natal_2": np.linspace(0.8, 6.0, 11),
        "lambda_natal": np.linspace(0.1, 0.8, 11),
    }
    for name, xs in grids.items():
        shp, neg_mu, tot = [], [], []
        for x in xs:
            td = dict(theta_star); td[name] = float(x)
            s, mu = _shape_and_mu(dist_fn, estimator, events, td, args.n_quad)
            shp.append(s); neg_mu.append(-mu); tot.append(s - mu)
        shp = np.array(shp); neg_mu = np.array(neg_mu); tot = np.array(tot)
        truth = float(theta_star[name])
        pk_s = xs[int(shp.argmax())]
        pk_t = xs[int(tot.argmax())]
        print(f"  {name:15s} truth={truth:+.3f}  "
              f"SHAPE-peak={pk_s:+.3f}  TOTAL-peak={pk_t:+.3f}  "
              f"Δμ-span={neg_mu.max()-neg_mu.min():.2f}")
    print("  → if SHAPE-peak is already off-truth, bias is in the density "
          "(KDE contamination / degeneracy), not the −μ metric.")


def d2_contamination(args, dist_fn, theta_star):
    print("\n=== D2  gen≥2 KDE contamination under each pulse ===")
    model = dist_fn(**theta_star)
    ev = model._response.mc_jax_response.evaluator
    if not bool(getattr(ev, "hybrid_1g", False)):
        print("  [skip] evaluator not hybrid_1g — can't split analytic vs KDE")
        return
    m = jnp.linspace(5.0, 45.0, 400)
    chi = jnp.zeros_like(m)
    # full density and analytic-1G-only; KDE part = full − analytic.
    log_mq = jnp.log(jnp.maximum(m, 1e-30))
    full = ev._density_flat(m, chi)
    analytic = ev._analytic_1g_density(m, chi)
    kde = jnp.clip(full - analytic, 0.0, None)
    m_np = np.asarray(m)
    kde_np = np.asarray(kde)
    full_np = np.asarray(full)

    def frac(lo, hi):
        sel = (m_np >= lo) & (m_np <= hi)
        f = np.trapezoid(kde_np[sel], m_np[sel])
        t = np.trapezoid(full_np[sel], m_np[sel])
        return f / max(t, 1e-30)

    print(f"  KDE(gen≥2) fraction of density  pulse-1 [8,12] M⊙ = {frac(8,12):.1%}")
    print(f"  KDE(gen≥2) fraction of density  pulse-2 [25,35] M⊙ = {frac(25,35):.1%}")
    print("  → a large pulse-2 fraction = the (μ₂,σ₂,λ) ↔ cascade-remnant "
          "degeneracy that biases pulse 2.")


def d3_resolution(args, key):
    print("\n=== D3  does the λ shape-peak walk to truth as n_seed grows? ===")
    fixed = demo.FIXED
    xs = np.linspace(0.1, 0.8, 11)
    truth_lam = float(F.THETA_STAR_2P["lambda_natal"])
    for n_seed in args.n_seed_list:
        a = argparse.Namespace(**vars(args)); a.n_seed = int(n_seed)
        dist_fn, estimator, fixed = _make(a, key)
        theta_star, events = _calib_and_catalogue(a, dist_fn, estimator, fixed, key)
        shp = []
        for x in xs:
            td = dict(theta_star); td["lambda_natal"] = float(x)
            s, _ = _shape_and_mu(dist_fn, estimator, events, td, a.n_quad)
            shp.append(s)
        pk = xs[int(np.argmax(shp))]
        print(f"  n_seed={n_seed:5d}  λ shape-peak={pk:+.3f}  (truth={truth_lam:.3f})")
    print("  → peak → truth as n_seed↑ ⇒ n_seed-resolution-limited gen≥2 KDE "
          "(fix the 2G+ estimator); peak STAYS off ⇒ a real degeneracy/metric bug.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obs", type=int, default=60)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-seed", type=int, default=600)
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--n-quad", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--sigma-lnm", type=float, default=0.1,
                    help="mock mass measurement scatter (default 0.1 = the fit's "
                         "default; pass 0 to DECONVOLUTION-test). Note 0.1 ≈ "
                         "1 M⊙ at pulse-1, 3 M⊙ at pulse-2 — comparable to the "
                         "natal widths being recovered.")
    ap.add_argument("--sigma-chi", type=float, default=0.05,
                    help="mock spin measurement scatter (pass 0 with "
                         "--sigma-lnm 0 for the no-scatter discriminator).")
    ap.add_argument("--kde-adaptive", action="store_true",
                    help="use the Abramson variable-bandwidth gen≥2 KDE "
                         "(re-run D1/D3 with it on to check the σ₂/λ peaks "
                         "move toward truth).")
    ap.add_argument("--d3", action="store_true", help="run only the n_seed sweep")
    ap.add_argument("--n-seed-list", type=int, nargs="+",
                    default=[400, 800, 1600])
    args = ap.parse_args()

    key = jax.random.PRNGKey(args.seed)
    k_mc, k_cat = jax.random.split(key, 2)

    if args.d3:
        d3_resolution(args, k_mc)
        return

    dist_fn, estimator, fixed = _make(args, k_mc)
    theta_star, events = _calib_and_catalogue(args, dist_fn, estimator, fixed, k_cat)
    print(f"[cfg] n_obs={args.n_obs} n_seed={args.n_seed} k_max={args.k_max}")
    d1_term_split(args, dist_fn, estimator, events, theta_star)
    d2_contamination(args, dist_fn, theta_star)
    print("\nRun with --d3 --n-seed-list 400 800 1600 for the resolution sweep.")


if __name__ == "__main__":
    main()
