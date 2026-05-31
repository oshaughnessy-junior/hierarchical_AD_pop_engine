#!/usr/bin/env python3
r"""Localize the non-finite-gradient source in the two-pulse MAP.

The MAP's ``-logL`` grad is non-finite from step 0 (the optimiser skips every
step and returns the start point).  Forward profiles are fine, so it is a
backward-only nan — the classic ``jnp.where``-guards-forward-not-backward
family.  This script grads each SUB-TERM of the likelihood separately at the
exact MAP start point and reports the first non-finite one, so we patch the
right place instead of guessing.

Decomposition of ``logL = Σ_n logsumexp_k log_prob(event) − μ``:

  T1  shape term   u -> Σ_n logsumexp_k log_prob(events_n)
  T2  mu term      u -> total_rate                 (= μ, the Poisson mean)

then, if T1 is the culprit, drill into the density at FIXED queries (so the
cascade-build grad is the only θ-dependence):

  D_in    density at in-support queries   (m=[10,30], χ=[0,0])
  D_out   density at out-of-support queries (m=[300], χ=[0])   ← the suspect
  D_anly  analytic-1G part only at D_out
  D_kde   (full − analytic) KDE part only at D_out

For each: print forward-finite and grad-finite.  The first grad-nan localises
the fix.  Runs at the SAME perturbed start point ``_run_map`` uses.

Run (workstation / full stack):
    python3 demo/localize_two_pulse_nan.py --n-obs 60 --n-seed 600 --k-max 3
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
import fit_two_pulse_gwkokab as F  # noqa: E402

from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (  # noqa: E402
    coagulation_poisson_mean_estimator,
    make_coagulation_dist_fn,
)


def _report(name, fn, u):
    """Print forward-finite + grad-finite for scalar fn(u) at u."""
    try:
        val = float(fn(u))
        fwd_ok = np.isfinite(val)
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:10s}  FORWARD RAISED: {type(exc).__name__}: {exc}")
        return
    try:
        g = jax.grad(fn)(u)
        g_ok = bool(jnp.all(jnp.isfinite(g)))
        # which components are nan?
        bad = np.where(~np.asarray(jnp.isfinite(g)))[0]
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:10s}  fwd={val:+.4g} [{ 'ok' if fwd_ok else 'NONFINITE'}]"
              f"  GRAD RAISED: {type(exc).__name__}: {exc}")
        return
    flag = "ok" if g_ok else f"NON-FINITE @ u[{list(bad)}]"
    print(f"  {name:10s}  fwd={val:+.4g} [{'ok' if fwd_ok else 'NONFINITE'}]"
          f"  grad=[{flag}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obs", type=int, default=60)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-seed", type=int, default=600)
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--n-quad", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260530)
    args = ap.parse_args()

    fixed = demo.FIXED
    fit_grid = demo._grid_for_fit()
    key = jax.random.PRNGKey(args.seed)
    k_mc, k_cat = jax.random.split(key, 2)

    dist_fn = make_coagulation_dist_fn(
        v_esc=fixed["v_esc"], m_min=fixed["m_min"], m_max=fixed["m_max"],
        kernel_a=fixed["kernel_a"], k_max=args.k_max, n_seed=args.n_seed,
        depletion_fraction=0.10, gate="soft", grid=fit_grid, mc_jax_key=k_mc,
        natal_family="two_gaussian", float_kernel_a=True)
    estimator = coagulation_poisson_mean_estimator

    theta_star = dict(F.THETA_STAR_2P)
    probe = dist_fn(**{**theta_star, "log_rate": 0.0})
    mu_unit, _ = estimator(probe, T_obs=1.0, n_quad=args.n_quad)
    theta_star["log_rate"] = float(jnp.log(args.n_obs / mu_unit))

    m_hi = fixed["m_max"] * 2.5
    truth = dist_fn(**theta_star)
    k_true, k_scat = jax.random.split(k_cat, 2)
    pts = sample_from_model(truth, args.n_obs, k_true,
                            m_lo=fixed["m_min"], m_hi=m_hi)
    events = scatter_to_posteriors(pts, args.n_samples, k_scat, m_hi=m_hi)

    theta0 = F._theta_vec(theta_star)
    # EXACT MAP start point: _from_physical(theta0) + the perturbation vector.
    u0 = F._from_physical(theta0) + jnp.asarray(
        [0.3, 0.2, 0.3, 0.2, 0.3, 0.3, 0.2, 0.4, 0.5, 0.4])

    def _model(u):
        return dist_fn(**F._unpack(F._to_physical(u)))

    def shape_term(u):
        lp = _model(u).log_prob(events)
        return jnp.sum(jax.scipy.special.logsumexp(lp, axis=1)
                       - jnp.log(events.shape[1]))

    def mu_term(u):
        return _model(u).total_rate(n_quad=args.n_quad)

    def dens_at(u, mq, cq):
        ev = _model(u)._response.mc_jax_response.evaluator
        return jnp.sum(ev.density(jnp.asarray(mq), jnp.asarray(cq)))

    def analytic_at(u, mq, cq):
        ev = _model(u)._response.mc_jax_response.evaluator
        return jnp.sum(ev._analytic_1g_density(jnp.asarray(mq), jnp.asarray(cq)))

    def kde_at(u, mq, cq):
        ev = _model(u)._response.mc_jax_response.evaluator
        full = ev.density(jnp.asarray(mq), jnp.asarray(cq))
        anly = ev._analytic_1g_density(jnp.asarray(mq), jnp.asarray(cq))
        return jnp.sum(full - anly)

    print(f"[cfg] n_obs={args.n_obs} n_seed={args.n_seed} k_max={args.k_max}")
    print("\n=== top-level likelihood terms (grad wrt u at MAP start) ===")
    _report("T1 shape", shape_term, u0)
    _report("T2 mu", mu_term, u0)

    print("\n=== density drill (fixed queries; θ-dep only via cascade build) ===")
    _report("D_in",  lambda u: dens_at(u, [10.0, 30.0], [0.0, 0.0]), u0)
    _report("D_out", lambda u: dens_at(u, [300.0], [0.0]), u0)
    _report("D_anly", lambda u: analytic_at(u, [300.0], [0.0]), u0)
    _report("D_kde", lambda u: kde_at(u, [10.0, 30.0], [0.0, 0.0]), u0)

    print("\n=== boundary-χ drill (χ exactly 0 and 1 — common quad nodes) ===")
    _report("D_chi0", lambda u: dens_at(u, [10.0], [0.0]), u0)
    _report("D_chi1", lambda u: dens_at(u, [10.0], [1.0]), u0)

    # --- T1 drill: isolate WHICH op in log_rate_density-over-events ------- #
    # T1 = Σ_n logsumexp_k [ log_R + logK(m1,m2) + log(max(N1/n_init,eps))
    #                        + log(max(N2/n_init,eps)) ].  Density-direct is
    # clean, so the nan is introduced by one of: the log-floor's 1/N grad at
    # near-empty events, the pairing kernel, or the events themselves.
    print("\n=== T1 drill: which op over the REAL events? ===")
    m1 = events[..., 0]; m2 = events[..., 1]
    c1 = events[..., 2]; c2 = events[..., 3]

    def _logfloor_dens(u, mq, cq, eps):
        ev = _model(u)._response.mc_jax_response.evaluator
        N = ev.density(mq, cq)
        return jnp.sum(jnp.log(jnp.maximum(N, eps)))

    # T1a: ONE density factor, log-floored, over the real event m1/χ1.
    _report("T1a log N1 (eps=1e-300)",
            lambda u: _logfloor_dens(u, m1, c1, 1e-300), u0)
    # T1b: same with a RAISED floor — if this is finite, the 1/N at the
    #      floor edge (tiny-but-positive N) is the nan source → fix the floor.
    _report("T1b log N1 (eps=1e-12)",
            lambda u: _logfloor_dens(u, m1, c1, 1e-12), u0)
    # T1c: the pairing-kernel factor alone over events (no density).
    def _logK(u):
        ev = _model(u)._response.mc_jax_response.evaluator
        from hierarchical_ad_pop_engine.evaluation.mc_jax import (
            _log_pairing_kernel_jax,
        )
        return jnp.sum(_log_pairing_kernel_jax(m1, m2, ev.kernel_a, ev.kernel_b))
    _report("T1c logK", _logK, u0)
    # T1d: the FULL response log_rate_density over events, but with the
    #      density floored at 1e-12 instead of 1e-300 (monkey-free: re-derive).
    def _shape_floor(u, eps):
        ev = _model(u)._response.mc_jax_response.evaluator
        from hierarchical_ad_pop_engine.evaluation.mc_jax import (
            _log_pairing_kernel_jax,
        )
        n_init = float(ev.n_active)  # density already count-scaled; ratio cancels in grad
        N1 = ev.density(m1, c1); N2 = ev.density(m2, c2)
        logK = _log_pairing_kernel_jax(m1, m2, ev.kernel_a, ev.kernel_b)
        lp = (ev.log_R_env + logK
              + jnp.log(jnp.maximum(N1, eps)) + jnp.log(jnp.maximum(N2, eps)))
        return jnp.sum(jax.scipy.special.logsumexp(lp, axis=1))
    _report("T1d shape (eps=1e-12)", lambda u: _shape_floor(u, 1e-12), u0)

    # T1e: COUNT pathological events at θ* (forward only).
    ev0 = truth._response.mc_jax_response.evaluator
    N1_0 = np.asarray(ev0.density(m1, c1)); N2_0 = np.asarray(ev0.density(m2, c2))
    print("\n=== event pathology census (forward, at θ*) ===")
    print(f"  events {tuple(events.shape)}  m_max(support)={fixed['m_max']}")
    print(f"  N1==0 exactly : {int((N1_0 == 0).sum())}   "
          f"0<N1<1e-12 : {int(((N1_0 > 0) & (N1_0 < 1e-12)).sum())}")
    print(f"  N2==0 exactly : {int((N2_0 == 0).sum())}   "
          f"0<N2<1e-12 : {int(((N2_0 > 0) & (N2_0 < 1e-12)).sum())}")
    print(f"  m1>m_max : {int((np.asarray(m1) > fixed['m_max']).sum())}   "
          f"m2>m_max : {int((np.asarray(m2) > fixed['m_max']).sum())}")
    print(f"  χ1==0 : {int((np.asarray(c1) == 0).sum())}   "
          f"χ1==1 : {int((np.asarray(c1) == 1).sum())}   "
          f"χ2==0 : {int((np.asarray(c2) == 0).sum())}")
    print("\nVerdict logic:\n"
          "  T1b/T1d finite but T1a nan  → the 1e-300 log-floor's 1/N grad at\n"
          "      tiny-positive N is it: raise the floor (eps≈1e-12) in\n"
          "      MCResponseJax.log_rate_density (and evaluator.log_rate_density).\n"
          "  T1a/T1b BOTH nan            → density grad nan at some event even\n"
          "      though D_in clean → a pathology census row >0 points to which\n"
          "      (m>m_max, χ boundary, or N==0 where-branch).\n"
          "  T1c nan                     → pairing kernel (q=0 / m_sml=0).")

    print("\nRead: the FIRST grad=[NON-FINITE] localises the patch site.\n"
          "  T2 only         → total_rate quad grid (m>m_max or χ=1 node)\n"
          "  D_out/D_anly    → analytic-1G out-of-support (logpdf -inf grad)\n"
          "  D_kde           → KDE reflection / W_kde→0 floor\n"
          "  D_chi1          → χ=1 boundary (truncnorm spin logpdf or reflect)\n"
          "  all clean but T1→ the cascade build (particle_mc_jax_soft) grad")


if __name__ == "__main__":
    main()
