r"""Hybrid analytic-1G + KDE-(2G+) density — H1-H4 pytest tier.

Mirrors ``demo/validate_hybrid_1g_density.py``.  H5 (end-to-end through
``build_two_pulse_coagulation(solver='mc_jax')``) is the slow integration
check kept in the demo script / workstation run.

Implements ``demo/PLAN_hybrid_analytic_1g_density.md`` § Validation.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from hierarchical_ad_pop_engine.evaluation.mc_jax import MCEvaluatorJax

MU1, SIG1, MU2, SIG2, LAM2 = 10.0, 1.0, 30.0, 2.0, 0.4
M_MIN, M_MAX = 2.0, 200.0
SPIN_LOC, SPIN_SCALE = 0.0, 0.05
KERNEL = {"a": 0.0, "b": 2.0}
NATAL_MASS = dict(mu_1=MU1, sigma_1=SIG1, mu_2=MU2, sigma_2=SIG2,
                  lambda_2=LAM2, m_min=M_MIN, m_max=M_MAX)
NATAL_SPIN = dict(loc=SPIN_LOC, scale=SPIN_SCALE)

# Memory-frugality: density() builds a dense (Q×S) kernel matrix, so we keep
# the pool modest and evaluate queries in chunks (peak mem ≈ CHUNK·S·8 bytes).
CHUNK = 256


def _density_chunked(ev, m_flat, chi_flat, chunk=CHUNK):
    m_flat = jnp.asarray(m_flat).ravel()
    chi_flat = jnp.asarray(chi_flat).ravel()
    out = [ev.density(m_flat[i:i + chunk], chi_flat[i:i + chunk])
           for i in range(0, m_flat.shape[0], chunk)]
    return jnp.concatenate(out)


@pytest.fixture(scope="module")
def pool():
    rng = np.random.default_rng(0)
    n1, n2 = 1500, 600
    comp = rng.random(n1) < (1.0 - LAM2)
    m1 = np.where(comp, rng.normal(MU1, SIG1, n1), rng.normal(MU2, SIG2, n1))
    m1 = np.clip(m1, M_MIN + 1e-3, M_MAX - 1e-3)
    chi1 = np.clip(np.abs(rng.normal(SPIN_LOC, SPIN_SCALE, n1)), 0.0, 1.0)
    m2 = np.clip(rng.normal(25.0, 8.0, n2), M_MIN + 1e-3, M_MAX - 1e-3)
    chi2 = np.clip(rng.normal(0.68, 0.06, n2), 0.0, 1.0)
    m = jnp.asarray(np.concatenate([m1, m2]))
    chi = jnp.asarray(np.concatenate([chi1, chi2]))
    gen = jnp.asarray(np.concatenate([np.ones(n1, int), np.full(n2, 2, int)]))
    w = jnp.ones(m.shape[0])
    mergers = jnp.zeros((1, 8))
    return m, chi, gen, w, mergers


def _ev(hybrid, p):
    m, chi, gen, w, mergers = p
    return MCEvaluatorJax.from_run(
        m, chi, mergers, log_R_env=0.0, kernel_params=KERNEL, weights=w,
        gen_active=gen if hybrid else None, hybrid_1g=hybrid,
        natal_mass_kind="two_gaussian" if hybrid else None,
        natal_mass_params=NATAL_MASS if hybrid else None,
        natal_spin_params=NATAL_SPIN if hybrid else None,
    )


def _dN_dm(ev, m_grid):
    cg = jnp.linspace(0.0, 1.0, 80)
    M, C = jnp.meshgrid(m_grid, cg, indexing="ij")
    N = _density_chunked(ev, M, C).reshape(M.shape)
    return jnp.trapezoid(N, cg, axis=1)


def _fwhm(mg, y, lo, hi):
    mg = np.asarray(mg); y = np.asarray(y)
    s = (mg >= lo) & (mg <= hi); ms, ys = mg[s], y[s]
    half = ys.max() / 2.0; i = ys.argmax()
    li = np.where(ys[:i] <= half)[0]; ri = np.where(ys[i:] <= half)[0]
    assert li.size and ri.size
    left = np.interp(half, [ys[li[-1]], ys[li[-1] + 1]], [ms[li[-1]], ms[li[-1] + 1]])
    j = i + ri[0]
    right = np.interp(half, [ys[j], ys[j - 1]], [ms[j], ms[j - 1]])
    return right - left


def test_H1_pulse1_width_recovery(pool):
    mg = jnp.linspace(6.0, 16.0, 200)
    fwhm_h = _fwhm(mg, _dN_dm(_ev(True, pool), mg), 6.0, 16.0)
    fwhm_k = _fwhm(mg, _dN_dm(_ev(False, pool), mg), 6.0, 16.0)
    truth = 2.3548 * SIG1
    assert abs(fwhm_h - truth) / truth < 0.15, (fwhm_h, truth)
    assert fwhm_k > 1.4 * truth, (fwhm_k, truth)  # KDE-only oversmooths


def test_H2_convention_integral_equals_W_total(pool):
    m, chi, gen, w, mergers = pool
    ev = _ev(True, pool)
    mg = jnp.linspace(M_MIN, 80.0, 300); cg = jnp.linspace(0.0, 1.0, 100)
    M, C = jnp.meshgrid(mg, cg, indexing="ij")
    N = _density_chunked(ev, M, C).reshape(M.shape)
    integral = float(jnp.trapezoid(jnp.trapezoid(N, cg, axis=1), mg))
    W_total = float(jnp.sum(w))
    assert abs(integral - W_total) / W_total < 0.03, (integral, W_total)


def test_H3_AD_matches_FD_on_sigma1(pool):
    m, chi, gen, w, mergers = pool

    def peak(sig1):
        params = dict(NATAL_MASS); params["sigma_1"] = sig1
        ev = MCEvaluatorJax.from_run(
            m, chi, mergers, log_R_env=0.0, kernel_params=KERNEL, weights=w,
            gen_active=gen, hybrid_1g=True, natal_mass_kind="two_gaussian",
            natal_mass_params=params, natal_spin_params=NATAL_SPIN,
        )
        return ev.density(jnp.asarray(MU1), jnp.asarray(SPIN_LOC))

    g_ad = float(jax.grad(peak)(jnp.asarray(SIG1)))
    h = 1e-4
    g_fd = float((peak(jnp.asarray(SIG1 + h)) - peak(jnp.asarray(SIG1 - h))) / (2 * h))
    assert abs(g_ad - g_fd) / (abs(g_fd) + 1e-30) < 1e-3, (g_ad, g_fd)


def test_H4_hybrid_off_is_legacy_bitforbit(pool):
    m, chi, gen, w, mergers = pool
    ev_off = _ev(False, pool)
    ev_legacy = MCEvaluatorJax.from_run(
        m, chi, mergers, log_R_env=0.0, kernel_params=KERNEL, weights=w,
    )
    qm = jnp.linspace(5.0, 50.0, 64); qc = jnp.linspace(0.0, 1.0, 64)
    assert jnp.array_equal(
        _density_chunked(ev_off, qm, qc), _density_chunked(ev_legacy, qm, qc)
    )


def _ev_chunk(hybrid, p, chunk):
    """Build like ``_ev`` but with an explicit ``query_chunk``."""
    m, chi, gen, w, mergers = p
    return MCEvaluatorJax.from_run(
        m, chi, mergers, log_R_env=0.0, kernel_params=KERNEL, weights=w,
        gen_active=gen if hybrid else None, hybrid_1g=hybrid,
        natal_mass_kind="two_gaussian" if hybrid else None,
        natal_mass_params=NATAL_MASS if hybrid else None,
        natal_spin_params=NATAL_SPIN if hybrid else None,
        query_chunk=chunk,
    )


@pytest.mark.parametrize("hybrid", [False, True])
def test_H6_query_chunk_is_density_invariant(pool, hybrid):
    """Engine-level chunking (query_chunk set) ≡ unchunked, to round-off.

    chunk=37 is deliberately NOT a divisor of the 65×65 query grid so the
    pad-and-slice path is exercised.  density() is the SVI workhorse, so any
    discrepancy here would silently bias the likelihood on small machines.

    NOT bit-for-bit: the flat path reduces one fused ``(Q×S)`` matrix while
    the chunked path reduces ``(chunk×S)`` blocks under ``scan`` +
    ``checkpoint``; XLA may reassociate those sums, so results can differ in
    the last ULP.  A tight ``allclose`` (rtol 1e-11) still catches any real
    indexing / ordering / pad-slice bug while tolerating reassociation.
    """
    ev_full = _ev_chunk(hybrid, pool, None)
    ev_chunked = _ev_chunk(hybrid, pool, 37)
    qm = jnp.linspace(5.0, 50.0, 65)
    qc = jnp.linspace(0.0, 1.0, 65)
    M, C = jnp.meshgrid(qm, qc, indexing="ij")
    d_full = ev_full.density(M, C)
    d_chunked = ev_chunked.density(M, C)
    assert d_full.shape == d_chunked.shape == M.shape
    assert jnp.allclose(d_full, d_chunked, rtol=1e-11, atol=1e-300)


def test_H7_query_chunk_AD_matches_unchunked(pool):
    """The checkpoint+scan chunked path must give the SAME gradient as flat.

    Guards the memory-bounding refactor's reverse-mode rematerialisation:
    ``jax.checkpoint`` recomputes each block on the backward pass, which must
    not change the value of the gradient (only its memory footprint).
    """
    m, chi, gen, w, mergers = pool

    def peak(sig1, chunk):
        params = dict(NATAL_MASS); params["sigma_1"] = sig1
        ev = MCEvaluatorJax.from_run(
            m, chi, mergers, log_R_env=0.0, kernel_params=KERNEL, weights=w,
            gen_active=gen, hybrid_1g=True, natal_mass_kind="two_gaussian",
            natal_mass_params=params, natal_spin_params=NATAL_SPIN,
            query_chunk=chunk,
        )
        qm = jnp.linspace(8.0, 12.0, 50)
        qc = jnp.full_like(qm, SPIN_LOC)
        return jnp.sum(ev.density(qm, qc))

    g_full = float(jax.grad(lambda s: peak(s, None))(jnp.asarray(SIG1)))
    g_chunk = float(jax.grad(lambda s: peak(s, 16))(jnp.asarray(SIG1)))
    assert abs(g_full - g_chunk) / (abs(g_full) + 1e-30) < 1e-10, (g_full, g_chunk)


def test_H8_grad_finite_out_of_support(pool):
    """Density gradient must be FINITE even for out-of-support queries.

    The analytic-1G truncnorm logpdfs give -inf outside [m_min,m_max] / [0,1];
    exp(-inf)=0 is a fine forward value but -inf's gradient is nan, which used
    to poison reverse-mode the moment a query (a high-mass cascade remnant or a
    measurement-scattered event above m_max) landed out of support — the
    two-pulse MAP then skipped every step.  Guards the clamp-then-mask fix in
    _analytic_1g_density ([[feedback-where-guards-forward-not-backward]]).
    """
    m, chi, gen, w, mergers = pool

    def total_at(sig1, mu2):
        params = dict(NATAL_MASS); params["sigma_1"] = sig1; params["mu_2"] = mu2
        ev = MCEvaluatorJax.from_run(
            m, chi, mergers, log_R_env=0.0, kernel_params=KERNEL, weights=w,
            gen_active=gen, hybrid_1g=True, natal_mass_kind="two_gaussian",
            natal_mass_params=params, natal_spin_params=NATAL_SPIN,
        )
        # Deliberately include OUT-OF-SUPPORT queries: m far above m_max=200
        # and chi outside [0,1] (clipped queries a scattered catalogue hits).
        qm = jnp.array([10.0, 30.0, 150.0, 250.0, 400.0])
        qc = jnp.array([0.0, 0.5, 0.9, 1.0, 1.3])
        return jnp.sum(ev.density(qm, qc))

    g = jax.grad(total_at, argnums=(0, 1))(jnp.asarray(SIG1), jnp.asarray(MU2))
    assert all(bool(jnp.isfinite(gi)) for gi in g), g


def test_H9_log_rate_density_grad_finite_at_zero_density_event():
    """``log_rate_density`` grad must be finite when an event has ~zero density.

    The two-pulse MAP failed because a binary in the KDE-sparse inter-pulse
    region gets N≈0; a bare ``log(maximum(N,1e-300))`` routes ``1/N`` (→ inf)
    into the downstream logsumexp → ``0·inf = nan``, killing the gradient for
    every cascade-coupled parameter at once.  The ``where``-guarded
    ``_safe_log_floor`` (default floor 1e-12) must give a FINITE gradient.
    Builds a real MCResponseJax via build_two_pulse_coagulation so the exact
    MAP code path is exercised.
    """
    import jax as _jax
    from hierarchical_ad_pop_engine.response.coagulation import (
        build_two_pulse_coagulation,
    )

    def shape(mu2):
        resp = build_two_pulse_coagulation(
            mu_1=MU1, sigma_1=SIG1, mu_2=mu2, sigma_2=SIG2, lambda_2=LAM2,
            m_min=M_MIN, m_max=M_MAX,
            natal_spin_loc=SPIN_LOC, natal_spin_scale=SPIN_SCALE,
            kernel_a=KERNEL["a"], kernel_b=KERNEL["b"], log_R_env=0.0,
            solver="mc_jax", mc_v_esc=300.0, mc_n_seed=400,
            mc_jax_key=_jax.random.PRNGKey(0), mc_jax_gate="soft",
        )
        # An event smack in the inter-pulse void (≈20 M⊙, both components) where
        # the gen≥2 KDE is empty → N≈0 → the 1/N grad trap.
        m1 = jnp.array([20.0]); m2 = jnp.array([20.0])
        c1 = jnp.array([0.0]); c2 = jnp.array([0.0])
        lrd = resp.mc_jax_response.log_rate_density(m1, m2, c1, c2, {})
        return jnp.sum(lrd)

    g = float(_jax.grad(shape)(jnp.asarray(MU2)))
    assert np.isfinite(g), g


def _ev_adaptive(pool, adaptive):
    m, chi, gen, w, mergers = pool
    return MCEvaluatorJax.from_run(
        m, chi, mergers, log_R_env=0.0, kernel_params=KERNEL, weights=w,
        gen_active=gen, hybrid_1g=True, natal_mass_kind="two_gaussian",
        natal_mass_params=NATAL_MASS, natal_spin_params=NATAL_SPIN,
        kde_adaptive=adaptive,
    )


def test_H10_adaptive_kde_mass_preserving_and_sharper(pool):
    """Abramson adaptive gen≥2 KDE: still integrates to W_total, and SHARPER.

    Two guarantees the variable-bandwidth estimator must keep:
      (i)  mass preservation — ∫∫N dm dχ ≈ W_total to ~3% (per-sample kernel
           normalisation 1/(2π h_log_m h_chi λ_i²) keeps each kernel unit-mass),
           so H2 / total_rate convention is untouched;
      (ii) it concentrates probability — the gen≥2 component's peak density is
           HIGHER than the fixed-bandwidth version (the whole point: resolve the
           cascade-remnant peaks Silverman oversmooths).
    """
    m, chi, gen, w, mergers = pool
    ev_fix = _ev_adaptive(pool, False)
    ev_ada = _ev_adaptive(pool, True)
    W_total = float(jnp.sum(w))

    mg = jnp.linspace(M_MIN, 80.0, 300); cg = jnp.linspace(0.0, 1.0, 100)
    M, C = jnp.meshgrid(mg, cg, indexing="ij")
    integ = float(jnp.trapezoid(
        jnp.trapezoid(_density_chunked(ev_ada, M, C).reshape(M.shape), cg, axis=1), mg))
    assert abs(integ - W_total) / W_total < 0.03, (integ, W_total)

    # gen≥2 peak (the cascade-remnant region ~20-40 M⊙) is sharper → higher max.
    mr = jnp.linspace(15.0, 45.0, 200)
    cr = jnp.full_like(mr, 0.68)  # the synthetic gen≥2 χ pile-up
    peak_fix = float(jnp.max(ev_fix.density(mr, cr)))
    peak_ada = float(jnp.max(ev_ada.density(mr, cr)))
    assert peak_ada > peak_fix, (peak_ada, peak_fix)


def test_H11_adaptive_off_is_fixed_bandwidth_bitforbit(pool):
    """kde_adaptive=False ≡ no lam path (regression guard for the default)."""
    ev_off = _ev_adaptive(pool, False)
    assert ev_off.lam_kde is None
    qm = jnp.linspace(5.0, 50.0, 64); qc = jnp.linspace(0.0, 1.0, 64)
    # default-built (no kde_adaptive kwarg) must match adaptive=False exactly.
    ev_default = _ev(True, pool)
    assert jnp.array_equal(ev_off.density(qm, qc), ev_default.density(qm, qc))
