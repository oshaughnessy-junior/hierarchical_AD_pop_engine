r"""Stage-2 selection tests — VT toy models + VT-weighted Poisson-mean estimator.

Validates :mod:`hierarchical_ad_pop_engine.inference.selection`.  All of the
core tier (S1–S9) needs only ``jax`` (+ the bridge's pure-numpyro
``CoagulationPopulation``); none of it touches gwkokab, so it runs in the
lightweight sandbox.  S10 plugs the estimator into the REAL gwkokab
``analytical_poisson_likelihood_fn`` and is ``importorskip``-gated like the
bridge G6/G7.

Invariants (no frozen numbers):
* ``chirp_mass`` matches the closed form;
* ``VT ∝ M_chirp^{15/6}`` scales as ``2^{15/6}`` per chirp-mass doubling and is
  flat at ``exponent = 0``;
* the estimator is **exactly linear in VT** (``VT≡c`` ⇒ ``μ = c·μ_flat``),
  monotone in ``vt_ref``, reduces to the flat-grid estimator at ``VT≡1``;
* an ``expit`` pdet ∈ (0,1) gives ``μ_pdet ≤ μ_flat``;
* ``μ`` is finite, ``≥ 0``, AD-differentiable and **nonzero-grad** in the
  population params (rate and shape);
* the return obeys gwkokab's ``(μ, var)`` contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("numpyro")
import jax.numpy as jnp  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
for p in (str(_REPO), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (  # noqa: E402
    CoagulationPopulation,
)
from hierarchical_ad_pop_engine.inference.selection import (  # noqa: E402
    chirp_mass,
    make_flat_grid_poisson_mean_estimator,
    make_vt_poisson_mean_estimator,
    snr_threshold_pdet,
    vt_powerlaw,
)

_M_MIN, _M_MAX, _V_ESC = 4.0, 80.0, 1000.0


# ---------------------------------------------------------------------------
# Duck-typed response (AD-traceable; not physics — physics is M4/bridge G6/G7).
# Intensity ∝ exp(log_rate) · Gaussian(m;mu) so μ depends on rate AND shape.
# ---------------------------------------------------------------------------
def _population(theta_mu, theta_log_rate):
    env = {"v_esc": jnp.asarray(_V_ESC)}

    class _Resp:
        def log_rate_density(self, m1, m2, chi1, chi2, _env):
            return (theta_log_rate
                    - 0.5 * ((m1 - theta_mu) ** 2 + (m2 - theta_mu) ** 2)
                    - (chi1 + chi2))

    return CoagulationPopulation(
        _Resp(), env, lambda **kw: jnp.exp(theta_log_rate),
        m_min=_M_MIN, m_max=_M_MAX,
    )


def _est_kwargs(n_m=24, n_chi=10):
    return dict(m_min=_M_MIN, m_max=_M_MAX, m_max_factor=2.5, n_m=n_m, n_chi=n_chi)


# ===========================================================================
# S1–S4 — VT toy models (pure functions).
# ===========================================================================
def test_chirp_mass_closed_form():
    m1 = jnp.asarray([10.0, 30.0, 8.0])
    m2 = jnp.asarray([10.0, 10.0, 8.0])
    mc = chirp_mass(m1, m2)
    expect = (np.asarray(m1) * np.asarray(m2)) ** 0.6 / (np.asarray(m1) + np.asarray(m2)) ** 0.2
    np.testing.assert_allclose(np.asarray(mc), expect, rtol=1e-6)
    # equal-mass identity: M_c = m / 2^{1/5}
    np.testing.assert_allclose(float(mc[0]), 10.0 / 2 ** 0.2, rtol=1e-6)
    print(f"\n[S1] chirp_mass matches closed form; equal-mass M_c={float(mc[0]):.3f}")


def test_vt_powerlaw_scaling():
    vt = vt_powerlaw(exponent=15.0 / 6.0, m_chirp_ref=10.0, vt_ref=1.0)
    # pick two equal-mass systems whose chirp masses differ by 2×
    # M_c(m,m) = m/2^{1/5} → doubling m doubles M_c.
    a = vt(jnp.asarray(10.0), jnp.asarray(10.0), 0.0, 0.0)
    b = vt(jnp.asarray(20.0), jnp.asarray(20.0), 0.0, 0.0)
    np.testing.assert_allclose(float(b / a), 2 ** (15.0 / 6.0), rtol=1e-5)
    print(f"\n[S2] VT M_chirp^(15/6): ratio per M_c doubling = {float(b/a):.3f} "
          f"(= 2^(15/6) = {2**(15/6):.3f})")


def test_vt_exponent_zero_is_flat():
    vt = vt_powerlaw(exponent=0.0, vt_ref=3.0)
    vals = vt(jnp.asarray([5.0, 40.0, 70.0]), jnp.asarray([5.0, 9.0, 12.0]),
              jnp.zeros(3), jnp.zeros(3))
    np.testing.assert_allclose(np.asarray(vals), 3.0, rtol=0, atol=1e-6)
    print("\n[S3] exponent=0 → VT ≡ vt_ref (flat), mass-independent")


def test_snr_pdet_in_unit_interval_and_monotone():
    pdet = snr_threshold_pdet(snr_ref=8.0, m_chirp_ref=10.0, snr_threshold=8.0,
                              steepness=1.0, vt_ref=1.0)
    # straddle the SNR-8 horizon (snr_ref=8 at m_chirp_ref=10) with masses whose
    # chirp mass keeps expit away from its float-saturation rail: equal-mass
    # m=8 → M_c≈6.96 (below), m=15 → M_c≈13.1 (above).
    lo = float(pdet(jnp.asarray(8.0), jnp.asarray(8.0), 0.0, 0.0))
    hi = float(pdet(jnp.asarray(15.0), jnp.asarray(15.0), 0.0, 0.0))
    assert 0.0 < lo < 0.5 < hi < 1.0
    print(f"\n[S4] expit SNR pdet ∈(0,1) & monotone across the horizon: "
          f"low-mass {lo:.3f} < 0.5 < high-mass {hi:.3f}")


# ===========================================================================
# S5–S9 — the VT-weighted estimator.
# ===========================================================================
def test_estimator_gwkokab_contract():
    model = _population(jnp.asarray(9.0), jnp.asarray(0.5))
    est = make_vt_poisson_mean_estimator(vt_powerlaw(), **_est_kwargs())
    out = est(model, T_obs=2.0, n_quad=8)  # n_quad ignored; must not error
    assert isinstance(out, tuple) and len(out) == 2
    mu, var = out
    assert np.shape(mu) == () and np.shape(var) == ()
    assert np.isfinite(float(mu)) and float(mu) > 0.0
    assert float(var) == 0.0
    print(f"\n[S5] estimator returns (μ,var)=({float(mu):.4g}, {float(var):.1g}); μ>0")


def test_estimator_exactly_linear_in_vt():
    model = _population(jnp.asarray(9.0), jnp.asarray(0.5))
    c = 3.7
    flat = make_flat_grid_poisson_mean_estimator(**_est_kwargs())
    scaled = make_vt_poisson_mean_estimator(
        vt_powerlaw(exponent=0.0, vt_ref=c), **_est_kwargs())
    mu_flat, _ = flat(model, T_obs=1.0)
    mu_scaled, _ = scaled(model, T_obs=1.0)
    np.testing.assert_allclose(float(mu_scaled), c * float(mu_flat), rtol=1e-5)
    print(f"\n[S6] VT≡{c} ⇒ μ = {c}·μ_flat exactly ({float(mu_scaled):.4g} "
          f"= {c}·{float(mu_flat):.4g})")


def test_T_obs_scales_mu_linearly():
    model = _population(jnp.asarray(9.0), jnp.asarray(0.5))
    est = make_vt_poisson_mean_estimator(vt_powerlaw(), **_est_kwargs())
    mu1, _ = est(model, T_obs=1.0)
    mu5, _ = est(model, T_obs=5.0)
    np.testing.assert_allclose(float(mu5), 5.0 * float(mu1), rtol=1e-6)
    print(f"\n[S7] μ ∝ T_obs ({float(mu5):.4g} = 5·{float(mu1):.4g})")


def test_pdet_reduces_mu_below_flat():
    r"""A pdet ∈ (0,1] can only remove events: μ_pdet ≤ μ_flat (same grid)."""
    model = _population(jnp.asarray(9.0), jnp.asarray(0.5))
    flat = make_flat_grid_poisson_mean_estimator(**_est_kwargs())
    # vt_ref=1 so pdet ∈ (0,1); a real horizon at SNR 8.
    pdet_est = make_vt_poisson_mean_estimator(
        snr_threshold_pdet(snr_ref=8.0, m_chirp_ref=10.0, snr_threshold=8.0,
                           steepness=2.0, vt_ref=1.0),
        **_est_kwargs())
    mu_flat, _ = flat(model, T_obs=1.0)
    mu_pdet, _ = pdet_est(model, T_obs=1.0)
    assert 0.0 < float(mu_pdet) <= float(mu_flat) * (1.0 + 1e-6)
    print(f"\n[S8] μ_pdet={float(mu_pdet):.4g} ≤ μ_flat={float(mu_flat):.4g}")


def test_estimator_ad_nonzero_in_rate_and_shape():
    est = make_vt_poisson_mean_estimator(vt_powerlaw(), **_est_kwargs())

    def mu_of_log_rate(lr):
        return est(_population(jnp.asarray(9.0), lr), T_obs=1.0)[0]

    def mu_of_mu(mu):
        return est(_population(mu, jnp.asarray(0.5)), T_obs=1.0)[0]

    # μ ∝ exp(log_rate) ⇒ dμ/d log_rate = μ exactly.
    lr0 = jnp.asarray(0.5)
    mu0 = float(mu_of_log_rate(lr0))
    g_lr = float(jax.grad(mu_of_log_rate)(lr0))
    np.testing.assert_allclose(g_lr, mu0, rtol=1e-4)
    g_mu = float(jax.grad(mu_of_mu)(jnp.asarray(9.0)))
    assert np.isfinite(g_lr) and g_lr != 0.0
    assert np.isfinite(g_mu) and g_mu != 0.0
    print(f"\n[S9] dμ/d log_rate={g_lr:.4g} (=μ={mu0:.4g}); dμ/d mu_natal={g_mu:.4g}")


# ===========================================================================
# S10 — plug into the REAL gwkokab analytical-Poisson likelihood (gated).
# ===========================================================================
_ENV_IMPORT_ERRORS = (ImportError, ModuleNotFoundError, OSError)


def test_real_gwkokab_likelihood_with_vt_selection():
    try:
        from gwkokab.inference.poissonlikelihood_utils import (
            analytical_poisson_likelihood_fn,
        )
    except _ENV_IMPORT_ERRORS as exc:
        pytest.skip(
            f"gwkokab not importable here (environment, NOT a selection bug): "
            f"{type(exc).__name__}: {exc}"
        )

    est = make_vt_poisson_mean_estimator(vt_powerlaw(), **_est_kwargs())
    rng = np.random.default_rng(0)
    n = 10
    x = np.stack([
        rng.uniform(8.0, 30.0, n),
        np.minimum(rng.uniform(8.0, 30.0, n), rng.uniform(6.0, 25.0, n)),
        rng.uniform(0.0, 0.9, n), rng.uniform(0.0, 0.9, n),
    ], axis=-1)[:, None, :]
    x = jnp.asarray(x)
    ln_offsets = jnp.zeros((n, 1))

    def neg_ll(theta_mu):
        model = _population(theta_mu, jnp.asarray(0.5))
        ll = analytical_poisson_likelihood_fn(
            model, est, x, ln_offsets, {"T_obs": 1.0, "n_quad": 8}, None)
        return -ll

    val = float(neg_ll(jnp.asarray(9.0)))
    g = float(jax.grad(neg_ll)(jnp.asarray(9.0)))
    print(f"\n[S10] REAL gwkokab analytical-Poisson w/ VT selection: "
          f"-logL={val:.3f}, d/d mu_natal={g:.3f}")
    assert np.isfinite(val) and np.isfinite(g) and g != 0.0
