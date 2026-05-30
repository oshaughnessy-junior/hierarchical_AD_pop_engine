r"""Bridge tests — CoagulationResponse → gwkokab analytical-Poisson likelihood.

Validates :mod:`hierarchical_ad_pop_engine.inference.gwkokab_bridge`.

Two tiers:

* **In-sandbox tier (G1–G5)** — exercises the bridge's OWN logic against a
  fake duck-typed response, needing only ``jax`` + ``numpyro``.  This runs
  in the ephemeral sandbox fast-loop (``demo/bootstrap_jax_sandbox.sh``).
  The real engine ``log_rate_density`` / ``total_rate`` are already covered
  by the M4 integration test; here we pin the NEW glue: 4-column unpacking,
  the box support, the poisson-mean delegation, AD pass-through, and — G5 —
  an inline replica of gwkokab's ``analytical_poisson_likelihood_fn``
  formula to prove the bridge satisfies that contract shape.

* **Full-stack tier (G6–G7)** — builds the REAL ``solver="mc_jax"`` response
  and (G7) calls the REAL ``gwkokab.inference`` likelihood.  These require
  the engine's ``response`` package (which transitively imports gwkokab →
  RIFT/lalsuite) and so SKIP in the lightweight sandbox; they run on a
  workstation / cardassia where the full stack is installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
for p in (str(_REPO), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Bridge imports cleanly without gwkokab (heavy engine import is lazy).
from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (  # noqa: E402
    CoagulationPopulation,
    coagulation_poisson_mean_estimator,
    make_coagulation_dist_fn,
)

_M_MIN, _M_MAX, _V_ESC = 4.0, 80.0, 1000.0


# ---------------------------------------------------------------------------
# Fake duck-typed response — AD-traceable, parameter-dependent.
# ---------------------------------------------------------------------------
def _fake_population(theta_mu, theta_rate):
    r"""CoagulationPopulation around a fake response whose log-intensity and
    total_rate depend on (theta_mu, theta_rate) so gradients flow."""
    env = {"v_esc": jnp.asarray(_V_ESC)}

    class _FakeResp:
        class mc_jax_response:  # noqa: N801 - mimic attribute path
            @staticmethod
            def total_rate(*, n_quad=64):
                return theta_rate * jnp.asarray(1.0)

        def log_rate_density(self, m1, m2, chi1, chi2, env):
            # smooth log-intensity stand-in (NOT physics — physics is
            # validated by M4; this only drives the bridge glue + AD).
            return -0.5 * ((m1 - theta_mu) ** 2 + (m2 - theta_mu) ** 2) \
                - (chi1 + chi2)

    resp = _FakeResp()
    return CoagulationPopulation(
        resp, env, lambda **kw: resp.mc_jax_response.total_rate(**kw),
        m_min=_M_MIN, m_max=_M_MAX,
    )


def _events(n=16):
    rng = np.random.default_rng(0)
    m1 = rng.uniform(8.0, 30.0, n)
    m2 = np.minimum(m1, rng.uniform(6.0, 25.0, n))
    chi1 = rng.uniform(0.0, 0.9, n)
    chi2 = rng.uniform(0.0, 0.9, n)
    return jnp.asarray(np.stack([m1, m2, chi1, chi2], axis=-1))


# ===========================================================================
# In-sandbox tier (fake response) — G1–G5.
# ===========================================================================
def test_log_prob_unpacks_and_delegates():
    model = _fake_population(jnp.asarray(8.0), jnp.asarray(50.0))
    x = _events()
    lp = model.log_prob(x)
    direct = -0.5 * ((x[:, 0] - 8.0) ** 2 + (x[:, 1] - 8.0) ** 2) \
        - (x[:, 2] + x[:, 3])
    assert lp.shape == (x.shape[0],)
    np.testing.assert_allclose(np.asarray(lp), np.asarray(direct), rtol=1e-6)
    print(f"\n[G1] log_prob unpacks (m1,m2,χ1,χ2) and delegates; "
          f"shape {lp.shape}")


def test_support_masks_out_of_domain():
    model = _fake_population(jnp.asarray(8.0), jnp.asarray(50.0))
    good = jnp.asarray([[20.0, 10.0, 0.3, 0.4]])
    bad_chi = jnp.asarray([[20.0, 10.0, 1.5, 0.4]])
    bad_m = jnp.asarray([[1.0, 0.5, 0.3, 0.4]])
    assert bool(model.support.check(good)[0])
    assert not bool(model.support.check(bad_chi)[0])
    assert not bool(model.support.check(bad_m)[0])
    print("\n[G2] support.check: in-domain True, χ>1 & m<m_min False")


def test_poisson_mean_delegates_to_total_rate():
    model = _fake_population(jnp.asarray(8.0), jnp.asarray(50.0))
    mu, var = coagulation_poisson_mean_estimator(model, T_obs=2.0, n_quad=8)
    # μ = T_obs · total_rate = 2 · 50
    np.testing.assert_allclose(float(mu), 100.0, rtol=1e-6)
    assert float(var) == 0.0
    print(f"\n[G3] poisson_mean μ = T_obs·total_rate = {float(mu):.1f}; var=0")


def test_grad_flows_through_population():
    x = _events()

    def loss(theta_mu):
        model = _fake_population(theta_mu, jnp.asarray(50.0))
        return jnp.sum(model.log_prob(x))

    g = float(jax.grad(loss)(jnp.asarray(8.0)))
    # analytic: d/dμ Σ[-0.5((m1-μ)^2+(m2-μ)^2)] = Σ[(m1-μ)+(m2-μ)]
    expect = float(jnp.sum((x[:, 0] - 8.0) + (x[:, 1] - 8.0)))
    np.testing.assert_allclose(g, expect, rtol=1e-5)
    assert abs(g) > 0.0
    print(f"\n[G4] d Σlog_prob/dθμ = {g:.3f} (matches analytic {expect:.3f})")


def test_matches_analytical_poisson_contract_inline():
    r"""Inline replica of gwkokab analytical_poisson_likelihood_fn — proves
    the bridge plugs into that contract (support.check + vmap log_prob +
    poisson_mean) and is AD-differentiable end-to-end."""
    x = _events(12)[:, None, :]          # (n_events, n_samples=1, 4)
    ln_offsets = jnp.zeros((12, 1))
    n_events = x.shape[0]

    def neg_log_lik(theta):
        theta_mu, theta_rate = theta
        model = _fake_population(theta_mu, theta_rate)
        mask = model.support.check(x)                       # (12,1)
        lp = jax.vmap(model.log_prob)(x)                    # (12,1)
        total = jnp.sum(jax.nn.logsumexp(lp + ln_offsets, axis=1, where=mask))
        mu, _ = coagulation_poisson_mean_estimator(model, T_obs=1.0, n_quad=8)
        ll = total + n_events * jnp.log(1.0) - mu
        return -ll

    theta0 = jnp.asarray([8.0, 50.0])
    val = float(neg_log_lik(theta0))
    g = np.asarray(jax.grad(neg_log_lik)(theta0))
    print(f"\n[G5] inline analytical-Poisson -logL = {val:.3f}, "
          f"grad = {g}")
    assert np.isfinite(val)
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)


# ===========================================================================
# Full-stack tier (real engine + real gwkokab) — G6–G7.  Skip if the
# response package / gwkokab (→ RIFT/lalsuite) isn't importable here.
# ===========================================================================
_THETA = dict(
    mu_natal=8.0, sigma_natal=1.0, natal_spin_loc=0.05,
    natal_spin_scale=0.10, kernel_b=1.0, log_rate=0.0,
)


# Only an IMPORT/env failure (gwkokab → RIFT/lalsuite not installed, etc.)
# is a legitimate skip.  A genuine API/build error (signature mismatch from
# gwkokab drift, a bridge bug) must FAIL loudly, not hide as a skip — that
# was the lesson when the gwkokab main branch moved underneath us.
_ENV_IMPORT_ERRORS = (ImportError, ModuleNotFoundError, OSError)


def _try_real_dist_fn(gate="soft"):
    try:
        dist_fn = make_coagulation_dist_fn(
            v_esc=_V_ESC, m_min=_M_MIN, m_max=_M_MAX, kernel_a=0.0,
            k_max=4, n_seed=400, depletion_fraction=0.10, gate=gate,
            mc_jax_key=jax.random.PRNGKey(7),
        )
        return dist_fn(**_THETA)   # forces the heavy build (→ gwkokab)
    except _ENV_IMPORT_ERRORS as exc:
        pytest.skip(
            f"full engine+gwkokab stack not importable here "
            f"(environment, NOT a bridge bug): {type(exc).__name__}: {exc}"
        )
    # NB: any OTHER exception (TypeError/AttributeError/ValueError → gwkokab
    # API drift or a real bridge bug) propagates and FAILS the test.


def test_real_response_log_prob_matches_engine():
    model = _try_real_dist_fn()
    x = _events()
    lp = model.log_prob(x)
    env = {"v_esc": jnp.asarray(_V_ESC)}
    direct = model._response.log_rate_density(
        x[:, 0], x[:, 1], x[:, 2], x[:, 3], env,
    )
    assert np.all(np.isfinite(np.asarray(lp)))
    np.testing.assert_allclose(np.asarray(lp), np.asarray(direct), rtol=1e-6)


def test_real_end_to_end_gwkokab_analytical():
    model = _try_real_dist_fn()
    try:
        from gwkokab.inference.poissonlikelihood_utils import (
            analytical_poisson_likelihood_fn,
        )
    except _ENV_IMPORT_ERRORS as exc:
        pytest.skip(
            f"gwkokab not importable here (environment): "
            f"{type(exc).__name__}: {exc}"
        )
    x = _events(12)[:, None, :]
    ln_offsets = jnp.zeros((12, 1))
    ll = analytical_poisson_likelihood_fn(
        model, coagulation_poisson_mean_estimator,
        x, ln_offsets, {"T_obs": 1.0, "n_quad": 32}, None,
    )
    assert np.isfinite(float(ll))
