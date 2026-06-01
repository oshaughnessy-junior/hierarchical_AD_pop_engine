r"""Bridge tests — the IMPLICIT solver path through the gwkokab adapter.

Companion to ``test_gwkokab_bridge.py`` (which pins the ``mc_jax`` path).  This
file validates the wiring added when we pivoted the two-pulse fit off the
``mc_jax`` hybrid KDE and onto the grid Picard fixed point
(``demo/HANDOFF_two_pulse_recovery.md``): ``make_coagulation_dist_fn(...,
solver="implicit")`` must route BOTH likelihood pieces through the engine grid
density —

* ``log_prob`` → :meth:`CoagulationResponse.log_rate_density` grid path
  (``_log_total_density`` → ``log_density_at``), AD via the implicit solver's
  ``@jax.custom_vjp`` IFT, and
* ``total_rate`` → :func:`observables.total_rate` on the converged ``N*``, the
  SAME trapezoid measure ``log_rate_density`` reads (so the gwkokab shape term
  and the −μ term stay mutually consistent).

Two tiers, same convention as the companion file:

* **In-sandbox (I1–I2)** — needs only ``jax`` + the light ``observables`` /
  ``grid`` / ``kernels`` modules (no gwkokab).  Pins the NEW grid
  ``total_rate_fn`` numerics + AD that the bridge installs for non-MC solvers,
  using a fake duck-typed response so we exercise the bridge contract, not the
  physics (physics is the engine's own test suite).
* **Full-stack (I3–I6)** — builds the REAL ``solver="implicit"`` two-pulse
  response and drives the gwkokab-shaped likelihood.  Skips where the engine
  (→ gwkokab → RIFT/lalsuite) isn't importable; runs on the workstation.  I6
  is the headline: finite-difference vs AD on ``lambda_natal`` / ``sigma_natal_2``
  THROUGH the Picard fixed point — proves the IFT custom-VJP survives the
  bridge into an inference-shaped loss.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parent.parent
for p in (str(_REPO), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (  # noqa: E402
    CoagulationPopulation,
    coagulation_poisson_mean_estimator,
    make_coagulation_dist_fn,
)

_M_MIN, _M_MAX, _V_ESC = 4.0, 80.0, 1000.0


# ===========================================================================
# In-sandbox tier (I1–I2): the grid total_rate_fn the bridge installs for
# non-MC solvers, exercised against the REAL observables quadrature with a
# fake density so AD has something to flow through.
# ===========================================================================
def _grid_pieces():
    """A tiny real grid + kernel params + a smooth, param-dependent log N."""
    obs = pytest.importorskip(
        "hierarchical_ad_pop_engine.observables",
        reason="observables pulls the engine grid/kernels (light, but skip if "
               "the package isn't importable here)",
    )
    from hierarchical_ad_pop_engine.grids import BinaryGrid

    grid = BinaryGrid.default(m_min=_M_MIN, m_max=_M_MAX, n_m=24, n_chi=8)
    kernel_params = {"a": jnp.asarray(0.0), "b": jnp.asarray(2.0)}
    return obs, grid, kernel_params


def _fake_grid_total_rate_fn(grid, kernel_params, obs, log_R_env, peak):
    r"""Mirror of the bridge's non-MC ``total_rate_fn``: integrate a
    param-dependent single-BH density on the engine trapezoid measure.

    ``peak`` stands in for the converged Picard ``N*`` amplitude so a gradient
    exists; the real path gets ``N`` from ``_log_total_density`` (which the
    custom-VJP makes differentiable)."""
    mgrid = grid.m[:, None]
    chigrid = grid.chi[None, :]
    # smooth density bump (NOT physics — drives the quadrature + AD only)
    N = peak * jnp.exp(-0.5 * ((mgrid - 12.0) / 4.0) ** 2) \
        * jnp.exp(-2.0 * chigrid)
    return obs.total_rate(N, grid, kernel_params, log_R_env)


def test_grid_total_rate_is_finite_and_scales_with_R_env():
    obs, grid, kp = _grid_pieces()
    r0 = _fake_grid_total_rate_fn(grid, kp, obs, jnp.asarray(0.0), jnp.asarray(1.0))
    r1 = _fake_grid_total_rate_fn(grid, kp, obs, jnp.asarray(np.log(3.0)),
                                  jnp.asarray(1.0))
    assert np.isfinite(float(r0)) and float(r0) > 0.0
    # total_rate ∝ R_env (linear prefactor) — exp(log 3) ⇒ 3×.
    np.testing.assert_allclose(float(r1) / float(r0), 3.0, rtol=1e-6)
    print(f"\n[I1] grid total_rate finite={float(r0):.4e}; R_env scaling ✓")


def test_grid_total_rate_is_ad_traceable():
    obs, grid, kp = _grid_pieces()

    def loss(peak):
        return _fake_grid_total_rate_fn(grid, kp, obs, jnp.asarray(0.0), peak)

    val, g = jax.value_and_grad(loss)(jnp.asarray(1.3))
    # total_rate is quadratic in the density amplitude (N·N) ⇒ d/dpeak ∝ 2·peak.
    fd = (loss(jnp.asarray(1.3 + 1e-4)) - loss(jnp.asarray(1.3 - 1e-4))) / 2e-4
    assert np.isfinite(float(g)) and float(g) != 0.0
    np.testing.assert_allclose(float(g), float(fd), rtol=1e-5)
    print(f"\n[I2] d total_rate/d(amplitude) AD={float(g):.4f} vs FD={float(fd):.4f}")


# ===========================================================================
# Full-stack tier (I3–I6): real solver="implicit" two-pulse response.
# ===========================================================================
_THETA_2P = dict(
    mu_natal=10.0, sigma_natal=1.0, mu_natal_2=30.0, sigma_natal_2=2.0,
    lambda_natal=0.4, natal_spin_loc=0.0, natal_spin_scale=0.05,
    kernel_a=0.0, kernel_b=2.0, log_rate=0.0,
)

_ENV_IMPORT_ERRORS = (ImportError, ModuleNotFoundError, OSError)


def _try_implicit_two_pulse(epsilon=1.0):
    """Build the REAL implicit two-pulse model; skip ONLY on env-import errors.

    A signature/API/bridge bug must FAIL loudly (same policy as the companion
    file's G6/G7) — only a missing-stack import is a legitimate skip.
    """
    try:
        dist_fn = make_coagulation_dist_fn(
            v_esc=_V_ESC, m_min=_M_MIN, m_max=_M_MAX, kernel_a=0.0,
            k_max=3, natal_family="two_gaussian", float_kernel_a=True,
            solver="implicit", epsilon=epsilon,
        )
        return dist_fn(**_THETA_2P)   # forces the heavy implicit build
    except _ENV_IMPORT_ERRORS as exc:
        pytest.skip(
            f"engine stack not importable here (environment, NOT a bridge "
            f"bug): {type(exc).__name__}: {exc}"
        )


def _events(n=16):
    rng = np.random.default_rng(0)
    m1 = rng.uniform(8.0, 34.0, n)
    m2 = np.minimum(m1, rng.uniform(6.0, 30.0, n))
    chi1 = rng.uniform(0.0, 0.6, n)
    chi2 = rng.uniform(0.0, 0.6, n)
    return jnp.asarray(np.stack([m1, m2, chi1, chi2], axis=-1))


def test_implicit_log_prob_matches_engine_grid_path():
    model = _try_implicit_two_pulse()
    x = _events()
    lp = model.log_prob(x)
    env = {"v_esc": jnp.asarray(_V_ESC)}
    direct = model._response.log_rate_density(
        x[:, 0], x[:, 1], x[:, 2], x[:, 3], env,
    )
    assert np.all(np.isfinite(np.asarray(lp)))
    np.testing.assert_allclose(np.asarray(lp), np.asarray(direct), rtol=1e-6)
    print(f"\n[I3] implicit log_prob == engine grid log_rate_density; "
          f"finite, shape {lp.shape}")


def test_implicit_total_rate_finite_and_positive():
    model = _try_implicit_two_pulse()
    mu, var = coagulation_poisson_mean_estimator(model, T_obs=1.0, n_quad=32)
    assert np.isfinite(float(mu)) and float(mu) > 0.0
    assert float(var) == 0.0
    print(f"\n[I4] implicit μ = total_rate(N*) = {float(mu):.4e} (finite, >0)")


def test_implicit_grad_flows_through_picard():
    """jax.grad of an inference-shaped loss wrt lambda_natal must be finite and
    nonzero THROUGH the implicit fixed point (custom-VJP, not loop-unroll)."""
    _try_implicit_two_pulse()  # availability guard (skips if stack missing)
    x = _events()

    def loss(lam):
        dist_fn = make_coagulation_dist_fn(
            v_esc=_V_ESC, m_min=_M_MIN, m_max=_M_MAX, kernel_a=0.0, k_max=3,
            natal_family="two_gaussian", float_kernel_a=True,
            solver="implicit", epsilon=1.0,
        )
        m = dist_fn(**{**_THETA_2P, "lambda_natal": lam})
        return jnp.sum(m.log_prob(x))

    g = float(jax.grad(loss)(jnp.asarray(0.4)))
    assert np.isfinite(g) and g != 0.0
    print(f"\n[I5] dΣlog_prob/dλ through Picard = {g:.4f} (finite, nonzero)")


def test_implicit_fd_vs_ad_lambda_and_sigma2():
    r"""HEADLINE: finite-difference vs AD on (lambda_natal, sigma_natal_2)
    through the implicit solve + bridge.  Agreement at ~1e-4 confirms the IFT
    custom-VJP is exact end-to-end (PLAN_engine_v2_production Step 3 contract,
    now satisfied through the gwkokab adapter, not just the bare solver)."""
    x = _events()

    def neg_log_lik(theta_vec):
        lam, s2 = theta_vec[0], theta_vec[1]
        dist_fn = make_coagulation_dist_fn(
            v_esc=_V_ESC, m_min=_M_MIN, m_max=_M_MAX, kernel_a=0.0, k_max=3,
            natal_family="two_gaussian", float_kernel_a=True,
            solver="implicit", epsilon=1.0,
        )
        model = dist_fn(**{**_THETA_2P, "lambda_natal": lam, "sigma_natal_2": s2})
        shape = jnp.sum(model.log_prob(x))
        mu, _ = coagulation_poisson_mean_estimator(model, T_obs=1.0, n_quad=32)
        return -(shape - mu)

    # availability guard (skip cleanly if stack missing)
    _try_implicit_two_pulse()

    theta0 = jnp.asarray([0.4, 2.0])
    g_ad = np.asarray(jax.grad(neg_log_lik)(theta0))
    eps = 1e-4
    g_fd = np.empty(2)
    for i in range(2):
        tp = theta0.at[i].add(eps)
        tm = theta0.at[i].add(-eps)
        g_fd[i] = float((neg_log_lik(tp) - neg_log_lik(tm)) / (2 * eps))
    assert np.all(np.isfinite(g_ad)) and np.all(np.isfinite(g_fd))
    np.testing.assert_allclose(g_ad, g_fd, rtol=2e-3, atol=1e-4)
    print(f"\n[I6] FD vs AD through Picard: AD={g_ad} FD={g_fd}")
