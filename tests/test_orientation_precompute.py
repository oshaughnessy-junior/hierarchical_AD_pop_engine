"""Tests for the F1 orientation-marginalised precompute cache.

Covers the four PLAN-prescribed acceptance gates from
``demo/PLAN_implicit_footprint_refactor.md`` § Session F1 ::

    * tests/test_orientation_precompute.py:
      * Bit-for-bit Aligned shortcut (the cache is None under Aligned).
      * Precomputed loss-side equals scan-side to rel `< 1e-10` —
        these are literally the same integral.
      * AD-on-κ flows through the precomputed tensor.
      * Memory budget assertion: ``P_retain_marg`` array ``nbytes`` is
        within 2× the analytic bound ``N_m² × N_χ² × 8 B``.

All assertions are either bit-equality, scan-vs-cached residual gates,
or finite-and-non-zero gradient checks against a finite-difference
baseline.  No frozen ``.npz`` references per
``[[feedback-no-frozen-test-references]]``.

The tests live one layer above the implicit-solver wiring (they call
``loss_density`` and ``solve_implicit_fixed_point`` directly), so a
regression in either the cache module or the dispatch wiring will fail
at least one assertion.

Hard constraints honoured
-------------------------
* ``[[feedback-no-fake-physics-stand-ins]]`` — the cache is a
  controlled memo of the same ``P_retain_marginalised`` /
  ``chi_rem_at_orientation`` quadrature that the un-cached path uses;
  bit-stable invariant verifies no drift.
* ``[[feedback-no-fudge-factors]]`` — no tolerances tuned to make the
  test pass; AD-vs-FD gate is ``rel < 1e-5`` for both κ and v_esc,
  matching the precision of the rest of the gradient suite.
* ``[[feedback-no-frozen-test-references]]`` — all checks are
  invariants (bitwise / residual / gradient) rather than pinned
  reference outputs.
* PLAN Decision 2 — Aligned dispatch bit-for-bit at every layer.

Cross-references
----------------
* ``[[project-session-handoff-20260527-m1]]`` — the planning handoff
  that scoped F1.
* ``[[project-orientation-quadrature-defaults-flip]]`` — the (4, 8)
  quadrature default; this test fixture mirrors that for a small but
  faithful Stage-1 run.
* ``[[project-misalignment-session-c-done]]`` — the scan path the
  cache replaces.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.orientation_precompute import (
    OrientationCache,
    build_orientation_cache,
)
from hierarchical_ad_pop_engine.response.coagulation import (
    build_gaussian_coagulation,
)
from hierarchical_ad_pop_engine.response.implicit import (
    loss_density,
    solve_implicit_fixed_point,
)
from hierarchical_ad_pop_engine.spin_orientation import (
    Aligned,
    Fisher,
    Isotropic,
)


# ---------------------------------------------------------------------------
# Stage-1 fixture — small enough to keep the AD-vs-FD test under a few
# seconds, large enough that the 4-D cache tensors carry actual structure
# (not all-zero) for the residual gate.  Mirrors the
# ``test_misalignment_invariants.py`` Stage-1 fixture choice.
# ---------------------------------------------------------------------------
_M_MIN = 5.0
_M_MAX = 80.0
_N_M = 8
_N_CHI = 6
_V_ESC = 300.0
_EPS = 0.3
_PICARD = dict(epsilon=_EPS, max_iter=100, tol=1e-9, damping=0.4)
_KIND = "sigmoid"
_W_OVER = 1.0


def _grid() -> BinaryGrid:
    return BinaryGrid.default(
        m_min=_M_MIN, m_max=_M_MAX, n_m=_N_M, n_chi=_N_CHI
    )


def _build(orient, kind=_KIND):
    return build_gaussian_coagulation(
        mu=8.0, sigma=1.0, m_min=_M_MIN, m_max=_M_MAX,
        natal_spin_loc=0.0, natal_spin_scale=0.10,
        kernel_a=0.0, kernel_b=1.0, log_R_env=0.0, k_max=3,
        grid=_grid(),
        solver="implicit", epsilon=_EPS,
        orientation=orient, retention_kind=kind, w_over_v_esc=_W_OVER,
    )


# ===========================================================================
# Gate 1 — Aligned shortcut: cache is None and converged N* is bit-equal
# ===========================================================================
def test_aligned_cache_is_none():
    """``build_orientation_cache`` returns ``None`` under :class:`Aligned`.

    The Aligned shortcut at the cache layer is required to keep the
    PLAN Decision-2 bit-stability invariant fired through every
    downstream consumer.  ``loss_density``'s ``cache is None`` branch
    is what dispatches to the scalar ``retention_aligned`` path.
    """
    grid = _grid()
    cache_a = build_orientation_cache(
        grid, Aligned(), jnp.asarray(_V_ESC),
        retention_kind=_KIND, w_over_v_esc=_W_OVER,
    )
    assert cache_a is None


def test_aligned_solve_bitstable_under_f1():
    """Aligned-default and explicit-Aligned converged ``N*`` are bit-equal.

    This is the cross-cutting bit-stability gate at the solver level:
    the F1 wiring through ``cache`` must not perturb the Aligned
    arithmetic by even a float ULP.  Regressing this would mean the
    dispatch leaked through the non-Aligned cache path.
    """
    r_default = _build(None)
    r_aligned = _build(Aligned())
    N_def = np.asarray(
        solve_implicit_fixed_point(
            r_default, jnp.asarray(_V_ESC), **_PICARD
        ).N
    )
    N_ali = np.asarray(
        solve_implicit_fixed_point(
            r_aligned, jnp.asarray(_V_ESC), **_PICARD
        ).N
    )
    assert N_def.shape == N_ali.shape
    assert np.array_equal(N_def, N_ali), (
        "F1 cache path leaked through Aligned dispatch — converged N* diverged"
    )


# ===========================================================================
# Gate 2 — Cached loss-side equals scan-side to machine precision
# ===========================================================================
@pytest.mark.parametrize(
    "orient",
    [
        Isotropic(n_theta=3, n_dphi=4),
        Fisher(kappa=jnp.asarray(0.5), n_theta=3, n_dphi=4),
        Fisher(kappa=jnp.asarray(5.0), n_theta=3, n_dphi=4),
    ],
    ids=["isotropic", "fisher_k0p5", "fisher_k5"],
)
def test_loss_density_cache_equals_scan(orient):
    """``loss_density(..., cache=cache) ≡ loss_density(..., cache=None)``.

    The cache is a memo of the same ``P_retain_marginalised`` call the
    un-cached path makes per Picard iteration.  Bit-equality is the
    expected result (same arithmetic, same trace).  PLAN gate sets the
    bar at ``rel < 1e-10``; we expect machine precision in practice
    and assert against that as the stronger invariant.
    """
    grid = _grid()
    r = _build(orient, kind=_KIND)

    # Random N — keeps the bilinear-scatter pattern non-trivial.  Any
    # nonzero N would do; the residual is independent of N magnitude
    # by linearity of loss_density in N.
    key = jax.random.PRNGKey(7)
    N = jax.random.uniform(key, (grid.m.size, grid.chi.size))

    v_esc = jnp.asarray(_V_ESC)
    loss_scan = loss_density(N, r, v_esc, cache=None)
    cache = build_orientation_cache(
        grid, orient, v_esc, retention_kind=_KIND, w_over_v_esc=_W_OVER,
    )
    loss_cached = loss_density(N, r, v_esc, cache=cache)

    rel_diff = float(
        jnp.max(
            jnp.abs(loss_scan - loss_cached)
            / jnp.maximum(jnp.abs(loss_scan), 1e-30)
        )
    )
    assert rel_diff < 1e-12, (
        f"Cached loss differs from scan loss by rel {rel_diff:.2e} > 1e-12 — "
        "cache is no longer a faithful memo of the per-iter scan"
    )


def test_solve_implicit_converges_under_isotropic_f1():
    """The F1-wired solver still converges under Isotropic + sigmoid.

    Sanity check that the cache plumbing through ``_picard_while_loop``
    and the post-convergence diagnostics didn't break convergence.
    """
    iso = Isotropic(n_theta=3, n_dphi=4)
    r = _build(iso, kind="sigmoid")
    out = solve_implicit_fixed_point(r, jnp.asarray(_V_ESC), **_PICARD)
    assert bool(jnp.isfinite(out.final_diff))
    assert float(out.final_diff) < _PICARD["tol"]
    assert int(out.n_iter) > 1  # actually iterated, not a no-op


# ===========================================================================
# Gate 3 — AD on κ and v_esc flow through the precomputed tensor
# ===========================================================================
def test_ad_on_kappa_via_cache_matches_finite_difference():
    """``jax.grad(total_N)(κ)`` under Fisher agrees with finite-diff.

    The IFT VJP backward in ``_make_picard_solve_fn`` rebuilds the
    cache inside the ``vjp_theta_fn`` closure so ``∂/∂κ`` flows through
    both the cache and any direct response surface.  If the closure
    forgot to rebuild the cache (e.g. captured the forward ``cache``
    as a free variable), this AD-vs-FD gate would fail at the order of
    the FD step.
    """
    def total_N(k):
        fisher = Fisher(kappa=k, n_theta=3, n_dphi=4)
        r = _build(fisher, kind="sigmoid")
        return solve_implicit_fixed_point(
            r, jnp.asarray(_V_ESC), **_PICARD
        ).N.sum()

    k0 = jnp.asarray(2.0)
    g_ad = float(jax.grad(total_N)(k0))
    h = 1e-4
    g_fd = float((total_N(k0 + h) - total_N(k0 - h)) / (2.0 * h))

    assert jnp.isfinite(g_ad)
    assert g_ad != 0.0, (
        "∂N_sum/∂κ is zero under Fisher — cache build dropped the κ "
        "dependency from the trace"
    )
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
    assert rel < 1e-5, (
        f"AD vs FD on κ disagree by rel {rel:.2e} > 1e-5 "
        f"(AD={g_ad:.3e}, FD={g_fd:.3e})"
    )


def test_ad_on_v_esc_via_cache_matches_finite_difference():
    """``jax.grad(total_N)(v_esc)`` under Isotropic+sigmoid agrees with FD.

    The PLAN's two-for-one AD-on-v_esc win cashed in through the F1
    cache wiring.  Under Aligned+Heaviside this gradient is zero (v1
    limitation); under non-Aligned+sigmoid it flows through both the
    cache and the post-convergence diagnostics.
    """
    iso = Isotropic(n_theta=3, n_dphi=4)

    def total_N(v):
        r = _build(iso, kind="sigmoid")
        return solve_implicit_fixed_point(r, v, **_PICARD).N.sum()

    v0 = jnp.asarray(_V_ESC)
    g_ad = float(jax.grad(total_N)(v0))
    hv = 0.1
    g_fd = float((total_N(v0 + hv) - total_N(v0 - hv)) / (2.0 * hv))

    assert jnp.isfinite(g_ad)
    assert g_ad != 0.0, (
        "∂N_sum/∂v_esc is zero under Isotropic+sigmoid — cache build "
        "dropped the v_esc dependency from the trace"
    )
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
    assert rel < 1e-5, (
        f"AD vs FD on v_esc disagree by rel {rel:.2e} > 1e-5 "
        f"(AD={g_ad:.3e}, FD={g_fd:.3e})"
    )


# ===========================================================================
# Gate 4 — Memory budget
# ===========================================================================
def test_cache_memory_budget_within_2x_analytic_bound():
    """Cache tensors fit within 2× ``N_m² × N_χ² × 8 B`` per the PLAN.

    Analytic bound: ``N_m² × N_χ² × 8 B`` per tensor (float64).  With
    two tensors (``P_retain_marg`` + ``chi_rem_mean``) the total cache
    should fit within ``2 × 2 × N_m² × N_χ² × 8 B = 32 · (N_m N_χ)²``.

    Anything larger means the cache built more leading axes than
    intended (e.g. accidentally retained the orientation quadrature
    axis), and the PLAN's ~13 MB-per-tensor projection at production
    resolution would be wrong.
    """
    grid = _grid()
    iso = Isotropic(n_theta=3, n_dphi=4)
    cache = build_orientation_cache(
        grid, iso, jnp.asarray(_V_ESC),
        retention_kind=_KIND, w_over_v_esc=_W_OVER,
    )
    assert cache is not None
    analytic_per_tensor = _N_M**2 * _N_CHI**2 * 8
    p_nbytes = int(np.asarray(cache.P_retain_marg).nbytes)
    c_nbytes = int(np.asarray(cache.chi_rem_mean).nbytes)
    assert p_nbytes <= 2 * analytic_per_tensor, (
        f"P_retain_marg = {p_nbytes} B > 2× analytic bound "
        f"({2 * analytic_per_tensor} B) — extra leading axes?"
    )
    assert c_nbytes <= 2 * analytic_per_tensor, (
        f"chi_rem_mean = {c_nbytes} B > 2× analytic bound "
        f"({2 * analytic_per_tensor} B) — extra leading axes?"
    )
    # Expected shape — guards against silent transposition/squeeze drift.
    assert cache.P_retain_marg.shape == (_N_M, _N_CHI, _N_M, _N_CHI)
    assert cache.chi_rem_mean.shape == (_N_M, _N_CHI, _N_M, _N_CHI)


# ===========================================================================
# Supporting structural invariants
# ===========================================================================
def test_chi_rem_mean_in_unit_interval():
    """Orientation-marginalised ``|χ_rem|`` is bounded in ``[0, 1]``.

    The Barausse-Rezzolla vector remnant is bounded by 1 per-event
    (Kerr extremal limit), so any orientation marginal is also bounded
    by 1.  A regression where the χ_rem mean exceeds 1 would indicate
    a normalisation or unit bug in the scan accumulator.
    """
    grid = _grid()
    iso = Isotropic(n_theta=3, n_dphi=4)
    cache = build_orientation_cache(
        grid, iso, jnp.asarray(_V_ESC),
        retention_kind=_KIND, w_over_v_esc=_W_OVER,
    )
    assert cache is not None
    mn = float(jnp.min(cache.chi_rem_mean))
    mx = float(jnp.max(cache.chi_rem_mean))
    assert mn >= 0.0, f"chi_rem_mean min {mn} < 0"
    assert mx <= 1.0, f"chi_rem_mean max {mx} > 1"


def test_p_retain_in_unit_interval():
    """Orientation-marginalised retention is a probability in ``[0, 1]``."""
    grid = _grid()
    iso = Isotropic(n_theta=3, n_dphi=4)
    cache = build_orientation_cache(
        grid, iso, jnp.asarray(_V_ESC),
        retention_kind=_KIND, w_over_v_esc=_W_OVER,
    )
    assert cache is not None
    mn = float(jnp.min(cache.P_retain_marg))
    mx = float(jnp.max(cache.P_retain_marg))
    assert mn >= 0.0, f"P_retain_marg min {mn} < 0"
    # Σ w_k = 1 + sigmoid ∈ [0, 1] per orientation point ⇒ marginal ≤ 1.
    # Allow a 1e-12 slack for float arithmetic at the upper boundary.
    assert mx <= 1.0 + 1e-12, f"P_retain_marg max {mx} > 1 + 1e-12"


def test_cache_jit_traceable():
    """``build_orientation_cache`` is end-to-end JIT-traceable.

    The forward Picard loop call site relies on this — if the build
    were eager-only (e.g. via a Python ``if`` on a traced
    ``orientation.kappa``), the surrounding ``jax.jit`` would fail at
    trace time.  Direct ``jax.jit`` wrap verifies the path.
    """
    grid = _grid()

    @jax.jit
    def go(kappa, v):
        fisher = Fisher(kappa=kappa, n_theta=3, n_dphi=4)
        cache = build_orientation_cache(
            grid, fisher, v, retention_kind=_KIND, w_over_v_esc=_W_OVER,
        )
        return cache.P_retain_marg.sum() + cache.chi_rem_mean.sum()

    out = float(go(jnp.asarray(1.0), jnp.asarray(_V_ESC)))
    assert jnp.isfinite(out)


# ===========================================================================
# Cross-orientation invariant — Fisher κ → 0 limit
# ===========================================================================
def test_fisher_small_kappa_matches_isotropic_cache():
    """``Fisher(κ → 0)`` cache asymptotes to the Isotropic cache.

    Cross-physics invariant rather than a frozen numerical value: at
    very small κ, the Fisher PDF on the sphere collapses to uniform,
    and the per-axis quadrature weights match the isotropic weights to
    ``O(κ²)`` per the small-κ expansion in ``spin_orientation.py``.

    This guards against a κ-dispatch regression that silently routed
    Fisher through a different orientation-quadrature path than
    Isotropic.
    """
    grid = _grid()
    iso = Isotropic(n_theta=3, n_dphi=4)
    fisher = Fisher(kappa=jnp.asarray(1e-6), n_theta=3, n_dphi=4)
    cache_i = build_orientation_cache(
        grid, iso, jnp.asarray(_V_ESC),
        retention_kind=_KIND, w_over_v_esc=_W_OVER,
    )
    cache_f = build_orientation_cache(
        grid, fisher, jnp.asarray(_V_ESC),
        retention_kind=_KIND, w_over_v_esc=_W_OVER,
    )
    # Relative agreement at O(κ²) ≈ 1e-12 in exact arithmetic; with the
    # regularised κ→0 branch of ``Fisher.quadrature_3d`` (κ/sinh(κ) via
    # ``2κ/(exp(κ) − exp(−κ))``) we observe ~1e-7 rel on
    # ``chi_rem_mean`` and ~1e-10 on ``P_retain_marg`` at κ=1e-6 in
    # float64 — the χ_rem-mean is the looser of the two because it
    # divides by the chi_rem_at_orientation magnitude which can be small
    # on chi-low cells, amplifying the relative residual.  Gate at 1e-6
    # to leave a 10× margin above the observed residual; tighter than
    # that pins the implementation detail of the regularisation branch,
    # not the physics invariant.
    rel_p = float(
        jnp.max(
            jnp.abs(cache_i.P_retain_marg - cache_f.P_retain_marg)
            / jnp.maximum(jnp.abs(cache_i.P_retain_marg), 1e-30)
        )
    )
    rel_c = float(
        jnp.max(
            jnp.abs(cache_i.chi_rem_mean - cache_f.chi_rem_mean)
            / jnp.maximum(jnp.abs(cache_i.chi_rem_mean), 1e-30)
        )
    )
    assert rel_p < 1e-6, f"Fisher(κ→0).P_retain_marg vs Isotropic rel diff {rel_p:.2e}"
    assert rel_c < 1e-6, f"Fisher(κ→0).chi_rem_mean vs Isotropic rel diff {rel_c:.2e}"
