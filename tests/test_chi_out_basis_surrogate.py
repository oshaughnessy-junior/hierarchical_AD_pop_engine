r"""Tests for the F2 χ-out bilinear-hat surrogate gain side.

Covers the four PLAN-prescribed acceptance gates from
``demo/PLAN_implicit_footprint_refactor.md`` § Session F2 ::

    * tests/test_chi_out_basis_surrogate.py:
      * Convergence: rel-L1 of ``solve_implicit_fixed_point`` vs scan
        path falls monotonically with R; gates at R such that error
        < 0.5 %.
      * Aligned shortcut still bit-for-bit (basis is None under Aligned).
      * AD on κ flows through the basis coefficients linearly.

Plus a mass-conservation invariant on the F2 cache itself
(``Σ_{j_out} chi_scatter_marg[j_out, …] == P_retain_marg[…]`` to
machine precision — the partition-of-unity bilinear-hat invariant
that justifies the F2 reformulation away from the PLAN's original
Legendre-spectral compression).

The R-convergence sub-task is **trivial** in this implementation: the
basis is bilinear hats at the engine χ grid (``R = N_χ`` by
construction), so there is no R-knob to sweep.  The gate is instead
"F2-vs-scan rel-L1 < 0.5 % at every tested resolution"; we verify at
three (N_m, N_χ) configurations.

Hard constraints honoured
-------------------------
* ``[[feedback-no-fake-physics-stand-ins]]`` — F2 is a controlled
  factorisation of the scan path's bilinear-χ-scatter, mass-
  conserving by construction.
* ``[[feedback-no-fudge-factors]]`` — the 0.5 % rel-L1 gate is the
  PLAN's published convergence target, not tuned against this
  fixture.
* ``[[feedback-no-frozen-test-references]]`` — every assertion is
  bit-equality (Aligned), F2-vs-scan invariant, or analytic
  invariant (partition-of-unity).  No frozen ``.npz``.

Related
-------
* ``[[demo/PLAN_implicit_footprint_refactor.md]]`` § Session F2.
* ``[[project-engine-f1-done]]`` — F1 closed the loss side; F2
  closes the gain side.
* ``[[project-engine-f2-done]]`` — full session details, including
  the architectural pivot from Legendre to bilinear hats.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import hierarchical_ad_pop_engine.orientation_precompute as op
import hierarchical_ad_pop_engine.response.implicit as imp
from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.orientation_precompute import (
    OrientationCache,
    build_orientation_cache,
)
from hierarchical_ad_pop_engine.response.coagulation import (
    build_gaussian_coagulation,
)
from hierarchical_ad_pop_engine.response.implicit import (
    solve_implicit_fixed_point,
)
from hierarchical_ad_pop_engine.spin_orientation import (
    Aligned,
    Fisher,
    Isotropic,
)


# ---------------------------------------------------------------------------
# Stage-1 fixture parameters
# ---------------------------------------------------------------------------
_M_MIN = 5.0
_M_MAX = 80.0
_V_ESC = 300.0
_EPS = 0.3
_PICARD = dict(epsilon=_EPS, max_iter=100, tol=1e-9, damping=0.4)
_KIND = "sigmoid"
_W_OVER = 1.0
_N_THETA = 3
_N_DPHI = 4


def _build(orient, n_m=8, n_chi=6, kind=_KIND):
    return build_gaussian_coagulation(
        mu=8.0, sigma=1.0, m_min=_M_MIN, m_max=_M_MAX,
        natal_spin_loc=0.0, natal_spin_scale=0.10,
        kernel_a=0.0, kernel_b=1.0, log_R_env=0.0, k_max=3,
        grid=BinaryGrid.default(m_min=_M_MIN, m_max=_M_MAX, n_m=n_m, n_chi=n_chi),
        solver="implicit", epsilon=_EPS,
        orientation=orient, retention_kind=kind, w_over_v_esc=_W_OVER,
    )


def _solve_with_f2(response, v_esc=_V_ESC, f2_on=True):
    """Run solve_implicit_fixed_point with/without the F2 fast path.

    F2 is gated by the module-level ``R_CHI_OUT_DEFAULT`` in
    ``response.implicit``; flip it temporarily for the comparison.
    """
    orig = imp.R_CHI_OUT_DEFAULT
    try:
        imp.R_CHI_OUT_DEFAULT = op.R_CHI_OUT_DEFAULT if f2_on else None
        return solve_implicit_fixed_point(
            response, jnp.asarray(v_esc), **_PICARD
        )
    finally:
        imp.R_CHI_OUT_DEFAULT = orig


# ===========================================================================
# Gate 1 — Aligned still bit-stable (cache None at every layer)
# ===========================================================================
def test_aligned_f2_cache_is_none():
    r"""``build_orientation_cache`` returns ``None`` under :class:`Aligned`,
    even when ``R_chi_out`` is truthy.

    The Aligned shortcut at the cache layer is non-negotiable for the
    PLAN's Decision-2 bit-stability invariant.  F2 must not erode it.
    """
    grid = BinaryGrid.default(m_min=_M_MIN, m_max=_M_MAX, n_m=8, n_chi=6)
    cache = build_orientation_cache(
        grid, Aligned(), jnp.asarray(_V_ESC),
        retention_kind=_KIND, w_over_v_esc=_W_OVER, R_chi_out=True,
    )
    assert cache is None


def test_aligned_solve_bitstable_under_f2():
    r"""Aligned-default and explicit-Aligned converged ``N*`` are bit-equal.

    The F2 wiring through ``cache`` must not perturb the Aligned
    arithmetic by even a float ULP — regression would mean the
    dispatch leaked through the non-Aligned cache path.  Mirror of
    the F1 bit-stability gate at the solver level.
    """
    r_default = _build(orient=None)
    r_aligned = _build(orient=Aligned())
    N_def = np.asarray(_solve_with_f2(r_default).N)
    N_ali = np.asarray(_solve_with_f2(r_aligned).N)
    assert N_def.shape == N_ali.shape
    assert np.array_equal(N_def, N_ali), (
        "F2 path leaked through Aligned dispatch — converged N* diverged"
    )


# ===========================================================================
# Gate 2 — F2 cache has bilinear-hat fields under non-Aligned
# ===========================================================================
@pytest.mark.parametrize(
    "orient",
    [
        Isotropic(n_theta=_N_THETA, n_dphi=_N_DPHI),
        Fisher(kappa=jnp.asarray(0.5), n_theta=_N_THETA, n_dphi=_N_DPHI),
        Fisher(kappa=jnp.asarray(5.0), n_theta=_N_THETA, n_dphi=_N_DPHI),
    ],
    ids=["isotropic", "fisher_k0p5", "fisher_k5"],
)
def test_f2_cache_fields_populated(orient):
    r"""Under non-Aligned + ``R_chi_out=True``, F2 fields are populated."""
    grid = BinaryGrid.default(m_min=_M_MIN, m_max=_M_MAX, n_m=8, n_chi=6)
    cache = build_orientation_cache(
        grid, orient, jnp.asarray(_V_ESC),
        retention_kind=_KIND, w_over_v_esc=_W_OVER, R_chi_out=True,
    )
    assert isinstance(cache, OrientationCache)
    assert cache.chi_scatter_marg is not None
    assert cache.M_rem_mean_retained is not None
    # Shape contract.
    assert cache.chi_scatter_marg.shape == (
        grid.chi.size, grid.m.size, grid.chi.size, grid.m.size, grid.chi.size,
    )
    assert cache.M_rem_mean_retained.shape == (
        grid.m.size, grid.chi.size, grid.m.size, grid.chi.size,
    )


# ===========================================================================
# Gate 3 — Mass conservation invariant (partition-of-unity)
# ===========================================================================
def test_chi_scatter_marg_partition_of_unity():
    r"""``Σ_{j_out} chi_scatter_marg[j_out, …] = P_retain_marg[…]``.

    The bilinear-hat partition-of-unity invariant that justifies
    the F2 reformulation.  Holds to machine precision by
    construction (bilinear hats sum to 1 at every point inside the
    χ grid).  If this fails, either the hats are mis-normalised or
    the orientation scan reweighting drifted.
    """
    grid = BinaryGrid.default(m_min=_M_MIN, m_max=_M_MAX, n_m=8, n_chi=6)
    iso = Isotropic(n_theta=_N_THETA, n_dphi=_N_DPHI)
    cache = build_orientation_cache(
        grid, iso, jnp.asarray(_V_ESC),
        retention_kind=_KIND, w_over_v_esc=_W_OVER, R_chi_out=True,
    )
    assert cache is not None
    chi_total_per_p = jnp.sum(cache.chi_scatter_marg, axis=0)
    max_diff = float(jnp.max(jnp.abs(chi_total_per_p - cache.P_retain_marg)))
    assert max_diff < 1e-12, (
        f"partition-of-unity violation: max |Σ_j hat[j,...] − P_retain_marg[...]| "
        f"= {max_diff:.2e} > 1e-12"
    )


# ===========================================================================
# Gate 4 — Convergence: F2 vs scan rel-L1 < 0.5%
# ===========================================================================
@pytest.mark.parametrize(
    "n_m, n_chi",
    [(8, 6), (12, 8), (16, 12)],
    ids=["8x6", "12x8", "16x12"],
)
def test_f2_rel_l1_under_half_percent(n_m, n_chi):
    r"""``solve_implicit_fixed_point`` under F2 matches scan-path to <0.5%.

    The PLAN-published convergence target.  Per the architectural
    pivot from Legendre to bilinear hats (see PLAN-superseded note
    in :class:`OrientationCache` docstring), the bilinear-hat F2 is
    mass-conserving by construction and matches the scan-path
    χ-marginal exactly per progenitor cell; the residual is just
    the M_rem orientation-spread approximation collapsed to its
    P_ret-weighted mean.  Empirically << 0.5 % at every tested
    resolution.
    """
    iso = Isotropic(n_theta=_N_THETA, n_dphi=_N_DPHI)
    r = _build(iso, n_m=n_m, n_chi=n_chi)
    out_f2 = _solve_with_f2(r, f2_on=True)
    out_scan = _solve_with_f2(r, f2_on=False)
    N_f2 = np.asarray(out_f2.N)
    N_scan = np.asarray(out_scan.N)
    rel_l1 = float(np.sum(np.abs(N_f2 - N_scan)) / np.sum(np.abs(N_scan)))
    assert rel_l1 < 5e-3, (
        f"F2-vs-scan rel-L1 = {rel_l1:.4%} at (N_m, N_χ) = ({n_m}, {n_chi}) "
        f"> 0.5 % PLAN gate"
    )


# ===========================================================================
# Gate 5 — AD on κ flows through F2 cache
# ===========================================================================
def test_ad_on_kappa_through_f2_matches_finite_difference():
    r"""``jax.grad(total_N)(κ)`` under Fisher + F2 matches finite-diff.

    The F2 cache build is part of the JAX trace inside both the
    forward Picard loop and the IFT VJP backward; ``κ`` flows
    through ``chi_scatter_marg`` (via per-orientation P_ret) and
    ``M_rem_mean_retained`` (via per-orientation Fisher weights).

    AD-vs-FD agreement at rel < 1e-5 confirms no κ-cotangent leak.
    This is the F2 analogue of F1's
    ``test_ad_on_kappa_via_cache_matches_finite_difference``.
    """
    def total_N(k):
        fisher = Fisher(kappa=k, n_theta=_N_THETA, n_dphi=_N_DPHI)
        r = _build(fisher)
        return _solve_with_f2(r).N.sum()

    k0 = jnp.asarray(2.0)
    g_ad = float(jax.grad(total_N)(k0))
    h = 1e-4
    g_fd = float((total_N(k0 + h) - total_N(k0 - h)) / (2.0 * h))

    assert jnp.isfinite(g_ad)
    assert g_ad != 0.0, (
        "∂N_sum/∂κ is zero under Fisher + F2 — cache build dropped "
        "the κ dependency from the trace"
    )
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
    assert rel < 1e-5, (
        f"AD vs FD on κ disagree by rel {rel:.2e} > 1e-5 "
        f"(AD={g_ad:.3e}, FD={g_fd:.3e})"
    )


def test_ad_on_v_esc_through_f2_matches_finite_difference():
    r"""``jax.grad(total_N)(v_esc)`` under Isotropic + F2 matches FD.

    Two-for-one AD-on-v_esc win from F1, preserved under F2.  The
    F2 cache build depends on v_esc via P_ret at every orientation
    point; ``vjp_theta_fn`` rebuilds the cache so the v_esc
    cotangent flows through.
    """
    iso = Isotropic(n_theta=_N_THETA, n_dphi=_N_DPHI)

    def total_N(v):
        r = _build(iso)
        return _solve_with_f2(r, v_esc=float(v)).N.sum()

    # Wrap the v_esc-dependent path through the solver carry, not
    # through ``_build`` (which calls Python float()).  Re-route:
    def total_N_traced(v):
        r = _build(iso)
        return solve_implicit_fixed_point(r, v, **_PICARD).N.sum()

    v0 = jnp.asarray(_V_ESC)
    g_ad = float(jax.grad(total_N_traced)(v0))
    hv = 0.1
    g_fd = float(
        (total_N_traced(v0 + hv) - total_N_traced(v0 - hv)) / (2.0 * hv)
    )

    assert jnp.isfinite(g_ad)
    assert g_ad != 0.0
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
    assert rel < 1e-5, (
        f"AD vs FD on v_esc disagree by rel {rel:.2e} > 1e-5 "
        f"(AD={g_ad:.3e}, FD={g_fd:.3e})"
    )


# ===========================================================================
# Gate 6 — Cache JIT-traceable end-to-end
# ===========================================================================
def test_f2_cache_jit_traceable():
    r"""``build_orientation_cache(R_chi_out=True)`` is JIT-traceable."""
    grid = BinaryGrid.default(m_min=_M_MIN, m_max=_M_MAX, n_m=8, n_chi=6)

    @jax.jit
    def go(kappa, v):
        fisher = Fisher(kappa=kappa, n_theta=_N_THETA, n_dphi=_N_DPHI)
        cache = build_orientation_cache(
            grid, fisher, v,
            retention_kind=_KIND, w_over_v_esc=_W_OVER, R_chi_out=True,
        )
        return (
            cache.chi_scatter_marg.sum()
            + cache.M_rem_mean_retained.sum()
            + cache.P_retain_marg.sum()
        )

    out = float(go(jnp.asarray(1.0), jnp.asarray(_V_ESC)))
    assert jnp.isfinite(out)
