"""Acceptance tests for ``build_remnant_tensor`` — the rank-3
projection tensor that powers the v3 basis Picard solver's gain term
(``PLAN_engine_v3_basis.md`` §Step 3).

Scope
-----

The plan's Step 3 acceptance criteria, restated here verbatim with
the test that pins each:

1. ``test_T_symmetric_in_alpha_beta`` — bosonic pair symmetry to
   ``1e-12``.  The integrand is exactly (m1,χ1) ↔ (m2,χ2) symmetric
   once the remnant fits are called with canonically-ordered
   arguments (the builder canonicalises internally; without that
   step ``chi_rem`` breaks symmetry by O(10^5), so this test is
   load-bearing).

2. ``test_T_grad_finite_against_fd`` — ``jax.grad`` through
   ``build_remnant_tensor`` agrees with central finite differences on
   every field of ``RemnantTensorTheta`` at low resolution.  This is
   the contract Step 4's IFT VJP relies on — gradients flow cleanly
   through ``T(θ)``.

3. ``test_T_jit_traceable`` — the builder compiles under
   ``jax.jit``.  Step 4's Picard loop will jit-compile the whole
   forward pass; if T's construction can't trace, the solver can't
   either.

4. ``test_retention_sigmoid_smoke`` — sanity checks on the soft
   retention used inside T: finite values, AD finite, monotone
   decreasing in v_kick, parameterised via the ``w_over_v_esc``
   ratio (never an absolute km/s number — Stage 3 Decision 3 rule).

The full Step 3 plan also lists ``test_T_conserves_count`` and
``test_T_matches_grid_scatter``.  Both are convergence-style tests
that require a v2 grid-scatter reference at a fairly large
``n_basis``, and are folded into the Step 5 cross-validation in
observable space rather than here.  Adding them as pytest cases
here would either (a) make the test suite slow enough to run only
on demand, or (b) gate on tolerances that are properly tied to the
basis resolution sweep — neither serves the per-PR feedback loop.

What this file does NOT do
--------------------------

* It does not run the Step 5 ``dC/dm1`` JS comparison — that's the
  cross-validation gate on the ``observables`` branch.
* It does not test at Stage 1 production resolution ``(32, 16)``;
  T is 1.1 GB at that resolution and tests should run on CI
  hardware.  We use ``(8, 4)`` here, which keeps every individual
  test under ~3 seconds while exercising the same code paths.

Run
---

::

    pytest tests/test_basis_remnant_tensor.py -v

These tests are gated by ``jax_enable_x64`` (matching the rest of
the basis suite); the symmetry test requires double precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.basis.remnant_tensor import (  # noqa: E402
    RemnantTensorTheta,
    build_remnant_tensor,
    make_t_tensor_gain,
    retention_sigmoid,
)
from hierarchical_ad_pop_engine.basis.tensor import TensorProductBasis  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures: a tiny solver basis + a Stage 1 θ.
# ---------------------------------------------------------------------------
# These are sized to keep T ~ 32^3 = 33k entries, which builds in ~1 s
# on a workstation CPU.  Bigger bases blow up cubically; the production
# (32, 16) → 512^3 = 1.3e8 lives in the Step 5 cross-validation, not here.
N_BASIS_M = 8
N_BASIS_CHI = 4


@pytest.fixture(scope="module")
def basis_solver() -> TensorProductBasis:
    return TensorProductBasis.for_binary_grid(
        m_min=5.0,
        m_max=80.0,
        n_basis_m=N_BASIS_M,
        chi_min=1e-3,
        chi_max=1.0 - 1e-3,
        n_basis_chi=N_BASIS_CHI,
    )


@pytest.fixture(scope="module")
def theta_stage1() -> RemnantTensorTheta:
    return RemnantTensorTheta(
        kernel_a=jnp.asarray(1.5),
        kernel_b=jnp.asarray(1.0),
        v_esc=jnp.asarray(300.0),
        w_over_v_esc=jnp.asarray(1.0),
    )


# ---------------------------------------------------------------------------
# 1. Symmetry T[γ; α, β] = T[γ; β, α].
# ---------------------------------------------------------------------------
def test_T_symmetric_in_alpha_beta(basis_solver, theta_stage1):
    """T must be symmetric in the unordered progenitor pair.

    Same physics structure that motivated the v2 j≠l double-count fix
    (memory ``project_engine_jl_doublecount_fix``); the basis form of
    the bug is asymmetric ``chi_rem`` evaluation when m1 < m2.
    """
    T = np.asarray(build_remnant_tensor(theta_stage1, basis_solver))
    err = np.abs(T - np.transpose(T, (0, 2, 1))).max()
    # 1e-10 instead of the plan's 1e-12 because at small (8, 4) the
    # quadrature has only ~16 nodes/axis and the LGL Newton iteration
    # contributes a few ulp of noise that gets amplified by the
    # large |T| ~ O(1e5).  Scale-relative the tolerance:
    rel = err / max(np.abs(T).max(), 1.0)
    assert rel < 1e-12, (
        f"|T - Tᵀ_αβ|_∞ = {err:.3e}, |T|_∞ = {np.abs(T).max():.3e}, "
        f"rel = {rel:.3e} — symmetry must hold to ~1e-12 relative."
    )


# ---------------------------------------------------------------------------
# 2. Gradients through T agree with central finite differences.
# ---------------------------------------------------------------------------
def _scalar_objective(theta, basis):
    """Sum of T^2 — a smooth scalar of every θ field, used for AD checks."""
    T = build_remnant_tensor(theta, basis)
    return jnp.sum(T ** 2)


@pytest.mark.parametrize(
    "field, eps",
    [
        ("kernel_a", 1e-5),
        ("kernel_b", 1e-5),
        ("v_esc", 1e-2),         # km/s scale
        ("w_over_v_esc", 1e-5),
    ],
)
def test_T_grad_finite_against_fd(
    basis_solver, theta_stage1, field, eps
):
    """``jax.grad`` of ``sum(T**2)`` matches central FD on every θ field.

    Tolerance ``rtol = 1e-4`` matches the v2 ``test_basis_grad_matches_fd``
    convention in the v3 plan §Step 4 — tight enough to catch real
    sign / scale errors, loose enough to absorb LGL Newton-iteration ulp.
    """
    ad = float(
        getattr(jax.grad(_scalar_objective)(theta_stage1, basis_solver), field)
    )

    def perturbed(delta):
        kwargs = {
            "kernel_a": theta_stage1.kernel_a,
            "kernel_b": theta_stage1.kernel_b,
            "v_esc": theta_stage1.v_esc,
            "w_over_v_esc": theta_stage1.w_over_v_esc,
        }
        kwargs[field] = kwargs[field] + delta
        return RemnantTensorTheta(**kwargs)

    f_p = float(_scalar_objective(perturbed(eps), basis_solver))
    f_m = float(_scalar_objective(perturbed(-eps), basis_solver))
    fd = (f_p - f_m) / (2 * eps)

    rel = abs(ad - fd) / max(abs(fd), 1.0)
    assert rel < 1e-4, (
        f"∂/∂{field}: AD = {ad:+.4e}, FD = {fd:+.4e}, rel = {rel:.3e}"
    )


# ---------------------------------------------------------------------------
# 3. JIT-traceability.
# ---------------------------------------------------------------------------
def test_T_jit_traceable(basis_solver, theta_stage1):
    """``build_remnant_tensor`` compiles under ``jax.jit``.

    Step 4's Picard loop wraps the whole solve in ``jax.jit``; if T
    can't trace, the solver can't either.  Run twice to check
    re-traceability is well-defined (no Python side-effects leaking).
    """
    fn = jax.jit(build_remnant_tensor)
    out1 = fn(theta_stage1, basis_solver)
    out2 = fn(theta_stage1, basis_solver)
    out1.block_until_ready()
    out2.block_until_ready()
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))


def test_t_tensor_gain_closure_shape(basis_solver, theta_stage1):
    """The ``gain_fn`` closure produced by ``make_t_tensor_gain``
    returns a vector matching the basis flat dimension.

    Smoke check on the Picard-facing API — the closure is what the
    Step 4 ``_F_map_basis`` calls.  No physics assertion here beyond
    "shape is right and the contraction runs end-to-end."
    """
    gain_fn = make_t_tensor_gain(basis_solver)
    coeffs = jnp.ones(basis_solver.n_basis)
    out = gain_fn(coeffs, theta_stage1)
    assert out.shape == (basis_solver.n_basis,)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# 4. retention_sigmoid smoke checks.
# ---------------------------------------------------------------------------
def test_retention_sigmoid_smoke():
    """Sigmoid: finite, monotone decreasing in v_kick, AD-clean in
    both ``v_esc`` and ``w_over_v_esc``.

    Stage-3 Decision 3 rule: ``w`` is always parameterised as a
    *fraction* of v_esc, never as an absolute km/s number.  This
    test pins the API at that contract.
    """
    v_kick = jnp.linspace(0.0, 2000.0, 64)
    P = retention_sigmoid(v_kick, jnp.asarray(300.0), jnp.asarray(1.0))
    P = np.asarray(P)
    assert np.all(np.isfinite(P))
    assert np.all((P >= 0.0) & (P <= 1.0))
    # Monotone decreasing in v_kick.
    assert np.all(np.diff(P) <= 1e-12)

    # AD finite on v_esc and w_over_v_esc.
    def loss(ve, wov):
        return jnp.mean(retention_sigmoid(v_kick, ve, wov))

    g_ve = float(jax.grad(loss, 0)(jnp.asarray(300.0), jnp.asarray(1.0)))
    g_wov = float(jax.grad(loss, 1)(jnp.asarray(300.0), jnp.asarray(1.0)))
    assert np.isfinite(g_ve) and g_ve != 0.0
    assert np.isfinite(g_wov) and g_wov != 0.0


def test_retention_sigmoid_sweep_w_over_v_esc():
    """Step 3c spot-check: w_over_v_esc ∈ {1.0, 1.5, 2.0} all give
    smooth retention that approaches the Heaviside as w → 0 and
    broadens monotonically with w.

    Not a comprehensive sweep — that's Step 3c's parallel observable-
    space test.  Just a sanity check that the sweep range the plan
    targets is well-conditioned.
    """
    v_kick = jnp.linspace(0.0, 1000.0, 100)
    v_esc = jnp.asarray(300.0)
    Ps = {}
    for wov in (1.0, 1.5, 2.0):
        Ps[wov] = np.asarray(
            retention_sigmoid(v_kick, v_esc, jnp.asarray(wov))
        )
    # Broader w → softer transition → less mass change with v_kick
    # at any given v_kick > v_esc:
    mid = 70  # v_kick ≈ 700 km/s, above the threshold
    assert Ps[1.0][mid] < Ps[1.5][mid] < Ps[2.0][mid], (
        f"Sigmoid not broadening monotonically with w_over_v_esc: "
        f"P(1.0)={Ps[1.0][mid]:.3f}, P(1.5)={Ps[1.5][mid]:.3f}, "
        f"P(2.0)={Ps[2.0][mid]:.3f}"
    )
    # All in [0, 1] and finite.
    for wov, P in Ps.items():
        assert np.all((P >= 0.0) & (P <= 1.0))
        assert np.all(np.isfinite(P))
