"""Acceptance tests for the v3 basis Picard solver
(:func:`response.basis.solve_basis_fixed_point`) — Step 4.

The plan's Step 4 acceptance criteria, restated here:

1. ``test_basis_solver_jit_traceable`` — the solver runs under
   ``jax.jit``.  Step 5's observable-space comparison will jit the
   whole forward pass.
2. ``test_basis_perturbative_limit`` — at ε → 0 the solver
   reproduces ``coeffs ≈ n_natal_α`` to a tolerance set by ε times
   the bilinear-operator magnitude.  Verifies the Picard fixed
   point is correctly seeded and that the steady-state equation
   has no spurious offsets.
3. ``test_basis_grad_matches_fd`` — ``jax.grad`` of a scalar
   observable through the solver matches central finite differences
   at low resolution.  This is the contract Step 4 ships: IFT VJP
   gives the same gradient as unrolling the loop, modulo IFT GMRES
   tol.

The plan's fourth Step 4 acceptance is the v2-vs-basis observable-
space JS comparison; that's Step 5, on the parallel ``observables``
branch.

Scope and sizing
----------------

These tests run at solver basis ``(n_basis_m, n_basis_χ) = (8, 4)``.
The bare kernel ``K = (m₁+m₂)^a · q^b`` is unnormalised in this test
configuration (no ``log_R_env`` scaling), so the absolute ε that
corresponds to "fully developed cascade" is small — we use ε in
``[1e-8, 1e-4]`` here.  Bigger ε values need a bigger basis to
represent the resulting cascade; that's a Step 5 question, not a
Step 4 one.

Float-64 mode is required (the IFT GMRES backward is set to
``tol = 1e-7`` in :mod:`response.basis` and float32 doesn't have
the headroom for the AD-vs-FD comparison).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.basis.natal import project_natal_density  # noqa: E402
from hierarchical_ad_pop_engine.basis.tensor import TensorProductBasis  # noqa: E402
from hierarchical_ad_pop_engine.grids import BinaryGrid  # noqa: E402
from hierarchical_ad_pop_engine.response.basis import (  # noqa: E402
    BasisSolveResult,
    solve_basis_fixed_point,
)
from hierarchical_ad_pop_engine.response.coagulation import (  # noqa: E402
    CoagulationResponse,
)


# ---------------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def basis_solver() -> TensorProductBasis:
    return TensorProductBasis.for_binary_grid(
        m_min=5.0,
        m_max=80.0,
        n_basis_m=8,
        chi_min=1e-3,
        chi_max=1.0 - 1e-3,
        n_basis_chi=4,
    )


@pytest.fixture(scope="module")
def response_gaussian() -> CoagulationResponse:
    """Minimal CoagulationResponse with Gaussian natal — avoids the
    PL+Peak boundary-taper code path that depends on gwkokab
    primitives.
    """
    grid = BinaryGrid.default()
    return CoagulationResponse(
        grid=grid,
        k_max=4,
        natal_mass_params=dict(
            mu=jnp.asarray(20.0),
            sigma=jnp.asarray(3.0),
            m_min=jnp.asarray(5.0),
            m_max=jnp.asarray(80.0),
        ),
        natal_spin_params=dict(
            loc=jnp.asarray(0.0),
            scale=jnp.asarray(0.15),
        ),
        kernel_params=dict(
            a=jnp.asarray(1.5),
            b=jnp.asarray(1.0),
        ),
        log_R_env=jnp.asarray(0.0),
        natal_mass_kind="gaussian",
        solver="basis",
        epsilon=jnp.asarray(1.0),
    )


# ---------------------------------------------------------------------------
# 1. Picard converges, n_iter reasonable, output shape correct.
# ---------------------------------------------------------------------------
def test_basis_solver_converges(basis_solver, response_gaussian):
    """Forward solve at small ε produces a valid BasisSolveResult."""
    result = solve_basis_fixed_point(
        response_gaussian,
        basis_solver,
        v_esc=jnp.asarray(300.0),
        epsilon=jnp.asarray(1e-4),
        w_over_v_esc=jnp.asarray(1.0),
        max_iter=200,
        tol=1e-7,
        damping=0.5,
    )
    assert isinstance(result, BasisSolveResult)
    assert result.coeffs.shape == (basis_solver.n_basis,)
    assert bool(jnp.all(jnp.isfinite(result.coeffs)))
    assert int(result.n_iter) < 200, (
        f"Picard did not converge: n_iter = {int(result.n_iter)}"
    )
    assert float(result.final_diff) < 1e-7, (
        f"Picard did not reach tol: final_diff = {float(result.final_diff)}"
    )


# ---------------------------------------------------------------------------
# 2. Perturbative limit: ε → 0 gives coeffs → n_natal_α.
# ---------------------------------------------------------------------------
def test_basis_perturbative_limit(basis_solver, response_gaussian):
    """At ε = 1e-8 the converged coefficients differ from the natal
    coefficients by O(ε · |gain - loss|).

    For this fixture, ``|gain - loss|`` at the natal is ≈ 15 in absolute
    units, ``||n_natal_α||_∞`` ≈ 0.037, so the expected relative
    deviation is ≈ ε · 15 / 0.037 ≈ 4e-6 at ε = 1e-8.  We allow a
    factor of 10 headroom for the small (8, 4) basis representation
    noise.
    """
    n_natal_alpha = np.asarray(
        project_natal_density(response_gaussian, basis_solver).reshape(-1)
    )
    result = solve_basis_fixed_point(
        response_gaussian,
        basis_solver,
        v_esc=jnp.asarray(300.0),
        epsilon=jnp.asarray(1e-8),
        w_over_v_esc=jnp.asarray(1.0),
        max_iter=200,
        tol=1e-12,
        damping=0.5,
    )
    coeffs = np.asarray(result.coeffs)
    rel = np.max(np.abs(coeffs - n_natal_alpha)) / max(
        np.max(np.abs(n_natal_alpha)), 1.0
    )
    assert rel < 1e-4, (
        f"Perturbative limit failed at ε = 1e-8: rel = {rel:.3e}"
    )


# ---------------------------------------------------------------------------
# 3. AD through the solve matches FD.
# ---------------------------------------------------------------------------
def test_basis_grad_matches_fd_epsilon(basis_solver, response_gaussian):
    """``jax.grad`` of ``Σ coeffs²`` w.r.t. epsilon agrees with
    central FD at low resolution.

    Cheapest single param to perturb; v_esc and w_over_v_esc are
    exercised in the smoke test below for finite-and-nonzero.
    """
    def loss_fn(epsilon):
        res = solve_basis_fixed_point(
            response_gaussian,
            basis_solver,
            v_esc=jnp.asarray(300.0),
            epsilon=epsilon,
            w_over_v_esc=jnp.asarray(1.0),
            max_iter=200,
            tol=1e-7,
            damping=0.5,
        )
        return jnp.sum(res.coeffs ** 2)

    eps0 = jnp.asarray(1e-4)
    ad = float(jax.grad(loss_fn)(eps0))
    delta = 1e-6
    f_p = float(loss_fn(jnp.asarray(1e-4 + delta)))
    f_m = float(loss_fn(jnp.asarray(1e-4 - delta)))
    fd = (f_p - f_m) / (2 * delta)
    rel = abs(ad - fd) / max(abs(fd), 1.0)
    assert rel < 1e-3, (
        f"AD ≠ FD on epsilon: AD = {ad:+.4e}, FD = {fd:+.4e}, rel = {rel:.3e}"
    )


def test_basis_grad_finite_for_each_input(basis_solver, response_gaussian):
    """``jax.grad`` returns finite values for every differentiable
    input — v_esc, w_over_v_esc, epsilon.

    The soft-sigmoid retention specifically makes ``∂/∂v_esc``
    finite (v2's Heaviside gives zero gradient w.r.t. v_esc) — this
    test pins that property.
    """
    def loss_fn(v_esc, w_over_v_esc, epsilon):
        res = solve_basis_fixed_point(
            response_gaussian,
            basis_solver,
            v_esc=v_esc,
            epsilon=epsilon,
            w_over_v_esc=w_over_v_esc,
            max_iter=200,
            tol=1e-7,
            damping=0.5,
        )
        return jnp.sum(res.coeffs ** 2)

    g_ve = float(jax.grad(loss_fn, 0)(
        jnp.asarray(300.0), jnp.asarray(1.0), jnp.asarray(1e-4)
    ))
    g_wov = float(jax.grad(loss_fn, 1)(
        jnp.asarray(300.0), jnp.asarray(1.0), jnp.asarray(1e-4)
    ))
    g_eps = float(jax.grad(loss_fn, 2)(
        jnp.asarray(300.0), jnp.asarray(1.0), jnp.asarray(1e-4)
    ))
    for name, val in [("v_esc", g_ve), ("w_over_v_esc", g_wov), ("epsilon", g_eps)]:
        assert np.isfinite(val), f"grad wrt {name} is not finite: {val}"
    # v_esc gradient should be NON-ZERO — soft sigmoid makes it
    # differentiable.  v2's Heaviside gives exactly zero here.
    assert g_ve != 0.0, "∂/∂v_esc is zero — soft sigmoid is supposed to make this nonzero"
    assert g_wov != 0.0, "∂/∂w_over_v_esc is zero"


# ---------------------------------------------------------------------------
# 4. JIT-traceability.
# ---------------------------------------------------------------------------
def test_basis_solver_jit_traceable(basis_solver, response_gaussian):
    """``solve_basis_fixed_point`` (forward path) compiles under
    ``jax.jit`` and yields finite coefficients.

    The custom-VJP plumbing inside the solver is also exercised by
    the AD tests above, but those don't necessarily go through jit;
    this one does.
    """
    def solve_for_jit(v_esc, w_over_v_esc, epsilon):
        return solve_basis_fixed_point(
            response_gaussian,
            basis_solver,
            v_esc=v_esc,
            epsilon=epsilon,
            w_over_v_esc=w_over_v_esc,
            max_iter=200,
            tol=1e-7,
            damping=0.5,
        ).coeffs

    fn = jax.jit(solve_for_jit)
    out = fn(jnp.asarray(300.0), jnp.asarray(1.0), jnp.asarray(1e-4))
    out.block_until_ready()
    assert out.shape == (basis_solver.n_basis,)
    assert bool(jnp.all(jnp.isfinite(out)))
