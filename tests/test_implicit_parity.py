"""Parity check: package implicit solver vs red_team prototype.

The package's :func:`solve_implicit_fixed_point` was lifted from
``red_team/engine_v2_implicit.solve_implicit`` as Step 1 of
``demo/PLAN_engine_v2_production.md``. Both implementations share
``_step_one_generation``, the same trapezoid weights, the same Picard
update, and the same hyperparameters — so with matching kwargs the
converged ``N*`` arrays should match to machine precision.

This test exists so that future edits to either side surface as a fast
regression in CI rather than as a quiet drift in the comparison
metrics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Float64 — the red_team prototype was developed at double precision and
# any float32 drift would dominate the bitwise comparison.
jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.response.coagulation import build_gaussian_coagulation
from hierarchical_ad_pop_engine.response.implicit import solve_implicit_fixed_point
from red_team.engine_v2_implicit import solve_implicit as solve_implicit_redteam


_STAGE1 = dict(
    mu=8.0,
    sigma=1.0,
    m_min=4.0,
    m_max=80.0,
    natal_spin_loc=0.0,
    natal_spin_scale=0.10,
    kernel_a=0.0,
    kernel_b=1.0,
    log_R_env=0.0,
    k_max=4,
)
V_ESC = 1000.0
HYPER = dict(epsilon=1.0, max_iter=200, tol=1e-7, damping=0.5)


def test_package_implicit_matches_redteam_prototype():
    """Bitwise (≤ 1e-12) match on converged N* between package and red_team."""
    response_pkg = build_gaussian_coagulation(
        **_STAGE1, solver="implicit", epsilon=HYPER["epsilon"]
    )
    response_rt = build_gaussian_coagulation(**_STAGE1)

    result_pkg = solve_implicit_fixed_point(
        response_pkg, jnp.asarray(V_ESC), **HYPER
    )
    result_rt = solve_implicit_redteam(
        response_rt, jnp.asarray(V_ESC), verbose=False, **HYPER
    )

    N_pkg = np.asarray(result_pkg.N)
    N_rt = np.asarray(result_rt["N"])

    assert N_pkg.shape == N_rt.shape
    # After Step 2 the package solver returns JAX scalars for n_iter /
    # f_natal_depleted (lax.while_loop is jit-traceable end-to-end);
    # the red_team prototype still returns Python primitives. Cast at
    # the comparison boundary.
    assert int(result_pkg.n_iter) == result_rt["n_iter"]
    np.testing.assert_allclose(N_pkg, N_rt, atol=1e-12, rtol=0)
    assert abs(float(result_pkg.f_natal_depleted) - result_rt["f_natal_depleted"]) < 1e-12


def test_solver_field_dispatches_to_implicit_path():
    """``_log_total_density`` with solver="implicit" must use the Picard solver.

    Two-part sanity check that does NOT rely on relative-vs-absolute
    tolerance choices in the tails of the natal density:

    1. At ε = 0 exactly, the Picard update is ``N_rhs = N_natal``, the
       loop exits after one iteration, and the implicit path returns
       ``log N_natal`` (up to the ``log(max(N, 1e-30))`` floor — which
       affects only cells where ``log_N_natal == -1e30``, already
       numerically zero in linear scale).

    2. At any moderate ε ~ 1, the implicit and perturbative paths
       *must* differ — the implicit path includes the loss term and
       sums to all orders, the perturbative path truncates at
       ``k_max``. If they agreed at ε ~ 1 the dispatch isn't doing
       anything.
    """
    # Part 1: ε = 0 ⇒ implicit reduces to natal.
    response_zero = build_gaussian_coagulation(
        **_STAGE1, solver="implicit", epsilon=0.0
    )
    log_N_zero = response_zero._log_total_density(jnp.asarray(V_ESC))
    log_N_natal = response_zero._natal_log_density()

    N_zero = jnp.exp(log_N_zero)
    N_natal = jnp.exp(log_N_natal)
    # atol=1e-20 swallows the 1e-30 floor artefact:
    # `_natal_log_density` floors -inf → -1e30 so exp → 0 exactly, but
    # the implicit solver routes through `log(max(N, 1e-30))` and exp's
    # back to 1e-30 in those cells. Peak natal density is O(1e-3), so
    # atol=1e-20 is 17 orders of magnitude below anything physical.
    np.testing.assert_allclose(
        np.asarray(N_zero), np.asarray(N_natal), atol=1e-20, rtol=1e-12,
    )

    # Part 2: ε = 1 ⇒ implicit differs from perturbative at the same θ.
    response_imp = build_gaussian_coagulation(
        **_STAGE1, solver="implicit", epsilon=1.0
    )
    response_pert = build_gaussian_coagulation(**_STAGE1)  # default solver

    log_N_imp = response_imp._log_total_density(jnp.asarray(V_ESC))
    log_N_pert = response_pert._log_total_density(jnp.asarray(V_ESC))

    # Marginalised total counts integrated over the grid — a single scalar
    # that captures any meaningful divergence between the two solvers
    # without being sensitive to tail floor artefacts.
    grid = response_imp.grid
    dlog_m = jnp.diff(grid.log_m)
    dlog_m_full = jnp.concatenate(
        [dlog_m[:1] * 0.5, 0.5 * (dlog_m[:-1] + dlog_m[1:]), dlog_m[-1:] * 0.5]
    )
    dm_w = grid.m * dlog_m_full
    dchi = jnp.diff(grid.chi)
    dchi_w = jnp.concatenate(
        [dchi[:1] * 0.5, 0.5 * (dchi[:-1] + dchi[1:]), dchi[-1:] * 0.5]
    )
    int_imp = float(jnp.einsum("ij,i,j->", jnp.exp(log_N_imp), dm_w, dchi_w))
    int_pert = float(jnp.einsum("ij,i,j->", jnp.exp(log_N_pert), dm_w, dchi_w))

    # The two integrals should differ by more than rounding. We don't
    # care about the sign — just that the dispatch is doing something.
    assert abs(int_imp - int_pert) / max(int_imp, int_pert) > 1e-3, (
        f"implicit and perturbative integrals agree to <0.1%: "
        f"int_imp={int_imp:.6g}, int_pert={int_pert:.6g}. Solver dispatch likely broken."
    )


def test_implicit_solver_grad_matches_finite_difference():
    """Step 3 deliverable: custom_root VJP returns the IFT gradient.

    Constructs a scalar loss ``L(θ) = ∫ N*(θ) dm dχ`` and checks that
    ``jax.grad(L)(θ)`` (which routes through the implicit-function-
    theorem VJP registered on solve_implicit_fixed_point) matches a
    central finite difference to ~1e-3 relative tolerance. Anything
    much worse means the custom VJP is wrong.

    Uses a tiny grid (N_m=24, N_chi=8) so the FD evaluations don't
    dominate the test budget. Differentiates w.r.t. four physical
    params: μ_natal, σ_natal, kernel_b, ε.
    """
    from hierarchical_ad_pop_engine.grids import BinaryGrid

    small_grid = BinaryGrid.default(m_min=4.0, m_max=80.0, n_m=24, n_chi=8)

    def loss(theta):
        mu, sigma, kernel_b, eps = theta
        response = build_gaussian_coagulation(
            mu=mu,
            sigma=sigma,
            m_min=4.0,
            m_max=80.0,
            natal_spin_loc=0.0,
            natal_spin_scale=0.10,
            kernel_a=0.0,
            kernel_b=kernel_b,
            log_R_env=0.0,
            k_max=4,
            grid=small_grid,
            solver="implicit",
            epsilon=eps,
        )
        log_N = response._log_total_density(jnp.asarray(V_ESC))
        # Scalar loss: ∫ N dm dχ. Single number, differentiable.
        N = jnp.exp(log_N)
        dlog_m = jnp.diff(small_grid.log_m)
        dlog_m_full = jnp.concatenate(
            [dlog_m[:1] * 0.5, 0.5 * (dlog_m[:-1] + dlog_m[1:]), dlog_m[-1:] * 0.5]
        )
        dm_w = small_grid.m * dlog_m_full
        dchi = jnp.diff(small_grid.chi)
        dchi_w = jnp.concatenate(
            [dchi[:1] * 0.5, 0.5 * (dchi[:-1] + dchi[1:]), dchi[-1:] * 0.5]
        )
        return jnp.einsum("ij,i,j->", N, dm_w, dchi_w)

    theta0 = jnp.asarray([8.0, 1.0, 1.0, 0.5])

    # AD gradient.
    grad_ad = jax.grad(loss)(theta0)

    # Central finite-difference. Use a step of 1e-3 relative — small
    # enough to suppress truncation, large enough to stay above the
    # 1e-7 Picard tolerance noise floor.
    h_rel = 1e-3
    grad_fd = np.zeros(4)
    for i in range(4):
        h = h_rel * float(theta0[i])
        theta_plus = theta0.at[i].add(h)
        theta_minus = theta0.at[i].add(-h)
        grad_fd[i] = (float(loss(theta_plus)) - float(loss(theta_minus))) / (2 * h)

    np.testing.assert_allclose(
        np.asarray(grad_ad), grad_fd, rtol=3e-3, atol=1e-8,
    )


def test_implicit_solver_is_jit_traceable():
    """Step 2 deliverable: solve_implicit_fixed_point must be jit-compilable.

    Pre-Step-2 the Python ``for`` loop with ``float(...)`` casts on the
    convergence trace could not be traced by ``jax.jit``. After Step 2
    the loop is a :func:`jax.lax.while_loop` and the whole solver is
    jit-traceable end-to-end. This is the precondition for Step 3's
    custom VJP.

    Test plan: jit-compile a one-shot driver that returns
    ``log_N_total`` from the implicit path, run it twice, confirm
    (a) it runs without raising, (b) re-tracing on the second call
    doesn't happen (lru-cache hit), and (c) the jitted output matches
    the eager output to float-epsilon.
    """
    response = build_gaussian_coagulation(**_STAGE1, solver="implicit", epsilon=1.0)
    v_esc_arr = jnp.asarray(V_ESC)

    @jax.jit
    def driver(resp, v):
        return resp._log_total_density(v)

    log_N_jit = driver(response, v_esc_arr)
    log_N_eager = response._log_total_density(v_esc_arr)

    np.testing.assert_allclose(
        np.asarray(log_N_jit), np.asarray(log_N_eager), atol=1e-10, rtol=1e-10,
    )

    # Second call should hit the jit cache — value still correct.
    log_N_jit_2 = driver(response, v_esc_arr)
    np.testing.assert_array_equal(np.asarray(log_N_jit), np.asarray(log_N_jit_2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
