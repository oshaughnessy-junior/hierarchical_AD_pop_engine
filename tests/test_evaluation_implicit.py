r"""Tests for :class:`ImplicitEvaluator` — point-query API over an implicit solve.

What this suite pins
--------------------
1. **Grid-point equivalence.**  The new arbitrary-point evaluator must
   reproduce the bulk grid-quadrature path
   (:mod:`observables`) **exactly** when queried at the engine grid
   points.  This is the headline regression invariant — if it ever
   breaks, either the evaluator's interpolation or the underlying
   bulk path has drifted.

   *  ``ev.density`` at every ``(m_i, χ_j)`` equals ``result.N[i, j]``.
   *  ``ev.marginal_1d`` for each of the four grid-aligned axes at the
      kept-axis grid points equals ``observables.marginal_1d``'s
      ``dC_dx`` output.
   *  ``ev.marginal_2d`` for each of the six grid-aligned pairs at the
      kept-axes meshgrid equals ``observables.marginal_2d``'s ``dC``
      output.
   *  ``ev.total_rate`` equals ``observables.total_rate`` to bitwise
      precision.

2. **4-D rate density factorisation.**  ``ev.rate_density_4d`` at
   arbitrary points matches the explicit
   ``½ R_env K(m₁,m₂) N(m₁,χ₁) N(m₂,χ₂)`` formula built from the
   lowest-level primitives (kernel + density), giving an independent
   re-derivation rather than self-consistency.

3. **Out-of-grid handling.**  Query points outside the engine grid
   return 0 for ``density``, ``marginal_1d``, ``marginal_2d`` and
   ``rate_density_4d`` — same density-of-zero convention as
   :func:`log_density_at`.

4. **Derived-axis axes raise actionably.**  Step-1 deliberately ships
   only the four grid-aligned axes; querying ``'chi_eff'`` or ``'q'``
   etc. raises :class:`NotImplementedError` with a pointer to the
   observables-side 2-D derived-axis helpers.

5. **Protocol conformance.**  The evaluator passes
   ``isinstance(ev, RateDensityEvaluator)`` under the
   ``runtime_checkable`` Protocol.

6. **AD path is intact.**  ``jax.grad`` of a scalar reduction over
   ``ev.density`` runs to completion without an integer-tangent or
   tracer error.  We do not pin a numerical gradient here — the
   implicit-function VJP is exercised by ``test_implicit_parity.py``
   and ``test_gradients.py``; we just want to confirm the new
   evaluator layer does not break the AD chain.

The fixture uses a small Gaussian natal mass + implicit solver — same
recipe as ``test_implicit_parity.py``, scaled down for speed.  Anything
that converges and gives a non-degenerate ``N*`` works.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine import observables as obs
from hierarchical_ad_pop_engine.evaluation import (
    ImplicitEvaluator,
    RateDensityEvaluator,
)
from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.kernels import log_pairing_kernel
from hierarchical_ad_pop_engine.response.coagulation import (
    build_gaussian_coagulation,
)
from hierarchical_ad_pop_engine.response.implicit import (
    ImplicitSolveResult,
    solve_implicit_fixed_point,
)


# ---------------------------------------------------------------------------
# Fixtures — small implicit solve.  Same recipe as test_implicit_parity.py.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_response():
    r"""Small Gaussian-natal response with the implicit solver."""
    return build_gaussian_coagulation(
        mu=8.0,
        sigma=1.0,
        m_min=4.0,
        m_max=80.0,
        natal_spin_loc=0.0,
        natal_spin_scale=0.10,
        kernel_a=0.0,
        kernel_b=1.0,
        log_R_env=0.5,
        k_max=4,
        solver="implicit",
        epsilon=1.0,
        grid=BinaryGrid.default(m_min=4.0, m_max=80.0, n_m=20, n_chi=8),
    )


@pytest.fixture(scope="module")
def implicit_result(small_response) -> ImplicitSolveResult:
    return solve_implicit_fixed_point(
        small_response,
        jnp.asarray(1000.0),
        epsilon=1.0,
        max_iter=200,
        tol=1e-8,
        damping=0.5,
    )


@pytest.fixture(scope="module")
def evaluator(small_response, implicit_result) -> ImplicitEvaluator:
    return implicit_result.evaluator(small_response)


# ---------------------------------------------------------------------------
# 1. Grid-point equivalence — density.
# ---------------------------------------------------------------------------
def test_density_at_grid_points_matches_result_N(
    small_response, implicit_result, evaluator
):
    r"""``ev.density(m_i, χ_j)`` == ``result.N[i, j]`` at every grid point."""
    grid = small_response.grid
    m_mesh, chi_mesh = jnp.meshgrid(grid.m, grid.chi, indexing="ij")
    N_eval = evaluator.density(m_mesh, chi_mesh)
    np.testing.assert_allclose(
        np.asarray(N_eval),
        np.asarray(implicit_result.N),
        rtol=1e-12,
        atol=1e-300,
    )


# ---------------------------------------------------------------------------
# 1. Grid-point equivalence — marginals.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("axis", list(obs.VALID_AXES))
def test_marginal_1d_at_grid_points_matches_observables(
    small_response, implicit_result, evaluator, axis
):
    r"""``ev.marginal_1d(axis, x_grid)`` reproduces ``observables.marginal_1d``."""
    x_grid, dC_dx_ref = obs.marginal_1d(
        implicit_result.N,
        small_response.grid,
        small_response.kernel_params,
        small_response.log_R_env,
        axis=axis,
    )
    dC_dx_eval = evaluator.marginal_1d(axis, x_grid)
    np.testing.assert_allclose(
        np.asarray(dC_dx_eval),
        np.asarray(dC_dx_ref),
        rtol=1e-12,
        atol=1e-300,
    )


_PAIRS = [
    (obs.AXIS_M1, obs.AXIS_M2),
    (obs.AXIS_M1, obs.AXIS_CHI1),
    (obs.AXIS_M1, obs.AXIS_CHI2),
    (obs.AXIS_M2, obs.AXIS_CHI1),
    (obs.AXIS_M2, obs.AXIS_CHI2),
    (obs.AXIS_CHI1, obs.AXIS_CHI2),
]


@pytest.mark.parametrize("axes", _PAIRS)
def test_marginal_2d_at_grid_points_matches_observables(
    small_response, implicit_result, evaluator, axes
):
    r"""``ev.marginal_2d(axes, x_grid, y_grid)`` reproduces
    ``observables.marginal_2d``."""
    x_grid, y_grid, dC_ref = obs.marginal_2d(
        implicit_result.N,
        small_response.grid,
        small_response.kernel_params,
        small_response.log_R_env,
        axes=axes,
    )
    x_mesh, y_mesh = jnp.meshgrid(x_grid, y_grid, indexing="ij")
    dC_eval = evaluator.marginal_2d(axes, x_mesh, y_mesh)
    np.testing.assert_allclose(
        np.asarray(dC_eval),
        np.asarray(dC_ref),
        rtol=1e-12,
        atol=1e-300,
    )


def test_total_rate_matches_observables(small_response, implicit_result, evaluator):
    ref = obs.total_rate(
        implicit_result.N,
        small_response.grid,
        small_response.kernel_params,
        small_response.log_R_env,
    )
    got = evaluator.total_rate()
    np.testing.assert_allclose(float(got), float(ref), rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. 4-D rate density — factorisation against explicit ½ R_env K N N.
# ---------------------------------------------------------------------------
def test_rate_density_4d_factorisation_at_grid_points(
    small_response, implicit_result, evaluator
):
    r"""``rate_density_4d`` == ½ R_env K(m1,m2) N(m1,χ1) N(m2,χ2).

    Built from the lowest-level primitives (point ``log_pairing_kernel``
    and ``result.N`` directly) so this is an independent
    re-implementation, not a self-consistency check.
    """
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env
    R_env = float(jnp.exp(jnp.asarray(log_R)))

    # Pick 5 random index quadruples from the grid corners
    rng = np.random.default_rng(20260522)
    i1 = rng.integers(0, grid.m.size, size=5)
    j1 = rng.integers(0, grid.chi.size, size=5)
    i2 = rng.integers(0, grid.m.size, size=5)
    j2 = rng.integers(0, grid.chi.size, size=5)

    m1 = grid.m[i1]
    m2 = grid.m[i2]
    chi1 = grid.chi[j1]
    chi2 = grid.chi[j2]

    got = evaluator.rate_density_4d(m1, m2, chi1, chi2)

    log_K = log_pairing_kernel(m1, m2, **kp)
    N1 = implicit_result.N[i1, j1]
    N2 = implicit_result.N[i2, j2]
    ref = 0.5 * R_env * jnp.exp(log_K) * N1 * N2

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-300
    )


# ---------------------------------------------------------------------------
# 3. Out-of-grid → zero.
# ---------------------------------------------------------------------------
def test_density_outside_grid_returns_zero(small_response, evaluator):
    grid = small_response.grid
    m_lo = jnp.asarray(grid.m[0] * 0.5)
    m_hi = jnp.asarray(grid.m[-1] * 2.0)
    chi_mid = jnp.asarray(0.5 * (grid.chi[0] + grid.chi[-1]))
    chi_lo = jnp.asarray(grid.chi[0] - 1.0)  # < 0
    chi_hi = jnp.asarray(grid.chi[-1] + 1.0)  # > 1

    # mass out-of-range (chi inside)
    assert float(evaluator.density(m_lo, chi_mid)) == 0.0
    assert float(evaluator.density(m_hi, chi_mid)) == 0.0
    # chi out-of-range (mass inside)
    m_mid = jnp.asarray(0.5 * (grid.m[0] + grid.m[-1]))
    assert float(evaluator.density(m_mid, chi_lo)) == 0.0
    assert float(evaluator.density(m_mid, chi_hi)) == 0.0


def test_marginal_1d_outside_grid_returns_zero(small_response, evaluator):
    grid = small_response.grid
    x_outside = jnp.asarray([grid.m[0] * 0.5, grid.m[-1] * 2.0])
    out = evaluator.marginal_1d("m1", x_outside)
    np.testing.assert_array_equal(np.asarray(out), np.zeros_like(np.asarray(out)))

    chi_outside = jnp.asarray([grid.chi[0] - 0.1, grid.chi[-1] + 0.1])
    out_chi = evaluator.marginal_1d("chi1", chi_outside)
    np.testing.assert_array_equal(
        np.asarray(out_chi), np.zeros_like(np.asarray(out_chi))
    )


def test_marginal_2d_outside_grid_returns_zero(small_response, evaluator):
    grid = small_response.grid
    x_outside = jnp.asarray([grid.m[0] * 0.5])
    y_outside = jnp.asarray([grid.chi[0] - 0.1])
    out = evaluator.marginal_2d(("m1", "chi1"), x_outside, y_outside)
    np.testing.assert_array_equal(np.asarray(out), np.zeros_like(np.asarray(out)))


def test_rate_density_4d_outside_grid_returns_zero(small_response, evaluator):
    grid = small_response.grid
    # One progenitor outside → ½ R_env K · 0 · N = 0.
    m1 = jnp.asarray(grid.m[0] * 0.5)
    m2 = jnp.asarray(grid.m[5])
    chi1 = jnp.asarray(0.5 * (grid.chi[0] + grid.chi[-1]))
    chi2 = jnp.asarray(0.5 * (grid.chi[0] + grid.chi[-1]))
    assert float(evaluator.rate_density_4d(m1, m2, chi1, chi2)) == 0.0


# ---------------------------------------------------------------------------
# 4. Derived axes raise actionably.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "axis", ["q", "chi_eff", "M_chirp", "m_heavy", "m_light"]
)
def test_marginal_1d_derived_axes_raise(evaluator, axis):
    with pytest.raises(NotImplementedError, match="derived axis"):
        evaluator.marginal_1d(axis, jnp.asarray([0.5]))


# ---------------------------------------------------------------------------
# 5. Protocol conformance.
# ---------------------------------------------------------------------------
def test_protocol_runtime_isinstance(evaluator):
    assert isinstance(evaluator, RateDensityEvaluator)


# ---------------------------------------------------------------------------
# 6. AD smoke test — gradient through density runs.
# ---------------------------------------------------------------------------
def test_density_grad_runs(small_response):
    r"""``jax.grad`` of a scalar reduction over ``ev.density`` completes.

    Numerical correctness of the implicit-function VJP is pinned by
    ``test_gradients.py``; we only want to confirm the evaluator layer
    does not break the AD chain (e.g. by introducing a non-traceable
    branch).
    """

    def loss(log_R_env_val):
        response = build_gaussian_coagulation(
            mu=8.0, sigma=1.0, m_min=4.0, m_max=80.0,
            natal_spin_loc=0.0, natal_spin_scale=0.10,
            kernel_a=0.0, kernel_b=1.0,
            log_R_env=log_R_env_val,
            k_max=4, solver="implicit", epsilon=1.0,
            grid=BinaryGrid.default(m_min=4.0, m_max=80.0, n_m=20, n_chi=8),
        )
        result = solve_implicit_fixed_point(
            response, jnp.asarray(1000.0), epsilon=1.0,
            max_iter=200, tol=1e-8, damping=0.5,
        )
        ev = result.evaluator(response)
        # Use a meshgrid so m / chi shapes broadcast cleanly.
        m_q, chi_q = jnp.meshgrid(
            response.grid.m[::4], response.grid.chi[::2], indexing="ij"
        )
        return jnp.sum(ev.density(m_q, chi_q))

    g = jax.grad(loss)(jnp.asarray(0.5))
    assert jnp.isfinite(g)
