r"""Tests for :class:`BasisEvaluator` — point-query API over a v3 basis solve.

What this suite pins
--------------------
1. **Grid-point equivalence — density.**  ``ev.density`` at every
   ``(m_i, χ_j)`` of the engine ``BinaryGrid`` equals
   :func:`observables.density_from_basis` at the same point.  This is
   the headline regression invariant — the breadcrumb from
   ``project_evaluation_step1_done`` named ``density_from_basis``
   explicitly as the Step-2 regression target.

2. **Grid-point equivalence — marginals + total rate.**  Same
   two-layer pattern as the implicit evaluator: ``ev.marginal_1d`` at
   the kept-axis grid points and ``ev.marginal_2d`` at the kept-axes
   meshgrid reproduce :func:`observables.marginal_1d` /
   ``marginal_2d`` applied to the same basis-evaluated grid density;
   ``ev.total_rate`` matches :func:`observables.total_rate`.  Together
   with (1), this means the new API agrees with
   :func:`observables.marginals_from_basis` exactly at every grid
   point — and bilinear-interpolates everywhere else.

3. **4-D rate density factorisation.**  ``ev.rate_density_4d`` at
   arbitrary grid-aligned points matches the explicit
   ``½ R_env K(m₁,m₂) N(m₁,χ₁) N(m₂,χ₂)`` formula built from the
   lowest-level primitives (:func:`log_pairing_kernel` +
   :func:`density_from_basis` directly).  Pins the "K analytic, not
   Galerkin" decision — see PLAN § "Open-question resolutions".

4. **Out-of-domain handling.**  Queries outside the basis rectangle
   return 0 for ``density``, ``marginal_1d``, ``marginal_2d`` and
   ``rate_density_4d``.

5. **Derived-axis axes raise actionably.**  Step-2 ships only the
   four grid-aligned axes; querying ``'chi_eff'`` or ``'q'`` etc.
   raises :class:`NotImplementedError` with a pointer to the
   observables-side 2-D derived-axis helpers.

6. **Protocol conformance + AD smoke.**  The evaluator passes
   ``isinstance(ev, RateDensityEvaluator)`` under
   ``runtime_checkable``; ``jax.grad`` of a scalar reduction over
   ``ev.density`` runs to completion.  Numerical correctness of the
   basis IFT VJP is pinned by ``test_basis_solver.py``; we only
   confirm the evaluator layer does not break the AD chain.

The fixture uses a small Gaussian natal mass + the basis Picard
solver on a coarse (8 × 6) basis with the engine grid at (20, 8) so
the suite is fast.  Anything that converges and gives a non-degenerate
``coeffs`` works.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine import observables as obs
from hierarchical_ad_pop_engine.basis.tensor import TensorProductBasis
from hierarchical_ad_pop_engine.evaluation import (
    BasisEvaluator,
    RateDensityEvaluator,
)
from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.kernels import log_pairing_kernel
from hierarchical_ad_pop_engine.response.basis import (
    BasisSolveResult,
    solve_basis_fixed_point,
)
from hierarchical_ad_pop_engine.response.coagulation import (
    build_gaussian_coagulation,
)


# ---------------------------------------------------------------------------
# Fixtures — small basis solve.  Same recipe as test_evaluation_implicit.py
# but solving via the v3 basis Picard instead of the v2 grid Picard.
# ---------------------------------------------------------------------------
_M_MIN = 4.0
_M_MAX = 80.0
_CHI_MIN = 1e-3
_CHI_MAX = 1.0 - 1e-3


@pytest.fixture(scope="module")
def small_response():
    r"""Small Gaussian-natal response on a coarse engine grid."""
    return build_gaussian_coagulation(
        mu=8.0,
        sigma=1.0,
        m_min=_M_MIN,
        m_max=_M_MAX,
        natal_spin_loc=0.0,
        natal_spin_scale=0.10,
        kernel_a=0.0,
        kernel_b=1.0,
        log_R_env=0.5,
        k_max=4,
        solver="implicit",  # ignored by solve_basis_fixed_point
        epsilon=1.0,
        grid=BinaryGrid.default(m_min=_M_MIN, m_max=_M_MAX, n_m=20, n_chi=8),
    )


@pytest.fixture(scope="module")
def basis_solver() -> TensorProductBasis:
    r"""Coarse tensor-product basis matching the engine grid's physical range.

    8 modes × 6 modes is small enough to converge fast and dense enough
    to support a non-trivial Gaussian.  The basis lives on
    ``(log m, χ)`` — :py:meth:`for_binary_grid` handles the
    log-mass mapping.
    """
    return TensorProductBasis.for_binary_grid(
        m_min=_M_MIN,
        m_max=_M_MAX,
        n_basis_m=8,
        chi_min=_CHI_MIN,
        chi_max=_CHI_MAX,
        n_basis_chi=6,
    )


@pytest.fixture(scope="module")
def basis_result(small_response, basis_solver) -> BasisSolveResult:
    # ε=1e-2 chosen for Picard *stability* — not for physics realism.
    # At this fixture's natal scale (μ=8, σ=1) with kernel_a=0 +
    # kernel_b=1 + v_esc=1000 + w_over_v_esc=1.0, the soft sigmoid
    # retention is ~σ(1) ≈ 0.73 even on the heaviest LZ kicks, so loss
    # dominates gain and damped Picard at ε≥0.1 diverges (coefficients
    # blow up ~30×/iter, NaN at ~iter 8).  ε=1e-2 leaves a non-trivial
    # cascade (1G/2G both visible) while staying well inside the
    # stability basin.  The basis IFT VJP is pinned for AD correctness
    # by ``test_basis_solver.py`` at ε=1e-4; this file pins the
    # ``BasisEvaluator`` *API* (grid-point equivalence, out-of-domain
    # zero, derived-axis raise, protocol conformance) which is
    # independent of ε's absolute value.
    return solve_basis_fixed_point(
        small_response,
        basis_solver,
        v_esc=jnp.asarray(1000.0),
        epsilon=1e-2,
        w_over_v_esc=1.0,
        max_iter=500,
        tol=1e-8,
        damping=0.2,
    )


@pytest.fixture(scope="module")
def evaluator(
    small_response, basis_solver, basis_result
) -> BasisEvaluator:
    return basis_result.evaluator(small_response, basis_solver)


@pytest.fixture(scope="module")
def N_grid(small_response, basis_solver, basis_result):
    r"""``N(m, χ)`` evaluated on the engine grid via the basis adapter.

    This is the reference the new evaluator must reproduce at every
    grid point — both directly (via :meth:`BasisEvaluator.density`)
    and indirectly (via the marginal / total-rate paths, which call
    it internally).
    """
    return obs.density_from_basis(basis_result, small_response, basis_solver)


# ---------------------------------------------------------------------------
# 1. Grid-point equivalence — density via density_from_basis.
# ---------------------------------------------------------------------------
def test_density_at_grid_points_matches_density_from_basis(
    small_response, evaluator, N_grid
):
    r"""``ev.density(m_i, χ_j)`` == ``density_from_basis(...)[i, j]`` at every
    engine-grid point.  This is the headline Step-2 regression invariant."""
    grid = small_response.grid
    m_mesh, chi_mesh = jnp.meshgrid(grid.m, grid.chi, indexing="ij")
    N_eval = evaluator.density(m_mesh, chi_mesh)
    np.testing.assert_allclose(
        np.asarray(N_eval),
        np.asarray(N_grid),
        rtol=1e-12,
        atol=1e-300,
    )


# ---------------------------------------------------------------------------
# 1. Grid-point equivalence — 1-D marginals.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("axis", list(obs.VALID_AXES))
def test_marginal_1d_at_grid_points_matches_observables(
    small_response, evaluator, N_grid, axis
):
    r"""``ev.marginal_1d(axis, x_grid)`` reproduces ``observables.marginal_1d``."""
    x_grid, dC_dx_ref = obs.marginal_1d(
        N_grid,
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
    small_response, evaluator, N_grid, axes
):
    r"""``ev.marginal_2d(axes, x_grid, y_grid)`` reproduces
    ``observables.marginal_2d``."""
    x_grid, y_grid, dC_ref = obs.marginal_2d(
        N_grid,
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


def test_total_rate_matches_observables(small_response, evaluator, N_grid):
    ref = obs.total_rate(
        N_grid,
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
    small_response, evaluator, N_grid
):
    r"""``rate_density_4d`` == ½ R_env K(m1,m2) N(m1,χ1) N(m2,χ2).

    Built from the lowest-level primitives (point
    :func:`log_pairing_kernel` and the basis-evaluated grid density
    directly) so this is an independent re-implementation, not a
    self-consistency check.  Pins the "K analytic, not Galerkin"
    decision from the PLAN.
    """
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env
    R_env = float(jnp.exp(jnp.asarray(log_R)))

    # Pick 5 random index quadruples from the grid corners
    rng = np.random.default_rng(20260525)
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
    N1 = N_grid[i1, j1]
    N2 = N_grid[i2, j2]
    ref = 0.5 * R_env * jnp.exp(log_K) * N1 * N2

    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-300
    )


# ---------------------------------------------------------------------------
# 3. Out-of-domain → zero.
# ---------------------------------------------------------------------------
def test_density_outside_domain_returns_zero(
    small_response, basis_solver, evaluator
):
    r"""Queries outside the basis rectangle return 0 — both axes, both ends."""
    # Below mass minimum / above mass maximum.
    a_m, b_m = basis_solver.basis_m.domain  # log-mass bounds
    a_c, b_c = basis_solver.basis_chi.domain
    m_lo = jnp.asarray(jnp.exp(a_m) * 0.5)
    m_hi = jnp.asarray(jnp.exp(b_m) * 2.0)
    chi_mid = jnp.asarray(0.5 * (a_c + b_c))

    assert float(evaluator.density(m_lo, chi_mid)) == 0.0
    assert float(evaluator.density(m_hi, chi_mid)) == 0.0

    m_mid = jnp.asarray(jnp.exp(0.5 * (a_m + b_m)))
    chi_lo = jnp.asarray(a_c - 1.0)
    chi_hi = jnp.asarray(b_c + 1.0)

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


def test_rate_density_4d_outside_domain_returns_zero(
    small_response, basis_solver, evaluator
):
    grid = small_response.grid
    a_m, _ = basis_solver.basis_m.domain
    # One progenitor outside the basis → ½ R_env K · 0 · N = 0.
    m1 = jnp.asarray(jnp.exp(a_m) * 0.5)
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
def test_density_grad_runs(small_response, basis_solver):
    r"""``jax.grad`` of a scalar reduction over ``ev.density`` completes.

    Numerical correctness of the basis IFT VJP is pinned by
    ``test_basis_solver.py``; we only want to confirm the evaluator
    layer does not break the AD chain (e.g. by introducing a
    non-traceable branch).
    """

    def loss(log_R_env_val):
        response = build_gaussian_coagulation(
            mu=8.0, sigma=1.0, m_min=_M_MIN, m_max=_M_MAX,
            natal_spin_loc=0.0, natal_spin_scale=0.10,
            kernel_a=0.0, kernel_b=1.0,
            log_R_env=log_R_env_val,
            k_max=4, solver="implicit", epsilon=1.0,
            grid=BinaryGrid.default(m_min=_M_MIN, m_max=_M_MAX, n_m=20, n_chi=8),
        )
        # ε=1e-2 for Picard stability — see basis_result fixture docstring.
        result = solve_basis_fixed_point(
            response, basis_solver,
            v_esc=jnp.asarray(1000.0),
            epsilon=1e-2, w_over_v_esc=1.0,
            max_iter=500, tol=1e-8, damping=0.2,
        )
        ev = result.evaluator(response, basis_solver)
        m_q, chi_q = jnp.meshgrid(
            response.grid.m[::4], response.grid.chi[::2], indexing="ij"
        )
        return jnp.sum(ev.density(m_q, chi_q))

    g = jax.grad(loss)(jnp.asarray(0.5))
    assert jnp.isfinite(g)
