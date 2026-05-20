"""Acceptance tests for the v3 basis infrastructure (Step 1).

Mirrors PLAN_engine_v3_basis.md §Step 1 acceptance criteria:

a. ``project → evaluate`` recovers smooth functions to relative ``1e-8``
   at ``n_basis ≥ 24``.
b. ``mass_matrix`` is symmetric positive-definite with condition number
   ``< 1e6`` at ``n_basis = 32``.
c. Gauss-Lobatto quadrature is exact (to ``< 1e-8``) for polynomials of
   degree ``2n − 3``.

Plus tensor-product sanity checks (the basis used by the v3 Picard).

Tolerances
----------
All "exact" / "machine-precision" checks use ``1e-8`` as the threshold
and all convergence checks use ``1e-8`` as the absolute relative-error
bar — float64 algorithms in this file empirically hit ~1e-12 so this
leaves ~4 orders of margin against accumulated rounding, future
refactors, or jax-version drift, while still catching real algorithm
bugs (a 1e-7 failure mode like float32 ulp leakage would still trip).

Float-64 mode is enabled explicitly below — same pattern as
``test_implicit_parity.py`` and ``test_gradients.py``.  Under the
default float32 these tolerances would not hold (float32 ulp ~ 1e-7).

The math behind every test has been independently validated in a pure-
numpy parallel script (``outputs/validate_legendre_basis.py`` in the
development session); this file exercises the actual jax/equinox code
shipped under ``src/hierarchical_ad_pop_engine/basis/``.

Naming note: the plan refers to "Chebyshev-Lobatto"; the implementation
is Legendre-Gauss-Lobatto (see ``basis/__init__.py`` for the math
justification of the refinement).  All acceptance criteria hold for the
implemented basis.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.basis import (  # noqa: E402
    LegendreLobatto1D,
    TensorProductBasis,
)

# Common tolerances — see module docstring.
TOL_EXACT = 1e-8       # checks that "should" be exact (polynomials, factored vs Kron, modal recovery)
TOL_SYM = 1e-10        # symmetry of stored matrices (purely algebraic)
TOL_CONVERGE = 1e-8    # spectral convergence on smooth functions at n_basis ≥ 24


# ---------------------------------------------------------------------------
# 1-D basis tests.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_quadrature_exact_for_polynomials_up_to_degree_2n_minus_3(n):
    """Reference-interval LGL with n nodes integrates polynomials of
    degree ≤ 2n-3 exactly.  Uses the basis's reference-interval nodes
    and weights (recovered by setting domain = (-1, 1))."""
    basis = LegendreLobatto1D.on(domain=(-1.0, 1.0), n_basis=n)
    x = basis.nodes
    w = basis.weights
    for d in range(2 * n - 2):  # degrees 0, 1, ..., 2n-3
        f = x ** d
        approx = float(jnp.sum(w * f))
        # Exact integral of x^d on [-1, 1]
        exact = 0.0 if (d % 2 == 1) else 2.0 / (d + 1)
        assert abs(approx - exact) < TOL_EXACT, (
            f"n={n}, d={d}: LGL gave {approx}, exact {exact}"
        )


@pytest.mark.parametrize("n", [8, 16, 32])
def test_mass_matrix_symmetric_positive_definite(n):
    """Mass matrix is symmetric PD with cond < 1e6 at n=32."""
    basis = LegendreLobatto1D.on(domain=(0.0, 1.0), n_basis=n)
    M = np.asarray(basis.mass_matrix)
    assert np.allclose(M, M.T, atol=TOL_SYM), "mass matrix not symmetric"
    eigs = np.linalg.eigvalsh(M)
    assert np.all(eigs > 0), f"mass matrix not PD; min eig = {eigs.min()}"
    cond = float(eigs.max() / eigs.min())
    assert cond < 1e6, f"n={n}: cond(M) = {cond:.2e}"


@pytest.mark.parametrize("n", [24, 32, 48])
def test_project_evaluate_inverse_smooth_function(n):
    """Project a smooth function then evaluate; recover to relative TOL_CONVERGE.

    The test function (σ=0.25 Gaussian on [0,1]) has Legendre
    coefficients that decay below machine ε by n≈24; narrower features
    (e.g. the engine's natal-density peak at ξ-scale ≈ 0.09) need
    proportionally higher n_basis but follow the same exponential
    convergence pattern.
    """
    basis = LegendreLobatto1D.on(domain=(0.0, 1.0), n_basis=n)

    def f(x):
        return jnp.exp(-0.5 * ((x - 0.3) / 0.25) ** 2) / (
            0.25 * jnp.sqrt(2 * jnp.pi)
        )

    f_at_nodes = f(basis.nodes)
    coeffs = basis.project(f_at_nodes)
    x_eval = jnp.linspace(0.01, 0.99, 200)
    f_recon = basis.evaluate(coeffs, x_eval)
    f_true = f(x_eval)
    rel_err = float(jnp.max(jnp.abs(f_recon - f_true)) / jnp.max(jnp.abs(f_true)))
    assert rel_err < TOL_CONVERGE, f"n={n}: rel_err = {rel_err:.2e}"


def test_nodes_endpoint_inclusion_and_monotone():
    """LGL nodes include both endpoints and are strictly increasing."""
    for n in (2, 3, 4, 8, 32):
        basis = LegendreLobatto1D.on(domain=(-1.0, 1.0), n_basis=n)
        x = np.asarray(basis.nodes)
        assert x.shape == (n,)
        assert np.all(np.diff(x) > 0), f"n={n}: nodes not monotone"
        assert x[0] == pytest.approx(-1.0)
        assert x[-1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "domain",
    [
        (-1.0, 1.0),
        (0.0, 1.0),
        (-5.0, 7.0),
        (math.log(4.0), math.log(80.0)),
    ],
)
def test_weights_sum_to_interval_length(domain):
    """Physical-domain LGL weights sum to (b-a)."""
    for n in (4, 8, 16, 32):
        basis = LegendreLobatto1D.on(domain=domain, n_basis=n)
        total = float(jnp.sum(basis.weights))
        assert total == pytest.approx(domain[1] - domain[0], abs=TOL_EXACT)


def test_evaluate_matches_evaluate_at_nodes():
    """``evaluate(coeffs, nodes)`` should match the cached
    ``eval_matrix @ coeffs``."""
    basis = LegendreLobatto1D.on(domain=(0.0, 1.0), n_basis=16)
    rng = np.random.default_rng(0)
    coeffs = jnp.asarray(rng.standard_normal(16))
    via_eval = basis.evaluate(coeffs, basis.nodes)
    via_mat = basis.evaluate_at_nodes(coeffs)
    assert jnp.allclose(via_eval, via_mat, atol=TOL_EXACT)


def test_project_then_evaluate_recovers_basis_function():
    """Projecting P_α evaluated at the nodes recovers δ_{αβ} coefficients.

    A unit test for the basis's own modal recovery: f(x) = P_2(ξ(x))
    should project to coeffs = [0, 0, 1, 0, ...].
    """
    n = 16
    basis = LegendreLobatto1D.on(domain=(-1.0, 1.0), n_basis=n)
    # Build P_α(ξ_nodes) for α=2.  The eval matrix has E[i, α] = P_α(ξ_i)
    # so the desired nodal values are E[:, 2].
    target_alpha = 2
    nodal = basis.eval_matrix[:, target_alpha]
    coeffs = basis.project(nodal)
    expected = jnp.zeros(n).at[target_alpha].set(1.0)
    # Top-mode aliasing only affects α = n-1; for α=2 with n=16 we have
    # 2 + 2 = 4 < 2n - 3 = 29, so projection is exact.
    assert jnp.allclose(coeffs, expected, atol=TOL_EXACT)


# ---------------------------------------------------------------------------
# Tensor-product basis tests.
# ---------------------------------------------------------------------------
def test_tensor_product_shape_and_n_basis():
    tp = TensorProductBasis.for_binary_grid(
        n_basis_m=24, n_basis_chi=16
    )
    assert tp.shape == (24, 16)
    assert tp.n_basis == 24 * 16


def test_tensor_product_meshgrid_consistency():
    tp = TensorProductBasis.for_binary_grid(n_basis_m=8, n_basis_chi=6)
    m_2d, c_2d = tp.meshgrid_nodes()
    assert m_2d.shape == (8, 6)
    assert c_2d.shape == (8, 6)
    # All rows of m_2d are identical (m varies along axis 0).
    assert jnp.allclose(m_2d[:, 0], m_2d[:, -1])
    # Same for c_2d along axis 1.
    assert jnp.allclose(c_2d[0, :], c_2d[-1, :])


def test_tensor_product_weights_outer_product():
    tp = TensorProductBasis.for_binary_grid(n_basis_m=8, n_basis_chi=6)
    w2d = tp.weights_2d()
    expected = (
        tp.basis_m.weights[:, None] * tp.basis_chi.weights[None, :]
    )
    assert jnp.allclose(w2d, expected, atol=TOL_EXACT)


def test_tensor_product_project_evaluate_inverse_2d():
    """2-D project → evaluate inverse on a smooth product of 1-D Gaussians."""
    tp = TensorProductBasis.from_1d(
        LegendreLobatto1D.on(domain=(0.0, 1.0), n_basis=28),
        LegendreLobatto1D.on(domain=(0.0, 1.0), n_basis=28),
    )
    m_2d, c_2d = tp.meshgrid_nodes()

    def f(m, chi):
        return (
            jnp.exp(-0.5 * ((m - 0.3) / 0.25) ** 2)
            * jnp.exp(-0.5 * ((chi - 0.5) / 0.3) ** 2)
        )

    coeffs = tp.project(f(m_2d, c_2d))

    # Evaluate on a dense (50, 50) grid.
    x_m = jnp.linspace(0.01, 0.99, 50)
    x_c = jnp.linspace(0.01, 0.99, 50)
    X_m, X_c = jnp.meshgrid(x_m, x_c, indexing="ij")
    f_recon = tp.evaluate(coeffs, X_m, X_c)
    f_true = f(X_m, X_c)
    rel_err = float(jnp.max(jnp.abs(f_recon - f_true)) / jnp.max(jnp.abs(f_true)))
    assert rel_err < TOL_CONVERGE, f"2-D rel err = {rel_err:.2e}"


def test_tensor_product_factored_matches_kronecker():
    """The factored ``P_m @ f @ P_χ^T`` projection equals the Kronecker form."""
    tp = TensorProductBasis.from_1d(
        LegendreLobatto1D.on(domain=(0.0, 1.0), n_basis=6),
        LegendreLobatto1D.on(domain=(-1.0, 1.0), n_basis=5),
    )
    rng = np.random.default_rng(7)
    f_at_nodes = jnp.asarray(rng.standard_normal((6, 5)))

    coeffs_factored = tp.project(f_at_nodes)

    P_kron = jnp.kron(
        tp.basis_m.project_matrix, tp.basis_chi.project_matrix
    )
    coeffs_flat = P_kron @ f_at_nodes.reshape(-1)
    coeffs_kron = coeffs_flat.reshape(6, 5)

    assert jnp.allclose(coeffs_factored, coeffs_kron, atol=TOL_EXACT)


def test_tensor_product_flatten_unflatten_roundtrip():
    tp = TensorProductBasis.for_binary_grid(n_basis_m=8, n_basis_chi=5)
    rng = np.random.default_rng(11)
    c2d = jnp.asarray(rng.standard_normal(tp.shape))
    flat = tp.flatten(c2d)
    assert flat.shape == (tp.n_basis,)
    # reshape roundtrip is bit-exact (no arithmetic).
    assert jnp.allclose(tp.unflatten(flat), c2d, atol=TOL_EXACT)


def test_tensor_product_mass_diag_factorisation():
    """``mass_diag_2d[α_m, α_χ] = M^m_{αα} · M^χ_{αα}``."""
    tp = TensorProductBasis.from_1d(
        LegendreLobatto1D.on(domain=(math.log(2.0), math.log(200.0)), n_basis=16),
        LegendreLobatto1D.on(domain=(0.0, 1.0), n_basis=8),
    )
    diag = tp.mass_diag_2d()
    assert diag.shape == (16, 8)
    # Explicit reconstruction.
    diag_m = jnp.diag(tp.basis_m.mass_matrix)
    diag_chi = jnp.diag(tp.basis_chi.mass_matrix)
    expected = diag_m[:, None] * diag_chi[None, :]
    assert jnp.allclose(diag, expected, atol=TOL_EXACT)


def test_tensor_product_jit_traceable():
    """The basis should compose under ``jax.jit`` (it's a static pytree)."""
    import jax

    tp = TensorProductBasis.for_binary_grid(n_basis_m=8, n_basis_chi=6)

    @jax.jit
    def project_eval_roundtrip(f_at_nodes):
        coeffs = tp.project(f_at_nodes)
        return tp.evaluate_at_nodes(coeffs)

    m_2d, c_2d = tp.meshgrid_nodes()
    # Use a function that lies EXACTLY in the basis span — a low-degree
    # bivariate polynomial.  For such f the project→evaluate round-trip
    # is mathematically the identity, so any deviation > TOL_EXACT
    # signals a jit-tracing bug (e.g. accidental dtype demotion).
    # Constants/linears are well within span{P_α(m) ⊗ P_β(χ) : α<8, β<6}.
    f_vals = 1.0 + 0.3 * m_2d - 0.7 * c_2d + 0.2 * m_2d * c_2d
    roundtripped = project_eval_roundtrip(f_vals)
    assert jnp.max(jnp.abs(roundtripped - f_vals)) < TOL_EXACT
