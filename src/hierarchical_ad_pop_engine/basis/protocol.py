"""``Basis`` protocol — the shape every 1-D and tensor-product basis must
satisfy so the v3 Picard solver can be written once against the abstract
interface.

A basis stores:

* ``n_basis`` modes ``φ_α`` and ``n_basis`` collocation/quadrature nodes
  ``x_i``.  In a Lobatto polynomial basis these are the same count
  (square eval matrix); other bases may decouple the two.
* The quadrature weights ``w_i`` corresponding to the nodes — Galerkin
  inner products of the form ``⟨f, g⟩ = Σ_i w_i f(x_i) g(x_i)`` are
  exact for polynomials up to a basis-specific degree.
* The eval matrix ``E[i, α] = φ_α(x_i)`` — node-space ← coefficient-
  space evaluation.
* The project matrix ``P[α, i]`` such that ``n_α = Σ_i P[α, i] f(x_i)``
  realises the discrete Galerkin projection of ``f`` onto the basis.
* The mass matrix ``M[α, β] = ⟨φ_α, φ_β⟩`` — used wherever an integral
  of ``f`` against a basis function is needed (e.g. inner products with
  natal densities or test functions).

All five quantities are precomputed once at construction.  Inside JIT
boundaries they appear as constant arrays.

The class is an ``equinox.Module`` so it composes as a JAX pytree and
participates in ``jax.jit`` / ``jax.grad`` without surprise.  Differentiable
parameters of the model (kernel coefficients, natal-density θ, ε) live
*outside* the basis; the basis itself is fully static.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Array


@runtime_checkable
class Basis(Protocol):
    """Abstract polynomial basis protocol.

    Attributes
    ----------
    n_basis : int
        Number of modes (= number of nodes for square Lobatto-style bases).
    nodes : Array, shape ``(n_basis, ...)``
        Collocation points in *physical* coordinates.  For 1-D bases the
        shape is ``(n_basis,)``; for tensor-product bases the leading
        dimensions enumerate basis modes in row-major order and the
        trailing axis indexes the physical dimension (e.g. ``(n_basis, 2)``
        for ``(m, χ)``).
    weights : Array, shape ``(n_basis,)``
        Quadrature weights for the discrete Galerkin inner product.
    eval_matrix : Array, shape ``(n_basis, n_basis)``
        ``E[i, α] = φ_α(x_i)``.
    project_matrix : Array, shape ``(n_basis, n_basis)``
        ``P[α, i]`` such that ``n_α = Σ_i P[α, i] f(x_i)``.
    mass_matrix : Array, shape ``(n_basis, n_basis)``
        ``M[α, β] = ⟨φ_α, φ_β⟩`` under the Lebesgue inner product on
        the basis domain.

    The protocol does not prescribe methods — direct array contractions
    against the precomputed matrices are the production calling pattern
    (so the Picard inner loops are pure ``einsum`` over JAX arrays with
    no Python dispatch).
    """

    n_basis: int
    nodes: Array
    weights: Array
    eval_matrix: Array
    project_matrix: Array
    mass_matrix: Array
