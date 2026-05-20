"""Legendre-Gauss-Lobatto (LGL) polynomial basis on a 1-D interval.

Implements :class:`LegendreLobatto1D` — an ``eqx.Module`` carrying
precomputed LGL nodes, LGL weights, and the eval/project/mass matrices
for a modal Legendre basis ``P_α(ξ(x))`` for ``α = 0, …, n_basis − 1``
on a physical interval ``[a, b]``.

Inner product convention
------------------------
Lebesgue: ``⟨f, g⟩ = ∫_a^b f(x) g(x) dx``.  Under Lebesgue the Legendre
polynomials are orthogonal::

    ∫_{-1}^{1} P_α(ξ) P_β(ξ) dξ = (2 / (2α+1)) · δ_{αβ}

so the mass matrix is diagonal in physical coordinates::

    M_{αα} = ((b − a) / 2) · (2 / (2α + 1)) = (b − a) / (2α + 1).

The Lebesgue choice (not the Chebyshev weight ``1/√(1−ξ²)``) is what
makes the Legendre coefficients ``n_α`` dimensionless density amplitudes
on the physical domain; this is the property ``ε`` needs to retain
absolute meaning (``feedback_no_fudge_factors``).

Quadrature accuracy
-------------------
LGL with ``n = n_basis`` nodes is **exact for polynomials of degree
``2n − 3``** on the reference interval.  This is the property the v3
plan's Step 1 acceptance tests rely on.

For nonlinear products (e.g. the gain-density ``K · N · N`` integrand
inside Picard) the discrete inner product can alias on the top mode.
Overintegration at ``≥ 2 n_basis`` nodes is the remedy and is handled
by Step 3 of the v3 plan, not here.

Construction is one-shot at module-build time (Python-level Newton
iteration; not JIT-traced).  The stored arrays are then constant under
``jax.jit``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array


# ---------------------------------------------------------------------------
# Internal helpers — Legendre polynomial evaluation + LGL node/weight solve.
# ---------------------------------------------------------------------------
def _legendre_recurrence(degree: int, x: Array) -> Array:
    """Evaluate Legendre polynomials ``P_0, …, P_degree`` at points ``x``.

    Uses the Bonnet recursion::

        (k + 1) P_{k+1}(x) = (2k + 1) x P_k(x) − k P_{k-1}(x).

    Parameters
    ----------
    degree : int
        Maximum polynomial degree to evaluate (``≥ 0``).
    x : Array
        Points at which to evaluate.

    Returns
    -------
    Array of shape ``(degree + 1, *x.shape)`` — first axis indexes the
    polynomial degree, trailing axes broadcast over ``x``.
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    x = jnp.asarray(x)
    P0 = jnp.ones_like(x)
    if degree == 0:
        return P0[None, ...]
    P1 = x.astype(P0.dtype)
    rows = [P0, P1]
    for k in range(1, degree):
        next_P = ((2 * k + 1) * x * rows[-1] - k * rows[-2]) / (k + 1)
        rows.append(next_P)
    return jnp.stack(rows, axis=0)


def _lgl_nodes_weights(n: int) -> tuple[Array, Array]:
    """Legendre-Gauss-Lobatto nodes and weights on the reference interval.

    Endpoints ``x_0 = −1``, ``x_{n−1} = 1``; interior nodes are the roots
    of ``P'_{n−1}(x)``.  Weights are::

        w_i = 2 / (n (n − 1) [P_{n−1}(x_i)]²).

    Computed by Newton iteration on ``P'_{n−1}`` starting from Chebyshev-
    Lobatto (cosine) initial guesses, which lie within the basin of
    quadratic convergence for Legendre.  100 iterations is a generous
    safety bound; in practice ~6 are needed to reach machine precision.
    """
    if n < 2:
        raise ValueError(f"LGL requires n >= 2, got n = {n}")
    if n == 2:
        return jnp.array([-1.0, 1.0]), jnp.array([1.0, 1.0])

    # Initial guess: CGL interior nodes in increasing order on (−1, 1).
    k_idx = jnp.arange(1, n - 1)
    x = -jnp.cos(jnp.pi * k_idx / (n - 1))

    # Newton: x ← x − P'_{n−1}(x) / P''_{n−1}(x).
    # P'_{n−1}(x)  = (n−1) (x P_{n−1}(x) − P_{n−2}(x)) / (x² − 1)
    # P''_{n−1}(x) = (2 x P'_{n−1}(x) − (n−1) n P_{n−1}(x)) / (1 − x²)
    #              (Legendre ODE).
    for _ in range(100):
        P_all = _legendre_recurrence(n - 1, x)
        P_nm1 = P_all[-1]
        P_nm2 = P_all[-2]
        dP = (n - 1) * (x * P_nm1 - P_nm2) / (x * x - 1.0)
        ddP = (2.0 * x * dP - (n - 1) * n * P_nm1) / (1.0 - x * x)
        dx = dP / ddP
        x = x - dx
        if float(jnp.max(jnp.abs(dx))) < 1e-15:
            break

    nodes = jnp.concatenate([jnp.array([-1.0]), x, jnp.array([1.0])])
    P_nm1_at_nodes = _legendre_recurrence(n - 1, nodes)[-1]
    weights = 2.0 / (n * (n - 1) * P_nm1_at_nodes * P_nm1_at_nodes)
    return nodes, weights


# ---------------------------------------------------------------------------
# Public class — Legendre-Lobatto 1-D basis on physical [a, b].
# ---------------------------------------------------------------------------
class LegendreLobatto1D(eqx.Module):
    """Legendre-Gauss-Lobatto basis on a 1-D physical interval ``[a, b]``.

    Attributes
    ----------
    n_basis : int  (static field)
        Number of LGL nodes ≡ number of Legendre modal coefficients.
    domain : tuple[float, float]  (static field)
        Physical domain ``(a, b)`` with ``a < b``.
    nodes : Array, shape (n_basis,)
        LGL nodes in physical coordinates.
    weights : Array, shape (n_basis,)
        LGL quadrature weights in physical coordinates: ``((b − a)/2) · ŵ_i``
        where ``ŵ_i`` are the reference-interval weights.
    eval_matrix : Array, shape (n_basis, n_basis)
        ``E[i, α] = P_α(ξ_i)`` where ``ξ_i`` is the i-th LGL node on
        the reference interval ``[−1, 1]``.  Note row index is the node,
        column index is the mode (so ``E @ coeffs`` evaluates at nodes).
    project_matrix : Array, shape (n_basis, n_basis)
        ``P[α, i] = ((2α + 1)/2) · ŵ_i · P_α(ξ_i)``.  Discrete Galerkin
        projection of nodal values onto modal coefficients.  Independent
        of ``(b − a)`` (cancels in the mass-matrix inverse).
    mass_matrix : Array, shape (n_basis, n_basis)
        Diagonal ``M[α, α] = (b − a) / (2α + 1)`` under Lebesgue inner
        product.  Stored as a dense array (not just the diagonal) so
        downstream code can use a uniform pattern; storage cost is
        ``n_basis²`` floats which is negligible.

    Construction
    ------------
    Use :py:meth:`on`::

        basis = LegendreLobatto1D.on(domain=(0.0, 1.0), n_basis=32)
    """

    n_basis: int = eqx.field(static=True)
    domain: tuple[float, float] = eqx.field(static=True)
    nodes: Array
    weights: Array
    eval_matrix: Array
    project_matrix: Array
    mass_matrix: Array

    @classmethod
    def on(
        cls, domain: tuple[float, float], n_basis: int
    ) -> "LegendreLobatto1D":
        """Build a LGL basis with ``n_basis`` modes on physical ``[a, b]``."""
        a, b = float(domain[0]), float(domain[1])
        if not (a < b):
            raise ValueError(f"domain must satisfy a < b, got ({a}, {b})")
        if n_basis < 2:
            raise ValueError(f"n_basis must be >= 2, got {n_basis}")

        # Reference-interval LGL data on [-1, 1].
        xi_ref, w_ref = _lgl_nodes_weights(n_basis)

        # Affine map ξ ∈ [−1, 1] → x ∈ [a, b]: x = (a+b)/2 + (b−a)/2 · ξ
        half = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        nodes_phys = mid + half * xi_ref
        weights_phys = half * w_ref

        # Eval matrix E[i, α] = P_α(ξ_i).  _legendre_recurrence returns
        # (n_basis, n_basis) with the degree axis first; transpose so
        # row is node and column is mode.
        P_at_nodes = _legendre_recurrence(n_basis - 1, xi_ref)  # (n_basis, n_basis)
        eval_mat = P_at_nodes.T  # (n_basis, n_basis)

        # Projection matrix P[α, i] = ((2α+1)/2) · ŵ_i · P_α(ξ_i)
        alpha = jnp.arange(n_basis)
        project_mat = (
            (2.0 * alpha[:, None] + 1.0) * 0.5 * w_ref[None, :] * P_at_nodes
        )

        # Mass matrix (diagonal under Lebesgue): M[α, α] = (b−a)/(2α+1)
        mass_diag = (b - a) / (2.0 * alpha + 1.0)
        mass_mat = jnp.diag(mass_diag)

        return cls(
            n_basis=int(n_basis),
            domain=(a, b),
            nodes=nodes_phys,
            weights=weights_phys,
            eval_matrix=eval_mat,
            project_matrix=project_mat,
            mass_matrix=mass_mat,
        )

    # -----------------------------------------------------------------
    # Operational helpers — all JIT-friendly (pure array math).
    # -----------------------------------------------------------------
    def project(self, f_at_nodes: Array) -> Array:
        """Project nodal values onto modal coefficients.

        Parameters
        ----------
        f_at_nodes : Array, shape ``(n_basis,)`` or ``(n_basis, ...)``
            Values of ``f`` at :attr:`nodes`.  Leading axis is contracted;
            trailing axes broadcast.

        Returns
        -------
        Array of the same trailing shape carrying Legendre coefficients
        ``n_α`` such that ``f(x) ≈ Σ_α n_α P_α(ξ(x))``.
        """
        return jnp.tensordot(self.project_matrix, f_at_nodes, axes=(1, 0))

    def evaluate_at_nodes(self, coeffs: Array) -> Array:
        """Evaluate the modal expansion at this basis's own LGL nodes.

        Faster than :py:meth:`evaluate` when the query points are the
        nodes — uses the precomputed :attr:`eval_matrix`.
        """
        return jnp.tensordot(self.eval_matrix, coeffs, axes=(1, 0))

    def evaluate(self, coeffs: Array, x: Array) -> Array:
        """Evaluate the modal expansion at arbitrary physical points ``x``.

        Stable via the Bonnet recurrence; ``x`` may be any shape and the
        result has the same shape.
        """
        a, b = self.domain
        xi = (2.0 * x - (a + b)) / (b - a)
        P_all = _legendre_recurrence(self.n_basis - 1, xi)
        return jnp.einsum("a...,a->...", P_all, coeffs)
