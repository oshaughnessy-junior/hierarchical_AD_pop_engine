"""Tensor-product basis on two 1-D component bases.

For the v3 engine the binary parameter space is ``(log m, χ)`` and the
basis is the tensor product of two :class:`LegendreLobatto1D` bases on
those axes.  The class is also useful as a building block for higher-D
extensions (eventual tilted spin, redshift) — wrap two tensor products
to get four axes, etc.

Storage
-------
Component bases are stored directly; the full ``(n_m·n_χ) × (n_m·n_χ)``
eval / project / mass matrices are **not** materialised — they are
applied factored, axis-by-axis, with cost ``O(n_m² n_χ + n_m n_χ²)``
instead of ``O((n_m n_χ)²)``.  For ``(32, 32)`` this is 32× faster per
projection and uses 1024× less memory; for larger bases the win grows.

Coefficient layout
------------------
Modal coefficients are 2-D arrays of shape ``(n_basis_m, n_basis_χ)``.
Flat ordering for vector-style APIs (GMRES, custom-VJP matvec) is
row-major::

    flat_idx = α_m · n_basis_χ + α_χ.

Helpers :py:meth:`flatten` and :py:meth:`unflatten` move between layouts.

Inner product
-------------
Lebesgue on ``[a_m, b_m] × [a_χ, b_χ]``::

    ⟨f, g⟩ = ∫∫ f(m, χ) g(m, χ) dm dχ.

The mass matrix is diagonal in flat coords (each component basis is
diagonal under Lebesgue), with diagonal
``M_diag_2d[α_m, α_χ] = (b_m − a_m)/(2α_m + 1) · (b_χ − a_χ)/(2α_χ + 1)``.

Quadrature accuracy
-------------------
Exact for polynomials of total degree ``2 min(n_m, n_χ) − 3``.  Step 3
of the v3 plan handles overintegration for the nonlinear gain tensor.

Cross-basis operations
----------------------
:py:meth:`restrict_to` and :py:meth:`prolong_to` move modal coefficients
between two ``TensorProductBasis`` instances that share the same
physical interval on each axis but differ in modal resolution.
Restriction is exact Legendre modal truncation (drop high modes);
prolongation is exact zero-padding.  Both are O(1) memory operations
on the source coefficient array — no quadrature error introduced.  They
are class methods (NOT a separate ``basis/restriction.py`` module) so
the compatibility checks live next to the basis representation they
operate on.  See ``PLAN_engine_v3_basis.md`` §Step 3.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .legendre import LegendreLobatto1D, _legendre_recurrence


class TensorProductBasis(eqx.Module):
    """Tensor product of two 1-D Legendre-Lobatto bases.

    Attributes
    ----------
    basis_m : LegendreLobatto1D
        Basis on the "mass-like" axis (typically log m).
    basis_chi : LegendreLobatto1D
        Basis on the "spin-like" axis (typically χ).
    """

    basis_m: LegendreLobatto1D
    basis_chi: LegendreLobatto1D

    @classmethod
    def from_1d(
        cls,
        basis_m: LegendreLobatto1D,
        basis_chi: LegendreLobatto1D,
    ) -> "TensorProductBasis":
        return cls(basis_m=basis_m, basis_chi=basis_chi)

    @classmethod
    def for_binary_grid(
        cls,
        m_min: float = 2.0,
        m_max: float = 200.0,
        n_basis_m: int = 32,
        chi_min: float = 1e-3,
        chi_max: float = 1.0 - 1e-3,
        n_basis_chi: int = 32,
    ) -> "TensorProductBasis":
        """Convenience constructor matching the v2 ``BinaryGrid.default()``
        ranges.

        Builds the m-axis basis on ``log m ∈ [log m_min, log m_max]`` (so
        modal coefficients live in log-mass space, consistent with the
        v2 ``log_m`` grid axis used by :mod:`hierarchical_ad_pop_engine.grids`),
        and the χ-axis basis on ``[chi_min, chi_max]`` (the shifted unit
        interval used by the v2 grid for the same beta-distribution
        regularity reason).
        """
        import math

        log_m_min, log_m_max = math.log(m_min), math.log(m_max)
        basis_m = LegendreLobatto1D.on(
            domain=(log_m_min, log_m_max), n_basis=n_basis_m
        )
        basis_chi = LegendreLobatto1D.on(
            domain=(chi_min, chi_max), n_basis=n_basis_chi
        )
        return cls.from_1d(basis_m, basis_chi)

    # -----------------------------------------------------------------
    # Shape / layout properties.
    # -----------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        """``(n_basis_m, n_basis_χ)``."""
        return (self.basis_m.n_basis, self.basis_chi.n_basis)

    @property
    def n_basis(self) -> int:
        """Total flat dimension ``n_basis_m · n_basis_χ``."""
        return self.basis_m.n_basis * self.basis_chi.n_basis

    # -----------------------------------------------------------------
    # Nodes / weights / mass matrix (all factored, never materialised).
    # -----------------------------------------------------------------
    def meshgrid_nodes(self) -> tuple[Array, Array]:
        """Cartesian product of 1-D node sets.

        Returns ``(m_nodes_2d, chi_nodes_2d)``, each of shape
        ``(n_basis_m, n_basis_χ)``.  Note ``m_nodes_2d`` carries
        log-mass values if :py:meth:`for_binary_grid` was used to
        construct this basis — exponentiate if you need physical mass.
        """
        m_2d, chi_2d = jnp.meshgrid(
            self.basis_m.nodes, self.basis_chi.nodes, indexing="ij"
        )
        return m_2d, chi_2d

    def weights_2d(self) -> Array:
        """Outer-product LGL quadrature weights, ``(n_basis_m, n_basis_χ)``."""
        return (
            self.basis_m.weights[:, None] * self.basis_chi.weights[None, :]
        )

    def mass_diag_2d(self) -> Array:
        """Diagonal of the 2-D mass matrix, ``(n_basis_m, n_basis_χ)``.

        Diagonal under Lebesgue because each component is diagonal::

            M[α_m, α_χ] = (b_m − a_m)/(2α_m + 1) · (b_χ − a_χ)/(2α_χ + 1).
        """
        diag_m = jnp.diag(self.basis_m.mass_matrix)
        diag_chi = jnp.diag(self.basis_chi.mass_matrix)
        return diag_m[:, None] * diag_chi[None, :]

    # -----------------------------------------------------------------
    # Project / evaluate.
    # -----------------------------------------------------------------
    def project(self, f_at_nodes: Array) -> Array:
        """Project nodal values onto 2-D modal coefficients.

        Parameters
        ----------
        f_at_nodes : Array, shape ``(n_basis_m, n_basis_χ)``
            Values of ``f`` at the tensor-product LGL nodes.

        Returns
        -------
        Array of shape ``(n_basis_m, n_basis_χ)`` carrying the Legendre
        coefficients ``n_{α_m, α_χ}``.

        Factored application: ``P_m @ f @ P_χ^T``.  Cost
        ``O(n_m² n_χ + n_m n_χ²)``; the materialised Kronecker form
        would cost ``O((n_m n_χ)²)``.
        """
        tmp = self.basis_m.project_matrix @ f_at_nodes
        return tmp @ self.basis_chi.project_matrix.T

    def evaluate_at_nodes(self, coeffs: Array) -> Array:
        """Evaluate the 2-D expansion at the LGL nodes.

        Factored: ``E_m @ coeffs @ E_χ^T``.
        """
        tmp = self.basis_m.eval_matrix @ coeffs
        return tmp @ self.basis_chi.eval_matrix.T

    def evaluate(self, coeffs: Array, x_m: Array, x_chi: Array) -> Array:
        """Evaluate the expansion at arbitrary ``(x_m, x_chi)`` points.

        ``x_m`` and ``x_chi`` are broadcast against each other; the
        result has the broadcast shape.  Uses the Bonnet recurrence on
        each axis directly — numerically stable.
        """
        x_m_b, x_chi_b = jnp.broadcast_arrays(
            jnp.asarray(x_m), jnp.asarray(x_chi)
        )
        a_m, b_m = self.basis_m.domain
        a_c, b_c = self.basis_chi.domain
        xi_m = (2.0 * x_m_b - (a_m + b_m)) / (b_m - a_m)
        xi_c = (2.0 * x_chi_b - (a_c + b_c)) / (b_c - a_c)
        P_m = _legendre_recurrence(self.basis_m.n_basis - 1, xi_m)
        P_c = _legendre_recurrence(self.basis_chi.n_basis - 1, xi_c)
        return jnp.einsum("ab,a...,b...->...", coeffs, P_m, P_c)

    # -----------------------------------------------------------------
    # Flat ↔ 2-D layout helpers (for GMRES / custom-VJP matvec).
    # -----------------------------------------------------------------
    def flatten(self, coeffs_2d: Array) -> Array:
        """``(n_basis_m, n_basis_χ)`` → ``(n_basis_m · n_basis_χ,)`` row-major."""
        return coeffs_2d.reshape(-1)

    def unflatten(self, coeffs_flat: Array) -> Array:
        """Inverse of :py:meth:`flatten`."""
        return coeffs_flat.reshape(self.shape)

    # -----------------------------------------------------------------
    # Cross-basis operations (restrict / prolong).
    # -----------------------------------------------------------------
    def _check_same_domain(self, target: "TensorProductBasis") -> None:
        """Raise if ``target`` lives on a different physical interval.

        Restrict and prolong are exact only when the modal coefficient
        arrays of ``self`` and ``target`` refer to the *same* Legendre
        polynomials on the *same* physical domain.  Mismatched domains
        would silently mean ``coeffs[i]`` represents a different
        function on each side, and the truncation/zero-pad would
        introduce arbitrary error.  Compared with a relative tolerance
        so that float round-off across construction sites doesn't
        trigger spurious rejections.
        """
        rtol = 1e-12
        atol = 1e-15
        for name, dom_src, dom_tgt in (
            ("m", self.basis_m.domain, target.basis_m.domain),
            ("χ", self.basis_chi.domain, target.basis_chi.domain),
        ):
            a_s, b_s = dom_src
            a_t, b_t = dom_tgt
            scale = max(abs(a_s), abs(b_s), abs(a_t), abs(b_t), 1.0)
            ok_lo = abs(a_s - a_t) <= rtol * scale + atol
            ok_hi = abs(b_s - b_t) <= rtol * scale + atol
            if not (ok_lo and ok_hi):
                raise ValueError(
                    f"axis {name} domain mismatch: source {dom_src} vs "
                    f"target {dom_tgt}; restrict/prolong require "
                    "identical physical intervals."
                )

    def restrict_to(
        self, coeffs: Array, target: "TensorProductBasis"
    ) -> Array:
        """Restrict modal coefficients onto a coarser tensor-product basis.

        Restriction is exact Legendre modal truncation: modes
        ``α_m ∈ [n_target_m, n_self_m)`` and similarly for χ are dropped.
        No quadrature error.  ``self.shape`` must componentwise ≥
        ``target.shape``; use :py:meth:`prolong_to` for the opposite
        direction.  ``self`` and ``target`` must share the same physical
        domain on each axis.

        Parameters
        ----------
        coeffs : Array, shape ``self.shape``
            Modal coefficients in the source (``self``) basis.
        target : TensorProductBasis
            The target basis.  Must be coarser or equal on every axis.

        Returns
        -------
        Array of shape ``target.shape`` carrying the truncated
        coefficients.

        Raises
        ------
        ValueError
            If ``target.shape`` is not componentwise ≤ ``self.shape``,
            or if the physical domains disagree.
        """
        self._check_same_domain(target)
        n_m_s, n_c_s = self.shape
        n_m_t, n_c_t = target.shape
        if n_m_t > n_m_s or n_c_t > n_c_s:
            raise ValueError(
                f"restrict_to: target shape {target.shape} must be ≤ "
                f"source shape {self.shape} on every axis "
                "(use prolong_to for the opposite direction)."
            )
        return coeffs[:n_m_t, :n_c_t]

    def prolong_to(
        self, coeffs: Array, target: "TensorProductBasis"
    ) -> Array:
        """Prolong modal coefficients onto a finer tensor-product basis.

        Prolongation is exact Legendre modal zero-padding.  No
        interpolation error.  ``target.shape`` must componentwise ≥
        ``self.shape``; use :py:meth:`restrict_to` for the opposite
        direction.  ``self`` and ``target`` must share the same physical
        domain on each axis.

        Parameters
        ----------
        coeffs : Array, shape ``self.shape``
            Modal coefficients in the source (``self``) basis.
        target : TensorProductBasis
            The target basis.  Must be finer or equal on every axis.

        Returns
        -------
        Array of shape ``target.shape`` carrying the zero-padded
        coefficients.

        Raises
        ------
        ValueError
            If ``target.shape`` is not componentwise ≥ ``self.shape``,
            or if the physical domains disagree.

        Notes
        -----
        ``restrict_to(prolong_to(c, target), self) == c`` exactly,
        because zero-padding followed by truncating the padding back
        off is the identity at the bit level.  This identity is a
        Step 3 acceptance test.
        """
        self._check_same_domain(target)
        n_m_s, n_c_s = self.shape
        n_m_t, n_c_t = target.shape
        if n_m_t < n_m_s or n_c_t < n_c_s:
            raise ValueError(
                f"prolong_to: target shape {target.shape} must be ≥ "
                f"source shape {self.shape} on every axis "
                "(use restrict_to for the opposite direction)."
            )
        return jnp.pad(
            coeffs,
            ((0, n_m_t - n_m_s), (0, n_c_t - n_c_s)),
            mode="constant",
        )
