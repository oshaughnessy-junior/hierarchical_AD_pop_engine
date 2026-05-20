"""Spectral / pseudospectral basis infrastructure for the v3 engine.

Implements polynomial bases on bounded 1-D intervals and tensor-product
extensions to ``(log m, χ)``.  The engine v3 refactor
(``demo/PLAN_engine_v3_basis.md``) replaces the (m, χ) grid Picard with
a Galerkin spectral discretisation; this module owns the basis
infrastructure underneath that.

Naming note
-----------
The v3 plan refers throughout to "Chebyshev-Lobatto" because that name
is colloquial for "polynomial Lobatto spectral basis".  The actual
implementation here is **Legendre-Gauss-Lobatto** (LGL) — nodes are
roots of ``(1-ξ²) P'_{n-1}(ξ)`` (Legendre), the modal basis is Legendre
polynomials ``P_α``, and the mass matrix is diagonal under the Lebesgue
inner product ``⟨f, g⟩ = ∫ f g dx`` (the choice locked by the plan).

Why Legendre and not Chebyshev:
* The plan's acceptance criterion — Gauss-Lobatto quadrature exact for
  polynomials of degree ``2n-3`` — is the Legendre-Gauss-Lobatto rule.
  The corresponding Chebyshev-Lobatto rule (Clenshaw-Curtis) is only
  exact to degree ``n`` under Lebesgue.
* The plan locks the Lebesgue inner product (so ``ε`` retains absolute
  physical meaning — see ``feedback_no_fudge_factors``).  Chebyshev
  polynomials of the first kind ``T_α`` are orthogonal under the
  *Chebyshev* weight ``1/√(1-ξ²)``, not Lebesgue; their Lebesgue mass
  matrix has off-diagonal structure.  Legendre polynomials are diagonal
  under Lebesgue.

The polynomial choice is therefore a faithful refinement of the plan,
not a deviation.  Everything else — node placement, quadrature accuracy,
domain conventions, the public API surface — matches the plan.

Public exports
--------------
``LegendreLobatto1D``    — 1-D basis on ``[a, b]`` (an ``eqx.Module``).
``TensorProductBasis``    — tensor product of two 1-D bases.
"""

from __future__ import annotations

from .legendre import LegendreLobatto1D
from .natal import (
    evaluate_natal_density,
    evaluate_natal_density_from_log,  # DEPRECATED — see basis/natal.py
    natal_log_density_at,
    project_natal_density,
    project_natal_log_density,  # DEPRECATED — see basis/natal.py
)
from .tensor import TensorProductBasis

# project_natal_log_density / evaluate_natal_density_from_log are deprecated
# (the symmetric-floor design caps at rel_err ≈ 0.2; production path is the
# linear projection — project_natal_density / evaluate_natal_density). The
# log-projection functions remain exported so existing call sites continue
# to work and so they're available for future revival; see their docstrings.
__all__ = [
    "LegendreLobatto1D",
    "TensorProductBasis",
    "natal_log_density_at",
    "project_natal_density",
    "evaluate_natal_density",
    "project_natal_log_density",
    "evaluate_natal_density_from_log",
]
