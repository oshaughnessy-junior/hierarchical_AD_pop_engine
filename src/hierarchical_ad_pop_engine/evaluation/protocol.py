r"""``RateDensityEvaluator`` — common point-query interface across solvers.

The forward merger-rate density

.. math::

    \frac{dC_{\rm forward}}{dt\, dm_1\, dm_2\, d\chi_1\, d\chi_2}
        \;=\;
        \tfrac{1}{2}\, R_{\rm env}\, K(m_1, m_2)\,
        N(m_1, \chi_1)\, N(m_2, \chi_2)

is produced by three different solver backends (implicit grid Picard,
basis Galerkin, particle MC).  Each backend stores its state in a
solver-specific representation — a grid array, a coefficient vector,
or a sample cloud — but every downstream consumer (plotting code,
gwkokab likelihood, ad-hoc inspection) needs the *same* point queries:

* a single-BH density ``N(m, χ)`` at arbitrary ``(m, χ)``;
* the 4-D rate density at arbitrary ``(m_1, m_2, χ_1, χ_2)``;
* 1-D / 2-D marginals at arbitrary query points on the kept axis (or
  pair of axes);
* a scalar total rate.

This Protocol pins that contract.  See ``demo/PLAN_arbitrary_point_evaluation.md``
for the resolved design questions (k-NN for the MC backend, no AD
on marginals, single-BH density backed by the surviving particle pool,
basis evaluators recompute K analytically per call, etc.).
"""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

from jax import Array

__all__ = ["RateDensityEvaluator"]


@runtime_checkable
class RateDensityEvaluator(Protocol):
    r"""Point-query interface to a forward merger-rate density.

    All methods accept JAX arrays of any broadcast-compatible shape;
    output shape matches the input under standard broadcasting.  Query
    points outside the evaluator's representable region return 0
    (a density-of-zero convention; callers must clamp / mask
    separately if they need a sentinel).

    AD
    --
    The :class:`ImplicitEvaluator` and the (forthcoming) basis
    evaluator are JAX-traceable end-to-end on :meth:`density` and
    :meth:`rate_density_4d`; the MC evaluator may be numpy-only
    (validation backstop, not in the inference AD path).

    :meth:`marginal_1d`, :meth:`marginal_2d` and :meth:`total_rate`
    are **not required** to be AD-traceable — they exist for plots
    and one-off export queries.  Inference paths use the
    grid-quadrature reductions in
    :mod:`hierarchical_ad_pop_engine.observables` directly.
    """

    def density(self, m: Array, chi: Array) -> Array:
        r"""Single-BH density ``N(m, χ)``.  Non-negative."""
        ...

    def rate_density_4d(
        self,
        m1: Array,
        m2: Array,
        chi1: Array,
        chi2: Array,
    ) -> Array:
        r"""Forward-rate density ``½ R_env K(m₁,m₂) N(m₁,χ₁) N(m₂,χ₂)``."""
        ...

    def marginal_1d(self, axis: str, x: Array) -> Array:
        r"""Marginal forward-rate density ``dC/dx`` at query points ``x``.

        Parameters
        ----------
        axis
            One of ``'m1'``, ``'m2'``, ``'chi1'``, ``'chi2'`` (grid-aligned)
            or — when the backend supports it — a derived axis label
            ``'q'``, ``'chi_eff'``, ``'M_chirp'``, ``'m_heavy'``,
            ``'m_light'``.  Backends that do not support a particular
            derived axis raise ``NotImplementedError`` with an actionable
            message; the grid-aligned four are required of every
            implementation.
        x
            Query values on the kept axis.
        """
        ...

    def marginal_2d(
        self,
        axes: Tuple[str, str],
        x_a: Array,
        x_b: Array,
    ) -> Array:
        r"""2-D marginal ``dC/(dx_a dx_b)`` at query points.

        ``axes`` ordering determines which axis broadcasts where; the
        implementation does **not** silently transpose.  Same
        grid-aligned / derived-axis convention as :meth:`marginal_1d`.
        """
        ...

    def total_rate(self) -> Array:
        r"""Scalar ``∫∫∫∫ ½ R_env K N N dm₁ dm₂ dχ₁ dχ₂``."""
        ...
