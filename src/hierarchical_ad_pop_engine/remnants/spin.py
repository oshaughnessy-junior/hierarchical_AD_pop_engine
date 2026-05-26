"""Remnant spin fit χ_rem(m1, m2, χ1, χ2).

Barausse-Rezzolla (2009) "final spin formula" for aligned-spin progenitors:

.. math::

    \\chi_{\\rm rem} = \\frac{1}{(1+q)^2} \\,
                     \\sqrt{ \\chi_1^2 + \\chi_2^2 q^4 + 2 \\chi_1 \\chi_2 q^2
                             + 2 (\\chi_1 + \\chi_2 q^2) \\, q \\, l(\\eta)
                             + q^2 \\, l(\\eta)^2 }

with ``q = m2/m1`` (so ``q ≤ 1``) and the ISCO orbital-angular-momentum
function (Barausse-Rezzolla 2009 eq. (10), aligned-spin reduction —
the canonical ``a_1 \\cdot a_2 \\cos\\alpha`` cross term reduces to
``\\chi_1 \\chi_2`` for signed scalar aligned spins):

.. math::

    l(\\eta) = \\frac{s_4}{(1+q^2)^2} \\,
              \\bigl( \\chi_1^2 + \\chi_2^2 q^4 + 2 \\chi_1 \\chi_2 q^2 \\bigr)
              + \\frac{s_5 \\eta + t_0 + 2}{1+q^2} \\,
              \\bigl( \\chi_1 + \\chi_2 q^2 \\bigr)
              + 2\\sqrt{3} + t_2 \\eta + t_3 \\eta^2 .

The first parenthesised group is ``(\\chi_1 + \\chi_2 q^2)^2``, the
square of the same linear combination that appears multiplying
``q l(\\eta)`` below.  Coefficients ``(s_4, s_5, t_0, t_2, t_3)`` from
Barausse & Rezzolla 2009.  Smooth, differentiable, bounded by
``|χ_rem| ≤ 1`` for physical inputs.

.. note::
    The ``2 \\chi_1 \\chi_2 q^2`` cross term in ``l`` was originally
    omitted in this project's v1 implementation — an unintentional
    simplification that produced ``O(10^{-5})`` errors at non-trivial
    same-sign spins relative to canonical BR 2009 (and larger at
    extremal anti-aligned spins where the cross term changes sign).
    Restored 2026-05-25 as part of misalignment Session A (no-fake-
    physics rule, ``feedback-no-fake-physics-stand-ins``).  The
    vector formula in :mod:`remnants.spin_vector` already includes
    the natural inner-product generalisation
    ``2 q^2 \\chi_1 \\cdot \\chi_2``; the aligned reduction is exact.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

# Barausse-Rezzolla 2009 coefficients.
_S4 = -0.1229
_S5 = 0.4537
_T0 = -2.8904
_T2 = -3.5171
_T3 = 2.5763
_SQRT3 = jnp.sqrt(3.0)


def _l(q: Array, eta_val: Array, chi1: Array, chi2: Array) -> Array:
    one_plus_q2 = 1.0 + jnp.square(q)
    # Canonical BR 2009 eq. (10) for aligned spins:
    #   (χ_1 + χ_2 q²)² = χ_1² + χ_2² q⁴ + 2 χ_1 χ_2 q².
    # The cross term was originally omitted (see module note).
    a_sum_sq = (
        jnp.square(chi1)
        + jnp.square(chi2) * jnp.square(jnp.square(q))
        + 2.0 * chi1 * chi2 * jnp.square(q)
    )
    a_lin = chi1 + chi2 * jnp.square(q)
    return (
        _S4 / jnp.square(one_plus_q2) * a_sum_sq
        + (_S5 * eta_val + _T0 + 2.0) / one_plus_q2 * a_lin
        + 2.0 * _SQRT3
        + _T2 * eta_val
        + _T3 * jnp.square(eta_val)
    )


def chi_rem(m1: Array, m2: Array, chi1: Array, chi2: Array) -> Array:
    """Dimensionless remnant spin (aligned-spin progenitors, Barausse-Rezzolla 2009).

    Returns a value bounded in ``[-1, 1]`` for physical inputs.

    .. warning::
        **Convention: m1 ≥ m2 with χ1 the heavier-mass spin.**  The
        Barausse-Rezzolla formula weights ``χ1`` by 1 and ``χ2`` by
        ``q² = (m_light/m_heavy)²``, so the formula is NOT symmetric
        under ``(m1, χ1) ↔ (m2, χ2)`` despite the ``q = min/max``
        mass ratio.  Callers must canonicalise the spin labels
        themselves; this function does not.

        ``v_kick_aligned`` canonicalises internally, but this one
        does not — preserved for differentiability semantics and
        cheap call sites (callers like
        :func:`response.coagulation._step_one_generation` and
        :mod:`basis.remnant_tensor` do the swap once outside).  See
        memory ``project_chi_rem_asymmetry_finding`` for the
        history.
    """
    q = jnp.minimum(m1, m2) / jnp.maximum(m1, m2)
    eta_val = q / jnp.square(1.0 + q)
    l_val = _l(q, eta_val, chi1, chi2)
    inside_sqrt = (
        jnp.square(chi1)
        + jnp.square(chi2) * jnp.square(jnp.square(q))
        + 2.0 * chi1 * chi2 * jnp.square(q)
        + 2.0 * (chi1 + chi2 * jnp.square(q)) * q * l_val
        + jnp.square(q * l_val)
    )
    inside_sqrt = jnp.maximum(inside_sqrt, 0.0)
    a_fin = jnp.sqrt(inside_sqrt) / jnp.square(1.0 + q)
    return jnp.clip(a_fin, -0.999, 0.999)
