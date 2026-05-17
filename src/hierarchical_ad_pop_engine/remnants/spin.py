"""Remnant spin fit χ_rem(m1, m2, χ1, χ2).

Barausse-Rezzolla (2009) "final spin formula" for aligned-spin progenitors:

.. math::

    \\chi_{\\rm rem} = \\frac{1}{(1+q)^2} \\,
                     \\sqrt{ \\chi_1^2 + \\chi_2^2 q^4 + 2 \\chi_1 \\chi_2 q^2
                             + 2 (\\chi_1 + \\chi_2 q^2) \\, q \\, l(\\eta)
                             + q^2 \\, l(\\eta)^2 }

with ``q = m2/m1`` (so ``q ≤ 1``) and the ISCO-orbital-angular-momentum
function

.. math::

    l(\\eta) = \\frac{s_4}{(1+q^2)^2} \\, (\\chi_1^2 + \\chi_2^2 q^4)
              + \\frac{s_5 \\eta + t_0 + 2}{1+q^2} \\, (\\chi_1 + \\chi_2 q^2)
              + 2\\sqrt{3} + t_2 \\eta + t_3 \\eta^2 .

Coefficients ``(s_4, s_5, t_0, t_2, t_3)`` from Barausse & Rezzolla 2009.
Smooth, differentiable, bounded by ``|χ_rem| ≤ 1`` for physical inputs.
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
    a_sum_sq = jnp.square(chi1) + jnp.square(chi2) * jnp.square(jnp.square(q))
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
    """
    # Convention: m1 >= m2 inside the engine, but the formula is symmetric
    # if we use q = min/max.
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
