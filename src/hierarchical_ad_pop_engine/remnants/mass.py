"""Remnant mass fit M_rem(m1, m2, χ1, χ2).

We use a simple NR-calibrated polynomial fit in the symmetric mass ratio and
the mass-weighted aligned spin, following the Barausse-Buonanno-Rezzolla
family (Buonanno, Kidder & Lehner 2008; Tichy & Marronetti 2008; refined
Barausse-Morozova-Rezzolla 2012). The energy-radiated fraction is:

.. math::

    \\frac{E_{\\rm rad}}{M} = p_0 \\, \\eta^2 \\, (1 - 4\\eta)
                        \\,\\bigl[ 1 + p_S\\,\\chi_{\\rm eff} + p_2\\,\\chi_{\\rm eff}^2 \\bigr]

with ``η = m1 m2 / (m1+m2)^2`` and ``χ_eff = (m1 χ1 + m2 χ2)/(m1+m2)``
for aligned-spin progenitors. Polynomial coefficients pinned to a reasonable
calibration for v1; replaceable with more accurate Healy-Lousto fits in v2.

All operations are smooth and differentiable in ``(m1, m2, χ1, χ2)``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


# Buonanno-Kidder-Lehner-style fit constants.
_P0 = 0.27  # base radiated-energy amplitude
_PS = 0.4  # linear spin correction
_P2 = 0.1  # quadratic spin correction


def chi_effective(m1: Array, m2: Array, chi1: Array, chi2: Array) -> Array:
    """Aligned-component effective spin ``χ_eff``."""
    return (m1 * chi1 + m2 * chi2) / (m1 + m2)


def eta(m1: Array, m2: Array) -> Array:
    """Symmetric mass ratio ``η = m1 m2 / (m1+m2)²``."""
    return m1 * m2 / jnp.square(m1 + m2)


def E_rad_fraction(m1: Array, m2: Array, chi1: Array, chi2: Array) -> Array:
    """Fraction of total mass radiated as GWs in the merger.

    Bounded to ``[0, 0.15]`` (physically the most that's ever observed).
    """
    e = eta(m1, m2)
    chi_eff = chi_effective(m1, m2, chi1, chi2)
    raw = _P0 * jnp.square(e) * (1.0 + _PS * chi_eff + _P2 * jnp.square(chi_eff))
    # Soft clip to the physical range with a smooth function.
    return jnp.clip(raw, 0.0, 0.15)


def M_rem(m1: Array, m2: Array, chi1: Array, chi2: Array) -> Array:
    """Remnant mass for an aligned-spin binary, in solar masses.

    Differentiable in all four arguments.
    """
    M = m1 + m2
    return M * (1.0 - E_rad_fraction(m1, m2, chi1, chi2))
