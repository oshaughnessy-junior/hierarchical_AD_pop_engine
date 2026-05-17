"""Parametric pairing / encounter kernel K(m1, m2; θ_K).

For v1 we use a separable mass kernel:

.. math::

    K(m_1, m_2)  \\propto  (m_1 + m_2)^a \\, q^b

with ``q = \\min(m_1,m_2) / \\max(m_1,m_2) \\in (0, 1]``.

- ``a`` controls the total-mass dependence (cross-section scaling): values
  around ``a ≈ 4/3`` correspond to gravitational-focusing-dominated pairing
  in dense environments, ``a = 0`` to a flat encounter rate.
- ``b`` controls the mass-ratio bias: ``b > 0`` favours equal-mass pairs,
  ``b < 0`` favours extreme mass ratios.

This is the same functional form used in the GWTC-3 power-law-peak
mass-ratio model and in many population-synthesis papers. v2 may extend with
spin-dependent factors.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def log_pairing_kernel(
    m1: Array,
    m2: Array,
    a: Array,
    b: Array,
) -> Array:
    """Return ``log K(m1, m2)`` (unnormalised) on the input mass points.

    Inputs ``m1`` and ``m2`` may be any broadcast-compatible shape. The kernel
    is symmetric in ``m1 ↔ m2`` by construction (uses min/max).
    """
    m_big = jnp.maximum(m1, m2)
    m_sml = jnp.minimum(m1, m2)
    q = m_sml / m_big
    return a * jnp.log(m1 + m2) + b * jnp.log(q)


def log_pairing_kernel_grid(
    m_grid: Array,
    a: Array,
    b: Array,
) -> Array:
    """Evaluate the kernel on the cross-product of ``m_grid`` with itself.

    Returns an ``(N_m, N_m)`` array of ``log K(m_i, m_j)``.
    """
    m1 = m_grid[:, None]
    m2 = m_grid[None, :]
    return log_pairing_kernel(m1, m2, a, b)
