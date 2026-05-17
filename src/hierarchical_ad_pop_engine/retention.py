"""Retention probability P(retain | binary, environment).

Re-exports the aligned-only Heaviside retention from
:mod:`hierarchical_ad_pop_engine.remnants.kick`. Future tilt-marginalised
variants will plug in here behind the same function name.
"""

from __future__ import annotations

from jax import Array

from .remnants.kick import retention_aligned


def P_retain(
    m1: Array,
    m2: Array,
    chi1: Array,
    chi2: Array,
    v_esc: Array,
) -> Array:
    """Aligned-only Heaviside retention: ``1.0`` if ``v_kick < v_esc`` else ``0.0``.

    Not differentiable in ``v_esc``. See :mod:`remnants.kick` for the AD
    consequences of aligned-only physics.
    """
    return retention_aligned(m1, m2, chi1, chi2, v_esc)
