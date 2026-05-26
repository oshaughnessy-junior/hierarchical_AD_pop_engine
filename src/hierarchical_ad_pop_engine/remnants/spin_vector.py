"""Full vector-spin Barausse-Rezzolla 2009 remnant spin.

This is the v2 generalisation of :mod:`remnants.spin` (aligned-only)
to vector progenitor spins.  Returns the *vector* remnant spin
``χ_rem_vec`` from progenitor masses and vector dimensionless spins.

Physical setup
--------------
The orbital angular momentum at merger is along ``L̂ = ẑ`` by convention.
Progenitor spin vectors ``χ_i_vec`` are dimensionless ``(a_i)``, with
the heavier-mass spin played by ``χ_1`` (see canonicalisation warning
below).

For aligned progenitors (``χ_i_vec = χ_i ẑ``) this module reduces to
``remnants.spin.chi_rem`` evaluated on the signed ``χ_i`` *exactly*
(verified by ``tests/test_remnant_vector.py``); the ``z``-component of
the returned vector matches the scalar value to machine precision.

Formula (this project's simplified-BR variant)
----------------------------------------------
Approximate ``J_fin ≈ S_1 + S_2 + L̃`` with ``L̃`` along ``L̂``,
neglecting radiated angular momentum.  Then::

    a_fin_vec = J_fin / M_fin²
              ≈ (χ_1_vec + q² χ_2_vec + q · l̃ · L̂) / (1 + q)²

with ``q = m_2 / m_1`` (``q ≤ 1`` after canonicalisation) and the
ISCO orbital-angular-momentum magnitude::

    l̃(q, η, χ_1_vec, χ_2_vec)
        = s_4 / (1+q²)² · (|χ_1|² + |χ_2|² q⁴)
        + (s_5 η + t_0 + 2) / (1+q²) · (χ_1_z + q² χ_2_z)
        + 2√3 + t_2 η + t_3 η²

with the same coefficients ``(s_4, s_5, t_0, t_2, t_3)`` as the scalar
formula.  The aligned reduction is bit-equivalent: ``|χ_i| cos β_i =
χ_i_z`` since ``L̂ = ẑ`` — no division by spin magnitudes is needed
(so ``|χ_i| = 0`` is handled cleanly).

.. note::
    The canonical Barausse-Rezzolla 2009 ``l̃`` formula carries an
    extra ``+ s_4 / (1+q²)² · 2 q² χ_1·χ_2 cos α`` cross term that
    the project's scalar :mod:`remnants.spin` ``chi_rem`` does NOT
    include.  This vector module mirrors the scalar choice
    deliberately, so the aligned-reduction invariant (vector at
    ẑ-aligned input → scalar exactly, machine precision) holds.
    A separate decision is whether to upgrade *both* the scalar and
    vector formulas to canonical BR 2009 in a later session; that
    would re-baseline a small handful of pinned scalar outputs and
    is *not* in scope for Session A.  This discrepancy is recorded
    in the Session A handoff memory.

The output magnitude is clipped to ``[0, 0.999]`` for safety (matches
the scalar code's ``±0.999`` clip).

Canonicalisation
----------------
Like the aligned ``chi_rem``, the BR formula carries an implicit
``m_1 ≥ m_2`` convention and is **NOT** symmetric under
``(m_1, χ_1) ↔ (m_2, χ_2)``.  ``chi_rem_vec`` here canonicalises
internally (mirroring ``v_kick_aligned``'s pattern, not the scalar
``chi_rem``'s caller-canonicalises pattern) — see memory
``project_chi_rem_asymmetry_finding`` for the history.  This makes the
public surface harder to misuse; the basis-side T-build call sites
that canonicalise externally can still pre-swap and pass an already-
canonical ``(m_1, χ_1)`` pair without harm.

References
----------
Barausse & Rezzolla 2009, ApJ 704:L40 — final spin formula.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

# Barausse-Rezzolla 2009 coefficients — kept identical to the scalar
# ``remnants.spin`` module so the aligned reduction is bit-equivalent at
# the algebra level (and machine-precision after FP rounding).
_S4 = -0.1229
_S5 = 0.4537
_T0 = -2.8904
_T2 = -3.5171
_T3 = 2.5763
_SQRT3 = jnp.sqrt(3.0)


def _l_vec(
    q: Array,
    eta_val: Array,
    chi1_vec: Float[Array, "... 3"],
    chi2_vec: Float[Array, "... 3"],
) -> Array:
    """ISCO orbital-angular-momentum magnitude for vector inputs.

    See module docstring.  Written entirely in dot products so
    ``|χ_i| = 0`` is well-defined.  Mirrors the simplified-BR ``l`` in
    :mod:`remnants.spin` (no ``χ_1·χ_2`` cross term — see module note).
    """
    one_plus_q2 = 1.0 + jnp.square(q)
    mag1_sq = jnp.sum(jnp.square(chi1_vec), axis=-1)
    mag2_sq = jnp.sum(jnp.square(chi2_vec), axis=-1)
    chi1_z = chi1_vec[..., 2]
    chi2_z = chi2_vec[..., 2]
    a_sum_sq = mag1_sq + mag2_sq * jnp.square(jnp.square(q))
    parallel_combo = chi1_z + jnp.square(q) * chi2_z
    return (
        _S4 / jnp.square(one_plus_q2) * a_sum_sq
        + (_S5 * eta_val + _T0 + 2.0) / one_plus_q2 * parallel_combo
        + 2.0 * _SQRT3
        + _T2 * eta_val
        + _T3 * jnp.square(eta_val)
    )


def chi_rem_vec(
    m1: Array,
    m2: Array,
    chi1_vec: Float[Array, "... 3"],
    chi2_vec: Float[Array, "... 3"],
) -> Float[Array, "... 3"]:
    """Dimensionless vector remnant spin ``χ_rem_vec`` (Barausse-Rezzolla 2009).

    Canonicalises ``m_1 ≥ m_2`` internally (swaps the spin vectors along
    with the masses).  The magnitude of the returned vector is clipped
    to ``[0, 0.999]`` for safety.  Reduces *exactly* to
    :func:`remnants.spin.chi_rem` (in the ``z``-component) when both
    input spins are along ``ẑ``.

    Differentiable in every input — uses only dot products and
    arithmetic, no branches on spin magnitude.

    Parameters
    ----------
    m1, m2
        Progenitor masses, scalar arrays.
    chi1_vec, chi2_vec
        Vector dimensionless spins, shape ``(..., 3)``, components in
        the ``(x, y, z)`` orbital-plane / ``L̂`` frame.

    Returns
    -------
    chi_rem_vec
        Vector remnant spin, shape ``(..., 3)``.  Always has finite
        magnitude ``≤ 0.999``.
    """
    # Defensive wrap so Python-float / numpy-array inputs work; keeps
    # downstream JAX-array slicing (swap[..., None]) safe.
    m1 = jnp.asarray(m1)
    m2 = jnp.asarray(m2)
    chi1_vec = jnp.asarray(chi1_vec)
    chi2_vec = jnp.asarray(chi2_vec)
    # Canonicalise so m1 carries the heavier mass and chi1_vec the
    # corresponding spin.  Pre-broadcast the mass-comparison mask
    # against the vector trailing dim.
    swap = m1 < m2
    m_big = jnp.where(swap, m2, m1)
    m_sml = jnp.where(swap, m1, m2)
    swap_v = swap[..., None]
    chi1_v = jnp.where(swap_v, chi2_vec, chi1_vec)
    chi2_v = jnp.where(swap_v, chi1_vec, chi2_vec)

    q = m_sml / m_big
    eta_val = q / jnp.square(1.0 + q)
    l_val = _l_vec(q, eta_val, chi1_v, chi2_v)

    # Vector sum a_fin = (χ_1_vec + q² χ_2_vec + q l̃ ẑ) / (1+q)².
    z_hat = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), chi1_v.shape)
    j_over_M2 = (
        chi1_v
        + jnp.square(q)[..., None] * chi2_v
        + (q * l_val)[..., None] * z_hat
    )
    a_fin = j_over_M2 / jnp.square(1.0 + q)[..., None]

    # Clip magnitude to |a_fin| ≤ 0.999 — match the scalar code's
    # safety clip.  Smooth: scale by min(1, 0.999/mag).
    mag = jnp.sqrt(jnp.sum(jnp.square(a_fin), axis=-1))
    scale = jnp.minimum(1.0, 0.999 / jnp.maximum(mag, 1e-30))
    return a_fin * scale[..., None]
