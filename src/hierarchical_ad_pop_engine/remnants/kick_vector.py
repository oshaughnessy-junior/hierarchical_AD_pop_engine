"""Full vector-spin Lousto-Zlochower remnant kick — with superkick.

This is the v2 generalisation of :mod:`remnants.kick` (aligned-only)
to vector progenitor spins.  It is the fulfilment of the
"tilt-marginalised kicks are a separate v2 module" docstring in
:mod:`remnants.kick`.

Three components contribute to the recoil velocity:

* **Mass-asymmetry kick** ``v_m`` — in the orbital plane, along the
  binary separation axis at merger.  Magnitude::

      v_m = A η² · (1 − q) / (1 + q) · (1 + B η)

  This is identical to the aligned-only ``v_kick_aligned`` mass term
  (Fitchett-Detweiler form).

* **Aligned-spin in-plane kick** ``v_⊥`` — in the orbital plane,
  perpendicular to ``v_m``.  Magnitude::

      v_⊥ = H · η² / (1 + q) · |a_2_∥ − q · a_1_∥|

  where ``a_i_∥ = a_i_vec · L̂`` is the signed aligned component.
  Identical algebra to the existing aligned ``v_par`` term (note
  the existing module's misnamed ``_K`` constant is physically the
  in-plane aligned-spin amplitude ``H``).

* **Superkick** ``v_∥`` — out of the orbital plane, along ``L̂``.
  Sourced by the *perpendicular* components of the spin vectors;
  vanishes for aligned input.  Magnitude::

      v_∥ = K_s · η² / (1 + q) · |a_2_⊥ − q · a_1_⊥| · cos Θ

  where ``a_i_⊥ = a_i_vec − a_i_∥ L̂`` is the in-plane vector,
  ``|·|`` is its Euclidean magnitude, and ``cos Θ`` is the phase
  factor between the in-plane spin difference and a reference
  in-plane direction at merger.

Phase angle convention
----------------------
For aligned inputs ``a_i_⊥ = 0`` ⇒ ``v_∥ = 0`` regardless of phase
choice; the aligned reduction is exact for any ``cos Θ``.  For
misaligned inputs we adopt **``cos Θ ≡ 1`` (max-superkick) per
quadrature point**.  This is a deterministic, smooth function of the
input vectors and is the right physics primitive to feed into a
quadrature wrapper.  Refinements (phase averaging, distribution over
``Θ``) live in the Session B :mod:`remnants.marginal` wrappers, not
here.

The three components are combined with ``v_m`` and ``v_⊥`` orthogonal
in the plane and ``v_∥`` orthogonal to both::

    v_kick = √( v_m² + v_⊥² + v_∥² )

This convention preserves the aligned reduction exactly:
when ``a_i_⊥ = 0``, ``v_kick = √(v_m² + v_⊥²)`` — identical algebra
to :func:`remnants.kick.v_kick_aligned`.

Coefficients
------------
* ``A = 1.2 × 10⁴ km/s``, ``B = −0.93`` — mass-asymmetry (matches
  :mod:`remnants.kick`).
* ``H = 6.0 × 10⁴ km/s`` — in-plane aligned-spin kick (matches the
  ``_K`` value in :mod:`remnants.kick`, retaining bit-equivalent
  aligned reduction; renamed here to its standard physical symbol).
* ``K_s = 6.0 × 10⁴ km/s`` — superkick amplitude.  This is the
  leading Lousto-Zlochower 2008 normalisation; gives ``≳ 10³ km/s``
  recoils for maximally spinning, equal-mass, anti-aligned in-plane
  configurations — easily exceeding any realistic ``v_esc``.  More
  refined NR-calibrated polynomial fits (Healy & Lousto 2018) live
  as a Session-D follow-up.

Canonicalisation
----------------
Mirrors :func:`remnants.kick.v_kick_aligned`: canonicalises ``m_1 ≥
m_2`` internally and swaps the spin vectors accordingly.  Safe to
call from already-canonical call sites.

References
----------
Campanelli, Lousto, Zlochower & Merritt 2007, PRL 98:231102 —
  superkick discovery.
Lousto & Zlochower 2008, ApJL 685:L129 — fitting formula.
Lousto & Zlochower 2011, PRD 83:024003 — polynomial extension
  (Session-D upgrade target).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from .mass import eta as _eta

# Lousto-Zlochower coefficients (km/s).  ``_H`` matches the
# (mis-named) ``_K`` in remnants/kick.py — keeping the same numerical
# value preserves aligned-reduction.
_A = 1.2e4
_B = -0.93
_H = 6.0e4
_K_S = 6.0e4


def v_kick_vec(
    m1: Array,
    m2: Array,
    chi1_vec: Float[Array, "... 3"],
    chi2_vec: Float[Array, "... 3"],
) -> Array:
    """Vector-spin remnant kick magnitude in km/s.

    Internally canonicalises ``m_1 ≥ m_2`` (swaps spin vectors with
    the masses).  Returns the recoil magnitude ``v_kick``, smooth and
    differentiable in every input.  Reduces *exactly* (modulo FP
    rounding) to :func:`remnants.kick.v_kick_aligned` when both input
    spins lie along ``ẑ``.

    Phase convention for the out-of-plane "superkick" term: ``cos Θ ≡
    1`` (max superkick), see module docstring.

    Parameters
    ----------
    m1, m2
        Progenitor masses, scalar arrays.
    chi1_vec, chi2_vec
        Vector dimensionless spins, shape ``(..., 3)``.

    Returns
    -------
    v_kick : Array
        Magnitude of the recoil velocity in km/s.  Bounded
        (numerically) by a few × ``10⁴`` km/s for extreme inputs.
    """
    # Defensive wrap so Python-float / numpy-array inputs work; keeps
    # downstream JAX-array slicing (swap[..., None]) safe.
    m1 = jnp.asarray(m1)
    m2 = jnp.asarray(m2)
    chi1_vec = jnp.asarray(chi1_vec)
    chi2_vec = jnp.asarray(chi2_vec)
    # Canonicalise: m_big = max(m1, m2), and the corresponding spin
    # vector plays the role of χ_1 in the formula.
    swap = m1 < m2
    m_big = jnp.where(swap, m2, m1)
    m_sml = jnp.where(swap, m1, m2)
    swap_v = swap[..., None]
    chi_big_vec = jnp.where(swap_v, chi2_vec, chi1_vec)
    chi_sml_vec = jnp.where(swap_v, chi1_vec, chi2_vec)

    q = m_sml / m_big
    e = _eta(m_big, m_sml)

    # Decompose each spin into parallel (along L̂ = ẑ) and
    # perpendicular (in orbital plane) components.
    a_big_par = chi_big_vec[..., 2]  # χ · ẑ, signed
    a_sml_par = chi_sml_vec[..., 2]
    a_big_perp = chi_big_vec[..., :2]  # (x, y) components
    a_sml_perp = chi_sml_vec[..., :2]

    # Mass-asymmetry kick (in-plane).
    v_m = _A * jnp.square(e) * (1.0 - q) / (1.0 + q) * (1.0 + _B * e)

    # Aligned-spin in-plane kick.  In the convention where chi_big is
    # the heavier-mass spin, the formula in v_kick_aligned reads
    # |a_2_∥ − q a_1_∥| = |chi_sml_par − q · chi_big_par|.
    v_perp = (
        _H * jnp.square(e) * jnp.abs(a_sml_par - q * a_big_par) / (1.0 + q)
    )

    # Superkick (out-of-plane).  |a_2_⊥ − q a_1_⊥|, Euclidean norm of
    # the in-plane vector difference.  cos Θ = 1 (see module
    # docstring).
    delta_perp = a_sml_perp - q[..., None] * a_big_perp
    delta_perp_mag = jnp.sqrt(jnp.sum(jnp.square(delta_perp), axis=-1))
    v_par_outplane = (
        _K_S * jnp.square(e) * delta_perp_mag / (1.0 + q)
    )

    # Combine with mutually-orthogonal components.
    return jnp.sqrt(
        jnp.square(v_m) + jnp.square(v_perp) + jnp.square(v_par_outplane)
    )
