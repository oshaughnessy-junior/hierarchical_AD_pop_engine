"""Tilt-marginalised wrappers over an :class:`OrientationModel`.

This module is **pure plumbing** — it integrates / samples the existing
vector remnant primitives :func:`remnants.spin_vector.chi_rem_vec` and
:func:`remnants.kick_vector.v_kick_vec` against a prescribed orientation
distribution ``P(θ̂)``.  No new physics lives here.

Two consumers:

* **Deterministic side** (basis / implicit / coagulation).  Iterates the
  flat quadrature array ``orientation.quadrature_3d()`` and accumulates
  weighted contributions.  The high-level ``chi_rem_distribution`` /
  ``v_kick_distribution`` wrappers return ``(scalar_values_at_quad,
  weights)`` so engine call sites can plug into their existing
  bilinear-scatter / Galerkin-projection machinery without a special
  per-orientation routine.
* **Monte-Carlo side** (``red_team/particle_mc.py``).  Single draw from
  ``orientation.sample(key, 1)``, evaluates the per-orientation
  scalar, returns it.

Per-orientation primitives are exposed as :func:`chi_rem_at_orientation`
/ :func:`v_kick_at_orientation` for the basis-side ``jax.lax.scan``
call sites — those need *one* scalar contribution alive at a time per
the Decision 3 memory-budget constraint and so cannot consume the
``_distribution`` wrappers directly.

Convention for the azimuthal collapse
-------------------------------------
``OrientationModel.quadrature_3d()`` returns ``(cos θ_1, cos θ_2, Δφ,
w)`` with one azimuthal integration folded into ``w``.  Because all
three concrete orientation models ship with axisymmetric ``P(θ̂)``
about ``L̂``, only the relative azimuth ``Δφ`` is physical.  We pick
``φ_1 = 0`` and ``φ_2 = Δφ`` at the per-orientation primitive level —
``v_kick_vec`` and ``|χ_rem_vec|`` are both invariant under rigid
rotation about ``L̂`` so the choice is unobservable.

Aligned shortcut at every entry point
-------------------------------------
``Aligned()`` is dispatched to the scalar
:func:`remnants.spin.chi_rem` / :func:`remnants.kick.v_kick_aligned` /
:func:`remnants.kick.retention_aligned` paths bit-for-bit.  The input
``(chi1_mag, chi2_mag)`` is interpreted as the signed scalar χ in the
``Aligned`` case (the aligned engine treats χ as signed ∈ [-1, 1]).
For non-Aligned orientations, ``chi_mag`` is the magnitude
``|χ| ≥ 0`` per the misalignment-mode convention — the orientation
quadrature carries the sign / tilt.

Differentiability
-----------------
Every wrapper here is end-to-end differentiable in ``(m1, m2,
chi1_mag, chi2_mag, v_esc)`` and in ``orientation.kappa`` (for the
:class:`spin_orientation.Fisher` case) via the quadrature weights.

Cross-references: ``demo/PLAN_misalignment_mode.md`` § Session B, step 4;
memories ``project-misalignment-session-a-done``,
``feedback-no-fake-physics-stand-ins``,
``feedback-no-frozen-test-references``.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from ..spin_orientation import Aligned
from .kick import retention_aligned, v_kick_aligned
from .kick_vector import v_kick_vec
from .spin import chi_rem
from .spin_vector import chi_rem_vec


# ---------------------------------------------------------------------------
# Per-orientation primitives.
#
# These are the scalar-output building blocks the basis-side and the
# coagulation-side ``jax.lax.scan`` over-orientations call sites consume.
# A single ``(cos θ_1, cos θ_2, Δφ)`` triple in, a single scalar out
# (broadcasting cleanly over any progenitor-mesh leading shape).
# ---------------------------------------------------------------------------
def _vectors_at_orientation(
    chi1_mag: Array,
    chi2_mag: Array,
    cos_theta_1: Array,
    cos_theta_2: Array,
    dphi: Array,
) -> tuple[Array, Array]:
    """Promote scalar spin magnitudes to vectors at one orientation point.

    ``φ_1 = 0``, ``φ_2 = Δφ`` (azimuthal-symmetry collapse — both
    output magnitudes are invariant under rigid rotation about ``L̂``,
    so this choice is unobservable).

    Returns ``(chi1_vec, chi2_vec)`` of shape ``(..., 3)`` where the
    leading shape is the broadcast of the inputs.
    """
    # |sin θ| ≥ 0 always.  Clip before sqrt against round-off-induced
    # tiny-negatives at |cos θ| = 1.
    sin_t1 = jnp.sqrt(jnp.clip(1.0 - jnp.square(cos_theta_1), 0.0, 1.0))
    sin_t2 = jnp.sqrt(jnp.clip(1.0 - jnp.square(cos_theta_2), 0.0, 1.0))
    cos_dphi = jnp.cos(dphi)
    sin_dphi = jnp.sin(dphi)

    chi1_x = chi1_mag * sin_t1
    chi1_y = jnp.zeros_like(chi1_x)
    chi1_z = chi1_mag * cos_theta_1
    chi2_x = chi2_mag * sin_t2 * cos_dphi
    chi2_y = chi2_mag * sin_t2 * sin_dphi
    chi2_z = chi2_mag * cos_theta_2

    chi1_vec = jnp.stack([chi1_x, chi1_y, chi1_z], axis=-1)
    chi2_vec = jnp.stack([chi2_x, chi2_y, chi2_z], axis=-1)
    return chi1_vec, chi2_vec


def chi_rem_at_orientation(
    m1: Array,
    m2: Array,
    chi1_mag: Array,
    chi2_mag: Array,
    cos_theta_1: Array,
    cos_theta_2: Array,
    dphi: Array,
) -> Array:
    """Scalar ``|χ_rem|`` at one orientation quadrature point.

    Returns the **magnitude** of the vector remnant spin
    :func:`remnants.spin_vector.chi_rem_vec`.  Smooth and
    differentiable in every input.  Broadcasts cleanly when the
    progenitor inputs ``(m1, m2, chi1_mag, chi2_mag)`` carry a leading
    progenitor-mesh shape and the orientation triple is scalar — the
    canonical use pattern inside a ``jax.lax.scan`` body.
    """
    chi1_vec, chi2_vec = _vectors_at_orientation(
        chi1_mag, chi2_mag, cos_theta_1, cos_theta_2, dphi
    )
    chi_rem_v = chi_rem_vec(m1, m2, chi1_vec, chi2_vec)
    return jnp.sqrt(jnp.sum(jnp.square(chi_rem_v), axis=-1))


def v_kick_at_orientation(
    m1: Array,
    m2: Array,
    chi1_mag: Array,
    chi2_mag: Array,
    cos_theta_1: Array,
    cos_theta_2: Array,
    dphi: Array,
) -> Array:
    """Scalar ``v_kick`` (km/s) at one orientation quadrature point.

    Wraps :func:`remnants.kick_vector.v_kick_vec`.  Same broadcast
    semantics as :func:`chi_rem_at_orientation`.
    """
    chi1_vec, chi2_vec = _vectors_at_orientation(
        chi1_mag, chi2_mag, cos_theta_1, cos_theta_2, dphi
    )
    return v_kick_vec(m1, m2, chi1_vec, chi2_vec)


# ---------------------------------------------------------------------------
# Marginalised distributions — vmap over the orientation quadrature.
#
# Suited to scalar progenitor inputs (one (m1, m2, chi1_mag, chi2_mag)
# point at a time).  The basis-side / coagulation-side use a
# ``jax.lax.scan`` over orientations at the engine level (not these
# wrappers) because they carry a 4-D progenitor mesh and the
# ``vmap``-over-quadrature here would blow peak memory.
# ---------------------------------------------------------------------------
def chi_rem_distribution(
    m1: Array,
    m2: Array,
    chi1_mag: Array,
    chi2_mag: Array,
    orientation,
) -> tuple[Array, Array]:
    """Marginalised ``|χ_rem|`` distribution at scalar progenitor inputs.

    Returns
    -------
    chi_rem_quad : Array of shape ``(n_quad,)``
        ``|χ_rem|`` evaluated at each orientation quadrature point.
    weights : Array of shape ``(n_quad,)``
        Normalised orientation-distribution weights ``Σ w_k = 1``
        (to quadrature accuracy).

    Aligned shortcut: returns
    ``([chi_rem(m1, m2, chi1_mag, chi2_mag)], [1.0])`` interpreting
    ``chi*_mag`` as the signed χ.
    """
    if isinstance(orientation, Aligned):
        x = chi_rem(m1, m2, chi1_mag, chi2_mag)
        return jnp.asarray(x)[None], jnp.ones((1,))
    quad = orientation.quadrature_3d()
    cos_t1 = quad[..., 0]
    cos_t2 = quad[..., 1]
    dphi = quad[..., 2]
    w = quad[..., 3]
    chi_rem_q = jax.vmap(
        lambda ct1, ct2, dp: chi_rem_at_orientation(
            m1, m2, chi1_mag, chi2_mag, ct1, ct2, dp
        )
    )(cos_t1, cos_t2, dphi)
    return chi_rem_q, w


def v_kick_distribution(
    m1: Array,
    m2: Array,
    chi1_mag: Array,
    chi2_mag: Array,
    orientation,
) -> tuple[Array, Array]:
    """Marginalised ``v_kick`` distribution at scalar progenitor inputs.

    Aligned shortcut returns the existing scalar
    :func:`remnants.kick.v_kick_aligned` value with unit weight.
    """
    if isinstance(orientation, Aligned):
        v = v_kick_aligned(m1, m2, chi1_mag, chi2_mag)
        return jnp.asarray(v)[None], jnp.ones((1,))
    quad = orientation.quadrature_3d()
    cos_t1 = quad[..., 0]
    cos_t2 = quad[..., 1]
    dphi = quad[..., 2]
    w = quad[..., 3]
    v_q = jax.vmap(
        lambda ct1, ct2, dp: v_kick_at_orientation(
            m1, m2, chi1_mag, chi2_mag, ct1, ct2, dp
        )
    )(cos_t1, cos_t2, dphi)
    return v_q, w


def P_retain_marginalised(
    m1: Array,
    m2: Array,
    chi1_mag: Array,
    chi2_mag: Array,
    v_esc: Array,
    orientation,
    *,
    retention_kind: Literal["heaviside", "sigmoid"] = "heaviside",
    w_over_v_esc: Array | float = 1.0,
) -> Array:
    """``P_retain`` marginalised over the orientation distribution.

    The integral of ``1[v_kick(orient) < v_esc]`` (or its sigmoid
    smoothing) against ``P(θ̂)``::

        P_retain_marginalised
            = Σ_k w_k · retention(v_kick_k; v_esc, w)

    Smooth in ``v_esc`` for any non-Aligned orientation — even with
    ``retention_kind="heaviside"`` the orientation quadrature acts as
    a Riemann sum over a smooth CDF (the integral of a Heaviside over a
    smooth distribution is a smooth CDF).  This is what cashes in the
    AD-on-``v_esc`` two-for-one promise from the PLAN.

    Aligned shortcut: routes to
    :func:`remnants.kick.retention_aligned` — preserves the v2
    Heaviside behaviour bit-for-bit.

    Parameters
    ----------
    retention_kind : {"heaviside", "sigmoid"}
        ``"heaviside"`` for the v2-style hard cut at each orientation
        quadrature point.  ``"sigmoid"`` for the v3-basis-style soft
        cut, with sharpness ``w = w_over_v_esc · v_esc``.  Both yield
        a smooth marginalised ``P_retain`` under any non-Aligned
        orientation; the sigmoid variant is additionally smooth in
        ``v_esc`` *at each orientation point* and is the right choice
        for the basis-side Galerkin projection.
    w_over_v_esc : Array or float
        Sigmoid sharpness as a fraction of ``v_esc``.  Ignored for
        ``retention_kind="heaviside"``.
    """
    if isinstance(orientation, Aligned):
        return retention_aligned(m1, m2, chi1_mag, chi2_mag, v_esc)
    v_q, w = v_kick_distribution(m1, m2, chi1_mag, chi2_mag, orientation)
    # v_q shape is ``(n_quad, *progenitor_broadcast_shape)`` — the
    # vmap inside ``v_kick_distribution`` puts the quadrature axis
    # leading.  At scalar progenitor inputs the trailing shape is
    # empty; at the implicit solver's 4-D progenitor-mesh broadcast
    # ``(m1, m2, chi1, chi2)`` it is ``(N_m, N_chi, N_m, N_chi)``.
    # ``w`` is always ``(n_quad,)`` and must be reshaped to broadcast
    # over the trailing axes; the sum reduces *only* over the
    # orientation axis, returning a per-cell marginalised retention.
    if retention_kind == "heaviside":
        retain_q = jnp.where(v_q < v_esc, 1.0, 0.0)
    elif retention_kind == "sigmoid":
        w_v = jnp.asarray(w_over_v_esc) * v_esc
        retain_q = jax.nn.sigmoid((v_esc - v_q) / w_v)
    else:
        raise ValueError(
            f"Unknown retention_kind {retention_kind!r}; expected "
            "'heaviside' or 'sigmoid'."
        )
    # Reshape ``w`` to ``(n_quad, 1, ..., 1)`` so it broadcasts cleanly
    # over any trailing progenitor-mesh shape.  At scalar inputs
    # (``retain_q.ndim == 1``) the reshape is a no-op so the original
    # scalar-output behaviour is preserved exactly (no breaking
    # change to Session B callers).
    w_b = w.reshape((w.shape[0],) + (1,) * (retain_q.ndim - 1))
    return jnp.sum(w_b * retain_q, axis=0)


# ---------------------------------------------------------------------------
# Single-event MC samplers.  Scalar in, scalar out per call.
# ---------------------------------------------------------------------------
def chi_rem_sample(
    key: Array,
    m1: Array,
    m2: Array,
    chi1_mag: Array,
    chi2_mag: Array,
    orientation,
) -> Array:
    """Single-event MC draw of ``|χ_rem|`` at one orientation realisation.

    Aligned shortcut returns the existing scalar ``chi_rem`` value
    (key unused).
    """
    if isinstance(orientation, Aligned):
        return chi_rem(m1, m2, chi1_mag, chi2_mag)
    theta_1, phi_1, theta_2, phi_2 = orientation.sample(key, n_events=1)
    cos_t1 = jnp.cos(theta_1[0])
    cos_t2 = jnp.cos(theta_2[0])
    # Δφ wraps modulo 2π; chi_rem_at_orientation depends on (cos Δφ,
    # sin Δφ) which are 2π-periodic so the modular reduction is moot.
    dphi = phi_2[0] - phi_1[0]
    return chi_rem_at_orientation(
        m1, m2, chi1_mag, chi2_mag, cos_t1, cos_t2, dphi
    )


def v_kick_sample(
    key: Array,
    m1: Array,
    m2: Array,
    chi1_mag: Array,
    chi2_mag: Array,
    orientation,
) -> Array:
    """Single-event MC draw of ``v_kick`` (km/s) at one orientation realisation.

    Aligned shortcut returns the existing scalar ``v_kick_aligned``
    value (key unused).
    """
    if isinstance(orientation, Aligned):
        return v_kick_aligned(m1, m2, chi1_mag, chi2_mag)
    theta_1, phi_1, theta_2, phi_2 = orientation.sample(key, n_events=1)
    cos_t1 = jnp.cos(theta_1[0])
    cos_t2 = jnp.cos(theta_2[0])
    dphi = phi_2[0] - phi_1[0]
    return v_kick_at_orientation(
        m1, m2, chi1_mag, chi2_mag, cos_t1, cos_t2, dphi
    )


__all__ = [
    "chi_rem_at_orientation",
    "v_kick_at_orientation",
    "chi_rem_distribution",
    "v_kick_distribution",
    "P_retain_marginalised",
    "chi_rem_sample",
    "v_kick_sample",
]
