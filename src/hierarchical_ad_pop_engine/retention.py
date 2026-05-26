"""Retention probability ``P(retain | binary, environment)``.

Dispatches on the type of the :class:`spin_orientation.OrientationModel`
passed in:

* :class:`spin_orientation.Aligned` (default) → existing scalar
  Heaviside path :func:`remnants.kick.retention_aligned` **bit-for-bit
  unchanged**.  Every existing call site that doesn't pass an explicit
  ``orientation`` gets the v1 semantics.  Gradients w.r.t. ``v_esc``
  are zero almost everywhere here (Heaviside).
* Any non-Aligned :class:`spin_orientation.OrientationModel` →
  marginalised retention :func:`remnants.marginal.P_retain_marginalised`.
  Under any non-trivial ``P(θ̂)`` the marginalised retention is smooth
  in ``v_esc`` (the orientation integral acts as a Riemann sum over a
  smooth CDF), recovering ``∂/∂v_esc`` gradients — the AD-on-``v_esc``
  two-for-one promise from ``demo/PLAN_misalignment_mode.md``.

Dispatch is at trace time via ``isinstance(orientation, Aligned)`` —
Python-time branch, jit-stable.  Two code paths, both jit'd; no
runtime dispatch cost.

Spin convention
---------------
The ``chi*`` arguments are interpreted differently per branch, by the
same logic as :mod:`remnants.marginal`:

* **Aligned**: ``chi1``, ``chi2`` are signed scalars ``∈ [-1, 1]`` (v1
  convention).  Sign carries the alignment direction.
* **Non-Aligned**: ``chi1``, ``chi2`` are magnitudes ``|χ_i| ∈ [0, 1]``
  (misalignment-mode convention).  The orientation distribution
  ``P(θ̂)`` carries the tilt; the spin magnitude is the only physical
  scalar that survives the marginalisation.

Callers that switch from aligned to misaligned mode must therefore
re-interpret their stored χ-grid as a magnitude grid; that wiring
lives at the engine call sites (Session B step 6 on the coagulation
side; the basis tensors already take ``chi_mag`` semantics by virtue
of the soft sigmoid being applied to a deterministic ``v_kick_vec``
evaluation).
"""

from __future__ import annotations

from typing import Literal, Optional

from jax import Array

from .remnants.kick import retention_aligned
from .remnants.marginal import P_retain_marginalised
from .spin_orientation import Aligned


def P_retain(
    m1: Array,
    m2: Array,
    chi1: Array,
    chi2: Array,
    v_esc: Array,
    orientation: Optional[object] = None,
    *,
    retention_kind: Literal["heaviside", "sigmoid"] = "heaviside",
    w_over_v_esc: float = 1.0,
) -> Array:
    """Retention probability, dispatched on the orientation model.

    Parameters
    ----------
    m1, m2 : Array
        Progenitor masses.
    chi1, chi2 : Array
        Progenitor spins.  Signed ``∈ [-1, 1]`` under :class:`Aligned`;
        magnitudes ``|χ_i| ∈ [0, 1]`` under any non-Aligned orientation
        model.  See module docstring.
    v_esc : Array
        Host escape velocity (km/s).
    orientation : OrientationModel, optional
        Prescribed spin-orientation distribution.  Defaults to
        :class:`Aligned` (v1 behaviour, bit-for-bit Heaviside).  Pass
        :class:`Isotropic` or :class:`Fisher` to recover ``v_esc``
        dependence via tilt marginalisation.
    retention_kind : {"heaviside", "sigmoid"}
        Per-orientation retention model under non-Aligned.  Ignored
        for :class:`Aligned` (which always shortcuts to the
        :func:`remnants.kick.retention_aligned` Heaviside).

        * ``"heaviside"`` (default): hard cut at each orientation
          quadrature point.  Function value is the correct
          orientation-weighted retention, but ``jax.grad`` w.r.t.
          ``v_esc`` is 0 almost everywhere (the discrete quadrature
          is a finite sum of Heavisides — analytically a smooth CDF,
          numerically a step function with many tiny steps).
        * ``"sigmoid"``: ``σ((v_esc - v_kick)/w)`` per orientation
          with ``w = w_over_v_esc · v_esc``.  Smooth ``v_esc``
          gradients — the AD-on-``v_esc`` two-for-one win from the
          PLAN.  Bias from the soft cut is ``O(w_over_v_esc)``
          relative to the Heaviside; for population-level work the
          two agree to a few percent at ``w_over_v_esc = 1``.
    w_over_v_esc : float
        Sigmoid sharpness as a fraction of ``v_esc`` (only used when
        ``retention_kind="sigmoid"``).  See ``demo/PLAN_engine_v3_basis.md``
        § Step 3 Decision 3 for the rationale.

    Returns
    -------
    Array
        Retention probability ``∈ [0, 1]``.
    """
    if orientation is None:
        orientation = Aligned()
    if isinstance(orientation, Aligned):
        return retention_aligned(m1, m2, chi1, chi2, v_esc)
    return P_retain_marginalised(
        m1, m2, chi1, chi2, v_esc, orientation,
        retention_kind=retention_kind,
        w_over_v_esc=w_over_v_esc,
    )
