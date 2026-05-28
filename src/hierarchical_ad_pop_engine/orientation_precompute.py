"""Orientation-marginalised tensors precomputed once per implicit solve.

Session F1 of ``demo/PLAN_implicit_footprint_refactor.md`` — "Strategy A:
precompute, don't surrogate".  The orientation marginalisation does **not**
iterate inside the Picard loop (the loop varies ``N``, the orientation
quadrature does not).  This module hoists the marginalisation out of the
loop body by eagerly evaluating the two scalar 4-D tensors once at the top
of a solve and threading the cache through the inner Picard iteration.

What gets cached
----------------
* ``P_retain_marg[i1, j1, i2, j2; v_esc]`` — the orientation-marginalised
  retention probability at every progenitor cell.  4-D shape
  ``(N_m, N_chi, N_m, N_chi)`` per ``v_esc``.  Drives the **loss side**
  scan-elimination in Session F1.
* ``chi_rem_mean[i1, j1, i2, j2]`` — the orientation-marginalised
  *mean* ``|χ_rem|`` over each progenitor cell.  Same 4-D shape; no
  ``v_esc`` dependence.  Shipped as a building block for Session F2's
  χ-out distribution surrogate; **not wired into the gain side in F1**.
  The Session F1 gain side continues to run the per-iteration scan in
  ``response/coagulation.py:_step_one_generation_marginalised`` — the
  mean alone is insufficient to drive the bilinear scatter (a single
  progenitor cell contributes to many χ-out cells under non-Aligned;
  see ``demo/PLAN_implicit_footprint_refactor.md`` § Strategy A
  limitations).

Aligned shortcut at the cache layer
-----------------------------------
:func:`build_orientation_cache` returns ``None`` when
``isinstance(orientation, Aligned)``.  Every downstream consumer
(``response/implicit.py``: :func:`loss_density`, :func:`_F_map`) checks
for ``cache is None`` and routes through the existing scalar Aligned
path bit-for-bit.  The cache is only ever populated under non-Aligned,
so the Aligned-default solver state pays no extra construction cost and
its arithmetic is unchanged (preserves
``test_aligned_default_equals_explicit_aligned_bitwise``).

AD-on-κ via the cache
---------------------
The cache is built inside the JAX trace of
:func:`response.implicit._picard_while_loop` (forward) and inside the
``vjp_theta_fn`` closure of :func:`response.implicit._make_picard_solve_fn`'s
backward pass.  Both call sites are jit-traced; the cache itself depends
on ``orientation.kappa`` through ``orientation.quadrature_3d()`` (which
is differentiable in ``κ`` for :class:`spin_orientation.Fisher`).  The
implicit-function-theorem VJP therefore flows ∂(N*)/∂κ naturally — no
extra plumbing.  This is the cross-cutting Decision (b) of
``demo/PLAN_implicit_footprint_refactor.md`` § Cross-cutting decisions
(separate ``OrientationCache`` pytree threaded through the solve,
**not** an ``eqx.field`` on the response).

Construction cost / memory budget
---------------------------------
At Stage 1 ``(N_m, N_chi) = (8, 6)``: ~64 KB per tensor.
At production ``(N_m, N_chi) = (64, 20)``: ~13 MB per tensor (×2 tensors
= 26 MB total), well inside the PLAN's "must fit in single-node RAM"
budget of ~1 GB.

The construction call evaluates ``P_retain_marginalised`` once at the
4-D-broadcast progenitor mesh — same arithmetic, same orientation scan
that ``loss_density`` already invokes per Picard iteration.  Net effect
in F1: ``n_iter ≈ 20`` orientation scans per solve → 1 orientation scan
per solve on the loss side.  Gain side untouched (F2).

Cross-references
----------------
* PLAN: ``demo/PLAN_implicit_footprint_refactor.md`` § Session F1.
* Roadmap: ``demo/PLAN_refactor_roadmap.md`` — F1 is the
  largest-single-speedup item in the implicit refactor stream.
* Memories: ``[[project-session-handoff-20260527-m1]]``,
  ``[[project-orientation-quadrature-defaults-flip]]``,
  ``[[project-misalignment-session-c-done]]``,
  ``[[feedback-no-fake-physics-stand-ins]]``,
  ``[[feedback-no-fudge-factors]]``,
  ``[[feedback-no-frozen-test-references]]``.
"""

from __future__ import annotations

from typing import Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .grids import BinaryGrid
from .remnants.marginal import (
    P_retain_marginalised,
    chi_rem_at_orientation,
)
from .spin_orientation import Aligned


class OrientationCache(eqx.Module):
    """Precomputed orientation-marginalised tensors.

    Two 4-D ``(N_m, N_chi, N_m, N_chi)`` tensors over the progenitor
    mesh of the engine grid.  Each is built by a *single* scan over the
    orientation quadrature (no per-iteration cost inside the Picard
    loop).

    Attributes
    ----------
    P_retain_marg : Array
        ``⟨ retention(v_kick(m1, m2, |χ1|, |χ2|, θ̂); v_esc) ⟩_{P(θ̂)}``
        evaluated at every progenitor cell of the engine grid, for a
        single scalar ``v_esc``.  Replaces the per-Picard-iteration call
        to :func:`P_retain_marginalised` on the **loss side**.
    chi_rem_mean : Array
        ``⟨ |χ_rem(m1, m2, |χ1|, |χ2|, θ̂)| ⟩_{P(θ̂)}`` on the same
        progenitor mesh.  Shipped for Session F2; F1 does **not** wire
        this into the gain side (the bilinear scatter needs the full
        χ-out distribution, not just the mean — see PLAN § Strategy A
        limitations).

    JAX-pytree
    ----------
    Both tensors are pytree leaves (``eqx.Module`` defaults).  The
    container is itself a pytree; threading it through the Picard loop
    keeps reverse-mode AD on ``orientation.kappa`` (Fisher) flowing
    naturally — the cache build is part of the trace, so ``vjp_theta_fn``
    sees κ through the cache tensors as well as through any direct
    response surface.
    """

    P_retain_marg: Array  # (N_m, N_chi, N_m, N_chi)
    chi_rem_mean: Array  # (N_m, N_chi, N_m, N_chi)


def build_orientation_cache(
    grid: BinaryGrid,
    orientation,
    v_esc: Array,
    *,
    retention_kind: Literal["heaviside", "sigmoid"] = "sigmoid",
    w_over_v_esc: float = 1.0,
) -> Optional[OrientationCache]:
    """Build the per-solve orientation-marginalised tensor cache.

    Returns ``None`` under :class:`Aligned`.  Every downstream consumer
    checks for ``cache is None`` and routes to the existing scalar
    Aligned path — see :func:`response.implicit.loss_density`.  This
    keeps the Aligned dispatch bit-stable per PLAN Decision 2 (the
    Stage-1 cascade-pathology study depends on it being literally the
    same arithmetic as the v1 path).

    Under any non-Aligned orientation, calls into the existing
    :func:`P_retain_marginalised` and :func:`chi_rem_at_orientation`
    primitives at the broadcast 4-D progenitor mesh.  No new physics
    here — just eager evaluation + container packaging of the same
    quadrature that the loss / gain hot loops already invoke per
    Picard iteration.

    Parameters
    ----------
    grid : BinaryGrid
        The engine grid.  Defines ``N_m, N_chi`` and the progenitor
        mesh axes the precompute broadcasts over.
    orientation : OrientationModel
        :class:`spin_orientation.Aligned` (returns ``None``),
        :class:`Isotropic`, or :class:`Fisher`.
    v_esc : Array
        Host escape velocity (km/s); scalar.  Feeds the retention
        marginal.  ``chi_rem_mean`` is ``v_esc``-independent and is
        built once with no v_esc dependence.
    retention_kind : {"heaviside", "sigmoid"}, default "sigmoid"
        Per-orientation retention model under non-Aligned.  Forwarded
        verbatim to :func:`P_retain_marginalised`; same convention as
        the rest of the misalignment-mode dispatch surface.
    w_over_v_esc : float, default 1.0
        Sigmoid sharpness as a fraction of ``v_esc`` (only used when
        ``retention_kind="sigmoid"``).

    Returns
    -------
    OrientationCache or None
        ``None`` under Aligned; populated cache otherwise.

    Notes
    -----
    JIT-traceable end-to-end.  Use cases call it from inside a jitted
    solver entry point so the cache is materialised once per trace,
    not once per Python invocation.
    """
    if isinstance(orientation, Aligned):
        return None

    m = grid.m
    chi = grid.chi
    # 4-D broadcast over the progenitor mesh, same shape contract as
    # ``response.implicit.loss_density``'s internal ``(m1, chi1, m2, chi2)``.
    m1 = m[:, None, None, None]
    chi1 = chi[None, :, None, None]
    m2 = m[None, None, :, None]
    chi2 = chi[None, None, None, :]

    # Loss-side marginal.  P_retain_marginalised already scans the
    # orientation quadrature internally with a Decision-3 memory contract
    # (one full-shape progenitor tensor alive per scan iteration).  Calling
    # it once with the broadcast 4-D inputs yields the full
    # (N_m, N_chi, N_m, N_chi) tensor in one shot.
    P_retain_marg = P_retain_marginalised(
        m1, m2, chi1, chi2, v_esc, orientation,
        retention_kind=retention_kind,
        w_over_v_esc=w_over_v_esc,
    )

    # Gain-side mean (shipped for F2; not used by F1 callers).  Mirrors
    # the Decision-3 scan pattern: accumulate one weighted contribution
    # per orientation quadrature point into a 4-D progenitor mesh
    # accumulator.  No vmap-over-quadrature — would blow peak memory
    # at production resolution.
    chi_rem_mean = _chi_rem_mean_4d(m1, m2, chi1, chi2, orientation)

    return OrientationCache(
        P_retain_marg=P_retain_marg,
        chi_rem_mean=chi_rem_mean,
    )


def _chi_rem_mean_4d(
    m1: Array,
    m2: Array,
    chi1: Array,
    chi2: Array,
    orientation,
) -> Array:
    """Orientation-marginalised mean ``|χ_rem|`` over a 4-D progenitor mesh.

    ``⟨ |χ_rem(m1, m2, |χ1|, |χ2|, θ̂_1, θ̂_2)| ⟩_{P(θ̂)}`` via a
    ``jax.lax.scan`` over ``orientation.quadrature_3d()``.  One
    full-shape progenitor tensor alive per scan iteration (Decision-3
    memory contract; same pattern as
    :func:`remnants.marginal.P_retain_marginalised` and
    :func:`response.coagulation._step_one_generation_marginalised`).

    No ``v_esc`` dependence — the retention factor lives in the loss
    marginal, not here.  ``chi_rem_at_orientation`` canonicalises
    ``m1 ≥ m2`` internally via ``chi_rem_vec`` (the
    ``[[project-chi-rem-asymmetry-finding]]`` fix), so this call site
    does not need to reorder progenitors.
    """
    quad = orientation.quadrature_3d()  # (n_quad, 4)
    bcast_shape = jnp.broadcast_shapes(
        jnp.shape(m1),
        jnp.shape(m2),
        jnp.shape(chi1),
        jnp.shape(chi2),
    )

    def scan_body(acc: Array, orient_pt: Array) -> tuple[Array, None]:
        cos_t1 = orient_pt[0]
        cos_t2 = orient_pt[1]
        dphi = orient_pt[2]
        w_orient = orient_pt[3]
        x = chi_rem_at_orientation(
            m1, m2, chi1, chi2, cos_t1, cos_t2, dphi
        )
        return acc + w_orient * x, None

    init = jnp.zeros(bcast_shape)
    out, _ = jax.lax.scan(scan_body, init, quad)
    return out


__all__ = [
    "OrientationCache",
    "build_orientation_cache",
]
