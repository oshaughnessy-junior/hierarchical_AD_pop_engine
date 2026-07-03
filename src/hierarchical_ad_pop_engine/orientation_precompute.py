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
    v_kick_at_orientation,
)
from .remnants.mass import M_rem
from .spin_orientation import Aligned


# Session F2 — basis-rank sentinel.
# ``R_CHI_OUT_DEFAULT = None`` means "use the engine χ-grid bilinear-hat
# basis with R = N_χ".  This basis is **mass-conserving by construction**
# (bilinear hats partition unity at every point in [chi[0], chi[-1]]),
# which matches what the original scan path's bilinear χ-scatter does.
#
# The PLAN's original Strategy-B intent was to spectrally compress
# (R ≪ N_χ via shifted Legendre), but the projection-reconstruction
# loop on a coarse engine χ grid loses mass-conservation for r ≥ 2
# because ``Σ_j P_r^*(χ[j]) · dχ[j] ≠ L·δ_{r,0}`` at low N_χ
# (Σ_j P_2^* · dχ ≈ 0.24 vs the exact 0 at N_χ=6).  The fix is
# either an internal fine χ-grid for orthogonality or an explicit
# mass-conservation correction — both are deferred to a follow-up.
#
# Memory cost at production (N_χ=32 with R=N_χ): same as the
# scan-path 4-D tensor for the χ-out side; the cache cost stays
# linear in N_χ, not constant in R.  Acceptable per the PLAN budget.
#
# Truthy default ``True`` tells :func:`build_orientation_cache` to
# build the F2 surrogate; the actual basis rank is fixed at
# ``N_χ`` (bilinear hats — see :class:`OrientationCache` field
# docstring).  Set to ``None``/``0`` to skip F2 and fall back to
# the F1-only path (gain side per-iter scan).
#
# HIGH-N_χ MEMORY WARNING: the F2 surrogate + its basis contraction scale
# ~``N_χ⁴`` and blow up fast — a single ~154 GB allocation at N_m·N_χ=24×64.
# For isotropic forward runs that need N_χ = 64–128 (e.g. the fractal dN/dχ
# in ``demo/forward_modeling_observables.py``), set ``R_CHI_OUT_DEFAULT =
# None`` to use the memory-lean per-Picard-iteration orientation scan (F1
# only; documented bit-equivalent) — the solve then stays ~4–6 GB and fits
# a 24 GB GPU.  That demo does exactly this when ``COAG_LEAN_GAIN`` is set;
# see its module docstring "efficient high-resolution settings" for the full
# recipe.  Leave ``True`` for the low-N_χ inference solves (F2 is faster there).
R_CHI_OUT_DEFAULT = True


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
    # ── Session F2 fields ────────────────────────────────────────────
    # Strategy B χ-out spectral surrogate, **bilinear-hat basis** at
    # R = N_χ (mass-conserving by construction; partition of unity at
    # every χ_out point).  Populated only when the caller passes
    # ``R_chi_out`` to :func:`build_orientation_cache`.  ``None`` (the
    # default sentinel) means "F2 fast path not built; gain side falls
    # back to per-iter scan in _step_one_generation_marginalised".
    #
    # The PLAN originally wanted a smaller-R Legendre spectral
    # compression (R ≪ N_χ), but the projection-reconstruction loop
    # loses mass conservation on coarse engine χ grids (the
    # bilinear-grid weights ``Σ_j P_r^*(χ[j]) · dχ[j]`` aren't
    # δ_{r,0} L for r ≥ 2 at N_χ ~ 6).  Bilinear hats keep
    # mass-conservation trivially; spectral compression is deferred.
    #
    # ``chi_scatter_marg[j_out, i1, j1, i2, j2]`` — orientation-summed
    # bilinear-hat coefficient at engine χ-cell ``j_out``::
    #
    #     chi_scatter_marg[j_out, ...]
    #         = Σ_k w_orient_k · P_ret_k
    #               · hat_at_j_out(chi_rem_k(i1, j1, i2, j2))
    #
    # where ``hat_at_j(x) = max(0, 1 − |x − χ[j]| / Δχ[j])``.  Mass-
    # conservation: ``Σ_{j_out} chi_scatter_marg[j_out, ...] =
    # P_retain_marg[...]`` by partition-of-unity of the bilinear hats.
    # Reduces to bit-equivalent of the scan path's bilinear-χ-scatter,
    # but precomputed once per solve rather than per Picard iter.
    #
    # ``M_rem_mean_retained[i1, j1, i2, j2]`` — P_ret-weighted
    # orientation-mean remnant mass.  Used by the F2 fast gain path
    # to anchor the bilinear-m scatter onto a single m_out value per
    # progenitor cell (the Strategy B assumption that M_rem's
    # orientation variation is dominated by its mean at production
    # m-resolution — see PLAN § Strategy A limitations and the
    # M_rem-derivation note in the F2 fast-gain docstring).
    chi_scatter_marg: Optional[Array] = eqx.field(default=None)
    M_rem_mean_retained: Optional[Array] = eqx.field(default=None)


def build_orientation_cache(
    grid: BinaryGrid,
    orientation,
    v_esc: Array,
    *,
    retention_kind: Literal["heaviside", "sigmoid"] = "sigmoid",
    w_over_v_esc: float = 1.0,
    R_chi_out: Optional[int] = None,
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

    # ── Session F2 ──────────────────────────────────────────────────
    # When the caller asks to build the F2 surrogate (truthy
    # ``R_chi_out``), build the bilinear-hat χ-scatter marginal +
    # P_ret-weighted M_rem mean in one combined scan over the
    # orientation quadrature.  ``R_chi_out`` value beyond truthiness
    # is currently ignored — the basis rank is fixed at ``N_χ`` by
    # the partition-of-unity mass-conservation requirement (see the
    # ``OrientationCache`` docstring section on the basis choice).
    # Skipping F2 keeps the F1 cache cost unchanged for callers that
    # haven't migrated.
    chi_scatter_marg = None
    M_rem_mean_retained = None
    if R_chi_out:
        chi_scatter_marg, M_rem_mean_retained = _f2_combined_scan(
            m1, m2, chi1, chi2, grid,
            v_esc=v_esc,
            orientation=orientation,
            retention_kind=retention_kind,
            w_over_v_esc=w_over_v_esc,
            P_retain_marg=P_retain_marg,
        )

    return OrientationCache(
        P_retain_marg=P_retain_marg,
        chi_rem_mean=chi_rem_mean,
        chi_scatter_marg=chi_scatter_marg,
        M_rem_mean_retained=M_rem_mean_retained,
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


# ---------------------------------------------------------------------------
# Session F2 — bilinear-hat χ-out surrogate (Strategy B, mass-conserving).
#
# Replaces the per-Picard-iter orientation scan in
# ``response/coagulation.py:_step_one_generation_marginalised`` (the
# gain side, in F1's complement to the loss side).  The orientation
# marginal of the χ-out distribution is **bilinear-hat-scattered onto
# the engine χ grid** at cache build time, then contracted per Picard
# iter against ``K · N · N · cell`` without re-scanning orientation.
#
# Mass-conservation: bilinear hats are a partition of unity at every
# χ_out, so ``Σ_{j_out} hat_at_j_out(x) = 1`` for any ``x ∈
# [grid.chi[0], grid.chi[-1]]``.  The orientation-summed coefficient
# tensor therefore obeys ``Σ_{j_out} chi_scatter_marg[j_out, …] =
# P_retain_marg[…]`` — bit-equivalent to the scan path on the χ side.
#
# This trades the PLAN's original Legendre-spectral compression for
# correctness: a low-R shifted-Legendre projection on the coarse
# engine χ grid does **not** discretely integrate to ``L · δ_{r,0}``
# for r ≥ 2 (e.g. ``Σ_j P_2^*(χ[j]) · dχ[j] ≈ 0.24`` at N_χ = 6 vs
# the exact 0), so the reconstruct-at-grid loop spills mass into the
# r=0 mode and the forward solve diverges by ~16 % rel L1 at Stage 1.
# Spectral compression with proper internal-grid orthogonality is a
# follow-up; for F2 we ship correct-and-simple.
# ---------------------------------------------------------------------------
def _bilinear_chi_hats(x: Array, chi_grid: Array) -> Array:
    r"""Bilinear-hat basis values at ``x`` for each engine χ-grid node.

    Returns ``hats[j, *x.shape]`` where ``hats[j, …]`` is the value
    of the bilinear hat centred at ``chi_grid[j]`` evaluated at
    ``x``.  Mass-conservation invariant::

        Σ_j hats[j, …] = 1   (for x inside [chi_grid[0], chi_grid[-1]])
                       = 0   (outside)

    Implementation: compute fractional grid index ``i_frac`` via
    ``jnp.interp``, then form ``(1 − di)`` at index ``i_lo`` and
    ``di`` at ``i_lo + 1``, scatter to length-``N_χ`` per node.
    """
    N = chi_grid.size
    x_arr = jnp.asarray(x)
    i_frac = jnp.interp(
        x_arr, chi_grid, jnp.arange(N, dtype=x_arr.dtype),
    )
    inside = (x_arr >= chi_grid[0]) & (x_arr <= chi_grid[-1])
    i_lo = jnp.clip(jnp.floor(i_frac).astype(jnp.int32), 0, N - 2)
    di = i_frac - i_lo

    # Build the (N, *x.shape) one-hot-like tensor by scattering
    # (1 - di) and di onto the appropriate j index.  We use
    # ``jnp.where`` over a comparison broadcast rather than ``.at``
    # because the latter doesn't vmap cleanly over the j-axis here
    # and the comparison broadcast is constant-time per j per query.
    j_axis = jnp.arange(N).reshape((N,) + (1,) * x_arr.ndim)
    lo_mask = (j_axis == i_lo)
    hi_mask = (j_axis == (i_lo + 1))
    hats = jnp.where(lo_mask, 1.0 - di, 0.0) + jnp.where(hi_mask, di, 0.0)
    # Zero out everything outside the χ grid.
    return jnp.where(inside, hats, 0.0)


def _f2_combined_scan(
    m1: Array,
    m2: Array,
    chi1: Array,
    chi2: Array,
    grid: BinaryGrid,
    *,
    v_esc: Array,
    orientation,
    retention_kind: Literal["heaviside", "sigmoid"],
    w_over_v_esc: float,
    P_retain_marg: Optional[Array],
) -> tuple[Array, Optional[Array]]:
    r"""Combined orientation scan: bilinear-hat χ scatter + P_ret-weighted M_rem.

    One pass over the orientation quadrature accumulates **two**
    tensors over the 4-D progenitor mesh:

    * ``chi_scatter_marg[j_out, i1, j1, i2, j2]`` —
      ``Σ_k w_k · P_ret_k · hat_at_j_out(chi_rem_k)`` (shape
      ``(N_χ, N_m, N_χ, N_m, N_χ)``).  Mass-conservation: summed
      over ``j_out`` this equals ``P_retain_marg`` by partition-of-
      unity of the hats.
    * ``M_rem_weighted_accum[i1, j1, i2, j2]`` —
      ``Σ_k w_k · P_ret_k · M_rem_k``.  Divided by ``P_retain_marg``
      at the caller to give the P_ret-weighted orientation-mean
      remnant mass.  ``None`` if ``P_retain_marg is None`` (the
      caller is in the standalone helper-function path).

    Both products share the same per-orientation ``P_ret`` and the
    same ``chi_rem`` evaluation, so combining them is half the cost
    of two separate scans.

    Returns
    -------
    (chi_scatter_marg, M_rem_mean_retained) : tuple
    """
    bcast_shape = jnp.broadcast_shapes(
        jnp.shape(m1), jnp.shape(m2),
        jnp.shape(chi1), jnp.shape(chi2),
    )
    quad = orientation.quadrature_3d()  # (n_quad, 4)
    v_esc_arr = jnp.asarray(v_esc)
    chi_grid = grid.chi
    N_chi = chi_grid.size

    if retention_kind not in ("heaviside", "sigmoid"):
        raise ValueError(
            f"Unknown retention_kind {retention_kind!r}; expected "
            "'heaviside' or 'sigmoid'."
        )

    def scan_body(carry, orient_pt):
        chi_acc, m_rem_acc = carry
        cos_t1 = orient_pt[0]
        cos_t2 = orient_pt[1]
        dphi = orient_pt[2]
        w_orient = orient_pt[3]

        v_k = v_kick_at_orientation(
            m1, m2, chi1, chi2, cos_t1, cos_t2, dphi,
        )
        if retention_kind == "heaviside":
            P_ret = jnp.where(v_k < v_esc_arr, 1.0, 0.0)
        else:
            w_v = w_over_v_esc * v_esc_arr
            P_ret = jax.nn.sigmoid((v_esc_arr - v_k) / w_v)

        chi_rem_k = chi_rem_at_orientation(
            m1, m2, chi1, chi2, cos_t1, cos_t2, dphi,
        )
        # M_rem under per-orientation χ_z projection — same convention
        # as ``_step_one_generation_marginalised``.
        chi1_z = chi1 * cos_t1
        chi2_z = chi2 * cos_t2
        M_rem_k = M_rem(m1, m2, chi1_z, chi2_z)

        w_eff = w_orient * P_ret  # 4-D
        hats = _bilinear_chi_hats(chi_rem_k, chi_grid)  # (N_χ, *bcast)
        chi_inc = hats * w_eff[None, ...]  # (N_χ, *bcast)
        m_rem_inc = w_eff * M_rem_k        # bcast

        return (chi_acc + chi_inc, m_rem_acc + m_rem_inc), None

    init_chi = jnp.zeros((N_chi,) + bcast_shape)
    init_m_rem = jnp.zeros(bcast_shape)
    (chi_scatter_marg, m_rem_accum), _ = jax.lax.scan(
        scan_body, (init_chi, init_m_rem), quad,
    )

    if P_retain_marg is not None:
        # Floor the divisor: progenitor cells with zero retention
        # contribute zero weight to the gain anyway; we keep
        # M_rem_mean finite for index-lookup well-definedness.
        m_rem_mean_retained = m_rem_accum / jnp.maximum(P_retain_marg, 1e-30)
    else:
        m_rem_mean_retained = None

    return chi_scatter_marg, m_rem_mean_retained


__all__ = [
    "OrientationCache",
    "build_orientation_cache",
    "R_CHI_OUT_DEFAULT",
]
