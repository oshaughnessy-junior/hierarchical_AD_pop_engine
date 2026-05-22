r"""Hand-rolled "obviously correct" reference accumulator for the generation cascade.

Used by :mod:`test_redteam_Nx_parity`'s **j ≠ l double-count invariant
guard** — assertion (4) in
``demo/PLAN_redteam_observable_validation.md`` §Phase A.2.

What this module does
---------------------
Reimplements the engine's generation accumulator (the loop in
:meth:`hierarchical_ad_pop_engine.response.coagulation.CoagulationResponse._solve`)
as an *explicit nested loop over unordered* ``(j, l)`` progenitor
pairs.  Each unordered pair contributes exactly one
:func:`_step_one_generation` call.  There is no swap-and-add — by
inspection — so the j ≠ l double-count bug
(``project_engine_jl_doublecount_fix``) cannot occur here.

The per-pair primitive ``_step_one_generation`` is reused **unchanged**
from the engine.  The intent of this reference is to pin the
*pairing rule* (which pairs contribute, with what symmetry factor),
not to re-derive the bilinear-scatter physics.  Layered tests:

* The symmetry-fix invariant (Phase A.2 assertion 3) pins the
  primitive itself: ``_step_one_generation`` is symmetric under the
  joint ``(m1, χ1) ↔ (m2, χ2)`` swap of its first two arguments.
* This reference pins the loop *around* the primitive.

Together the two invariants exhaust the load-bearing surfaces of
``_solve``.

Why not also re-implement ``_step_one_generation``?  Two reasons:

1. The bilinear scatter is ~100 lines of axis bookkeeping; a
   parallel implementation drifts toward "test the implementation by
   re-implementing it", which is its own anti-pattern.
2. The primitive's correctness is already pinned by the symmetry
   invariant — the only nontrivial way for it to be wrong (chi_rem's
   implicit m1≥m2 convention) shows up as broken symmetry.

Pairing convention (matches the engine's ``_solve``)
----------------------------------------------------
``n_k`` includes contributions from progenitor pairs ``(j, l)`` with
``max(j, l) + 1 == k``.  In practice the engine uses ``l = k − 1``
(deepest progenitor) and ``j = 1, …, k − 1``, giving the unordered
set::

    k = 2:  {(1, 1)}
    k = 3:  {(1, 2), (2, 2)}
    k = 4:  {(1, 3), (2, 3), (3, 3)}
    k = K:  {(1, K-1), (2, K-1), …, (K-1, K-1)}

Each pair contributes one ``_step_one_generation`` call.  The
symmetry factor is ``0.5`` for ``j == l`` (self-pairs are counted with
half-weight because the unordered double-counts itself) and ``1.0``
for ``j != l`` (each ordering of the unordered pair is enumerated
exactly once by construction — there is no second call to add).
"""

from __future__ import annotations

from typing import List

import jax.numpy as jnp
from jax import Array

from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.kernels import log_pairing_kernel_grid
from hierarchical_ad_pop_engine.response.coagulation import _step_one_generation


# ---------------------------------------------------------------------------
# Cell-area weights — re-derived locally to match the engine's
# asymmetric trapezoidal convention used by ``_step_one_generation``
# (see comments in :func:`_solve` for why the perturbative path uses
# asymmetric, full-at-endpoints weights as opposed to the implicit
# path's symmetric ones).  These have to match exactly or the
# reference would be off by an unrelated cell-area artefact.
# ---------------------------------------------------------------------------
def _cell_area_perturbative(grid: BinaryGrid) -> Array:
    r"""``dm_out · dχ_out`` on ``grid``, asymmetric-trapezoid convention.

    This is the divisor that ``_solve`` applies to convert
    bilinear-scatter count-per-cell back into density-per-cell.
    Replicated here verbatim (not imported) so the reference does not
    depend on any private symbol of :mod:`response.coagulation`.
    """
    dlog_m = jnp.diff(grid.log_m)
    dlog_m_full = jnp.concatenate(
        [dlog_m[:1], 0.5 * (dlog_m[:-1] + dlog_m[1:]), dlog_m[-1:]]
    )
    dchi = jnp.diff(grid.chi)
    dchi_full = jnp.concatenate(
        [dchi[:1], 0.5 * (dchi[:-1] + dchi[1:]), dchi[-1:]]
    )
    dm_out = grid.m * dlog_m_full
    return dm_out[:, None] * dchi_full[None, :]


# ---------------------------------------------------------------------------
# Pair enumeration — one row per unordered progenitor pair contributing
# to ``n_k``.  By inspection: no swap-and-add, no symmetrisation
# afterward, every contribution is a *single* ``_step_one_generation``
# call with the documented symmetry factor.
# ---------------------------------------------------------------------------
def _unordered_pairs_for_generation(k: int) -> List[tuple[int, int, float]]:
    r"""Return ``[(j, l, sym), …]`` for generation ``k``.

    ``j <= l`` always (we enumerate the unordered pair canonically),
    and ``l == k − 1`` per the engine's "deepest progenitor"
    convention.  Symmetry factor ``sym`` is ``0.5`` for ``j == l``,
    ``1.0`` otherwise.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2 (1G is natal); got {k}")
    l = k - 1
    pairs: List[tuple[int, int, float]] = []
    for j in range(1, l + 1):  # j in 1..l, so j <= l
        sym = 0.5 if j == l else 1.0
        pairs.append((j, l, sym))
    return pairs


# ---------------------------------------------------------------------------
# Reference solver.
# ---------------------------------------------------------------------------
def reference_solve_log_n_k(
    log_n_natal: Array,
    grid: BinaryGrid,
    kernel_params: dict,
    v_esc: Array,
    k_max: int,
) -> Array:
    r"""Reference reimplementation of ``CoagulationResponse._solve``.

    Parameters
    ----------
    log_n_natal : Array, shape ``(N_m, N_chi)``
        1G (natal) log-density on the grid.  Same input the engine
        receives via :meth:`_natal_log_density`.
    grid : BinaryGrid
        Engine grid the densities live on.
    kernel_params : dict
        Pairing-kernel parameters.
    v_esc : Array
        Per-environment escape velocity, scalar.
    k_max : int
        Maximum generation tracked.

    Returns
    -------
    Array, shape ``(k_max, N_m, N_chi)``
        ``log n_k`` for ``k = 1, …, k_max`` — identical layout to
        :meth:`CoagulationResponse._solve`'s return value.

    Notes
    -----
    By construction this is a single nested loop over unordered
    ``(j, l)`` pairs with no swap-and-add.  Compare against the
    engine's ``_solve`` output as the double-count invariant.
    """
    log_K_mm = log_pairing_kernel_grid(grid.m, **kernel_params)
    cell_area = _cell_area_perturbative(grid)

    log_n: List[Array] = [log_n_natal]
    for k in range(2, k_max + 1):
        n_lin = jnp.zeros((grid.m.size, grid.chi.size))
        for j, l, sym in _unordered_pairs_for_generation(k):
            # One call per unordered pair.  No second call with
            # arguments swapped — that would be the j != l
            # double-count bug.
            contrib = _step_one_generation(
                log_n[j - 1], log_n[l - 1], log_K_mm, grid, v_esc, sym
            )
            n_lin = n_lin + contrib / cell_area
        log_n.append(jnp.log(jnp.maximum(n_lin, 1e-30)))
    return jnp.stack(log_n, axis=0)


__all__ = [
    "_unordered_pairs_for_generation",
    "reference_solve_log_n_k",
]
