"""Implicit Picard fixed-point solver for the coagulation response.

The current perturbative ``CoagulationResponse._solve`` truncates the
Volterra series at ``k_max`` and omits the loss term entirely. Empirically
that path matches the gold-standard implicit solver only at unphysically
small ε ≈ 0.025. The production-physics regime is ε ~ O(1), where 2G/3G
peaks are observable in real GW catalogues.

This module solves the steady-state coagulation equation::

    N(m, χ) = N_natal(m, χ) + ε · (gain[N, N](m, χ) - loss[N](m, χ))

by damped Picard iteration. The bilinear gain re-uses the engine's
existing :func:`_step_one_generation` (with ``sym = 1/2`` for the
unordered-pair correction), divided by the output cell area to recover
a density; the loss term is the part the perturbative accumulator
omits.

ε has absolute physical meaning — it is cluster evolution time × kernel
rate constant. ε ≪ 1 leaves the natal pool mostly intact; ε ≈ 1 sets
the cascade to a fully-developed steady state.

The Picard iteration is implemented as a :func:`jax.lax.while_loop`, so
the whole solver is ``jax.jit``-traceable end-to-end. Step 3 of
``demo/PLAN_engine_v2_production.md`` will register a custom VJP at the
fixed point so AD doesn't unroll the iterations. Until that lands, raw
``solver="implicit"`` is correct for forward evaluation and gives
approximately-correct gradients (the unrolled-loop gradient converges to
the implicit-function-theorem gradient as iteration count rises, but is
brittle when the loop hasn't fully converged).

The cell-area Jacobian fix (``project_engine_cell_area_fix.md``) is
applied in :func:`gain_density`: ``_step_one_generation`` returns
count-per-output-cell, and we divide by ``dm_out · dχ_out`` to recover a
density on the same measure scale as ``N_natal`` and :func:`loss_density`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..kernels import log_pairing_kernel, log_pairing_kernel_grid
from ..retention import P_retain
from .coagulation import _step_one_generation

if TYPE_CHECKING:
    from .coagulation import CoagulationResponse


# ---------------------------------------------------------------------------
# Trapezoid weights on the engine grids.
#
# NOTE: these are *symmetric trapezoidal* weights (half at both endpoints),
# used for integrals ``∫ f dm dχ``. The cell-area divisor inside
# :func:`gain_density` uses the same weights, which is what
# ``red_team/engine_v2_implicit.gain_density`` does. The engine's
# perturbative ``_solve`` uses asymmetric weights (full at endpoints) for
# its own cell-area divisor to match ``_step_one_generation``'s progenitor
# measure exactly. Boundary-cell populations are ~0 in every Stage 1
# scenario tested so the discrepancy never surfaces; if it does later,
# unify on the asymmetric convention (matches ``_step_one_generation``'s
# cell measure on the progenitor side).
# ---------------------------------------------------------------------------
def _dm_log_weights(grid_m: Array, grid_log_m: Array) -> Array:
    """``dm`` weights on a geomspace grid (``dm = m · d log m``, trapezoid)."""
    dlog_m = jnp.diff(grid_log_m)
    dlog_m_full = jnp.concatenate(
        [dlog_m[:1] * 0.5, 0.5 * (dlog_m[:-1] + dlog_m[1:]), dlog_m[-1:] * 0.5]
    )
    return grid_m * dlog_m_full


def _dchi_weights(grid_chi: Array) -> Array:
    """``dχ`` weights on a linear grid (trapezoid)."""
    d = jnp.diff(grid_chi)
    return jnp.concatenate([d[:1] * 0.5, 0.5 * (d[:-1] + d[1:]), d[-1:] * 0.5])


# ---------------------------------------------------------------------------
# Loss density — the part the perturbative accumulator omits.
# ---------------------------------------------------------------------------
def loss_density(
    N: Array,
    response: "CoagulationResponse",
    v_esc: Array,
) -> Array:
    """``N(m, χ) · ⟨K · N · P_retain⟩(m, χ)`` on the response's grid.

    Memory footprint: builds a ``(N_m, N_chi, N_m, N_chi)`` 4-D array.
    At (96, 32) and float64 that is ~75 MB — comfortable.
    """
    grid = response.grid
    m = grid.m
    chi = grid.chi
    # 4-D broadcast over (m, χ, m', χ').
    m1 = m[:, None, None, None]
    chi1 = chi[None, :, None, None]
    m2 = m[None, None, :, None]
    chi2 = chi[None, None, None, :]

    log_K = log_pairing_kernel(m1, m2, **response.kernel_params)  # (N_m, 1, N_m, 1)
    P = P_retain(m1, m2, chi1, chi2, v_esc)  # (N_m, N_chi, N_m, N_chi)
    K_P = jnp.exp(log_K) * P  # broadcasts on j

    integrand = K_P * N[None, None, :, :]  # multiply by N(m', χ')

    dm_w = _dm_log_weights(grid.m, grid.log_m)  # (N_m,)
    dchi_w = _dchi_weights(grid.chi)  # (N_chi,)

    avg = jnp.einsum("ijkl,k,l->ij", integrand, dm_w, dchi_w)
    return N * avg


# ---------------------------------------------------------------------------
# Gain density — the engine's existing _step_one_generation with sym = 1/2,
# divided by the output cell area to recover a density at (m_out, χ_out).
#
# Why the division: ``_step_one_generation`` returns the bilinear-scattered
# count per output cell (it already absorbed the four progenitor measure
# factors ``dm₁ dχ₁ dm₂ dχ₂`` via its internal ``cell`` term). To get a
# density at (m_out, χ_out) we divide by ``dm_out · dχ_out``. On a
# geomspace m grid this Jacobian varies ~20× across the grid, so no
# constant rescaling can absorb it.
# ---------------------------------------------------------------------------
def gain_density(
    N: Array,
    log_K_mm: Array,
    response: "CoagulationResponse",
    v_esc: Array,
) -> Array:
    """Density-form gain term ``(1/2) ∫∫ K · N · N · P_retain`` on the grid."""
    log_N_safe = jnp.log(jnp.maximum(N, 1e-30))
    raw_weight_per_cell = _step_one_generation(
        log_N_safe,
        log_N_safe,
        log_K_mm,
        response.grid,
        v_esc,
        0.5,  # (1/2) ∫∫ K N N P_retain — unordered-pair correction
    )
    grid = response.grid
    dm_out = _dm_log_weights(grid.m, grid.log_m)
    dchi_out = _dchi_weights(grid.chi)
    return raw_weight_per_cell / (dm_out[:, None] * dchi_out[None, :])


# ---------------------------------------------------------------------------
# Result container.
# ---------------------------------------------------------------------------
class ImplicitSolveResult(eqx.Module):
    """Result of :func:`solve_implicit_fixed_point`.

    All fields are JAX arrays (Step 2 wrote the Picard loop as
    ``jax.lax.while_loop`` so diagnostics are now produced traceably).
    Cast scalars to Python with ``int(...)``, ``float(...)`` if you
    need them eagerly — e.g. for assertions or logging outside a jit
    boundary.

    Attributes
    ----------
    N : Array
        Converged single-BH density on the (m, χ) grid, ``(N_m, N_chi)``.
    log_N : Array
        ``log(N)`` with a numerical floor, same shape — what
        :meth:`CoagulationResponse._log_total_density` returns.
    epsilon : Array
        Coupling × time used (echoed for downstream diagnostics).
    n_iter : Array
        Number of Picard iterations actually run. JAX int32 scalar.
    final_diff : Array
        ``‖ΔN‖∞`` at the final iteration. Should be ``< tol`` on success.
    loss_integral : Array
        ``∫ L[N*](m, χ) dm dχ`` at the fixed point.
    natal_integral : Array
        ``∫ N_natal(m, χ) dm dχ``.
    f_natal_depleted : Array
        ``ε · loss_integral / natal_integral``, clipped to ``[0, 1]``.
        This is the principled mapping between ε and the closed-system
        particle-MC stopping fraction (see
        ``project_engine_implicit_refactor.md`` for the derivation).

    Notes
    -----
    The per-iteration ``∫ N dm dχ`` history is no longer recorded —
    ``jax.lax.while_loop`` cannot accumulate variable-length traces.
    If you want a trace for diagnostics, call
    :func:`solve_implicit_fixed_point_with_history` (uses
    ``jax.lax.scan`` with a fixed iteration count). The plain solver is
    the one to use inside inference; it does fewer FLOPs per call and
    its output pytree is smaller (faster ``jax.jit`` re-tracing).
    """

    N: Array
    log_N: Array
    epsilon: Array
    n_iter: Array
    final_diff: Array
    loss_integral: Array
    natal_integral: Array
    f_natal_depleted: Array


# ---------------------------------------------------------------------------
# Solver.
# ---------------------------------------------------------------------------
def solve_implicit_fixed_point(
    response: "CoagulationResponse",
    v_esc: Array,
    *,
    epsilon: Array | float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-7,
    damping: float = 0.5,
) -> ImplicitSolveResult:
    """Damped Picard fixed-point for ``N = N_natal + ε · (gain(N) - loss(N))``.

    Parameters
    ----------
    response : CoagulationResponse
        The response whose ``grid``, ``kernel_params`` and natal
        densities define the implicit equation. The ``solver`` and
        ``epsilon`` fields on ``response`` are ignored — pass ``epsilon``
        explicitly as a keyword.
    v_esc : Array
        Per-environment escape velocity. Scalar today; vmap externally
        for multi-environment.
    epsilon : Array | float, default 1.0
        Coupling × time. Absolute physical meaning, not a tuning knob.
    max_iter, tol, damping : int, float, float
        Picard hyperparameters. Damping = 0.5 is conservative; 0.4 helps
        at high ε; >0.7 oscillates. Tol on ``‖ΔN‖∞``.

    Returns
    -------
    ImplicitSolveResult
        Converged fixed point plus diagnostics. See class docstring.

    Notes
    -----
    The Picard loop is a :func:`jax.lax.while_loop` so the whole solver
    is ``jax.jit``-traceable end-to-end. Step 3 of
    ``demo/PLAN_engine_v2_production.md`` will register a custom VJP at
    the fixed point so AD won't unroll the iterations; until then,
    gradients through this function are mathematically correct only as
    long as the loop converges (a non-converged loop produces noisy
    gradients that scale with iteration count).

    ``max_iter`` is taken statically — it is the upper bound on the
    ``while_loop`` step count, not a runtime knob. Pass it as a Python
    ``int``; the loop terminates early on ``final_diff < tol``.
    """
    log_K_mm = log_pairing_kernel_grid(response.grid.m, **response.kernel_params)
    log_N_natal = response._natal_log_density()
    N_natal = jnp.exp(log_N_natal)

    grid = response.grid
    dm_w = _dm_log_weights(grid.m, grid.log_m)
    dchi_w = _dchi_weights(grid.chi)

    epsilon_arr = jnp.asarray(epsilon)
    v_esc_arr = jnp.asarray(v_esc)
    tol_arr = jnp.asarray(tol, dtype=N_natal.dtype)
    max_iter_arr = jnp.asarray(max_iter, dtype=jnp.int32)

    # Carry: (N, n_iter, final_diff). Initial final_diff = +inf ensures
    # ``cond_fun`` is True at entry so the body runs at least once even
    # in the degenerate ε = 0 case (one iter, then exits with diff = 0).
    init_carry = (
        N_natal,
        jnp.zeros((), dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=N_natal.dtype),
    )

    def cond_fun(carry):
        _, n_iter, final_diff = carry
        return (final_diff > tol_arr) & (n_iter < max_iter_arr)

    def body_fun(carry):
        N, n_iter, _ = carry
        gain = gain_density(N, log_K_mm, response, v_esc_arr)
        loss = loss_density(N, response, v_esc_arr)
        N_rhs = jnp.maximum(N_natal + epsilon_arr * (gain - loss), 0.0)
        N_new = (1.0 - damping) * N + damping * N_rhs
        diff = jnp.max(jnp.abs(N_new - N))
        return (N_new, n_iter + 1, diff)

    N, n_iter, final_diff = jax.lax.while_loop(cond_fun, body_fun, init_carry)

    # Final loss evaluation at the converged N* — needed for f_natal_depleted.
    loss_final = loss_density(N, response, v_esc_arr)
    loss_integral = jnp.einsum("ij,i,j->", loss_final, dm_w, dchi_w)
    natal_integral = jnp.einsum("ij,i,j->", N_natal, dm_w, dchi_w)
    f_natal_depleted = jnp.clip(
        epsilon_arr * loss_integral / jnp.maximum(natal_integral, 1e-30),
        0.0,
        1.0,
    )

    return ImplicitSolveResult(
        N=N,
        log_N=jnp.log(jnp.maximum(N, 1e-30)),
        epsilon=epsilon_arr,
        n_iter=n_iter,
        final_diff=final_diff,
        loss_integral=loss_integral,
        natal_integral=natal_integral,
        f_natal_depleted=f_natal_depleted,
    )


__all__ = [
    "ImplicitSolveResult",
    "gain_density",
    "loss_density",
    "solve_implicit_fixed_point",
]
