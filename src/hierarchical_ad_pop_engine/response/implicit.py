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

The Picard iteration is implemented as a :func:`jax.lax.while_loop`
(Step 2) with an implicit-function-theorem VJP registered via a
hand-rolled ``@jax.custom_vjp`` factory (Step 3 of
``demo/PLAN_engine_v2_production.md``). Reverse-mode AD
(``jax.grad``, ``jax.vjp``) solves the transposed linearised system

    (I − ∂F/∂N)ᵀ · v = λ

via GMRES at ``N*``, then pulls back through ``∂F/∂θ``. Forward eval,
jit, and grad are all supported end-to-end. ``solver="implicit"`` is
now the right physics *and* the right gradient — safe for NUTS / SVI.

Note: ``jax.lax.custom_root`` was attempted first but raised
``NotImplementedError`` on ``jax.grad`` due to closure over an
``equinox.Module`` with mixed static/dynamic fields. The hand-rolled
``@jax.custom_vjp`` factory (:func:`_make_picard_solve_fn`) sidesteps
this by making ``response``, ``v_esc``, and ``epsilon`` explicit
positional arguments and fixing the hyperparameters (``max_iter``,
``tol``, ``damping``) via :func:`functools.lru_cache`.

The cell-area Jacobian fix (``project_engine_cell_area_fix.md``) is
applied in :func:`gain_density`: ``_step_one_generation`` returns
count-per-output-cell, and we divide by ``dm_out · dχ_out`` to recover a
density on the same measure scale as ``N_natal`` and :func:`loss_density`.
"""

from __future__ import annotations

import functools
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

    Forwards ``response.orientation`` / ``retention_kind`` /
    ``w_over_v_esc`` to :func:`P_retain` so the misalignment dispatch
    set at construction time fires through the production implicit
    solver.  ``Aligned`` (the default) routes to the bit-for-bit v1
    Heaviside; non-Aligned routes to the marginalised retention.
    Without this forward, the implicit solver silently ignored the
    orientation field — caught by
    ``tests/test_misalignment_invariants.py`` Step 11.
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
    P = P_retain(
        m1, m2, chi1, chi2, v_esc,
        orientation=response.orientation,
        retention_kind=response.retention_kind,
        w_over_v_esc=response.w_over_v_esc,
    )  # (N_m, N_chi, N_m, N_chi)
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
    """Density-form gain term ``(1/2) ∫∫ K · N · N · P_retain`` on the grid.

    Forwards ``response.orientation`` / ``retention_kind`` /
    ``w_over_v_esc`` to :func:`_step_one_generation` so the
    misalignment dispatch fires through the production implicit
    solver — same wiring as :func:`loss_density`.  Without this
    forward, the bilinear gain accumulator ignored the orientation
    field while the per-generation perturbative ``_solve`` honoured
    it, producing a silent divergence between the two solver paths.
    """
    log_N_safe = jnp.log(jnp.maximum(N, 1e-30))
    raw_weight_per_cell = _step_one_generation(
        log_N_safe,
        log_N_safe,
        log_K_mm,
        response.grid,
        v_esc,
        0.5,  # (1/2) ∫∫ K N N P_retain — unordered-pair correction
        orientation=response.orientation,
        retention_kind=response.retention_kind,
        w_over_v_esc=response.w_over_v_esc,
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
        Number of Picard iterations actually run. Cast to the float dtype
        of ``N`` (not int) so that the custom-VJP aux pytree can carry
        well-defined tangents under AD — JAX's custom-VJP machinery
        rejects integer tangents in any output, including pass-through
        aux. Read back with ``int(result.n_iter)`` at the eager boundary.
    final_diff : Array
        ``‖ΔN‖∞`` at the final iteration. Should be ``< tol`` on success.
    loss_integral : Array
        ``∫ L[N*](m, χ) dm dχ`` at the fixed point — diagnostic only.
        Note: the double integral ``∫∫ K P_retain N* N* dm dχ dm' dχ'``
        counts each merger pair **twice** (once as (i,j) and once as
        (j,i)), so this quantity is 2 × the physical merger rate.  Do
        not use it directly as a depletion fraction; use
        :attr:`f_natal_depleted` instead.
    natal_integral : Array
        ``∫ N_natal(m, χ) dm dχ``.
    f_natal_depleted : Array
        ``1 − ∫N*(m,χ) dm dχ / ∫N_natal(m,χ) dm dχ``, clipped to
        ``[0, 1]``.

        This is the fraction of the natal pool depleted by the cascade
        at the converged fixed point — the principled stopping-fraction
        for the closed-system particle-MC backstop.

        **Derivation.** Integrating the steady-state equation over
        ``(m, χ)`` and using ``∫G[N*] = (1/2)·loss_integral``
        (the gain counts each pair once; the loss double-integral counts
        it twice) gives::

            ∫N* = ∫N_natal − (ε/2)·loss_integral

        so ``f = 1 − ∫N*/∫N_natal`` exactly at the Picard fixed point.
        Using ``ε·loss_integral/∫N_natal`` directly would overestimate
        by a factor of 2.

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

    def evaluator(self, response: "CoagulationResponse"):
        r"""Return a :class:`RateDensityEvaluator` over this result.

        Thin constructor for
        :class:`hierarchical_ad_pop_engine.evaluation.ImplicitEvaluator`,
        co-located on the result type per
        ``demo/PLAN_arbitrary_point_evaluation.md`` § "Construction
        sites".  Callers can write::

            ev = result.evaluator(response)
            N_at_obs = ev.density(m_obs, chi_obs)
            R_at_obs = ev.rate_density_4d(m1, m2, chi1, chi2)

        without naming the evaluator class.  Swapping in the (future)
        basis evaluator at the same call site requires changing only
        the solver invocation, not the evaluator construction.

        Imported lazily to break the
        ``response.implicit ↔ evaluation.implicit`` reference cycle —
        same pattern as ``CoagulationResponse._log_total_density``'s
        local import of :func:`solve_implicit_fixed_point`.
        """
        from ..evaluation.implicit import ImplicitEvaluator
        return ImplicitEvaluator(result=self, response=response)


# ---------------------------------------------------------------------------
# Picard map F — explicit in all arguments for the custom VJP.
# ---------------------------------------------------------------------------
def _F_map(
    N: Array,
    response: "CoagulationResponse",
    v_esc_arr: Array,
    epsilon_arr: Array,
) -> Array:
    """Picard RHS: N_natal(θ) + ε·(gain[N, θ] − loss[N, θ]).

    No clipping at 0 here — at the converged fixed point N* ≥ 0 by
    construction (the forward while_loop clips), so the linearisation
    used by the IFT backward is evaluated where the clip is inactive
    and the gradient is correct.  Do **not** use this for the forward
    solve; use :func:`_picard_while_loop` instead.
    """
    log_K_mm = log_pairing_kernel_grid(response.grid.m, **response.kernel_params)
    log_N_natal = response._natal_log_density()
    N_natal = jnp.exp(log_N_natal)
    gain = gain_density(N, log_K_mm, response, v_esc_arr)
    loss = loss_density(N, response, v_esc_arr)
    return N_natal + epsilon_arr * (gain - loss)


# ---------------------------------------------------------------------------
# Raw Picard while_loop (no AD support; used inside the custom VJP forward).
# ---------------------------------------------------------------------------
def _picard_while_loop(
    response: "CoagulationResponse",
    v_esc_arr: Array,
    epsilon_arr: Array,
    max_iter: int,
    tol: float,
    damping: float,
) -> ImplicitSolveResult:
    """Damped-Picard iteration via :func:`jax.lax.while_loop`.

    JIT-traceable but not AD-safe in isolation.  Wrap via
    :func:`_make_picard_solve_fn` (which registers the IFT VJP) before
    calling inside ``jax.grad`` / ``jax.vjp``.
    """
    log_K_mm = log_pairing_kernel_grid(response.grid.m, **response.kernel_params)
    log_N_natal = response._natal_log_density()
    N_natal = jnp.exp(log_N_natal)

    grid = response.grid
    dm_w = _dm_log_weights(grid.m, grid.log_m)
    dchi_w = _dchi_weights(grid.chi)
    tol_arr = jnp.asarray(tol, dtype=N_natal.dtype)
    max_iter_arr = jnp.asarray(max_iter, dtype=jnp.int32)

    def F(N):
        g = gain_density(N, log_K_mm, response, v_esc_arr)
        l = loss_density(N, response, v_esc_arr)
        return jnp.maximum(N_natal + epsilon_arr * (g - l), 0.0)

    init_carry = (
        N_natal,
        jnp.zeros((), dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=N_natal.dtype),
    )

    def cond_fun(carry):
        _, n_iter, diff = carry
        return (diff > tol_arr) & (n_iter < max_iter_arr)

    def body_fun(carry):
        N, n_iter, _ = carry
        # damped Picard: N_new = (1 - d)·N + d·F(N) = N - d·(N - F(N))
        N_new = N - damping * (N - F(N))
        diff = jnp.max(jnp.abs(N_new - N))
        return (N_new, n_iter + 1, diff)

    N_star, n_iter_final, diff_final = jax.lax.while_loop(
        cond_fun, body_fun, init_carry
    )

    # Post-convergence diagnostics.
    loss_final = loss_density(N_star, response, v_esc_arr)
    loss_integral = jnp.einsum("ij,i,j->", loss_final, dm_w, dchi_w)
    natal_integral = jnp.einsum("ij,i,j->", N_natal, dm_w, dchi_w)
    n_star_integral = jnp.einsum("ij,i,j->", N_star, dm_w, dchi_w)
    # f = 1 - ∫N*/∫N_natal.  Derived by integrating the steady-state
    # equation: ∫G = (1/2)·loss_integral (each pair counted once in G,
    # twice in the double integral of L), so
    #   ∫N* = ∫N_natal - (ε/2)·loss_integral  →  f = 1 - ∫N*/∫N_natal.
    # Using ε·loss_integral/∫N_natal directly would overestimate by 2×.
    f_natal_depleted = jnp.clip(
        1.0 - n_star_integral / jnp.maximum(natal_integral, 1e-30),
        0.0,
        1.0,
    )

    return ImplicitSolveResult(
        N=N_star,
        log_N=jnp.log(jnp.maximum(N_star, 1e-30)),
        epsilon=epsilon_arr,
        n_iter=n_iter_final.astype(N_natal.dtype),
        final_diff=diff_final,
        loss_integral=loss_integral,
        natal_integral=natal_integral,
        f_natal_depleted=f_natal_depleted,
    )


# ---------------------------------------------------------------------------
# AD-safe solver factory — registers IFT VJP via @jax.custom_vjp.
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=16)
def _make_picard_solve_fn(max_iter: int, tol: float, damping: float):
    """Return an AD-safe Picard solver for fixed hyperparameters.

    The returned callable carries a ``@jax.custom_vjp`` that implements
    the implicit-function-theorem backward pass.

    Using :func:`functools.lru_cache` keyed on ``(max_iter, tol,
    damping)`` means that calls with the same hyperparameters return
    the **same Python function object**.  ``jax.jit`` caches compiled
    XLA kernels per function identity, so repeated calls with the same
    hyperparameters do not retrace.

    Why a factory instead of a single module-level
    ``@jax.custom_vjp``?  ``@jax.custom_vjp`` cannot take static
    (non-JAX-array) arguments without either tracing them as arrays
    (introducing unnecessary trace instability) or closing over them
    (which is exactly what this factory does).  The cache ensures we
    don't pay new-closure overhead on every ``solve_implicit_fixed_point``
    call.

    Only ``response``, ``v_esc``, and ``epsilon`` carry gradients.
    ``max_iter``, ``tol``, and ``damping`` are hyperparameters fixed at
    trace time and are *not* differentiated.
    """

    @jax.custom_vjp
    def _solve(
        response: "CoagulationResponse",
        v_esc_arr: Array,
        epsilon_arr: Array,
    ) -> ImplicitSolveResult:
        return _picard_while_loop(
            response, v_esc_arr, epsilon_arr, max_iter, tol, damping
        )

    def _solve_fwd(
        response: "CoagulationResponse",
        v_esc_arr: Array,
        epsilon_arr: Array,
    ):
        result = _solve(response, v_esc_arr, epsilon_arr)
        residuals = (result.N, response, v_esc_arr, epsilon_arr)
        return result, residuals

    def _solve_bwd(residuals, g: ImplicitSolveResult):
        """IFT VJP at the Picard fixed point.

        Given cotangent ``g`` on the :class:`ImplicitSolveResult`, return
        cotangents on ``(response, v_esc_arr, epsilon_arr)``.

        The implicit-function theorem for ``N* = F(N*, θ)``::

            dN*/dθ = (I − ∂F/∂N)⁻¹ · ∂F/∂θ

        In VJP (pull-back) form with incoming cotangent λ on N*:

        1. Solve  ``(I − J_N)ᵀ · v = λ``  via GMRES  (J_N = ∂F/∂N|_{N*}).
        2. ``cotangent_θ = vjp_θ(F(N*, ·))(v)``

        The GMRES operator is a matvec-only call — we never form
        ``(N_m·N_χ, N_m·N_χ)`` explicitly.  Each matvec is one VJP of
        :func:`_F_map` with respect to N (one gain + one loss evaluation).
        """
        N_star, response, v_esc_arr, epsilon_arr = residuals

        # ── Aggregate cotangent on N_star from result fields ────────────
        # N field: direct cotangent.
        # log_N field: chain rule log(max(N, 1e-30)) → dlog_N/dN = 1/N*.
        # epsilon field: echoed verbatim → contributes to cotangent_eps below.
        # natal_integral: ∫ N_natal dm dχ — no N* dependence, handled below.
        # n_iter, final_diff: no differentiable path; cotangent is effectively 0.
        # loss_integral, f_natal_depleted: depend on N* via loss_density; for
        #   now treated as zero (inference losses never differentiate through
        #   diagnostics — they use N / log_N exclusively).  TODO: add when needed.
        g_N_star = g.N + g.log_N / jnp.maximum(N_star, 1e-30)

        # ── GMRES: solve (I − J_N)ᵀ v = g_N_star ───────────────────────
        shape = N_star.shape

        def jac_N_T_v(v: Array) -> Array:
            """(∂F/∂N)ᵀ v  via VJP of _F_map wrt N at N_star."""
            _, vjp_fn = jax.vjp(
                lambda N: _F_map(N, response, v_esc_arr, epsilon_arr),
                N_star,
            )
            return vjp_fn(v)[0]

        def linear_op_T_flat(v_flat: Array) -> Array:
            v = v_flat.reshape(shape)
            return (v - jac_N_T_v(v)).reshape(-1)

        v_flat, _ = jax.scipy.sparse.linalg.gmres(
            linear_op_T_flat,
            g_N_star.reshape(-1),
            tol=1e-7,
            atol=0.0,
            restart=20,
            maxiter=200,
        )
        v = v_flat.reshape(shape)

        # ── Propagate: cotangent_θ = vjp_θ(F(N*, ·))(v) ────────────────
        _, vjp_theta_fn = jax.vjp(
            lambda resp, ve, eps: _F_map(N_star, resp, ve, eps),
            response,
            v_esc_arr,
            epsilon_arr,
        )
        cotangent_response, cotangent_v_esc, cotangent_eps = vjp_theta_fn(v)

        # epsilon field of ImplicitSolveResult is epsilon_arr echoed verbatim;
        # its cotangent feeds back directly into cotangent_eps.
        cotangent_eps = cotangent_eps + g.epsilon

        # natal_integral = ∫ N_natal dm dχ depends only on response, not N*.
        # Gradient bypasses the IFT and flows directly into cotangent_response.
        grid = response.grid
        dm_w = _dm_log_weights(grid.m, grid.log_m)
        dchi_w = _dchi_weights(grid.chi)
        _, vjp_natal_fn = jax.vjp(
            lambda resp: jnp.einsum(
                "ij,i,j->",
                jnp.exp(resp._natal_log_density()),
                dm_w,
                dchi_w,
            ),
            response,
        )
        ct_response_natal = vjp_natal_fn(g.natal_integral)[0]
        cotangent_response = jax.tree_util.tree_map(
            jnp.add, cotangent_response, ct_response_natal
        )

        return (cotangent_response, cotangent_v_esc, cotangent_eps)

    _solve.defvjp(_solve_fwd, _solve_bwd)
    return _solve


# ---------------------------------------------------------------------------
# Public solver.
# ---------------------------------------------------------------------------
def solve_implicit_fixed_point(
    response: "CoagulationResponse",
    v_esc: Array,
    *,
    epsilon: "Array | float" = 1.0,
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
    Reverse-mode AD (``jax.grad``, ``jax.vjp``) is supported via an
    implicit-function-theorem VJP registered on the forward Picard
    iteration.  The backward pass solves the linearised system

    .. math::

        (I - \\partial F / \\partial N)^{\\mathsf T}\\, v = \\lambda

    via GMRES at ``N*`` and then pulls back through
    ``\\partial F / \\partial \\theta``.  This gives exact IFT gradients
    regardless of the number of Picard iterations taken by the forward
    pass — far cheaper and more accurate than unrolling the loop.

    ``max_iter``, ``tol``, and ``damping`` are static hyperparameters
    (not differentiated). Pass them as Python scalars; they are fixed
    at ``jax.jit`` trace time via :func:`functools.lru_cache`.
    """
    _solve = _make_picard_solve_fn(max_iter, tol, damping)
    return _solve(response, jnp.asarray(v_esc), jnp.asarray(epsilon))


__all__ = [
    "ImplicitSolveResult",
    "_F_map",
    "gain_density",
    "loss_density",
    "solve_implicit_fixed_point",
]
