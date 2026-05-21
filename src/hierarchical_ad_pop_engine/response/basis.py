r"""Basis Picard solver — Step 4 of the v3 basis refactor.

Mirrors :mod:`response.implicit`'s structure one-for-one so the diff
review between solvers is clean:

* :class:`BasisSolveResult` ↔ :class:`ImplicitSolveResult`.
* :func:`_F_map_basis` ↔ :func:`_F_map`.
* :func:`_picard_while_loop_basis` ↔ :func:`_picard_while_loop`.
* :func:`_make_basis_solve_fn` ↔ :func:`_make_picard_solve_fn`.
* :func:`solve_basis_fixed_point` ↔ :func:`solve_implicit_fixed_point`.

Differentiable inputs are ``(response, v_esc, w_over_v_esc, epsilon)``.
``basis_solver`` and the Picard hyperparameters are static (closed
over in :func:`_make_basis_solve_fn`).  The IFT VJP is the same
machinery as v2: GMRES on ``(I - ∂F/∂N)ᵀ`` at the fixed point, then
``vjp_θ(F(N*, ·))``.  Only the matvec dimension changes (from grid
``N_m·N_χ`` to basis ``n_basis``); the IFT bookkeeping is unchanged.

Why a separate file (not a switch inside ``implicit.py``)
--------------------------------------------------------

The plan calls for diff-vs-implicit.py review as a correctness check.
A single file with two solver paths would make that review opaque.
Two parallel files with identical structure is the cleanest setup —
the diff highlights exactly which lines change between gain/loss on
a grid vs gain/loss on a basis.

Cross-references: ``demo/PLAN_engine_v3_basis.md`` §Step 4;
memories ``project_engine_v3_step3d_done``, ``project_jax_custom_root_gotchas``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..basis.loss_tensor import build_loss_tensor
from ..basis.natal import project_natal_density
from ..basis.remnant_tensor import RemnantTensorTheta, build_remnant_tensor

if TYPE_CHECKING:
    from ..basis.tensor import TensorProductBasis
    from .coagulation import CoagulationResponse


# ---------------------------------------------------------------------------
# Result container.
# ---------------------------------------------------------------------------
class BasisSolveResult(eqx.Module):
    r"""Result of :func:`solve_basis_fixed_point`.

    All fields are JAX arrays.  Mirrors
    :class:`response.implicit.ImplicitSolveResult` field-for-field
    except that ``N`` and ``log_N`` become ``coeffs`` (flat
    coefficient vector on the solver basis) and there is no
    ``log_N`` field — log is taken at evaluation time on the basis
    expansion, not on the coefficient vector itself (coefficients
    can be negative even when the underlying density is everywhere
    non-negative).

    Attributes
    ----------
    coeffs : Array, shape ``(n_basis,)``
        Converged single-BH density coefficients in flat layout.
        Reshape to ``basis_solver.shape`` via ``coeffs.reshape(*basis_solver.shape)``
        to recover the 2-D ``(n_basis_m, n_basis_χ)`` form.
    epsilon : Array
        Coupling × time used (echoed for downstream diagnostics).
    w_over_v_esc : Array
        Retention sigmoid sharpness used (echoed).  Stage 1 default 1.0.
    n_iter : Array
        Number of Picard iterations actually run.  Cast to the float
        dtype of ``coeffs`` so custom-VJP aux-tangent bookkeeping
        works; read back with ``int(result.n_iter)`` at the eager
        boundary.
    final_diff : Array
        ``‖Δcoeffs‖∞`` at the final iteration.  Should be ``< tol``
        on success.
    loss_integral : Array
        ``∫ loss(m, χ) dm dχ`` at the fixed point — diagnostic
        only.  Computed by LGL quadrature on the basis nodes with
        the ``dm = m d(log m)`` Jacobian.  Note: as in v2, this
        double integral counts each merger pair **twice**; use
        :attr:`f_natal_depleted` for depletion fractions.
    natal_integral : Array
        ``∫ N_natal(m, χ) dm dχ`` — physical density natal integral.
    f_natal_depleted : Array
        ``1 − ∫N*(m,χ) dm dχ / ∫N_natal(m,χ) dm dχ``, clipped to
        ``[0, 1]``.  Same definition as in v2.

    Notes
    -----
    The per-iteration coefficient trace is not recorded
    (``jax.lax.while_loop`` cannot accumulate variable-length traces).
    """

    coeffs: Array
    epsilon: Array
    w_over_v_esc: Array
    n_iter: Array
    final_diff: Array
    loss_integral: Array
    natal_integral: Array
    f_natal_depleted: Array


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------
def _theta_from(
    response: "CoagulationResponse",
    v_esc_arr: Array,
    w_over_v_esc_arr: Array,
) -> RemnantTensorTheta:
    r"""Pack the θ subset that ``T(θ)``, ``L(θ)`` depend on."""
    return RemnantTensorTheta(
        kernel_a=jnp.asarray(response.kernel_params["a"]),
        kernel_b=jnp.asarray(response.kernel_params["b"]),
        v_esc=v_esc_arr,
        w_over_v_esc=w_over_v_esc_arr,
    )


def _integrate_density(
    coeffs_2d: Array,
    basis_solver: "TensorProductBasis",
) -> Array:
    r"""Compute ``∫ f(m, χ) dm dχ`` for a coefficient array ``coeffs_2d``.

    The basis lives on ``(log m, χ)`` with Lebesgue inner product, so
    the physical-mass integral needs the ``dm = m · d(log m)``
    Jacobian carried explicitly.  We evaluate at the basis's own LGL
    nodes and integrate by LGL quadrature; this is exact for
    polynomials of degree ``2 n_basis − 3`` in ``log m``, modulo the
    Jacobian ``m = exp(log m)``.  Acceptable for diagnostics — finer
    accuracy would overintegrate on a higher-order LGL grid.
    """
    f_at_nodes = basis_solver.evaluate_at_nodes(coeffs_2d)
    m_at_nodes = jnp.exp(basis_solver.basis_m.nodes)
    w_m = basis_solver.basis_m.weights
    w_chi = basis_solver.basis_chi.weights
    return jnp.einsum("ij,i,i,j->", f_at_nodes, m_at_nodes, w_m, w_chi)


# ---------------------------------------------------------------------------
# Picard RHS — F(coeffs; theta).  Rebuilds T, L per call so AD flows
# through every θ field.  The forward while_loop calls F once per
# iteration; the IFT backward VJP calls it twice (once for ∂F/∂N,
# once for ∂F/∂θ).  For inference efficiency the while_loop variant
# below caches T, L outside the loop.
# ---------------------------------------------------------------------------
def _F_map_basis(
    coeffs_flat: Array,
    response: "CoagulationResponse",
    v_esc_arr: Array,
    w_over_v_esc_arr: Array,
    epsilon_arr: Array,
    basis_solver: "TensorProductBasis",
    n_quad_factor: int,
) -> Array:
    r"""Picard RHS on basis coefficients: ``n_natal + ε·(gain - loss)``.

    Rebuilds ``T(θ)``, ``L(θ)``, and ``n_natal_α(response)`` per call
    — slow but the gradient pulls cleanly through every θ field via
    standard JAX VJP.  Used by the custom-VJP backward to compute
    ``∂F/∂N`` and ``∂F/∂θ``; the forward path uses the cached
    version inside ``_picard_while_loop_basis``.
    """
    theta = _theta_from(response, v_esc_arr, w_over_v_esc_arr)
    n_natal_2d = project_natal_density(response, basis_solver)
    n_natal_alpha = n_natal_2d.reshape(-1)
    T = build_remnant_tensor(theta, basis_solver, n_quad_factor=n_quad_factor)
    L = build_loss_tensor(theta, basis_solver, n_quad_factor=n_quad_factor)
    gain = 0.5 * jnp.einsum("gab,a,b->g", T, coeffs_flat, coeffs_flat)
    loss = jnp.einsum("gab,a,b->g", L, coeffs_flat, coeffs_flat)
    return n_natal_alpha + epsilon_arr * (gain - loss)


# ---------------------------------------------------------------------------
# Raw Picard while_loop — JIT-traceable, not AD-safe in isolation.
# Caches T, L, n_natal outside the loop.
# ---------------------------------------------------------------------------
def _picard_while_loop_basis(
    response: "CoagulationResponse",
    v_esc_arr: Array,
    w_over_v_esc_arr: Array,
    epsilon_arr: Array,
    basis_solver: "TensorProductBasis",
    max_iter: int,
    tol: float,
    damping: float,
    n_quad_factor: int,
) -> BasisSolveResult:
    r"""Damped-Picard fixed-point on coefficient space via
    :func:`jax.lax.while_loop`.

    Builds ``T(θ)``, ``L(θ)``, and the natal coefficients ONCE
    outside the loop; the loop body is a pair of cheap einsums.
    """
    theta = _theta_from(response, v_esc_arr, w_over_v_esc_arr)
    n_natal_2d = project_natal_density(response, basis_solver)
    n_natal_alpha = n_natal_2d.reshape(-1)
    T = build_remnant_tensor(theta, basis_solver, n_quad_factor=n_quad_factor)
    L = build_loss_tensor(theta, basis_solver, n_quad_factor=n_quad_factor)

    n_basis = basis_solver.n_basis
    tol_arr = jnp.asarray(tol, dtype=n_natal_alpha.dtype)
    max_iter_arr = jnp.asarray(max_iter, dtype=jnp.int32)

    def F(coeffs: Array) -> Array:
        gain = 0.5 * jnp.einsum("gab,a,b->g", T, coeffs, coeffs)
        loss = jnp.einsum("gab,a,b->g", L, coeffs, coeffs)
        return n_natal_alpha + epsilon_arr * (gain - loss)

    init_carry = (
        n_natal_alpha,
        jnp.zeros((), dtype=jnp.int32),
        jnp.asarray(jnp.inf, dtype=n_natal_alpha.dtype),
    )

    def cond_fun(carry):
        _, n_iter, diff = carry
        return (diff > tol_arr) & (n_iter < max_iter_arr)

    def body_fun(carry):
        coeffs, n_iter, _ = carry
        # Damped Picard:  c_new = (1 - d)·c + d·F(c) = c - d·(c - F(c)).
        # Unlike v2 we do **not** clamp coefficients ≥ 0 — modal
        # coefficients can be negative (e.g. Gibbs ringing at sharp
        # support boundaries) even when the underlying density is
        # everywhere non-negative.  Clamping would break Galerkin
        # orthogonality.  If a downstream observable requires
        # non-negative density, it should clamp the *evaluated*
        # density at evaluation time.
        coeffs_new = coeffs - damping * (coeffs - F(coeffs))
        diff = jnp.max(jnp.abs(coeffs_new - coeffs))
        return (coeffs_new, n_iter + 1, diff)

    coeffs_star, n_iter_final, diff_final = jax.lax.while_loop(
        cond_fun, body_fun, init_carry
    )

    # Diagnostics — done on the cached L for loss_integral, on the
    # basis-evaluated density for natal/n* integrals.
    loss_star = jnp.einsum("gab,a,b->g", L, coeffs_star, coeffs_star)
    # ∫ loss dm dχ:  the loss vector is in coefficient space at solver
    # basis; turn back into a function on (m, χ) and integrate by
    # LGL quadrature with the dm Jacobian.
    loss_2d = loss_star.reshape(basis_solver.shape)
    loss_integral = _integrate_density(loss_2d, basis_solver)
    natal_2d = n_natal_alpha.reshape(basis_solver.shape)
    natal_integral = _integrate_density(natal_2d, basis_solver)
    n_star_integral = _integrate_density(
        coeffs_star.reshape(basis_solver.shape), basis_solver
    )
    # f = 1 - ∫N*/∫N_natal.  Same derivation as v2 (∫G = (1/2)·loss_integral
    # at fixed point, so ∫N* = ∫N_natal - (ε/2)·loss_integral and the
    # depletion fraction is what's left).
    f_natal_depleted = jnp.clip(
        1.0 - n_star_integral / jnp.maximum(natal_integral, 1e-30),
        0.0,
        1.0,
    )

    return BasisSolveResult(
        coeffs=coeffs_star,
        epsilon=epsilon_arr,
        w_over_v_esc=w_over_v_esc_arr,
        n_iter=n_iter_final.astype(n_natal_alpha.dtype),
        final_diff=diff_final,
        loss_integral=loss_integral,
        natal_integral=natal_integral,
        f_natal_depleted=f_natal_depleted,
    )


# ---------------------------------------------------------------------------
# AD-safe factory — registers IFT VJP via @jax.custom_vjp.
# ---------------------------------------------------------------------------
def _make_basis_solve_fn(
    basis_solver: "TensorProductBasis",
    max_iter: int,
    tol: float,
    damping: float,
    n_quad_factor: int,
):
    r"""Return an AD-safe basis-Picard solver closure.

    ``basis_solver`` and the Picard hyperparameters are static
    (closed over).  Differentiable inputs are
    ``(response, v_esc_arr, w_over_v_esc_arr, epsilon_arr)``.

    Why a factory: ``@jax.custom_vjp`` doesn't accept non-differentiable
    JAX pytrees gracefully.  Closing over ``basis_solver`` keeps it
    out of the differentiated-arg surface, the same trick v2's
    ``_make_picard_solve_fn`` uses for ``(max_iter, tol, damping)``.

    No :func:`functools.lru_cache` here (unlike v2) — keying on a
    hashable proxy of ``basis_solver`` adds complexity for a
    micro-optimization that isn't load-bearing at Stage 1.  The
    JIT cache inside ``jax.jit`` does the heavy lifting per
    ``(input shape, hyperparam)`` combination.
    """

    @jax.custom_vjp
    def _solve(
        response: "CoagulationResponse",
        v_esc_arr: Array,
        w_over_v_esc_arr: Array,
        epsilon_arr: Array,
    ) -> BasisSolveResult:
        return _picard_while_loop_basis(
            response,
            v_esc_arr,
            w_over_v_esc_arr,
            epsilon_arr,
            basis_solver,
            max_iter,
            tol,
            damping,
            n_quad_factor,
        )

    def _solve_fwd(
        response: "CoagulationResponse",
        v_esc_arr: Array,
        w_over_v_esc_arr: Array,
        epsilon_arr: Array,
    ):
        result = _solve(response, v_esc_arr, w_over_v_esc_arr, epsilon_arr)
        residuals = (
            result.coeffs,
            response,
            v_esc_arr,
            w_over_v_esc_arr,
            epsilon_arr,
        )
        return result, residuals

    def _solve_bwd(residuals, g: BasisSolveResult):
        r"""IFT VJP at the basis Picard fixed point.

        Pull-back through ``c* = F(c*, θ)``.  Same algebra as v2's
        ``_solve_bwd``: GMRES on ``(I − ∂F/∂c)ᵀ`` to obtain ``v``,
        then ``cotangent_θ = vjp_θ(F(c*, ·))(v)``.

        Differences from v2:
        * Carry is flat ``(n_basis,)`` not ``(N_m, N_χ)``.
        * Four differentiable args instead of three (``w_over_v_esc``
          is new at Stage 1).
        * Natal-integral cotangent flows through
          ``project_natal_density`` instead of v2's
          ``response._natal_log_density`` on the grid.
        """
        (
            coeffs_star,
            response,
            v_esc_arr,
            w_over_v_esc_arr,
            epsilon_arr,
        ) = residuals

        # Aggregate cotangent on coeffs_star from result fields.
        # Only ``coeffs`` carries a direct cotangent into the IFT
        # GMRES; the others (epsilon, w_over_v_esc, n_iter, etc.)
        # echo or are diagnostic and bypass the IFT.
        g_coeffs = g.coeffs

        # GMRES: solve (I − ∂F/∂c)ᵀ v = g_coeffs.
        def jac_c_T_v(v: Array) -> Array:
            r"""(∂F/∂c)ᵀ v via VJP of _F_map_basis wrt c at c_star."""
            _, vjp_fn = jax.vjp(
                lambda c: _F_map_basis(
                    c,
                    response,
                    v_esc_arr,
                    w_over_v_esc_arr,
                    epsilon_arr,
                    basis_solver,
                    n_quad_factor,
                ),
                coeffs_star,
            )
            return vjp_fn(v)[0]

        def linear_op_T(v: Array) -> Array:
            return v - jac_c_T_v(v)

        v_sol, _ = jax.scipy.sparse.linalg.gmres(
            linear_op_T,
            g_coeffs,
            tol=1e-7,
            atol=0.0,
            restart=20,
            maxiter=200,
        )

        # Propagate: cotangent_θ = vjp_θ(F(c*, ·))(v).
        _, vjp_theta_fn = jax.vjp(
            lambda resp, ve, wov, eps: _F_map_basis(
                coeffs_star,
                resp,
                ve,
                wov,
                eps,
                basis_solver,
                n_quad_factor,
            ),
            response,
            v_esc_arr,
            w_over_v_esc_arr,
            epsilon_arr,
        )
        (
            cotangent_response,
            cotangent_v_esc,
            cotangent_w_over_v_esc,
            cotangent_eps,
        ) = vjp_theta_fn(v_sol)

        # Echo-field cotangents (epsilon, w_over_v_esc fields of
        # BasisSolveResult are pass-throughs).
        cotangent_eps = cotangent_eps + g.epsilon
        cotangent_w_over_v_esc = cotangent_w_over_v_esc + g.w_over_v_esc

        # natal_integral cotangent — bypasses the IFT, flows
        # directly into cotangent_response.
        _, vjp_natal_fn = jax.vjp(
            lambda resp: _integrate_density(
                project_natal_density(resp, basis_solver), basis_solver
            ),
            response,
        )
        ct_response_natal = vjp_natal_fn(g.natal_integral)[0]
        cotangent_response = jax.tree_util.tree_map(
            jnp.add, cotangent_response, ct_response_natal
        )

        return (
            cotangent_response,
            cotangent_v_esc,
            cotangent_w_over_v_esc,
            cotangent_eps,
        )

    _solve.defvjp(_solve_fwd, _solve_bwd)
    return _solve


# ---------------------------------------------------------------------------
# Public solver.
# ---------------------------------------------------------------------------
def solve_basis_fixed_point(
    response: "CoagulationResponse",
    basis_solver: "TensorProductBasis",
    v_esc: "Array | float",
    *,
    epsilon: "Array | float" = 1.0,
    w_over_v_esc: "Array | float" = 1.0,
    max_iter: int = 200,
    tol: float = 1e-7,
    damping: float = 0.5,
    n_quad_factor: int = 2,
) -> BasisSolveResult:
    r"""Damped-Picard fixed point on coefficient space for the v3 basis
    solver.

    Parameters
    ----------
    response : CoagulationResponse
        Source of natal density (via :func:`basis.natal.project_natal_density`)
        and kernel parameters (``response.kernel_params``).  The
        ``solver`` and ``epsilon`` fields on ``response`` are ignored —
        pass ``epsilon`` and ``w_over_v_esc`` explicitly as keywords.
    basis_solver : TensorProductBasis
        Solver basis on ``(log m, χ)``.  Use
        :py:meth:`TensorProductBasis.for_binary_grid` to construct.
    v_esc : Array | float
        Per-environment escape velocity in km/s.  Differentiable.
    epsilon : Array | float, default 1.0
        Coupling × time.  Differentiable.  Absolute physical meaning.
    w_over_v_esc : Array | float, default 1.0
        Retention sigmoid sharpness as a fraction of ``v_esc`` —
        Stage 3 Decision 3 rule, never an absolute km/s number.
        Differentiable.  Useful range ``[1.0, 2.0]``.
    max_iter, tol, damping : int, float, float
        Picard hyperparameters.  Same conventions as v2.
    n_quad_factor : int, default 2
        Overintegration factor for ``T`` and ``L`` quadrature.

    Returns
    -------
    BasisSolveResult
        Converged fixed point plus diagnostics; see class docstring.

    Notes
    -----
    Reverse-mode AD is supported via the IFT VJP — identical machinery
    to v2 :func:`response.implicit.solve_implicit_fixed_point`, just on
    a coefficient-space carry.  Use ``jax.grad`` / ``jax.vjp``
    transparently; the static hyperparameters (``max_iter``, ``tol``,
    ``damping``, ``n_quad_factor``, ``basis_solver``) are closed over
    in :func:`_make_basis_solve_fn` and are not differentiated.

    A small Python-time cost (closure creation) is paid per call;
    add :func:`functools.lru_cache` keyed on hashable proxies for
    ``basis_solver`` if profiling shows this matters.
    """
    _solve = _make_basis_solve_fn(
        basis_solver, max_iter, tol, damping, n_quad_factor
    )
    return _solve(
        response,
        jnp.asarray(v_esc),
        jnp.asarray(w_over_v_esc),
        jnp.asarray(epsilon),
    )


__all__ = [
    "BasisSolveResult",
    "_F_map_basis",
    "solve_basis_fixed_point",
]
