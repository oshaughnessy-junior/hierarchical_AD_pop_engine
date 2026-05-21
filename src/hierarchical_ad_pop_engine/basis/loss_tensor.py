r"""Loss-term projection tensor ``L(θ)`` for the v3 basis Picard solver.

Step 4 of ``demo/PLAN_engine_v3_basis.md``.  Companion to
:mod:`basis.remnant_tensor`.

The v2 loss density is the pointwise product of the single-BH density
with the pair-averaged "encounter integrand"::

    loss(m, χ) = N(m, χ) · ∫∫ K(m, m') P_{ret}(m, m', χ, χ')
                              · N(m', χ') dm' dχ'.

Projecting onto the basis with the Lebesgue inner product and
folding both factors of ``N`` into modal coefficients gives a
rank-3 tensor ``L`` such that

.. math::

    \mathrm{loss}_\gamma  \;=\;
        \sum_{\alpha, \beta} n_\alpha \, n_\beta \, L_{\gamma;\alpha,\beta}(\theta).

Concretely, with the same ``(log m, χ)`` tensor-product basis the
natal pipeline uses:

.. math::

    L_{\gamma;\alpha,\beta}(\theta)
        \;=\;
        \frac{1}{M_{\gamma\gamma}}
        \int\!\!\!\int\!\!\!\int\!\!\!\int
            \varphi_\gamma(\log m_1, \chi_1) \,
            \varphi_\alpha(\log m_1, \chi_1) \,
            K(m_1, m_2; \theta_K) \,
            P_{ret}(m_1, m_2, \chi_1, \chi_2; v_{\rm esc}, w) \,
            \varphi_\beta(\log m_2, \chi_2) \,
            \mathrm{d}m_1 \mathrm{d}\chi_1 \mathrm{d}m_2 \mathrm{d}\chi_2

— so ``γ`` is "twinned" with ``α`` at the surviving-particle leg of
the integrand and ``β`` is contracted at the partner leg.  The
``dm = m \, d\log m`` Jacobian is carried explicitly (the basis is
on ``log m``, the physical measure is ``dm``).

Compared with :func:`basis.remnant_tensor.build_remnant_tensor`, the
loss tensor:

* **Does not see the remnant map.**  No ``M_rem`` / ``\chi_{rem}``
  evaluation, no in-domain mask on the output — the surviving
  particle is at the same point as the integrand evaluation, which
  by construction sits inside the basis domain.
* **Is not symmetric in α ↔ β.**  ``γ`` is twinned with ``α`` only.
  The bosonic-pair correction (factor 1/2 in the gain term) does
  **not** appear in the loss — each ordered ``(m_1, m_2)`` pair
  contributes once because both progenitors leave the population.

Memory at chosen resolution
---------------------------

Same as ``T``: ``(n_basis, n_basis, n_basis)`` flat.  At Stage 1
solver resolution ``(32, 16)``, that's 1.1 GB float64.  Combined
with ``T``, the basis Picard solver footprint is 2.2 GB — fine for
a modest workstation, and the brute-force ``C[N,N]`` fallback (see
``feedback_dont_straightjacket_basis``) remains the eventual swap
path if Stage 2 spin resolution pushes us past memory.

Why bake K and P_retain inside L (matching T)
---------------------------------------------

Same rationale as the gain tensor (see
:mod:`basis.remnant_tensor`'s docstring): θ-dependent quantities
(kernel exponents, retention thresholds, retention sharpness) flow
through one contraction.  Step 4's Picard map then reduces to two
cheap einsums on the cached ``T``, ``L`` per iteration.

Cross-references: ``PLAN_engine_v3_basis.md`` §Step 4;
:mod:`basis.remnant_tensor`; memory
``project_engine_v3_step3d_done``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array

from ..kernels import log_pairing_kernel
from ..remnants.kick import v_kick_aligned
from .legendre import _legendre_recurrence
from .remnant_tensor import (
    RemnantTensorTheta,
    _lgl_nodes_weights_np,
    retention_sigmoid,
)

if TYPE_CHECKING:
    from .tensor import TensorProductBasis


def build_loss_tensor(
    theta: RemnantTensorTheta,
    basis_solver: "TensorProductBasis",
    *,
    n_quad_factor: int = 2,
) -> Array:
    r"""Build the loss projection tensor ``L_{γ;α,β}(θ)``.

    Signature mirrors :func:`basis.remnant_tensor.build_remnant_tensor`
    — same ``theta``, same ``basis_solver``, same overintegration
    factor — so the Step 4 Picard map can swap one helper for the
    other without rewiring.

    Parameters and conventions: see module docstring above.

    Returns
    -------
    Array of shape ``(n_basis, n_basis, n_basis)`` where
    ``n_basis = n_basis_m · n_basis_χ``.  Flat index ordering on every
    axis is ``α_m · n_basis_χ + α_χ`` (matches
    :py:meth:`TensorProductBasis.flatten`).

    Notes
    -----
    Unlike ``T``, ``L`` is not symmetric in ``α ↔ β``.  The
    acceptance test in ``tests/test_basis_solver.py`` checks the
    weaker conservation invariant

    .. math::

        \sum_\gamma M_{\gamma\gamma} \, L_{\gamma;\alpha,\beta}
            = \langle \mathrm{const}, \mathrm{loss}_{\alpha,\beta}\rangle

    which says "the projected loss density integrates to the same
    total loss count as the raw double quadrature."
    """
    basis_m = basis_solver.basis_m
    basis_chi = basis_solver.basis_chi
    n_m = basis_m.n_basis
    n_chi = basis_chi.n_basis
    a_m, b_m = basis_m.domain
    a_chi, b_chi = basis_chi.domain

    # --- Quadrature nodes / weights on the progenitor axes ---------------
    # Same overintegration story as build_remnant_tensor.  Numpy mirror
    # avoids the Python float() early-exit that breaks jax.jit.
    n_quad_m = n_quad_factor * n_m
    n_quad_chi = n_quad_factor * n_chi
    xi_m_q_np, w_m_q_ref_np = _lgl_nodes_weights_np(n_quad_m)
    xi_chi_q_np, w_chi_q_ref_np = _lgl_nodes_weights_np(n_quad_chi)
    xi_m_q = jnp.asarray(xi_m_q_np)
    w_m_q_ref = jnp.asarray(w_m_q_ref_np)
    xi_chi_q = jnp.asarray(xi_chi_q_np)
    w_chi_q_ref = jnp.asarray(w_chi_q_ref_np)

    half_m = 0.5 * (b_m - a_m)
    mid_m = 0.5 * (a_m + b_m)
    half_chi = 0.5 * (b_chi - a_chi)
    mid_chi = 0.5 * (a_chi + b_chi)

    log_m_q = mid_m + half_m * xi_m_q  # (n_quad_m,) — log m
    chi_q = mid_chi + half_chi * xi_chi_q  # (n_quad_chi,)
    m_q = jnp.exp(log_m_q)

    w_m_q = half_m * w_m_q_ref
    w_chi_q = half_chi * w_chi_q_ref

    # --- Basis table at the progenitor quadrature nodes ------------------
    # phi_m_q   : (n_quad_m,   n_m) — P_α at the m-quadrature nodes
    # phi_chi_q : (n_quad_chi, n_chi)
    P_m_table = _legendre_recurrence(n_m - 1, xi_m_q)         # (n_m, n_quad_m)
    phi_m_q = P_m_table.T                                      # (n_quad_m, n_m)
    P_chi_table = _legendre_recurrence(n_chi - 1, xi_chi_q)   # (n_chi, n_quad_chi)
    phi_chi_q = P_chi_table.T                                  # (n_quad_chi, n_chi)

    # --- Build the 4-D progenitor mesh -----------------------------------
    m1 = m_q[:, None, None, None]
    chi1 = chi_q[None, :, None, None]
    m2 = m_q[None, None, :, None]
    chi2 = chi_q[None, None, None, :]

    # Pairing kernel.  Symmetric in (m1, m2) by construction.
    log_K = log_pairing_kernel(m1, m2, theta.kernel_a, theta.kernel_b)
    K = jnp.exp(log_K)

    # Canonical mass / spin ordering — same fix pattern as in
    # build_remnant_tensor.  v_kick_aligned canonicalises internally
    # but applying the swap once at the call site keeps the integrand
    # exactly (m1,χ1) ↔ (m2,χ2) symmetric in the K·P factor.  Note
    # that L is not symmetric in α↔β (γ is twinned with α), but K·P
    # must still respect the unordered-pair symmetry of the physics.
    heavier_first = m1 >= m2
    m1_c = jnp.where(heavier_first, m1, m2)
    m2_c = jnp.where(heavier_first, m2, m1)
    chi1_c = jnp.where(heavier_first, chi1, chi2)
    chi2_c = jnp.where(heavier_first, chi2, chi1)

    v_k = v_kick_aligned(m1_c, m2_c, chi1_c, chi2_c)
    P_ret = retention_sigmoid(v_k, theta.v_esc, theta.w_over_v_esc)

    # --- Integrand pre-multiplied by weights + Jacobians -----------------
    # See build_remnant_tensor for the dm = m d(log m) Jacobian
    # rationale.
    w_quad = (
        w_m_q[:, None, None, None]
        * w_chi_q[None, :, None, None]
        * w_m_q[None, None, :, None]
        * w_chi_q[None, None, None, :]
    )
    jacobian = m1 * m2
    integrand_no_basis = K * P_ret * jacobian * w_quad
    # Shape: (qm, qχ, qm, qχ).  γ-basis evaluation at (log m1, χ1)
    # is just phi_m_q[i, γ_m] · phi_chi_q[j, γ_χ] — a 2-D tensor
    # that we'll broadcast against (i, j, k, l) below.

    # --- The contraction -------------------------------------------------
    # L[γ_m, γ_χ, α_m, α_χ, β_m, β_χ]
    #   = Σ_{i,j,k,l} integrand_no_basis[i,j,k,l]
    #     · phi_m_q[i, γ_m] · phi_chi_q[j, γ_χ]
    #     · phi_m_q[i, α_m] · phi_chi_q[j, α_χ]
    #     · phi_m_q[k, β_m] · phi_chi_q[l, β_χ]
    #
    # γ is twinned with α at axis (i, j), so the product
    #   phi_m_q[i, γ_m] · phi_m_q[i, α_m]
    # cannot be linearly separated into independent contractions —
    # we have to bring the γ basis into the integrand at the (i, j)
    # axis level.  In contrast, β at axis (k, l) factors cleanly.
    #
    # We can either fold γ into the integrand up front (peak memory
    # n_m · n_χ · qm · qχ · qm · qχ = same as T's intermediate)
    # or do a tensordot in steps.  The step-by-step approach is the
    # same peak memory but easier to read; use it.

    # Step 1: contract m2 axis (k) against β_m.
    # integrand_no_basis : (qm, qχ, qm, qχ)
    # phi_m_q            : (qm, n_m) — used as (K, β_m)
    step_1 = jnp.einsum("ijkl,kc->ijcl", integrand_no_basis, phi_m_q)
    # Shape: (qm, qχ, β_m, qχ).

    # Step 2: contract χ2 axis (l) against β_χ.
    step_2 = jnp.einsum("ijcl,ld->ijcd", step_1, phi_chi_q)
    # Shape: (qm, qχ, β_m, β_χ).

    # Step 3: bring γ basis in at axis (i, j).
    #   step_3[γ_m, γ_χ, i, j, β_m, β_χ]
    #     = phi_m_q[i, γ_m] · phi_chi_q[j, γ_χ] · step_2[i, j, β_m, β_χ]
    # Memory: n_m · n_χ · qm · qχ · n_m · n_χ — at (32, 16) this is
    # 32·16·64·32·32·16 = 1.7e8 floats = 1.3 GB.  Larger than T's
    # intermediate.  We do it in two pieces to keep the size of the
    # in-flight intermediate the same as T's worst case:
    #
    #   first fold the γ_m basis on axis i:
    step_3a = jnp.einsum("ijcd,iG->Gjcd", step_2, phi_m_q)
    # Shape: (γ_m, qχ, β_m, β_χ).  Size: 32·32·32·16 = 5.2e5 floats — fine.

    # Step 4: fold γ_χ basis on axis j.
    step_3b = jnp.einsum("Gjcd,jg->Ggcd", step_3a, phi_chi_q)
    # Shape: (γ_m, γ_χ, β_m, β_χ).  Size: 32·16·32·16 = 2.6e5 floats.

    # Step 5: bring α in at axis (i, j) — this requires going back
    # to the (i, j) axes, which we already contracted out above.  So
    # we need a slightly different order: don't contract γ first.
    #
    # Re-plan:
    #   1) Contract β (axes k, l) — done; gives (i, j, β_m, β_χ).
    #   2) Fold γ_χ basis on axis j twinned with α_χ on axis j:
    #      that doesn't factor either — γ_χ and α_χ both attach to j.
    #
    # The "γ twinned with α at (i, j)" structure means we need to
    # materialise the (γ_m, γ_χ, α_m, α_χ) outer product at each
    # (i, j), or equivalently the (γ_m, α_m) outer at i and the
    # (γ_χ, α_χ) outer at j.  Build those as 2-tensors on each axis:
    #
    #   M_alpha_gamma_at_i[i, γ_m, α_m] = phi_m_q[i, γ_m] · phi_m_q[i, α_m]
    #   M_alpha_gamma_at_j[j, γ_χ, α_χ] = phi_chi_q[j, γ_χ] · phi_chi_q[j, α_χ]
    #
    # Then contract these against step_2 along (i, j).
    #
    # Memory of the outer-product tables:
    #   m-axis: qm · n_m · n_m = 64 · 32 · 32 = 65k floats — trivial
    #   χ-axis: qχ · n_χ · n_χ = 32 · 16 · 16 = 8k floats — trivial
    # Way better than the dense expansion above.
    #
    # Discard step_3a, step_3b; they were a wrong turn.
    del step_3a, step_3b

    M_ga_m = jnp.einsum("iG,ia->iGa", phi_m_q, phi_m_q)
    M_ga_chi = jnp.einsum("jg,jb->jgb", phi_chi_q, phi_chi_q)

    # Contract step_2[i, j, β_m, β_χ] against M_ga_m[i, γ_m, α_m]
    # and M_ga_chi[j, γ_χ, α_χ]:
    #   L[γ_m, γ_χ, α_m, α_χ, β_m, β_χ]
    #     = Σ_{i,j} M_ga_m[i, γ_m, α_m] · M_ga_chi[j, γ_χ, α_χ]
    #       · step_2[i, j, β_m, β_χ]
    # We do it in two steps to control peak memory.
    #
    # First contract i:
    #   T_m[γ_m, α_m, j, β_m, β_χ] = Σ_i M_ga_m[i, γ_m, α_m] · step_2[i, j, β_m, β_χ]
    T_m = jnp.einsum("iGa,ijcd->Gajcd", M_ga_m, step_2)
    # Shape: (γ_m, α_m, qχ, β_m, β_χ).  Size at (32,16): 32·32·32·32·16 = 1.7e7 — 130 MB.

    # Then contract j:
    #   L[γ_m, α_m, γ_χ, α_χ, β_m, β_χ] = Σ_j M_ga_chi[j, γ_χ, α_χ] · T_m[γ_m, α_m, j, β_m, β_χ]
    L_dense = jnp.einsum("jgb,Gajcd->GgaBbcd"[:13], M_ga_chi, T_m)  # noqa: E501
    # Rewriting the einsum cleanly:
    L_dense = jnp.einsum("jgb,Gajcd->Ggabcd", M_ga_chi, T_m)
    # Shape: (γ_m, γ_χ, α_m, α_χ, β_m, β_χ) — matches the target layout.

    # Apply (1/M_γγ) — diagonal Lebesgue mass matrix on γ.
    M_diag_2d = basis_solver.mass_diag_2d()
    L_dense = L_dense / M_diag_2d[:, :, None, None, None, None]

    n_basis = n_m * n_chi
    return L_dense.reshape(n_basis, n_basis, n_basis)


def make_t_l_gain_loss(basis_solver: "TensorProductBasis", *, n_quad_factor: int = 2):
    r"""Return ``(gain_fn, loss_fn)`` closures for the Step 4 Picard map.

    Both closures consume ``(coeffs_flat, theta)`` and return a flat
    coefficient vector of shape ``(basis_solver.n_basis,)``.  The
    factor of ``1/2`` (unordered-pair correction) lives in
    ``gain_fn``, not inside ``T``.  ``loss_fn`` has no symmetry
    factor — each ordered pair contributes once because both
    progenitors leave.

    Both closures call :func:`build_remnant_tensor` /
    :func:`build_loss_tensor` on every invocation.  In Step 4's
    Picard while_loop the tensors are built **once** outside the
    loop; this closure is the form the IFT VJP's per-call linearisation
    uses (where rebuilding is desired so AD flows through θ).
    """
    from .remnant_tensor import build_remnant_tensor

    def gain_fn(coeffs_flat: Array, theta: RemnantTensorTheta) -> Array:
        T = build_remnant_tensor(theta, basis_solver, n_quad_factor=n_quad_factor)
        return 0.5 * jnp.einsum("gab,a,b->g", T, coeffs_flat, coeffs_flat)

    def loss_fn(coeffs_flat: Array, theta: RemnantTensorTheta) -> Array:
        L = build_loss_tensor(theta, basis_solver, n_quad_factor=n_quad_factor)
        return jnp.einsum("gab,a,b->g", L, coeffs_flat, coeffs_flat)

    return gain_fn, loss_fn


__all__ = [
    "build_loss_tensor",
    "make_t_l_gain_loss",
]
