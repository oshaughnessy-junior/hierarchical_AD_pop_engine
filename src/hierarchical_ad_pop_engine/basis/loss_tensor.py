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

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..kernels import log_pairing_kernel
from ..remnants.kick import v_kick_aligned
from ..remnants.marginal import v_kick_at_orientation
from ..spin_orientation import Aligned
from .legendre import _legendre_recurrence
from .remnant_tensor import (
    RemnantTensorTheta,
    _lgl_nodes_weights_np,
    retention_sigmoid,
)

if TYPE_CHECKING:
    from .tensor import TensorProductBasis


def _build_loss_tensor_quadrature_scaffolding(
    theta: RemnantTensorTheta,
    basis_solver: "TensorProductBasis",
    n_quad_factor: int,
):
    """Quadrature/basis-table/4D-mesh scaffolding shared by both the
    aligned and the marginalised L-build paths."""
    basis_m = basis_solver.basis_m
    basis_chi = basis_solver.basis_chi
    n_m = basis_m.n_basis
    n_chi = basis_chi.n_basis
    a_m, b_m = basis_m.domain
    a_chi, b_chi = basis_chi.domain

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

    log_m_q = mid_m + half_m * xi_m_q
    chi_q = mid_chi + half_chi * xi_chi_q
    m_q = jnp.exp(log_m_q)

    w_m_q = half_m * w_m_q_ref
    w_chi_q = half_chi * w_chi_q_ref

    P_m_table = _legendre_recurrence(n_m - 1, xi_m_q)
    phi_m_q = P_m_table.T
    P_chi_table = _legendre_recurrence(n_chi - 1, xi_chi_q)
    phi_chi_q = P_chi_table.T

    m1 = m_q[:, None, None, None]
    chi1 = chi_q[None, :, None, None]
    m2 = m_q[None, None, :, None]
    chi2 = chi_q[None, None, None, :]

    log_K = log_pairing_kernel(m1, m2, theta.kernel_a, theta.kernel_b)
    K = jnp.exp(log_K)

    w_quad = (
        w_m_q[:, None, None, None]
        * w_chi_q[None, :, None, None]
        * w_m_q[None, None, :, None]
        * w_chi_q[None, None, None, :]
    )
    jacobian = m1 * m2

    M_diag_2d = basis_solver.mass_diag_2d()

    return {
        "m1": m1, "m2": m2, "chi1": chi1, "chi2": chi2,
        "K": K, "w_quad": w_quad, "jacobian": jacobian,
        "n_m": n_m, "n_chi": n_chi,
        "phi_m_q": phi_m_q, "phi_chi_q": phi_chi_q,
        "M_diag_2d": M_diag_2d,
    }


def _contract_loss_from_integrand(integrand_no_basis: Array, qd: dict) -> Array:
    """Apply the γ-twinned-α / β-on-partner contractions to produce L.

    Identical to the original ``build_loss_tensor`` contraction
    sequence; lifted out so the aligned and marginalised paths share
    it.
    """
    phi_m_q = qd["phi_m_q"]
    phi_chi_q = qd["phi_chi_q"]
    n_m = qd["n_m"]
    n_chi = qd["n_chi"]

    step_1 = jnp.einsum("ijkl,kc->ijcl", integrand_no_basis, phi_m_q)
    step_2 = jnp.einsum("ijcl,ld->ijcd", step_1, phi_chi_q)

    M_ga_m = jnp.einsum("iG,ia->iGa", phi_m_q, phi_m_q)
    M_ga_chi = jnp.einsum("jg,jb->jgb", phi_chi_q, phi_chi_q)

    T_m = jnp.einsum("iGa,ijcd->Gajcd", M_ga_m, step_2)
    L_dense = jnp.einsum("jgb,Gajcd->Ggabcd", M_ga_chi, T_m)

    M_diag_2d = qd["M_diag_2d"]
    L_dense = L_dense / M_diag_2d[:, :, None, None, None, None]

    n_basis = n_m * n_chi
    return L_dense.reshape(n_basis, n_basis, n_basis)


def _build_loss_tensor_aligned(
    theta: RemnantTensorTheta,
    basis_solver: "TensorProductBasis",
    n_quad_factor: int,
) -> Array:
    """v1 aligned-only L-build — unchanged from pre-Session-B."""
    qd = _build_loss_tensor_quadrature_scaffolding(theta, basis_solver, n_quad_factor)
    m1, m2, chi1, chi2 = qd["m1"], qd["m2"], qd["chi1"], qd["chi2"]

    heavier_first = m1 >= m2
    m1_c = jnp.where(heavier_first, m1, m2)
    m2_c = jnp.where(heavier_first, m2, m1)
    chi1_c = jnp.where(heavier_first, chi1, chi2)
    chi2_c = jnp.where(heavier_first, chi2, chi1)

    v_k = v_kick_aligned(m1_c, m2_c, chi1_c, chi2_c)
    P_ret = retention_sigmoid(v_k, theta.v_esc, theta.w_over_v_esc)

    integrand_no_basis = qd["K"] * P_ret * qd["jacobian"] * qd["w_quad"]
    return _contract_loss_from_integrand(integrand_no_basis, qd)


def _build_loss_tensor_marginalised(
    theta: RemnantTensorTheta,
    basis_solver: "TensorProductBasis",
    orientation,
    n_quad_factor: int,
) -> Array:
    """Marginalised L-build via ``jax.lax.scan`` over orientation quadrature.

    Same memory/compute scaling as
    :func:`basis.remnant_tensor._build_remnant_tensor_marginalised`:
    one full-shape L-contribution alive per scan iteration; compute
    cost is ``n_orient × aligned``.

    L is NOT symmetric in ``α ↔ β`` — γ is twinned with α at the
    surviving-particle leg.  The orientation marginalisation acts on
    ``v_kick`` (which only enters the K·P factor symmetric in
    (m1, m2), so the asymmetry is preserved).
    """
    qd = _build_loss_tensor_quadrature_scaffolding(theta, basis_solver, n_quad_factor)
    m1, m2, chi1, chi2 = qd["m1"], qd["m2"], qd["chi1"], qd["chi2"]
    K = qd["K"]
    w_quad = qd["w_quad"]
    jacobian = qd["jacobian"]

    quad = orientation.quadrature_3d()
    n_m = qd["n_m"]; n_chi = qd["n_chi"]
    n_basis = n_m * n_chi
    init_L = jnp.zeros((n_basis, n_basis, n_basis))

    def scan_body(acc_L: Array, orient_pt: Array) -> tuple[Array, None]:
        cos_t1 = orient_pt[0]
        cos_t2 = orient_pt[1]
        dphi = orient_pt[2]
        w_orient = orient_pt[3]

        v_k = v_kick_at_orientation(m1, m2, chi1, chi2, cos_t1, cos_t2, dphi)
        P_ret = retention_sigmoid(v_k, theta.v_esc, theta.w_over_v_esc)
        integrand_no_basis = K * P_ret * jacobian * w_quad * w_orient
        contrib = _contract_loss_from_integrand(integrand_no_basis, qd)
        return acc_L + contrib, None

    L_total, _ = jax.lax.scan(scan_body, init_L, quad)
    return L_total


def build_loss_tensor(
    theta: RemnantTensorTheta,
    basis_solver: "TensorProductBasis",
    *,
    n_quad_factor: int = 2,
    orientation=None,
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

    The ``orientation`` parameter dispatches the same way as
    :func:`basis.remnant_tensor.build_remnant_tensor`: default
    :class:`Aligned` routes to the v1 unchanged path; any non-Aligned
    model engages a ``jax.lax.scan`` over the orientation quadrature.
    """
    if orientation is None or isinstance(orientation, Aligned):
        return _build_loss_tensor_aligned(theta, basis_solver, n_quad_factor)
    return _build_loss_tensor_marginalised(
        theta, basis_solver, orientation, n_quad_factor,
    )


def make_t_l_gain_loss(
    basis_solver: "TensorProductBasis",
    *,
    n_quad_factor: int = 2,
    orientation=None,
):
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
        T = build_remnant_tensor(
            theta, basis_solver,
            n_quad_factor=n_quad_factor,
            orientation=orientation,
        )
        return 0.5 * jnp.einsum("gab,a,b->g", T, coeffs_flat, coeffs_flat)

    def loss_fn(coeffs_flat: Array, theta: RemnantTensorTheta) -> Array:
        L = build_loss_tensor(
            theta, basis_solver,
            n_quad_factor=n_quad_factor,
            orientation=orientation,
        )
        return jnp.einsum("gab,a,b->g", L, coeffs_flat, coeffs_flat)

    return gain_fn, loss_fn


__all__ = [
    "build_loss_tensor",
    "make_t_l_gain_loss",
]
