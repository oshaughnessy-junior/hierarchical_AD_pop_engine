r"""Remnant-scatter projection tensor ``T(θ)`` for the v3 basis Picard solver.

Step 3d of ``demo/PLAN_engine_v3_basis.md``.  Builds the θ-dependent
rank-3 tensor ``T`` such that the bilinear gain density of the Picard
map projects onto the solver basis as

.. math::

    G_\gamma  \;=\;  \tfrac{1}{2} \, \sum_{\alpha,\beta}
                       n_\alpha \, n_\beta \, T_{\gamma;\alpha,\beta}(\theta).

Equivalently, in tensor-product index form ``α = (α_m, α_χ)``,
``β = (β_m, β_χ)``, ``γ = (γ_m, γ_χ)``, with the engine convention
that the natal density is represented as a function of ``(m, χ)`` on
the basis ``φ_α(\log m, χ) = P_{α_m}(\xi_{\log m}) \, P_{α_χ}(\xi_χ)``
(so coefficients have units of "density per ``dm`` per ``dχ``"; see
:mod:`basis.natal`):

.. math::

    T_{\gamma;\alpha,\beta}(\theta)
        \;=\; \frac{1}{M_{\gamma\gamma}}
              \int\!\!\!\int\!\!\!\int\!\!\!\int
                \varphi_\alpha(\log m_1, \chi_1) \,
                \varphi_\beta (\log m_2, \chi_2) \,
                K(m_1, m_2; \theta_K) \,
                P_{\rm ret}(m_1,m_2,\chi_1,\chi_2; v_{\rm esc}, w) \,
                \varphi_\gamma\bigl(\log M_{\rm rem}, \chi_{\rm rem}\bigr) \,
                \mathbb{1}_{\rm in\ domain}\,
                \mathrm{d}m_1 \mathrm{d}\chi_1 \mathrm{d}m_2 \mathrm{d}\chi_2

where ``M_{γγ}`` is the diagonal of the 2-D Lebesgue mass matrix of the
solver basis, the integration measure is ``dm dχ`` (so the Jacobian
``dm = m \, d\log m`` is carried explicitly inside the quadrature), and
``\mathbb{1}_{\rm in\ domain}`` is the indicator that
``(M_{\rm rem}, \chi_{\rm rem})`` falls inside the solver basis's
physical interval.  Out-of-domain remnants are dropped — this matches
v2's ``_bilinear_scatter`` behaviour and is the same conservative
truncation noted in ``PLAN_engine_v3_basis.md`` §3 as an existing
small leak at Stage 1 generation depths.

Why bake K and P_retain inside T
-------------------------------

The Step 3 architecture decision (memory ``engine-v3-basis-step-3-
architecture-locked``, Decision 1) is that ``T(θ)`` is θ-dependent
and rebuilt per call so that ``jax.grad`` cleanly flows through every
ingredient of the integrand.  We follow the same design here:

* **Kernel parameters** ``(a, b)`` of :func:`hierarchical_ad_pop_engine
  .kernels.log_pairing_kernel` enter through the integrand of ``T``.
* **Retention parameters** ``v_{\rm esc}`` and ``w_{\rm over\_v\_esc}``
  enter through the soft-sigmoid retention (see below).
* **Remnant-fit coefficients** sit inside :func:`hierarchical_ad_pop_engine
  .remnants.mass.M_rem` and :func:`...spin.chi_rem` and are
  currently hard-coded; the API is shaped so an eventual θ-fittable
  remnant model only needs to be threaded through the same call.

Soft-sigmoid retention (not the v2 Heaviside)
---------------------------------------------

Galerkin-projecting the Heaviside ``\mathbf{1}[v_{\rm kick} < v_{\rm esc}]``
produces fixed-amplitude Gibbs ringing in ``T`` regardless of basis
size — same failure mode as the deprecated log-projection from Step 2.
The plan (§Step 3, Decision 3) therefore replaces the cutoff with a
soft sigmoid

.. math::

    P_{\rm ret}(v_{\rm kick}; v_{\rm esc}, w)
        \;=\; \sigma\!\left(\frac{v_{\rm esc} - v_{\rm kick}}{w}\right),
        \qquad w = w_{\rm over\_v\_esc} \cdot v_{\rm esc}.

``w_{\rm over\_v\_esc}`` parameterises sharpness as a **fraction** of
``v_{\rm esc}``, never an absolute km/s number.  Stage 1 default
``v_{\rm esc} = 300\,{\rm km/s}``, ``w_{\rm over\_v\_esc} = 1.0``;
sweep range ``w_{\rm over\_v\_esc} \in [1.0, 2.0]`` per the Step 3c
acceptance.  Spot-checked under ``outputs/probe_retention_sigmoid.py``.

Quadrature
----------

We use a Legendre-Gauss-Lobatto product rule on the four progenitor
axes at overintegration factor ``n_quad_factor`` (default 2) relative
to the solver basis.  The plan calls for "Gauss-Lobatto quadrature at
order ≥ ``2 n_basis``" to maintain Galerkin orthogonality; at
``factor = 2`` and ``n_basis_solve = (32, 16)`` the four-axis tensor
has ``(64 · 32)^2 ≈ 4.2 · 10^6`` points, which is small compared to
the ``(32 · 16)^3 ≈ 1.3 · 10^8`` entries of ``T`` itself.

Memory at the chosen resolution
-------------------------------

``T`` has shape ``(n_basis, n_basis, n_basis)`` with
``n_basis = n_basis_m_solve · n_basis_χ_solve``.  At ``(32, 16)``,
``n_basis = 512``, ``T`` is ``≈ 1.3 · 10^8`` entries — 1.1 GB float64.
Per Step 3 §"Memory at chosen resolution".  Larger resolutions are
cheap to *build* but expensive to *store*; the brute-force ``C[N,N]``
fallback (per ``feedback_dont_straightjacket_basis``) replaces ``T``
when memory becomes binding.  The API consumer of ``T`` therefore
sees just a callable ``gain_fn(coeffs, theta) -> coeffs`` — *not*
a hard-coded T-tensor contraction — so a future ``BruteForceGain``
swap-in requires zero changes to the Picard loop.  See
:func:`make_t_tensor_gain` below.

What lives where
----------------

* :func:`build_remnant_tensor` — public entry point.  Builds ``T``
  given the solver basis and a ``RemnantTensorTheta`` bundle.
* :class:`RemnantTensorTheta` — frozen ``eqx.Module`` holding the
  θ subset that ``T`` depends on.  Splitting θ out of
  ``CoagulationResponse`` for now keeps the basis module physically
  separable from the v2 response container; the production response
  glue will pack ``theta`` from the same fields it already holds.
* :func:`retention_sigmoid` — soft-sigmoid retention, kept module-local
  so the v2 :func:`retention.P_retain` Heaviside is unaffected.
* :func:`make_t_tensor_gain` — produces a closure
  ``gain_fn(coeffs, theta)`` over the T-tensor contraction recipe,
  honouring the polymorphism constraint of
  ``feedback_dont_straightjacket_basis``.

Cross-references: ``PLAN_engine_v3_basis.md`` §Step 3; memories
``engine-v3-basis-step-3-architecture-locked``,
``dont-straightjacket-basis-api``, ``feedback_no_fudge_factors``,
``feedback_no_fake_physics_stand_ins``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..kernels import log_pairing_kernel
from ..remnants.kick import v_kick_aligned
from ..remnants.marginal import (
    chi_rem_at_orientation,
    v_kick_at_orientation,
)
from ..remnants.mass import M_rem
from ..remnants.spin import chi_rem
from ..spin_orientation import Aligned
from .legendre import LegendreLobatto1D, _legendre_recurrence


# ---------------------------------------------------------------------------
# Numpy mirror of legendre._lgl_nodes_weights.  Used at trace time only.
#
# Why a separate copy: the jnp version in legendre.py terminates early
# via ``float(jnp.max(jnp.abs(dx))) < 1e-15``.  That works eagerly (it
# runs once at LegendreLobatto1D construction) but breaks under
# ``jax.jit`` because ``dx`` is a tracer and ``float(...)`` raises
# ``ConcretizationTypeError``.  ``build_remnant_tensor`` calls into
# this for the progenitor quadrature axes inside a function that the
# Step 4 Picard loop will jit — so we resolve the LGL data with plain
# numpy, then ``jnp.asarray`` the result.  ``n_quad_m`` /
# ``n_quad_chi`` are statically known from the basis (which exposes
# ``n_basis`` as an ``eqx.field(static=True)``), so this side-steps
# the trace-time path cleanly without changing the eager LGL solver
# used by every other basis caller.
# ---------------------------------------------------------------------------
def _lgl_nodes_weights_np(n: int) -> tuple[np.ndarray, np.ndarray]:
    """LGL nodes and weights on the reference interval ``[-1, 1]``,
    computed in pure numpy.

    Bit-equivalent to :func:`legendre._lgl_nodes_weights` to ~ulp
    precision; the only differences are (a) pure numpy ops and (b) no
    ``jax`` tracer interaction.  Pinned to a small acceptance margin
    in the symmetry test below — if Newton ever drifts more than
    ~1e-15 across the two implementations the symmetry would loosen.
    """
    if n < 2:
        raise ValueError(f"LGL requires n >= 2, got n = {n}")
    if n == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    k_idx = np.arange(1, n - 1)
    x = -np.cos(np.pi * k_idx / (n - 1))

    def _legendre_np(degree: int, xx: np.ndarray) -> np.ndarray:
        out = [np.ones_like(xx)]
        if degree == 0:
            return np.stack(out, axis=0)
        out.append(xx.copy())
        for k in range(1, degree):
            nxt = ((2 * k + 1) * xx * out[-1] - k * out[-2]) / (k + 1)
            out.append(nxt)
        return np.stack(out, axis=0)

    for _ in range(100):
        P_all = _legendre_np(n - 1, x)
        P_nm1 = P_all[-1]
        P_nm2 = P_all[-2]
        dP = (n - 1) * (x * P_nm1 - P_nm2) / (x * x - 1.0)
        ddP = (2.0 * x * dP - (n - 1) * n * P_nm1) / (1.0 - x * x)
        dx = dP / ddP
        x = x - dx
        if float(np.max(np.abs(dx))) < 1e-15:
            break

    nodes = np.concatenate([[-1.0], x, [1.0]])
    P_nm1_at_nodes = _legendre_np(n - 1, nodes)[-1]
    weights = 2.0 / (n * (n - 1) * P_nm1_at_nodes * P_nm1_at_nodes)
    return nodes, weights

if TYPE_CHECKING:
    from .tensor import TensorProductBasis


# ---------------------------------------------------------------------------
# θ bundle.
# ---------------------------------------------------------------------------
class RemnantTensorTheta(eqx.Module):
    r"""Parameter subset on which the remnant-scatter tensor depends.

    Attributes
    ----------
    kernel_a, kernel_b : Array (scalar)
        Pairing-kernel exponents — see :func:`kernels.log_pairing_kernel`.
        Gradients of observables w.r.t. these flow through ``T(θ)``.
    v_esc : Array (scalar, km/s)
        Host escape velocity.  Enters the soft retention sigmoid.
    w_over_v_esc : Array (scalar)
        Retention sigmoid sharpness as a fraction of ``v_esc``.  Always
        passed as a *ratio*, never as an absolute km/s number, per the
        Step 3 Decision 3 rule.  Stage 1 default ``1.0``; useful sweep
        range ``[1.0, 2.0]``.

    Notes
    -----
    Remnant-fit coefficients (Buonanno-Kidder-Lehner for ``M_rem``,
    Barausse-Rezzolla for ``\chi_{\rm rem}``, Lousto-Zlochower for
    ``v_{\rm kick}``) are currently hard-coded in
    :mod:`hierarchical_ad_pop_engine.remnants`.  When Stage 2+ fits
    them, add fields here; :func:`build_remnant_tensor` will pick them
    up via the same closure.
    """

    kernel_a: Array
    kernel_b: Array
    v_esc: Array
    w_over_v_esc: Array


# ---------------------------------------------------------------------------
# Soft-sigmoid retention — module-local so the v2 Heaviside is unaffected.
# ---------------------------------------------------------------------------
def retention_sigmoid(
    v_kick: Array, v_esc: Array, w_over_v_esc: Array
) -> Array:
    r"""``σ((v_esc - v_kick) / w)`` with ``w = w_over_v_esc · v_esc``.

    AD-clean replacement for the Heaviside
    :func:`hierarchical_ad_pop_engine.retention.P_retain`.  At
    ``w_over_v_esc → 0`` it approaches the Heaviside.  See module
    docstring for the physical motivation of
    ``w_over_v_esc ∈ [1.0, 2.0]``.
    """
    w = w_over_v_esc * v_esc
    return jax.nn.sigmoid((v_esc - v_kick) / w)


# ---------------------------------------------------------------------------
# Internal helper: Legendre polynomial table at reference quadrature nodes.
# ---------------------------------------------------------------------------
def _legendre_table_for(
    basis_1d: LegendreLobatto1D, xi_quad: Array
) -> Array:
    r"""Evaluate ``P_0, …, P_{n-1}`` of ``basis_1d`` at reference
    quadrature nodes ``xi_quad`` (on ``[-1, 1]``).

    Returns shape ``(n_quad, n_basis)`` so that the contraction
    ``table @ coeffs`` evaluates the basis expansion at the quadrature
    nodes.
    """
    P_table = _legendre_recurrence(basis_1d.n_basis - 1, xi_quad)
    return P_table.T  # (n_quad, n_basis)


def _affine_to_ref(x: Array, a: float, b: float) -> Array:
    r"""Affine map ``x ∈ [a, b] → ξ ∈ [-1, 1]``."""
    return (2.0 * x - (a + b)) / (b - a)


# ---------------------------------------------------------------------------
# Build T.
# ---------------------------------------------------------------------------
def _build_remnant_tensor_quadrature_data(
    theta: RemnantTensorTheta,
    basis_solver: "TensorProductBasis",
    n_quad_factor: int,
):
    """Shared quadrature scaffolding for the aligned & marginalised paths.

    Returns a bundle of ``(m1, m2, chi1, chi2, K, w_quad, jacobian,
    a_m, b_m, a_chi, b_chi, n_m, n_chi, phi_m_q, phi_chi_q,
    M_diag_2d)`` — everything except the per-orientation physics
    (``v_kick``, ``chi_rem``, ``M_rem``, retention) which lives inside
    the per-path body.
    """
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

    phi_m_q = _legendre_table_for(basis_m, xi_m_q)
    phi_chi_q = _legendre_table_for(basis_chi, xi_chi_q)

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
        "a_m": a_m, "b_m": b_m, "a_chi": a_chi, "b_chi": b_chi,
        "n_m": n_m, "n_chi": n_chi,
        "phi_m_q": phi_m_q, "phi_chi_q": phi_chi_q,
        "M_diag_2d": M_diag_2d,
    }


def _contract_t_from_integrand(
    integrand_no_basis: Array,
    log_M_rem: Array,
    chi_rem_arr: Array,
    in_domain: Array,
    qd: dict,
) -> Array:
    """Apply the (P_gamma_m · P_gamma_chi)-weighted contraction over
    the 4-D progenitor quadrature to produce the dense ``T`` tensor.

    Identical to the original ``build_remnant_tensor`` contraction
    sequence; lifted out so the aligned and marginalised paths share
    it.
    """
    n_m = qd["n_m"]
    n_chi = qd["n_chi"]
    a_m = qd["a_m"]; b_m = qd["b_m"]
    a_chi = qd["a_chi"]; b_chi = qd["b_chi"]

    xi_M = _affine_to_ref(log_M_rem, a_m, b_m)
    xi_X = _affine_to_ref(chi_rem_arr, a_chi, b_chi)
    P_gamma_m = _legendre_recurrence(n_m - 1, xi_M)
    P_gamma_chi = _legendre_recurrence(n_chi - 1, xi_X)
    P_gamma_m = jnp.where(in_domain[None], P_gamma_m, 0.0)
    P_gamma_chi = jnp.where(in_domain[None], P_gamma_chi, 0.0)

    integrand_no_basis = integrand_no_basis * jnp.where(in_domain, 1.0, 0.0)
    integrand_g = (
        integrand_no_basis[None, None, ...]
        * P_gamma_m[:, None, ...]
        * P_gamma_chi[None, :, ...]
    )

    phi_m_q = qd["phi_m_q"]
    phi_chi_q = qd["phi_chi_q"]
    step_a = jnp.einsum("GgIJKL,Ia->GgaJKL", integrand_g, phi_m_q)
    step_b = jnp.einsum("GgaJKL,Jb->GgabKL", step_a, phi_chi_q)
    step_c = jnp.einsum("GgabKL,Kc->GgabcL", step_b, phi_m_q)
    T_dense = jnp.einsum("GgabcL,Ld->Ggabcd", step_c, phi_chi_q)

    M_diag_2d = qd["M_diag_2d"]
    T_dense = T_dense / M_diag_2d[:, :, None, None, None, None]
    n_basis = n_m * n_chi
    return T_dense.reshape(n_basis, n_basis, n_basis)


def _build_remnant_tensor_aligned(
    theta: RemnantTensorTheta,
    basis_solver: "TensorProductBasis",
    n_quad_factor: int,
) -> Array:
    """v1 aligned-only T-build — unchanged from pre-Session-B."""
    qd = _build_remnant_tensor_quadrature_data(theta, basis_solver, n_quad_factor)
    m1, m2, chi1, chi2 = qd["m1"], qd["m2"], qd["chi1"], qd["chi2"]

    heavier_first = m1 >= m2
    m1_c = jnp.where(heavier_first, m1, m2)
    m2_c = jnp.where(heavier_first, m2, m1)
    chi1_c = jnp.where(heavier_first, chi1, chi2)
    chi2_c = jnp.where(heavier_first, chi2, chi1)

    v_k = v_kick_aligned(m1_c, m2_c, chi1_c, chi2_c)
    P_ret = retention_sigmoid(v_k, theta.v_esc, theta.w_over_v_esc)
    M_rem_arr = M_rem(m1_c, m2_c, chi1_c, chi2_c)
    chi_rem_arr = chi_rem(m1_c, m2_c, chi1_c, chi2_c)

    log_M_rem = jnp.log(jnp.maximum(M_rem_arr, 1e-30))
    in_domain = (
        (log_M_rem >= qd["a_m"])
        & (log_M_rem <= qd["b_m"])
        & (chi_rem_arr >= qd["a_chi"])
        & (chi_rem_arr <= qd["b_chi"])
    )
    integrand_no_basis = qd["K"] * P_ret * qd["jacobian"] * qd["w_quad"]
    return _contract_t_from_integrand(
        integrand_no_basis, log_M_rem, chi_rem_arr, in_domain, qd,
    )


def _build_remnant_tensor_marginalised(
    theta: RemnantTensorTheta,
    basis_solver: "TensorProductBasis",
    orientation,
    n_quad_factor: int,
) -> Array:
    """Marginalised T-build via ``jax.lax.scan`` over orientation quadrature.

    One full-shape ``(n_basis, n_basis, n_basis)`` T-contribution alive
    per scan iteration (carry) plus the per-iteration integrand
    intermediates — peak memory the same order as the aligned build.
    Compute cost is ``n_orient × aligned-build``.  Decision 3 of
    ``demo/PLAN_misalignment_mode.md``.

    Each scan iteration evaluates :func:`remnants.kick_vector.v_kick_vec`
    and :func:`remnants.spin_vector.chi_rem_vec` at one orientation
    quadrature point, feeds the soft retention sigmoid (which keeps
    AD on ``v_esc`` everywhere — the basis side has always used
    sigmoid retention, this preserves that), and accumulates a
    ``w_orient``-weighted T-contribution.
    """
    qd = _build_remnant_tensor_quadrature_data(theta, basis_solver, n_quad_factor)
    m1, m2, chi1, chi2 = qd["m1"], qd["m2"], qd["chi1"], qd["chi2"]
    K = qd["K"]
    w_quad = qd["w_quad"]
    jacobian = qd["jacobian"]
    a_m = qd["a_m"]; b_m = qd["b_m"]
    a_chi = qd["a_chi"]; b_chi = qd["b_chi"]

    quad = orientation.quadrature_3d()  # (n_orient, 4)
    n_m = qd["n_m"]; n_chi = qd["n_chi"]
    n_basis = n_m * n_chi
    init_T = jnp.zeros((n_basis, n_basis, n_basis))

    def scan_body(acc_T: Array, orient_pt: Array) -> tuple[Array, None]:
        cos_t1 = orient_pt[0]
        cos_t2 = orient_pt[1]
        dphi = orient_pt[2]
        w_orient = orient_pt[3]

        chi1_z = chi1 * cos_t1
        chi2_z = chi2 * cos_t2
        M_k = M_rem(m1, m2, chi1_z, chi2_z)
        # |χ_rem| at this orientation (chi_rem_vec canonicalises m1≥m2
        # internally).
        chi_rem_k = chi_rem_at_orientation(m1, m2, chi1, chi2, cos_t1, cos_t2, dphi)
        v_k = v_kick_at_orientation(m1, m2, chi1, chi2, cos_t1, cos_t2, dphi)
        P_ret = retention_sigmoid(v_k, theta.v_esc, theta.w_over_v_esc)

        log_M_rem = jnp.log(jnp.maximum(M_k, 1e-30))
        in_domain = (
            (log_M_rem >= a_m)
            & (log_M_rem <= b_m)
            & (chi_rem_k >= a_chi)
            & (chi_rem_k <= b_chi)
        )
        integrand_no_basis = K * P_ret * jacobian * w_quad * w_orient
        contrib = _contract_t_from_integrand(
            integrand_no_basis, log_M_rem, chi_rem_k, in_domain, qd,
        )
        return acc_T + contrib, None

    T_total, _ = jax.lax.scan(scan_body, init_T, quad)
    return T_total


def build_remnant_tensor(
    theta: RemnantTensorTheta,
    basis_solver: "TensorProductBasis",
    *,
    n_quad_factor: int = 2,
    orientation=None,
) -> Array:
    r"""Build the gain projection tensor ``T_{γ;α,β}(θ)``.

    Parameters
    ----------
    theta : RemnantTensorTheta
        θ subset that ``T`` depends on.  Gradients of observables w.r.t.
        any field here flow through this call via the standard JAX
        machinery (no special ``custom_vjp`` needed at the build site;
        the Step 4 IFT VJP differentiates the *fixed point* through
        ``T(θ)``).
    basis_solver : TensorProductBasis
        Tensor-product basis on ``(\log m, \chi)`` at the *solver*
        resolution.  The natal pipeline at the (finer) density
        resolution restricts to this basis via
        :py:meth:`TensorProductBasis.restrict_to` before ``T`` is
        contracted in.
    n_quad_factor : int, default 2
        Overintegration factor.  Quadrature node count per progenitor
        axis is ``n_quad_factor · n_basis_axis``.  The plan calls for
        ``≥ 2 n_basis``; the default satisfies that.  Bump to 3 if the
        Step 3c sweep shows residual sensitivity to retention
        sharpness.
    orientation : OrientationModel, optional
        Spin-orientation distribution.  Defaults to :class:`Aligned` —
        the v1 aligned-only T-build, unchanged from pre-Session-B.
        Any non-Aligned model engages a ``jax.lax.scan`` over the
        flat 3-D orientation quadrature, evaluating the full vector
        BR / LZ physics at each orientation point and accumulating a
        ``w_orient``-weighted T-contribution.  Build cost scales
        ``n_orient × aligned``; peak memory unchanged (one
        T-contribution alive per scan iteration plus the running
        accumulator).  Decision 3 of
        ``demo/PLAN_misalignment_mode.md``.

    Returns
    -------
    Array of shape ``(n_basis, n_basis, n_basis)`` where
    ``n_basis = n_basis_m · n_basis_χ``.  The flat index ordering on
    every axis is ``α_m · n_basis_χ + α_χ`` — same as
    :py:meth:`TensorProductBasis.flatten`.

    Gradient signature: AD-clean in every field of ``theta``, and also
    in ``orientation.kappa`` when ``orientation`` is :class:`Fisher`
    (the Fisher quadrature weights are smooth in ``κ``).

    Memory
    ------
    At default Stage 1 resolution ``(32, 16)`` this allocates a
    ``(512, 512, 512)`` float64 array (~1.1 GB).  Use the
    :func:`make_t_tensor_gain` wrapper if you only need the
    contracted ``gain_fn`` and want to potentially swap to a
    brute-force backend later (see module docstring).
    """
    if orientation is None or isinstance(orientation, Aligned):
        return _build_remnant_tensor_aligned(theta, basis_solver, n_quad_factor)
    return _build_remnant_tensor_marginalised(
        theta, basis_solver, orientation, n_quad_factor,
    )


# ---------------------------------------------------------------------------
# Gain closure — keeps the Picard loop polymorphic over the gain backend.
# ---------------------------------------------------------------------------
def make_t_tensor_gain(
    basis_solver: "TensorProductBasis",
    *,
    n_quad_factor: int = 2,
    orientation=None,
):
    r"""Return ``gain_fn(coeffs_flat, theta) -> coeffs_flat`` using the
    T-tensor recipe.

    The returned closure is what the v3 Picard map consumes.  It
    encapsulates the T-tensor contraction so that a future
    ``BruteForceGain`` swap-in (see
    ``feedback_dont_straightjacket_basis``) requires zero changes to
    the Picard loop or the IFT VJP.

    Parameters
    ----------
    basis_solver, n_quad_factor
        Forwarded to :func:`build_remnant_tensor`.
    orientation : OrientationModel, optional
        Spin-orientation distribution; forwarded.  Defaults to
        :class:`Aligned` (v1 behaviour).

    Returns
    -------
    callable ``(coeffs_flat, theta) -> coeffs_flat``
        Computes ``G_γ = (1/2) Σ_{α,β} n_α n_β T_{γ;α,β}(theta)`` on
        flat coefficient vectors.  ``coeffs_flat`` has shape
        ``(basis_solver.n_basis,)``.

    Notes
    -----
    The factor of ``1/2`` (unordered-pair correction) lives **here**,
    not inside ``T``.  This matches the Step 4 signature in
    ``PLAN_engine_v3_basis.md`` and matches the implicit-solver
    ``gain_density`` factor-1/2 convention.
    """

    def gain_fn(coeffs_flat: Array, theta: RemnantTensorTheta) -> Array:
        T = build_remnant_tensor(
            theta, basis_solver,
            n_quad_factor=n_quad_factor,
            orientation=orientation,
        )
        # G_γ = 1/2 Σ_α,β T_{γ;α,β} · n_α · n_β
        return 0.5 * jnp.einsum("gab,a,b->g", T, coeffs_flat, coeffs_flat)

    return gain_fn


__all__ = [
    "RemnantTensorTheta",
    "build_remnant_tensor",
    "make_t_tensor_gain",
    "retention_sigmoid",
]
