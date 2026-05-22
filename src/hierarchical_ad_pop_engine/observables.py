r"""1-D and 2-D marginal merger-rate observables.

Pure-quadrature reductions of the per-environment forward merger-rate
density

.. math::

    \frac{dC_{\rm forward}}{dt\, dm_1\, dm_2\, d\chi_1\, d\chi_2}
        \;=\;
        \tfrac{1}{2}\, R_{\rm env}\, K(m_1, m_2)\,
        N(m_1, \chi_1)\, N(m_2, \chi_2)

— the same integrand that
:meth:`hierarchical_ad_pop_engine.response.coagulation.CoagulationResponse.log_rate_density`
returns *pointwise*.  This module integrates it over subsets of the
four-dimensional binary parameter space ``(m_1, m_2, \chi_1, \chi_2)``
to produce 1-D and 2-D marginal rate densities suitable for plotting
and for cross-validation against the particle-MC and basis solvers.

Conventions
-----------
* **Forward rate, no depletion / no retention factor.**  The forward
  rate ``½ R_env K N N`` is the LIGO-observable merger rate: every
  merger contributes whether or not its remnant is retained.
  ``P_retain`` enters only inside the implicit solve that produces
  ``N*`` (it shapes the steady-state density), it does *not* appear in
  the integrand here.  This matches
  :meth:`CoagulationResponse.log_rate_density` (which has no
  ``P_retain`` factor) and the user's "no depletion terms" phrasing.
* **Unordered binaries.**  The ``½`` prefactor corrects for the
  unordered-pair double-counting in the symmetric integrand
  ``K(m_1, m_2) N(m_1, \chi_1) N(m_2, \chi_2)``.  Marginals returned
  by this module are therefore *unordered* densities: integrating the
  full 4-D rate over all four axes gives the total merger rate, and
  each merger is counted once.  Use :func:`fold_ordered_m` (or your
  own ``Θ(m_1 − m_2)`` mask) if you want plots in the LVK ``m_1 ≥ m_2``
  convention; that flips the prefactor from ``½`` to a full count over
  the ordered triangle.
* **Quadrature.**  Trapezoidal on the response's ``BinaryGrid``,
  matching the implicit solver's
  :func:`response.implicit._dm_log_weights` and
  :func:`response.implicit._dchi_weights` so the total-rate identity

  .. math::

      \iiiint \tfrac{1}{2}\, R_{\rm env} K N N \, dm_1 dm_2 d\chi_1 d\chi_2

  reproduces the same number whether reduced as
  ``marginal_2d → integrate`` or ``marginal_1d → integrate`` or
  :func:`total_rate` directly (consistency tests pin this).

Scope
-----
This module is solver-agnostic.  Adapters
:func:`density_from_implicit` and :func:`density_from_basis` extract a
single-BH density ``N(m, \chi)`` on the engine ``BinaryGrid`` from the
v2 (grid Picard) and v3 (basis Picard) solve results respectively, so
the same marginal routines work for either path.  See the docstrings of
those adapters for the precise shapes and any basis-evaluation
considerations.

AD
--
Every function below is a JAX trace of grid operations only — no
Python branching on traced shapes, no ``.item()`` calls — so the full
pipeline (response → ``N*`` → marginal → loss) is end-to-end
differentiable.  Tests cover finite-difference vs ``jax.grad``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .grids import BinaryGrid
from .kernels import log_pairing_kernel_grid

if TYPE_CHECKING:
    from .basis.tensor import TensorProductBasis
    from .response.basis import BasisSolveResult
    from .response.coagulation import CoagulationResponse
    from .response.implicit import ImplicitSolveResult


__all__ = [
    "AXIS_M1",
    "AXIS_M2",
    "AXIS_CHI1",
    "AXIS_CHI2",
    "VALID_AXES",
    "trapezoid_weights",
    "rate_density_4d",
    "total_rate",
    "marginal_1d",
    "marginal_2d",
    "fold_ordered_m",
    "density_from_implicit",
    "density_from_basis",
    "marginals_from_implicit",
    "marginals_from_basis",
    "dN_dm1_dchi_eff",
    "dN_dm1_dq",
]


# ---------------------------------------------------------------------------
# Axis labels.
# ---------------------------------------------------------------------------
AXIS_M1 = "m1"
AXIS_M2 = "m2"
AXIS_CHI1 = "chi1"
AXIS_CHI2 = "chi2"
VALID_AXES = (AXIS_M1, AXIS_M2, AXIS_CHI1, AXIS_CHI2)


# ---------------------------------------------------------------------------
# Quadrature weights — same convention as response.implicit.
#
# We deliberately re-derive the weights here rather than importing the private
# helpers from :mod:`response.implicit`: the marginal module is a public
# observable layer and should not couple to a solver-internal symbol.  The
# definitions are identical (symmetric-trapezoid on ``log m``, symmetric-
# trapezoid on ``χ``) so :func:`total_rate` here matches the implicit solver's
# ``∫∫ … dm dχ`` arithmetic exactly.
# ---------------------------------------------------------------------------
def _dm_log_weights(grid_m: Array, grid_log_m: Array) -> Array:
    r"""``dm = m · d(log m)`` trapezoid weights on a geomspace grid."""
    dlog_m = jnp.diff(grid_log_m)
    dlog_m_full = jnp.concatenate(
        [dlog_m[:1] * 0.5, 0.5 * (dlog_m[:-1] + dlog_m[1:]), dlog_m[-1:] * 0.5]
    )
    return grid_m * dlog_m_full


def _dchi_weights(grid_chi: Array) -> Array:
    r"""``dχ`` trapezoid weights on a linear grid."""
    d = jnp.diff(grid_chi)
    return jnp.concatenate([d[:1] * 0.5, 0.5 * (d[:-1] + d[1:]), d[-1:] * 0.5])


def trapezoid_weights(grid: BinaryGrid) -> Tuple[Array, Array]:
    r"""Return ``(w_m, w_chi)`` trapezoid weights on the engine grid.

    These are the same symmetric-trapezoid weights the implicit solver
    uses for its ``∫∫ … dm dχ`` integrals, so marginals reduced with
    them satisfy the consistency identity ``∫ marginal_1d dx_i =
    total_rate``.

    Parameters
    ----------
    grid : BinaryGrid
        Engine binary grid.

    Returns
    -------
    w_m : Array, shape ``(N_m,)``
        Mass-axis weights with the ``dm = m · d(log m)`` Jacobian
        applied (i.e. ``∫ f(m) dm ≈ Σ w_m[i] · f(m_i)``).
    w_chi : Array, shape ``(N_chi,)``
        Spin-axis weights.
    """
    w_m = _dm_log_weights(grid.m, grid.log_m)
    w_chi = _dchi_weights(grid.chi)
    return w_m, w_chi


# ---------------------------------------------------------------------------
# 4-D forward-rate density on the engine grid.
# ---------------------------------------------------------------------------
def rate_density_4d(
    N_mchi: Array,
    grid: BinaryGrid,
    kernel_params: dict,
    log_R_env: Array | float = 0.0,
) -> Array:
    r"""Per-environment forward merger-rate density on the 4-D grid.

    .. math::

        R[i_1, j_1, i_2, j_2]
            = \tfrac{1}{2}\, R_{\rm env}\, K(m_{i_1}, m_{i_2})\,
              N(m_{i_1}, \chi_{j_1})\, N(m_{i_2}, \chi_{j_2})

    Parameters
    ----------
    N_mchi : Array, shape ``(N_m, N_chi)``
        Single-BH density on the engine grid.  Non-negative;
        constructed externally from whichever solver path produced
        it (see :func:`density_from_implicit`, :func:`density_from_basis`).
    grid : BinaryGrid
        Engine grid the density lives on.
    kernel_params : dict
        Pairing-kernel parameters; passed straight to
        :func:`hierarchical_ad_pop_engine.kernels.log_pairing_kernel_grid`.
    log_R_env : Array | float, default 0.0
        ``log(per-environment-per-time merger amplitude)``.  Matches
        :class:`CoagulationResponse.log_R_env`.

    Returns
    -------
    Array, shape ``(N_m, N_chi, N_m, N_chi)``
        ``R[i_1, j_1, i_2, j_2]``.  Note this is **not** the log; we
        return the linear density so quadrature is a simple
        ``einsum`` downstream.

    Notes
    -----
    Memory: at the default ``(N_m, N_chi) = (96, 32)`` and float64
    this is ``96·32·96·32·8 ≈ 75 MB`` — comfortable.  Marginals are
    contractions of this tensor; we do not materialise it inside
    :func:`marginal_1d` / :func:`marginal_2d` (they fuse the einsum
    with the integration weights), so the peak working-set for those
    routines stays below ~10 MB.  This function exists for callers
    that want the full tensor for plotting (e.g. ``dC/dm_1 dm_2``
    integrated only over the spin axes via custom slicing).
    """
    log_R_env = jnp.asarray(log_R_env)
    R_env = jnp.exp(log_R_env)
    K_mm = jnp.exp(log_pairing_kernel_grid(grid.m, **kernel_params))  # (N_m, N_m)
    # ½ R_env K(m1, m2) N(m1, χ1) N(m2, χ2)
    return 0.5 * R_env * (
        K_mm[:, None, :, None]
        * N_mchi[:, :, None, None]
        * N_mchi[None, None, :, :]
    )


# ---------------------------------------------------------------------------
# Total rate — ∫∫∫∫ R d^4 x.
# ---------------------------------------------------------------------------
def total_rate(
    N_mchi: Array,
    grid: BinaryGrid,
    kernel_params: dict,
    log_R_env: Array | float = 0.0,
) -> Array:
    r"""Total per-environment merger rate ``∫∫∫∫ ½ R_env K N N d^4 x``.

    Computed by a fused ``einsum`` over the 4-D rate density and the
    trapezoid weights — never materialises the ``(N_m, N_chi, N_m,
    N_chi)`` tensor.

    Returns
    -------
    Array, scalar
        Total per-environment forward merger rate.  Has the same
        ``R_env`` dependence and units convention as
        :meth:`CoagulationResponse.log_rate_density` (rate density;
        not unit-normalised).
    """
    w_m, w_chi = trapezoid_weights(grid)
    log_R_env = jnp.asarray(log_R_env)
    R_env = jnp.exp(log_R_env)
    K_mm = jnp.exp(log_pairing_kernel_grid(grid.m, **kernel_params))

    # ½ R_env · Σ_{i,j,k,l} K_{ik} N_{ij} N_{kl} w_i w_j w_k w_l
    # Reduce over χ first to get an (N_m, N_m) "density times weight" tensor,
    # then contract with K and the mass weights.
    N_int_chi = jnp.einsum("ij,j->i", N_mchi, w_chi)  # ∫ N(m, χ) dχ
    return 0.5 * R_env * jnp.einsum(
        "i,k,ik,i,k->", N_int_chi, N_int_chi, K_mm, w_m, w_m
    )


# ---------------------------------------------------------------------------
# 1-D marginals.
# ---------------------------------------------------------------------------
def _validate_axis(axis: str) -> None:
    if axis not in VALID_AXES:
        raise ValueError(
            f"axis must be one of {VALID_AXES!r}; got {axis!r}"
        )


def marginal_1d(
    N_mchi: Array,
    grid: BinaryGrid,
    kernel_params: dict,
    log_R_env: Array | float = 0.0,
    *,
    axis: str,
) -> Tuple[Array, Array]:
    r"""1-D marginal of the forward rate over three of the four axes.

    .. math::

        \frac{dC_{\rm forward}}{dx_i}(x_i)
            = \int \frac{dC_{\rm forward}}{dm_1 dm_2 d\chi_1 d\chi_2}
              \,\, \prod_{j \neq i} dx_j .

    Parameters
    ----------
    N_mchi : Array, shape ``(N_m, N_chi)``
        Single-BH density on the engine grid.
    grid : BinaryGrid
        Engine grid.
    kernel_params : dict
        Pairing-kernel parameters.
    log_R_env : Array | float, default 0.0
        Per-environment merger-amplitude log.
    axis : {'m1', 'm2', 'chi1', 'chi2'}
        Axis to keep — the marginal is evaluated *as a function of*
        this binary parameter.

    Returns
    -------
    x : Array
        Node values of the kept axis (``grid.m`` if mass-like,
        ``grid.chi`` if spin-like).
    dC_dx : Array, same shape as ``x``
        Marginal density ``dC_forward / dx_i`` at each node.

    Notes
    -----
    By the ``m_1 ↔ m_2`` and ``\chi_1 ↔ \chi_2`` symmetry of the
    forward-rate integrand, ``marginal_1d(..., axis='m1')`` and
    ``marginal_1d(..., axis='m2')`` are mathematically identical — and
    likewise for the two spin axes.  We compute them anyway via the
    same code path for diff cleanliness; the test suite pins the
    symmetry.
    """
    _validate_axis(axis)
    w_m, w_chi = trapezoid_weights(grid)
    log_R_env = jnp.asarray(log_R_env)
    R_env = jnp.exp(log_R_env)
    K_mm = jnp.exp(log_pairing_kernel_grid(grid.m, **kernel_params))

    # Precompute ∫ N(m, χ) dχ and ∫ N(m, χ) χ-weighted as needed.  All four
    # marginals can be written as a contraction over (m_i × χ_j × m_k × χ_l)
    # of the 4-D rate density with a "weight tensor" that has 1's on the
    # kept axis and the trapezoid weight on the others.  We do the
    # contractions axis-by-axis to keep memory under (N_m, N_m).
    N_int_chi = jnp.einsum("ij,j->i", N_mchi, w_chi)  # (N_m,)  ∫ N(m, χ) dχ

    if axis == AXIS_M1:
        # dC/dm1 = ½ R_env · [∫ N(m1, χ1) dχ1] · ∫ K(m1, m2) · [∫ N(m2, χ2) dχ2] dm2
        inner = jnp.einsum("ik,k,k->i", K_mm, N_int_chi, w_m)  # (N_m,)
        dC_dx = 0.5 * R_env * N_int_chi * inner
        return grid.m, dC_dx
    if axis == AXIS_M2:
        # Symmetric to m1 by m1 ↔ m2.
        inner = jnp.einsum("ik,i,i->k", K_mm, N_int_chi, w_m)  # (N_m,)
        dC_dx = 0.5 * R_env * N_int_chi * inner
        return grid.m, dC_dx
    if axis == AXIS_CHI1:
        # dC/dχ1 = ½ R_env · ∫∫∫ K(m1, m2) N(m1, χ1) N(m2, χ2) dm1 dm2 dχ2
        # Reduce over χ2 first, then over (m1, m2).
        # T[m1] = N(m1, χ1) ·  ∫ K(m1, m2) · [∫ N(m2, χ2) dχ2] dm2
        # Σ_{m1} w_m[m1] T[m1] is the χ1-marginal.
        inner_m1 = jnp.einsum("ik,k,k->i", K_mm, N_int_chi, w_m)  # (N_m,)
        dC_dx = 0.5 * R_env * jnp.einsum("ij,i,i->j", N_mchi, inner_m1, w_m)
        return grid.chi, dC_dx
    if axis == AXIS_CHI2:
        # Symmetric to χ1.
        inner_m2 = jnp.einsum("ik,i,i->k", K_mm, N_int_chi, w_m)  # (N_m,)
        dC_dx = 0.5 * R_env * jnp.einsum("kl,k,k->l", N_mchi, inner_m2, w_m)
        return grid.chi, dC_dx
    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# 2-D marginals.
# ---------------------------------------------------------------------------
def _axes_canonical(axes: Tuple[str, str]) -> Tuple[str, str]:
    r"""Return the input ``axes`` as a tuple, validating each entry.

    We do *not* sort the axes — the returned marginal carries the user's
    ordering so that ``marginal_2d(..., axes=('m1', 'chi1'))`` returns an
    array of shape ``(N_m, N_chi)`` with mass on the first axis, while
    ``axes=('chi1', 'm1')`` returns the transpose.  This keeps plot
    code unambiguous.
    """
    if not (isinstance(axes, tuple) and len(axes) == 2):
        raise ValueError(
            f"axes must be a length-2 tuple; got {axes!r}"
        )
    a, b = axes
    _validate_axis(a)
    _validate_axis(b)
    if a == b:
        raise ValueError(
            f"axes must be distinct; got ({a!r}, {b!r})"
        )
    return (a, b)


def _axis_values(axis: str, grid: BinaryGrid) -> Array:
    return grid.m if axis in (AXIS_M1, AXIS_M2) else grid.chi


def marginal_2d(
    N_mchi: Array,
    grid: BinaryGrid,
    kernel_params: dict,
    log_R_env: Array | float = 0.0,
    *,
    axes: Tuple[str, str],
) -> Tuple[Array, Array, Array]:
    r"""2-D marginal of the forward rate over the two unkept axes.

    .. math::

        \frac{dC_{\rm forward}}{dx_i dx_j}(x_i, x_j)
            = \int \frac{dC_{\rm forward}}{dm_1 dm_2 d\chi_1 d\chi_2}
              \,\, \prod_{k \notin \{i, j\}} dx_k .

    Parameters
    ----------
    N_mchi : Array, shape ``(N_m, N_chi)``
        Single-BH density.
    grid : BinaryGrid
    kernel_params : dict
    log_R_env : Array | float, default 0.0
    axes : tuple of two str
        Pair of axes to keep, e.g. ``('m1', 'm2')``, ``('m1', 'chi1')``.
        Order determines the output shape: first axis labels rows.

    Returns
    -------
    x : Array
        Node values for the first kept axis.
    y : Array
        Node values for the second kept axis.
    dC_dxdy : Array, shape ``(len(x), len(y))``
        Marginal density.

    Notes
    -----
    Six distinct pairs out of the C(4, 2) = 6 axis combinations are
    physically distinct only up to ``m_1 ↔ m_2`` / ``\chi_1 ↔ \chi_2``
    symmetries (e.g. ``(m_1, m_2)`` is symmetric, ``(m_1, \chi_1)``
    equals ``(m_2, \chi_2)``).  We compute the requested pair literally
    via the appropriate ``einsum`` contraction; the symmetry tests pin
    the equivalences.
    """
    a, b = _axes_canonical(axes)
    w_m, w_chi = trapezoid_weights(grid)
    log_R_env = jnp.asarray(log_R_env)
    R_env = jnp.exp(log_R_env)
    K_mm = jnp.exp(log_pairing_kernel_grid(grid.m, **kernel_params))
    half_R = 0.5 * R_env

    # All six pairs reduce to a contraction over the 4-D rate density
    # ``½ R_env K_{ik} N_{ij} N_{kl}`` with appropriate weight tensors.
    # We dispatch the pair as an unordered set ``frozenset({a, b})`` —
    # the contraction is symmetric in axes labelled (1) vs (2); whether
    # we transpose the result at the end determines the output ordering.
    pair = frozenset({a, b})

    if pair == frozenset({AXIS_M1, AXIS_M2}):
        # ∫∫ N(m1, χ1) N(m2, χ2) dχ1 dχ2 = [∫ N dχ](m1) · [∫ N dχ](m2)
        N_int_chi = jnp.einsum("ij,j->i", N_mchi, w_chi)  # (N_m,)
        dens = half_R * K_mm * N_int_chi[:, None] * N_int_chi[None, :]
        # rows = a, cols = b
        if a == AXIS_M1 and b == AXIS_M2:
            return grid.m, grid.m, dens
        return grid.m, grid.m, dens.T  # (m2, m1)

    if pair == frozenset({AXIS_CHI1, AXIS_CHI2}):
        # ∫∫ K(m1, m2) N(m1, χ1) N(m2, χ2) dm1 dm2
        #   = Σ_{m1, m2} K_{m1, m2} w_{m1} w_{m2} N_{m1, χ1} N_{m2, χ2}
        Kw = K_mm * w_m[:, None] * w_m[None, :]  # (N_m, N_m)
        # contract: rows = χ1, cols = χ2
        dens_chi1_chi2 = half_R * jnp.einsum(
            "ik,ij,kl->jl", Kw, N_mchi, N_mchi
        )
        if a == AXIS_CHI1 and b == AXIS_CHI2:
            return grid.chi, grid.chi, dens_chi1_chi2
        return grid.chi, grid.chi, dens_chi1_chi2.T

    if pair == frozenset({AXIS_M1, AXIS_CHI1}):
        # ∫∫ K(m1, m2) N(m2, χ2) dm2 dχ2  =  ∫ K(m1, m2) [∫ N(m2,χ2) dχ2] dm2
        N_int_chi = jnp.einsum("kl,l->k", N_mchi, w_chi)  # (N_m,)
        inner_m1 = jnp.einsum("ik,k,k->i", K_mm, N_int_chi, w_m)  # (N_m,)
        # dC/dm1 dχ1 = ½ R_env · N(m1, χ1) · inner(m1)
        dens_m1_chi1 = half_R * N_mchi * inner_m1[:, None]
        if a == AXIS_M1 and b == AXIS_CHI1:
            return grid.m, grid.chi, dens_m1_chi1
        return grid.chi, grid.m, dens_m1_chi1.T

    if pair == frozenset({AXIS_M2, AXIS_CHI2}):
        # Symmetric to (m1, χ1) by m1 ↔ m2 / χ1 ↔ χ2.
        N_int_chi = jnp.einsum("ij,j->i", N_mchi, w_chi)  # (N_m,) — over progenitor 1
        inner_m2 = jnp.einsum("ik,i,i->k", K_mm, N_int_chi, w_m)  # (N_m,)
        dens_m2_chi2 = half_R * N_mchi * inner_m2[:, None]
        if a == AXIS_M2 and b == AXIS_CHI2:
            return grid.m, grid.chi, dens_m2_chi2
        return grid.chi, grid.m, dens_m2_chi2.T

    if pair == frozenset({AXIS_M1, AXIS_CHI2}):
        # The interesting cross-pair: marginalises over (m2, χ1).
        # dC/dm1 dχ2 = ½ R_env · [∫ N(m1, χ1) dχ1] · ∫ K(m1, m2) N(m2, χ2) dm2
        N_int_chi_1 = jnp.einsum("ij,j->i", N_mchi, w_chi)  # (N_m,)
        # cross[m1, χ2] = ∫ K(m1, m2) N(m2, χ2) w_m[m2] dm2
        cross = jnp.einsum("ik,kl,k->il", K_mm, N_mchi, w_m)  # (N_m, N_chi)
        dens_m1_chi2 = half_R * N_int_chi_1[:, None] * cross
        if a == AXIS_M1 and b == AXIS_CHI2:
            return grid.m, grid.chi, dens_m1_chi2
        return grid.chi, grid.m, dens_m1_chi2.T

    if pair == frozenset({AXIS_M2, AXIS_CHI1}):
        # Symmetric to (m1, χ2).
        N_int_chi_2 = jnp.einsum("kl,l->k", N_mchi, w_chi)  # (N_m,)
        cross = jnp.einsum("ik,ij,i->kj", K_mm, N_mchi, w_m)  # (N_m_k, N_chi_j)
        dens_m2_chi1 = half_R * N_int_chi_2[:, None] * cross
        if a == AXIS_M2 and b == AXIS_CHI1:
            return grid.m, grid.chi, dens_m2_chi1
        return grid.chi, grid.m, dens_m2_chi1.T

    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Ordered-mass fold (for LVK m1 ≥ m2 plots).
# ---------------------------------------------------------------------------
def fold_ordered_m(
    m_axis: Array,
    dens_m1_m2: Array,
) -> Array:
    r"""Fold a symmetric ``dC/dm_1 dm_2`` onto the ``m_1 ≥ m_2`` triangle.

    The forward-rate marginal :func:`marginal_2d` with ``axes=('m1',
    'm2')`` returns an *unordered* density (symmetric in ``m_1 ↔ m_2``,
    each merger counted with the ``½`` prefactor).  For LVK-style plots
    the convention is ``m_1 ≥ m_2`` with each merger counted **once** —
    obtained by doubling the upper triangle (``m_1 > m_2``) and zeroing
    the lower triangle.

    The diagonal ``m_1 = m_2`` is *kept as-is*, not doubled: those are
    self-pairs in the unordered measure (one merger per cell, not two)
    and doubling would over-count by a factor of 2 on the diagonal.

    Parameters
    ----------
    m_axis : Array, shape ``(N_m,)``
        Mass axis values (used only to construct the ``m_1 ≥ m_2``
        mask; the density is not re-interpolated).
    dens_m1_m2 : Array, shape ``(N_m, N_m)``
        Unordered ``dC/dm_1 dm_2`` density (typical output of
        ``marginal_2d(..., axes=('m1', 'm2'))``).

    Returns
    -------
    Array, shape ``(N_m, N_m)``
        Ordered density with support on ``m_1 ≥ m_2``; zero elsewhere.
        ``∫∫_{m_1 ≥ m_2} dens dm_1 dm_2`` equals the total merger rate.
    """
    m1 = m_axis[:, None]
    m2 = m_axis[None, :]
    upper = m1 > m2
    diag = m1 == m2
    return jnp.where(upper, 2.0 * dens_m1_m2, jnp.where(diag, dens_m1_m2, 0.0))


# ---------------------------------------------------------------------------
# Solver-aware adapters.
# ---------------------------------------------------------------------------
def density_from_implicit(
    result: "ImplicitSolveResult",
) -> Array:
    r"""Extract ``N(m, χ)`` from an :class:`ImplicitSolveResult`.

    Identity adapter — the v2 implicit solver already stores ``N`` on
    the engine ``BinaryGrid`` as :attr:`ImplicitSolveResult.N`.  We
    keep this thin wrapper so callers can write the same line of code
    regardless of solver path::

        N_mchi = density_from_implicit(result)   # v2 grid Picard
        N_mchi = density_from_basis(result, response, basis_solver)  # v3 basis Picard
    """
    return result.N


def density_from_basis(
    result: "BasisSolveResult",
    response: "CoagulationResponse",
    basis_solver: "TensorProductBasis",
) -> Array:
    r"""Evaluate the v3 basis solution on the engine ``BinaryGrid``.

    The v3 basis solver stores coefficients on a Legendre-Lobatto
    tensor-product basis over ``(log m, χ)``.  Evaluating the
    expansion at the engine grid points produces an ``N(m, χ)`` array
    in the same layout as the v2 grid Picard solver, which the
    marginal routines can consume directly.

    Parameters
    ----------
    result : BasisSolveResult
        Output of :func:`response.basis.solve_basis_fixed_point`.
    response : CoagulationResponse
        Source of ``response.grid`` — the engine grid the marginals
        are evaluated on.  (We do not re-use the *solver* basis nodes
        for the marginals because the quadrature weights baked into
        :func:`trapezoid_weights` are matched to the engine grid, not
        the solver basis.)
    basis_solver : TensorProductBasis
        The same basis the solver ran on (needed to evaluate the
        expansion).

    Returns
    -------
    Array, shape ``(N_m, N_chi)``
        ``N(m, χ)`` evaluated at the engine grid points.  Clamped at 0
        below — basis expansions can ring slightly negative near
        sharp features (Gibbs) but a merger rate density must be
        non-negative.  The clamp is differentiable everywhere except
        the exact zero-crossing, which has measure zero under
        practical quadrature.

    Notes
    -----
    Evaluating the basis at the engine grid is ``O(N_m · n_basis_m +
    N_chi · n_basis_chi)`` per axis, so even for the default ``96 ×
    32`` engine grid and ``32 × 32`` solver basis the cost is well
    below a single Picard iteration.
    """
    coeffs_2d = result.coeffs.reshape(basis_solver.shape)
    grid = response.grid

    # Evaluate the basis on the Cartesian product of the engine grid
    # mass-log and spin axes.  TensorProductBasis.evaluate broadcasts
    # x_m and x_chi against each other, so feed grid-shaped arrays via
    # meshgrid to get an (N_m, N_chi) result.
    log_m_2d, chi_2d = jnp.meshgrid(grid.log_m, grid.chi, indexing="ij")
    N_evaluated = basis_solver.evaluate(coeffs_2d, log_m_2d, chi_2d)
    return jnp.maximum(N_evaluated, 0.0)


def _marginals_from_density(
    N_mchi: Array,
    response: "CoagulationResponse",
) -> dict:
    r"""Compute the canonical 1-D + 2-D marginal bundle from ``N(m, χ)``.

    Internal helper shared by :func:`marginals_from_implicit` and
    :func:`marginals_from_basis`.  Builds the four 1-D marginals
    (``m_1, m_2, χ_1, χ_2``) and the six 2-D pairs in a single dict
    keyed by axis labels.  Total rate is also returned for
    self-consistency checks.
    """
    grid = response.grid
    kp = response.kernel_params
    log_R = response.log_R_env

    out: dict = {
        "grid": grid,
        "total_rate": total_rate(N_mchi, grid, kp, log_R),
        "1d": {},
        "2d": {},
    }
    for ax in VALID_AXES:
        x, dC = marginal_1d(N_mchi, grid, kp, log_R, axis=ax)
        out["1d"][ax] = (x, dC)
    pairs = [
        (AXIS_M1, AXIS_M2),
        (AXIS_M1, AXIS_CHI1),
        (AXIS_M1, AXIS_CHI2),
        (AXIS_M2, AXIS_CHI1),
        (AXIS_M2, AXIS_CHI2),
        (AXIS_CHI1, AXIS_CHI2),
    ]
    for a, b in pairs:
        x, y, dC = marginal_2d(N_mchi, grid, kp, log_R, axes=(a, b))
        out["2d"][(a, b)] = (x, y, dC)
    return out


def marginals_from_implicit(
    result: "ImplicitSolveResult",
    response: "CoagulationResponse",
) -> dict:
    r"""Convenience: full marginal bundle from a v2 implicit solve.

    Equivalent to ``_marginals_from_density(result.N, response)``;
    exposes the same dict layout for both solvers so plotting code can
    treat them interchangeably.

    Returns
    -------
    dict
        Keys:

        - ``"grid"`` → :class:`BinaryGrid`
        - ``"total_rate"`` → scalar Array
        - ``"1d"`` → dict ``{axis_label → (x, dC_dx)}`` for all four
          axes in :data:`VALID_AXES`.
        - ``"2d"`` → dict ``{(a, b) → (x, y, dC_dxdy)}`` for all six
          distinct axis pairs (axis order matches the key order).
    """
    return _marginals_from_density(density_from_implicit(result), response)


def marginals_from_basis(
    result: "BasisSolveResult",
    response: "CoagulationResponse",
    basis_solver: "TensorProductBasis",
) -> dict:
    r"""Convenience: full marginal bundle from a v3 basis solve.

    Same return layout as :func:`marginals_from_implicit`.  Uses
    :func:`density_from_basis` to evaluate the basis expansion on the
    engine ``BinaryGrid`` first.
    """
    N_mchi = density_from_basis(result, response, basis_solver)
    return _marginals_from_density(N_mchi, response)


# ---------------------------------------------------------------------------
# Derived-axis observables — for the implicit-vs-MC observable comparison.
#
# These integrate the same 4-D forward-rate integrand ``½ R_env K N N``
# but bin into observables that are *not* axis-aligned on the engine
# grid: ``χ_eff = (m1 χ1 + m2 χ2) / (m1 + m2)`` and ``q = m_light /
# m_heavy``.  Linear-interpolation bin scatter keeps the result
# AD-safe in every parameter that flows through ``N``, ``log_R_env``,
# and the kernel params.
#
# Memory note: the 4-D integrand is built once at ``(N_m, N_chi, N_m,
# N_chi)`` (~75 MB at the default grid).  Smaller working-set is
# possible via ``jax.lax.scan`` over ``m_heavy`` rows, but adds
# complexity unwarranted at the default grid size.
# ---------------------------------------------------------------------------
def _bin_scatter_1d(
    bin_edges: Array, x_src: Array, w_src: Array
) -> Array:
    r"""Hard-histogram scatter onto a 1-D bin axis.

    Returns a length-``(len(bin_edges) - 1)`` array of accumulated
    weight per bin.  Each source point lands entirely in the bin
    ``[edges[i], edges[i+1])``; the upper edge ``x == edges[-1]`` is
    included in the last bin (closed interval at the top, matching the
    numpy ``np.histogram`` convention).

    AD note
    -------
    The derived-axis values (``χ_eff``, ``q``) at each cell are
    functions of the *fixed* engine grid points alone — independent of
    every inference parameter.  We therefore do not need a smoothly
    differentiable scatter in ``x_src``; gradient w.r.t. ``w_src``
    flows cleanly through :func:`jnp.zeros.at[i].add` and that is the
    only path inference cares about.  A bilinear / linear-interp
    scatter is possible but adds boundary subtleties (points exactly
    at the bin range edges land in a single neighbour, not two) for
    no inference benefit.

    Out-of-range points contribute zero.
    """
    n_bins = bin_edges.size - 1
    # ``searchsorted(edges, x, side='right') - 1`` gives the bin index
    # under the half-open convention ``[edges[i], edges[i+1])``.
    bin_idx = jnp.searchsorted(bin_edges, x_src, side="right") - 1
    # Include the right edge in the last bin (closed-on-the-right
    # convention) so ``x == edges[-1]`` (e.g. q = 1 exactly) is
    # counted.
    at_upper_edge = x_src == bin_edges[-1]
    bin_idx = jnp.where(at_upper_edge, n_bins - 1, bin_idx)
    inside = (x_src >= bin_edges[0]) & (x_src <= bin_edges[-1])
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)
    w = jnp.where(inside, w_src, 0.0)
    out = jnp.zeros((n_bins,), dtype=w.dtype)
    out = out.at[bin_idx.reshape(-1)].add(w.reshape(-1))
    return out


def _derived_axis_observable(
    N_mchi: Array,
    grid: BinaryGrid,
    kernel_params: dict,
    log_R_env: Array | float,
    *,
    derived_axis_values: Array,  # (N_m, N_chi, N_m, N_chi)
    derived_bin_edges: Array,
    ordered: bool,
) -> Array:
    r"""Internal: scatter the 4-D forward-rate integrand onto (m_heavy,
    derived_axis) bins.

    The derived axis is precomputed per 4-D cell by the caller (so the
    function is generic between ``χ_eff`` and ``q``).  Returns a
    ``(N_m, len(derived_bin_edges) - 1)`` array.
    """
    log_R_env = jnp.asarray(log_R_env)
    R_env = jnp.exp(log_R_env)
    K_mm = jnp.exp(log_pairing_kernel_grid(grid.m, **kernel_params))
    w_m, w_chi = trapezoid_weights(grid)

    # cell_volume[i, j, k, l] = w_m[i] w_chi[j] w_m[k] w_chi[l]
    cell_vol = (
        w_m[:, None, None, None]
        * w_chi[None, :, None, None]
        * w_m[None, None, :, None]
        * w_chi[None, None, None, :]
    )
    # rate_per_cell = ½ R_env K(m_i, m_k) N(m_i, χ_j) N(m_k, χ_l) · cell_vol.
    rate_per_cell = (
        0.5 * R_env
        * (K_mm[:, None, :, None]
           * N_mchi[:, :, None, None]
           * N_mchi[None, None, :, :])
        * cell_vol
    )

    # m_heavy[i, k] = max(m[i], m[k]); m_light = min.
    m_grid = grid.m
    m_i = m_grid[:, None]  # (N_m, 1)
    m_k = m_grid[None, :]  # (1, N_m)
    heavier_first = m_i >= m_k
    # We always bin on the *heavier* progenitor mass; the labels (1,
    # χ_1) and (2, χ_2) follow.  Under ``ordered=False`` we relax to
    # using m_i (first label) as the binning axis — for symmetric
    # observables (χ_eff, q is by definition m_light/m_heavy) this
    # changes only the bookkeeping, not the histogram values, because
    # the integrand and derived axis are themselves swap-symmetric.

    n_m = m_grid.size
    n_bins = derived_bin_edges.size - 1
    out = jnp.zeros((n_m, n_bins), dtype=rate_per_cell.dtype)

    # The natural way to bin into m_heavy is to upper-triangulate the
    # (m_i, m_k) plane: scatter the i ≥ k half twice (heavier label =
    # i), zero the rest.  Below we instead use the (i, k) plane in
    # full and a heavier-row mask, multiplied by 2 for off-diagonal
    # cells to reflect "two orderings, one merger event".  Diagonal
    # (i == k) keeps weight 1.
    #
    # This is equivalent to the unordered-pair count:
    #     ∫_{m1 ≥ m2} dC_{ordered}/(dm_heavy dχ_eff) dm_heavy dχ_eff
    #         = ∫∫∫∫ ½ R_env K N N d^4 x
    # (each unordered pair is counted as one ordered event under
    # m1≥m2; the ½ in the unordered integrand cancels against the 2×
    # over the i > k half).
    if ordered:
        # i > k:  off-diagonal — count twice (the unordered (i, k) and
        # (k, i) entries both map to the same ordered event); i == k:
        # diagonal — count once.
        is_diag = (m_i == m_k)
        upper = m_i > m_k
        mask = jnp.where(upper, 2.0, jnp.where(is_diag, 1.0, 0.0))
        weight_4d = rate_per_cell * mask[:, None, :, None]
        # The m_heavy axis is the first (i) axis; binning is trivial
        # in m_heavy (it IS the grid).  Aggregate the (k, χ_j, χ_l)
        # axes per (i, derived) bin.

        # Bin per i-row in derived axis.  We flatten (j, k, l) and
        # apply the 1-D scatter helper on the corresponding subset of
        # derived_axis_values.
        def _scatter_row(i: int) -> Array:
            x = derived_axis_values[i].reshape(-1)  # (N_chi · N_m · N_chi,)
            w = weight_4d[i].reshape(-1)
            return _bin_scatter_1d(derived_bin_edges, x, w)

        out = jax.vmap(_scatter_row)(jnp.arange(n_m))
        return out

    # ordered=False — use m_i directly as the binning axis (no
    # heavier-mask, no factor of 2).  This is the "unordered" version
    # of the histogram and integrates back to ½ ∫∫∫∫ K N N over the
    # full (m_i, χ_eff)-plane, with each unordered pair counted twice
    # (once at m1 = m_i, once at m2 = m_k).  Less natural for plots
    # but cheaper and useful for unit-testing the ordered version.
    def _scatter_row_unordered(i: int) -> Array:
        x = derived_axis_values[i].reshape(-1)
        w = rate_per_cell[i].reshape(-1)
        return _bin_scatter_1d(derived_bin_edges, x, w)

    return jax.vmap(_scatter_row_unordered)(jnp.arange(n_m))


def dN_dm1_dchi_eff(
    N_mchi: Array,
    grid: BinaryGrid,
    kernel_params: dict,
    log_R_env: Array | float = 0.0,
    *,
    chi_eff_bin_edges: Array,
    ordered: bool = True,
) -> tuple[Array, Array, Array]:
    r"""Forward-rate marginal in ``(m_1, χ_{\rm eff})`` space.

    .. math::

        \chi_{\rm eff} = \frac{m_1 \chi_1 + m_2 \chi_2}{m_1 + m_2}

    Bilinear-scatters the per-cell forward-rate integrand
    ``½ R_env K(m_i, m_k) N(m_i, χ_j) N(m_k, χ_l)`` (times cell volume)
    onto a histogram indexed by the engine grid's mass axis and a user-
    supplied ``χ_eff`` bin edge array.

    Parameters
    ----------
    N_mchi : Array, shape ``(N_m, N_chi)``
        Single-BH density on the engine grid.
    grid : BinaryGrid
        Engine grid.
    kernel_params : dict
    log_R_env : Array | float, default 0.0
    chi_eff_bin_edges : Array, shape ``(N_bins + 1,)``
        Bin edges on the χ_eff axis.  At Stage 1 (aligned only),
        χ_eff ∈ [0, 1]; reasonable default ``jnp.linspace(0, 1, 17)``.
    ordered : bool, default True
        If True, bin against the heavier-mass label ``m_1 ≥ m_2`` —
        the LVK convention.  Each unordered pair is counted once at
        ``m_1 = m_heavy``.  If False, bin against ``m_i`` directly —
        equivalent to "axis label 1 is just a label, no ordering";
        each unordered pair appears in the histogram twice (once as
        ``m_i = m_heavy``, once as ``m_i = m_light``).

    Returns
    -------
    m_axis : Array, shape ``(N_m,)``
        Engine mass-grid values.
    chi_eff_bin_centres : Array, shape ``(N_bins,)``
        χ_eff bin centres.
    dC : Array, shape ``(N_m, N_bins)``
        Forward-rate density ``dC_forward / dm_1 dχ_eff``.

    Notes
    -----
    Sanity invariant: integrating ``dC`` over the ``χ_eff`` axis with
    the bin widths reproduces :func:`marginal_1d` with
    ``axis='m1'`` (if ``ordered=False``) or twice it (if
    ``ordered=True``, because under the ordered convention each
    merger is counted at ``m_1 = m_heavy`` only).  The helper test
    ``test_redteam_observables_helpers.test_dN_dm1_dchi_eff_reduces_to_marginal_1d``
    pins this.
    """
    m_grid = grid.m
    chi_grid = grid.chi

    # χ_eff[i, j, k, l] = (m_i χ_j + m_k χ_l) / (m_i + m_k)
    m_i = m_grid[:, None, None, None]
    chi_j = chi_grid[None, :, None, None]
    m_k = m_grid[None, None, :, None]
    chi_l = chi_grid[None, None, None, :]
    chi_eff = (m_i * chi_j + m_k * chi_l) / (m_i + m_k)

    chi_eff_bin_edges_a = jnp.asarray(chi_eff_bin_edges)
    dC = _derived_axis_observable(
        N_mchi, grid, kernel_params, log_R_env,
        derived_axis_values=chi_eff,
        derived_bin_edges=chi_eff_bin_edges_a,
        ordered=ordered,
    )
    bin_widths = jnp.diff(chi_eff_bin_edges_a)
    bin_centres = 0.5 * (chi_eff_bin_edges_a[:-1] + chi_eff_bin_edges_a[1:])
    # Convert bin-summed weight → density on (m_1, χ_eff).
    # ``dC[i, j]`` is the integrated count in the m-grid row ``i`` and
    # χ_eff bin ``j``; dividing by ``w_m[i] · bin_width[j]`` returns a
    # proper density satisfying ``∫∫ dC_density dm_1 dχ_eff =
    # total_rate``.  The ``w_m`` divisor here matches the convention
    # used by :func:`marginal_1d` (which returns a density in ``m_1``,
    # not a per-grid-cell count).
    w_m, _ = trapezoid_weights(grid)
    dC_density = dC / (w_m[:, None] * bin_widths[None, :])
    return m_grid, bin_centres, dC_density


def dN_dm1_dq(
    N_mchi: Array,
    grid: BinaryGrid,
    kernel_params: dict,
    log_R_env: Array | float = 0.0,
    *,
    q_bin_edges: Array,
    ordered: bool = True,
) -> tuple[Array, Array, Array]:
    r"""Forward-rate marginal in ``(m_1, q)`` space.

    .. math::

        q = m_{\rm light} \,/\, m_{\rm heavy} \in (0, 1]

    Same machinery as :func:`dN_dm1_dchi_eff`.  ``ordered=True``
    is essentially mandatory here — ``q`` is itself defined by the
    heavier-mass convention — but we keep the flag for API symmetry;
    ``ordered=False`` produces the same histogram (``q`` is
    invariant under label swap by definition).

    Parameters
    ----------
    N_mchi, grid, kernel_params, log_R_env
        As in :func:`dN_dm1_dchi_eff`.
    q_bin_edges : Array, shape ``(N_bins + 1,)``
        ``q`` bin edges in ``(0, 1]``.  Recommend clipping the very-
        extreme-q tail (e.g. ``jnp.linspace(0.05, 1.0, 20)``) — both
        algorithms are noisy at the tiny-q end.
    ordered : bool, default True

    Returns
    -------
    m_axis : Array, shape ``(N_m,)``
    q_bin_centres : Array, shape ``(N_bins,)``
    dC : Array, shape ``(N_m, N_bins)``
        Forward-rate density ``dC_forward / dm_1 dq``.
    """
    m_grid = grid.m

    m_i = m_grid[:, None, None, None]
    m_k = m_grid[None, None, :, None]
    # q is symmetric in (i, k); chi axes don't enter.  Broadcast to
    # (N_m, N_chi, N_m, N_chi) so the scatter helper sees the right
    # shape.
    q_2d = jnp.minimum(m_i, m_k) / jnp.maximum(m_i, m_k)  # (N_m, 1, N_m, 1)
    n_chi = grid.chi.size
    q_4d = jnp.broadcast_to(q_2d, (m_grid.size, n_chi, m_grid.size, n_chi))

    q_bin_edges_a = jnp.asarray(q_bin_edges)
    dC = _derived_axis_observable(
        N_mchi, grid, kernel_params, log_R_env,
        derived_axis_values=q_4d,
        derived_bin_edges=q_bin_edges_a,
        ordered=ordered,
    )
    bin_widths = jnp.diff(q_bin_edges_a)
    bin_centres = 0.5 * (q_bin_edges_a[:-1] + q_bin_edges_a[1:])
    w_m, _ = trapezoid_weights(grid)
    dC_density = dC / (w_m[:, None] * bin_widths[None, :])
    return m_grid, bin_centres, dC_density
