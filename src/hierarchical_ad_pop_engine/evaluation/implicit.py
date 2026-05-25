r""":class:`ImplicitEvaluator` — point-query wrapper over an implicit solve.

Backing data is an :class:`ImplicitSolveResult` (the v2 grid Picard
solver's output) plus the :class:`CoagulationResponse` it was solved
from.  The two together are enough to answer every point query in the
:class:`RateDensityEvaluator` protocol without re-running the solver.

Design choices
--------------
* **Single-BH density at arbitrary ``(m, χ)`` uses
  :func:`hierarchical_ad_pop_engine.interp.log_density_at`** —
  bilinear interpolation in ``(log m, χ)`` of the stored ``log N*``.
  This is the same primitive that
  :meth:`CoagulationResponse.log_rate_density` uses for the existing
  4-D point evaluation, so the new evaluator is bit-equivalent to the
  current gwkokab path at the same query points.
* **4-D rate density uses point-K via
  :func:`log_pairing_kernel`** (not the precomputed grid kernel).
  Point queries are cheap one-offs; recomputing K analytically per
  call avoids any dependency on the solver's grid kernel
  precomputation and lets the evaluator be sliced freely with no
  hidden tensor materialisations.
* **1-D / 2-D marginals reuse the grid-quadrature path in
  :mod:`hierarchical_ad_pop_engine.observables`**, then interpolate
  the result onto the user's query points.  This guarantees that the
  new API agrees with the old API exactly at the engine grid points
  — a property the test suite pins as a regression invariant.
  Interpolation is in ``log m`` for mass-like axes (matching the
  geomspace grid layout used by ``log_density_at``) and linear in
  ``χ`` for spin axes.
* **Out-of-grid queries return 0** for densities, marginals, and the
  rate density — the same density-of-zero convention as
  :func:`log_density_at` (which returns ``-inf`` outside, so
  ``exp(...) = 0`` after :meth:`density`).

Sequence step 1 (this commit) implements the four grid-aligned 1-D
marginal axes (``'m1'``, ``'m2'``, ``'chi1'``, ``'chi2'``) and the
six grid-aligned 2-D pairs; derived axes (``'q'``, ``'chi_eff'``,
``'M_chirp'``, ``'m_heavy'``, ``'m_light'``) raise
:class:`NotImplementedError` with a pointer to the relevant
``observables`` helper.  They land in a follow-up commit alongside
the diagnostic-quadrature path required for arbitrary-point queries
on derived axes — see ``demo/PLAN_arbitrary_point_evaluation.md``
§ "Implicit (grid Picard) → marginal_1d".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.scipy.ndimage import map_coordinates

from .. import observables as _obs
from ..kernels import log_pairing_kernel

if TYPE_CHECKING:
    from ..response.coagulation import CoagulationResponse
    from ..response.implicit import ImplicitSolveResult


__all__ = ["ImplicitEvaluator"]


# ---------------------------------------------------------------------------
# Axis-aware interpolation helpers.
# ---------------------------------------------------------------------------
# The engine mass grid is geomspaced; densities and marginals interpolate
# more naturally in ``log m``.  The χ grid is linear.  Both axes return
# zero outside the grid range — same density-of-zero convention as
# ``log_density_at``.
# ---------------------------------------------------------------------------
_MASS_AXES = frozenset({_obs.AXIS_M1, _obs.AXIS_M2})
_CHI_AXES = frozenset({_obs.AXIS_CHI1, _obs.AXIS_CHI2})
_DERIVED_AXES = frozenset({"q", "chi_eff", "M_chirp", "m_heavy", "m_light"})


def _check_axis_supported(axis: str) -> None:
    r"""Gate derived axes with an actionable :class:`NotImplementedError`.

    Step-1 of ``PLAN_arbitrary_point_evaluation.md`` ships only the
    four grid-aligned axes — derived-axis support (``q``, ``chi_eff``,
    ``M_chirp``, ``m_heavy``, ``m_light``) requires an additional
    constrained-integral path (adaptive quadrature or fine internal
    grid + bilinear interp) and lands in a follow-up commit.  We
    raise here, *before* the underlying ``observables.marginal_1d``
    would raise ``ValueError``, so callers get a precise message that
    distinguishes "axis I have not yet implemented" from "axis label I
    do not recognise".
    """
    if axis in _DERIVED_AXES:
        raise NotImplementedError(
            f"axis {axis!r} is a derived axis (q / chi_eff / M_chirp / "
            f"m_heavy / m_light); ImplicitEvaluator step-1 supports only the "
            f"four grid-aligned axes {_obs.VALID_AXES!r}.  Use "
            f"observables.dN_dm1_dchi_eff or .dN_dm1_dq for the 2-D "
            f"derived-axis marginals today; arbitrary-point derived-axis "
            f"queries land in step-1-followup per PLAN_arbitrary_point_evaluation.md."
        )


def _axis_grid_and_query(axis: str, grid_m: Array, grid_chi: Array,
                         x_query: Array) -> Tuple[Array, Array, Array]:
    r"""Return ``(grid_coord, query_coord, valid_mask)`` for one axis.

    Mass axes are interpolated in ``log m``; spin axes are linear in
    ``χ``.  ``valid_mask`` flags query points strictly inside the
    grid range — outside points are zeroed by the caller.

    Derived axes are gated upstream by :func:`_check_axis_supported`;
    reaching this helper with an unrecognised axis indicates an
    internal logic error.
    """
    if axis in _MASS_AXES:
        gc = jnp.log(grid_m)
        qc = jnp.log(x_query)
    elif axis in _CHI_AXES:
        gc = grid_chi
        qc = x_query
    else:
        raise ValueError(
            f"unrecognised axis {axis!r}; expected one of {_obs.VALID_AXES!r}"
        )
    valid = (qc >= gc[0]) & (qc <= gc[-1])
    return gc, qc, valid


def _interp1_with_zero_outside(gc: Array, vals: Array,
                               qc: Array, valid: Array) -> Array:
    r"""``jnp.interp`` with zero outside the grid range (broadcast-safe)."""
    out = jnp.interp(qc, gc, vals)
    return jnp.where(valid, out, 0.0)


def _interp2_with_zero_outside(
    gc_a: Array, gc_b: Array, vals_2d: Array,
    qc_a: Array, qc_b: Array, valid: Array,
) -> Array:
    r"""Bilinear interpolation with zero outside the (a, b) rectangle.

    Mirrors :func:`hierarchical_ad_pop_engine.interp.log_density_at` —
    ``map_coordinates`` with ``order=1`` is the JAX-friendly bilinear
    primitive and handles arbitrary query shapes by stacking the
    fractional indices.
    """
    i = jnp.interp(qc_a, gc_a, jnp.arange(gc_a.size, dtype=vals_2d.dtype))
    j = jnp.interp(qc_b, gc_b, jnp.arange(gc_b.size, dtype=vals_2d.dtype))
    out = map_coordinates(
        vals_2d, jnp.stack([i, j], axis=0), order=1, mode="constant", cval=0.0,
    )
    return jnp.where(valid, out, 0.0)


# ---------------------------------------------------------------------------
# Evaluator.
# ---------------------------------------------------------------------------
class ImplicitEvaluator(eqx.Module):
    r"""Point-query evaluator over an :class:`ImplicitSolveResult`.

    Implements :class:`hierarchical_ad_pop_engine.evaluation.protocol.RateDensityEvaluator`.

    Parameters
    ----------
    result
        The converged Picard solve.  Stores ``N`` and ``log_N`` on the
        engine grid.
    response
        The :class:`CoagulationResponse` the implicit equation was
        solved from.  Carries the grid, kernel params, and
        ``log_R_env``.

    Notes
    -----
    The evaluator is an :class:`equinox.Module`, so it is a valid JAX
    pytree and can be passed across ``jax.jit`` boundaries.  Inference
    paths that prefer the existing grid-quadrature in
    :mod:`observables` remain unchanged.
    """

    result: "ImplicitSolveResult"
    response: "CoagulationResponse"

    # ---- 1. Single-BH density ---------------------------------------------
    def density(self, m: Array, chi: Array) -> Array:
        r"""``N(m, χ)`` by bilinear interp of ``N*`` in ``(log m, χ)``.

        Out-of-grid queries return 0 (an explicit inside mask).
        AD-traceable in every solver / response parameter via the
        existing implicit-function-theorem VJP on
        :func:`solve_implicit_fixed_point`.

        Notes
        -----
        We interpolate the **linear** ``N`` directly rather than going
        through :func:`interp.log_density_at` because the log-space
        path uses ``map_coordinates(..., mode='constant', cval=-inf)``
        and that produces ``0 · (-inf) = NaN`` for any query whose
        bilinear stencil straddles the upper grid boundary — including
        queries *exactly* at the last grid node (the bilinear weight
        on the out-of-bounds neighbour is zero, but the
        ``cval=-jnp.inf`` neighbour value still poisons the sum).
        Linear interp with ``cval=0.0`` has no such failure mode at
        the boundary, and is bit-equivalent to the log-space path for
        every strictly-interior query.
        """
        grid = self.response.grid
        N = self.result.N
        m_arr = jnp.asarray(m)
        chi_arr = jnp.asarray(chi)
        log_m = jnp.log(m_arr)
        i = jnp.interp(log_m, grid.log_m, jnp.arange(grid.log_m.size, dtype=N.dtype))
        j = jnp.interp(chi_arr, grid.chi, jnp.arange(grid.chi.size, dtype=N.dtype))
        out = map_coordinates(
            N, jnp.stack([i, j], axis=0),
            order=1, mode="constant", cval=0.0,
        )
        inside = (
            (log_m >= grid.log_m[0])
            & (log_m <= grid.log_m[-1])
            & (chi_arr >= grid.chi[0])
            & (chi_arr <= grid.chi[-1])
        )
        return jnp.where(inside, out, 0.0)

    # ---- 2. 4-D rate density ----------------------------------------------
    def rate_density_4d(
        self,
        m1: Array,
        m2: Array,
        chi1: Array,
        chi2: Array,
    ) -> Array:
        r"""``½ R_env K(m₁,m₂) N(m₁,χ₁) N(m₂,χ₂)`` at point ``(m₁,m₂,χ₁,χ₂)``.

        Matches :meth:`CoagulationResponse.log_rate_density` bit-for-bit
        at any common query point — same kernel formula, same
        bilinear-interp density.
        """
        kp = self.response.kernel_params
        log_K = log_pairing_kernel(m1, m2, **kp)
        N1 = self.density(m1, chi1)
        N2 = self.density(m2, chi2)
        R_env = jnp.exp(jnp.asarray(self.response.log_R_env))
        return 0.5 * R_env * jnp.exp(log_K) * N1 * N2

    # ---- 3. 1-D marginal ---------------------------------------------------
    def marginal_1d(self, axis: str, x: Array) -> Array:
        r"""``dC/dx`` at query points ``x``.

        Step-1 scope: ``axis ∈ {'m1','m2','chi1','chi2'}``.  Implemented
        as a two-layer reduction:

        1. Compute the full grid marginal via :func:`observables.marginal_1d`
           — exact grid-quadrature contraction of the 4-D integrand.
        2. Interpolate the resulting 1-D density at the user's query
           points (linear in ``log m`` for mass axes, linear in ``χ``
           for spin axes).

        This guarantees grid-point equivalence with the bulk
        :mod:`observables` path — pinned by the test suite as a
        regression invariant.
        """
        _check_axis_supported(axis)
        grid = self.response.grid
        x_grid, dC_dx = _obs.marginal_1d(
            self.result.N,
            grid,
            self.response.kernel_params,
            self.response.log_R_env,
            axis=axis,
        )
        gc, qc, valid = _axis_grid_and_query(axis, grid.m, grid.chi, jnp.asarray(x))
        # x_grid is grid.m or grid.chi (linear); convert to log m if mass.
        # We deliberately reuse _axis_grid_and_query for both gc (from
        # grid.m / grid.chi) and qc (from x_query) so they share the
        # same coordinate transform — same as log_density_at.
        return _interp1_with_zero_outside(gc, dC_dx, qc, valid)

    # ---- 4. 2-D marginal ---------------------------------------------------
    def marginal_2d(
        self,
        axes: Tuple[str, str],
        x_a: Array,
        x_b: Array,
    ) -> Array:
        r"""``dC/(dx_a dx_b)`` at query points.

        Same two-layer reduction as :meth:`marginal_1d`: grid-quadrature
        via :func:`observables.marginal_2d`, then bilinear interp at the
        query points.  ``axes`` ordering controls broadcast layout — we
        do not silently transpose.
        """
        grid = self.response.grid
        a, b = axes
        _check_axis_supported(a)
        _check_axis_supported(b)
        x_grid, y_grid, dC = _obs.marginal_2d(
            self.result.N,
            grid,
            self.response.kernel_params,
            self.response.log_R_env,
            axes=(a, b),
        )
        gc_a, qc_a, valid_a = _axis_grid_and_query(
            a, grid.m, grid.chi, jnp.asarray(x_a)
        )
        gc_b, qc_b, valid_b = _axis_grid_and_query(
            b, grid.m, grid.chi, jnp.asarray(x_b)
        )
        valid = valid_a & valid_b
        return _interp2_with_zero_outside(gc_a, gc_b, dC, qc_a, qc_b, valid)

    # ---- 5. Total rate -----------------------------------------------------
    def total_rate(self) -> Array:
        r"""Scalar ``∫∫∫∫ ½ R_env K N N dm₁ dm₂ dχ₁ dχ₂``.

        Pure delegate to :func:`observables.total_rate` — the
        evaluator's grid-quadrature contract is identical to the bulk
        path's contract.
        """
        return _obs.total_rate(
            self.result.N,
            self.response.grid,
            self.response.kernel_params,
            self.response.log_R_env,
        )
