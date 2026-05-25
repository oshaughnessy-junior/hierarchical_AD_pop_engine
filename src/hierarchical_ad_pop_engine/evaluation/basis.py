r""":class:`BasisEvaluator` — point-query wrapper over a v3 basis solve.

Backing data is a :class:`BasisSolveResult` (the v3 Legendre-Lobatto
Galerkin solver's output), the :class:`CoagulationResponse` it was
solved from, and the :class:`TensorProductBasis` the solver ran on.
The three together are enough to answer every point query in the
:class:`RateDensityEvaluator` protocol without re-running the solver.

Design choices
--------------
* **Single-BH density at arbitrary ``(m, χ)`` uses
  :py:meth:`TensorProductBasis.evaluate`** on ``(log m, χ)`` — the
  basis lives on log-mass and linear χ.  Out-of-domain queries return
  0.  The evaluated density is also clamped at 0 below — modal
  expansions can ring slightly negative near sharp features (Gibbs),
  but a merger rate density must be non-negative.  Same convention as
  :func:`observables.density_from_basis`.

* **4-D rate density recomputes K analytically per call** via
  :func:`hierarchical_ad_pop_engine.kernels.log_pairing_kernel`, then
  multiplies the two single-BH densities.  We deliberately do **not**
  route through the basis Galerkin tensors ``T_{γαβ}`` or ``L_{γα}``
  — those are appropriate for the solver inner loop (where their
  quadrature accuracy is the cost we pay for fixed-point closure) but
  here we want the evaluator's systematic to be cleanly separable from
  the solver's quadrature systematic.  See
  ``demo/PLAN_arbitrary_point_evaluation.md`` § "Open-question
  resolutions" — this was an explicit user direction.

* **1-D / 2-D marginals and total rate are a two-layer reduction.**
  Layer 1: evaluate the basis on the engine ``BinaryGrid`` (via the
  inlined :func:`density_from_basis` body — kept self-contained so the
  diff against :class:`ImplicitEvaluator` is line-aligned).  Layer 2:
  delegate to :func:`observables.marginal_1d` / ``marginal_2d`` /
  ``total_rate`` for the contraction and to ``jnp.interp`` /
  ``map_coordinates`` for the query-point interpolation.  This
  guarantees grid-point equivalence with both
  :class:`ImplicitEvaluator` (when run on the same response) and
  :func:`observables.marginals_from_basis` — pinned by the test
  suite as a regression invariant.

* **Out-of-grid queries return 0** for densities, marginals, and the
  rate density — same density-of-zero convention as
  :class:`ImplicitEvaluator`.

Why the interp helpers are *not* shared with :mod:`evaluation.implicit`
---------------------------------------------------------------------
The mass / χ axis-coordinate helpers (``_axis_grid_and_query``,
``_interp1_with_zero_outside``, ``_interp2_with_zero_outside``) are
deliberately duplicated from :mod:`evaluation.implicit` rather than
hoisted into a shared private module.  Step 1 (implicit) chose
locality, and matching that pattern keeps the diff between the two
evaluators short and reviewable.  If Step 3 (MC) needs the same
helpers, that's the right time to factor them out — three call sites
beat two.

Latent ``interp.log_density_at`` NaN — not relevant here
--------------------------------------------------------
:class:`ImplicitEvaluator` documents a workaround for an upper-grid-
boundary NaN bug in :func:`interp.log_density_at`.  The basis
evaluator's :meth:`density` does **not** route through that primitive
— it uses :py:meth:`TensorProductBasis.evaluate` directly — and the
marginal/total paths use :func:`jnp.interp` (no ``cval=-inf``) and
:func:`map_coordinates` with ``cval=0.0``.  The latent NaN cannot
surface through this evaluator, so the upstream fix stays deferred
without consequence.  See memory ``project_log_density_at_boundary_nan``
for the eventual upstream change.

Sequence step 2 (this commit) implements the four grid-aligned 1-D
marginal axes (``'m1'``, ``'m2'``, ``'chi1'``, ``'chi2'``) and the
six grid-aligned 2-D pairs; derived axes (``'q'``, ``'chi_eff'``,
``'M_chirp'``, ``'m_heavy'``, ``'m_light'``) raise
:class:`NotImplementedError` with a pointer to the relevant
``observables`` helper.  They land in a follow-up commit alongside
the corresponding implicit-side derived-axis path — see
``demo/PLAN_arbitrary_point_evaluation.md`` § "Basis (v3 Galerkin) →
marginal_1d".
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
    from ..basis.tensor import TensorProductBasis
    from ..response.basis import BasisSolveResult
    from ..response.coagulation import CoagulationResponse


__all__ = ["BasisEvaluator"]


# ---------------------------------------------------------------------------
# Axis-aware interpolation helpers (mirror of evaluation/implicit.py).
# ---------------------------------------------------------------------------
# Same convention as the implicit evaluator and ``interp.log_density_at``:
# mass-like axes interpolate in ``log m`` (the engine grid is geomspaced),
# spin-like axes interpolate linearly in ``χ``.  Out-of-grid queries
# return 0 (density-of-zero convention).
# ---------------------------------------------------------------------------
_MASS_AXES = frozenset({_obs.AXIS_M1, _obs.AXIS_M2})
_CHI_AXES = frozenset({_obs.AXIS_CHI1, _obs.AXIS_CHI2})
_DERIVED_AXES = frozenset({"q", "chi_eff", "M_chirp", "m_heavy", "m_light"})


def _check_axis_supported(axis: str) -> None:
    r"""Gate derived axes with an actionable :class:`NotImplementedError`.

    Step-2 of ``PLAN_arbitrary_point_evaluation.md`` ships only the
    four grid-aligned axes — derived-axis support (``q``, ``chi_eff``,
    ``M_chirp``, ``m_heavy``, ``m_light``) requires an additional
    constrained-integral path (adaptive quadrature or fine internal
    grid + bilinear interp) and lands in a follow-up commit, in
    lockstep with the implicit-side derived-axis path.  We raise
    here, *before* the underlying ``observables.marginal_1d`` would
    raise ``ValueError``, so callers get a precise message that
    distinguishes "axis I have not yet implemented" from "axis label I
    do not recognise".
    """
    if axis in _DERIVED_AXES:
        raise NotImplementedError(
            f"axis {axis!r} is a derived axis (q / chi_eff / M_chirp / "
            f"m_heavy / m_light); BasisEvaluator step-2 supports only the "
            f"four grid-aligned axes {_obs.VALID_AXES!r}.  Use "
            f"observables.dN_dm1_dchi_eff or .dN_dm1_dq for the 2-D "
            f"derived-axis marginals today; arbitrary-point derived-axis "
            f"queries land in step-2-followup per PLAN_arbitrary_point_evaluation.md."
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

    ``map_coordinates`` with ``order=1`` is the JAX-friendly bilinear
    primitive and handles arbitrary query shapes by stacking the
    fractional indices.  ``cval=0.0`` (not ``-inf``) — this is the
    linear-N path, deliberately, so no boundary NaN can leak in.
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
class BasisEvaluator(eqx.Module):
    r"""Point-query evaluator over a :class:`BasisSolveResult`.

    Implements :class:`hierarchical_ad_pop_engine.evaluation.protocol.RateDensityEvaluator`.

    Parameters
    ----------
    result
        Converged basis Picard solve.  Stores ``coeffs`` in flat layout;
        :py:meth:`density` reshapes to ``basis_solver.shape``.
    response
        The :class:`CoagulationResponse` the basis equation was solved
        from.  Carries the engine grid, kernel params, and ``log_R_env``.
    basis_solver
        The same :class:`TensorProductBasis` the solver ran on.
        Required to evaluate the modal expansion at arbitrary points
        and (via :py:meth:`shape`) to reshape the flat coefficient
        vector to its 2-D form.

    Notes
    -----
    The evaluator is an :class:`equinox.Module`, so it is a valid JAX
    pytree and can be passed across ``jax.jit`` boundaries.  Inference
    paths that prefer the existing grid-quadrature in
    :mod:`observables` (via :func:`observables.density_from_basis`)
    remain unchanged — this evaluator is additive.
    """

    result: "BasisSolveResult"
    response: "CoagulationResponse"
    basis_solver: "TensorProductBasis"

    # ---- 1. Single-BH density ---------------------------------------------
    def density(self, m: Array, chi: Array) -> Array:
        r"""``N(m, χ)`` from the basis modal expansion.

        Evaluates the tensor-product basis at ``(log m, χ)`` — the
        component basis on the mass-like axis lives on log-mass, the χ
        basis is linear.  Out-of-domain queries return 0 (an explicit
        inside mask against the basis rectangles).  The evaluated
        density is clamped at 0 below to match the non-negativity
        convention used by :func:`observables.density_from_basis`
        (Gibbs ringing near sharp features can give modestly negative
        values; rate densities must be non-negative).

        AD-traceable in every solver / response parameter via the
        basis-solver's implicit-function-theorem VJP on
        :func:`solve_basis_fixed_point`.
        """
        coeffs_2d = self.result.coeffs.reshape(self.basis_solver.shape)
        m_arr = jnp.asarray(m)
        chi_arr = jnp.asarray(chi)
        log_m = jnp.log(m_arr)

        # Basis domain rectangles — the basis is defined only inside
        # ``[a_m, b_m] × [a_χ, b_χ]``; the Legendre recurrence extrapolates
        # outside but with rapidly amplifying error.  Mask hard.
        #
        # Tolerance: engine-grid endpoints that *should* equal the basis
        # domain bounds can differ by float ULP slop (e.g.
        # ``jnp.linspace(1e-3, 1-1e-3, n)[-1]`` is not bit-exact ``1-1e-3``
        # for arbitrary ``n``).  Without a tolerance the inside mask
        # spuriously zeros the corner cells, breaking grid-point
        # equivalence with :func:`observables.density_from_basis` (which
        # does no masking).  Convention matches
        # :py:meth:`TensorProductBasis._check_same_domain`.
        a_m, b_m = self.basis_solver.basis_m.domain
        a_c, b_c = self.basis_solver.basis_chi.domain
        _RTOL = 1e-12
        _ATOL = 1e-15
        eps_m = _RTOL * max(abs(a_m), abs(b_m), 1.0) + _ATOL
        eps_c = _RTOL * max(abs(a_c), abs(b_c), 1.0) + _ATOL

        N_raw = self.basis_solver.evaluate(coeffs_2d, log_m, chi_arr)
        inside = (
            (log_m >= a_m - eps_m)
            & (log_m <= b_m + eps_m)
            & (chi_arr >= a_c - eps_c)
            & (chi_arr <= b_c + eps_c)
        )
        return jnp.where(inside, jnp.maximum(N_raw, 0.0), 0.0)

    # ---- 2. 4-D rate density ----------------------------------------------
    def rate_density_4d(
        self,
        m1: Array,
        m2: Array,
        chi1: Array,
        chi2: Array,
    ) -> Array:
        r"""``½ R_env K(m₁,m₂) N(m₁,χ₁) N(m₂,χ₂)`` — K analytic, not Galerkin.

        Recomputes the pairing kernel pointwise via
        :func:`log_pairing_kernel`; does **not** route through the basis
        Galerkin tensors ``T``/``L``.  Keeps the evaluator's systematic
        cleanly separable from the solver's quadrature systematic —
        see ``demo/PLAN_arbitrary_point_evaluation.md`` § "Open-question
        resolutions" for the user's direction on this.
        """
        kp = self.response.kernel_params
        log_K = log_pairing_kernel(m1, m2, **kp)
        N1 = self.density(m1, chi1)
        N2 = self.density(m2, chi2)
        R_env = jnp.exp(jnp.asarray(self.response.log_R_env))
        return 0.5 * R_env * jnp.exp(log_K) * N1 * N2

    # ---- internal: ``N`` on the engine grid -------------------------------
    def _N_on_engine_grid(self) -> Array:
        r"""Evaluate the basis at the engine ``BinaryGrid`` mesh.

        Bit-equivalent to :func:`observables.density_from_basis` — we
        inline the body here so the evaluator is self-contained for diff
        review against :class:`ImplicitEvaluator` (which carries
        ``result.N`` directly on the engine grid).
        """
        coeffs_2d = self.result.coeffs.reshape(self.basis_solver.shape)
        grid = self.response.grid
        log_m_2d, chi_2d = jnp.meshgrid(grid.log_m, grid.chi, indexing="ij")
        N_eval = self.basis_solver.evaluate(coeffs_2d, log_m_2d, chi_2d)
        return jnp.maximum(N_eval, 0.0)

    # ---- 3. 1-D marginal ---------------------------------------------------
    def marginal_1d(self, axis: str, x: Array) -> Array:
        r"""``dC/dx`` at query points ``x``.

        Step-2 scope: ``axis ∈ {'m1','m2','chi1','chi2'}``.  Two-layer
        reduction:

        1. Evaluate the basis on the engine grid (:py:meth:`_N_on_engine_grid`).
        2. Reduce to the requested 1-D marginal via
           :func:`observables.marginal_1d` — exact grid-quadrature
           contraction of the 4-D integrand.
        3. Interpolate the resulting 1-D density at the user's query
           points (linear in ``log m`` for mass axes, linear in ``χ``
           for spin axes).

        This guarantees grid-point equivalence with both
        :func:`observables.marginals_from_basis` and (when the same
        response is also solved with the v2 implicit Picard)
        :class:`ImplicitEvaluator.marginal_1d` up to the difference
        between the two solver paths — pinned by the test suite as a
        regression invariant.
        """
        _check_axis_supported(axis)
        grid = self.response.grid
        N_mchi = self._N_on_engine_grid()
        _x_grid, dC_dx = _obs.marginal_1d(
            N_mchi,
            grid,
            self.response.kernel_params,
            self.response.log_R_env,
            axis=axis,
        )
        gc, qc, valid = _axis_grid_and_query(
            axis, grid.m, grid.chi, jnp.asarray(x)
        )
        return _interp1_with_zero_outside(gc, dC_dx, qc, valid)

    # ---- 4. 2-D marginal ---------------------------------------------------
    def marginal_2d(
        self,
        axes: Tuple[str, str],
        x_a: Array,
        x_b: Array,
    ) -> Array:
        r"""``dC/(dx_a dx_b)`` at query points.

        Same two-layer reduction as :meth:`marginal_1d`: basis →
        engine grid → :func:`observables.marginal_2d` → bilinear
        interpolation at the query points.  ``axes`` ordering controls
        broadcast layout — we do not silently transpose.
        """
        a, b = axes
        _check_axis_supported(a)
        _check_axis_supported(b)
        grid = self.response.grid
        N_mchi = self._N_on_engine_grid()
        _x_grid, _y_grid, dC = _obs.marginal_2d(
            N_mchi,
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

        Same two-layer reduction reduced to a scalar: basis → engine
        grid → :func:`observables.total_rate`.  Bit-equivalent to
        ``observables.total_rate(density_from_basis(...))``.
        """
        return _obs.total_rate(
            self._N_on_engine_grid(),
            self.response.grid,
            self.response.kernel_params,
            self.response.log_R_env,
        )
