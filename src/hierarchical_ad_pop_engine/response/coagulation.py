"""Generational coagulation response function.

Implementation of the :class:`ResponseFunction` protocol via the Doctor+2020
generational solver. See ``DESIGN.md`` §3.1 for the mathematics.

Pipeline per environment ``θ_env = (v_esc,)``:

1. ``n_1(m, χ)`` = natal IMF × natal spin (1G).
2. For ``k = 2, ..., K_max``: compute ``n_k(m, χ)`` from all ``(j, l)`` pairs
   with ``max(j, l) + 1 == k`` via 4D quadrature over the progenitor
   ``(m_1, χ_1, m_2, χ_2)`` grid, mapping through remnant fits with
   bilinear scatter onto the output ``(m, χ)`` grid, weighted by retention.
3. Per-environment merger-rate density (all generations):

    .. math::

        dN_R/dt/dm_1 dm_2 d\\chi_1 d\\chi_2
            = \\tfrac12 \\, K(m_1, m_2)
              \\, N(m_1, \\chi_1) \\, N(m_2, \\chi_2)

   with ``N = Σ_k n_k`` the total per-env single-BH density and the factor
   ``1/2`` correcting unordered-pair double counting.

Internal log-densities are stored on the :class:`BinaryGrid`; evaluation at
arbitrary binary parameter points uses bilinear interpolation in
``(log m, χ)``.
"""

from __future__ import annotations

from typing import Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike

from ..grids import BinaryGrid
from ..interp import log_density_at
from ..kernels import log_pairing_kernel_grid
from ..natal.mass import log_natal_mass, log_natal_mass_gaussian
from ..natal.spin import build_natal_spin, log_natal_spin
from ..remnants.kick_vector import v_kick_vec
from ..remnants.marginal import (
    chi_rem_at_orientation,
    v_kick_at_orientation,
)
from ..remnants.mass import M_rem
from ..remnants.spin import chi_rem
from ..retention import P_retain
from ..spin_orientation import Aligned


def _bilinear_scatter(
    grid_log_m: Array,  # (N_m,)
    grid_chi: Array,  # (N_chi,)
    src_m: Array,  # (..., ) M_rem values
    src_chi: Array,  # (..., ) chi_rem values
    src_weight: Array,  # (..., ) weights (linear, not log)
) -> Array:
    """Scatter weighted source points onto the (m, χ) grid with bilinear weights.

    Returns a ``(N_m, N_chi)`` array of accumulated weights. Out-of-grid
    source points contribute zero. Differentiable in ``src_m``, ``src_chi``,
    ``src_weight`` via JAX.
    """
    N_m = grid_log_m.size
    N_chi = grid_chi.size
    log_src_m = jnp.log(jnp.maximum(src_m, 1e-30))

    # Fractional indices into the target grid.
    i_frac = jnp.interp(log_src_m, grid_log_m, jnp.arange(N_m, dtype=jnp.float32))
    j_frac = jnp.interp(src_chi, grid_chi, jnp.arange(N_chi, dtype=jnp.float32))

    inside = (
        (log_src_m >= grid_log_m[0])
        & (log_src_m <= grid_log_m[-1])
        & (src_chi >= grid_chi[0])
        & (src_chi <= grid_chi[-1])
    )
    w = jnp.where(inside, src_weight, 0.0)

    i_lo = jnp.clip(jnp.floor(i_frac).astype(jnp.int32), 0, N_m - 2)
    j_lo = jnp.clip(jnp.floor(j_frac).astype(jnp.int32), 0, N_chi - 2)
    di = i_frac - i_lo
    dj = j_frac - j_lo

    flat = jnp.zeros((N_m, N_chi))
    flat = flat.at[i_lo, j_lo].add(w * (1.0 - di) * (1.0 - dj))
    flat = flat.at[i_lo + 1, j_lo].add(w * di * (1.0 - dj))
    flat = flat.at[i_lo, j_lo + 1].add(w * (1.0 - di) * dj)
    flat = flat.at[i_lo + 1, j_lo + 1].add(w * di * dj)
    return flat


def _trapezoidal_cell_widths(grid: BinaryGrid) -> tuple[Array, Array]:
    """Trapezoidal-rule progenitor-cell widths ``(dm1[i], dchi1[j])``.

    Shared by both the aligned and the marginalised
    :func:`_step_one_generation` paths.  ``dm = m · d(log m)`` is the
    Jacobian converting d(log m) to dm.
    """
    dlog_m = jnp.diff(grid.log_m)
    dlog_m_full = jnp.concatenate(
        [dlog_m[:1], 0.5 * (dlog_m[:-1] + dlog_m[1:]), dlog_m[-1:]]
    )
    dchi = jnp.diff(grid.chi)
    dchi_full = jnp.concatenate(
        [dchi[:1], 0.5 * (dchi[:-1] + dchi[1:]), dchi[-1:]]
    )
    dm1 = grid.m * dlog_m_full
    dchi1 = dchi_full
    return dm1, dchi1


def _step_one_generation_aligned(
    log_n_j: Array,
    log_n_l: Array,
    log_K_mm: Array,
    grid: BinaryGrid,
    v_esc: ArrayLike,
    symmetry_factor: float,
) -> Array:
    """Aligned-only bilinear scatter — v1 path, **unchanged** from
    pre-Session-B.  Returns ``(N_m, N_chi)`` count-per-cell.

    Spin convention here is the v1 signed-χ convention: ``chi1`` and
    ``chi2`` are signed scalars (``∈ [-1, 1]`` mathematically; the
    default grid has positive support).  ``chi_rem`` canonicalises
    ``m1 ≥ m2`` externally at this call site (memory
    ``project_chi_rem_asymmetry_finding``).
    """
    m = grid.m
    chi = grid.chi

    m1 = m[:, None, None, None]
    chi1 = chi[None, :, None, None]
    m2 = m[None, None, :, None]
    chi2 = chi[None, None, None, :]

    # Remnant mass / spin / retention on the 4D grid.
    M = M_rem(m1, m2, chi1, chi2)
    heavier_first = m1 >= m2
    A = chi_rem(
        jnp.where(heavier_first, m1, m2),
        jnp.where(heavier_first, m2, m1),
        jnp.where(heavier_first, chi1, chi2),
        jnp.where(heavier_first, chi2, chi1),
    )
    # P_retain default routes to retention_aligned (Heaviside) under
    # the Aligned shortcut — v1 semantics preserved bit-for-bit.
    P_ret = P_retain(m1, m2, chi1, chi2, jnp.asarray(v_esc))

    dm1, dchi1 = _trapezoidal_cell_widths(grid)

    log_weight = (
        log_K_mm[:, None, :, None]
        + log_n_j[:, :, None, None]
        + log_n_l[None, None, :, :]
        + jnp.log(jnp.maximum(P_ret, 1e-30))
    )
    cell = (
        dm1[:, None, None, None]
        * dchi1[None, :, None, None]
        * dm1[None, None, :, None]
        * dchi1[None, None, None, :]
    )
    weight = jnp.exp(log_weight) * cell * symmetry_factor

    return _bilinear_scatter(
        grid.log_m,
        grid.chi,
        M.reshape(-1),
        A.reshape(-1),
        weight.reshape(-1),
    )


def _step_one_generation_marginalised(
    log_n_j: Array,
    log_n_l: Array,
    log_K_mm: Array,
    grid: BinaryGrid,
    v_esc: ArrayLike,
    symmetry_factor: float,
    orientation,
    retention_kind: Literal["heaviside", "sigmoid"] = "heaviside",
    w_over_v_esc: float = 1.0,
) -> Array:
    """Orientation-marginalised bilinear scatter via ``jax.lax.scan``.

    Iterates the orientation quadrature points from
    ``orientation.quadrature_3d()`` and accumulates one
    ``(N_m, N_chi)`` contribution per iteration into the output
    density.  **One full-shape 4-D progenitor tensor alive at a time**
    (Decision 3 of ``demo/PLAN_misalignment_mode.md``) — naïve vmap
    over orientation would multiply the 4-D tensor by ``n_orient`` and
    blow peak memory.

    Spin convention here is the misalignment-mode convention:
    ``chi1, chi2`` are magnitudes ``|χ_i| ∈ [0, 1]``.  Per orientation,
    the L̂-component is ``χ_z = χ_mag · cos θ`` (signed) and feeds into
    ``M_rem``'s ``χ_eff``.  Full vector spins feed into
    :func:`remnants.spin_vector.chi_rem_vec` (returning the scalar
    magnitude ``|χ_rem|``) and
    :func:`remnants.kick_vector.v_kick_vec` (scalar km/s).

    ``retention_kind="heaviside"`` reproduces the v1 hard-cut at each
    orientation point — same retention surface as the aligned path on
    a per-orientation basis but smoothed by the orientation
    averaging.  ``retention_kind="sigmoid"`` makes the per-orientation
    retention smooth, recovering ``∂/∂v_esc`` gradients through JAX
    (see ``remnants/kick.py`` module docstring).
    """
    m = grid.m
    chi = grid.chi
    N_m = m.size
    N_chi = chi.size

    m1 = m[:, None, None, None]
    chi1 = chi[None, :, None, None]
    m2 = m[None, None, :, None]
    chi2 = chi[None, None, None, :]

    dm1, dchi1 = _trapezoidal_cell_widths(grid)
    cell = (
        dm1[:, None, None, None]
        * dchi1[None, :, None, None]
        * dm1[None, None, :, None]
        * dchi1[None, None, None, :]
    )

    log_K_4d = log_K_mm[:, None, :, None]
    log_n_j_4d = log_n_j[:, :, None, None]
    log_n_l_4d = log_n_l[None, None, :, :]
    # Pair-physics part of the integrand (no orientation dep):
    log_pair = log_K_4d + log_n_j_4d + log_n_l_4d
    cell_sym = cell * symmetry_factor

    quad = orientation.quadrature_3d()  # (n_quad, 4)
    v_esc_arr = jnp.asarray(v_esc)

    def scan_body(acc: Array, orient_pt: Array) -> tuple[Array, None]:
        cos_t1 = orient_pt[0]
        cos_t2 = orient_pt[1]
        dphi = orient_pt[2]
        w_orient = orient_pt[3]

        # χ_z per progenitor = |χ| · cos θ.  Signed; feeds M_rem.
        chi1_z = chi1 * cos_t1
        chi2_z = chi2 * cos_t2

        M_k = M_rem(m1, m2, chi1_z, chi2_z)
        # |χ_rem| from the vector BR formula at this orientation
        # quadrature point.  chi_rem_at_orientation canonicalises
        # m_1 ≥ m_2 internally via chi_rem_vec.
        chi_rem_k = chi_rem_at_orientation(
            m1, m2, chi1, chi2, cos_t1, cos_t2, dphi
        )
        v_kick_k = v_kick_at_orientation(
            m1, m2, chi1, chi2, cos_t1, cos_t2, dphi
        )

        if retention_kind == "heaviside":
            P_ret_k = jnp.where(v_kick_k < v_esc_arr, 1.0, 0.0)
        else:
            w_v = w_over_v_esc * v_esc_arr
            P_ret_k = jax.nn.sigmoid((v_esc_arr - v_kick_k) / w_v)

        log_weight = log_pair + jnp.log(jnp.maximum(P_ret_k, 1e-30))
        weight = jnp.exp(log_weight) * cell_sym * w_orient

        contrib = _bilinear_scatter(
            grid.log_m,
            grid.chi,
            M_k.reshape(-1),
            chi_rem_k.reshape(-1),
            weight.reshape(-1),
        )
        return acc + contrib, None

    init = jnp.zeros((N_m, N_chi))
    out, _ = jax.lax.scan(scan_body, init, quad)
    return out


def _step_one_generation(
    log_n_j: Array,  # (N_m, N_chi) progenitor j density
    log_n_l: Array,  # (N_m, N_chi) progenitor l density
    log_K_mm: Array,  # (N_m, N_m) pairing kernel
    grid: BinaryGrid,
    v_esc: ArrayLike,
    symmetry_factor: float,  # 1/2 if j==l else 1
    orientation=None,
    retention_kind: Literal["heaviside", "sigmoid"] = "heaviside",
    w_over_v_esc: float = 1.0,
) -> Array:
    """Compute the contribution of one ``(j, l)`` progenitor pair to ``n_k(m, χ)``.

    Dispatches on the orientation model at trace time:

    * ``orientation is None`` or ``isinstance(orientation, Aligned)``:
      routes to :func:`_step_one_generation_aligned` — v1 code path,
      bit-for-bit unchanged.  Spins are signed scalars in the v1 sense.
    * Any non-Aligned :class:`spin_orientation.OrientationModel`:
      routes to :func:`_step_one_generation_marginalised` — scan over
      orientation quadrature, vector remnant physics per point.
      Spins are magnitudes ``|χ_i|``; orientation distribution
      carries tilt.

    Returns a ``(N_m, N_chi)`` linear-scale density (positive) — the
    caller accumulates across all ``(j, l)`` pairs and finally takes
    the log.
    """
    if orientation is None or isinstance(orientation, Aligned):
        return _step_one_generation_aligned(
            log_n_j, log_n_l, log_K_mm, grid, v_esc, symmetry_factor
        )
    return _step_one_generation_marginalised(
        log_n_j, log_n_l, log_K_mm, grid, v_esc, symmetry_factor,
        orientation,
        retention_kind=retention_kind,
        w_over_v_esc=w_over_v_esc,
    )


class CoagulationResponse(eqx.Module):
    """Generational-coagulation response function.

    Parameters
    ----------
    grid : BinaryGrid
        Fixed (m, χ) grid on which ``n_k`` are stored.
    k_max : int
        Maximum generation tracked (default 4).
    natal_mass_params : dict
        Talbot-Thrane PL+Peak parameters (see :func:`natal.mass.build_natal_mass`).
    natal_spin_params : dict
        Truncated-normal parameters on ``χ ∈ [0, 1]`` (``loc``, ``scale``).
    kernel_params : dict
        Pairing-kernel parameters (``a``, ``b``).
    log_R_env : Array
        ``log(per-env-per-time merger amplitude)`` — overall normalisation
        of the bilinear ``K · N · N`` rate. Differentiable.

    Notes
    -----
    Retention uses aligned-only :func:`retention_aligned` (Heaviside). As a
    consequence, gradients of the rate w.r.t. ``v_esc`` are zero almost
    everywhere; gradients w.r.t. all other (natal / kernel / response) params
    flow normally. To recover AD in ``v_esc``, swap in a tilt-marginalised
    retention model (v2).
    """

    grid: BinaryGrid
    k_max: int = eqx.field(static=True)
    natal_mass_params: dict
    natal_spin_params: dict
    kernel_params: dict
    log_R_env: Array
    natal_mass_kind: str = eqx.field(static=True, default="pl_peak")
    # Solver dispatch. "perturbative" is the legacy generational accumulator
    # in :meth:`_solve`; "implicit" is the all-orders damped Picard fixed-
    # point in :mod:`hierarchical_ad_pop_engine.response.implicit`. The
    # implicit path is the physics-correct one (it includes the loss term
    # and sums the Volterra series to all orders); the perturbative path
    # is retained as a diagnostic and as the AD-safe baseline until the
    # implicit solver gets its custom VJP. See
    # ``demo/PLAN_engine_v2_production.md``.
    solver: str = eqx.field(static=True, default="perturbative")
    # Coupling × time used by the implicit solver. Has absolute physical
    # meaning — ε ≈ 1 is a fully-developed cascade, ε ≪ 1 leaves the
    # natal pool intact. Ignored when ``solver == "perturbative"``.
    epsilon: Array = eqx.field(default_factory=lambda: jnp.asarray(1.0))
    # Spin-orientation model.  Defaults to Aligned() — preserves the v1
    # signed-χ semantics bit-for-bit at every internal call site.
    # Setting to Isotropic / Fisher engages the full vector remnant
    # physics path inside ``_step_one_generation`` via the
    # ``isinstance(orientation, Aligned)`` dispatch.  The orientation
    # model is a pytree (leaf-bearing for Fisher's κ); jit cache keys
    # on its tree structure, so swapping orientation types triggers a
    # retrace, but changing κ inside Fisher is a normal leaf update.
    # See ``demo/PLAN_misalignment_mode.md`` § Session B.
    orientation: object = eqx.field(default_factory=Aligned)
    # Per-orientation retention model under non-Aligned (ignored when
    # ``orientation`` is Aligned, which always routes to the Heaviside
    # path).  ``"heaviside"`` preserves the v1 hard-cut at each
    # orientation point; ``"sigmoid"`` smooths each cut, recovering
    # ``∂/∂v_esc`` gradients through JAX.
    retention_kind: str = eqx.field(static=True, default="heaviside")
    w_over_v_esc: float = eqx.field(static=True, default=1.0)

    def _natal_log_density(self) -> Array:
        spin_dist = build_natal_spin(**self.natal_spin_params)
        if self.natal_mass_kind == "gaussian":
            log_f_m = log_natal_mass_gaussian(self.grid.m, **self.natal_mass_params)
        else:
            log_f_m = log_natal_mass(self.grid.m, **self.natal_mass_params)  # (N_m,)
        log_g_chi = log_natal_spin(spin_dist, self.grid.chi)  # (N_chi,)
        # Replace -inf with a finite very-negative number so the 4D log-sum
        # arithmetic doesn't propagate NaN. Cells flagged -inf carry zero
        # exp() weight downstream.
        log_f_m = jnp.where(jnp.isneginf(log_f_m), -1e30, log_f_m)
        log_g_chi = jnp.where(jnp.isneginf(log_g_chi), -1e30, log_g_chi)
        return log_f_m[:, None] + log_g_chi[None, :]  # (N_m, N_chi)

    def _solve(self, v_esc: Array) -> Array:
        """Return ``log n_k(m, χ)`` for ``k = 1..k_max``; shape ``(K, N_m, N_chi)``.

        Each ``n_k`` is a **density** on (m, χ) — units of dN / dm / dχ —
        consistent with ``log_n[0]`` from :meth:`_natal_log_density`. The
        :func:`_step_one_generation` bilinear scatter returns *count per
        output cell* (the 4D progenitor cell-measure ``dm1·dχ1·dm2·dχ2``
        is already absorbed). Dividing by the output cell area
        ``dm_out · dχ_out`` converts back to a density, so that
        :meth:`log_rate_density`'s bilinear ``K · N · N`` evaluation has
        consistent units across generations.
        """
        log_K_mm = log_pairing_kernel_grid(self.grid.m, **self.kernel_params)

        # Output cell area for converting bilinear-scatter weight per cell
        # → density on (m, χ). Trapezoidal-rule weights matching
        # ``_step_one_generation``'s progenitor measure.
        m = self.grid.m
        chi = self.grid.chi
        dlog_m = jnp.diff(self.grid.log_m)
        dlog_m_full = jnp.concatenate(
            [dlog_m[:1], 0.5 * (dlog_m[:-1] + dlog_m[1:]), dlog_m[-1:]]
        )
        dchi = jnp.diff(chi)
        dchi_full = jnp.concatenate(
            [dchi[:1], 0.5 * (dchi[:-1] + dchi[1:]), dchi[-1:]]
        )
        dm_out = m * dlog_m_full  # (N_m,)
        cell_area = dm_out[:, None] * dchi_full[None, :]  # (N_m, N_chi)

        log_n = [self._natal_log_density()]
        for k in range(2, self.k_max + 1):
            # Sum contributions from all (j, l) pairs with max(j, l) + 1 == k.
            # I.e. j in 1..k-1, l == k-1 (and symmetric).
            n_lin = jnp.zeros((self.grid.m.size, self.grid.chi.size))
            for j in range(1, k):
                l = k - 1  # max progenitor generation
                if j == l:
                    sym = 0.5
                    contrib = _step_one_generation(
                        log_n[j - 1],
                        log_n[l - 1],
                        log_K_mm,
                        self.grid,
                        v_esc,
                        sym,
                        orientation=self.orientation,
                        retention_kind=self.retention_kind,
                        w_over_v_esc=self.w_over_v_esc,
                    )
                else:
                    # K, M_rem, P_retain are all symmetric in (m1, m2), so the
                    # 4D ``_step_one_generation`` integral with ``n_j`` on axis 1
                    # and ``n_l`` on axis 2 already captures the full
                    # ``∫∫ K n_j n_l dm1 dm2`` contribution. Adding a swapped
                    # ``(n_l, n_j)`` call would double-count the same unordered
                    # pair. (Red-team Phase 1 validation found this: 1G and 2G
                    # had no j != l cross terms and matched to JS ≈ 1e-4; 3G+
                    # showed JS ≈ 0.2-0.35 from the 2× over-count.)
                    sym = 1.0
                    contrib = _step_one_generation(
                        log_n[j - 1],
                        log_n[l - 1],
                        log_K_mm,
                        self.grid,
                        v_esc,
                        sym,
                        orientation=self.orientation,
                        retention_kind=self.retention_kind,
                        w_over_v_esc=self.w_over_v_esc,
                    )
                # contrib is count per output cell; divide by cell area to
                # convert to density so that n_k is consistent with n_1.
                n_lin = n_lin + contrib / cell_area
            log_n.append(jnp.log(jnp.maximum(n_lin, 1e-30)))
        return jnp.stack(log_n, axis=0)

    def _log_total_density(self, v_esc: Array) -> Array:
        """``log N(m, χ) = log Σ_k n_k(m, χ)`` on the binary grid.

        Dispatches on :attr:`solver`. With ``solver == "perturbative"`` we
        return ``logsumexp_k log n_k`` from the generational accumulator;
        with ``solver == "implicit"`` we return the converged ``log N*``
        from :func:`response.implicit.solve_implicit_fixed_point`.
        """
        if self.solver == "implicit":
            # Local import to break the implicit ↔ coagulation cycle.
            from .implicit import solve_implicit_fixed_point

            result = solve_implicit_fixed_point(self, v_esc, epsilon=self.epsilon)
            return result.log_N
        log_n_k = self._solve(v_esc)  # (K, N_m, N_chi)
        return _logsumexp_axis(log_n_k, axis=0)

    def log_rate_density(
        self,
        m1: Array,
        m2: Array,
        chi1: Array,
        chi2: Array,
        env: dict,
    ) -> Array:
        """Return ``log[dN_R / dt / dm1 dm2 dχ1 dχ2]`` per environment.

        Uses

        .. math::

            \\frac{dN_R}{dt\\, dm_1 dm_2 d\\chi_1 d\\chi_2}
            = \\tfrac12 \\, R_{\\rm env} \\, K(m_1, m_2)
              \\, N(m_1, \\chi_1) \\, N(m_2, \\chi_2) .
        """
        v_esc = env["v_esc"]
        log_N = self._log_total_density(v_esc)
        from ..kernels import log_pairing_kernel

        log_K_eval = log_pairing_kernel(m1, m2, **self.kernel_params)
        log_N_1 = log_density_at(self.grid.log_m, self.grid.chi, log_N, m1, chi1)
        log_N_2 = log_density_at(self.grid.log_m, self.grid.chi, log_N, m2, chi2)
        return (
            self.log_R_env
            - jnp.log(2.0)
            + log_K_eval
            + log_N_1
            + log_N_2
        )


def _logsumexp_axis(x: Array, axis: int) -> Array:
    m = jnp.max(x, axis=axis, keepdims=True)
    safe_m = jnp.where(jnp.isneginf(m), 0.0, m)
    return jnp.log(jnp.sum(jnp.exp(x - safe_m), axis=axis)) + jnp.squeeze(safe_m, axis=axis)


def build_gaussian_coagulation(
    *,
    mu: ArrayLike = 8.0,
    sigma: ArrayLike = 1.0,
    m_min: ArrayLike = 2.0,
    m_max: ArrayLike = 200.0,
    natal_spin_loc: ArrayLike = 0.0,
    natal_spin_scale: ArrayLike = 0.05,
    kernel_a: ArrayLike = 0.0,
    kernel_b: ArrayLike = 2.0,
    log_R_env: ArrayLike = 0.0,
    k_max: int = 4,
    grid: Optional[BinaryGrid] = None,
    solver: str = "perturbative",
    epsilon: ArrayLike = 1.0,
) -> "CoagulationResponse":
    """Convenience constructor for an impulse-response demo / unit test.

    Uses ``jax.scipy.stats.truncnorm.logpdf`` (the same primitive gwkokab's
    own ``GaussianPrimaryMassRatio`` uses) as the natal IMF. The defaults
    ``μ = 8 M⊙``, ``σ = 1 M⊙`` give a narrow Gaussian centred on the natal
    8 M⊙ BHs of Doctor+2020's working example, so successive generations
    pile up at the expected ``M_rem(m, m)`` cascade locations.
    """
    if grid is None:
        grid = BinaryGrid.default(
            m_min=float(jnp.asarray(m_min)),
            m_max=float(jnp.asarray(m_max)),
        )
    return CoagulationResponse(
        grid=grid,
        k_max=k_max,
        natal_mass_params=dict(
            mu=jnp.asarray(mu),
            sigma=jnp.asarray(sigma),
            m_min=jnp.asarray(m_min),
            m_max=jnp.asarray(m_max),
        ),
        natal_spin_params=dict(
            loc=jnp.asarray(natal_spin_loc),
            scale=jnp.asarray(natal_spin_scale),
        ),
        kernel_params=dict(
            a=jnp.asarray(kernel_a),
            b=jnp.asarray(kernel_b),
        ),
        log_R_env=jnp.asarray(log_R_env),
        natal_mass_kind="gaussian",
        solver=solver,
        epsilon=jnp.asarray(epsilon),
    )


def build_default_coagulation(
    *,
    alpha: ArrayLike = 3.0,
    delta_m: ArrayLike = 5.0,
    lambda_peak: ArrayLike = 0.03,
    mu_peak: ArrayLike = 33.0,
    sigma_peak: ArrayLike = 3.0,
    m_min: ArrayLike = 5.0,
    m_max: ArrayLike = 100.0,
    natal_spin_loc: ArrayLike = 0.1,
    natal_spin_scale: ArrayLike = 0.15,
    kernel_a: ArrayLike = 0.0,
    kernel_b: ArrayLike = 2.0,
    log_R_env: ArrayLike = 0.0,
    k_max: int = 4,
    grid: Optional[BinaryGrid] = None,
    solver: str = "perturbative",
    epsilon: ArrayLike = 1.0,
) -> CoagulationResponse:
    """Convenience constructor with reasonable defaults for a single AGN-disk-like environment."""
    if grid is None:
        grid = BinaryGrid.default(
            m_min=float(jnp.asarray(m_min)),
            m_max=float(jnp.asarray(m_max)),
        )
    return CoagulationResponse(
        grid=grid,
        k_max=k_max,
        natal_mass_params=dict(
            alpha=jnp.asarray(alpha),
            delta_m=jnp.asarray(delta_m),
            lambda_peak=jnp.asarray(lambda_peak),
            mu_peak=jnp.asarray(mu_peak),
            sigma_peak=jnp.asarray(sigma_peak),
            m_min=jnp.asarray(m_min),
            m_max=jnp.asarray(m_max),
        ),
        natal_spin_params=dict(
            loc=jnp.asarray(natal_spin_loc),
            scale=jnp.asarray(natal_spin_scale),
        ),
        kernel_params=dict(
            a=jnp.asarray(kernel_a),
            b=jnp.asarray(kernel_b),
        ),
        log_R_env=jnp.asarray(log_R_env),
        solver=solver,
        epsilon=jnp.asarray(epsilon),
    )
