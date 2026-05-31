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
from ..natal.mass import (
    log_natal_mass,
    log_natal_mass_gaussian,
    log_natal_mass_two_gaussian,
)
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
    retention_kind: Literal["heaviside", "sigmoid"] = "sigmoid",
    w_over_v_esc: float = 1.0,
    cache: Optional[object] = None,
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

    ``retention_kind="sigmoid"`` (the default after 2026-05-27) keeps
    the per-orientation retention smooth, which (a) lets the
    ``spin_orientation`` module's ``(4, 8) = 128``-point default
    quadrature converge spectrally on the smooth integrand, and (b)
    recovers ``∂/∂v_esc`` gradients through JAX (see ``remnants/kick.py``
    module docstring).  ``retention_kind="heaviside"`` is the v1
    bit-equality opt-in — reproduces the hard cut at each
    orientation point but is piecewise-discontinuous in orientation,
    so quadrature convergence drops; pair with
    ``Isotropic(n_theta=8, n_dphi=16)`` for the v1 1024-point
    accuracy.

    Session F2 fast path
    --------------------
    If ``cache`` carries a non-``None``
    ``chi_out_basis_coeffs`` field (built by
    :func:`orientation_precompute.build_orientation_cache` with
    ``R_chi_out > 0``), the per-iter orientation scan over both
    ``M_rem`` and ``chi_rem`` is replaced by a one-shot tensor
    contraction against the precomputed spectral coefficients and
    a precomputed P_ret-weighted ``M_rem`` mean.  See
    :func:`_step_one_generation_basis_fast` for the contraction
    geometry.

    Speedup: ~``n_quad / 1`` per Picard iter on the gain side; the
    orientation cost reappears once at cache construction (one scan
    per solve).  Aligned dispatch bypasses this branch by routing
    to ``_step_one_generation_aligned`` at the upstream
    ``_step_one_generation`` site — Aligned never reaches here.
    """
    # F2 fast path — if a basis cache is present, contract against
    # the precomputed surrogate instead of scanning orientation.
    if (
        cache is not None
        and getattr(cache, "chi_scatter_marg", None) is not None
    ):
        return _step_one_generation_basis_fast(
            log_n_j, log_n_l, log_K_mm, grid, symmetry_factor, cache,
        )

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


def _step_one_generation_basis_fast(
    log_n_j: Array,
    log_n_l: Array,
    log_K_mm: Array,
    grid: BinaryGrid,
    symmetry_factor: float,
    cache,
) -> Array:
    """F2 fast-path gain — basis-surrogate contraction in place of the scan.

    Replaces ``_step_one_generation_marginalised``'s per-Picard-iter
    orientation scan with a one-shot tensor contraction against the
    F2 surrogate fields on ``cache``.  Same output convention
    (count-per-output-cell) so the upstream
    ``response/implicit.py:gain_density`` cell-area divisor
    continues to produce a density on ``(m_out, χ_out)`` consistent
    with ``log_n[0]`` from :meth:`CoagulationResponse._natal_log_density`.

    Geometry
    --------
    The orientation-averaged gain at progenitor cell
    ``(i_1, j_1, i_2, j_2)`` produces a 2-D distribution on
    ``(m_out, χ_out)``.  F2 (bilinear-hat surrogate; PLAN § Strategy
    B, simplified for mass-conservation at coarse engine grids)
    factorises:

    * ``χ_out`` direction: the orientation-summed bilinear-hat
      scattered coefficients ``cache.chi_scatter_marg[j_out, …]`` —
      a partition-of-unity decomposition that conserves mass by
      construction.
    * ``m_out`` direction: the P_ret-weighted orientation-mean
      ``M_rem_mean_retained[…]`` — collapsed to a single mean per
      progenitor cell.  Strategy A's limitation: the M_rem
      orientation spread is dropped.  At production m-resolution
      (N_m ≥ 32) the spread is well-resolved by adjacent grid bins
      and the bias is small; at sandbox-tier resolution (N_m = 8)
      the bias surfaces as ~few % rel-L1 vs the scan path.

    Mathematically::

        contribution[i_out, j_out]
            = Σ_{i1,j1,i2,j2}
                  K_pair · N_j(i1,j1) · N_l(i2,j2) · cell · sym
                  · cache.chi_scatter_marg[j_out, i1, j1, i2, j2]
                  · bilinear-m-hat(i_out, M_rem_mean_retained[i1, …])

    Mass conservation::

        Σ_{j_out} cache.chi_scatter_marg[j_out, …]
            = P_retain_marg[…]   (partition of unity for hats)

    so the total mass on (i_out, j_out) equals the scan-path total
    up to the m-side approximation.

    Aligned bit-stability
    ---------------------
    Aligned never reaches this branch; the
    :func:`_step_one_generation` dispatcher routes Aligned to
    :func:`_step_one_generation_aligned` upstream.  The cache itself
    is also ``None`` under Aligned (see
    :func:`orientation_precompute.build_orientation_cache`).
    """
    m = grid.m
    chi = grid.chi
    log_m = grid.log_m
    N_m = m.size
    N_chi = chi.size

    dm1, dchi1 = _trapezoidal_cell_widths(grid)
    cell = (
        dm1[:, None, None, None]
        * dchi1[None, :, None, None]
        * dm1[None, None, :, None]
        * dchi1[None, None, None, :]
    )

    log_K_4d = log_K_mm[:, None, :, None]
    log_pair = log_K_4d + log_n_j[:, :, None, None] + log_n_l[None, None, :, :]
    base_weight = jnp.exp(log_pair) * cell * symmetry_factor  # (N_m, N_chi, N_m, N_chi)

    # ── Per-(j_out) weighted progenitor tensor ───────────────────
    # cache.chi_scatter_marg has shape (N_χ, N_m, N_χ, N_m, N_χ).
    # weighted[j_out, i1, j1, i2, j2] = base_weight · chi_scatter_marg[j_out, …]
    weighted = base_weight[None, ...] * cache.chi_scatter_marg

    # Flatten progenitor mesh into a single axis so the m-side
    # bilinear scatter is a 1-D operation per j_out.
    weighted_flat = weighted.reshape(N_chi, -1)              # (N_χ, N_p)
    M_rem_flat = cache.M_rem_mean_retained.reshape(-1)        # (N_p,)

    # ── m-side bilinear scatter onto i_out ──────────────────────
    log_M = jnp.log(jnp.maximum(M_rem_flat, 1e-30))
    i_frac = jnp.interp(
        log_M, log_m, jnp.arange(N_m, dtype=jnp.float64),
    )
    inside_m = (log_M >= log_m[0]) & (log_M <= log_m[-1])
    i_lo = jnp.clip(jnp.floor(i_frac).astype(jnp.int32), 0, N_m - 2)
    di = i_frac - i_lo

    w_lo = jnp.where(inside_m, weighted_flat * (1.0 - di), 0.0)  # (N_χ, N_p)
    w_hi = jnp.where(inside_m, weighted_flat * di, 0.0)

    # Scatter on the m_out axis.  Shape (N_χ, N_m), transposed at
    # the end to match the (N_m, N_χ) caller convention.
    out_jm = jnp.zeros((N_chi, N_m))
    out_jm = out_jm.at[:, i_lo].add(w_lo)
    out_jm = out_jm.at[:, i_lo + 1].add(w_hi)
    return out_jm.T  # (N_m, N_χ)


def _step_one_generation(
    log_n_j: Array,  # (N_m, N_chi) progenitor j density
    log_n_l: Array,  # (N_m, N_chi) progenitor l density
    log_K_mm: Array,  # (N_m, N_m) pairing kernel
    grid: BinaryGrid,
    v_esc: ArrayLike,
    symmetry_factor: float,  # 1/2 if j==l else 1
    orientation=None,
    retention_kind: Literal["heaviside", "sigmoid"] = "sigmoid",
    w_over_v_esc: float = 1.0,
    cache: Optional[object] = None,
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
        cache=cache,
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
    # 2026-05-27: default flipped from "heaviside" to "sigmoid" to make
    # the (4, 8) = 128-point orientation quadrature default converge
    # spectrally on the smooth integrand (see
    # ``spin_orientation`` module Convergence Notes).  Aligned dispatch
    # is unaffected (Aligned shortcut bypasses retention_kind entirely).
    retention_kind: str = eqx.field(static=True, default="sigmoid")
    w_over_v_esc: float = eqx.field(static=True, default=1.0)
    # Particle-MC stash for ``solver="mc"`` (Session M2 of
    # ``demo/PLAN_mc_first_reprioritization.md``).  ``None`` under the
    # default implicit / basis / perturbative paths — they ignore it.
    # When ``solver="mc"`` the response factory eagerly runs the MC
    # driver at construction time and parks the resulting stash here;
    # :meth:`log_rate_density` then dispatches to
    # :meth:`response.mc.MCResponse.log_rate_density` instead of the
    # grid-based path below.  See ``[[project-engine-m2-done]]``.
    mc_response: object = eqx.field(default=None)
    # JAX-pure MC response for ``solver="mc_jax"`` (Session J4 of
    # ``demo/PLAN_jax_mc.md``).  Sibling of ``mc_response``: the J4
    # factory builds this via the JAX-traceable ``particle_mc_jax`` +
    # ``MCEvaluatorJax`` pipeline so AD flows from log_rate_density
    # back to θ_pop — what unblocks the M4 SVI integration test.
    # Carrier is :class:`response.mc_jax.MCResponseJax`.  See
    # ``[[project-jax-mc-j4-done]]``.
    mc_jax_response: object = eqx.field(default=None)

    def _natal_log_density(self) -> Array:
        spin_dist = build_natal_spin(**self.natal_spin_params)
        if self.natal_mass_kind == "gaussian":
            log_f_m = log_natal_mass_gaussian(self.grid.m, **self.natal_mass_params)
        elif self.natal_mass_kind == "two_gaussian":
            log_f_m = log_natal_mass_two_gaussian(self.grid.m, **self.natal_mass_params)
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

        Dispatches on :attr:`solver`:

        * ``"mc"`` — delegates to
          :meth:`response.mc.MCResponse.log_rate_density` on the
          stashed :attr:`mc_response`.  The MC driver was run eagerly
          at construction time by
          :func:`response.mc.build_mc_response`; ``env["v_esc"]`` is
          ignored on this path (the MC driver baked ``v_esc`` at
          construction time — see PLAN § "Hard gotcha").
        * Any other ``solver`` value (``"implicit"``, ``"basis"``,
          ``"perturbative"``) — falls through to the existing grid-
          based path below, which calls :meth:`_log_total_density` and
          looks up at the query points via :func:`log_density_at`.

        Under ``solver="mc"`` we require :attr:`mc_response` to be
        non-``None``; the factory contract is that
        ``coagulation_model(solver="mc")`` always populates it.  A
        ``ValueError`` here is a misuse signal, not a runtime
        condition.
        """
        if self.solver == "mc":
            if self.mc_response is None:
                raise ValueError(
                    "CoagulationResponse(solver='mc') requires "
                    "mc_response to be populated; build via "
                    "coagulation_model(..., solver='mc') or attach "
                    "an MCResponse explicitly via response.mc."
                    "build_mc_response(...)."
                )
            return self.mc_response.log_rate_density(
                m1, m2, chi1, chi2, env
            )

        if self.solver == "mc_jax":
            if self.mc_jax_response is None:
                raise ValueError(
                    "CoagulationResponse(solver='mc_jax') requires "
                    "mc_jax_response to be populated; build via "
                    "build_gaussian_coagulation(..., solver='mc_jax') "
                    "or attach an MCResponseJax explicitly via "
                    "response.mc_jax.build_mc_jax_response(...)."
                )
            return self.mc_jax_response.log_rate_density(
                m1, m2, chi1, chi2, env
            )

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
    orientation: object = None,
    retention_kind: Literal["heaviside", "sigmoid"] = "sigmoid",
    w_over_v_esc: float = 1.0,
    # Session M2 MC-path kwargs.  Ignored unless ``solver="mc"``.
    mc_v_esc: Optional[ArrayLike] = None,
    mc_n_seed: int = 5000,
    mc_depletion_fraction: float = 0.5,
    mc_rng_seed: Optional[int] = None,
    mc_orientation_key_seed: int = 0,
    mc_k: Optional[int] = None,
    # Session J4 MC-JAX-path kwargs.  Ignored unless ``solver="mc_jax"``.
    # ``mc_jax_key`` defaults to ``jax.random.PRNGKey(0)`` so the SVI
    # objective is deterministic — change it via the test fixture if
    # multi-key averaging is desired (typically not, for SVI; the
    # variance estimator is already a stochastic outer loop).
    mc_jax_key: Optional[jax.Array] = None,
    mc_jax_n_steps: Optional[int] = None,
    # Round-3 (B.1 Bernoulli K-gate): K_max/⟨K⟩ estimate sized so
    # n_steps yields expected K-accepts = depletion × n_seed (numpy
    # parity).  See
    # ``demo/PLAN_jax_mc_time_decoupling_diagnosis.md`` § 5.
    mc_jax_k_ratio: float = 1.5,
    # Session B.2 (soft K-gate): selects the cascade gate mechanism.
    # ``"bernoulli"`` keeps the round-3 B.1 path (default, unchanged);
    # ``"soft"`` routes to particle_mc_jax_soft — deterministic
    # occupancy-flow gate that removes the k_ratio precision knob and the
    # straight-through bias.  See
    # ``demo/PLAN_jax_mc_time_decoupling_diagnosis.md`` § 5.B.2 and
    # ``project-jax-mc-b2-soft-gate-followup``.  ``mc_jax_n_steps_mult``
    # sizes the soft path's n_steps upper bound (ignored under bernoulli).
    mc_jax_gate: str = "bernoulli",
    mc_jax_n_steps_mult: float = 3.0,
) -> "CoagulationResponse":
    """Convenience constructor for an impulse-response demo / unit test.

    Uses ``jax.scipy.stats.truncnorm.logpdf`` (the same primitive gwkokab's
    own ``GaussianPrimaryMassRatio`` uses) as the natal IMF. The defaults
    ``μ = 8 M⊙``, ``σ = 1 M⊙`` give a narrow Gaussian centred on the natal
    8 M⊙ BHs of Doctor+2020's working example, so successive generations
    pile up at the expected ``M_rem(m, m)`` cascade locations.

    Misalignment-mode arguments
    ---------------------------
    ``orientation`` — defaults to ``None`` which keeps the v1 ``Aligned``
    behaviour bit-for-bit at every internal call site (the orientation
    field falls back to ``CoagulationResponse``'s default factory).  Pass
    a concrete :mod:`spin_orientation` model (``Isotropic`` / ``Fisher``)
    to engage the per-orientation vector physics path inside
    ``_step_one_generation``.

    ``retention_kind`` and ``w_over_v_esc`` configure the per-orientation
    retention under non-``Aligned`` models (ignored otherwise).  See
    ``CoagulationResponse`` and
    :func:`remnants.marginal.P_retain_marginalised` for details.
    """
    if grid is None:
        grid = BinaryGrid.default(
            m_min=float(jnp.asarray(m_min)),
            m_max=float(jnp.asarray(m_max)),
        )
    kwargs = dict(
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
        retention_kind=retention_kind,
        w_over_v_esc=w_over_v_esc,
    )
    if orientation is not None:
        kwargs["orientation"] = orientation
    response = CoagulationResponse(**kwargs)
    return _finalize_response_solver(
        response,
        solver=solver,
        mc_v_esc=mc_v_esc,
        mc_n_seed=mc_n_seed,
        mc_depletion_fraction=mc_depletion_fraction,
        mc_rng_seed=mc_rng_seed,
        mc_orientation_key_seed=mc_orientation_key_seed,
        mc_k=mc_k,
        mc_jax_key=mc_jax_key,
        mc_jax_n_steps=mc_jax_n_steps,
        mc_jax_k_ratio=mc_jax_k_ratio,
        mc_jax_gate=mc_jax_gate,
        mc_jax_n_steps_mult=mc_jax_n_steps_mult,
    )


def _finalize_response_solver(
    response: "CoagulationResponse",
    *,
    solver: str,
    mc_v_esc: Optional[ArrayLike],
    mc_n_seed: int,
    mc_depletion_fraction: float,
    mc_rng_seed: Optional[int],
    mc_orientation_key_seed: int,
    mc_k: Optional[int],
    mc_jax_key: Optional[jax.Array],
    mc_jax_n_steps: Optional[int],
    mc_jax_k_ratio: float,
    mc_jax_gate: str,
    mc_jax_n_steps_mult: float,
    mc_jax_kde_adaptive: bool = False,
) -> "CoagulationResponse":
    """Run + attach the eager MC solver state (shared by the builders).

    Factored out of :func:`build_gaussian_coagulation` so the two-pulse
    builder (:func:`build_two_pulse_coagulation`) reuses the identical
    ``solver`` dispatch without duplication.  No-op for the analytic
    solvers (``perturbative`` / ``implicit`` / ``basis``); returns the
    response unchanged.
    """
    # Session M2: under ``solver="mc"`` we eagerly run the particle MC
    # driver and stash the result on the response via the new
    # ``mc_response`` field.  ``mc_v_esc`` is the MC's escape-velocity
    # bake-in — passed separately from the formation-side ``env["v_esc"]``
    # which the MC path ignores (PLAN § "Hard gotcha").
    if solver == "mc":
        if mc_v_esc is None:
            raise ValueError(
                "build_gaussian_coagulation(solver='mc') requires "
                "mc_v_esc to be set (the MC driver bakes v_esc at "
                "construction time)."
            )
        from .mc import build_mc_response
        mc_response = build_mc_response(
            response,
            v_esc=mc_v_esc,
            n_seed=mc_n_seed,
            depletion_fraction=mc_depletion_fraction,
            rng_seed=mc_rng_seed,
            orientation_key_seed=mc_orientation_key_seed,
            k=mc_k,
        )
        # Attach via eqx.tree_at (CoagulationResponse is frozen).
        import equinox as _eqx
        response = _eqx.tree_at(
            lambda r: r.mc_response, response, mc_response,
        )

    if solver == "mc_jax":
        if mc_v_esc is None:
            raise ValueError(
                "build_gaussian_coagulation(solver='mc_jax') requires "
                "mc_v_esc to be set (the MC driver bakes v_esc at "
                "construction time)."
            )
        from .mc_jax import build_mc_jax_response
        # Default key is deterministic so the SVI objective is
        # deterministic in θ.  Multi-key averaging is the caller's
        # decision (typically not used for SVI).
        if mc_jax_key is None:
            mc_jax_key = jax.random.PRNGKey(0)
        mc_jax_response = build_mc_jax_response(
            response,
            v_esc=mc_v_esc,
            key=mc_jax_key,
            n_seed=mc_n_seed,
            depletion_fraction=mc_depletion_fraction,
            n_steps=mc_jax_n_steps,
            k_ratio=float(mc_jax_k_ratio),
            gate=mc_jax_gate,
            n_steps_mult=float(mc_jax_n_steps_mult),
            kde_adaptive=bool(mc_jax_kde_adaptive),
        )
        import equinox as _eqx
        response = _eqx.tree_at(
            lambda r: r.mc_jax_response, response, mc_jax_response,
            is_leaf=lambda x: x is None,
        )
    return response


def build_two_pulse_coagulation(
    *,
    mu_1: ArrayLike = 10.0,
    sigma_1: ArrayLike = 1.0,
    mu_2: ArrayLike = 30.0,
    sigma_2: ArrayLike = 2.0,
    lambda_2: ArrayLike = 0.4,
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
    orientation: object = None,
    retention_kind: Literal["heaviside", "sigmoid"] = "sigmoid",
    w_over_v_esc: float = 1.0,
    mc_v_esc: Optional[ArrayLike] = None,
    mc_n_seed: int = 5000,
    mc_depletion_fraction: float = 0.5,
    mc_rng_seed: Optional[int] = None,
    mc_orientation_key_seed: int = 0,
    mc_k: Optional[int] = None,
    mc_jax_key: Optional[jax.Array] = None,
    mc_jax_n_steps: Optional[int] = None,
    mc_jax_k_ratio: float = 1.5,
    mc_jax_gate: str = "bernoulli",
    mc_jax_n_steps_mult: float = 3.0,
    mc_jax_kde_adaptive: bool = False,
) -> "CoagulationResponse":
    r"""Two-pulse (two-Gaussian) natal-IMF coagulation response.

    The ``natal_family="two_gaussian"`` sibling of
    :func:`build_gaussian_coagulation`: the natal mass marginal is the
    normalised mixture

    .. math::

        f_{\rm IMF}(m) = (1 - \lambda_2)\,\mathrm{TruncN}(m; \mu_1, \sigma_1)
                       + \lambda_2\,\mathrm{TruncN}(m; \mu_2, \sigma_2)

    (see :func:`natal.mass.log_natal_mass_two_gaussian`), with a single
    shared natal-spin truncnorm.  The defaults — pulses at ``μ₁ = 10`` and
    ``μ₂ = 30 M⊙`` with near-zero spin — are the methods-paper two-pulse
    small-spin recovery experiment (INSTRUCTIONS_gwkokab_stage1 §6).

    Caller owns the ``μ₁ < μ₂`` label-switch convention; the fit-side
    priors / reparametrisation in ``inference/stage1_wiring`` enforce it.

    Same ``solver`` dispatch as :func:`build_gaussian_coagulation`
    (``"mc_jax"`` for the AD-traceable SVI/NUTS path); the cascade,
    remnant physics, and selection downstream are entirely natal-family
    agnostic.
    """
    if grid is None:
        grid = BinaryGrid.default(
            m_min=float(jnp.asarray(m_min)),
            m_max=float(jnp.asarray(m_max)),
        )
    kwargs = dict(
        grid=grid,
        k_max=k_max,
        natal_mass_params=dict(
            mu_1=jnp.asarray(mu_1),
            sigma_1=jnp.asarray(sigma_1),
            mu_2=jnp.asarray(mu_2),
            sigma_2=jnp.asarray(sigma_2),
            lambda_2=jnp.asarray(lambda_2),
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
        natal_mass_kind="two_gaussian",
        solver=solver,
        epsilon=jnp.asarray(epsilon),
        retention_kind=retention_kind,
        w_over_v_esc=w_over_v_esc,
    )
    if orientation is not None:
        kwargs["orientation"] = orientation
    response = CoagulationResponse(**kwargs)
    return _finalize_response_solver(
        response,
        solver=solver,
        mc_v_esc=mc_v_esc,
        mc_n_seed=mc_n_seed,
        mc_depletion_fraction=mc_depletion_fraction,
        mc_rng_seed=mc_rng_seed,
        mc_orientation_key_seed=mc_orientation_key_seed,
        mc_k=mc_k,
        mc_jax_key=mc_jax_key,
        mc_jax_n_steps=mc_jax_n_steps,
        mc_jax_k_ratio=mc_jax_k_ratio,
        mc_jax_gate=mc_jax_gate,
        mc_jax_n_steps_mult=mc_jax_n_steps_mult,
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
