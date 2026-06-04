"""Fixed grids for binary parameters (m, χ), redshift z, and environment params.

The engine represents every internal density n_k(m, χ) as values on a fixed grid.
This module owns the grid definitions and conventions so they're the same
everywhere (response function, formation rate, convolution).

Conventions (defaults; configurable):

- ``m``: log-spaced ∈ [2 M⊙, 200 M⊙], ``N_m = 96``.
- ``χ``: linear ∈ [0, 1], ``N_χ = 32``.
- ``z``: linear ∈ [0, 4], ``N_z = 64``.
- ``v_esc``: log-spaced ∈ [1, 1000] km/s, ``N_ve = 24``.

All grids are immutable equinox.Modules so they are JAX-tree leaves but never
recompile-trigger on identity changes.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

# ---------------------------------------------------------------------------
# Physics anchors for the default clustered-χ node placement.
#
# ``_CHI_CASCADE_PEAK`` is the remnant spin of an equal-mass, non-spinning
# 1G+1G merger, ``χ_rem(30, 30, 0, 0) = 0.6865`` from the engine's own
# Healy-Lousto-Zlochower aligned remnant fit (``remnants/spin.chi_rem``).
# Successive generations and unequal-mass pairs spread this peak over
# roughly ``[0.62, 0.72]`` (e.g. a 2:1 mass ratio gives 0.624), so the
# cascade bump width covers that span; the uniform floor catches the
# higher-generation tail beyond it.
#
# ``_CHI_NATAL_PEAK`` / ``_CHI_NATAL_WIDTH`` track the default natal Beta /
# truncnorm spin distribution (``natal_spin_loc=0.0``,
# ``natal_spin_scale=0.05``), whose mass piles up near ``χ ≈ 0``.
# ---------------------------------------------------------------------------
_CHI_NATAL_PEAK = 0.05
_CHI_NATAL_WIDTH = 0.05
_CHI_CASCADE_PEAK = 0.6865
_CHI_CASCADE_WIDTH = 0.04

# (center, width, weight) bumps for the default target node-density.
_DEFAULT_CHI_PEAKS: tuple[tuple[float, float, float], ...] = (
    (_CHI_NATAL_PEAK, _CHI_NATAL_WIDTH, 1.0),
    (_CHI_CASCADE_PEAK, _CHI_CASCADE_WIDTH, 1.0),
)


def clustered_chi_nodes(
    n_chi: int = 32,
    chi_min: float = 1e-3,
    chi_max: float = 1.0 - 1e-3,
    peaks: tuple[tuple[float, float, float], ...] = _DEFAULT_CHI_PEAKS,
    floor_frac: float = 0.25,
    n_fine: int = 4001,
) -> Array:
    r"""Smooth density-mapped nonuniform χ nodes via inverse-CDF equidistribution.

    Places ``n_chi`` nodes on ``[chi_min, chi_max]`` so their local density
    follows a **target node-density** ``g(χ)`` that is a uniform floor plus
    a Gaussian bump per physics peak::

        g(χ) = floor_frac / span
               + (1 − floor_frac) · Σ_p w_p exp(−½ ((χ − c_p)/σ_p)²) / Z

    where ``span = chi_max − chi_min`` and ``Z`` normalises the bump sum to
    unit integral.  Nodes sit at equal increments of the cumulative
    ``G(χ) = ∫ g`` (equidistribution), so spacing ``∝ 1/g`` — dense at the
    natal peak (``χ ≈ 0.05``) and the cascade remnant peak (``χ ≈ 0.69``),
    sparse in the flat valley between, with the ``floor_frac`` keeping a
    uniform backbone everywhere (it also guarantees ``g > 0`` so ``G`` is
    strictly monotone and the nodes are strictly increasing — no
    coincident nodes, no zero-width cells).

    The nodes are **static** — computed once in numpy and frozen as a JAX
    array (same pattern as the Gauss-Legendre nodes in
    :mod:`spin_orientation`).  They carry no dependence on the population
    parameters ``θ``, so they never enter the ``jax.jit`` trace or the AD
    graph: every downstream consumer reaches the grid only through
    ``jnp.interp(·, grid.chi, ·)`` (bilinear scatter / hats) and
    ``jnp.diff(grid.chi)`` (trapezoid weights), both of which are exact on
    an arbitrary monotone grid.

    Parameters
    ----------
    n_chi
        Number of χ nodes (endpoints inclusive).
    chi_min, chi_max
        Grid endpoints.  Defaults shift slightly inside ``[0, 1]`` for the
        same reason as :meth:`BinaryGrid.default` (finite natal log-density
        at the Beta singularity).
    peaks
        Tuple of ``(center, width, weight)`` Gaussian bumps in the target
        density.  Defaults: natal ``(0.05, 0.05, 1)`` + cascade
        ``(0.6865, 0.04, 1)``.
    floor_frac
        Fraction of the target density mass assigned to the uniform floor
        (the rest goes to the bumps).  ``0`` would cluster nodes entirely
        in the peaks and starve the valley; the ``0.25`` default keeps a
        usable backbone.  Must satisfy ``0 < floor_frac < 1``.
    n_fine
        Resolution of the internal fine grid used to build the cumulative.

    Returns
    -------
    Array
        Sorted ``(n_chi,)`` χ nodes with ``nodes[0] == chi_min`` and
        ``nodes[-1] == chi_max``.
    """
    if not (0.0 < floor_frac < 1.0):
        raise ValueError(f"floor_frac must be in (0, 1); got {floor_frac}")
    if n_chi < 2:
        raise ValueError(f"n_chi must be >= 2; got {n_chi}")

    chi_fine = np.linspace(chi_min, chi_max, n_fine)
    span = chi_max - chi_min

    bump = np.zeros_like(chi_fine)
    for center, width, weight in peaks:
        bump += weight * np.exp(-0.5 * ((chi_fine - center) / width) ** 2)
    # Trapezoid integral (manual — avoids the np.trapz/np.trapezoid
    # deprecation churn across numpy versions).
    bump_integral = float(
        np.sum(0.5 * (bump[1:] + bump[:-1]) * np.diff(chi_fine))
    )
    if bump_integral <= 0.0:
        # Degenerate (no peaks / zero weight) → fall back to uniform.
        return jnp.asarray(np.linspace(chi_min, chi_max, n_chi))

    # Target node-density: floor (integrates to floor_frac) + normalised
    # bumps (integrate to 1 − floor_frac).  Total integral = 1.
    g = floor_frac / span + (1.0 - floor_frac) * bump / bump_integral

    # Cumulative via trapezoid; normalise to [0, 1].
    G = np.concatenate(
        [[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(chi_fine))]
    )
    G /= G[-1]

    # Equidistribution: invert G at equally spaced cumulative targets.
    targets = np.linspace(0.0, 1.0, n_chi)
    nodes = np.interp(targets, G, chi_fine)
    nodes[0] = chi_min
    nodes[-1] = chi_max
    return jnp.asarray(nodes)


class BinaryGrid(eqx.Module):
    """Fixed grid over (m, χ) for storing per-generation densities n_k(m, χ).

    Stored as 1D edge arrays plus pre-computed log-edges; convolutions
    elsewhere broadcast on the cross product.
    """

    m: Array  # shape (N_m,), solar masses
    chi: Array  # shape (N_chi,)
    log_m: Array  # cached log(m) for log-spacing convolutions

    @classmethod
    def default(
        cls,
        m_min: float = 2.0,
        m_max: float = 200.0,
        n_m: int = 96,
        n_chi: int = 32,
        chi_min: float = 1e-3,
        chi_max: float = 1.0 - 1e-3,
    ) -> "BinaryGrid":
        """Default (m, χ) grid.

        ``chi_min`` and ``chi_max`` are shifted slightly inside the unit
        interval so the natal Beta spin distribution — which has a PDF
        singularity at ``χ = 0`` whenever its shape parameter ``α < 1`` —
        evaluates to a finite log-density on every grid point. The default
        offset (1e-3) is well below the resolution of any realistic spin
        measurement and propagates negligible bias.
        """
        m = jnp.geomspace(m_min, m_max, n_m)
        chi = jnp.linspace(chi_min, chi_max, n_chi)
        return cls(m=m, chi=chi, log_m=jnp.log(m))

    @classmethod
    def clustered_chi(
        cls,
        m_min: float = 2.0,
        m_max: float = 200.0,
        n_m: int = 96,
        n_chi: int = 32,
        chi_min: float = 1e-3,
        chi_max: float = 1.0 - 1e-3,
        peaks: tuple[tuple[float, float, float], ...] = _DEFAULT_CHI_PEAKS,
        floor_frac: float = 0.25,
    ) -> "BinaryGrid":
        """``(m, χ)`` grid with the χ axis clustered at the spin peaks.

        Identical to :meth:`default` except the χ nodes are placed by
        :func:`clustered_chi_nodes` — concentrated at the natal peak
        (``χ ≈ 0.05``) and the cascade remnant peak (``χ ≈ 0.69``) — so a
        coarse ``n_chi`` resolves the narrow ``dN/dχ`` cascade structure
        that a uniform grid blurs.  This is the "believable spin" grid: the
        downstream scatter, integration and F2 surrogate are all exact on
        an arbitrary monotone χ grid (they reach it only via
        ``jnp.interp`` / ``jnp.diff``), so swapping the nodes is the whole
        change.

        The m axis, endpoints and the ``chi_min`` / ``chi_max`` Beta-edge
        offsets are unchanged from :meth:`default`, so a clustered grid is
        a drop-in for any call site that takes a ``grid=`` argument
        (including the gwkokab bridge's ``make_coagulation_dist_fn``).
        """
        m = jnp.geomspace(m_min, m_max, n_m)
        chi = clustered_chi_nodes(
            n_chi=n_chi,
            chi_min=chi_min,
            chi_max=chi_max,
            peaks=peaks,
            floor_frac=floor_frac,
        )
        return cls(m=m, chi=chi, log_m=jnp.log(m))


class RedshiftGrid(eqx.Module):
    """Fixed grid over redshift z, used by formation-rate and convolution layers."""

    z: Array  # shape (N_z,)

    @classmethod
    def default(cls, z_max: float = 4.0, n_z: int = 64) -> "RedshiftGrid":
        return cls(z=jnp.linspace(0.0, z_max, n_z))


class EnvGrid(eqx.Module):
    """Fixed grid over environment parameters (v_esc for v1).

    Future env parameters (SMBH mass, NSC mass, disk surface density, ...)
    will extend this; for v1 only ``v_esc`` is varied.
    """

    v_esc: Array  # shape (N_ve,), km/s

    @classmethod
    def default(
        cls, v_esc_min: float = 1.0, v_esc_max: float = 1000.0, n_ve: int = 24
    ) -> "EnvGrid":
        return cls(v_esc=jnp.geomspace(v_esc_min, v_esc_max, n_ve))
