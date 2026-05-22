r"""Particle-MC histogram adapters for the observable comparison.

For continuous (grid / basis) representations the merger-rate
marginals are 4-D quadrature contractions of ``½ R_env K N N`` (see
:mod:`hierarchical_ad_pop_engine.observables`).  For the particle MC
the marginals are **trivial**: each accepted merger row in
``backup_particle_mc.particle_mc(...).mergers`` is one physical event
parameterised by ``(m_1, m_2, χ_1, χ_2, g_1, g_2, retained, m_rem)``
with the ``m_1 ≥ m_2`` convention enforced at write-time.  We compute
observable histograms by raw ``np.histogram`` / ``np.histogram2d`` on
those rows.

The functions in this module each consume a ``mergers`` array as
produced by :func:`backup_particle_mc.particle_mc` and return a
histogram in the same shape and units conventions as their continuous
counterparts in :mod:`observables`, so the two paths can be compared
on equal footing.

Rate scale
----------
The MC counts merger *events* during a finite simulated cascade; the
continuous engine returns a per-environment merger *rate density*
(per time).  Converting between them requires a single multiplicative
``rate_scale`` factor that depends on the natal pool size, the kernel
amplitude, and the implicit-solver time scale ``ε``.  Per the
no-fudge-factors rule we expose ``rate_scale`` as an explicit caller-
supplied argument; it is *not* tuned per comparison.  The convention
matches :mod:`compare_implicit_vs_mc` exactly: ``rate_scale`` is the
factor that makes ``Σ_i histogram[i] · rate_scale`` equal the
engine's ``total_rate`` at the same ``θ``.  See the helper
:func:`infer_rate_scale_from_total` for the bookkeeping if you don't
want to derive it by hand.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Mergers-array column convention.  Mirrors ``backup_particle_mc.particle_mc``
# line 151.  Centralised here so any reordering shows up in one place.
# ---------------------------------------------------------------------------
COL_M1 = 0
COL_M2 = 1
COL_CHI1 = 2
COL_CHI2 = 3
COL_G1 = 4
COL_G2 = 5
COL_RETAINED = 6
COL_M_REM = 7


def _split_mergers(mergers: np.ndarray) -> tuple[np.ndarray, ...]:
    r"""Return the (m_1, m_2, χ_1, χ_2) columns under m_1 ≥ m_2.

    No additional sorting needed — :func:`backup_particle_mc.particle_mc`
    already records each event heavier-first.  We re-sort defensively
    in case a caller hands in a custom array.
    """
    m1 = mergers[:, COL_M1].astype(np.float64)
    m2 = mergers[:, COL_M2].astype(np.float64)
    chi1 = mergers[:, COL_CHI1].astype(np.float64)
    chi2 = mergers[:, COL_CHI2].astype(np.float64)
    swap = m1 < m2
    if np.any(swap):
        m1, m2 = np.where(swap, m2, m1), np.where(swap, m1, m2)
        chi1, chi2 = np.where(swap, chi2, chi1), np.where(swap, chi1, chi2)
    return m1, m2, chi1, chi2


# ---------------------------------------------------------------------------
# 1-D: dN_mgr / dm_1.
# ---------------------------------------------------------------------------
def dN_dm1_mc(
    mergers: np.ndarray,
    m_bin_edges: np.ndarray,
    rate_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""``dN/dm_1`` from a particle-MC mergers array.

    Parameters
    ----------
    mergers : np.ndarray, shape ``(N_accept, 8)``
        Output of :func:`backup_particle_mc.particle_mc`.  ``m_1 ≥ m_2``
        already enforced at write-time.
    m_bin_edges : np.ndarray, shape ``(N_bins + 1,)``
        Bin edges on the ``m_1`` axis.  Typical: the engine's
        ``response.grid.m`` extended at top / bottom by one bin so the
        histogram covers all events.
    rate_scale : float
        Multiplicative conversion from per-bin event count → per-bin
        rate density (so ``Σ_i hist[i] · bin_width[i] · rate_scale``
        equals the engine's ``total_rate``).

    Returns
    -------
    bin_centres : np.ndarray, shape ``(N_bins,)``
    dN_dm1 : np.ndarray, shape ``(N_bins,)``
        Rate density on the ``m_1`` axis.
    """
    m1, _, _, _ = _split_mergers(mergers)
    counts, edges = np.histogram(m1, bins=m_bin_edges)
    widths = np.diff(edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, counts.astype(np.float64) * rate_scale / widths


# ---------------------------------------------------------------------------
# 2-D: dN_mgr / dm_1 dχ_eff.
# ---------------------------------------------------------------------------
def dN_dm1_dchi_eff_mc(
    mergers: np.ndarray,
    m_bin_edges: np.ndarray,
    chi_eff_bin_edges: np.ndarray,
    rate_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""``dN/(dm_1 dχ_eff)`` from a particle-MC mergers array.

    ``χ_eff = (m_1 χ_1 + m_2 χ_2) / (m_1 + m_2)``, aligned-only at
    Stage 1 (range ``[0, 1]``).

    Returns
    -------
    m_centres : np.ndarray, shape ``(N_m_bins,)``
    chi_eff_centres : np.ndarray, shape ``(N_chi_eff_bins,)``
    density : np.ndarray, shape ``(N_m_bins, N_chi_eff_bins)``
        Forward-rate density on the ``(m_1, χ_eff)`` plane.
    """
    m1, m2, chi1, chi2 = _split_mergers(mergers)
    chi_eff = (m1 * chi1 + m2 * chi2) / (m1 + m2)
    counts, m_edges, chi_edges = np.histogram2d(
        m1, chi_eff, bins=[m_bin_edges, chi_eff_bin_edges]
    )
    m_widths = np.diff(m_edges)
    chi_widths = np.diff(chi_edges)
    density = counts * rate_scale / (m_widths[:, None] * chi_widths[None, :])
    m_centres = 0.5 * (m_edges[:-1] + m_edges[1:])
    chi_centres = 0.5 * (chi_edges[:-1] + chi_edges[1:])
    return m_centres, chi_centres, density


# ---------------------------------------------------------------------------
# 2-D: dN_mgr / dm_1 dq.
# ---------------------------------------------------------------------------
def dN_dm1_dq_mc(
    mergers: np.ndarray,
    m_bin_edges: np.ndarray,
    q_bin_edges: np.ndarray,
    rate_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""``dN/(dm_1 dq)`` from a particle-MC mergers array.

    ``q = m_{\rm light} / m_{\rm heavy} = m_2 / m_1`` under the
    ``m_1 ≥ m_2`` convention.  ``q ∈ (0, 1]``.

    Returns
    -------
    m_centres, q_centres, density : np.ndarray
    """
    m1, m2, _, _ = _split_mergers(mergers)
    # m_1 ≥ m_2 already; q ∈ (0, 1].
    q = m2 / m1
    counts, m_edges, q_edges = np.histogram2d(
        m1, q, bins=[m_bin_edges, q_bin_edges]
    )
    m_widths = np.diff(m_edges)
    q_widths = np.diff(q_edges)
    density = counts * rate_scale / (m_widths[:, None] * q_widths[None, :])
    m_centres = 0.5 * (m_edges[:-1] + m_edges[1:])
    q_centres = 0.5 * (q_edges[:-1] + q_edges[1:])
    return m_centres, q_centres, density


# ---------------------------------------------------------------------------
# Rate-scale bookkeeping helper.
# ---------------------------------------------------------------------------
def infer_rate_scale_from_total(
    mergers: np.ndarray,
    engine_total_rate: float,
) -> float:
    r"""Return the multiplicative factor that aligns MC counts to the
    engine total rate.

    Useful when you don't want to derive ``rate_scale`` from first
    principles (natal pool size × time × kernel amplitude).  Computes
    ``engine_total_rate / N_accept``, so the MC histograms — when
    summed over all bins and multiplied by the bin widths — match the
    engine's total rate exactly.

    Caveat: this is a normalisation-only scale, not a physics
    derivation.  Use it for shape comparisons (JS, L1) where the
    overall amplitude is not the science target; do NOT use it as a
    physics conversion factor in inference.  See
    ``compare_implicit_vs_mc.py`` for the principled derivation when
    the absolute rate matters.
    """
    n_accept = int(mergers.shape[0])
    if n_accept == 0:
        return 0.0
    return float(engine_total_rate) / n_accept


__all__ = [
    "COL_M1",
    "COL_M2",
    "COL_CHI1",
    "COL_CHI2",
    "COL_G1",
    "COL_G2",
    "COL_RETAINED",
    "COL_M_REM",
    "dN_dm1_mc",
    "dN_dm1_dchi_eff_mc",
    "dN_dm1_dq_mc",
    "infer_rate_scale_from_total",
]
