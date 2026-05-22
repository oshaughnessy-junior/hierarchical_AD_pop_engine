r"""JS / L1 / peak-location metrics with an engine-derived bulk mask.

Used by ``tests/test_redteam_observables_parity.py`` (Phase C of
``demo/PLAN_redteam_observable_validation.md``) to compare engine and
particle-MC histograms in the regime where both algorithms resolve
the underlying physics cleanly.

The bulk mask
-------------
For every observable, the comparison region is the set of bins where
the engine density is above a small fraction ``tail_frac`` of its own
peak::

    bulk_mask = engine_density >= tail_frac · max(engine_density)

Two design notes:

1. **Mask from the engine, not the MC.**  Using the MC density to
   define the bulk would let MC shot noise (or, worse, a real bias
   under test) leak into the mask itself and hide its own effect.
   The engine density is smooth and reproducible; that's what we
   anchor on.
2. **Fixed ``tail_frac`` per test.**  Per ``feedback_no_fudge_factors``,
   ``tail_frac`` is set once in the test file at ``0.01`` and never
   tuned per run.  If a test fails inside the resulting bulk, the
   answer is to fix the physics, not to widen the mask.

The bulk mask covers ``≥ 90 %`` of the engine total rate at the
default ``tail_frac=0.01`` for every observable considered here;
:func:`bulk_mask_coverage_fraction` returns this number for
diagnostic logging.
"""
from __future__ import annotations

import numpy as np

from red_team.phase1_overlay import js_divergence


# ---------------------------------------------------------------------------
# Bulk mask.
# ---------------------------------------------------------------------------
def bulk_mask(
    engine_density: np.ndarray,
    *,
    tail_frac: float = 0.01,
) -> np.ndarray:
    r"""Boolean mask: cells with engine density ≥ ``tail_frac`` · peak.

    Same shape as ``engine_density``.  Cells outside the mask are
    considered "tail" and excluded from JS / L1 metrics — they are
    where MC shot noise and engine truncation dominate and the per-
    bin comparison is uninformative.
    """
    peak = float(np.max(engine_density)) if engine_density.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(engine_density, dtype=bool)
    return engine_density >= tail_frac * peak


def bulk_mask_coverage_fraction(
    engine_density: np.ndarray,
    bin_widths_axes: list[np.ndarray] | None = None,
    *,
    tail_frac: float = 0.01,
) -> float:
    r"""Fraction of the engine's *integrated* density inside the bulk mask.

    Used as a sanity check that the mask captures the physics.  At
    the default ``tail_frac=0.01`` this should be ``≥ 0.9`` for the
    observables considered in this plan; if it isn't, the bulk
    threshold is too aggressive for the chosen binning.

    ``bin_widths_axes`` is a list of 1-D bin-width arrays — one per
    axis of ``engine_density``.  If omitted, we treat the bins as
    unit-area (the fraction is then count-fraction, not rate-fraction).
    """
    mask = bulk_mask(engine_density, tail_frac=tail_frac)
    if bin_widths_axes is None:
        return float(np.sum(engine_density[mask]) / np.sum(engine_density))
    widths_outer = bin_widths_axes[0]
    for w in bin_widths_axes[1:]:
        widths_outer = np.multiply.outer(widths_outer, w)
    total = float(np.sum(engine_density * widths_outer))
    if total <= 0.0:
        return 0.0
    bulk = float(np.sum(engine_density * widths_outer * mask))
    return bulk / total


# ---------------------------------------------------------------------------
# JS on grid, restricted to bulk.
# ---------------------------------------------------------------------------
def js_on_grid_bulk(
    engine_density: np.ndarray,
    mc_density: np.ndarray,
    axis_values: np.ndarray,
    *,
    tail_frac: float = 0.01,
) -> float:
    r"""JS divergence between two 1-D densities on the bulk of the engine.

    Bulk mask is derived from ``engine_density`` at ``tail_frac``.
    Densities are renormalised inside the mask before the JS is
    computed (the metric reports *shape* disagreement, not amplitude).
    Returns ``NaN`` if fewer than 4 cells fall inside the mask.
    """
    mask = bulk_mask(engine_density, tail_frac=tail_frac)
    if int(mask.sum()) < 4:
        return float("nan")
    return float(
        js_divergence(
            engine_density[mask], mc_density[mask], axis_values[mask]
        )
    )


# ---------------------------------------------------------------------------
# Relative-L1, restricted to bulk.  For the 2-D observables — JS on a
# 2-D grid is awkward to renormalise without arbitrary marginalisation
# choices, but the bulk-L1 metric is well-defined and the natural
# scalar summary of agreement.
# ---------------------------------------------------------------------------
def relative_L1_bulk(
    engine_density: np.ndarray,
    mc_density: np.ndarray,
    bin_widths_axes: list[np.ndarray],
    *,
    tail_frac: float = 0.01,
) -> float:
    r"""``Σ |engine - mc| · widths inside bulk / Σ engine · widths inside bulk``.

    Closer to zero is better.  ``> 1`` indicates the disagreement is
    larger than the engine signal itself in the bulk — usually a
    catastrophic failure of the comparison (units bug, missing
    rate_scale conversion, etc.).
    """
    mask = bulk_mask(engine_density, tail_frac=tail_frac)
    widths_outer = bin_widths_axes[0]
    for w in bin_widths_axes[1:]:
        widths_outer = np.multiply.outer(widths_outer, w)
    if not mask.any():
        return float("nan")
    diff = float(np.sum(np.abs(engine_density - mc_density) * widths_outer * mask))
    base = float(np.sum(engine_density * widths_outer * mask))
    if base <= 0.0:
        return float("nan")
    return diff / base


# ---------------------------------------------------------------------------
# Total-rate ratio.
# ---------------------------------------------------------------------------
def total_rate_ratio(
    engine_density: np.ndarray,
    mc_density: np.ndarray,
    bin_widths_axes: list[np.ndarray],
) -> float:
    r"""``∫ mc_density / ∫ engine_density`` (over the full grid).

    ``1.0`` means the MC and engine agree on the total event rate.
    The Phase C tests allow a ±15 % window per the plan, reflecting
    MC shot noise at the chosen ``n_seed = 20 000``.
    """
    widths_outer = bin_widths_axes[0]
    for w in bin_widths_axes[1:]:
        widths_outer = np.multiply.outer(widths_outer, w)
    eng = float(np.sum(engine_density * widths_outer))
    mc = float(np.sum(mc_density * widths_outer))
    if eng <= 0.0:
        return float("nan")
    return mc / eng


# ---------------------------------------------------------------------------
# Peak-location.
# ---------------------------------------------------------------------------
def peak_location(
    density: np.ndarray, axis_values: np.ndarray
) -> float:
    r"""Axis value at the argmax of a 1-D density.

    Returns ``NaN`` if the density is all zero / NaN.
    """
    if density.size == 0 or not np.isfinite(density).any():
        return float("nan")
    idx = int(np.nanargmax(density))
    return float(axis_values[idx])


def peak_location_2d(
    density: np.ndarray,
    axis_a_values: np.ndarray,
    axis_b_values: np.ndarray,
) -> tuple[float, float]:
    r"""``(axis_a, axis_b)`` value at the argmax of a 2-D density."""
    if density.size == 0 or not np.isfinite(density).any():
        return (float("nan"), float("nan"))
    flat_idx = int(np.nanargmax(density))
    i, j = np.unravel_index(flat_idx, density.shape)
    return float(axis_a_values[i]), float(axis_b_values[j])


__all__ = [
    "bulk_mask",
    "bulk_mask_coverage_fraction",
    "js_on_grid_bulk",
    "relative_L1_bulk",
    "total_rate_ratio",
    "peak_location",
    "peak_location_2d",
]
