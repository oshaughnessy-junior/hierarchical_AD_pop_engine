"""Tests for the clustered (nonuniform) χ grid — the "believable spin" grid.

Invariant-based (no frozen numerical references — see
``feedback-no-frozen-test-references``).  Every assertion is an algebraic
identity (monotone nodes, partition of unity, trapezoid exactness on a
linear integrand), a placement inequality (denser at the spin peaks than
in the valley), or a *comparative* statement (clustered beats uniform on a
sharp test density at matched node count).

This file imports only :mod:`grids` and
:func:`orientation_precompute._bilinear_chi_hats`, both of which are
gwkokab-free, so the whole module runs in the lightweight JAX sandbox.
The AD-through-the-solver and Fisher-bridge checks that need the full
engine live in ``test_fisher_bridge_clustered.py`` (engine tier, skips if
gwkokab is unavailable).
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.grids import (  # noqa: E402
    BinaryGrid,
    clustered_chi_nodes,
    _CHI_NATAL_PEAK,
    _CHI_CASCADE_PEAK,
)
from hierarchical_ad_pop_engine.orientation_precompute import (  # noqa: E402
    _bilinear_chi_hats,
)

CHI_MIN = 1e-3
CHI_MAX = 1.0 - 1e-3


# ---------------------------------------------------------------------------
# Node placement.
# ---------------------------------------------------------------------------
def test_nodes_strictly_increasing_and_endpoints():
    nodes = np.asarray(clustered_chi_nodes(n_chi=32))
    assert np.all(np.diff(nodes) > 0), "nodes must be strictly increasing"
    assert nodes[0] == pytest.approx(CHI_MIN)
    assert nodes[-1] == pytest.approx(CHI_MAX)


def test_nodes_cluster_at_both_spin_peaks():
    """More nodes near the natal (χ≈0.05) and cascade (χ≈0.69) peaks than
    in the flat valley between them."""
    nodes = np.asarray(clustered_chi_nodes(n_chi=32))
    near = lambda c, h=0.06: int(np.sum(np.abs(nodes - c) < h))
    n_natal = near(_CHI_NATAL_PEAK)
    n_cascade = near(_CHI_CASCADE_PEAK)
    n_valley = near(0.35)  # flat region, no bump
    assert n_natal > n_valley
    assert n_cascade > n_valley


def test_floor_frac_validation():
    with pytest.raises(ValueError):
        clustered_chi_nodes(floor_frac=0.0)
    with pytest.raises(ValueError):
        clustered_chi_nodes(floor_frac=1.0)


def test_degenerate_peaks_fall_back_to_uniform():
    """Zero-weight bumps → the generator returns a uniform grid (no NaNs,
    still monotone)."""
    nodes = np.asarray(
        clustered_chi_nodes(n_chi=16, peaks=((0.5, 0.1, 0.0),))
    )
    uni = np.linspace(CHI_MIN, CHI_MAX, 16)
    assert np.allclose(nodes, uni)


def test_node_count_honoured():
    for n in (8, 16, 32, 64):
        assert clustered_chi_nodes(n_chi=n).shape == (n,)


def test_max_gap_cap_enforced():
    """No cell wider than max_gap_factor × the uniform spacing — the guard
    against the χ→1 tail artifact (wide tail cell cell-averages the steep
    post-cascade falloff onto the χ=1 node)."""
    span = CHI_MAX - CHI_MIN
    for n in (12, 24, 32):
        for fac in (1.5, 2.0, 3.0):
            nodes = np.asarray(
                clustered_chi_nodes(n_chi=n, max_gap_factor=fac)
            )
            assert np.max(np.diff(nodes)) <= fac * span / (n - 1) * (1 + 1e-6)


def test_max_gap_cap_disabled_reproduces_uncapped():
    a = np.asarray(clustered_chi_nodes(n_chi=32, max_gap_factor=None))
    b = np.asarray(clustered_chi_nodes(n_chi=32, max_gap_factor=float("inf")))
    assert np.allclose(a, b)
    # and the uncapped grid genuinely violates the default cap (so the cap
    # is doing real work at the default n_chi)
    span = CHI_MAX - CHI_MIN
    assert np.max(np.diff(a)) > 2.0 * span / 31


# ---------------------------------------------------------------------------
# Partition of unity of the production bilinear-hat scatter on the
# nonuniform nodes — the mass-conservation guarantee the gain/F2 paths rely
# on.  This is the load-bearing invariant: it is what makes the entire χ
# machinery exact on an arbitrary monotone grid.
# ---------------------------------------------------------------------------
def test_bilinear_hats_partition_of_unity_on_clustered_grid():
    chi = clustered_chi_nodes(n_chi=32)
    xq = jnp.linspace(float(chi[0]), float(chi[-1]), 501)
    hats = _bilinear_chi_hats(xq, chi)  # (N_chi, 501)
    psum = jnp.sum(hats, axis=0)
    assert float(jnp.max(jnp.abs(psum - 1.0))) < 1e-10
    assert bool(jnp.all(hats >= -1e-12))


def test_bilinear_hats_zero_outside_grid():
    chi = clustered_chi_nodes(n_chi=16)
    xq = jnp.asarray([CHI_MIN - 0.1, CHI_MAX + 0.1])
    hats = _bilinear_chi_hats(xq, chi)
    assert float(jnp.max(jnp.abs(hats))) < 1e-12


# ---------------------------------------------------------------------------
# Trapezoid integration on the nonuniform grid (the loss-side measure).
# Replicated inline (3 lines) to keep this file gwkokab-free — identical to
# ``response.implicit._dchi_weights``.
# ---------------------------------------------------------------------------
def _dchi_weights(chi):
    d = jnp.diff(chi)
    return jnp.concatenate([d[:1] * 0.5, 0.5 * (d[:-1] + d[1:]), d[-1:] * 0.5])


def test_trapezoid_weights_exact_on_constant_and_linear():
    chi = clustered_chi_nodes(n_chi=32)
    w = _dchi_weights(chi)
    span = float(chi[-1] - chi[0])
    assert float(jnp.sum(w)) == pytest.approx(span, abs=1e-12)
    # trapezoid is exact for linear integrands on any grid
    exact = 0.5 * (float(chi[-1]) ** 2 - float(chi[0]) ** 2)
    assert float(jnp.sum(w * chi)) == pytest.approx(exact, abs=1e-10)


# ---------------------------------------------------------------------------
# The headline: clustered resolves a sharp two-peak density better than a
# uniform grid of the SAME node count.  Comparative invariant — no frozen
# number, just "clustered < uniform".
# ---------------------------------------------------------------------------
def test_clustered_beats_uniform_on_sharp_two_peak_density():
    n_chi = 32

    def truth(x):
        return (
            np.exp(-0.5 * ((x - _CHI_NATAL_PEAK) / 0.04) ** 2)
            + np.exp(-0.5 * ((x - _CHI_CASCADE_PEAK) / 0.03) ** 2)
        )

    xfine = np.linspace(CHI_MIN, CHI_MAX, 6001)
    tfine = truth(xfine)

    def _trap(y, x):
        return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))

    def rel_l1(grid_nodes):
        approx = np.interp(xfine, grid_nodes, truth(grid_nodes))
        return _trap(np.abs(approx - tfine), xfine) / _trap(tfine, xfine)

    err_uniform = rel_l1(np.linspace(CHI_MIN, CHI_MAX, n_chi))
    err_clustered = rel_l1(np.asarray(clustered_chi_nodes(n_chi=n_chi)))
    assert err_clustered < err_uniform


# ---------------------------------------------------------------------------
# BinaryGrid.clustered_chi is a drop-in for BinaryGrid.default.
# ---------------------------------------------------------------------------
def test_clustered_grid_is_drop_in_for_default():
    g_def = BinaryGrid.default(m_min=4.0, m_max=80.0, n_m=32, n_chi=20)
    g_clu = BinaryGrid.clustered_chi(m_min=4.0, m_max=80.0, n_m=32, n_chi=20)
    # same m axis + shapes; only χ nodes differ
    assert g_clu.m.shape == g_def.m.shape
    assert g_clu.chi.shape == g_def.chi.shape
    assert jnp.allclose(g_clu.m, g_def.m)
    assert jnp.allclose(g_clu.log_m, g_def.log_m)
    assert not jnp.allclose(g_clu.chi, g_def.chi)
    assert bool(jnp.all(jnp.diff(g_clu.chi) > 0))
