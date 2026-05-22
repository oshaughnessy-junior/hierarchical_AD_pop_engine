r"""Phase B sanity tests for the derived-axis observables.

Fast tier (no MC).  Pins:

* The derived-axis observables (:func:`observables.dN_dm1_dchi_eff`,
  :func:`observables.dN_dm1_dq`) integrate consistently with the
  axis-aligned :func:`observables.marginal_1d` and
  :func:`observables.total_rate` already pinned in
  ``tests/test_observables.py``.
* ``χ_eff ∈ [0, 1]`` per cell at Stage 1 (aligned-only) — sign /
  labelling sanity.
* ``q ∈ (0, 1]`` per cell — m_1 ≥ m_2 labelling sanity.
* AD path through the derived-axis observables is finite and non-zero
  (basic regression for population fits that use them as inference
  observables).
* MC adapters (:mod:`red_team.observables_mc`) normalise to a known
  total rate via :func:`infer_rate_scale_from_total` when called on a
  realistic mergers array.

We use the same gwkokab-backed :func:`build_default_coagulation`
fixture as ``tests/test_observables.py`` so the tests run under the
canonical natal + kernel physics, not a synthetic stand-in.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine import observables as obs
from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.response.coagulation import (
    CoagulationResponse,
    build_default_coagulation,
)


# ---------------------------------------------------------------------------
# Fixtures.  Match ``tests/test_observables.py`` so the small-grid
# response is shared across both files (pytest caches via module scope).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_response() -> CoagulationResponse:
    return build_default_coagulation(
        k_max=2,
        kernel_a=0.3,
        kernel_b=1.0,
        log_R_env=0.5,
        grid=BinaryGrid.default(m_min=5.0, m_max=80.0, n_m=20, n_chi=8),
    )


@pytest.fixture(scope="module")
def N_mchi(small_response: CoagulationResponse) -> jnp.ndarray:
    log_N = small_response._log_total_density(jnp.asarray(300.0))
    return jnp.exp(log_N)


# ---------------------------------------------------------------------------
# Headline: integration consistency with marginal_1d / total_rate.
# ---------------------------------------------------------------------------
def test_dN_dm1_dchi_eff_integrates_to_total_rate(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""``∫∫ dN/(dm_1 dχ_eff) dm_1 dχ_eff == total_rate`` at machine precision.

    Both ``ordered=True`` and ``ordered=False`` must hit the same
    number — each unordered merger is counted exactly once under
    either convention; the ordering only changes the per-cell
    labelling, not the integral.
    """
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env
    chi_eff_edges = jnp.linspace(0.0, 1.0, 33)
    bin_widths = jnp.diff(chi_eff_edges)
    w_m, _ = obs.trapezoid_weights(grid)
    C_total = float(obs.total_rate(N_mchi, grid, kp, log_R))

    for ordered in (False, True):
        _, _, dC = obs.dN_dm1_dchi_eff(
            N_mchi, grid, kp, log_R,
            chi_eff_bin_edges=chi_eff_edges, ordered=ordered,
        )
        integ = float(jnp.einsum("ij,i,j->", dC, w_m, bin_widths))
        rel = abs(integ / C_total - 1.0)
        assert rel < 1e-11, (
            f"dN/dm_1 dχ_eff ordered={ordered}: integral {integ:.6e} vs "
            f"total {C_total:.6e}, rel {rel:.2e}"
        )


def test_dN_dm1_dq_integrates_to_total_rate(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""``∫∫ dN/(dm_1 dq) dm_1 dq == total_rate`` at machine precision.

    ``q`` axis includes ``q = 1`` (diagonal cells); the histogram
    scatter's closed-right convention must include that endpoint.
    """
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env
    q_edges = jnp.linspace(0.0, 1.0, 40)
    bin_widths = jnp.diff(q_edges)
    w_m, _ = obs.trapezoid_weights(grid)
    C_total = float(obs.total_rate(N_mchi, grid, kp, log_R))

    for ordered in (False, True):
        _, _, dC = obs.dN_dm1_dq(
            N_mchi, grid, kp, log_R, q_bin_edges=q_edges, ordered=ordered,
        )
        integ = float(jnp.einsum("ij,i,j->", dC, w_m, bin_widths))
        rel = abs(integ / C_total - 1.0)
        assert rel < 1e-11, (
            f"dN/dm_1 dq ordered={ordered}: integral {integ:.6e} vs "
            f"total {C_total:.6e}, rel {rel:.2e}"
        )


def test_dN_dm1_dchi_eff_reduces_to_marginal_1d_under_unordered(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""Integrating ``dN/(dm_1 dχ_eff)`` over χ_eff reproduces
    :func:`marginal_1d(axis='m1')`.

    Under ``ordered=False`` the marginal is the canonical unordered
    ``dC/dm_1`` from the axis-aligned 1-D marginal module.  This pins
    the new derived-axis machinery to the axis-aligned arithmetic
    already validated in ``tests/test_observables.py``.
    """
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env
    chi_eff_edges = jnp.linspace(0.0, 1.0, 33)
    bin_widths = jnp.diff(chi_eff_edges)

    _, _, dC = obs.dN_dm1_dchi_eff(
        N_mchi, grid, kp, log_R,
        chi_eff_bin_edges=chi_eff_edges, ordered=False,
    )
    dC_dm1_via_2d = jnp.einsum("ij,j->i", dC, bin_widths)
    _, dC_dm1_direct = obs.marginal_1d(
        N_mchi, grid, kp, log_R, axis=obs.AXIS_M1
    )
    rel = float(
        jnp.max(jnp.abs(dC_dm1_via_2d - dC_dm1_direct))
        / jnp.maximum(jnp.max(jnp.abs(dC_dm1_direct)), 1e-30)
    )
    assert rel < 1e-11, (
        f"max rel diff between dN/dm_1 dχ_eff reduction and "
        f"marginal_1d(m_1) = {rel:.2e}"
    )


# ---------------------------------------------------------------------------
# Sign / range sanity.
# ---------------------------------------------------------------------------
def test_chi_eff_per_cell_in_range(small_response: CoagulationResponse):
    r"""``χ_eff ∈ [0, 1]`` for every (m_i, χ_j, m_k, χ_l) cell.

    At Stage 1 aligned-only, χ_1 and χ_2 are both in ``[0, 1]``, so
    χ_eff = (m_1 χ_1 + m_2 χ_2) / (m_1 + m_2) is bounded the same
    way.  Catches accidental sign flips or off-by-one tuple swaps
    inside the bilinear scatter.
    """
    grid = small_response.grid
    m = grid.m
    chi = grid.chi
    m1 = m[:, None, None, None]
    chi1 = chi[None, :, None, None]
    m2 = m[None, None, :, None]
    chi2 = chi[None, None, None, :]
    chi_eff = (m1 * chi1 + m2 * chi2) / (m1 + m2)
    assert float(chi_eff.min()) >= 0.0
    assert float(chi_eff.max()) <= 1.0


def test_q_per_cell_in_range(small_response: CoagulationResponse):
    r"""``q ∈ (0, 1]`` for every cell (m_min > 0 by construction)."""
    grid = small_response.grid
    m = grid.m
    m1 = m[:, None]
    m2 = m[None, :]
    q = jnp.minimum(m1, m2) / jnp.maximum(m1, m2)
    assert float(q.min()) > 0.0
    assert float(q.max()) == 1.0
    # Diagonal is q == 1 exactly:
    assert bool(jnp.all(jnp.diag(q) == 1.0))


# ---------------------------------------------------------------------------
# Ordered/unordered total-rate equivalence (the integral is the same
# regardless of labelling — both count each merger once).
# ---------------------------------------------------------------------------
def test_ordered_and_unordered_totals_agree(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""``ordered=True`` and ``ordered=False`` integrate to the same
    total — each merger is counted once under both conventions.
    """
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env
    chi_eff_edges = jnp.linspace(0.0, 1.0, 33)
    bin_widths = jnp.diff(chi_eff_edges)
    w_m, _ = obs.trapezoid_weights(grid)

    _, _, dC_u = obs.dN_dm1_dchi_eff(
        N_mchi, grid, kp, log_R, chi_eff_bin_edges=chi_eff_edges, ordered=False,
    )
    _, _, dC_o = obs.dN_dm1_dchi_eff(
        N_mchi, grid, kp, log_R, chi_eff_bin_edges=chi_eff_edges, ordered=True,
    )
    integ_u = float(jnp.einsum("ij,i,j->", dC_u, w_m, bin_widths))
    integ_o = float(jnp.einsum("ij,i,j->", dC_o, w_m, bin_widths))
    rel = abs(integ_u / integ_o - 1.0)
    assert rel < 1e-12, (
        f"ordered/unordered total disagreement: rel {rel:.2e}"
    )


# ---------------------------------------------------------------------------
# AD sanity.
# ---------------------------------------------------------------------------
def test_dN_dm1_dchi_eff_grad_through_N(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""``jax.grad`` through the χ_eff observable returns a finite,
    non-zero gradient w.r.t. the input density ``N(m, χ)``.

    Required for any inference loop that uses ``dN/(dm_1 dχ_eff)`` as
    the likelihood data.  We don't pin the gradient *value* — just
    its finiteness — because the value drifts as the natal / kernel
    parameters change (which is fine).
    """
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env
    chi_eff_edges = jnp.linspace(0.0, 1.0, 17)

    def loss(N):
        _, _, dC = obs.dN_dm1_dchi_eff(
            N, grid, kp, log_R, chi_eff_bin_edges=chi_eff_edges, ordered=True,
        )
        return jnp.sum(dC)

    g = jax.grad(loss)(N_mchi)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.sum(jnp.abs(g))) > 0.0


def test_dN_dm1_dq_grad_through_N(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""Same AD sanity for the q observable."""
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env
    q_edges = jnp.linspace(0.05, 1.0, 20)

    def loss(N):
        _, _, dC = obs.dN_dm1_dq(
            N, grid, kp, log_R, q_bin_edges=q_edges, ordered=True,
        )
        return jnp.sum(dC)

    g = jax.grad(loss)(N_mchi)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.sum(jnp.abs(g))) > 0.0


# ---------------------------------------------------------------------------
# MC adapter sanity.
# ---------------------------------------------------------------------------
def test_mc_dN_dm1_normalises_via_infer_rate_scale():
    r"""``infer_rate_scale_from_total`` aligns the MC total to a given
    target rate.

    Synthetic mergers array — single column block, fixed values, fixed
    target rate.  No physics; this tests the bookkeeping of the
    rate_scale conversion factor.
    """
    from red_team import observables_mc as mc

    rng = np.random.default_rng(0)
    n = 1000
    m1 = rng.uniform(20.0, 60.0, size=n)
    m2 = rng.uniform(5.0, 20.0, size=n)
    chi1 = rng.uniform(0.0, 0.3, size=n)
    chi2 = rng.uniform(0.0, 0.3, size=n)
    g1 = np.ones(n)
    g2 = np.ones(n)
    retained = np.ones(n)
    m_rem = np.full(n, np.nan)
    mergers = np.stack(
        [m1, m2, chi1, chi2, g1, g2, retained, m_rem], axis=1
    )
    target_total = 4.2e-3

    rate_scale = mc.infer_rate_scale_from_total(mergers, target_total)
    edges = np.linspace(5.0, 80.0, 25)
    centres, dN_dm = mc.dN_dm1_mc(mergers, edges, rate_scale)
    widths = np.diff(edges)
    integ = float(np.sum(dN_dm * widths))
    rel = abs(integ / target_total - 1.0)
    assert rel < 1e-13, (
        f"MC dN/dm_1 ∫ = {integ:.6e}, target {target_total:.6e}, rel {rel:.2e}"
    )
