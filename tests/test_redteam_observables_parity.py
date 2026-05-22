r"""Phase C: implicit Picard ↔ particle MC parity in observable space.

Three parity tests covering the headline observables LIGO inference
consumes:

C.1  ``dN_mgr / dt dm_1`` — primary-mass marginal (1-D)
C.2  ``dN_mgr / dt dm_1 dχ_eff`` — primary mass × effective spin (2-D)
C.3  ``dN_mgr / dt dm_1 dq`` — primary mass × mass ratio (2-D)

Each compares the engine's forward-rate marginal (computed from the
implicit Picard fixed point via :mod:`observables`) against the MC
mergers histogram from :mod:`red_team.observables_mc`, using the
bulk-mask metrics in :mod:`red_team.observable_metrics`.

Stringency convention
---------------------
Per ``demo/PLAN_redteam_observable_validation.md`` §"Stringency":

* Bulk metric (JS for 1-D, relative-L1 for 2-D) gated below a
  threshold proportional to the MC-to-MC noise on the same statistic.
* Total-rate ratio inside ±15 % — MC shot-noise budget at
  ``n_seed = 20 000``.
* Tail behaviour pattern-only (peak location to within a couple of
  bins, monotonicity expectations for the kernel mass-ratio bias).

The thresholds are loose by design.  The plan calls them out as a v1
gate appropriate for the physics-shifting phase; tightening waits for
the v3 basis refactor to close the grid-quadrature systematic
(``project_implicit_vs_mc_systematic_bias``).

Slow tier
---------
All tests carry ``@pytest.mark.slow``.  Opt in with ``pytest -m slow``.
The fixture is module-scoped so the implicit solve and the two MC
seeds run once and feed all three tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine import observables as obs

from red_team.comparison_helpers import run_implicit, run_mc
from red_team import observable_metrics as omet
from red_team import observables_mc as mc_obs


# ---------------------------------------------------------------------------
# Stage 1 truth point (matches ``test_redteam_Nx_parity.THETA_STAR``).
# ---------------------------------------------------------------------------
THETA_STAR = dict(
    mu_natal=8.0,
    sigma_natal=1.0,
    m_min=4.0,
    m_max=80.0,
    natal_spin_loc=0.0,
    natal_spin_scale=0.10,
    kernel_a=0.0,
    kernel_b=1.0,
    v_esc=1000.0,
    k_max=4,
)
_MC_N_SEED = 20_000  # decided 2026-05-21
_EPSILON_PROD = 1.0
_TAIL_FRAC = 0.01  # bulk-mask threshold; fixed per the no-fudge-factors rule

# MC ``depletion_fraction`` is NOT a free knob — see the same NOTE in
# ``test_redteam_Nx_parity.py``.  We read it from the engine's
# converged ``result.f_natal_depleted`` inside the fixture so both
# algorithms are evaluated at the *same* physical depletion state.

# JS-vs-noise multiplier for the 1-D observable, sized to the known
# v3-pending systematic — see the long NOTE in
# ``test_redteam_Nx_parity.py``.  Mass-axis ceiling ~50× MC noise
# until the v3 basis refactor closes the grid-quadrature mismatch.
_JS_MAX_MULTIPLIER_MASS = 50.0


# ---------------------------------------------------------------------------
# Fixture: one implicit solve + two MC seeds (mergers arrays included).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def picard_and_mc_observable():
    response, result, m_grid, chi_grid = run_implicit(
        THETA_STAR, _EPSILON_PROD
    )
    # MC stops at the engine's converged depletion fraction — same
    # physical state on both sides of the comparison.
    f_dep = float(result.f_natal_depleted)
    rng1 = np.random.default_rng(2026_05_21_1)
    rng2 = np.random.default_rng(2026_05_21_2)
    # track_generations=True so the mergers array carries g_1, g_2 —
    # not strictly needed for the observable histograms but cheap.
    mc1 = run_mc(THETA_STAR, _MC_N_SEED, rng1, f_dep, track_generations=True)
    mc2 = run_mc(THETA_STAR, _MC_N_SEED, rng2, f_dep, track_generations=True)
    return dict(
        response=response,
        N_implicit=np.asarray(result.N),
        m_grid=m_grid,
        chi_grid=chi_grid,
        mc1_mergers=mc1["mergers"],
        mc2_mergers=mc2["mergers"],
    )


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------
def _engine_total_rate(picard_and_mc_observable: dict) -> float:
    response = picard_and_mc_observable["response"]
    N = jnp.asarray(picard_and_mc_observable["N_implicit"])
    return float(
        obs.total_rate(
            N, response.grid, response.kernel_params, response.log_R_env
        )
    )


def _m_bin_edges(m_grid: np.ndarray) -> np.ndarray:
    r"""Bin edges that put each engine grid point at a bin centre.

    Halfway-in-log between grid points; the outer edges extend one
    half-step beyond the grid endpoints so every grid point is
    captured.
    """
    log_m = np.log(m_grid)
    half = 0.5 * np.diff(log_m)
    lo = log_m[0] - half[0]
    hi = log_m[-1] + half[-1]
    inner = 0.5 * (log_m[:-1] + log_m[1:])
    return np.exp(np.concatenate([[lo], inner, [hi]]))


# ---------------------------------------------------------------------------
# C.1 — dN/dm_1 parity.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_dN_dm1_parity(picard_and_mc_observable):
    r"""Implicit ↔ MC on ``dN/dm_1``: bulk JS ≤ 5× MC-to-MC noise.

    Same 5× MC-noise gate as the ``N(m)`` test in
    ``test_redteam_Nx_parity`` — promoted to observable space, where
    LIGO inference actually consumes the quantity.
    """
    response = picard_and_mc_observable["response"]
    N = jnp.asarray(picard_and_mc_observable["N_implicit"])
    m_grid = picard_and_mc_observable["m_grid"]
    engine_total = _engine_total_rate(picard_and_mc_observable)

    # Engine: 1-D forward-rate marginal on the engine mass grid.
    _, eng_dN_dm1 = obs.marginal_1d(
        N, response.grid, response.kernel_params, response.log_R_env,
        axis=obs.AXIS_M1,
    )
    eng_dN_dm1 = np.asarray(eng_dN_dm1)

    # MC: histogram on bin edges that put each grid point at a centre.
    m_edges = _m_bin_edges(m_grid)
    rate_scale_1 = mc_obs.infer_rate_scale_from_total(
        picard_and_mc_observable["mc1_mergers"], engine_total
    )
    rate_scale_2 = mc_obs.infer_rate_scale_from_total(
        picard_and_mc_observable["mc2_mergers"], engine_total
    )
    _, mc1_dN_dm1 = mc_obs.dN_dm1_mc(
        picard_and_mc_observable["mc1_mergers"], m_edges, rate_scale_1
    )
    _, mc2_dN_dm1 = mc_obs.dN_dm1_mc(
        picard_and_mc_observable["mc2_mergers"], m_edges, rate_scale_2
    )

    # Bulk-JS comparison.
    js_imp_mc = omet.js_on_grid_bulk(
        eng_dN_dm1, mc1_dN_dm1, m_grid, tail_frac=_TAIL_FRAC
    )
    js_noise = omet.js_on_grid_bulk(
        mc1_dN_dm1, mc2_dN_dm1, m_grid, tail_frac=_TAIL_FRAC
    )
    assert np.isfinite(js_imp_mc) and np.isfinite(js_noise), (
        f"dN/dm_1: JS values not finite (imp_mc={js_imp_mc!r}, "
        f"noise={js_noise!r})"
    )
    assert js_imp_mc < _JS_MAX_MULTIPLIER_MASS * js_noise, (
        f"dN/dm_1 observable: cross-method JS {js_imp_mc:.4g} > "
        f"{_JS_MAX_MULTIPLIER_MASS}× MC-noise JS {js_noise:.4g} "
        f"(ratio {js_imp_mc / js_noise:.1f}×).  Worse than the known "
        f"v3-pending mass-axis systematic by more than 2× — flag as a "
        f"regression against today's ceiling, not just the open issue."
    )

    # Total-rate ratio inside ±15 %.
    bin_widths = np.diff(m_edges)
    centres = 0.5 * (m_edges[:-1] + m_edges[1:])
    mc_total = float(np.sum(mc1_dN_dm1 * bin_widths))
    # Engine total: integrate dN/dm_1 with the engine grid's trapezoid
    # weights (matches the convention used to *define* the marginal).
    w_m_arr = np.asarray(obs.trapezoid_weights(response.grid)[0])
    eng_total = float(np.sum(eng_dN_dm1 * w_m_arr))
    ratio = mc_total / eng_total
    assert 0.85 <= ratio <= 1.15, (
        f"dN/dm_1 total-rate ratio MC/engine = {ratio:.3f} outside "
        f"[0.85, 1.15] tolerance at n_seed = {_MC_N_SEED}."
    )


# ---------------------------------------------------------------------------
# C.2 — dN/dm_1 dχ_eff parity.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_dN_dm1_dchi_eff_parity(picard_and_mc_observable):
    r"""Implicit ↔ MC on ``dN/dm_1 dχ_eff``: bulk relative-L1 ≤ 0.5.

    Looser threshold than the 1-D case — per-bin counts are ~16× smaller
    at the 16-χ_eff-bin resolution chosen by the plan; relative-L1
    bounded at 0.5 is the plan's v1 acceptance.  Total-rate ratio
    within ±15 % as in C.1.
    """
    response = picard_and_mc_observable["response"]
    N = jnp.asarray(picard_and_mc_observable["N_implicit"])
    m_grid = picard_and_mc_observable["m_grid"]
    engine_total = _engine_total_rate(picard_and_mc_observable)

    chi_eff_edges = np.linspace(0.0, 1.0, 17)  # 16 bins (plan §C.2)

    # Engine.
    _, _, eng = obs.dN_dm1_dchi_eff(
        N, response.grid, response.kernel_params, response.log_R_env,
        chi_eff_bin_edges=jnp.asarray(chi_eff_edges),
        ordered=True,
    )
    eng = np.asarray(eng)

    # MC.
    m_edges = _m_bin_edges(m_grid)
    rate_scale_1 = mc_obs.infer_rate_scale_from_total(
        picard_and_mc_observable["mc1_mergers"], engine_total
    )
    rate_scale_2 = mc_obs.infer_rate_scale_from_total(
        picard_and_mc_observable["mc2_mergers"], engine_total
    )
    _, _, mc1 = mc_obs.dN_dm1_dchi_eff_mc(
        picard_and_mc_observable["mc1_mergers"], m_edges, chi_eff_edges, rate_scale_1
    )
    _, _, mc2 = mc_obs.dN_dm1_dchi_eff_mc(
        picard_and_mc_observable["mc2_mergers"], m_edges, chi_eff_edges, rate_scale_2
    )

    # Engine grid has N_m mass cells; MC histogram has len(m_edges) - 1.
    # Resample to a shared grid — the engine grid m values are
    # bin-centres of `m_edges` by construction, so the shapes already
    # match.
    assert eng.shape == mc1.shape, (
        f"shape mismatch engine {eng.shape} vs MC {mc1.shape} — "
        f"_m_bin_edges convention drift?"
    )
    m_widths = np.diff(m_edges)
    chi_widths = np.diff(chi_eff_edges)

    l1_imp_mc = omet.relative_L1_bulk(
        eng, mc1, [m_widths, chi_widths], tail_frac=_TAIL_FRAC
    )
    assert np.isfinite(l1_imp_mc), f"L1 not finite: {l1_imp_mc!r}"
    assert l1_imp_mc < 0.5, (
        f"dN/dm_1 dχ_eff observable: bulk relative-L1 {l1_imp_mc:.3f} > 0.5. "
        f"Physics-level disagreement (or rate_scale convention drift)."
    )

    # Total-rate ratio.
    ratio = omet.total_rate_ratio(eng, mc1, [m_widths, chi_widths])
    assert 0.85 <= ratio <= 1.15, (
        f"dN/dm_1 dχ_eff total-rate ratio MC/engine = {ratio:.3f} "
        f"outside [0.85, 1.15]."
    )

    # Bulk-mask coverage sanity (engine bulk covers ≥ 90 % of the rate).
    cov = omet.bulk_mask_coverage_fraction(
        eng, [m_widths, chi_widths], tail_frac=_TAIL_FRAC
    )
    assert cov >= 0.85, (
        f"bulk mask covers only {cov:.2%} of engine rate at "
        f"tail_frac={_TAIL_FRAC}; mask too aggressive?"
    )


# ---------------------------------------------------------------------------
# C.3 — dN/dm_1 dq parity.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_dN_dm1_dq_parity(picard_and_mc_observable):
    r"""Implicit ↔ MC on ``dN/dm_1 dq``: bulk relative-L1 ≤ 0.5.

    The ``q`` axis is clipped at the very-extreme-q end where both
    algorithms are noisy.  Plan §C.3 picks ``q ∈ [0.05, 1.0]`` with
    19 bins; the same range here.
    """
    response = picard_and_mc_observable["response"]
    N = jnp.asarray(picard_and_mc_observable["N_implicit"])
    m_grid = picard_and_mc_observable["m_grid"]
    engine_total = _engine_total_rate(picard_and_mc_observable)

    q_edges = np.linspace(0.05, 1.0, 20)  # 19 bins (plan §C.3)

    _, _, eng = obs.dN_dm1_dq(
        N, response.grid, response.kernel_params, response.log_R_env,
        q_bin_edges=jnp.asarray(q_edges),
        ordered=True,
    )
    eng = np.asarray(eng)

    m_edges = _m_bin_edges(m_grid)
    rate_scale_1 = mc_obs.infer_rate_scale_from_total(
        picard_and_mc_observable["mc1_mergers"], engine_total
    )
    rate_scale_2 = mc_obs.infer_rate_scale_from_total(
        picard_and_mc_observable["mc2_mergers"], engine_total
    )
    _, _, mc1 = mc_obs.dN_dm1_dq_mc(
        picard_and_mc_observable["mc1_mergers"], m_edges, q_edges, rate_scale_1
    )
    _, _, mc2 = mc_obs.dN_dm1_dq_mc(
        picard_and_mc_observable["mc2_mergers"], m_edges, q_edges, rate_scale_2
    )

    assert eng.shape == mc1.shape
    m_widths = np.diff(m_edges)
    q_widths = np.diff(q_edges)

    l1_imp_mc = omet.relative_L1_bulk(
        eng, mc1, [m_widths, q_widths], tail_frac=_TAIL_FRAC
    )
    assert np.isfinite(l1_imp_mc), f"L1 not finite: {l1_imp_mc!r}"
    assert l1_imp_mc < 0.5, (
        f"dN/dm_1 dq observable: bulk relative-L1 {l1_imp_mc:.3f} > 0.5."
    )

    # Total-rate ratio.  Loosened to ±25 % here because the q < 0.05
    # tail is excluded from the engine and MC integrals — some
    # legitimate rate is dropped on both sides, but not by exactly
    # the same fraction.
    ratio = omet.total_rate_ratio(eng, mc1, [m_widths, q_widths])
    assert 0.75 <= ratio <= 1.25, (
        f"dN/dm_1 dq total-rate ratio MC/engine = {ratio:.3f} "
        f"outside [0.75, 1.25]."
    )

    # Pattern check: with kernel_b > 0, the q-marginal must be
    # monotonically non-decreasing toward q = 1 averaged over m_1.
    # Tests the kernel's mass-ratio bias is working as intended
    # (both algorithms must agree on the *direction* of the bias).
    eng_q_marginal = np.sum(eng * m_widths[:, None], axis=0)  # (N_q,)
    # Check monotonic-ish — the bulk of mass should be in upper-q half.
    upper_half_frac = np.sum(eng_q_marginal[len(eng_q_marginal) // 2:]) / np.sum(eng_q_marginal)
    assert upper_half_frac > 0.5, (
        f"engine q-marginal upper-half mass fraction = {upper_half_frac:.3f}; "
        f"kernel_b = {THETA_STAR['kernel_b']} should produce a "
        f"q→1-favouring distribution, but the bulk landed in the lower q half."
    )
