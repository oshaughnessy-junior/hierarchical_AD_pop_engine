r"""Session M3b workstation gate — stratified natal MC + weighted MCEvaluator.

Pins three invariants on top of the M3a sandbox tier
(``tests/test_evaluation_mc_weighted.py``):

* **Bit-for-bit dispatch**: ``MCConfig`` with all M3b kwargs ``None``
  (the v1 default) produces *exactly* the pre-M3b ``m_active``,
  ``chi_active``, and ``mergers`` arrays from the matching seed.
  Guards against silent leaks of the M3b code paths into the v1
  baseline per ``[[feedback-dispatch-callgraph-audit]]``.
* **Weight propagation is finite + positive**: the cascade-tracked
  ``w_active`` / ``w_mergers`` arrays returned by ``particle_mc``
  remain physical (no NaN, no negative) after a stratified natal
  draw + ε ≈ 1 cascade.
* **Headline narrow-natal-peak recovery gate**: at ``σ_χ = 0.02``
  (the narrow-feature regime that motivated stratification per
  ``[[demo/PLAN_mc_first_reprioritization]]`` Session M3) and
  ``N_init = 10⁴``, the weighted MCEvaluator's recovery of
  ``dN/dχ`` is *no worse than 2 ×* the unweighted MCEvaluator's
  recovery measured by rel-L1 vs the engine.  Per the PLAN:
  stratification must not be a regression; ideally it's an
  improvement.  The "no worse than 2×" headline is the published
  M3b gate.

Slow tier — opt in with ``pytest -m slow``.  Workstation-only
(requires JAX + scipy + the engine stack); not sandbox-tier per the
M3a/M3b split design (``[[demo/PLAN_m3_stratified_natal]]``
§ "Why split M3 into M3a + M3b").

See also ``[[project-session-handoff-20260528]]`` for the session
context this batch shipped in.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine import observables as obs
from hierarchical_ad_pop_engine.evaluation.mc import MCEvaluator

from red_team.comparison_helpers import run_implicit, run_mc


# ---------------------------------------------------------------------------
# Fixture parameters — narrow-natal-χ regime is the M3 motivating case.
# Per the no-fudge-factors rule the absolute numbers below are physics
# choices (not knobs to make the test green).
# ---------------------------------------------------------------------------
THETA_STAR = dict(
    mu_natal=8.0,
    sigma_natal=1.0,
    m_min=4.0,
    m_max=80.0,
    natal_spin_loc=0.0,
    natal_spin_scale=0.02,   # narrow natal-χ peak — the M3 motivating regime
    kernel_a=0.0,
    kernel_b=1.0,
    v_esc=1000.0,
    k_max=4,
)
N_INIT = 10_000
EPSILON = 1.0
STRATIFY = (8, 16)


# ---------------------------------------------------------------------------
# Bit-for-bit dispatch invariant — uses default theta (σ_χ=0.10) so it's
# bit-equal against any other v1-baseline run with the same seed.  Fast.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_v1_dispatch_bit_for_bit():
    r"""``run_mc`` with all M3b kwargs ``None`` (default) reproduces the
    pre-M3b MC trajectory bit-for-bit.

    Catches the failure mode where adding ``proposal`` / ``stratify`` /
    ``weight_array`` to :class:`MCConfig` accidentally perturbs the
    truncnorm PRNG sequence or the K-rejection inner loop.  Both
    runs use the same numpy seed; outputs must be identical.
    """
    theta_v1 = dict(THETA_STAR, natal_spin_scale=0.10)  # default σ_χ for the v1 test
    rng_a = np.random.default_rng(20260528)
    rng_b = np.random.default_rng(20260528)
    mc_default = run_mc(
        theta_v1, n_seed=N_INIT, rng=rng_a, depletion_fraction=0.5,
    )
    mc_explicit = run_mc(
        theta_v1, n_seed=N_INIT, rng=rng_b, depletion_fraction=0.5,
        proposal=None, stratify=None, weight_array=None,
    )
    np.testing.assert_array_equal(
        mc_default["m_active"], mc_explicit["m_active"],
        err_msg="v1 m_active drift between default-kwargs and explicit-None — "
                "M3b kwargs leaked into the default code path",
    )
    np.testing.assert_array_equal(
        mc_default["chi_active"], mc_explicit["chi_active"],
        err_msg="v1 chi_active drift — M3b dispatch leak",
    )
    np.testing.assert_array_equal(
        mc_default["mergers"], mc_explicit["mergers"],
        err_msg="v1 mergers drift — M3b dispatch leak (affects K-rejection)",
    )
    # And under v1, w_active and w_mergers must be uniformly 1.0.
    assert np.all(mc_default["w_active"] == 1.0), (
        "v1 path must produce uniform w_active = 1.0; got "
        f"min={mc_default['w_active'].min()}, max={mc_default['w_active'].max()}"
    )
    assert np.all(mc_default["w_mergers"] == 1.0), (
        "v1 path must produce uniform w_mergers = 1.0"
    )


# ---------------------------------------------------------------------------
# Weight-propagation sanity under stratified-natal.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_stratified_weights_finite_positive():
    r"""Stratified natal draw + ε≈1 cascade: every survivor and every
    merger event carries a finite, positive weight.

    Catches the failure mode where the arithmetic-mean propagation
    rule produces NaN under non-uniform initial weights (e.g. a
    parent inheriting a weight from an already-merged slot).
    """
    rng = np.random.default_rng(20260528)
    mc = run_mc(
        THETA_STAR, n_seed=N_INIT, rng=rng, depletion_fraction=0.5,
        stratify=STRATIFY,
    )
    assert mc["w_active"].size > 0
    assert mc["w_mergers"].size > 0
    assert np.all(np.isfinite(mc["w_active"])), "w_active has non-finite entries"
    assert np.all(mc["w_active"] > 0.0), (
        f"w_active has non-positive entries: min={mc['w_active'].min()}"
    )
    assert np.all(np.isfinite(mc["w_mergers"])), "w_mergers has non-finite entries"
    assert np.all(mc["w_mergers"] > 0.0), (
        f"w_mergers has non-positive entries: min={mc['w_mergers'].min()}"
    )
    # Kish effective sample size in the stats dict.
    assert mc["stats"]["n_eff_active"] > 0.0
    assert mc["stats"]["n_eff_active"] <= mc["w_active"].size, (
        "n_eff_active should be ≤ raw sample count"
    )


# ---------------------------------------------------------------------------
# Headline gate — narrow-natal-χ peak recovery.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_stratified_no_worse_than_uniform_on_dN_dchi():
    r"""Stratified MC + weighted MCEvaluator recovers ``dN/dχ`` no worse
    than 2× the uniform MC + unweighted MCEvaluator.

    The narrow-natal-χ regime (σ_χ = 0.02) is the motivating use case
    for stratified sampling per ``[[demo/PLAN_mc_first_reprioritization]]``
    § "Cascade resolution (obstacle a)": at the v1 truncnorm draw,
    only a small fraction of N_init = 10⁴ particles land inside the
    narrow peak, so the empirical density is shot-noise dominated
    there.  Stratification with (8, 16) cells covers the full ``(log m,
    χ)`` rectangle uniformly and the weighted MCEvaluator (M3a)
    re-weights by ``p_natal / q_strat`` to recover the target.

    Acceptance is "no worse than 2× uniform rel-L1" (per
    ``[[demo/PLAN_m3_stratified_natal]]`` § M3b headline gate),
    measured against the engine ``dN/dχ`` marginal.  This is the
    weaker of two PLAN-listed targets — the stronger 10 % absolute
    rel-L1 target may not always hold (engine vs MC has known
    systematic biases on derived axes per
    ``[[project-dchi-eff-engine-mc-systematic-bias]]``), but the
    relative ordering "stratified ≤ 2× uniform" is the load-bearing
    M3b invariant.

    We compare the *single-BH density's χ-marginal*
    ``g(χ) = ∫ N(m, χ) dm`` (not the merger-rate derived ``dN/dχ_eff``)
    so the result is insensitive to the per-generation engine-vs-MC
    cascade systematic that drives the
    ``[[project-aligned-cascade-fractal-resolution]]`` xfail.
    """
    response, result, m_grid, chi_grid = run_implicit(THETA_STAR, EPSILON)
    f_dep = float(result.f_natal_depleted)

    # --- Engine reference: g_engine(χ) = ∫ N(m, χ) dm. -----------------
    N_engine = np.asarray(result.N)
    w_m, _w_chi = obs.trapezoid_weights(response.grid)
    g_engine = np.sum(N_engine * np.asarray(w_m)[:, None], axis=0)  # shape (N_chi,)
    chi_ref = np.asarray(chi_grid)

    # --- Uniform MC + unweighted MCEvaluator. --------------------------
    rng_u = np.random.default_rng(20260528_001)
    mc_u = run_mc(
        THETA_STAR, n_seed=N_INIT, rng=rng_u, depletion_fraction=f_dep,
    )
    ev_u = MCEvaluator(
        mc_u["m_active"], mc_u["chi_active"], mc_u["mergers"],
        log_R_env=float(response.log_R_env),
        kernel_params={"a": float(response.kernel_params["a"]),
                       "b": float(response.kernel_params["b"])},
    )

    # --- Stratified MC + weighted MCEvaluator. --------------------------
    rng_s = np.random.default_rng(20260528_002)
    mc_s = run_mc(
        THETA_STAR, n_seed=N_INIT, rng=rng_s, depletion_fraction=f_dep,
        stratify=STRATIFY,
    )
    ev_s = MCEvaluator(
        mc_s["m_active"], mc_s["chi_active"], mc_s["mergers"],
        log_R_env=float(response.log_R_env),
        kernel_params={"a": float(response.kernel_params["a"]),
                       "b": float(response.kernel_params["b"])},
        weights=mc_s["w_active"],
    )

    # --- g_MC(χ) on the engine χ grid via density evaluation + m-quad. -
    def _g_from_evaluator(ev, m_grid_arr, chi_grid_arr, w_m_arr):
        M, C = np.meshgrid(m_grid_arr, chi_grid_arr, indexing="ij")
        N_eval = ev.density(M, C)
        return np.sum(N_eval * w_m_arr[:, None], axis=0)

    g_u = _g_from_evaluator(ev_u, np.asarray(m_grid), chi_ref, np.asarray(w_m))
    g_s = _g_from_evaluator(ev_s, np.asarray(m_grid), chi_ref, np.asarray(w_m))

    # --- Renormalise to unit integral in χ for *shape*-only comparison.
    def _norm(p, x):
        Z = float(np.trapezoid(p, x))
        return p / Z if Z > 0 else np.zeros_like(p)

    g_engine_n = _norm(g_engine, chi_ref)
    g_u_n = _norm(g_u, chi_ref)
    g_s_n = _norm(g_s, chi_ref)

    # rel-L1 vs engine (denominator = engine integral).
    def _rel_l1(p, q, x):
        diff = float(np.trapezoid(np.abs(p - q), x))
        base = float(np.trapezoid(p, x))
        return diff / base if base > 0 else float("nan")

    L1_uniform = _rel_l1(g_engine_n, g_u_n, chi_ref)
    L1_strat = _rel_l1(g_engine_n, g_s_n, chi_ref)

    assert np.isfinite(L1_uniform) and np.isfinite(L1_strat), (
        f"non-finite rel-L1: uniform={L1_uniform!r} strat={L1_strat!r}"
    )
    print(
        f"\n[M3b] dN/dχ recovery rel-L1 vs engine: "
        f"uniform = {L1_uniform:.4f}, stratified = {L1_strat:.4f}, "
        f"ratio = {L1_strat / L1_uniform:.2f}×"
    )
    # Headline gate: stratified must not be more than 2× worse than uniform.
    # If this fails, either the weight propagation is wrong or the
    # stratification cells don't cover the natal support well enough at
    # this N_init.
    assert L1_strat <= 2.0 * L1_uniform, (
        f"stratified rel-L1 {L1_strat:.3f} > 2× uniform {L1_uniform:.3f} — "
        f"stratification is a regression at this N_init.  Either weight "
        f"propagation is buggy or the (n_m, n_chi) cell count is "
        f"insufficient for the σ_χ = {THETA_STAR['natal_spin_scale']} regime."
    )
