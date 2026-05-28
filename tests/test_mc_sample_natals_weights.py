r"""Session M3b sandbox-tier tests — weight bookkeeping in ``sample_natals``.

Pins the four invariants from ``demo/PLAN_m3_stratified_natal.md`` §
M3b that are exercise-able without JAX (the engine + ``particle_mc``
itself imports JAX; the sampler helpers are pure numpy + scipy):

1. **v1 bit-for-bit** — default ``MCConfig(...)`` (no proposal,
   stratify, or weight_array) yields the pre-M3b truncnorm sequence
   and uniform weights = 1.0.
2. **Stratified non-negativity + zero-weight filter** — stratified
   draws over ``[m_min, m_max] × [0, 1]`` produce strictly positive
   weights after the underflow filter (truncnorm logpdf can reach
   −2500 in the tail; ``exp`` rounds to 0 and those samples are
   filtered).
3. **IS-integrand recovery** — for a normalised target ``p_natal``,
   the stratified-IS estimator of ``∫ p_natal dm dχ`` is ≈ 1 (Σw / n
   ratio), and the weighted-sample mean of ``m`` recovers the
   truncnorm mean of ``m`` within Monte-Carlo SE.
4. **Validation** — negative weights from ``weight_array`` or
   non-finite weights from a bad proposal raise ``ValueError`` with
   a useful message.

All tests are numpy + scipy only — imports are guarded with an
``importlib`` extraction so this file runs even where JAX is absent
(matches the M3a sandbox file pattern).  Workstation-tier coverage
of the full particle-MC cascade weight propagation lives in
``tests/test_mc_stratified_natal.py``.

See also:

* ``[[demo/PLAN_m3_stratified_natal.md]]`` — design and invariants.
* ``[[project-session-handoff-20260528]]`` — session context.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

scipy_stats = pytest.importorskip(
    "scipy.stats", reason="scipy required for truncnorm-backed natal pdf",
)
truncnorm = scipy_stats.truncnorm


# ---------------------------------------------------------------------------
# Pure-numpy extraction: lift ``MCConfig`` + the M3b sampler helpers out of
# ``red_team/backup_particle_mc.py`` without importing the JAX-dependent
# module body (JAX is unavailable in the Cowork VM).  Mirrors the M3a
# pattern in ``tests/test_evaluation_mc_weighted.py``.
# ---------------------------------------------------------------------------
_BPM_PATH = (
    Path(__file__).resolve().parent.parent
    / "red_team" / "backup_particle_mc.py"
)


def _extract_helpers() -> dict:
    src = _BPM_PATH.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    wanted_fn = {
        "_natal_log_pdf", "_stratified_natal", "sample_natals", "_kish_n_eff",
    }
    wanted_class = {"MCConfig"}
    parts = []
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef) and node.name in wanted_fn) or \
           (isinstance(node, ast.ClassDef) and node.name in wanted_class):
            start_line = (
                node.decorator_list[0].lineno - 1
                if node.decorator_list else node.lineno - 1
            )
            parts.append("\n".join(lines[start_line:node.end_lineno]))
    mod_src = (
        "from __future__ import annotations\n"
        "from dataclasses import dataclass, field\n"
        "from typing import Callable, Optional, Tuple\n"
        "import numpy as np\n"
        "from scipy.stats import truncnorm\n\n"
        + "\n\n".join(parts)
    )
    ns: dict = {}
    exec(mod_src, ns)
    return ns


_NS = _extract_helpers()
MCConfig = _NS["MCConfig"]
sample_natals = _NS["sample_natals"]
_kish_n_eff = _NS["_kish_n_eff"]


# ---------------------------------------------------------------------------
# Shared fixture — Stage 1 θ* with the v1 σ_χ.
# ---------------------------------------------------------------------------
THETA = dict(
    mu_natal=8.0, sigma_natal=1.0, m_min=4.0, m_max=80.0,
    natal_spin_loc=0.0, natal_spin_scale=0.10,
    kernel_a=0.0, kernel_b=1.0, v_esc=1000.0,
)
N = 5000


# ---------------------------------------------------------------------------
# INV-1: v1 bit-for-bit.
# ---------------------------------------------------------------------------
def test_v1_default_bit_for_bit():
    r"""``MCConfig(...)`` with all M3b kwargs ``None`` reproduces the
    pre-M3b truncnorm draw sequence + uniform weights = 1.0.

    Regression defence: the v1 code path inside ``sample_natals``
    must not be perturbed by the addition of the
    ``proposal`` / ``stratify`` / ``weight_array`` branches.
    """
    cfg = MCConfig(n_seed=N, **THETA)
    m, chi, w = sample_natals(cfg, np.random.default_rng(2026))
    assert m.shape == chi.shape == w.shape == (N,)
    np.testing.assert_array_equal(w, np.ones(N))

    # Independent re-derivation of the v1 truncnorm sequence — must match
    # bit-for-bit because nothing in sample_natals' default branch was
    # reordered.
    rng_ref = np.random.default_rng(2026)
    a_m = (THETA["m_min"] - THETA["mu_natal"]) / THETA["sigma_natal"]
    b_m = (THETA["m_max"] - THETA["mu_natal"]) / THETA["sigma_natal"]
    m_ref = truncnorm.rvs(
        a_m, b_m, loc=THETA["mu_natal"], scale=THETA["sigma_natal"],
        size=N, random_state=rng_ref,
    ).astype(np.float64)
    a_c = (0.0 - THETA["natal_spin_loc"]) / THETA["natal_spin_scale"]
    b_c = (1.0 - THETA["natal_spin_loc"]) / THETA["natal_spin_scale"]
    chi_ref = truncnorm.rvs(
        a_c, b_c, loc=THETA["natal_spin_loc"], scale=THETA["natal_spin_scale"],
        size=N, random_state=rng_ref,
    ).astype(np.float64)
    np.testing.assert_array_equal(m, m_ref)
    np.testing.assert_array_equal(chi, chi_ref)


# ---------------------------------------------------------------------------
# INV-2: stratified non-negativity + underflow filter.
# ---------------------------------------------------------------------------
def test_stratified_positive_after_filter():
    r"""Stratified draws over the full ``(log m, χ)`` rectangle are
    strictly positive *after* the underflow filter.

    At ``mu_natal=8, sigma=1, m_max=80`` the truncnorm logpdf at
    m≈80 reaches ~−2500 and exp underflows to 0.0 in float64.
    ``sample_natals`` filters these zero-weight samples so the
    output satisfies the strictly-positive contract of
    :class:`MCEvaluator` (M3a).
    """
    cfg = MCConfig(n_seed=N, stratify=(8, 16), **THETA)
    m, chi, w = sample_natals(cfg, np.random.default_rng(2026))
    n_per_cell = N // (8 * 16)
    n_drawn = 8 * 16 * n_per_cell
    assert m.size <= n_drawn
    assert m.size > 0
    assert np.all(np.isfinite(w))
    assert np.all(w > 0.0), f"post-filter weights must be > 0; min={w.min()}"
    # And the filter must have actually dropped something — at θ* the
    # high-m tail is many σ above the mean, so most cells in the
    # high-m band have zero p_natal.
    assert m.size < n_drawn, (
        "expected the underflow filter to drop *some* tail samples — "
        "if it didn't, either the natal distribution is wider than "
        "expected or the filter is mis-wired"
    )


# ---------------------------------------------------------------------------
# INV-3: importance-integrand recovery.
# ---------------------------------------------------------------------------
def test_stratified_is_recovers_natal_mean():
    r"""Stratified IS recovers ``E_p_natal[m]`` to within MC SE.

    Two parts:

    * Σ w / n_drawn ≈ 1 (estimate of ``∫ p_natal dm dχ = 1``).
    * ``Σ w·m / Σ w`` ≈ truncnorm.mean(m) (weighted MC estimate of
      the natal mass mean).

    Both are O(1) checks at N_drawn ≈ N_seed; the test budgets
    Monte-Carlo SE generously (within 30 % for ratio, within
    0.3 M_⊙ for ⟨m⟩).
    """
    cfg = MCConfig(n_seed=N, stratify=(8, 16), **THETA)
    m, chi, w = sample_natals(cfg, np.random.default_rng(2026))
    n_per_cell = N // (8 * 16)
    n_drawn = 8 * 16 * n_per_cell
    sum_w = float(w.sum())
    print(f"\n[INV-3] stratified IS:  Σw = {sum_w:.1f}  vs n_drawn = {n_drawn}")
    assert 0.7 * n_drawn <= sum_w <= 1.3 * n_drawn

    a_m = (THETA["m_min"] - THETA["mu_natal"]) / THETA["sigma_natal"]
    b_m = (THETA["m_max"] - THETA["mu_natal"]) / THETA["sigma_natal"]
    true_m_mean = float(truncnorm.mean(
        a_m, b_m, loc=THETA["mu_natal"], scale=THETA["sigma_natal"],
    ))
    weighted_m_mean = float(np.sum(w * m) / np.sum(w))
    print(
        f"[INV-3] ⟨m⟩ weighted stratified MC = {weighted_m_mean:.4f}, "
        f"true = {true_m_mean:.4f}"
    )
    assert abs(weighted_m_mean - true_m_mean) < 0.3


# ---------------------------------------------------------------------------
# INV-4: validation errors.
# ---------------------------------------------------------------------------
def test_validation_negative_weight_array():
    bad = np.ones(N)
    bad[0] = -0.5
    cfg = MCConfig(n_seed=N, weight_array=bad, **THETA)
    with pytest.raises(ValueError, match=r"negative weight"):
        sample_natals(cfg, np.random.default_rng(2026))


def test_validation_inf_weight_proposal():
    def bad_prop(rng):
        m = rng.uniform(THETA["m_min"], THETA["m_max"], N)
        chi = rng.uniform(0.0, 1.0, N)
        log_q = -np.inf * np.ones(N)   # zero-density proposal — weights blow up
        return m, chi, log_q
    cfg = MCConfig(n_seed=N, proposal=bad_prop, **THETA)
    with pytest.raises(ValueError, match=r"non-finite weight"):
        sample_natals(cfg, np.random.default_rng(2026))


def test_validation_weight_array_shape_mismatch():
    # weight_array must match the natal-sample shape — caller
    # responsibility per the docstring.  Default v1 path → shape = (N,).
    wa = np.ones(N - 1)
    cfg = MCConfig(n_seed=N, weight_array=wa, **THETA)
    with pytest.raises(ValueError, match=r"weight_array shape"):
        sample_natals(cfg, np.random.default_rng(2026))


# ---------------------------------------------------------------------------
# INV-5: Kish n_eff helper bit-equality check (sandbox-extracted).
# ---------------------------------------------------------------------------
def test_kish_n_eff_basic():
    assert _kish_n_eff(np.array([])) == 0.0
    assert _kish_n_eff(np.ones(100)) == 100.0
    # Constant-rescale invariance.
    assert _kish_n_eff(np.full(100, 7.0)) == 100.0
    # Skewed: one big outlier dominates → n_eff collapses.
    w = np.ones(100)
    w[0] = 1000.0
    expected = (99 + 1000) ** 2 / (99 + 1000 ** 2)
    assert _kish_n_eff(w) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# INV-6: Custom proposal + weight_array compose multiplicatively.
# ---------------------------------------------------------------------------
def test_proposal_and_weight_array_compose():
    def uniform_proposal(rng):
        m = rng.uniform(THETA["m_min"], THETA["m_max"], N)
        chi = rng.uniform(0.0, 1.0, N)
        # log q = -log(m_range) (chi range = 1).
        log_q = -np.log(THETA["m_max"] - THETA["m_min"]) * np.ones(N)
        return m, chi, log_q
    cfg_prop = MCConfig(n_seed=N, proposal=uniform_proposal, **THETA)
    m1, chi1, w1 = sample_natals(cfg_prop, np.random.default_rng(2026))

    # Same proposal + explicit weight_array = 2.0.  Same RNG seed → same
    # m/chi samples (modulo filtering which depends only on samples,
    # which match).  Weights should be exactly 2× the bare-proposal weights.
    ext_w = np.full(N, 2.0)
    cfg_combo = MCConfig(
        n_seed=N, proposal=uniform_proposal, weight_array=ext_w, **THETA,
    )
    m2, chi2, w2 = sample_natals(cfg_combo, np.random.default_rng(2026))
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(chi1, chi2)
    np.testing.assert_allclose(w2, 2.0 * w1, rtol=1e-15, atol=0.0)
