r"""Session M3a sandbox-tier tests — weighted ``MCEvaluator``.

Pins the six invariants from ``demo/PLAN_m3_stratified_natal.md``
§ "Sandbox-verifiable invariants":

1. ``weights=None`` and ``weights=np.ones(n)`` produce *bit-for-bit*
   identical ``density`` output (the unweighted path defends Step-3
   / M2 behaviour as a hard regression).
2. Constant non-unit weights scale the engine-units density by the
   weight (``∫ N dm dχ = W_total = w · n``).
3. The Kish (1965) effective sample size ``n_eff = (Σw)² / (Σw²)``
   drives the default-``k`` heuristic (one big outlier ⇒ ``n_eff``
   collapses to ~1, ``k`` floors at 4).
4. Importance recovery: drawing from a uniform proposal but
   weighting by ``target_pdf / proposal_pdf`` recovers the target
   density's *shape* in ``χ``, while the same samples unweighted
   recover the uniform proposal's shape (the L1-vs-target ordering
   is the gate).
5. ``W_total`` matches ``Σ w_i`` (= ``n_active`` for uniform
   weights).
6. ``__init__`` validates ``weights`` against shape, finiteness,
   and positivity (importance weights are positive by construction
   per ``[[demo/PLAN_m3_stratified_natal.md]]`` §Surface).

All tests are numpy + scipy only — no JAX, no engine, runnable
inside the Cowork VM.  Mirrors the structure of
``tests/test_evaluation_mc.py`` (Step 3,
``[[project-evaluation-step3-done]]``).

See also ``[[project-evaluation-mc-weighted-m3a]]`` for the session
breadcrumb.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

scipy_stats = pytest.importorskip(
    "scipy.stats", reason="scipy required for the importance-recovery test",
)
truncnorm = scipy_stats.truncnorm


# ---------------------------------------------------------------------------
# Import MCEvaluator directly from the module file to avoid the
# ``evaluation/__init__.py`` JAX dependency chain — keeps these tests
# numpy-only so they run anywhere scipy is installed.  This mirrors the
# sandbox-validation pattern used in the M3a planning session.
# ---------------------------------------------------------------------------
_MC_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "hierarchical_ad_pop_engine" / "evaluation" / "mc.py"
)
_spec = importlib.util.spec_from_file_location("hape_evaluation_mc_m3a", _MC_PATH)
_mc_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mc_mod)
MCEvaluator = _mc_mod.MCEvaluator


# ---------------------------------------------------------------------------
# Shared fixture: uniform-chi MC samples on a moderate mass grid.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def uniform_chi_samples():
    rng = np.random.default_rng(20260528)
    n = 500
    m = np.exp(rng.normal(np.log(20.0), 0.5, n))
    chi = rng.uniform(0.0, 1.0, n)
    return dict(m=m, chi=chi, n=n, mergers=np.empty((0, 8)),
                kernel_params=dict(a=0.0, b=1.0), log_R_env=0.0)


@pytest.fixture(scope="module")
def query_grid():
    m_q = np.geomspace(8.0, 50.0, 25)
    chi_q = np.linspace(0.1, 0.9, 25)
    M, C = np.meshgrid(m_q, chi_q, indexing="ij")
    return dict(M=M, C=C)


# ---------------------------------------------------------------------------
# INV-1: bit-for-bit invariance under uniform weights.
# ---------------------------------------------------------------------------
def test_bit_for_bit_none_vs_ones(uniform_chi_samples, query_grid):
    r"""``weights=None`` and ``weights=np.ones(n_active)`` produce
    *literally equal* ``density`` outputs.

    Step-3 / M2 regression defence: the unweighted code path
    (``sample_weights=None`` inside ``_knn_balloon_2d``) MUST
    short-circuit to the pre-M3a algebra exactly.  See
    ``[[demo/PLAN_m3_stratified_natal.md]]`` §"Sandbox-verifiable
    invariants" item 1.
    """
    s = uniform_chi_samples
    ev_none = MCEvaluator(
        s["m"], s["chi"], s["mergers"],
        log_R_env=s["log_R_env"], kernel_params=s["kernel_params"], k=40,
    )
    ev_ones = MCEvaluator(
        s["m"], s["chi"], s["mergers"],
        log_R_env=s["log_R_env"], kernel_params=s["kernel_params"], k=40,
        weights=np.ones(s["n"]),
    )
    d_none = ev_none.density(query_grid["M"], query_grid["C"])
    d_ones = ev_ones.density(query_grid["M"], query_grid["C"])
    np.testing.assert_array_equal(
        d_none, d_ones,
        err_msg="weights=None and weights=ones(n) MUST be bit-for-bit "
                "identical (Step-3 / M2 regression defence)",
    )


# ---------------------------------------------------------------------------
# INV-2: constant-weight rescaling of the engine-units density.
# ---------------------------------------------------------------------------
def test_constant_weight_scales_density(uniform_chi_samples, query_grid):
    r"""``weights = c · ones`` scales ``density`` by exactly ``c``.

    The engine-units convention ``∫∫ N dm dχ = W_total`` means a
    uniform constant rescale of the weights produces a uniform
    rescale of the density.  This pins the W_total normalisation
    inside ``density()`` against accidental drift to ``n_active``.
    """
    s = uniform_chi_samples
    ev_none = MCEvaluator(
        s["m"], s["chi"], s["mergers"],
        log_R_env=s["log_R_env"], kernel_params=s["kernel_params"], k=40,
    )
    ev_7 = MCEvaluator(
        s["m"], s["chi"], s["mergers"],
        log_R_env=s["log_R_env"], kernel_params=s["kernel_params"], k=40,
        weights=np.full(s["n"], 7.0),
    )
    d_none = ev_none.density(query_grid["M"], query_grid["C"])
    d_7 = ev_7.density(query_grid["M"], query_grid["C"])
    ratio = d_7 / np.maximum(d_none, 1e-30)
    # Machine-precision tight — algebra is exact for uniform weights.
    np.testing.assert_allclose(ratio, 7.0, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# INV-3: Kish n_eff drives the default-k heuristic.
# ---------------------------------------------------------------------------
def test_kish_n_eff_drives_default_k(uniform_chi_samples):
    r"""One huge outlier weight ⇒ ``n_eff ≈ 1`` ⇒ default ``k = 4`` (floor).

    Sanity that the Step-3 default-``k`` rule
    ``k = max(4, round(√n_eff))`` correctly uses the *effective*
    sample size (Kish 1965 design-effect) rather than ``n_active``.
    Without this, a heavily skewed weighted sample would inherit
    Step-3's ``k = √n_active`` and oversmooth across the high-
    weight outlier.
    """
    s = uniform_chi_samples
    w_skew = np.ones(s["n"])
    w_skew[0] = 1000.0
    ev = MCEvaluator(
        s["m"], s["chi"], s["mergers"],
        log_R_env=s["log_R_env"], kernel_params=s["kernel_params"],
        weights=w_skew,
    )
    # n_eff = (n - 1 + 1000)² / (n - 1 + 1000²)
    expected_n_eff = (s["n"] - 1 + 1000.0) ** 2 / (s["n"] - 1 + 1000.0 ** 2)
    assert ev.n_eff == pytest.approx(expected_n_eff, rel=1e-12)
    # √n_eff ≈ 1.5 < 4 so the floor binds.
    assert ev.k == 4, (
        f"default-k should be the floor at 4 when n_eff = "
        f"{ev.n_eff:.3f}; got k = {ev.k}"
    )


# ---------------------------------------------------------------------------
# INV-4: importance-recovery sanity.
# ---------------------------------------------------------------------------
def test_importance_recovery_chi(uniform_chi_samples):
    r"""Drawing from a uniform-χ proposal but weighting by the target
    pdf recovers the target's *shape* in χ.

    Setup: draw ``χ ~ Uniform[0, 1]`` (proposal ``q(χ) = 1``); target
    ``p(χ) = TruncNorm(μ=0.5, σ=0.05)`` on ``[0, 1]``.  Importance
    weights ``w = p(χ) / q(χ) = p(χ)``.

    Gate (magnitude, no frozen reference per
    ``[[feedback-no-frozen-test-references]]``):

    * weighted L1 vs target shape < 0.5 (target is recovered);
    * weighted L1 < unweighted L1 (weighting strictly improves).
    """
    rng = np.random.default_rng(7)
    n = 5_000
    m = np.exp(rng.normal(np.log(20.0), 0.3, n))
    chi_unif = rng.uniform(0.0, 1.0, n)

    mu_t, sg_t = 0.5, 0.05
    a_t = (0.0 - mu_t) / sg_t
    b_t = (1.0 - mu_t) / sg_t
    w_imp = truncnorm.pdf(chi_unif, a_t, b_t, loc=mu_t, scale=sg_t)
    w_imp = np.maximum(w_imp, 1e-12)

    kp = dict(a=0.0, b=1.0)
    ev_unif = MCEvaluator(
        m, chi_unif, np.empty((0, 8)),
        log_R_env=0.0, kernel_params=kp, k=70,
    )
    ev_imp = MCEvaluator(
        m, chi_unif, np.empty((0, 8)),
        log_R_env=0.0, kernel_params=kp, k=70, weights=w_imp,
    )

    # Marginalise over m to get dN/dχ; compare *shapes* (renormalise to
    # unit integral so the comparison is amplitude-free, the way Phase
    # C compares engine vs MC shapes).
    m_grid = np.geomspace(m.min(), m.max(), 64)
    chi_grid = np.linspace(0.05, 0.95, 64)
    Mg, Cg = np.meshgrid(m_grid, chi_grid, indexing="ij")
    dlogm = np.diff(np.log(m_grid))
    w_m_log = np.concatenate(
        [dlogm[:1] * 0.5, 0.5 * (dlogm[:-1] + dlogm[1:]), dlogm[-1:] * 0.5]
    ) * m_grid

    dN_unif = np.sum(ev_unif.density(Mg, Cg) * w_m_log[:, None], axis=0)
    dN_imp = np.sum(ev_imp.density(Mg, Cg) * w_m_log[:, None], axis=0)
    target = truncnorm.pdf(chi_grid, a_t, b_t, loc=mu_t, scale=sg_t)

    def _norm(p):
        Z = float(np.trapezoid(p, chi_grid))
        return p / Z if Z > 0 else np.zeros_like(p)

    target_norm = _norm(target)
    dN_unif_norm = _norm(dN_unif)
    dN_imp_norm = _norm(dN_imp)
    L1_unif = float(np.trapezoid(np.abs(dN_unif_norm - target_norm), chi_grid))
    L1_imp = float(np.trapezoid(np.abs(dN_imp_norm - target_norm), chi_grid))

    assert np.isfinite(L1_unif) and np.isfinite(L1_imp)
    assert L1_imp < L1_unif, (
        f"weighted should strictly beat unweighted on target shape "
        f"recovery: weighted L1 = {L1_imp:.3f} vs unweighted L1 = "
        f"{L1_unif:.3f}.  If this fails the weighted kernel sum is "
        f"not picking up the importance weights."
    )
    assert L1_imp < 0.5, (
        f"weighted L1 vs target should be < 0.5 at N = {n}, k = 70; "
        f"got {L1_imp:.3f}.  Likely the weight broadcast through "
        f"reflections is wrong — verify _build_density_tree augmented-"
        f"weights array."
    )


# ---------------------------------------------------------------------------
# INV-5: W_total bookkeeping.
# ---------------------------------------------------------------------------
def test_W_total_matches_sum_weights(uniform_chi_samples):
    r"""``self.W_total`` equals ``Σ w_i`` (= ``n_active`` for uniform).

    Pins the convention that ``W_total`` is the original-sample
    weight sum, *not* the reflection-augmented sum (reflections are
    virtual and don't add probability mass).
    """
    s = uniform_chi_samples
    ev_none = MCEvaluator(
        s["m"], s["chi"], s["mergers"],
        log_R_env=s["log_R_env"], kernel_params=s["kernel_params"], k=40,
    )
    assert ev_none.W_total == float(s["n"])

    w = np.full(s["n"], 7.0)
    ev_w = MCEvaluator(
        s["m"], s["chi"], s["mergers"],
        log_R_env=s["log_R_env"], kernel_params=s["kernel_params"], k=40,
        weights=w,
    )
    assert ev_w.W_total == float(np.sum(w))

    rng = np.random.default_rng(11)
    w_rand = rng.uniform(0.1, 10.0, s["n"])
    ev_r = MCEvaluator(
        s["m"], s["chi"], s["mergers"],
        log_R_env=s["log_R_env"], kernel_params=s["kernel_params"], k=40,
        weights=w_rand,
    )
    assert ev_r.W_total == pytest.approx(float(np.sum(w_rand)), rel=1e-15)


# ---------------------------------------------------------------------------
# INV-6: validation errors.
# ---------------------------------------------------------------------------
def test_validation_bad_shape(uniform_chi_samples):
    s = uniform_chi_samples
    with pytest.raises(ValueError, match=r"weights must have shape"):
        MCEvaluator(
            s["m"], s["chi"], s["mergers"],
            log_R_env=s["log_R_env"], kernel_params=s["kernel_params"],
            weights=np.ones(s["n"] - 1),
        )


def test_validation_negative_weight(uniform_chi_samples):
    s = uniform_chi_samples
    bad = np.ones(s["n"])
    bad[0] = -1.0
    with pytest.raises(ValueError, match=r"strictly positive"):
        MCEvaluator(
            s["m"], s["chi"], s["mergers"],
            log_R_env=s["log_R_env"], kernel_params=s["kernel_params"],
            weights=bad,
        )


def test_validation_zero_weight(uniform_chi_samples):
    s = uniform_chi_samples
    bad = np.ones(s["n"])
    bad[3] = 0.0
    with pytest.raises(ValueError, match=r"strictly positive"):
        MCEvaluator(
            s["m"], s["chi"], s["mergers"],
            log_R_env=s["log_R_env"], kernel_params=s["kernel_params"],
            weights=bad,
        )


def test_validation_nan_weight(uniform_chi_samples):
    s = uniform_chi_samples
    bad = np.ones(s["n"])
    bad[0] = np.nan
    with pytest.raises(ValueError, match=r"must all be finite"):
        MCEvaluator(
            s["m"], s["chi"], s["mergers"],
            log_R_env=s["log_R_env"], kernel_params=s["kernel_params"],
            weights=bad,
        )


def test_validation_inf_weight(uniform_chi_samples):
    s = uniform_chi_samples
    bad = np.ones(s["n"])
    bad[0] = np.inf
    with pytest.raises(ValueError, match=r"must all be finite"):
        MCEvaluator(
            s["m"], s["chi"], s["mergers"],
            log_R_env=s["log_R_env"], kernel_params=s["kernel_params"],
            weights=bad,
        )


# ---------------------------------------------------------------------------
# Bonus: reflection-boundary weight broadcast.
# ---------------------------------------------------------------------------
def test_reflection_inherits_weight(uniform_chi_samples):
    r"""Boundary-near queries with uniform weights match unweighted bit-for-bit.

    Per ``[[demo/PLAN_m3_stratified_natal.md]]`` §"Boundary handling":
    the reflection-augmentation broadcast must produce identical
    density for ``weights = ones`` vs ``weights = None``, including
    *near the χ boundary* where the reflected neighbours enter the
    kernel sum.  This is a tighter form of INV-1 specifically
    exercising the reflection path.
    """
    s = uniform_chi_samples
    ev_none = MCEvaluator(
        s["m"], s["chi"], s["mergers"],
        log_R_env=s["log_R_env"], kernel_params=s["kernel_params"], k=40,
    )
    ev_ones = MCEvaluator(
        s["m"], s["chi"], s["mergers"],
        log_R_env=s["log_R_env"], kernel_params=s["kernel_params"], k=40,
        weights=np.ones(s["n"]),
    )
    # Query at χ near each boundary — reflections enter the k-NN.
    m_b = np.full(20, 20.0)
    chi_b_lo = np.linspace(0.001, 0.05, 20)
    chi_b_hi = np.linspace(0.95, 0.999, 20)
    np.testing.assert_array_equal(
        ev_none.density(m_b, chi_b_lo),
        ev_ones.density(m_b, chi_b_lo),
        err_msg="lower-boundary χ density must be bit-for-bit None vs ones",
    )
    np.testing.assert_array_equal(
        ev_none.density(m_b, chi_b_hi),
        ev_ones.density(m_b, chi_b_hi),
        err_msg="upper-boundary χ density must be bit-for-bit None vs ones",
    )
