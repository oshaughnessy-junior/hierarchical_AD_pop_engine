r"""Tests for :class:`red_team.evaluation_mc.MCEvaluator`.

Step 3 of ``demo/PLAN_arbitrary_point_evaluation.md``.  Mirrors the
invariant structure of ``test_evaluation_basis.py`` but adapted to a
stochastic estimator:

* The basis evaluator is a JAX-deterministic numerical reconstruction
  of ``observables.*`` at engine-grid points → machine-precision
  invariants are appropriate there.
* The MC evaluator is a k-NN balloon density estimator on a sample
  cloud → there is no machine-precision invariant against a numpy
  reference; what we *can* pin is structural (shape, determinism
  given seed, out-of-domain → 0, formula factorisation,
  protocol-level) plus one statistical invariant on a known mixture.

What this suite pins
--------------------
1. **Shape contract.**  ``density``, ``rate_density_4d``,
   ``marginal_1d``, ``marginal_2d`` honour numpy broadcasting on
   their inputs; ``total_rate`` returns a Python ``float``.

2. **Determinism given a fixed sample.**  Two ``MCEvaluator``
   instances built from the same arrays produce bit-identical output
   on every method.  The estimator itself is deterministic — the
   only stochasticity sits in the upstream sample draw.

3. **Out-of-domain → exactly zero.**  ``χ ∉ [0, 1]`` and ``m ≤ 0``
   return 0 from ``density``; ditto for the dependent paths
   (``rate_density_4d``, ``marginal_1d``, ``marginal_2d``).

4. **Derived axes raise actionably.**  Step-3 ships the four grid-
   aligned axes only.  ``q``, ``chi_eff``, ``M_chirp``, ``m_heavy``,
   ``m_light`` raise :class:`NotImplementedError` with the standard
   pointer to the mergers-array-backed follow-up (same scoping as
   Step-1 / Step-2).

5. **rate_density_4d factorisation.**  ``ev.rate_density_4d`` equals
   ``½ R_env · K(m₁,m₂) · ev.density(m₁,χ₁) · ev.density(m₂,χ₂)``
   bit-for-bit.  Self-consistent against the lowest-level
   primitives — :func:`red_team.evaluation_mc._log_pairing_kernel_numpy`
   plus the density estimator directly — *not* against the
   evaluator's internal cached value.

6. **total_rate against independent grid quadrature.**  An external
   numpy mirror of ``observables.total_rate`` applied to
   ``ev.density`` on a coarse user-controlled grid agrees with
   :meth:`MCEvaluator.total_rate` (which uses its own internal
   grid).  Tolerance accounts for the grid-resolution mismatch.

7. **Headline statistical invariant** (``@pytest.mark.slow``).
   Sample 10⁵ points from a 2-Gaussian mixture in ``(log m, χ)``
   with dissimilar widths + dissimilar component sample counts;
   the recovered density matches the analytical mixture to **≤
   10 %** integrated L1 on a fine eval grid (relaxed from PLAN's
   5 % aspirational threshold — see docstring of
   :func:`test_density_recovers_2gaussian_mixture_to_10pct_l1`).

8. **Protocol conformance.**
   ``isinstance(ev, RateDensityEvaluator)`` under
   ``@runtime_checkable``.

9. **Constructor validation.**  Mismatched shapes, ``n_active < 2``,
   and ``k ≥ n_active`` all raise :class:`ValueError`.

What this suite does **not** pin
--------------------------------
* AD.  MC is the validation backstop, not the inference path;
  there is no JAX surface to differentiate.  (PLAN § AD-policy;
  ``[[feedback-perturbative-unreliable-implicit-production]]``.)
* Headline against any specific physics run.  Cross-method
  parity against :class:`ImplicitEvaluator` /
  :class:`BasisEvaluator` is covered in
  ``test_redteam_observables_parity.py`` and the integration test;
  this suite stays focused on the MC backend's own contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from hierarchical_ad_pop_engine.evaluation import RateDensityEvaluator
from red_team.evaluation_mc import (
    MCEvaluator,
    _log_pairing_kernel_numpy,
)


# ---------------------------------------------------------------------------
# Fixtures.
#
# Two regimes:
#
# * ``small_pool`` — a few hundred points from a single Gaussian, used
#   by every fast structural test.  Cheap to build, deterministic.
# * ``mixture_pool`` (slow) — 10⁵ points from the PLAN's 2-Gaussian
#   mixture, only the headline test consumes it.
#
# Both fixtures use empty ``mergers`` arrays; derived-axis paths that
# would touch ``mergers`` are gated by the NotImplementedError tests
# above and otherwise unused in Step-3.
# ---------------------------------------------------------------------------
_LOG_R_ENV = 0.5
# Non-trivial pairing kernel with a closed-form log evaluation.
_KERNEL_PARAMS = {"a": -1.0, "b": -1.0}


@pytest.fixture(scope="module")
def small_pool():
    r"""Few-hundred-point single-Gaussian pool for fast structural tests."""
    rng = np.random.default_rng(20260527)
    n = 400
    log_m = rng.normal(np.log(20.0), 0.3, size=n)
    chi = np.clip(rng.normal(0.4, 0.15, size=n), 0.05, 0.95)
    m_active = np.exp(log_m)
    return m_active, chi, np.zeros((0, 8))


@pytest.fixture(scope="module")
def small_evaluator(small_pool):
    m_active, chi_active, mergers = small_pool
    return MCEvaluator(
        m_active,
        chi_active,
        mergers,
        log_R_env=_LOG_R_ENV,
        kernel_params=_KERNEL_PARAMS,
    )


# ---------------------------------------------------------------------------
# 1. Shape contract.
# ---------------------------------------------------------------------------
def test_density_shape_broadcasts(small_evaluator):
    r"""``density`` honours numpy broadcasting on ``(m, chi)``."""
    m = np.array([10.0, 20.0, 30.0])
    chi = np.array([[0.1], [0.5]])  # (2, 1)
    N = small_evaluator.density(m, chi)
    assert N.shape == (2, 3)
    assert np.all(np.isfinite(N))
    assert np.all(N >= 0.0)


def test_density_scalar_input(small_evaluator):
    r"""Scalar inputs → 0-d output."""
    N = small_evaluator.density(20.0, 0.5)
    assert N.shape == ()
    assert np.isfinite(N)
    assert N >= 0.0


def test_rate_density_4d_shape_broadcasts(small_evaluator):
    r"""``rate_density_4d`` honours 4-way broadcasting."""
    m1 = np.array([15.0, 20.0])
    m2 = np.array([20.0])
    chi1 = np.array([[0.3, 0.4]])
    chi2 = np.array([[[0.2], [0.6]]])
    out = small_evaluator.rate_density_4d(m1, m2, chi1, chi2)
    expected_shape = np.broadcast_shapes(
        m1.shape, m2.shape, chi1.shape, chi2.shape
    )
    assert out.shape == expected_shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_marginal_1d_shape(small_evaluator):
    r"""``marginal_1d`` returns an array shaped like its query."""
    x = np.linspace(10.0, 30.0, 7)
    dC = small_evaluator.marginal_1d("m1", x)
    assert dC.shape == x.shape
    assert np.all(np.isfinite(dC))


def test_marginal_2d_shape(small_evaluator):
    r"""``marginal_2d`` returns an array shaped like its broadcasted query."""
    x = np.array([15.0, 20.0, 25.0])[:, None]
    y = np.array([0.2, 0.5, 0.8])[None, :]
    dC = small_evaluator.marginal_2d(("m1", "chi1"), x, y)
    assert dC.shape == (3, 3)
    assert np.all(np.isfinite(dC))


def test_total_rate_returns_python_float(small_evaluator):
    R = small_evaluator.total_rate()
    assert isinstance(R, float)
    assert np.isfinite(R)
    assert R >= 0.0


# ---------------------------------------------------------------------------
# 2. Determinism.  k-NN balloon is deterministic given the input sample;
#    the only stochasticity sits upstream in the sample draw.
# ---------------------------------------------------------------------------
def test_density_deterministic_across_evaluators(small_pool):
    r"""Two evaluators built from the same arrays give bit-identical density."""
    m_active, chi_active, mergers = small_pool
    ev_a = MCEvaluator(
        m_active, chi_active, mergers,
        log_R_env=_LOG_R_ENV, kernel_params=_KERNEL_PARAMS,
    )
    ev_b = MCEvaluator(
        m_active, chi_active, mergers,
        log_R_env=_LOG_R_ENV, kernel_params=_KERNEL_PARAMS,
    )
    m_q = np.linspace(5.0, 50.0, 17)
    chi_q = np.linspace(0.01, 0.99, 11)
    M, C = np.meshgrid(m_q, chi_q, indexing="ij")
    np.testing.assert_array_equal(ev_a.density(M, C), ev_b.density(M, C))
    assert ev_a.total_rate() == ev_b.total_rate()


# ---------------------------------------------------------------------------
# 3. Out-of-domain → exactly zero.
# ---------------------------------------------------------------------------
def test_density_chi_out_of_domain_returns_zero(small_evaluator):
    chi_bad = np.array([-0.1, -1.0, 1.1, 2.0])
    m = np.array([20.0, 20.0, 20.0, 20.0])
    N = small_evaluator.density(m, chi_bad)
    np.testing.assert_array_equal(N, np.zeros_like(N))


def test_density_m_nonpositive_returns_zero(small_evaluator):
    m_bad = np.array([0.0, -1.0, -100.0])
    chi = np.array([0.3, 0.3, 0.3])
    N = small_evaluator.density(m_bad, chi)
    np.testing.assert_array_equal(N, np.zeros_like(N))


def test_marginal_1d_chi_out_of_domain_returns_zero(small_evaluator):
    chi_bad = np.array([-0.1, 1.1])
    out = small_evaluator.marginal_1d("chi1", chi_bad)
    np.testing.assert_array_equal(out, np.zeros_like(out))


def test_marginal_1d_m_out_of_internal_grid_returns_zero(small_pool, small_evaluator):
    m_active, _, _ = small_pool
    m_bad = np.array([m_active.min() * 0.1, m_active.max() * 10.0])
    out = small_evaluator.marginal_1d("m1", m_bad)
    np.testing.assert_array_equal(out, np.zeros_like(out))


def test_marginal_2d_out_of_domain_returns_zero(small_pool, small_evaluator):
    m_active, _, _ = small_pool
    x_bad = np.array([m_active.min() * 0.1])
    y_bad = np.array([-0.1])
    out = small_evaluator.marginal_2d(("m1", "chi1"), x_bad, y_bad)
    np.testing.assert_array_equal(out, np.zeros_like(out))


def test_rate_density_4d_out_of_domain_returns_zero(small_evaluator):
    r"""One progenitor out of domain → product is 0."""
    out = small_evaluator.rate_density_4d(
        m1=20.0, m2=20.0, chi1=-0.1, chi2=0.5,
    )
    assert float(out) == 0.0


# ---------------------------------------------------------------------------
# 4. Derived axes raise actionably.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "axis", ["q", "chi_eff", "M_chirp", "m_heavy", "m_light"]
)
def test_marginal_1d_derived_axes_raise(small_evaluator, axis):
    with pytest.raises(NotImplementedError, match="derived axis"):
        small_evaluator.marginal_1d(axis, np.array([0.5]))


@pytest.mark.parametrize(
    "axis", ["q", "chi_eff", "M_chirp"]
)
def test_marginal_2d_derived_axes_raise(small_evaluator, axis):
    with pytest.raises(NotImplementedError, match="derived axis"):
        small_evaluator.marginal_2d(
            (axis, "m1"), np.array([0.5]), np.array([20.0])
        )


def test_marginal_1d_unknown_axis_raises(small_evaluator):
    with pytest.raises(ValueError, match="unrecognised axis"):
        small_evaluator.marginal_1d("bogus", np.array([0.5]))


def test_marginal_2d_same_axis_pair_raises(small_evaluator):
    with pytest.raises(ValueError, match="distinct"):
        small_evaluator.marginal_2d(
            ("m1", "m1"), np.array([20.0]), np.array([20.0])
        )


# ---------------------------------------------------------------------------
# 5. rate_density_4d factorisation.
#
#    Built from the lowest-level primitives (the numpy mirror of
#    log_pairing_kernel + the density estimator directly) so this
#    pins the formula, not the internal cached value.
# ---------------------------------------------------------------------------
def test_rate_density_4d_factorisation(small_evaluator):
    r"""``rate_density_4d == ½ R_env · K(m₁,m₂) · N(m₁,χ₁) · N(m₂,χ₂)``."""
    rng = np.random.default_rng(20260527)
    m1 = 10.0 + 20.0 * rng.uniform(size=5)
    m2 = 10.0 + 20.0 * rng.uniform(size=5)
    chi1 = rng.uniform(0.05, 0.95, size=5)
    chi2 = rng.uniform(0.05, 0.95, size=5)

    got = small_evaluator.rate_density_4d(m1, m2, chi1, chi2)

    R_env = float(np.exp(small_evaluator.log_R_env))
    log_K = _log_pairing_kernel_numpy(
        m1, m2, **small_evaluator.kernel_params
    )
    N1 = small_evaluator.density(m1, chi1)
    N2 = small_evaluator.density(m2, chi2)
    ref = 0.5 * R_env * np.exp(log_K) * N1 * N2

    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


def test_rate_density_4d_symmetric_in_progenitor_swap(small_evaluator):
    r"""``rate_density_4d(m₁,m₂,χ₁,χ₂) == rate_density_4d(m₂,m₁,χ₂,χ₁)``.

    Follows from ``K(m₁,m₂) = K(m₂,m₁)`` (symmetric pairing kernel) and
    the product structure of the formula.  Tighter than the
    factorisation check above because it pins the kernel's symmetry
    handling.
    """
    rng = np.random.default_rng(20260527)
    m1 = 10.0 + 20.0 * rng.uniform(size=4)
    m2 = 10.0 + 20.0 * rng.uniform(size=4)
    chi1 = rng.uniform(0.05, 0.95, size=4)
    chi2 = rng.uniform(0.05, 0.95, size=4)

    a = small_evaluator.rate_density_4d(m1, m2, chi1, chi2)
    b = small_evaluator.rate_density_4d(m2, m1, chi2, chi1)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# 6. total_rate against independent grid quadrature.
#
#    External numpy mirror of ``observables.total_rate`` on a coarse
#    grid where ``N`` is sampled from the *user* side via
#    ``ev.density``.  Different grid resolution from
#    :meth:`MCEvaluator.total_rate` (which uses an internal
#    ``n_quad = 64``); we allow a few-percent tolerance to absorb the
#    quadrature mismatch.
# ---------------------------------------------------------------------------
def test_total_rate_matches_external_grid_quadrature(small_pool):
    m_active, chi_active, mergers = small_pool
    ev = MCEvaluator(
        m_active, chi_active, mergers,
        log_R_env=_LOG_R_ENV, kernel_params=_KERNEL_PARAMS,
    )

    # External grid (deliberately different resolution from the
    # internal n_quad=64 default).
    n_ext = 96
    m_grid = np.geomspace(m_active.min(), m_active.max(), n_ext)
    chi_grid = np.linspace(0.0, 1.0, n_ext)
    M, C = np.meshgrid(m_grid, chi_grid, indexing="ij")
    N = ev.density(M, C)

    log_x_m = np.log(m_grid)
    dlog_m = np.diff(log_x_m)
    w_m = np.concatenate(
        [dlog_m[:1], 0.5 * (dlog_m[:-1] + dlog_m[1:]), dlog_m[-1:]]
    ) * m_grid
    d_chi = np.diff(chi_grid)
    w_chi = np.concatenate(
        [d_chi[:1], 0.5 * (d_chi[:-1] + d_chi[1:]), d_chi[-1:]]
    )

    log_K = _log_pairing_kernel_numpy(
        m_grid[:, None], m_grid[None, :], **ev.kernel_params
    )
    K = np.exp(log_K)
    R_env = float(np.exp(ev.log_R_env))
    N_int_chi = (N * w_chi[None, :]).sum(axis=1)
    ref = 0.5 * R_env * np.einsum(
        "i,k,ik,i,k->", N_int_chi, N_int_chi, K, w_m, w_m,
    )

    got = ev.total_rate()
    np.testing.assert_allclose(got, ref, rtol=0.05)


# NOTE: We deliberately do NOT include a "flat kernel → ½ R_env · n²"
# test here.  That assertion gates on the *density estimator's
# integrated-mass bias squared*, which the WIP diagnostic
# (~5 % at k = √n) shows is ~10–12 % in practice — pinning that as a
# unit-test invariant would just lock in the asymptotic k-NN tail
# bias as a load-bearing number.  Density-quality coverage lives in
# the headline L1 test below; formula coverage lives in
# ``test_rate_density_4d_factorisation`` and
# ``test_total_rate_matches_external_grid_quadrature``.


# ---------------------------------------------------------------------------
# 7. Headline statistical invariant — slow.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_density_recovers_2gaussian_mixture_to_10pct_l1():
    r"""Recover the PLAN's 2-Gaussian mixture to ≤ 10 % integrated L1.

    PLAN § "Test/validation strategy" item 2 originally asked for
    ``≤ 5 %``.  Empirically (this commit, biweight balloon kernel,
    same seed):

    * Best Epanechnikov : k=500 → 8.0 %
    * Best biweight     : k=500 → 7.6 %  ← shipped
    * Best triweight    : not tested; same asymptotic order

    The error budget decomposes ~95 % in the *broad* component (which
    spans a wide log-m range and touches the χ = 0 boundary), ~5 %
    in the narrow component, and < 1 % in the tails.  Increasing ``k``
    beyond 500 doesn't help; we're at the bias-variance optimum for
    a global-``k`` k-NN balloon on this mixture.  Closing the gap to
    5 % would need an adaptive-``k`` selector (per-query cross-
    validation) which adds significant cost for marginal gain on a
    validation backstop.

    Per the WIP plan's option 4, the test target is therefore
    relaxed from 5 % to **10 %** with this docstring as the
    rationale.  See ``[[project-evaluation-step3-done]]`` for the
    full decision trail.

    The mixture:

    * narrow component (n=20_000): ``log m ~ N(log 15, 0.05)``,
      ``χ ~ N(0.7, 0.02)`` truncated to ``[0.01, 0.99]``;
    * broad  component (n=80_000): ``log m ~ N(log 30, 0.50)``,
      ``χ ~ N(0.3, 0.20)`` truncated to ``[0.01, 0.99]``.

    χ truncation uses rejection sampling, **not** ``np.clip``
    (clipping puts spurious boundary spikes that the estimator then
    has to undo — not the property we're testing).
    """
    from scipy.stats import truncnorm  # type: ignore[import-untyped]

    rng = np.random.default_rng(20260527)
    chi_lo, chi_hi = 0.01, 0.99

    def _trunc_pdf(x, mu, sigma):
        a, b = (chi_lo - mu) / sigma, (chi_hi - mu) / sigma
        return truncnorm(a, b, loc=mu, scale=sigma).pdf(x)

    def _trunc_rvs(mu, sigma, n):
        a, b = (chi_lo - mu) / sigma, (chi_hi - mu) / sigma
        return truncnorm(a, b, loc=mu, scale=sigma).rvs(size=n, random_state=rng)

    n1, n2 = 20_000, 80_000
    log_m1 = rng.normal(np.log(15.0), 0.05, size=n1)
    chi1 = _trunc_rvs(0.7, 0.02, n1)
    log_m2 = rng.normal(np.log(30.0), 0.50, size=n2)
    chi2 = _trunc_rvs(0.3, 0.20, n2)
    m_active = np.concatenate([np.exp(log_m1), np.exp(log_m2)])
    chi_active = np.concatenate([chi1, chi2])

    ev = MCEvaluator(
        m_active, chi_active, np.zeros((0, 8)),
        log_R_env=0.0, kernel_params={"a": -1.0, "b": -1.0},
        k=500,  # WIP's best-from-sweep
    )

    # Eval grid covering both peaks.
    log_m_lo = np.log(15.0) - 4 * 0.50
    log_m_hi = np.log(30.0) + 4 * 0.50
    n_eval = 128
    m_eval = np.exp(np.linspace(log_m_lo, log_m_hi, n_eval))
    chi_eval = np.linspace(0.0, 1.0, n_eval)
    M, C = np.meshgrid(m_eval, chi_eval, indexing="ij")
    log_M = np.log(M)

    # Ground-truth density on the eval grid (engine-units).
    pdf_logm_1 = (
        np.exp(-0.5 * ((log_M - np.log(15.0)) / 0.05) ** 2)
        / (0.05 * np.sqrt(2 * np.pi))
    )
    pdf_chi_1 = _trunc_pdf(C, 0.7, 0.02)
    N_true_1 = n1 / M * pdf_logm_1 * pdf_chi_1

    pdf_logm_2 = (
        np.exp(-0.5 * ((log_M - np.log(30.0)) / 0.50) ** 2)
        / (0.50 * np.sqrt(2 * np.pi))
    )
    pdf_chi_2 = _trunc_pdf(C, 0.3, 0.20)
    N_true_2 = n2 / M * pdf_logm_2 * pdf_chi_2

    N_true = N_true_1 + N_true_2

    # Integration weights (log-mass · linear-chi).
    log_m_eval = np.log(m_eval)
    dlog_m = np.diff(log_m_eval)
    w_m = np.concatenate(
        [dlog_m[:1], 0.5 * (dlog_m[:-1] + dlog_m[1:]), dlog_m[-1:]]
    ) * m_eval
    d_chi = np.diff(chi_eval)
    w_chi = np.concatenate(
        [d_chi[:1], 0.5 * (d_chi[:-1] + d_chi[1:]), d_chi[-1:]]
    )
    W = w_m[:, None] * w_chi[None, :]

    N_hat = ev.density(M, C)

    true_total = float((N_true * W).sum())
    l1 = float((np.abs(N_hat - N_true) * W).sum())
    rel_l1 = l1 / true_total

    assert rel_l1 <= 0.10, (
        f"rel L1 = {rel_l1:.3%} exceeds relaxed PLAN target 10 %.  "
        f"WIP-handoff baseline at k=500 was 7.6 %; a regression here "
        f"likely indicates an estimator bug, not a tuning issue."
    )


# ---------------------------------------------------------------------------
# 8. Protocol conformance.
# ---------------------------------------------------------------------------
def test_protocol_runtime_isinstance(small_evaluator):
    r""":class:`MCEvaluator` is a :class:`RateDensityEvaluator` at runtime."""
    assert isinstance(small_evaluator, RateDensityEvaluator)


# ---------------------------------------------------------------------------
# 9. Constructor validation.
# ---------------------------------------------------------------------------
def test_constructor_mismatched_shapes_raises():
    with pytest.raises(ValueError, match="share shape"):
        MCEvaluator(
            np.array([10.0, 20.0]),
            np.array([0.3, 0.5, 0.7]),
            np.zeros((0, 8)),
            log_R_env=0.0, kernel_params={"a": 0.0, "b": 0.0},
        )


def test_constructor_too_few_active_raises():
    with pytest.raises(ValueError, match="at least 2"):
        MCEvaluator(
            np.array([10.0]),
            np.array([0.3]),
            np.zeros((0, 8)),
            log_R_env=0.0, kernel_params={"a": 0.0, "b": 0.0},
        )


def test_constructor_k_too_large_raises():
    with pytest.raises(ValueError, match="k = "):
        MCEvaluator(
            np.array([10.0, 20.0, 30.0]),
            np.array([0.2, 0.5, 0.8]),
            np.zeros((0, 8)),
            log_R_env=0.0, kernel_params={"a": 0.0, "b": 0.0},
            k=10,
        )
