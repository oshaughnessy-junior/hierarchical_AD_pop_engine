r"""Mock-catalogue scatter tests — engine-free (lightweight sandbox).

Pins the measurement-blur half of :mod:`demo.mock_catalog`
(``scatter_to_posteriors``); the engine-dependent ``make_mock_posteriors``
(grid inverse-CDF truth draw) is exercised by the fit runner where the engine
is available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

_DEMO = (Path(__file__).resolve().parent.parent / "demo").resolve()
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))
from mock_catalog import scatter_to_posteriors  # noqa: E402


def _true_events(n=64):
    rng = np.random.default_rng(0)
    m1 = rng.uniform(10.0, 40.0, n)
    m2 = np.minimum(m1, rng.uniform(8.0, 35.0, n))
    chi1 = rng.uniform(0.05, 0.8, n)
    chi2 = rng.uniform(0.05, 0.8, n)
    return jnp.asarray(np.stack([m1, m2, chi1, chi2], axis=-1))


def test_shape_and_support():
    ev = _true_events(64)
    out = scatter_to_posteriors(ev, 8, jax.random.PRNGKey(0),
                                m_lo=4.0, m_hi=200.0)
    assert out.shape == (64, 8, 4)
    o = np.asarray(out)
    assert (o[..., 0] >= 4.0).all() and (o[..., 0] <= 200.0).all()
    assert (o[..., 2] >= 0.0).all() and (o[..., 2] <= 1.0).all()
    assert (o[..., 3] >= 0.0).all() and (o[..., 3] <= 1.0).all()
    print(f"\n[MC1] scatter shape {out.shape}; all samples in support box")


def test_mass_ordering_preserved_per_sample():
    ev = _true_events(64)
    out = np.asarray(scatter_to_posteriors(ev, 16, jax.random.PRNGKey(1)))
    assert (out[..., 0] >= out[..., 1] - 1e-6).all()
    print("\n[MC2] m1 ≥ m2 holds for every (event, sample)")


def test_samples_centre_on_truth():
    r"""Posterior-sample mean ≈ true value (unbiased blur) at large S.

    Uses WELL-SEPARATED masses (gap ≫ blur σ) so the per-sample ``m₁≥m₂``
    re-sort never flips and hence introduces no order-statistic bias; near
    equal-mass events would bias the marginal means *by design* (the sort
    defines m₁ as the larger), so this test deliberately avoids them.
    """
    rng = np.random.default_rng(7)
    n = 40
    m1 = rng.uniform(34.0, 38.0, n)          # ~36 ± blur 3.6
    m2 = rng.uniform(9.0, 11.0, n)           # ~10 ± blur 1.0; gap ≈ 26 ≫ blur
    chi1 = rng.uniform(0.3, 0.6, n)          # well inside (0,1): no clip bias
    chi2 = rng.uniform(0.3, 0.6, n)
    ev = jnp.asarray(np.stack([m1, m2, chi1, chi2], axis=-1))
    out = np.asarray(scatter_to_posteriors(
        ev, 6000, jax.random.PRNGKey(2), sigma_log_m=0.1, sigma_chi=0.1,
        m_lo=1.0, m_hi=500.0))
    # mass blur is multiplicative log-normal → unbiased in log space.
    log_m1_mean = np.mean(np.log(out[..., 0]), axis=1)
    np.testing.assert_allclose(log_m1_mean, np.log(m1), atol=0.02)
    chi1_mean = np.mean(out[..., 2], axis=1)
    np.testing.assert_allclose(chi1_mean, chi1, atol=0.02)
    print("\n[MC3] sample means recover truth (log-mass & spin), well-separated")


def test_single_sample_is_clipped_truth():
    ev = _true_events(10)
    out = scatter_to_posteriors(ev, 1, jax.random.PRNGKey(3),
                                sigma_log_m=0.0, sigma_chi=0.0)
    # zero scatter, S=1 → exactly the (ordered, clipped) truth.
    np.testing.assert_allclose(np.asarray(out[:, 0, :]), np.asarray(ev), rtol=1e-6)
    print("\n[MC4] σ=0, S=1 → the delta-catalogue (M4 reduction)")
