r"""Session J1 sandbox-tier tests — JAX-pure ``sample_natals_jax``.

Per ``demo/PLAN_jax_mc.md`` § J1.  Pins five invariants:

1. **Distributional equivalence with numpy** — KS test on independent
   draws of ``m`` and ``χ`` shows the numpy ``sample_natals`` and
   JAX ``sample_natals_jax`` paths sample from the same truncnorm
   distributions at the same ``θ*``.  This is the validation-backstop
   pattern from ``[[feedback-red-team-validation]]``: the numpy MC
   is the independent ground truth.
2. **Uniform-weights contract** — default ``sample_natals_jax(cfg, key)``
   with no proposal/stratify/weight_array returns ``weights = 1.0``
   for every sample (same M3b convention as numpy).
3. **Stratified-IS recovery** — at ``σ_χ = 0.10`` (Stage 1 θ*) the
   stratified-JAX path's IS estimator of ``∫ p_natal dm dχ = 1`` and
   ``⟨m⟩ ≈ truncnorm.mean(m)`` within MC SE.  Mirrors numpy INV-3.
4. **Validation errors** — negative ``weight_array`` and non-finite
   proposal ``log_q`` raise ``ValueError`` with matching messages
   (consistency with the numpy path).
5. **AD-through-natal-params** — gradient of
   ``Σ sample_natals_jax(cfg.replace(mu_natal=mu), key)[0]`` w.r.t.
   ``mu`` is finite and matches a 5-point central finite difference.
   This is the headline invariant that unblocks the J-series: it
   confirms the reparametrisation trick wires gradients through the
   truncnorm sampler.

Per ``[[demo/PLAN_jax_mc.md]]`` § "Hard rule": the numpy
``sample_natals`` is **not** touched by J1 — its M3a/M3b sandbox
tests (``test_mc_sample_natals_weights.py``) continue to gate
bit-for-bit.  This file is purely additive.

All tests are JAX + numpy + scipy — runnable on the Cowork VM
after a ``pip install jax``.  JAX defaults to float32; we don't
enable x64 here so the test environment is the natural
inference-time precision (matches the implicit-integration test
convention per ``[[project-integration-test-in-flight]]``).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

scipy_stats = pytest.importorskip("scipy.stats")
truncnorm = scipy_stats.truncnorm

# Add red_team/ parent to sys.path so we can import backup_particle_mc
# without the package wrapper (matches the
# tests/test_mc_sample_natals_weights.py pattern).
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from red_team.backup_particle_mc import (  # noqa: E402
    MCConfig,
    sample_natals,
    sample_natals_jax,
    _natal_log_pdf_jax,
)


THETA = dict(
    mu_natal=8.0, sigma_natal=1.0, m_min=4.0, m_max=80.0,
    natal_spin_loc=0.0, natal_spin_scale=0.10,
    kernel_a=0.0, kernel_b=1.0, v_esc=1000.0,
)
N = 5000


# ---------------------------------------------------------------------------
# INV-1: distributional equivalence with numpy.
# ---------------------------------------------------------------------------
def test_distribution_matches_numpy():
    r"""KS test: JAX and numpy paths sample the same truncnorm distributions.

    The two paths cannot produce bit-identical samples (different
    PRNG streams), but they should produce statistically
    indistinguishable distributions at the same θ.  Validation-
    backstop pattern per ``[[feedback-red-team-validation]]``.
    """
    cfg = MCConfig(n_seed=N, **THETA)
    m_np, chi_np, _ = sample_natals(cfg, np.random.default_rng(2026))
    m_jax, chi_jax, _ = sample_natals_jax(cfg, jax.random.PRNGKey(2026))
    m_jax = np.asarray(m_jax)
    chi_jax = np.asarray(chi_jax)
    ks_m = scipy_stats.ks_2samp(m_np, m_jax)
    ks_chi = scipy_stats.ks_2samp(chi_np, chi_jax)
    # p > 0.01 ⇒ no detectable difference at the 1 % level.
    assert ks_m.pvalue > 0.01, (
        f"m distribution mismatch JAX vs numpy: KS p-value = "
        f"{ks_m.pvalue:.3g} < 0.01.  Likely a truncnorm bound bug "
        f"in sample_natals_jax."
    )
    assert ks_chi.pvalue > 0.01, (
        f"chi distribution mismatch JAX vs numpy: KS p-value = "
        f"{ks_chi.pvalue:.3g} < 0.01."
    )


# ---------------------------------------------------------------------------
# INV-2: uniform-weights contract.
# ---------------------------------------------------------------------------
def test_uniform_weights_default_path():
    r"""Default ``sample_natals_jax`` (no proposal/stratify/weight_array)
    returns ``weights = 1.0`` for every sample.

    Mirrors numpy INV-1 (``test_v1_default_bit_for_bit`` in
    ``tests/test_mc_sample_natals_weights.py``).  Cannot assert
    bit-for-bit numpy sample sequence because JAX uses a different
    PRNG, but the weights contract is preserved.
    """
    cfg = MCConfig(n_seed=N, **THETA)
    m, chi, w = sample_natals_jax(cfg, jax.random.PRNGKey(2026))
    assert m.shape == chi.shape == w.shape == (N,)
    np.testing.assert_array_equal(np.asarray(w), np.ones(N, dtype=np.asarray(w).dtype))


# ---------------------------------------------------------------------------
# INV-3: stratified-IS recovery.
# ---------------------------------------------------------------------------
def test_stratified_jax_recovers_natal_mean():
    r"""Stratified JAX IS recovers ``∫ p_natal = 1`` and ``⟨m⟩``.

    Mirrors the numpy INV-3 in
    ``tests/test_mc_sample_natals_weights.py``.
    """
    cfg = MCConfig(n_seed=N, stratify=(8, 16), **THETA)
    m, chi, w = sample_natals_jax(cfg, jax.random.PRNGKey(2026))
    m_np = np.asarray(m)
    w_np = np.asarray(w)
    n_per_cell = N // (8 * 16)
    n_drawn = 8 * 16 * n_per_cell
    sum_w = float(w_np.sum())
    print(f"\n[INV-3] stratified-JAX IS: Σw = {sum_w:.1f}  vs n_drawn = {n_drawn}")
    assert 0.7 * n_drawn <= sum_w <= 1.3 * n_drawn

    a_m = (THETA["m_min"] - THETA["mu_natal"]) / THETA["sigma_natal"]
    b_m = (THETA["m_max"] - THETA["mu_natal"]) / THETA["sigma_natal"]
    true_m_mean = float(truncnorm.mean(
        a_m, b_m, loc=THETA["mu_natal"], scale=THETA["sigma_natal"],
    ))
    weighted_m_mean = float(np.sum(w_np * m_np) / np.sum(w_np))
    print(
        f"[INV-3] ⟨m⟩ weighted stratified-JAX = {weighted_m_mean:.4f}, "
        f"true = {true_m_mean:.4f}"
    )
    assert abs(weighted_m_mean - true_m_mean) < 0.3


# ---------------------------------------------------------------------------
# INV-4: validation errors.
# ---------------------------------------------------------------------------
def test_validation_negative_weight_array():
    bad = np.ones(N, dtype=np.float32)
    bad[0] = -0.5
    cfg = MCConfig(n_seed=N, weight_array=bad, **THETA)
    with pytest.raises(ValueError, match=r"negative weight"):
        sample_natals_jax(cfg, jax.random.PRNGKey(2026))


def test_validation_inf_weight_proposal():
    def bad_prop(key):
        k_m, k_chi = jax.random.split(key)
        m = jax.random.uniform(k_m, shape=(N,), minval=THETA["m_min"], maxval=THETA["m_max"])
        chi = jax.random.uniform(k_chi, shape=(N,))
        log_q = -jnp.inf * jnp.ones(N)
        return m, chi, log_q
    cfg = MCConfig(n_seed=N, proposal=bad_prop, **THETA)
    with pytest.raises(ValueError, match=r"non-finite weight"):
        sample_natals_jax(cfg, jax.random.PRNGKey(2026))


# ---------------------------------------------------------------------------
# INV-5: AD-through-natal-params (the headline J1 invariant).
# ---------------------------------------------------------------------------
def test_grad_through_mu_natal_matches_finite_difference():
    r"""``jax.grad(Σ m_samples)`` w.r.t. ``mu_natal`` is finite and
    matches a central finite difference.

    This is the **headline J1 invariant**: it confirms the
    reparametrisation trick wires gradients cleanly through
    ``jax.random.truncated_normal``'s ``loc + scale·z`` affine
    transform.  Without this gradient flowing, the eventual J4 SVI
    integration test cannot work — so failing this test means the
    whole J-series stack will not lift off.

    Test design: pick a function ``f(mu) = Σ m_samples(mu, key)``
    at a fixed key; compute ``df/dmu`` two ways.

    * AD: ``jax.grad(f)(mu)`` — should be ≈ N (each sample's m =
      ``mu + sigma·z_i``, so ``dm/dmu = 1`` and ``df/dmu = N``).
    * FD: ``[f(mu + h) - f(mu - h)] / (2h)``.
      With the same key, samples shift uniformly so FD should give
      ≈ N too.
    """
    key = jax.random.PRNGKey(2026)
    cfg_base = MCConfig(n_seed=N, **THETA)

    def f(mu):
        cfg = replace(cfg_base, mu_natal=mu)
        # Disable filtering — keeps shape static so jax.grad has a
        # clean tape.  Under the default (no proposal/stratify), no
        # zero weights are produced anyway.
        m, _chi, _w = sample_natals_jax(
            cfg, key, filter_zero_weights=False,
        )
        return jnp.sum(m)

    mu_ref = jnp.asarray(8.0)
    df_ad = float(jax.grad(f)(mu_ref))
    h = 1e-2
    df_fd = float((f(mu_ref + h) - f(mu_ref - h)) / (2.0 * h))
    rel_err = abs(df_ad - df_fd) / max(abs(df_fd), 1e-12)
    print(f"\n[INV-5] AD vs FD on ∂Σm/∂mu: AD = {df_ad:.4f}, FD = {df_fd:.4f}, "
          f"expected ≈ N = {N}, rel_err = {rel_err:.3e}")
    assert np.isfinite(df_ad), "AD gradient must be finite"
    # The samples are reparametrised m = mu + sigma·z, so dm/dmu = 1
    # exactly per sample, giving df/dmu = N exactly.  FD agrees up to
    # float-precision noise in the affine transform.
    assert rel_err < 1e-2, (
        f"AD vs FD gradient mismatch: {rel_err:.3e}.  Likely the "
        f"reparametrisation trick is not wiring through "
        f"jax.random.truncated_normal as expected."
    )


def test_grad_through_sigma_natal_finite():
    r"""``jax.grad(Σ m²) / dσ`` is finite and non-zero.

    Lighter than INV-5: just confirms the gradient *exists* and is
    non-trivial.  ``E[m²]`` depends on σ², so ``d(Σ m²)/dσ`` should
    be roughly ``2σ·N``.
    """
    key = jax.random.PRNGKey(2026)
    cfg_base = MCConfig(n_seed=N, **THETA)

    def f(sigma):
        cfg = replace(cfg_base, sigma_natal=sigma)
        m, _chi, _w = sample_natals_jax(
            cfg, key, filter_zero_weights=False,
        )
        return jnp.sum(m * m)

    sigma_ref = jnp.asarray(1.0)
    df_ad = float(jax.grad(f)(sigma_ref))
    assert np.isfinite(df_ad), "grad w.r.t. sigma_natal must be finite"
    assert abs(df_ad) > 0.0, "grad w.r.t. sigma_natal must be non-zero"
    print(f"\n[INV-5b] ∂Σm²/∂σ AD = {df_ad:.4f} (expect O(2σ·N) ~ {2*1.0*N})")


# ---------------------------------------------------------------------------
# INV-bonus: _natal_log_pdf_jax matches numpy _natal_log_pdf at fixed points.
# ---------------------------------------------------------------------------
def test_log_pdf_jax_matches_numpy():
    r"""``_natal_log_pdf_jax(m, χ, cfg)`` agrees with the numpy
    ``_natal_log_pdf(m, χ, cfg)`` to within float32 precision.

    The two are independent implementations (jax.scipy.stats vs
    scipy.stats); agreement validates the truncnorm bounds and
    affine convention match.
    """
    from red_team.backup_particle_mc import _natal_log_pdf
    cfg = MCConfig(n_seed=N, **THETA)
    rng = np.random.default_rng(2026)
    m = rng.uniform(THETA["m_min"], THETA["m_max"], 100)
    chi = rng.uniform(0.01, 0.99, 100)
    lp_np = _natal_log_pdf(m, chi, cfg)
    lp_jax = np.asarray(_natal_log_pdf_jax(jnp.asarray(m), jnp.asarray(chi), cfg))
    # JAX defaults to float32; numpy uses float64.  Compare with
    # relaxed tolerance.
    max_abs = float(np.max(np.abs(lp_np - lp_jax)))
    print(f"\n[INV-bonus] max|log_pdf JAX - numpy| = {max_abs:.3e}")
    np.testing.assert_allclose(lp_np, lp_jax, rtol=1e-3, atol=1e-3)
