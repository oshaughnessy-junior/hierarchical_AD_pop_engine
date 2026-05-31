r"""Two-pulse natal IMF — sandbox-tier invariants (JAX-pure, no gwkokab).

The ``natal_family="two_gaussian"`` model (methods-paper inject-at-10&30
experiment, INSTRUCTIONS_gwkokab_stage1 §5.1/§6).  These tests exercise the
parts that run in the lightweight JAX sandbox (no gwkokab/RIFT): the
two-pulse natal pdf + the IS-weight :func:`sample_natals_jax` path + the AD
chain through the full ``particle_mc_jax_soft`` cascade.  The engine-wrapper
(``build_two_pulse_coagulation`` → ``mc_jax``) and the gwkokab analytical-
Poisson recovery (``demo/fit_two_pulse_gwkokab.py``) are workstation-tier.

Invariants pinned:

T1. **Density normalisation** — the two-pulse mass marginal integrates to 1
    over ``[m_min, m_max]`` (each truncnorm component is independently
    normalised, so the convex mix is too).
T2. **IS-weight unbiasedness** — the importance-weighted pool reproduces the
    analytic mixture mean ``(1−λ)μ₁ + λμ₂`` (here 18.0) and the weights
    average to ≈1 (proper normalisation of q=½c₁+½c₂).
T3. **λ gradient flows (the IS-weight payoff)** — AD-vs-FD on
    ``d⟨m⟩/dλ`` through the pool agrees, and matches the closed form
    ``μ₂−μ₁``.  This is what hard component assignment would zero out.
T4. **μ₁, μ₂ gradients flow** — AD-vs-FD on the pulse locations.
T5. **Single-pulse regression** — ``lambda_natal=None`` leaves the default
    truncnorm path bit-for-bit (weights all 1, no two-pulse fields touched).
T6. **Full-cascade AD in λ** — ``d/dλ`` of an occupancy-weighted functional
    of the ``particle_mc_jax_soft`` surviving pool agrees AD-vs-FD; proves
    the two-pulse natals are differentiable end-to-end through the cascade
    (the SVI/NUTS requirement).
T7. **Both pulses populate the 1G pool** — the raw 50/50 IS proposal seeds
    particles at both ~10 and ~30 M⊙.

Run (after ``source demo/bootstrap_jax_sandbox.sh``)::

    python3 -m pytest tests/test_two_pulse_natal.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from red_team.backup_particle_mc import (  # noqa: E402
    MCConfig,
    particle_mc_jax_soft,
    sample_natals_jax,
    _two_pulse_active,
    _two_pulse_mass_logpdf_jax,
)

_BASE = dict(
    m_min=5.0, m_max=80.0, natal_spin_loc=0.05, natal_spin_scale=0.05,
    kernel_a=0.0, kernel_b=1.0, v_esc=300.0, n_seed=4000,
)


def _cfg(mu1=10.0, sig1=1.0, mu2=30.0, sig2=2.0, lam=0.4, **over):
    base = dict(_BASE)
    base.update(over)
    return MCConfig(mu_natal=mu1, sigma_natal=sig1, mu_natal_2=mu2,
                    sigma_natal_2=sig2, lambda_natal=lam, **base)


def test_T1_density_normalisation():
    cfg = _cfg()
    m = jnp.linspace(5.0, 80.0, 200_000)
    Z = jnp.trapezoid(jnp.exp(_two_pulse_mass_logpdf_jax(m, cfg)), m)
    assert abs(float(Z) - 1.0) < 1e-3


def test_T2_is_weight_unbiased():
    cfg = _cfg()
    m, chi, w = sample_natals_jax(cfg, jax.random.PRNGKey(1),
                                  filter_zero_weights=False)
    mean_is = float((w * m).sum() / w.sum())
    assert abs(mean_is - 18.0) < 0.5          # (1-0.4)*10 + 0.4*30
    assert abs(float(w.mean()) - 1.0) < 0.05  # q properly normalised


def _pool_mean_mass(mu1, mu2, lam):
    cfg = _cfg(mu1=mu1, mu2=mu2, lam=lam)
    m, chi, w = sample_natals_jax(cfg, jax.random.PRNGKey(1),
                                  filter_zero_weights=False)
    return (w * m).sum() / w.sum()


def test_T3_lambda_gradient_flows():
    g = float(jax.grad(lambda l: _pool_mean_mass(10.0, 30.0, l))(0.4))
    eps = 1e-3
    fd = float((_pool_mean_mass(10.0, 30.0, 0.4 + eps)
                - _pool_mean_mass(10.0, 30.0, 0.4 - eps)) / (2 * eps))
    assert abs(g - fd) / abs(fd) < 1e-4
    assert abs(g - (30.0 - 10.0)) < 0.3       # closed form d⟨m⟩/dλ = μ₂−μ₁


def test_T4_mu_gradients_flow():
    g1 = float(jax.grad(lambda u: _pool_mean_mass(u, 30.0, 0.4))(10.0))
    g2 = float(jax.grad(lambda u: _pool_mean_mass(10.0, u, 0.4))(30.0))
    eps = 1e-3
    fd1 = float((_pool_mean_mass(10.0 + eps, 30.0, 0.4)
                 - _pool_mean_mass(10.0 - eps, 30.0, 0.4)) / (2 * eps))
    fd2 = float((_pool_mean_mass(10.0, 30.0 + eps, 0.4)
                 - _pool_mean_mass(10.0, 30.0 - eps, 0.4)) / (2 * eps))
    assert abs(g1 - fd1) / abs(fd1) < 1e-3
    assert abs(g2 - fd2) / abs(fd2) < 1e-3
    # mixture weights ≈ (1-λ, λ) so ∂⟨m⟩/∂μ₁ ≈ 0.6, ∂⟨m⟩/∂μ₂ ≈ 0.4
    assert abs(g1 - 0.6) < 0.1 and abs(g2 - 0.4) < 0.1


def test_T5_single_pulse_regression():
    cfg = MCConfig(mu_natal=8.0, sigma_natal=1.0, **_BASE)
    assert _two_pulse_active(cfg) is False
    m, chi, w = sample_natals_jax(cfg, jax.random.PRNGKey(1),
                                  filter_zero_weights=False)
    assert bool(jnp.allclose(w, 1.0))
    assert w.size == _BASE["n_seed"]


def _cascade_scalar(mu1, mu2, lam, kb=1.0):
    cfg = MCConfig(
        mu_natal=mu1, sigma_natal=1.0, mu_natal_2=mu2, sigma_natal_2=2.0,
        lambda_natal=lam, kernel_b=kb, m_min=5.0, m_max=120.0,
        natal_spin_loc=0.02, natal_spin_scale=0.03, kernel_a=0.0,
        v_esc=400.0, n_seed=1500, depletion_fraction=0.3,
    )
    o = particle_mc_jax_soft(cfg, jax.random.PRNGKey(0),
                             n_steps=None, n_steps_mult=3.0)
    w = o["w_active"] * o["occupancy"]
    return (w * o["m_active"]).sum() / w.sum()


def test_T6_full_cascade_ad_in_lambda():
    g = float(jax.grad(lambda l: _cascade_scalar(10.0, 30.0, l))(0.4))
    eps = 2e-3
    fd = float((_cascade_scalar(10.0, 30.0, 0.4 + eps)
                - _cascade_scalar(10.0, 30.0, 0.4 - eps)) / (2 * eps))
    assert np.isfinite(g) and abs(g) > 1e-3
    assert abs(g - fd) / abs(fd) < 1e-4


def test_T7_both_pulses_seed_pool():
    cfg = _cfg(lam=0.4, n_seed=4000, depletion_fraction=0.05)
    o = particle_mc_jax_soft(cfg, jax.random.PRNGKey(0),
                             n_steps=None, n_steps_mult=3.0)
    g1 = np.asarray(o["gen_active"]) == 1
    m1 = np.asarray(o["m_active"])[g1]
    assert np.mean(np.abs(m1 - 10.0) < 3.0) > 0.3
    assert np.mean(np.abs(m1 - 30.0) < 5.0) > 0.3
