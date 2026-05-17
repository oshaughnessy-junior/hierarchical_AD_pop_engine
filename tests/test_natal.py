"""Tests for the natal (1G) population layer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hierarchical_ad_pop_engine.natal import (
    build_natal_spin,
    log_natal_mass,
    log_natal_spin,
)


@pytest.fixture
def imf_params():
    return dict(
        alpha=jnp.asarray(3.0),
        delta_m=jnp.asarray(5.0),
        lambda_peak=jnp.asarray(0.03),
        mu_peak=jnp.asarray(33.0),
        sigma_peak=jnp.asarray(3.0),
        m_min=jnp.asarray(5.0),
        m_max=jnp.asarray(100.0),
    )


def test_log_natal_mass_finite_inside_support(imf_params):
    m = jnp.geomspace(6.0, 90.0, 32)
    logp = log_natal_mass(m, **imf_params)
    assert jnp.all(jnp.isfinite(logp))


def test_log_natal_mass_neg_inf_outside(imf_params):
    assert jnp.isneginf(log_natal_mass(jnp.asarray(2.0), **imf_params))
    assert jnp.isneginf(log_natal_mass(jnp.asarray(500.0), **imf_params))


def test_log_natal_mass_normalises_to_one(imf_params):
    m = jnp.geomspace(float(imf_params["m_min"]), float(imf_params["m_max"]), 4096)
    pdf = jnp.exp(log_natal_mass(m, **imf_params))
    integral = jnp.trapezoid(pdf, m)
    assert float(integral) == pytest.approx(1.0, rel=2e-2)


def test_log_natal_mass_differentiable(imf_params):
    def total_logp(alpha):
        params = {**imf_params, "alpha": alpha}
        m = jnp.geomspace(6.0, 90.0, 16)
        return log_natal_mass(m, **params).sum()

    g = jax.grad(total_logp)(jnp.asarray(3.0))
    assert jnp.isfinite(g)


def test_log_natal_spin_normalises_to_one():
    """TruncatedNormal on [0, 1] integrates to 1."""
    dist = build_natal_spin(loc=jnp.asarray(0.2), scale=jnp.asarray(0.15))
    chi = jnp.linspace(0.0, 1.0, 4096)
    integral = jnp.trapezoid(jnp.exp(log_natal_spin(dist, chi)), chi)
    assert float(integral) == pytest.approx(1.0, rel=2e-3)


def test_log_natal_spin_differentiable():
    def total_logp(loc):
        dist = build_natal_spin(loc=loc, scale=jnp.asarray(0.15))
        chi = jnp.linspace(0.01, 0.99, 32)
        return log_natal_spin(dist, chi).sum()

    g = jax.grad(total_logp)(jnp.asarray(0.2))
    assert jnp.isfinite(g)
