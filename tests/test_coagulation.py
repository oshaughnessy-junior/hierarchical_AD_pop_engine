"""Tests for the coagulation response function — limits, sanity, AD."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.response.coagulation import (
    CoagulationResponse,
    build_default_coagulation,
)


@pytest.fixture
def small_response():
    """Cheap CoagulationResponse for fast tests (small grid, k_max=2)."""
    return build_default_coagulation(
        k_max=2,
        grid=BinaryGrid.default(m_min=5.0, m_max=80.0, n_m=24, n_chi=8),
    )


def test_natal_density_finite_inside_support(small_response: CoagulationResponse):
    log_n = small_response._natal_log_density()
    assert log_n.shape == (24, 8)
    # Most cells should be finite (some edges go to -inf below m_min/above m_max).
    assert jnp.sum(jnp.isfinite(log_n)) > 24 * 8 // 2


def test_solve_returns_K_layers(small_response: CoagulationResponse):
    log_n_k = small_response._solve(jnp.asarray(500.0))
    assert log_n_k.shape == (2, 24, 8)
    # 2G density should be smaller than 1G integrated.
    n_1_total = jnp.sum(jnp.exp(log_n_k[0]))
    n_2_total = jnp.sum(jnp.exp(log_n_k[1]))
    assert float(n_2_total) < float(n_1_total)


def test_log_rate_density_finite(small_response: CoagulationResponse):
    out = small_response.log_rate_density(
        m1=jnp.asarray(40.0),
        m2=jnp.asarray(20.0),
        chi1=jnp.asarray(0.2),
        chi2=jnp.asarray(0.1),
        env={"v_esc": jnp.asarray(500.0)},
    )
    assert jnp.isfinite(out)


def test_zero_retention_kills_higher_generations():
    """v_esc = 0 → Heaviside `v_kick < v_esc` is never satisfied → n_k=0 for k≥2.

    Note: any v_esc > 0 would leave the q=1 / χ₁=χ₂ "diagonal" of the
    progenitor grid contributing (because v_kick = 0 there exactly), so the
    test must use v_esc = 0 to fully zero retention.
    """
    resp = build_default_coagulation(
        k_max=3,
        grid=BinaryGrid.default(m_min=5.0, m_max=80.0, n_m=24, n_chi=8),
    )
    log_n_k = resp._solve(jnp.asarray(0.0))
    n_2_total = jnp.sum(jnp.exp(log_n_k[1]))
    assert float(n_2_total) < 1e-6


def test_high_retention_grows_higher_generations():
    """v_esc → ∞ → high retention → meaningful n_2."""
    resp = build_default_coagulation(
        k_max=2,
        grid=BinaryGrid.default(m_min=5.0, m_max=200.0, n_m=24, n_chi=8),
    )
    log_n_low = resp._solve(jnp.asarray(50.0))
    log_n_high = resp._solve(jnp.asarray(5000.0))
    n_2_low = jnp.sum(jnp.exp(log_n_low[1]))
    n_2_high = jnp.sum(jnp.exp(log_n_high[1]))
    assert float(n_2_high) > float(n_2_low)


def test_log_rate_density_differentiable():
    """Gradient flows through the whole solver."""
    resp = build_default_coagulation(
        k_max=2,
        grid=BinaryGrid.default(m_min=5.0, m_max=80.0, n_m=16, n_chi=6),
    )

    def loss(alpha):
        natal_params = {**resp.natal_mass_params, "alpha": alpha}
        new = CoagulationResponse(
            grid=resp.grid,
            k_max=resp.k_max,
            natal_mass_params=natal_params,
            natal_spin_params=resp.natal_spin_params,
            kernel_params=resp.kernel_params,
            log_R_env=resp.log_R_env,
        )
        return new.log_rate_density(
            jnp.asarray(40.0),
            jnp.asarray(20.0),
            jnp.asarray(0.2),
            jnp.asarray(0.1),
            {"v_esc": jnp.asarray(500.0)},
        )

    g = jax.grad(loss)(jnp.asarray(3.0))
    assert jnp.isfinite(g)
