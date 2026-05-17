"""Sanity tests for remnant fits and retention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hierarchical_ad_pop_engine.remnants import (
    M_rem,
    chi_rem,
    eta,
    retention_aligned,
    v_kick_aligned,
)
from hierarchical_ad_pop_engine.retention import P_retain


def test_M_rem_is_less_than_total():
    m1, m2 = jnp.asarray(30.0), jnp.asarray(20.0)
    chi1, chi2 = jnp.asarray(0.0), jnp.asarray(0.0)
    M_total = m1 + m2
    M = M_rem(m1, m2, chi1, chi2)
    assert float(M) < float(M_total)
    # Radiated fraction should be a few percent for q ~ 0.67, χ=0.
    assert (float(M_total) - float(M)) / float(M_total) < 0.1


def test_M_rem_differentiable():
    g = jax.grad(lambda m1: M_rem(m1, jnp.asarray(20.0), jnp.asarray(0.0), jnp.asarray(0.0)))(
        jnp.asarray(30.0)
    )
    assert jnp.isfinite(g)


def test_eta_bounded():
    assert float(eta(jnp.asarray(10.0), jnp.asarray(10.0))) == pytest.approx(0.25)
    assert 0.0 <= float(eta(jnp.asarray(50.0), jnp.asarray(1.0))) <= 0.25


def test_chi_rem_bounded():
    # Strongly anti-aligned high spins shouldn't push χ_rem out of [-1, 1].
    for chi1, chi2 in [(-0.9, -0.9), (0.9, 0.9), (-0.9, 0.9)]:
        a = chi_rem(jnp.asarray(30.0), jnp.asarray(20.0), jnp.asarray(chi1), jnp.asarray(chi2))
        assert -1.0 < float(a) < 1.0


def test_chi_rem_q1_zero_spin_is_canonical_kerr_value():
    # Equal-mass, non-spinning merger should give χ_rem ~ 0.686 (Kerr/Schwarzschild ISCO).
    a = chi_rem(jnp.asarray(30.0), jnp.asarray(30.0), jnp.asarray(0.0), jnp.asarray(0.0))
    assert 0.5 < float(a) < 0.8


def test_v_kick_aligned_symmetric_zero():
    # Equal-mass, equal-spin binaries should have zero aligned-spin kick.
    v = v_kick_aligned(jnp.asarray(30.0), jnp.asarray(30.0), jnp.asarray(0.5), jnp.asarray(0.5))
    assert float(v) < 1.0  # km/s — numerical zero


def test_v_kick_aligned_finite_and_differentiable():
    f = lambda chi: v_kick_aligned(
        jnp.asarray(40.0), jnp.asarray(20.0), chi, jnp.asarray(0.0)
    )
    assert jnp.isfinite(f(jnp.asarray(0.3)))
    assert jnp.isfinite(jax.grad(f)(jnp.asarray(0.3)))


def test_P_retain_heaviside():
    """Aligned-only retention is a Heaviside step at v_kick."""
    m1, m2 = jnp.asarray(40.0), jnp.asarray(20.0)
    chi1, chi2 = jnp.asarray(0.3), jnp.asarray(0.0)
    v_kick = float(v_kick_aligned(m1, m2, chi1, chi2))
    assert float(P_retain(m1, m2, chi1, chi2, jnp.asarray(v_kick - 1.0))) == 0.0
    assert float(P_retain(m1, m2, chi1, chi2, jnp.asarray(v_kick + 1.0))) == 1.0


def test_P_retain_gradient_in_v_esc_is_zero():
    """Documented v1 limitation: Heaviside has zero gradient in v_esc."""
    f = lambda v: P_retain(
        jnp.asarray(40.0), jnp.asarray(20.0), jnp.asarray(0.3), jnp.asarray(0.0), v
    )
    g = jax.grad(f)(jnp.asarray(200.0))
    assert float(g) == 0.0


def test_retention_alias_matches_module_function():
    args = (
        jnp.asarray(40.0),
        jnp.asarray(20.0),
        jnp.asarray(0.3),
        jnp.asarray(0.0),
        jnp.asarray(500.0),
    )
    assert float(P_retain(*args)) == float(retention_aligned(*args))
