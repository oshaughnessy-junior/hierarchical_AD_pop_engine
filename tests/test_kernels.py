"""Tests for the pairing kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hierarchical_ad_pop_engine.kernels import (
    log_pairing_kernel,
    log_pairing_kernel_grid,
)


def test_pairing_kernel_symmetric():
    m1, m2 = jnp.asarray(30.0), jnp.asarray(20.0)
    a, b = jnp.asarray(1.5), jnp.asarray(2.0)
    assert float(log_pairing_kernel(m1, m2, a, b)) == pytest.approx(
        float(log_pairing_kernel(m2, m1, a, b))
    )


def test_pairing_kernel_q_dependence():
    # For b > 0, equal-mass should beat unequal-mass at fixed M_tot.
    a = jnp.asarray(0.0)
    b = jnp.asarray(2.0)
    eq = log_pairing_kernel(jnp.asarray(25.0), jnp.asarray(25.0), a, b)
    uneq = log_pairing_kernel(jnp.asarray(40.0), jnp.asarray(10.0), a, b)
    assert float(eq) > float(uneq)


def test_pairing_kernel_M_tot_dependence():
    # At fixed q=1, doubling M_tot scales kernel by 2^a.
    a, b = jnp.asarray(3.0), jnp.asarray(0.0)
    k1 = log_pairing_kernel(jnp.asarray(10.0), jnp.asarray(10.0), a, b)
    k2 = log_pairing_kernel(jnp.asarray(20.0), jnp.asarray(20.0), a, b)
    assert float(k2 - k1) == pytest.approx(3.0 * float(jnp.log(2.0)))


def test_pairing_kernel_grid_shape():
    m = jnp.linspace(5.0, 100.0, 8)
    K = log_pairing_kernel_grid(m, jnp.asarray(1.0), jnp.asarray(2.0))
    assert K.shape == (8, 8)
    # Symmetry on the diagonal.
    assert jnp.allclose(K, K.T)


def test_pairing_kernel_differentiable():
    g = jax.grad(
        lambda a: log_pairing_kernel(jnp.asarray(30.0), jnp.asarray(20.0), a, jnp.asarray(2.0))
    )(jnp.asarray(1.5))
    assert jnp.isfinite(g)
