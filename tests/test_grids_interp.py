"""Smoke tests for the grid and interpolation utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hierarchical_ad_pop_engine.grids import BinaryGrid, EnvGrid, RedshiftGrid
from hierarchical_ad_pop_engine.interp import log_density_at


def test_binary_grid_default_shapes():
    g = BinaryGrid.default()
    assert g.m.shape == (96,)
    assert g.chi.shape == (32,)
    assert jnp.all(jnp.diff(g.m) > 0)
    assert jnp.allclose(g.log_m, jnp.log(g.m))


def test_redshift_grid_default_shapes():
    g = RedshiftGrid.default()
    assert g.z.shape == (64,)
    assert g.z[0] == 0.0
    assert float(g.z[-1]) == pytest.approx(4.0)


def test_env_grid_default_shapes():
    g = EnvGrid.default()
    assert g.v_esc.shape == (24,)
    assert jnp.all(jnp.diff(g.v_esc) > 0)


def test_interpolation_recovers_grid_values():
    g = BinaryGrid.default(n_m=8, n_chi=4)
    log_dens = jnp.outer(jnp.linspace(-2.0, 1.0, 8), jnp.linspace(0.0, 1.0, 4))
    # Query at a grid point — bilinear interp should return the exact value.
    m_query = g.m[3]
    chi_query = g.chi[1]
    out = log_density_at(g.log_m, g.chi, log_dens, m_query, chi_query)
    assert jnp.allclose(out, log_dens[3, 1])


def test_interpolation_outside_grid_is_neg_inf():
    g = BinaryGrid.default(n_m=8, n_chi=4)
    log_dens = jnp.zeros((8, 4))
    out = log_density_at(g.log_m, g.chi, log_dens, jnp.array(1000.0), jnp.array(0.5))
    assert jnp.isneginf(out)


def test_interpolation_at_upper_edge_is_finite_not_nan():
    r"""Regression: a query EXACTLY at the upper grid edge (and in the last
    half-cell) must be finite.  The old ``mode='constant', cval=-inf`` gave
    ``0·(-inf)=NaN`` at the last node and spurious ``-inf`` in the last
    half-cell, which poisoned the cross-method likelihood (scattered spins land
    near the χ edges).  ``mode='nearest'`` returns the edge value."""
    g = BinaryGrid.default(n_m=8, n_chi=4)
    log_dens = jnp.outer(jnp.linspace(-2.0, 1.0, 8), jnp.linspace(0.0, 1.0, 4))
    # exact upper corner → the corner grid value, finite (no NaN)
    corner = log_density_at(g.log_m, g.chi, log_dens, g.m[-1], g.chi[-1])
    assert jnp.isfinite(corner)
    assert jnp.allclose(corner, log_dens[-1, -1])
    # a point in the last χ half-cell (between χ_{N-2} and χ_{N-1}) → finite
    chi_near_edge = 0.5 * (g.chi[-2] + g.chi[-1])
    mid = log_density_at(g.log_m, g.chi, log_dens, g.m[3], chi_near_edge)
    assert jnp.isfinite(mid)
    # grad at the edge must also be finite (the −inf variant gave NaN grads)
    gm = jax.grad(
        lambda c: log_density_at(g.log_m, g.chi, log_dens, g.m[3], c).sum()
    )(g.chi[-1])
    assert jnp.isfinite(gm)


def test_interpolation_is_jittable_and_grad_friendly():
    g = BinaryGrid.default(n_m=8, n_chi=4)
    log_dens = jnp.outer(jnp.linspace(-2.0, 1.0, 8), jnp.linspace(0.0, 1.0, 4))
    f = jax.jit(
        lambda m, chi: log_density_at(g.log_m, g.chi, log_dens, m, chi)
    )
    val = f(jnp.array(10.0), jnp.array(0.3))
    assert jnp.isfinite(val)
    grad_m = jax.grad(lambda m: f(m, jnp.array(0.3)).sum())(jnp.array(10.0))
    assert jnp.isfinite(grad_m)
