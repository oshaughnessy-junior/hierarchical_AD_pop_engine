"""Tests for the formation-rate layer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hierarchical_ad_pop_engine.formation import (
    DeltaFormation,
    LogNormalFormation,
)
from hierarchical_ad_pop_engine.natal import MadauDickinsonRedshift


@pytest.fixture
def md_redshift():
    return MadauDickinsonRedshift(
        z_max=jnp.asarray(3.0),
        kappa=jnp.asarray(2.7),
        gamma=jnp.asarray(2.9),
        z_peak=jnp.asarray(1.9),
    )


def test_delta_formation_redshift_profile(md_redshift):
    f = DeltaFormation(
        env_fiducial={"v_esc": jnp.asarray(500.0)},
        log_R0=jnp.asarray(0.0),
        redshift=md_redshift,
    )
    out = f.log_redshift_profile(jnp.asarray(0.5))
    assert jnp.isfinite(out)


def test_delta_formation_returns_neg_inf_off_fiducial(md_redshift):
    f = DeltaFormation(
        env_fiducial={"v_esc": jnp.asarray(500.0)},
        log_R0=jnp.asarray(0.0),
        redshift=md_redshift,
    )
    out = f.log_density({"v_esc": jnp.asarray(123.4)}, jnp.asarray(0.5))
    assert jnp.isneginf(out)


def test_lognormal_formation_density_finite_and_differentiable(md_redshift):
    f = LogNormalFormation(
        log_R0=jnp.asarray(0.0),
        log_mu_v_esc=jnp.asarray(jnp.log(300.0)),
        log_sigma_v_esc=jnp.asarray(0.5),
        redshift=md_redshift,
    )
    out = f.log_density({"v_esc": jnp.asarray(250.0)}, jnp.asarray(0.5))
    assert jnp.isfinite(out)

    def loss(mu):
        f2 = LogNormalFormation(
            log_R0=jnp.asarray(0.0),
            log_mu_v_esc=mu,
            log_sigma_v_esc=jnp.asarray(0.5),
            redshift=md_redshift,
        )
        return f2.log_density({"v_esc": jnp.asarray(250.0)}, jnp.asarray(0.5))

    g = jax.grad(loss)(jnp.asarray(jnp.log(300.0)))
    assert jnp.isfinite(g)


def test_lognormal_formation_grid_shape(md_redshift):
    f = LogNormalFormation(
        log_R0=jnp.asarray(0.0),
        log_mu_v_esc=jnp.asarray(jnp.log(300.0)),
        log_sigma_v_esc=jnp.asarray(0.5),
        redshift=md_redshift,
    )
    ve = jnp.geomspace(10.0, 1000.0, 12)
    z = jnp.linspace(0.01, 2.99, 16)
    out = f.log_density_on_grids(ve, z)
    assert out.shape == (12, 16)
    assert jnp.all(jnp.isfinite(out))
