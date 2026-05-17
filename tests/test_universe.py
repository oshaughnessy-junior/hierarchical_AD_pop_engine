"""Synthetic-universe convolution tests.

The headline test is the "delta-formation recovers the bare response" invariant
(plus the redshift profile factor): this is the trivial limit that the
response/formation factorisation must satisfy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hierarchical_ad_pop_engine.formation import DeltaFormation, LogNormalFormation
from hierarchical_ad_pop_engine.grids import BinaryGrid, EnvGrid, RedshiftGrid
from hierarchical_ad_pop_engine.natal import MadauDickinsonRedshift
from hierarchical_ad_pop_engine.response.coagulation import build_default_coagulation
from hierarchical_ad_pop_engine.universe import SyntheticUniverse


@pytest.fixture
def md_redshift():
    return MadauDickinsonRedshift(
        z_max=jnp.asarray(3.0),
        kappa=jnp.asarray(2.7),
        gamma=jnp.asarray(2.9),
        z_peak=jnp.asarray(1.9),
    )


@pytest.fixture
def small_response():
    return build_default_coagulation(
        k_max=2,
        grid=BinaryGrid.default(m_min=5.0, m_max=80.0, n_m=16, n_chi=6),
    )


def test_delta_formation_recovers_bare_response(small_response, md_redshift):
    """The headline architectural invariant."""
    v_esc_0 = jnp.asarray(500.0)
    formation = DeltaFormation(
        env_fiducial={"v_esc": v_esc_0},
        log_R0=jnp.asarray(0.0),
        redshift=md_redshift,
    )
    universe = SyntheticUniverse(
        response=small_response,
        formation=formation,
        env_grid=EnvGrid.default(n_ve=8),
        z_grid=RedshiftGrid.default(n_z=8),
    )

    m1, m2, chi1, chi2, z = (
        jnp.asarray(40.0),
        jnp.asarray(20.0),
        jnp.asarray(0.2),
        jnp.asarray(0.1),
        jnp.asarray(0.5),
    )

    universe_val = universe.log_rate_density(m1, m2, chi1, chi2, z)
    bare = small_response.log_rate_density(m1, m2, chi1, chi2, {"v_esc": v_esc_0})
    z_factor = md_redshift.log_prob(z)

    expected = bare + z_factor
    assert float(universe_val) == pytest.approx(float(expected), rel=1e-6)


def test_lognormal_formation_finite_and_jittable(small_response, md_redshift):
    formation = LogNormalFormation(
        log_R0=jnp.asarray(0.0),
        log_mu_v_esc=jnp.asarray(jnp.log(300.0)),
        log_sigma_v_esc=jnp.asarray(0.5),
        redshift=md_redshift,
    )
    universe = SyntheticUniverse(
        response=small_response,
        formation=formation,
        env_grid=EnvGrid.default(v_esc_min=20.0, v_esc_max=2000.0, n_ve=8),
        z_grid=RedshiftGrid.default(n_z=8),
    )
    f = jax.jit(
        lambda m1, m2, chi1, chi2, z: universe.log_rate_density(m1, m2, chi1, chi2, z)
    )
    val = f(
        jnp.asarray(40.0),
        jnp.asarray(20.0),
        jnp.asarray(0.2),
        jnp.asarray(0.1),
        jnp.asarray(0.5),
    )
    assert jnp.isfinite(val)


def test_universe_log_rate_density_differentiable(small_response, md_redshift):
    """End-to-end gradient: param of formation rate flows through convolution and response."""
    formation = LogNormalFormation(
        log_R0=jnp.asarray(0.0),
        log_mu_v_esc=jnp.asarray(jnp.log(300.0)),
        log_sigma_v_esc=jnp.asarray(0.5),
        redshift=md_redshift,
    )
    universe = SyntheticUniverse(
        response=small_response,
        formation=formation,
        env_grid=EnvGrid.default(v_esc_min=20.0, v_esc_max=2000.0, n_ve=8),
        z_grid=RedshiftGrid.default(n_z=8),
    )

    def loss(mu):
        new_formation = LogNormalFormation(
            log_R0=jnp.asarray(0.0),
            log_mu_v_esc=mu,
            log_sigma_v_esc=jnp.asarray(0.5),
            redshift=md_redshift,
        )
        new_universe = SyntheticUniverse(
            response=universe.response,
            formation=new_formation,
            env_grid=universe.env_grid,
            z_grid=universe.z_grid,
        )
        return new_universe.log_rate_density(
            jnp.asarray(40.0),
            jnp.asarray(20.0),
            jnp.asarray(0.2),
            jnp.asarray(0.1),
            jnp.asarray(0.5),
        )

    g = jax.grad(loss)(jnp.asarray(jnp.log(300.0)))
    assert jnp.isfinite(g)
