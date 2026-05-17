"""Finite-difference vs jax.grad verification of the engine.

This is the headline robustness test: for each differentiable physics
parameter, the analytic gradient via ``jax.grad`` of the full pipeline
``log_rate_density`` must match a central-finite-difference estimate.

The full pipeline includes: natal IMF + spin → coagulation solver (multiple
generations, scatter, retention) → synthetic-universe convolution → log_prob.
Passing this end-to-end is a strong correctness signal.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from hierarchical_ad_pop_engine.formation import LogNormalFormation
from hierarchical_ad_pop_engine.grids import BinaryGrid, EnvGrid, RedshiftGrid
from hierarchical_ad_pop_engine.natal import MadauDickinsonRedshift
from hierarchical_ad_pop_engine.response.coagulation import build_default_coagulation
from hierarchical_ad_pop_engine.universe import SyntheticUniverse


def _build_universe(alpha=3.0, kernel_b=2.0, log_mu_v_esc=jnp.log(300.0)):
    redshift = MadauDickinsonRedshift(
        z_max=jnp.asarray(3.0),
        kappa=jnp.asarray(2.7),
        gamma=jnp.asarray(2.9),
        z_peak=jnp.asarray(1.9),
    )
    response = build_default_coagulation(
        alpha=jnp.asarray(alpha),
        delta_m=jnp.asarray(5.0),
        lambda_peak=jnp.asarray(0.03),
        mu_peak=jnp.asarray(33.0),
        sigma_peak=jnp.asarray(3.0),
        m_min=jnp.asarray(5.0),
        m_max=jnp.asarray(80.0),
        natal_spin_loc=jnp.asarray(0.1),
        natal_spin_scale=jnp.asarray(0.15),
        kernel_a=jnp.asarray(0.0),
        kernel_b=jnp.asarray(kernel_b),
        log_R_env=jnp.asarray(0.0),
        k_max=2,
        grid=BinaryGrid.default(m_min=5.0, m_max=80.0, n_m=20, n_chi=8),
    )
    formation = LogNormalFormation(
        log_R0=jnp.asarray(0.0),
        log_mu_v_esc=jnp.asarray(log_mu_v_esc),
        log_sigma_v_esc=jnp.asarray(0.5),
        redshift=redshift,
    )
    return SyntheticUniverse(
        response=response,
        formation=formation,
        # Larger env grid → tighter trapezoidal quadrature precision; the
        # FD-vs-grad agreement test for log_mu_v_esc is sensitive to this.
        # With Heaviside retention (no AD on v_esc), each v_esc grid point
        # contributes a piecewise-constant function of the retained progenitor
        # mass; integration accuracy depends on resolving that staircase.
        env_grid=EnvGrid.default(v_esc_min=30.0, v_esc_max=2000.0, n_ve=32),
        z_grid=RedshiftGrid.default(n_z=8),
    )


def _eval_at(universe, m1=40.0, m2=20.0, chi1=0.2, chi2=0.1, z=0.5):
    return universe.log_rate_density(
        jnp.asarray(m1),
        jnp.asarray(m2),
        jnp.asarray(chi1),
        jnp.asarray(chi2),
        jnp.asarray(z),
    )


@pytest.mark.parametrize("param_name", ["alpha", "kernel_b", "log_mu_v_esc"])
def test_grad_matches_finite_difference(param_name):
    """jax.grad vs central-difference for each parameter in {alpha, kernel_b, log_mu_v_esc}."""
    if param_name == "alpha":
        x0 = 3.0
        h = 1e-3
    elif param_name == "kernel_b":
        x0 = 2.0
        h = 1e-3
    elif param_name == "log_mu_v_esc":
        x0 = float(jnp.log(300.0))
        h = 1e-3

    def f(x):
        kwargs = {param_name: x}
        u = _build_universe(**kwargs)
        return _eval_at(u)

    # Analytic gradient.
    g_analytic = jax.grad(f)(jnp.asarray(x0))

    # Central finite difference.
    g_fd = (f(jnp.asarray(x0 + h)) - f(jnp.asarray(x0 - h))) / (2.0 * h)

    # Loose-ish tolerance: the engine has a Heaviside retention step that
    # produces discrete contributions per v_esc grid cell, so the FD step
    # captures slightly different "set of retained cells" than the analytic
    # gradient when the FD perturbation crosses cell boundaries. 1% rel
    # agreement is the realistic ceiling for a coarse env grid.
    assert jnp.isfinite(g_analytic) and jnp.isfinite(g_fd)
    assert float(g_analytic) == pytest.approx(float(g_fd), rel=1e-2, abs=1e-5)


def test_full_pipeline_jit_compiles():
    """The end-to-end pipeline JITs without concrete-shape errors."""
    universe = _build_universe()
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
