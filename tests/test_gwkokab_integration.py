"""End-to-end gwkokab integration test for the coagulation Distribution wrapper."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hierarchical_ad_pop_engine.distributions import (
    coagulation_model,
    coagulation_model_delta_formation,
)


_PARAMS = dict(
    alpha=jnp.asarray(3.0),
    delta_m=jnp.asarray(5.0),
    lambda_peak=jnp.asarray(0.03),
    mu_peak=jnp.asarray(33.0),
    sigma_peak=jnp.asarray(3.0),
    m_min=jnp.asarray(5.0),
    m_max=jnp.asarray(80.0),
    natal_spin_loc=jnp.asarray(0.1),
    natal_spin_scale=jnp.asarray(0.15),
    kernel_a=jnp.asarray(0.0),
    kernel_b=jnp.asarray(2.0),
    log_R_env=jnp.asarray(0.0),
    log_R_formation=jnp.asarray(0.0),
    log_mu_v_esc=jnp.asarray(jnp.log(300.0)),
    log_sigma_v_esc=jnp.asarray(0.5),
    z_max=jnp.asarray(3.0),
    kappa=jnp.asarray(2.7),
    gamma=jnp.asarray(2.9),
    z_peak=jnp.asarray(1.9),
    log_rate=jnp.asarray(2.0),
    k_max=2,
    n_m=16,
    n_chi=6,
    n_ve=6,
)


def test_coagulation_model_builds_scaled_mixture():
    mix = coagulation_model(**_PARAMS, validate_args=True)
    # ScaledMixture interface: log_prob over (m1, m2, χ1, χ2, z).
    value = jnp.asarray([40.0, 20.0, 0.2, 0.1, 0.5])
    out = mix.log_prob(value)
    assert jnp.isfinite(out)


def test_coagulation_model_log_prob_jittable():
    mix = coagulation_model(**_PARAMS, validate_args=False)
    f = jax.jit(mix.log_prob)
    value = jnp.asarray([40.0, 20.0, 0.2, 0.1, 0.5])
    out = f(value)
    assert jnp.isfinite(out)


def test_coagulation_model_log_prob_differentiable_in_alpha():
    static = {k: v for k, v in _PARAMS.items() if k in ("k_max", "n_m", "n_chi", "n_ve")}
    diff_keys = [k for k in _PARAMS if k not in static]

    def loss(alpha):
        params = {k: _PARAMS[k] for k in diff_keys}
        params["alpha"] = alpha
        mix = coagulation_model(**params, **static, validate_args=False)
        return mix.log_prob(jnp.asarray([40.0, 20.0, 0.2, 0.1, 0.5]))

    g = jax.grad(loss)(jnp.asarray(3.0))
    assert jnp.isfinite(g)


def test_coagulation_delta_model_finite():
    static = {k: v for k, v in _PARAMS.items() if k in ("k_max", "n_m", "n_chi")}
    delta_kwargs = {k: v for k, v in _PARAMS.items() if k not in ("n_ve",)}
    delta_kwargs = {
        k: v
        for k, v in delta_kwargs.items()
        if k not in ("log_mu_v_esc", "log_sigma_v_esc")
    }
    delta_kwargs["v_esc_fiducial"] = jnp.asarray(500.0)
    mix = coagulation_model_delta_formation(**delta_kwargs, validate_args=False)
    out = mix.log_prob(jnp.asarray([40.0, 20.0, 0.2, 0.1, 0.5]))
    assert jnp.isfinite(out)
