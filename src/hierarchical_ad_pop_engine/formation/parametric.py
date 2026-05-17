"""Parametric formation-rate models.

For v1 we provide a log-normal-in-``v_esc`` × redshift formation rate:

.. math::

    F(v_{\\rm esc}, z) =
        R_0 \\, \\mathrm{LogNormal}(v_{\\rm esc};\\mu, \\sigma) \\, \\psi(z)

with :math:`\\psi(z)` any gwkokab redshift distribution (``MadauDickinsonRedshift``,
``PowerlawRedshift``, ...). Both factors are differentiable in all parameters,
so the convolved synthetic-universe rate is differentiable end-to-end.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from numpyro.distributions import Distribution


class LogNormalFormation(eqx.Module):
    """Log-normal-in-``v_esc`` × redshift formation rate."""

    log_R0: Array
    log_mu_v_esc: Array  # log of mean v_esc (km/s)
    log_sigma_v_esc: Array  # log-scale width (dimensionless)
    redshift: Distribution

    def log_density(self, env: dict, z: Array) -> Array:
        """Return ``log[dN_env / dV_c / dz / dv_esc]``."""
        v_esc = env["v_esc"]
        log_v = jnp.log(v_esc)
        log_lognormal = (
            -0.5 * jnp.square((log_v - self.log_mu_v_esc) / self.log_sigma_v_esc)
            - jnp.log(v_esc)
            - jnp.log(self.log_sigma_v_esc)
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        log_z = self.redshift.log_prob(z)
        return self.log_R0 + log_lognormal + log_z

    def log_density_on_grids(self, v_esc_grid: Array, z_grid: Array) -> Array:
        """Evaluate ``log F`` on the cross product of the env and z grids.

        Returns shape ``(N_ve, N_z)``.
        """
        # Outer-product evaluation, vectorised.
        ve = v_esc_grid[:, None]
        zz = z_grid[None, :]
        log_v = jnp.log(ve)
        log_lognormal = (
            -0.5 * jnp.square((log_v - self.log_mu_v_esc) / self.log_sigma_v_esc)
            - jnp.log(ve)
            - jnp.log(self.log_sigma_v_esc)
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        log_z = self.redshift.log_prob(z_grid)  # (N_z,)
        return self.log_R0 + log_lognormal + log_z[None, :]
