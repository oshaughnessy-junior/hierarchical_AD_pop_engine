"""Delta-function formation rate: single environment type at fixed ``θ_env``.

Useful for verification (the convolution at this formation rate must recover
the bare response function) and for representing a single-channel hypothesis
where environment properties are pinned and only the cosmic rate is fit.

The redshift profile is any gwkokab redshift Distribution (e.g.
``MadauDickinsonRedshift``); we use its ``log_prob`` which returns
``log[dN/dV_c dt dz]`` in the rate-density convention.
"""

from __future__ import annotations

import equinox as eqx
from jax import Array
from numpyro.distributions import Distribution


class DeltaFormation(eqx.Module):
    """Single-environment formation rate ``F(θ_env, z) = R_0 δ(θ_env - θ_env^*) ψ(z)``.

    The convolution layer recognises this class via ``isinstance`` and skips
    the env-grid integral, evaluating the response directly at
    ``env_fiducial``.

    Parameters
    ----------
    env_fiducial : dict
        Fiducial environment parameters (e.g. ``{"v_esc": jnp.asarray(500.0)}``).
    log_R0 : Array
        log of the total per-env-time normalisation (dimensionless rate scale).
    redshift : numpyro.distributions.Distribution
        Redshift rate-density distribution (gwkokab convention).
    """

    env_fiducial: dict
    log_R0: Array
    redshift: Distribution

    def log_density(self, env: dict, z: Array) -> Array:
        """For verification only: returns log_R0 + log p(z) when env matches
        ``env_fiducial`` (within ``1e-6``), else ``-inf``.

        Most users should use this class via :class:`SyntheticUniverse`, which
        bypasses the env integral entirely.
        """
        import jax.numpy as jnp

        match = jnp.asarray(True)
        for k, v in self.env_fiducial.items():
            match = match & jnp.isclose(env[k], v, rtol=1e-6)
        log_z = self.redshift.log_prob(z)
        return jnp.where(match, self.log_R0 + log_z, -jnp.inf)

    def log_redshift_profile(self, z: Array) -> Array:
        """Return ``log R_0 + log_prob_z(z)`` — convenience for fast-path eval."""
        return self.log_R0 + self.redshift.log_prob(z)
