"""numpyro.Distribution wrapper for a :class:`SyntheticUniverse`.

Wraps the synthetic-universe rate-density as a single
``numpyro.distributions.Distribution`` over the event vector
``[m1, m2, χ1, χ2, z]``. Follows gwkokab's rate-density convention:
``log_prob`` returns ``log[dN / dV_c dt dθ]`` (not unit-normalised) so it
composes correctly with gwkokab's :func:`numpyro_poisson_likelihood`.

Factory function
----------------

:func:`coagulation_model` is a drop-in replacement for
``gwkokab.models.hybrids.PowerlawPeak`` — pass it kwargs and it returns a
``ScaledMixture([log_rate], [CoagulationPopulation])`` ready to be used by
``gwkokab.inference.numpyro_poisson_likelihood``.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from gwkokab.models.constraints import mass_sandwich
from gwkokab.models.utils import ScaledMixture
from jax import Array
from numpyro.distributions import Distribution, constraints

from .formation.delta import DeltaFormation
from .formation.parametric import LogNormalFormation
from .grids import BinaryGrid, EnvGrid, RedshiftGrid
from .natal import MadauDickinsonRedshift
from .response.coagulation import build_default_coagulation
from .universe import SyntheticUniverse


class CoagulationPopulation(Distribution):
    """Coagulation-engine population as a numpyro Distribution.

    Event vector: ``[m1, m2, χ1, χ2, z]``.

    ``log_prob`` returns ``log[dN / dV_c dt dm1 dm2 dχ1 dχ2 dz]`` (rate
    density, matching gwkokab convention; the redshift factor includes
    ``dV_c/dz`` and ``1/(1+z)`` per the underlying redshift distribution).

    Parameters
    ----------
    universe : SyntheticUniverse
        The convolved response × formation rate.
    m_min, m_max : ArrayLike
        Mass-sandwich bounds for the support.
    z_max : ArrayLike
        Maximum redshift in the support.
    """

    pytree_data_fields = ("universe", "m_min", "m_max", "z_max", "_support")

    def __init__(
        self,
        universe: SyntheticUniverse,
        m_min: Array,
        m_max: Array,
        z_max: Array,
        *,
        validate_args: Optional[bool] = None,
    ):
        self.universe = universe
        self.m_min = m_min
        self.m_max = m_max
        self.z_max = z_max
        self._support = _CoagulationSupport(m_min, m_max, z_max)
        super().__init__(batch_shape=(), event_shape=(5,), validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self):
        return self._support

    def log_prob(self, value: Array) -> Array:
        m1, m2, chi1, chi2, z = jnp.unstack(value, axis=-1)
        return self.universe.log_rate_density(m1, m2, chi1, chi2, z)

    def sample(self, key, sample_shape=()):  # pragma: no cover - rejection sampling
        raise NotImplementedError(
            "CoagulationPopulation does not provide direct sampling. Use the "
            "rate-density log_prob with gwkokab's PoissonLikelihood."
        )


class _CoagulationSupport(constraints.Constraint):
    """Product constraint for [m1, m2, χ1, χ2, z]."""

    event_dim = 1

    def __init__(self, m_min: Array, m_max: Array, z_max: Array):
        self.m_min = m_min
        self.m_max = m_max
        self.z_max = z_max
        self._mass = mass_sandwich(m_min, m_max)

    def __call__(self, x: Array) -> Array:
        m1, m2, chi1, chi2, z = jnp.unstack(x, axis=-1)
        mass_ok = self._mass(jnp.stack([m1, m2], axis=-1))
        chi1_ok = (chi1 >= 0.0) & (chi1 <= 1.0)
        chi2_ok = (chi2 >= 0.0) & (chi2 <= 1.0)
        z_ok = (z >= 0.0) & (z <= self.z_max)
        return mass_ok & chi1_ok & chi2_ok & z_ok

    def feasible_like(self, prototype: Array) -> Array:
        m_mid = jnp.sqrt(self.m_min * self.m_max)
        return jnp.broadcast_to(
            jnp.asarray([m_mid, m_mid * 0.5, 0.0, 0.0, 0.0], dtype=prototype.dtype),
            prototype.shape,
        )
# --- ADD THESE TWO METHODS TO FIX THE ERROR ---
    def tree_flatten(self):
        # Return a tuple of dynamic parameters, and a tuple of auxiliary (static) data
        return (), ()  

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        return cls()


# -----------------------------------------------------------------------------
# Factory functions
# -----------------------------------------------------------------------------


def coagulation_model(
    *,
    # Natal IMF (Talbot-Thrane PL + Peak)
    alpha: Array,
    delta_m: Array,
    lambda_peak: Array,
    mu_peak: Array,
    sigma_peak: Array,
    m_min: Array,
    m_max: Array,
    # Natal spin (TruncatedNormal on [0, 1])
    natal_spin_loc: Array,
    natal_spin_scale: Array,
    # Pairing kernel
    kernel_a: Array,
    kernel_b: Array,
    # Per-environment response normalisation
    log_R_env: Array,
    # Formation rate: log-normal in v_esc × Madau-Dickinson
    log_R_formation: Array,
    log_mu_v_esc: Array,
    log_sigma_v_esc: Array,
    z_max: Array,
    kappa: Array,
    gamma: Array,
    z_peak: Array,
    # Population-level total rate
    log_rate: Array,
    # Static configuration (compile-time)
    k_max: int = 4,
    n_m: int = 96,
    n_chi: int = 16,
    n_ve: int = 16,
    validate_args: Optional[bool] = None,
) -> ScaledMixture:
    """Build a ``ScaledMixture`` over the coagulation population.

    Drop-in replacement for ``gwkokab.models.hybrids.PowerlawPeak`` — usable
    directly with ``gwkokab.inference.numpyro_poisson_likelihood``.

    The signature splits parameters into (1) physics (natal, kernel,
    response, formation) and (2) static grid configuration (compile-time).
    All physics parameters are differentiable.
    """
    grid = BinaryGrid.default(m_min=float(m_min), m_max=float(m_max), n_m=n_m, n_chi=n_chi)
    env_grid = EnvGrid.default(n_ve=n_ve)
    z_grid = RedshiftGrid.default(z_max=float(z_max))

    response = build_default_coagulation(
        alpha=alpha,
        delta_m=delta_m,
        lambda_peak=lambda_peak,
        mu_peak=mu_peak,
        sigma_peak=sigma_peak,
        m_min=m_min,
        m_max=m_max,
        natal_spin_loc=natal_spin_loc,
        natal_spin_scale=natal_spin_scale,
        kernel_a=kernel_a,
        kernel_b=kernel_b,
        log_R_env=log_R_env,
        k_max=k_max,
        grid=grid,
    )

    redshift = MadauDickinsonRedshift(
        z_max=z_max, kappa=kappa, gamma=gamma, z_peak=z_peak, validate_args=validate_args
    )

    formation = LogNormalFormation(
        log_R0=log_R_formation,
        log_mu_v_esc=log_mu_v_esc,
        log_sigma_v_esc=log_sigma_v_esc,
        redshift=redshift,
    )

    universe = SyntheticUniverse(
        response=response,
        formation=formation,
        env_grid=env_grid,
        z_grid=z_grid,
    )

    dist = CoagulationPopulation(
        universe=universe,
        m_min=m_min,
        m_max=m_max,
        z_max=z_max,
        validate_args=validate_args,
    )

    return ScaledMixture(
        log_scales=jnp.asarray([log_rate]),
        component_distributions=[dist],
        support=dist.support,
        validate_args=validate_args,
    )


def coagulation_model_delta_formation(
    *,
    alpha: Array,
    delta_m: Array,
    lambda_peak: Array,
    mu_peak: Array,
    sigma_peak: Array,
    m_min: Array,
    m_max: Array,
    natal_spin_loc: Array,
    natal_spin_scale: Array,
    kernel_a: Array,
    kernel_b: Array,
    log_R_env: Array,
    v_esc_fiducial: Array,
    log_R_formation: Array,
    z_max: Array,
    kappa: Array,
    gamma: Array,
    z_peak: Array,
    log_rate: Array,
    k_max: int = 4,
    n_m: int = 96,
    n_chi: int = 16,
    validate_args: Optional[bool] = None,
) -> ScaledMixture:
    """Same as :func:`coagulation_model` but with a :class:`DeltaFormation` —
    a single-environment hypothesis at fixed ``v_esc``. Useful for fitting
    the cosmic rate of a single channel without env-distribution freedom.
    """
    grid = BinaryGrid.default(m_min=float(m_min), m_max=float(m_max), n_m=n_m, n_chi=n_chi)
    env_grid = EnvGrid.default(n_ve=4)  # ignored on the delta-path
    z_grid = RedshiftGrid.default(z_max=float(z_max))

    response = build_default_coagulation(
        alpha=alpha,
        delta_m=delta_m,
        lambda_peak=lambda_peak,
        mu_peak=mu_peak,
        sigma_peak=sigma_peak,
        m_min=m_min,
        m_max=m_max,
        natal_spin_loc=natal_spin_loc,
        natal_spin_scale=natal_spin_scale,
        kernel_a=kernel_a,
        kernel_b=kernel_b,
        log_R_env=log_R_env,
        k_max=k_max,
        grid=grid,
    )

    redshift = MadauDickinsonRedshift(
        z_max=z_max, kappa=kappa, gamma=gamma, z_peak=z_peak, validate_args=validate_args
    )

    formation = DeltaFormation(
        env_fiducial={"v_esc": v_esc_fiducial},
        log_R0=log_R_formation,
        redshift=redshift,
    )

    universe = SyntheticUniverse(
        response=response,
        formation=formation,
        env_grid=env_grid,
        z_grid=z_grid,
    )

    dist = CoagulationPopulation(
        universe=universe,
        m_min=m_min,
        m_max=m_max,
        z_max=z_max,
        validate_args=validate_args,
    )

    return ScaledMixture(
        log_scales=jnp.asarray([log_rate]),
        component_distributions=[dist],
        support=dist.support,
        validate_args=validate_args,
    )
