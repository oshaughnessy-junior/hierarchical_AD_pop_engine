"""SyntheticUniverse — convolve a ResponseFunction with a FormationRate.

The observable rate density is the convolution

.. math::

    dN_{\\rm obs} / dV_c / dt / d\\theta_b / dz
      = \\int  R(\\theta_b \\mid \\theta_{\\rm env})  \\,
              F(\\theta_{\\rm env}, z)  \\, d\\theta_{\\rm env}

For :class:`DeltaFormation` we short-circuit: there's no env integral, just
``R(θ_b | env_fid) × log_R0 × ψ(z)``. For general formations we do a
trapezoidal quadrature on the :class:`EnvGrid` over ``v_esc``.

The response total-density grids ``log N(m, χ | v_esc_i)`` are evaluated
once via ``jax.vmap`` over the env grid; AD passes through every step.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .formation.delta import DeltaFormation
from .formation.protocol import FormationRate
from .grids import EnvGrid, RedshiftGrid
from .interp import log_density_at
from .kernels import log_pairing_kernel
from .response.coagulation import CoagulationResponse
from .response.protocol import ResponseFunction


def _logtrapezoid_weights(x: Array) -> Array:
    """Trapezoidal cell widths for a 1D grid ``x``: ``Δx_i = ½(x_{i+1} - x_{i-1})`` interior,
    ``½(x_1 - x_0)`` and ``½(x_N - x_{N-1})`` at edges. Returns ``log`` weights."""
    dx_interior = 0.5 * (x[2:] - x[:-2])
    dx_first = jnp.atleast_1d(0.5 * (x[1] - x[0]))
    dx_last = jnp.atleast_1d(0.5 * (x[-1] - x[-2]))
    dx = jnp.concatenate([dx_first, dx_interior, dx_last])
    return jnp.log(dx)


class SyntheticUniverse(eqx.Module):
    """Convolution of a per-environment response with a universe-scale formation rate.

    Parameters
    ----------
    response : ResponseFunction
        Per-env merger-rate density (the coagulation solver in v1).
    formation : FormationRate
        Universe-scale environment count density.
    env_grid : EnvGrid
        Grid over the env parameter(s) used in the quadrature. Ignored when
        ``formation`` is a :class:`DeltaFormation`.
    z_grid : RedshiftGrid
        Redshift grid (not directly used here, kept for downstream tools).
    """

    response: ResponseFunction
    formation: FormationRate
    env_grid: EnvGrid
    z_grid: RedshiftGrid

    # --- Delta-formation fast path ------------------------------------------------

    def _log_rate_density_delta(
        self,
        m1: Array,
        m2: Array,
        chi1: Array,
        chi2: Array,
        z: Array,
    ) -> Array:
        assert isinstance(self.formation, DeltaFormation)
        env = self.formation.env_fiducial
        log_R = self.response.log_rate_density(m1, m2, chi1, chi2, env)
        log_z = self.formation.log_redshift_profile(z)
        return log_R + log_z

    # --- General-formation path ---------------------------------------------------

    def _log_total_density_stack(self) -> Array:
        """``log N(m, χ | v_esc_i)`` stacked over the env grid: shape ``(N_ve, N_m, N_chi)``."""
        assert isinstance(self.response, CoagulationResponse), (
            "General-formation path currently assumes the response is a "
            "CoagulationResponse (other response types can be added)."
        )
        return jax.vmap(self.response._log_total_density)(self.env_grid.v_esc)

    def _log_rate_density_general(
        self,
        m1: Array,
        m2: Array,
        chi1: Array,
        chi2: Array,
        z: Array,
    ) -> Array:
        assert isinstance(self.response, CoagulationResponse)
        log_N_stack = self._log_total_density_stack()  # (N_ve, N_m, N_chi)

        # Evaluate log N at (m1, χ1) and (m2, χ2) for each env grid point.
        def at(logN_2d):
            return log_density_at(
                self.response.grid.log_m, self.response.grid.chi, logN_2d, m1, chi1
            ), log_density_at(
                self.response.grid.log_m, self.response.grid.chi, logN_2d, m2, chi2
            )

        log_N1, log_N2 = jax.vmap(at)(log_N_stack)  # each (N_ve,)

        log_K = log_pairing_kernel(m1, m2, **self.response.kernel_params)
        log_R_env = self.response.log_R_env
        log_resp = log_R_env - jnp.log(2.0) + log_K + log_N1 + log_N2  # (N_ve,)

        # Formation rate at each env grid point.
        log_form = jax.vmap(
            lambda ve: self.formation.log_density({"v_esc": ve}, z)
        )(self.env_grid.v_esc)  # (N_ve,)

        log_dve = _logtrapezoid_weights(self.env_grid.v_esc)  # (N_ve,)

        # ∫ R(θ_b | ve) F(ve, z) dve = Σ exp(log_resp_i + log_form_i + log_dve_i)
        return jax.nn.logsumexp(log_resp + log_form + log_dve)

    # --- Public ------------------------------------------------------------------

    def log_rate_density(
        self,
        m1: Array,
        m2: Array,
        chi1: Array,
        chi2: Array,
        z: Array,
    ) -> Array:
        """Return ``log[dN_obs / dV_c / dt / dm1 dm2 dχ1 dχ2 / dz]`` at the query point.

        Uses the delta-formation fast path when applicable; otherwise integrates
        the response over the env grid.
        """
        if isinstance(self.formation, DeltaFormation):
            return self._log_rate_density_delta(m1, m2, chi1, chi2, z)
        return self._log_rate_density_general(m1, m2, chi1, chi2, z)
