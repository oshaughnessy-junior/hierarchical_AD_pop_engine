"""Fixed grids for binary parameters (m, χ), redshift z, and environment params.

The engine represents every internal density n_k(m, χ) as values on a fixed grid.
This module owns the grid definitions and conventions so they're the same
everywhere (response function, formation rate, convolution).

Conventions (defaults; configurable):

- ``m``: log-spaced ∈ [2 M⊙, 200 M⊙], ``N_m = 96``.
- ``χ``: linear ∈ [0, 1], ``N_χ = 32``.
- ``z``: linear ∈ [0, 4], ``N_z = 64``.
- ``v_esc``: log-spaced ∈ [1, 1000] km/s, ``N_ve = 24``.

All grids are immutable equinox.Modules so they are JAX-tree leaves but never
recompile-trigger on identity changes.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array


class BinaryGrid(eqx.Module):
    """Fixed grid over (m, χ) for storing per-generation densities n_k(m, χ).

    Stored as 1D edge arrays plus pre-computed log-edges; convolutions
    elsewhere broadcast on the cross product.
    """

    m: Array  # shape (N_m,), solar masses
    chi: Array  # shape (N_chi,)
    log_m: Array  # cached log(m) for log-spacing convolutions

    @classmethod
    def default(
        cls,
        m_min: float = 2.0,
        m_max: float = 200.0,
        n_m: int = 96,
        n_chi: int = 32,
        chi_min: float = 1e-3,
        chi_max: float = 1.0 - 1e-3,
    ) -> "BinaryGrid":
        """Default (m, χ) grid.

        ``chi_min`` and ``chi_max`` are shifted slightly inside the unit
        interval so the natal Beta spin distribution — which has a PDF
        singularity at ``χ = 0`` whenever its shape parameter ``α < 1`` —
        evaluates to a finite log-density on every grid point. The default
        offset (1e-3) is well below the resolution of any realistic spin
        measurement and propagates negligible bias.
        """
        m = jnp.geomspace(m_min, m_max, n_m)
        chi = jnp.linspace(chi_min, chi_max, n_chi)
        return cls(m=m, chi=chi, log_m=jnp.log(m))


class RedshiftGrid(eqx.Module):
    """Fixed grid over redshift z, used by formation-rate and convolution layers."""

    z: Array  # shape (N_z,)

    @classmethod
    def default(cls, z_max: float = 4.0, n_z: int = 64) -> "RedshiftGrid":
        return cls(z=jnp.linspace(0.0, z_max, n_z))


class EnvGrid(eqx.Module):
    """Fixed grid over environment parameters (v_esc for v1).

    Future env parameters (SMBH mass, NSC mass, disk surface density, ...)
    will extend this; for v1 only ``v_esc`` is varied.
    """

    v_esc: Array  # shape (N_ve,), km/s

    @classmethod
    def default(
        cls, v_esc_min: float = 1.0, v_esc_max: float = 1000.0, n_ve: int = 24
    ) -> "EnvGrid":
        return cls(v_esc=jnp.geomspace(v_esc_min, v_esc_max, n_ve))
