"""FormationRate protocol — universe-scale environment count density.

A ``FormationRate`` describes how often environments of various kinds exist
across the universe:

.. math::

    F(\\theta_{\\rm env}, z)  =  dN_{\\rm env} / dV_c / dz / d\\theta_{\\rm env}

Implementations must be ``equinox.Module`` subclasses so they are JAX
pytrees, with ``log_density`` JIT-able and AD-differentiable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Array


@runtime_checkable
class FormationRate(Protocol):
    """Universe-scale environment number-density protocol.

    Implementations expose:

    ``log_density(env, z)`` →
        Array of ``log[dN_env / dV_c / dz / dθ_env]`` evaluated at
        ``(env, z)``. ``env`` is a pytree of environment parameters,
        details implementation-specific.

    Returned units: ``Gpc^-3 yr^-1`` (per unit env parameter, per unit z).
    """

    def log_density(self, env: dict, z: Array) -> Array:  # pragma: no cover
        ...
