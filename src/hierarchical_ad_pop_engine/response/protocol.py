"""ResponseFunction protocol — per-environment merger-rate density.

A ``ResponseFunction`` represents the physics inside a single dense
environment (one AGN disk, one NSC, one cluster). Given binary parameters
``θ_b = (m1, m2, χ1, χ2)`` and environment parameters ``θ_env``, it returns
the differential merger rate per environment per time.

Implementations must be ``equinox.Module`` subclasses so they are JAX
pytrees, with ``log_rate_density`` JIT-able and AD-differentiable end-to-end.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Array


@runtime_checkable
class ResponseFunction(Protocol):
    """Per-environment merger-rate density protocol.

    Implementations expose:

    ``log_rate_density(m1, m2, chi1, chi2, env)`` →
        Array of ``log[dN_merge / dt / dm1 dm2 dχ1 dχ2]`` per single
        environment, evaluated at the given binary parameter points. ``env``
        is a pytree of environment parameters (for v1, typically just
        ``v_esc``); details are implementation-specific.

    Returned units: ``yr^-1 (M⊙)^-2`` (per environment).
    """

    def log_rate_density(
        self,
        m1: Array,
        m2: Array,
        chi1: Array,
        chi2: Array,
        env: dict,
    ) -> Array:  # pragma: no cover - protocol stub
        ...
