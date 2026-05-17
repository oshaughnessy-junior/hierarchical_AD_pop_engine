"""Natal spin magnitude — truncated normal on ``χ ∈ [0, 1]``.

A truncated normal is smooth, well-behaved everywhere on the unit interval,
and AD-friendly in both parameters. No boundary singularities like the Beta
distribution can have at small mean / variance.

The two parameters ``loc`` and ``scale`` are the mean and standard
deviation of the *underlying* (untruncated) normal; the truncation to
``[0, 1]`` shifts the actual moments slightly, but for AGN-disk / dense-
cluster channels with typical loc ≪ 1 and small scale, the underlying
loc/scale are excellent proxies for the physical mean/std.
"""

from __future__ import annotations

from typing import Optional

from jax import Array
from jaxtyping import ArrayLike
from numpyro.distributions import TruncatedNormal


def build_natal_spin(
    loc: ArrayLike,
    scale: ArrayLike,
    *,
    validate_args: Optional[bool] = None,
) -> TruncatedNormal:
    """Construct the natal spin-magnitude distribution g_spin(χ) on ``χ ∈ [0, 1]``."""
    return TruncatedNormal(
        loc=loc,
        scale=scale,
        low=0.0,
        high=1.0,
        validate_args=validate_args,
    )


def log_natal_spin(dist: TruncatedNormal, chi: Array) -> Array:
    """Return ``log g_spin(χ)`` on ``chi`` (any shape).

    Outside ``[0, 1]`` numpyro's ``log_prob`` returns ``-inf`` per convention.
    """
    return dist.log_prob(chi)
