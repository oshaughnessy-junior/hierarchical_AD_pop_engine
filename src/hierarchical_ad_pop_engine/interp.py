"""Smooth, autodifferentiable interpolation of log-densities on the binary grid.

Internal densities ``n_k(m, χ)`` are stored as ``log n_k`` on a fixed
``(m, χ)`` grid (see :mod:`hierarchical_ad_pop_engine.grids`). Evaluation
at arbitrary points uses bilinear interpolation in (log m, χ) so the result
is C¹ in the query point and C∞ in the grid values — both required for
``jax.grad``.

Function signatures are kept pure (grid + log-density values + query points)
so they jit-compile and gradient cleanly.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def log_density_at(
    log_m_grid: Array,  # (N_m,)
    chi_grid: Array,  # (N_chi,)
    log_density: Array,  # (N_m, N_chi)
    m: Array,  # (...,)
    chi: Array,  # (...,)
) -> Array:
    """Bilinear interpolation of a log-density on the (m, χ) grid.

    Returns ``log_density`` evaluated at ``(m, chi)`` query points. Points
    outside the grid extrapolate to ``-jnp.inf`` (zero density).
    """
    log_m = jnp.log(m)
    # Locate query positions in grid coordinates [0, N-1].
    i = jnp.interp(log_m, log_m_grid, jnp.arange(log_m_grid.size))
    j = jnp.interp(chi, chi_grid, jnp.arange(chi_grid.size))

    # 2D map_coordinates with linear order is the JAX-friendly way:
    from jax.scipy.ndimage import map_coordinates

    out = map_coordinates(
        log_density,
        jnp.stack([i, j], axis=0),
        order=1,
        mode="constant",
        cval=-jnp.inf,
    )
    # Mask points strictly outside the grid box.
    inside = (
        (log_m >= log_m_grid[0])
        & (log_m <= log_m_grid[-1])
        & (chi >= chi_grid[0])
        & (chi <= chi_grid[-1])
    )
    return jnp.where(inside, out, -jnp.inf)
