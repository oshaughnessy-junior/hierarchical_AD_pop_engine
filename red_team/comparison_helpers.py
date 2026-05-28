r"""Pytest-friendly helpers extracted from ``compare_implicit_vs_mc.py``.

The original ``compare_implicit_vs_mc.py`` is a measurement script
with an ``argparse``-driven ``main()``.  The pytest target
``tests/test_redteam_Nx_parity.py`` needs ~half of its fixture
machinery — fixed-seed MC runs, KDE marginals on the engine grid, JS
on a mass-range subrange — but not the plotting or CLI scaffolding.

This module is the thin extraction: pure functions, no global state,
no CLI.  ``compare_implicit_vs_mc.py`` is left untouched so existing
research workflows continue to work; the pytest imports come from
here.

Functions kept verbatim from the source (so the comparison the
pytest gates on is identical to the script the team has already
been eyeballing for weeks):

* :func:`build_response_for_theta` — theta-dict → CoagulationResponse.
* :func:`run_implicit` — call ``solve_implicit_fixed_point`` and
  return the converged ``N(m, χ)`` together with axis arrays.
* :func:`run_mc` — run the particle MC with a fixed-seed RNG.
* :func:`mc_marginals_on_grid` — evaluate a KDE of the active pool
  at the solver's grid points (apples-to-apples JS).
* :func:`js_subrange` — JS divergence restricted to a mass sub-range.

The original ``compare_implicit_vs_mc.py`` keeps its full
``main()`` / plotting machinery; only the small primitives above
move.
"""
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np

# Engine production solver — same import path the script uses.
from hierarchical_ad_pop_engine.response.coagulation import (
    CoagulationResponse,
    build_gaussian_coagulation,
)
from hierarchical_ad_pop_engine.response.implicit import (
    ImplicitSolveResult,
    solve_implicit_fixed_point,
)

# Red-team particle MC + KDE primitives.
from red_team import backup_particle_mc as B1
from red_team.phase1_overlay import js_divergence
from red_team.sim_coagulation import make_marginal


def build_response_for_theta(
    theta: dict,
    *,
    n_m: int | None = None,
    n_chi: int | None = None,
    orientation: object | None = None,
    retention_kind: str = "sigmoid",
    w_over_v_esc: float = 1.0,
) -> CoagulationResponse:
    r"""Build a :class:`CoagulationResponse` from a theta dict.

    The solver field is irrelevant — the caller is expected to pass
    the result through :func:`solve_implicit_fixed_point` directly,
    which ignores ``response.solver``.  We use ``solver="perturbative"``
    (the default) so building the response is cheap.

    ``orientation``, ``retention_kind`` and ``w_over_v_esc`` forward
    to :func:`build_gaussian_coagulation` (Session B/C wired these
    through ``CoagulationResponse``).  Defaulting ``orientation`` to
    ``None`` preserves the v1 Aligned bit-for-bit path for every
    existing caller — see ``[[feedback-dispatch-callgraph-audit]]``.

    ``n_m`` / ``n_chi`` override the default grid resolution when
    provided.  When omitted, ``build_gaussian_coagulation`` uses
    :meth:`BinaryGrid.default` baselines.
    """
    extra: dict = {}
    if n_m is not None and n_chi is not None:
        # Construct an explicit grid only when the caller asks for a
        # non-default resolution on both axes; otherwise let the
        # factory default it.  Asymmetric override is intentionally
        # disallowed — N_m and N_χ tend to be tuned together.
        from hierarchical_ad_pop_engine.grids import BinaryGrid
        extra["grid"] = BinaryGrid.default(
            m_min=float(theta["m_min"]),
            m_max=float(theta["m_max"]),
            n_m=int(n_m),
            n_chi=int(n_chi),
        )
    elif n_m is not None or n_chi is not None:
        raise ValueError(
            "build_response_for_theta: n_m and n_chi must be set together "
            "(both None to use BinaryGrid.default, or both ints)."
        )
    return build_gaussian_coagulation(
        mu=theta["mu_natal"],
        sigma=theta["sigma_natal"],
        m_min=theta["m_min"],
        m_max=theta["m_max"],
        natal_spin_loc=theta["natal_spin_loc"],
        natal_spin_scale=theta["natal_spin_scale"],
        kernel_a=theta["kernel_a"],
        kernel_b=theta["kernel_b"],
        log_R_env=0.0,
        k_max=theta["k_max"],
        orientation=orientation,
        retention_kind=retention_kind,
        w_over_v_esc=w_over_v_esc,
        **extra,
    )


def run_implicit(
    theta: dict,
    epsilon: float,
    *,
    max_iter: int = 200,
    tol: float = 1e-7,
    damping: float = 0.4,
    n_m: int | None = None,
    n_chi: int | None = None,
    orientation: object | None = None,
    retention_kind: str = "sigmoid",
    w_over_v_esc: float = 1.0,
) -> tuple[CoagulationResponse, ImplicitSolveResult, np.ndarray, np.ndarray]:
    r"""Run the package implicit Picard solver at the given theta.

    ``orientation`` / ``retention_kind`` / ``w_over_v_esc`` forward
    through to :func:`build_response_for_theta`; defaults reproduce
    the v1 Aligned + Heaviside path bit-for-bit so existing callers
    are unaffected.  See ``[[feedback-dispatch-callgraph-audit]]``.

    Returns
    -------
    response : CoagulationResponse
        The constructed response (caller may want its grid / kernel
        params downstream).
    result : ImplicitSolveResult
        Converged fixed point.
    m_grid, chi_grid : np.ndarray
        Engine grid axes as numpy arrays (for plot / metric use).
    """
    response = build_response_for_theta(
        theta,
        n_m=n_m,
        n_chi=n_chi,
        orientation=orientation,
        retention_kind=retention_kind,
        w_over_v_esc=w_over_v_esc,
    )
    result = solve_implicit_fixed_point(
        response,
        jnp.asarray(theta["v_esc"]),
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
        damping=damping,
    )
    m_grid = np.asarray(response.grid.m)
    chi_grid = np.asarray(response.grid.chi)
    return response, result, m_grid, chi_grid


def run_mc(
    theta: dict,
    n_seed: int,
    rng: np.random.Generator,
    depletion_fraction: float,
    *,
    track_generations: bool = False,
    orientation: object | None = None,
    orientation_key_seed: int = 0,
) -> dict:
    r"""Run the particle MC at the given theta with a fixed-seed RNG.

    Returns the raw MC result dict (``m_active``, ``chi_active``,
    ``mergers``, ``stats``) — callers that need marginals on the
    engine grid pass ``m_active`` and ``chi_active`` through
    :func:`mc_marginals_on_grid`; callers that need observable
    histograms read ``mergers`` directly.

    ``orientation`` / ``orientation_key_seed`` forward to
    :class:`MCConfig` (Session C wired these in via a76fa8f).  Default
    ``orientation=None`` routes to the v1 Aligned scalar path
    bit-for-bit so existing callers are unaffected.  See
    ``[[feedback-dispatch-callgraph-audit]]``.
    """
    cfg = B1.MCConfig(
        n_seed=n_seed,
        depletion_fraction=depletion_fraction,
        orientation=orientation,
        orientation_key_seed=int(orientation_key_seed),
        **{k: theta[k] for k in (
            "mu_natal", "sigma_natal", "m_min", "m_max",
            "natal_spin_loc", "natal_spin_scale",
            "kernel_a", "kernel_b", "v_esc",
        )},
    )
    return B1.particle_mc(cfg, rng, track_generations=track_generations)


def mc_marginals_on_grid(
    m_active: np.ndarray,
    chi_active: np.ndarray,
    m_grid: np.ndarray,
    chi_grid: np.ndarray,
    theta: dict,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Evaluate the MC's KDE at the solver's grid points.

    The KDE is computed on (log m, χ) so that both axes have
    approximately uniform resolution after the geomspace stretch
    (matches the engine's internal representation).
    """
    if m_active.size < 2:
        return np.zeros_like(m_grid), np.zeros_like(chi_grid)
    dN_dm = make_marginal(
        m_active, m_grid, kind="kde",
        bounds=(theta["m_min"], theta["m_max"] * 2.5),
        log_space=True,
    )
    dN_dchi = make_marginal(
        chi_active, chi_grid, kind="kde",
        bounds=(0.0, 1.0),
        log_space=False,
    )
    return dN_dm, dN_dchi


def js_subrange(
    p: np.ndarray,
    q: np.ndarray,
    x: np.ndarray,
    x_lo: float,
    x_hi: float,
) -> float:
    r"""JS divergence on the sub-range ``x ∈ [x_lo, x_hi]``.

    Returns NaN if fewer than 4 grid points fall in the range — the
    metric is not well-defined on a degenerate axis.
    """
    mask = (x >= x_lo) & (x <= x_hi)
    if int(mask.sum()) < 4:
        return float("nan")
    return js_divergence(p[mask], q[mask], x[mask])


__all__ = [
    "build_response_for_theta",
    "run_implicit",
    "run_mc",
    "mc_marginals_on_grid",
    "js_subrange",
]
