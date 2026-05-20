"""Project the natal density ``N_natal(m, χ)`` onto a tensor-product basis.

Step 2 of ``demo/PLAN_engine_v3_basis.md``.

The v2 natal pipeline (Talbot-Thrane PL+Peak or truncated-normal in ``m``,
truncated-normal in ``χ``) lives in :mod:`hierarchical_ad_pop_engine.natal`.
This module is the bridge from that pipeline to the v3 basis
infrastructure — no edits to ``response/`` are needed.

Conventions
-----------
* The basis is on ``(log m, χ)`` (the natural axis for the engine —
  matches the v2 ``BinaryGrid.log_m`` axis and gives spectral
  convergence on power-law-tailed mass distributions).
* Coefficients ``n_α`` expand the density **on (m, χ)** (i.e.
  ``∫ N dm = total count``) viewed as a function of ``(log m, χ)``::

      N(m, χ) ≈ Σ_α n_α φ_α(log m, χ).

  No Jacobian is smuggled into the coefficients; the function we expand
  is the same density v2 stores on its grid, just sampled at LGL nodes
  in log-mass coordinates.  Downstream integrals against this density
  must include a ``dm = m · d(log m)`` Jacobian when integrating in
  log-m coordinates — exactly the same Jacobian v2 carries
  (``_dm_log_weights``).

* Negative-infinity ``log N`` values at out-of-support points are
  surrogated to ``-1e30`` (matching :meth:`CoagulationResponse._natal_log_density`)
  so the exponentiation gives a clean zero without NaN propagation.

Projection paths
----------------
* :func:`project_natal_density` (LINEAR) — **production path.**
  Projects ``N_natal`` directly.  Works for any ``natal_mass_kind``.
  At ``n_basis_m = 128`` reaches ``rel_err ≈ 3e-5`` on Stage 1 PL+Peak —
  about two orders of magnitude better than v2's bilinear-interp
  representation error (``≈ 2e-3``).
* :func:`project_natal_log_density` (LOG) — **DEPRECATED, kept for
  reference.**  Projects ``log N_natal`` with a symmetric floor (input
  clipped to ``[_LOG_N_FLOOR, ∞)``, output re-clipped before exp).
  The intended win was machine-precision recovery for natals whose
  analytic log-density is globally polynomial across the basis
  support (Gaussian, log-normal).  In practice every realistic
  inference config introduces a kink at the support boundary, and
  Gibbs ringing at the kink limits ``rel_err`` to ``~0.2`` even at
  ``n_basis_m = 128``.  See ``demo/PLAN_engine_v3_basis.md`` §Step 2
  for the empirical convergence table and the rationale for choosing
  the linear path as the production default.

  The log-projection code is retained for possible future revival
  (e.g. with a smooth soft-floor replacement, or with a basis better
  suited to compactly-supported densities such as Bernstein
  polynomials).  Callers wanting log-positivity guarantees should
  instead use the linear projection and clamp at zero downstream —
  v2's Picard solver does this already.

Both paths share :func:`natal_log_density_at` for the underlying
density evaluation.

API
---
* :func:`natal_log_density_at` — evaluate ``log N_natal(m, χ)`` at
  arbitrary points (utility, mirrors ``_natal_log_density`` for the
  arbitrary-grid case).
* :func:`project_natal_density` — LINEAR projection (production).
* :func:`evaluate_natal_density` — evaluate basis approximation from
  linear-projection coefficients (production).
* :func:`project_natal_log_density` — LOG projection. **DEPRECATED.**
* :func:`evaluate_natal_density_from_log` — evaluate from
  log-projection coefficients. **DEPRECATED.**
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array

from ..natal.mass import log_natal_mass, log_natal_mass_gaussian
from ..natal.spin import build_natal_spin, log_natal_spin

if TYPE_CHECKING:  # avoid circular import; only needed for static typing
    from ..response.coagulation import CoagulationResponse
    from .tensor import TensorProductBasis


def natal_log_density_at(
    m: Array,
    chi: Array,
    natal_mass_params: dict,
    natal_spin_params: dict,
    natal_mass_kind: str = "pl_peak",
) -> Array:
    """Evaluate ``log N_natal(m, χ)`` at arbitrary, broadcastable ``(m, χ)``.

    Mirrors :meth:`CoagulationResponse._natal_log_density` but does not
    bind to ``response.grid`` — accepts any-shape ``m`` and ``chi`` that
    broadcast against each other.  Used by :func:`project_natal_density`
    (to sample at the basis's LGL nodes) and by the Step 2 validation
    test (to sample on a dense reference grid).

    Parameters
    ----------
    m, chi
        Mass and spin query points.  Broadcast against each other; the
        result has the broadcast shape.
    natal_mass_params, natal_spin_params, natal_mass_kind
        Same dicts ``CoagulationResponse`` holds; this function does
        nothing more than dispatch on ``natal_mass_kind`` and add the
        two log-densities.
    """
    spin_dist = build_natal_spin(**natal_spin_params)
    if natal_mass_kind == "gaussian":
        log_f_m = log_natal_mass_gaussian(m, **natal_mass_params)
    else:
        log_f_m = log_natal_mass(m, **natal_mass_params)
    log_g_chi = log_natal_spin(spin_dist, chi)
    log_f_m = jnp.where(jnp.isneginf(log_f_m), -1e30, log_f_m)
    log_g_chi = jnp.where(jnp.isneginf(log_g_chi), -1e30, log_g_chi)
    return log_f_m + log_g_chi


def project_natal_density(
    response: "CoagulationResponse",
    basis: "TensorProductBasis",
) -> Array:
    """Project ``N_natal(m, χ)`` onto the basis.

    Evaluates the natal log-density at the basis's LGL meshgrid (in
    ``(log m, χ)`` coordinates; ``m = exp(log m)`` to call into the
    natal pipeline), exponentiates, then applies
    :py:meth:`TensorProductBasis.project`.

    Parameters
    ----------
    response
        ``CoagulationResponse`` carrying ``natal_mass_params``,
        ``natal_spin_params``, ``natal_mass_kind``.  Its ``grid`` is
        **not** used — projection samples at basis nodes, not grid
        nodes.
    basis
        Tensor-product basis on ``(log m, χ)``.

    Returns
    -------
    Array of shape ``(n_basis_m, n_basis_χ)``: modal coefficients
    ``n_natal_α`` such that ``N_natal(m, χ) ≈ Σ_α n_natal_α φ_α(log m, χ)``.
    """
    log_m_2d, chi_2d = basis.meshgrid_nodes()
    m_2d = jnp.exp(log_m_2d)
    log_N_at_nodes = natal_log_density_at(
        m_2d,
        chi_2d,
        natal_mass_params=response.natal_mass_params,
        natal_spin_params=response.natal_spin_params,
        natal_mass_kind=response.natal_mass_kind,
    )
    N_at_nodes = jnp.exp(log_N_at_nodes)
    return basis.project(N_at_nodes)


def evaluate_natal_density(
    coeffs: Array,
    basis: "TensorProductBasis",
    m: Array,
    chi: Array,
) -> Array:
    """Evaluate the basis approximation to ``N_natal`` at arbitrary points.

    Inverse of :func:`project_natal_density` in the L²-projection sense.
    Returns linear-scale density values (NOT log) so the result is
    directly comparable to ``exp(natal_log_density_at(m, chi))``.

    Parameters
    ----------
    coeffs : Array, shape ``(n_basis_m, n_basis_χ)``
        Output of :func:`project_natal_density`.
    basis : TensorProductBasis
        Same basis used to build ``coeffs``.
    m, chi : Array
        Physical mass and spin query points; broadcast against each
        other.

    Returns
    -------
    Array of the broadcast shape, carrying ``Σ_α n_α φ_α(log m, χ)``.
    May be slightly negative near sharp boundary transitions (Gibbs-
    style ringing); the Picard solver in Step 4 clips at zero, matching
    v2's behaviour.
    """
    log_m = jnp.log(m)
    return basis.evaluate(coeffs, log_m, chi)


# ---------------------------------------------------------------------------
# Log-space projection — only use for smooth analytic natals (Gaussian, etc).
# ---------------------------------------------------------------------------
# Density floor for the log-projection path.  exp(_LOG_N_FLOOR) is treated
# as "physically zero" — modes spent recovering density values below this
# floor are wasted.  -50 → exp ≈ 2e-22 of peak; well below any observable's
# sensitivity.
#
# The floor is applied symmetrically on BOTH ends of the projection:
#
#   1. INPUT side (in :func:`project_natal_log_density`): clip the natal
#      log-density at the LGL nodes to ``[_LOG_N_FLOOR, +∞)`` before
#      handing it to the polynomial projector.  This stops the polynomial
#      from spending its modes chasing a quadratic tail of order ``-1e4``
#      that contributes ``exp(-1e4) ≈ 0`` to every observable.  The clip
#      introduces a kink (C⁰, not C∞) where the natal log-density crosses
#      the floor; Gibbs ringing at the kink is algebraic-in-n, not stuck.
#
#   2. OUTPUT side (in :func:`evaluate_natal_density_from_log`): re-floor
#      the basis-evaluated log-density to ``[_LOG_N_FLOOR, +∞)`` before
#      exponentiation.  This eats the Gibbs ringing in the tail region —
#      where the true density is below the floor anyway — and guarantees
#      the returned density is in ``[exp(_LOG_N_FLOOR), exp(50)]``.
#
# (The upper clip at +50 stays defensive — prevents overflow if the basis
# expansion ever excursions wildly.)
#
# Empirical convergence of the symmetric-floor design
# (``outputs/probe_floored_log_projection.py``, Gaussian μ=8/σ=1):
#
#     n_basis_m   16      24      32      48      64      96      128
#     rel_err     ~2.5    ~0.7    ~1.4    ~0.4    ~0.2    ~0.2    ~0.2
#
# The kink-Gibbs overshoot is stuck — it sits adjacent to the kink and
# bleeds into the peak region, so rel_err caps at O(0.2) regardless of
# resolution.  This is why the log-projection path is deprecated; see
# the module docstring and the function-level deprecation notices.  The
# floor constant + symmetric-floor implementation are preserved as a
# starting point for a future revival (smoother floor, Bernstein basis,
# etc.).
_LOG_N_FLOOR = -50.0


def project_natal_log_density(
    response: "CoagulationResponse",
    basis: "TensorProductBasis",
) -> Array:
    """LOG-space projection of the natal density.  **DEPRECATED.**

    .. deprecated:: 2026-05-20
        Use :func:`project_natal_density` (linear) instead.  Empirical
        convergence on the symmetric-floor design caps at
        ``rel_err ≈ 0.2`` even at ``n_basis_m = 128``: the input floor
        introduces a C⁰ kink wherever the natal log-density crosses
        ``_LOG_N_FLOOR``, and Gibbs ringing at that kink bleeds into
        the peak region.  The linear path reaches ``3e-5`` at the
        same ``n_basis_m`` with no kinks, on any natal kind.  Retained
        in the codebase for possible future revival with a smoother
        floor design (softplus, or a compactly-supported basis like
        Bernstein polynomials).

    Projects ``log N_natal`` onto the basis, returning modal coefficients
    that expand the log-density::

        log N_natal(m, χ) ≈ Σ_α c_α φ_α(log m, χ).

    Implementation: input log-density at LGL nodes is clipped at
    ``_LOG_N_FLOOR`` from below before projection; this is undone
    symmetrically by :func:`evaluate_natal_density_from_log` which
    re-clips on output.  See ``_LOG_N_FLOOR`` for the design rationale
    and the failure mode that motivated the deprecation.

    Returns
    -------
    Array of shape ``(n_basis_m, n_basis_χ)``: coefficients of
    (floored) ``log N_natal``.  These expand log-density, NOT density.
    Evaluate via :func:`evaluate_natal_density_from_log`.
    """
    log_m_2d, chi_2d = basis.meshgrid_nodes()
    m_2d = jnp.exp(log_m_2d)
    log_N_at_nodes = natal_log_density_at(
        m_2d,
        chi_2d,
        natal_mass_params=response.natal_mass_params,
        natal_spin_params=response.natal_spin_params,
        natal_mass_kind=response.natal_mass_kind,
    )
    # Clip the log-density at the floor before projection — anything below
    # exp(_LOG_N_FLOOR) is physically zero, so the polynomial should not
    # waste modes resolving it.  This catches both the -1e30 surrogate
    # (out-of-support) and legitimate-but-irrelevant tail values (e.g. the
    # Gaussian quadratic at m=m_max ≈ -1e4).  See the _LOG_N_FLOOR docstring
    # above for the design rationale and trade-off.
    log_N_at_nodes = jnp.maximum(log_N_at_nodes, _LOG_N_FLOOR)
    return basis.project(log_N_at_nodes)


def evaluate_natal_density_from_log(
    log_coeffs: Array,
    basis: "TensorProductBasis",
    m: Array,
    chi: Array,
) -> Array:
    """Evaluate ``N_natal`` from log-space coefficients.  **DEPRECATED.**

    .. deprecated:: 2026-05-20
        Companion to :func:`project_natal_log_density`; see that
        function's docstring for the rationale.  Use
        :func:`evaluate_natal_density` instead.

    Inverse of :func:`project_natal_log_density`.  Evaluates the basis
    expansion of ``log N``, exponentiates, returns linear-scale density.

    Parameters
    ----------
    log_coeffs : Array, shape ``(n_basis_m, n_basis_χ)``
        Output of :func:`project_natal_log_density`.
    basis, m, chi
        Same semantics as :func:`evaluate_natal_density`.

    Returns
    -------
    Array of broadcast shape, carrying ``exp(Σ_α c_α φ_α(log m, χ))``
    after re-flooring at ``_LOG_N_FLOOR``.  Always positive.
    """
    log_m = jnp.log(m)
    log_N_eval = basis.evaluate(log_coeffs, log_m, chi)
    # Re-floor the basis evaluation symmetric with project_natal_log_density:
    # below the floor → exp(_LOG_N_FLOOR) (physically zero), eating Gibbs
    # ringing in the tail region.  Above +50 → defensive cap against
    # overflow if the expansion excursions wildly.
    return jnp.exp(jnp.clip(log_N_eval, min=_LOG_N_FLOOR, max=50.0))
