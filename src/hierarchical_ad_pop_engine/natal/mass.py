"""Natal black-hole mass function — Talbot-Thrane smoothed PL + Gaussian Peak.

Built directly from gwkokab's primitive functions
(``log_planck_taper_window``, ``doubly_truncated_power_law_log_norm_constant``)
rather than the ``SmoothedTwoComponentPrimaryMassRatio`` Distribution class.
The class's eager ``_logZ`` precomputation in ``__init__`` produces NaN
gradients via ``0 · ∞`` contamination at the singular Planck-taper boundary
``m = m_min``; we cannot defuse that from outside.

The physics is identical — same gwkokab primitives, same Talbot-Thrane
assembly — just with full autodiff control:

.. math::

    f_{\\rm IMF}(m)
    \\propto \\mathrm{taper}\\bigl(\\tfrac{m - m_{\\min}}{\\delta_m}\\bigr) \\;
            \\bigl[\\,
              (1 - \\lambda_{\\rm peak}) \\, p_{\\rm PL}(m; \\alpha, m_{\\min}, m_{\\max})
              + \\lambda_{\\rm peak} \\, \\mathcal{N}(m; \\mu_{\\rm peak}, \\sigma_{\\rm peak})
            \\,\\bigr]

with the normalisation ``Z`` computed by trapezoidal quadrature on
``[m_min + 0.05·δ_m, m_max]`` to skip the Planck-taper singular boundary
(< e⁻²⁰ density contribution lost, gradient stays clean).
"""

from __future__ import annotations

import jax.numpy as jnp
from gwkokab.models.utils import doubly_truncated_power_law_log_norm_constant
from gwkokab.utils.kernel import log_planck_taper_window
from jax import Array
from jax.scipy.stats import norm, truncnorm
from jaxtyping import ArrayLike


def _log_natal_mass_unnorm(
    m: Array,
    alpha: ArrayLike,
    delta_m: ArrayLike,
    lambda_peak: ArrayLike,
    mu_peak: ArrayLike,
    sigma_peak: ArrayLike,
    m_min: ArrayLike,
    m_max: ArrayLike,
) -> Array:
    """Unnormalised log f_IMF(m) — Talbot-Thrane assembly."""
    # Power-law component, with log-norm via gwkokab's AD-safe primitive (custom JVP).
    log_Z_pl = doubly_truncated_power_law_log_norm_constant(-alpha, m_min, m_max)
    log_pl = -alpha * jnp.log(m) - log_Z_pl
    # Gaussian peak component.
    log_peak = norm.logpdf(m, loc=mu_peak, scale=sigma_peak)
    # Weighted log-sum-exp combination.
    log_pl_w = jnp.log1p(-lambda_peak) + log_pl
    log_peak_w = jnp.log(lambda_peak) + log_peak
    log_combined = jnp.logaddexp(log_pl_w, log_peak_w)
    # Low-mass Planck-taper smoothing.
    x = (m - m_min) / delta_m
    log_taper = log_planck_taper_window(x)
    return log_combined + log_taper


def log_natal_mass(
    m: Array,
    alpha: ArrayLike,
    delta_m: ArrayLike,
    lambda_peak: ArrayLike,
    mu_peak: ArrayLike,
    sigma_peak: ArrayLike,
    m_min: ArrayLike,
    m_max: ArrayLike,
) -> Array:
    """AD-safe normalised log f_IMF(m).

    Parameters
    ----------
    m
        Mass query points (solar masses), any shape.
    alpha
        Power-law slope (positive convention; density goes as ``m^{-α}``).
    delta_m
        Planck-taper smoothing length above ``m_min``.
    lambda_peak
        Fractional weight of the Gaussian peak component.
    mu_peak, sigma_peak
        Peak location / width (solar masses).
    m_min, m_max
        Hard truncation limits (solar masses).

    Returns
    -------
    Array
        ``log f_IMF(m)``, normalised so ``∫ exp(log f) dm = 1`` over
        ``[m_min, m_max]``. Returns ``-inf`` outside the support, with
        clean (zero) gradients there.
    """
    # AD-safe normalisation: skip the bottom 5 % of the smoothing region
    # to avoid the Planck-taper singular boundary at x = 0.
    eps_lo = m_min + 0.05 * delta_m
    m_quad = jnp.linspace(eps_lo, m_max, 4096)
    log_p_quad = _log_natal_mass_unnorm(
        m_quad, alpha, delta_m, lambda_peak, mu_peak, sigma_peak, m_min, m_max
    )
    Z = jnp.trapezoid(jnp.exp(log_p_quad), m_quad)
    log_Z = jnp.log(Z)

    # Query path: substitute safe interior m for out-of-support points so
    # the singular branch is never differentiated.
    inside = (m > m_min) & (m < m_max)
    m_safe_interior = jnp.sqrt(m_min * m_max)
    m_safe = jnp.where(inside, m, m_safe_interior)
    log_unnorm = _log_natal_mass_unnorm(
        m_safe, alpha, delta_m, lambda_peak, mu_peak, sigma_peak, m_min, m_max
    )
    log_prob = log_unnorm - log_Z
    return jnp.where(inside, log_prob, -jnp.inf)


def log_natal_mass_gaussian(
    m: Array,
    mu: ArrayLike,
    sigma: ArrayLike,
    m_min: ArrayLike,
    m_max: ArrayLike,
) -> Array:
    """Truncated-normal natal IMF.

    Pedagogically useful: a narrow Gaussian gives an impulse-response view of
    the coagulation operator (peaks at successive generations land at the
    expected ``M_rem(m, m)`` locations).

    Uses ``jax.scipy.stats.truncnorm.logpdf`` — the same primitive gwkokab's
    own ``GaussianPrimaryMassRatio`` calls (see
    ``gwkokab.models.mass._models.GaussianPrimaryMassRatio.log_prob``). The
    result is a fully normalised, AD-safe truncated Gaussian on
    ``[m_min, m_max]``.

    Parameters
    ----------
    m
        Mass query points (solar masses).
    mu, sigma
        Mean and standard deviation of the underlying (untruncated) Gaussian.
    m_min, m_max
        Hard truncation limits.

    Returns
    -------
    Array
        ``log f_IMF(m)``, normalised so ``∫ exp(log f) dm = 1`` over
        ``[m_min, m_max]``.
    """
    a = (m_min - mu) / sigma
    b = (m_max - mu) / sigma
    return truncnorm.logpdf(m, a=a, b=b, loc=mu, scale=sigma)


def log_natal_mass_two_gaussian(
    m: Array,
    mu_1: ArrayLike,
    sigma_1: ArrayLike,
    mu_2: ArrayLike,
    sigma_2: ArrayLike,
    lambda_2: ArrayLike,
    m_min: ArrayLike,
    m_max: ArrayLike,
) -> Array:
    r"""Two-pulse (two-Gaussian) natal IMF — a normalised mixture of truncnorms.

    .. math::

        f_{\rm IMF}(m) = (1 - \lambda_2)\,\mathrm{TruncN}(m; \mu_1, \sigma_1)
                       + \lambda_2\,\mathrm{TruncN}(m; \mu_2, \sigma_2)

    Each component is a truncated normal on ``[m_min, m_max]`` (so each is
    independently normalised over the support), hence the convex
    combination is itself normalised: ``∫ exp(log f) dm = 1``.  This is the
    ``natal_family="two_gaussian"`` model of the two-pulse coagulation fit
    (``INSTRUCTIONS_gwkokab_stage1.md`` §5.1) — e.g. the methods-paper
    "inject at 10 & 30 M⊙" small-spin recovery experiment.

    Built from the same ``jax.scipy.stats.truncnorm`` primitive as
    :func:`log_natal_mass_gaussian`, so it is fully normalised and AD-safe;
    differentiable in ``mu_1, sigma_1, mu_2, sigma_2`` and (smoothly, via
    ``log1p(-lambda_2)`` / ``log(lambda_2)``) in the mixing weight
    ``lambda_2``.

    Parameters
    ----------
    m
        Mass query points (solar masses).
    mu_1, sigma_1
        Mean / std of the first (lower-mass-by-convention) pulse.
    mu_2, sigma_2
        Mean / std of the second pulse.  The caller owns the label-switch
        convention (``mu_1 < mu_2``); this function enforces no ordering.
    lambda_2
        Fractional weight of the **second** pulse, ``∈ (0, 1)``.
    m_min, m_max
        Hard truncation limits shared by both components.

    Returns
    -------
    Array
        ``log f_IMF(m)``, normalised over ``[m_min, m_max]``.
    """
    mu_1 = jnp.asarray(mu_1)
    sigma_1 = jnp.asarray(sigma_1)
    mu_2 = jnp.asarray(mu_2)
    sigma_2 = jnp.asarray(sigma_2)
    lambda_2 = jnp.asarray(lambda_2)
    a1 = (m_min - mu_1) / sigma_1
    b1 = (m_max - mu_1) / sigma_1
    a2 = (m_min - mu_2) / sigma_2
    b2 = (m_max - mu_2) / sigma_2
    log_c1 = truncnorm.logpdf(m, a=a1, b=b1, loc=mu_1, scale=sigma_1)
    log_c2 = truncnorm.logpdf(m, a=a2, b=b2, loc=mu_2, scale=sigma_2)
    log_w1 = jnp.log1p(-lambda_2) + log_c1
    log_w2 = jnp.log(lambda_2) + log_c2
    return jnp.logaddexp(log_w1, log_w2)


# Kept for backward compat with existing callers that pass a params dict;
# now just returns the dict so callers can splat it into log_natal_mass.
def build_natal_mass(
    alpha: ArrayLike,
    delta_m: ArrayLike,
    lambda_peak: ArrayLike,
    mu_peak: ArrayLike,
    sigma_peak: ArrayLike,
    m_min: ArrayLike,
    m_max: ArrayLike,
    *,
    validate_args=None,
) -> dict:
    """Return the natal-mass parameter dict (no Distribution object needed).

    The historical signature returned a ``SmoothedTwoComponentPrimaryMassRatio``
    instance; we no longer use that wrapper because its eager ``_logZ``
    poisons autodiff. Use :func:`log_natal_mass` directly with these params.
    """
    return dict(
        alpha=jnp.asarray(alpha),
        delta_m=jnp.asarray(delta_m),
        lambda_peak=jnp.asarray(lambda_peak),
        mu_peak=jnp.asarray(mu_peak),
        sigma_peak=jnp.asarray(sigma_peak),
        m_min=jnp.asarray(m_min),
        m_max=jnp.asarray(m_max),
    )
