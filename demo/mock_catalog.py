r"""Multi-sample mock catalogue for the gwkokab analytical-Poisson fit.

The gwkokab ``analytical_poisson_likelihood_fn`` marginalises each event's
measurement uncertainty with an inner Monte-Carlo sum over per-event posterior
samples,

.. math::

    \log L = \sum_n \log\!\sum_s \exp\!\big[\log\rho(\omega_{n,s}) + \Delta_{n,s}\big]
             \;+\; N\log T_{\rm obs}\;-\;\mu ,

so it wants a catalogue of shape ``(n_events, n_samples, 4)`` plus matching
``ln_offsets`` ``Δ_{n,s}`` (the reference-prior log-weights).  M4 used a single
delta-sample per event (``n_samples = 1``, ``Δ = 0``); this module builds the
realistic multi-sample version.

Two pieces, deliberately split so the **scatter** (which needs no engine) is
unit-testable in the lightweight sandbox while the **truth draw** (which needs
a built ``CoagulationResponse``) runs where the engine is available:

* :func:`scatter_to_posteriors` — turn true event parameters ``(N, 4)`` into
  per-event posterior samples ``(N, S, 4)`` by a simple measurement model
  (log-normal in mass, truncated-normal in spin), preserving ``m₁ ≥ m₂`` and
  clipping to the population support box.
* :func:`make_mock_posteriors` — draw the true events from a truth response
  (grid inverse-CDF, reusing ``demo_gwkokab_fit_stage1.sample_mock_catalog``)
  and scatter them.

**``ln_offsets = 0``.**  The samples are treated as posterior draws under a
**flat** PE reference prior over the box; the reference-prior log-weight is
then a per-event constant ``Δ = −log π_ref − log S`` that is independent of the
population parameters and so shifts ``log L`` by a θ-independent constant — it
does not change the MAP or the posterior over θ.  We therefore set
``ln_offsets ≡ 0`` (the constants are absorbed, exactly as in M4's
single-sample reduction).  A non-flat PE prior would instead set
``Δ_{n,s} = −log π_ref(ω_{n,s})``; pass your own ``ln_offsets`` to the
likelihood in that case.

Hard rules: this is mock *data generation*, not a physics model — the
measurement kernel is an explicit, declared toy (Gaussian-ish PE blur), not a
stand-in for real LIGO PE.  No fudge factors enter the rate; the scatter only
shapes the per-event likelihood term.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

__all__ = ["scatter_to_posteriors", "make_mock_posteriors"]


def scatter_to_posteriors(
    true_events: Array,
    n_samples: int,
    key: Array,
    *,
    sigma_log_m: float = 0.1,
    sigma_chi: float = 0.1,
    m_lo: float = 4.0,
    m_hi: float = 200.0,
) -> Array:
    r"""``(N, 4)`` true ``(m₁,m₂,χ₁,χ₂)`` → ``(N, S, 4)`` posterior samples.

    Measurement model (a declared toy PE blur):

    * mass: multiplicative log-normal, ``m_obs = m_true·exp(σ_log_m·z)`` —
      fractional mass error ``≈ σ_log_m``;
    * spin: additive, ``χ_obs = χ_true + σ_chi·z``;

    then each sample's masses are re-sorted so ``m₁ ≥ m₂`` (carrying spins) and
    all components clipped to the support box (``m ∈ [m_lo, m_hi]``,
    ``χ ∈ [0, 1]``).  ``S = 1`` returns the (clipped) true value as a single
    sample — i.e. the M4 delta-catalogue.

    Deterministic in ``key``; engine-free (sandbox-testable).
    """
    true_events = jnp.asarray(true_events)
    n_events = true_events.shape[0]
    m1 = true_events[:, 0][:, None]
    m2 = true_events[:, 1][:, None]
    chi1 = true_events[:, 2][:, None]
    chi2 = true_events[:, 3][:, None]

    k_m1, k_m2, k_c1, k_c2 = jax.random.split(key, 4)
    shape = (n_events, n_samples)
    m1s = m1 * jnp.exp(sigma_log_m * jax.random.normal(k_m1, shape))
    m2s = m2 * jnp.exp(sigma_log_m * jax.random.normal(k_m2, shape))
    c1s = chi1 + sigma_chi * jax.random.normal(k_c1, shape)
    c2s = chi2 + sigma_chi * jax.random.normal(k_c2, shape)

    # Enforce m1 >= m2 per sample, carrying the matching spins.
    hi_is_1 = m1s >= m2s
    M1 = jnp.where(hi_is_1, m1s, m2s)
    M2 = jnp.where(hi_is_1, m2s, m1s)
    C1 = jnp.where(hi_is_1, c1s, c2s)
    C2 = jnp.where(hi_is_1, c2s, c1s)

    M1 = jnp.clip(M1, m_lo, m_hi)
    M2 = jnp.clip(M2, m_lo, m_hi)
    C1 = jnp.clip(C1, 0.0, 1.0)
    C2 = jnp.clip(C2, 0.0, 1.0)
    return jnp.stack([M1, M2, C1, C2], axis=-1)  # (N, S, 4)


def make_mock_posteriors(
    truth,
    n_events: int,
    n_samples: int,
    key: Array,
    *,
    sigma_log_m: float = 0.1,
    sigma_chi: float = 0.1,
    m_lo: float = 4.0,
    m_hi: float = 200.0,
    return_truth_points: bool = False,
) -> Tuple[Array, Array]:
    r"""Mock catalogue ``(events (N,S,4), ln_offsets (N,S))`` from a truth model.

    Draws the ``N`` true events from ``truth`` via the demo's grid inverse-CDF
    sampler (``sample_mock_catalog``) — so ``truth`` must be a grid/implicit
    :class:`CoagulationResponse` (the data-generating oracle, same convention as
    the M4 integration test) — then scatters them with
    :func:`scatter_to_posteriors`.

    ``ln_offsets`` is all-zeros (flat PE reference prior; see module docstring).

    Returns ``(events, ln_offsets)``; if ``return_truth_points`` also the
    underlying ``(N, 4)`` true values (handy for plotting / diagnostics).
    """
    # Imported lazily so this module imports under jax alone; the demo import
    # pulls the engine.
    import demo_gwkokab_fit_stage1 as demo  # type: ignore

    k_true, k_scatter = jax.random.split(key, 2)
    true_events = jnp.asarray(demo.sample_mock_catalog(truth, n_events, k_true))
    events = scatter_to_posteriors(
        true_events, n_samples, k_scatter,
        sigma_log_m=sigma_log_m, sigma_chi=sigma_chi, m_lo=m_lo, m_hi=m_hi,
    )
    ln_offsets = jnp.zeros((n_events, n_samples))
    if return_truth_points:
        return events, ln_offsets, true_events  # type: ignore[return-value]
    return events, ln_offsets
