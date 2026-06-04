r"""Sampled-mode (likelihood-mode) gwkokab wiring — PLAN item 5 thread-back.

Closes the loop of ``demo/PLAN_gwkokab_sampled_likelihood.md``: the gwkokab
fork branch ``feature/sampled-poisson-likelihood`` provides
``sampled_poisson_likelihood_fn`` + ``numpyro_sampled_poisson_likelihood``
(model as weighted **draws**, events as per-event **likelihood evaluators**);
this module supplies the engine side —

1. :func:`make_coagulation_sampler_fn` — exposes the JAX particle MC
   (``particle_mc_jax`` / ``particle_mc_jax_soft``) actives + weights as the
   ``sampler_fn(**θ) -> (samples, log_w)`` the fork's wrappers consume.  This
   is the production promotion of the toy's
   ``demo/inference/toy_single_pulse_sampled_likelihood.py::make_sampler_fn``.
2. :func:`load_rift_event_log_likelihoods` /
   :func:`gaussian_event_log_likelihoods_from_posteriors` — per-event
   likelihood loading: RIFT marginal-likelihood interpolants in production,
   genie-style Gaussian fits to posterior clouds as the test/fallback path.
3. :func:`build_sampled_likelihood` — the one-call wiring analog of
   ``stage1_wiring.build_stage1_likelihood`` (same Stage-1 six free
   parameters, same priors, same ``sorted``-order ``variables_index``
   contract), returning the numpyro model body ready for NUTS/SVI.

This is the **Results-C capable path**: MC as the *recoverer* — exact draws
at any cascade depth, no grid (no ``n_chi`` ringing against the empty
natal–1g gap), no KDE.  The implicit solver remains the backstop
(``[[feedback-validation-downselect-two-algorithms]]``).

Conventions (inherited from the toy, see ``demo/inference/README.md``):

* Model samples are merger events ``(m1, m2, chi1, chi2)`` (heavier-first) —
  columns 0–3 of the MC's 8-column ``mergers`` schema.
* ``log_w = log(w_mergers·step_active) − logsumexp(·) + log_rate`` so that
  ``Σ_j w_j = e^{log_rate}`` = total merger rate per unit observing time.
  ``log_rate`` is then cleanly identifiable through the Poisson mean, and the
  open MC count-normalisation thread (``[[project-mc-jax-total-rate-bias]]``)
  is deliberately sidestepped — revisit the normalisation when that closes.
* The sampler key is FIXED inside ``sampler_fn`` (deterministic objective —
  same convention as ``make_coagulation_dist_fn``).  Validate gradients as
  finite + nonzero or many-key FD-mean, never single-key FD
  (``[[feedback-single-key-fd-not-ad-reference]]``).
* Dead merger slots carry NaN coordinates (fixed-shape scan padding); they are
  replaced by a benign in-domain point under ``jnp.where`` (kills NaN in both
  passes) and their ~0 weights are **floored, not where-guarded**
  (``[[feedback-where-guards-forward-not-backward]]``).

Per-event ESS is the survival metric of this whole approach (broad BBH
posteriors are forgiving; narrow/low-mass posteriors starve) — every
deliverable here threads the fork's ``ess_per_event`` diagnostic through.
Stratified natal proposals (M3a/M3b machinery) are the lever when it
collapses.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array


__all__ = [
    "make_coagulation_sampler_fn",
    "gaussian_event_log_likelihoods_from_posteriors",
    "load_rift_event_log_likelihoods",
    "build_sampled_likelihood",
]


# The Stage-1 free-parameter names, in the order sampler_fn receives them.
SAMPLER_FREE_PARAMS = (
    "mu_natal",
    "sigma_natal",
    "natal_spin_loc",
    "natal_spin_scale",
    "kernel_b",
    "log_rate",
)

# Benign in-domain stand-in for dead (NaN-padded) merger slots.  Any finite
# point works — the slot weight is floored to ~0 so it never contributes;
# the substitution exists purely to keep NaN out of fwd/bwd passes.
_BENIGN_MERGER = (10.0, 8.0, 0.0, 0.0)


def make_coagulation_sampler_fn(
    *,
    v_esc: float,
    m_min: float,
    m_max: float,
    kernel_a: float = 0.0,
    n_seed: int = 2000,
    depletion_fraction: float = 0.10,
    gate: str = "soft",
    n_steps_mult: float = 3.0,
    k_ratio: float = 1.5,
    mc_jax_key: Any = None,
    weight_floor: float = 1e-300,
) -> Callable[..., Tuple[Array, Array]]:
    r"""Expose the JAX particle MC as a gwkokab sample-mode ``sampler_fn``.

    Mirrors :func:`~hierarchical_ad_pop_engine.inference.gwkokab_bridge.make_coagulation_dist_fn`'s
    constant/free split: everything passed here is a ``constant_param`` of the
    fit; the returned closure takes the six Stage-1 free parameters
    (:data:`SAMPLER_FREE_PARAMS`) as keywords and returns
    ``(samples (n, 4), log_w (n,))`` with ``Σ_j w_j = e^{log_rate}``.

    Wire into the gwkokab fork::

        sampler_fn = make_coagulation_sampler_fn(v_esc=..., m_min=..., m_max=...)
        log_lik_fn = numpyro_sampled_poisson_likelihood(
            sampler_fn=sampler_fn,
            priors=..., variables=..., constant_params={},
            variables_index=...,
            event_log_likelihood_fns=...,   # RIFT interpolants / Gaussian fits
            pdet_fn=None,                   # flat selection (Stage-1)
            variance_cut_threshold=None,
        )

    Parameters
    ----------
    gate : {"soft", "bernoulli"}, default "soft"
        ``"soft"`` (B.2 deterministic occupancy-flow gate) is the default
        here — the recommended gate for gradient-based fits (no
        straight-through bias, no ``k_ratio`` precision knob).
        ``"bernoulli"`` routes to the B.1 path (``k_ratio`` applies).
    mc_jax_key : jax PRNG key, optional
        Fixed cascade key (default ``PRNGKey(0)``).  Draw catalogues with an
        INDEPENDENT key.
    weight_floor : float
        Floor applied to effective weights before ``log`` — keeps the
        backward pass finite without a ratio rescale.

    Notes
    -----
    AD: gradients flow through the weights (and, under ``"soft"``, the
    occupancy) to all six free parameters — the J1–J4 + B.2 chain.  The same
    variance caveats apply as documented in the fork's TP2 gate.
    """
    from red_team.backup_particle_mc import (  # lazy heavy import
        MCConfig,
        particle_mc_jax,
        particle_mc_jax_soft,
    )

    if gate not in ("soft", "bernoulli"):
        raise ValueError(f"gate must be 'soft' or 'bernoulli', got {gate!r}")
    key = jax.random.PRNGKey(0) if mc_jax_key is None else mc_jax_key
    benign = jnp.asarray(_BENIGN_MERGER)

    def sampler_fn(
        mu_natal: Array,
        sigma_natal: Array,
        natal_spin_loc: Array,
        natal_spin_scale: Array,
        kernel_b: Array,
        log_rate: Array,
    ) -> Tuple[Array, Array]:
        cfg = MCConfig(
            mu_natal=jnp.asarray(mu_natal),
            sigma_natal=jnp.asarray(sigma_natal),
            m_min=jnp.asarray(m_min),
            m_max=jnp.asarray(m_max),
            natal_spin_loc=jnp.asarray(natal_spin_loc),
            natal_spin_scale=jnp.asarray(natal_spin_scale),
            kernel_a=jnp.asarray(kernel_a),
            kernel_b=jnp.asarray(kernel_b),
            v_esc=jnp.asarray(v_esc),
            n_seed=int(n_seed),
            depletion_fraction=float(depletion_fraction),
        )
        if gate == "soft":
            out = particle_mc_jax_soft(cfg, key, n_steps_mult=n_steps_mult)
        else:
            out = particle_mc_jax(cfg, key, k_ratio=k_ratio)

        samples = out["mergers"][:, :4]  # (n, 4): m1 m2 chi1 chi2
        w_eff = out["w_mergers"] * out["step_active"]
        # NaN-slot hygiene + floored (not where-guarded) log-weights — see
        # module docstring for the two feedback rules this encodes.
        samples = jnp.where(jnp.isfinite(samples), samples, benign[None, :])
        log_w = jnp.log(jnp.maximum(w_eff, weight_floor))
        log_w = log_w - jax.scipy.special.logsumexp(log_w) + log_rate
        return samples, log_w

    return sampler_fn


def gaussian_event_log_likelihoods_from_posteriors(
    posteriors: Array,
    *,
    jitter: float = 1e-10,
) -> tuple:
    r"""Genie-style fallback: per-event Gaussian likelihood fits to PE clouds.

    Fits a mean + full covariance to each event's posterior samples and wraps
    them in the fork's :class:`GaussianEventLikelihood`.  This is the same
    construction validated end-to-end in
    ``internal_tests/gw_banana_sampled_vs_density.py`` (where the fits are
    exact by construction) and the test/CI stand-in for RIFT interpolants.

    Parameters
    ----------
    posteriors : Array
        ``(n_events, n_pe, n_dim)`` per-event posterior samples in the model
        coordinates ``(m1, m2, chi1, chi2)`` (or any subset the sampler
        emits).  NOTE: a Gaussian fit to a *posterior* equals the event
        likelihood only under a flat PE prior — divide out non-flat priors
        upstream.
    jitter : float
        Diagonal regulariser for near-degenerate clouds.

    Returns
    -------
    tuple of per-event evaluators ``L_i: (n, n_dim) -> (n,)``.
    """
    from gwkokab.inference.event_likelihoods import (  # lazy fork import
        gaussian_event_log_likelihoods,
    )

    posteriors = jnp.asarray(posteriors)
    n_events, _, n_dim = posteriors.shape
    means = posteriors.mean(axis=1)  # (n_ev, n_dim)
    centred = posteriors - means[:, None, :]
    covs = (
        jnp.einsum("eki,ekj->eij", centred, centred) / (posteriors.shape[1] - 1)
        + jitter * jnp.eye(n_dim)[None]
    )
    return gaussian_event_log_likelihoods(means, covs)


def load_rift_event_log_likelihoods(
    sources: Sequence[Any],
    *,
    builder: Optional[Callable[[Any], Callable[[Array], Array]]] = None,
    log_input: bool = True,
) -> tuple:
    r"""Wrap per-event RIFT marginal-likelihood interpolants for the fork.

    The production loading path of PLAN item 5: RIFT produces, per event, a
    fit/interpolant of the marginal likelihood over the binary parameters.
    Each becomes an ``L_i`` evaluator via the fork's
    :class:`RIFTMarginalLikelihood` adapter.

    Parameters
    ----------
    sources : sequence
        One entry per event — either an already-built interpolant **callable**
        ``(n, n_dim) -> (n,)``, or an opaque handle (e.g. a path to a RIFT
        ``*.composite`` / fit file) to be passed through ``builder``.
    builder : callable, optional
        ``builder(handle) -> interpolant`` — the RIFT-side constructor
        (requires the RIFT stack; supply it on the workstation).  Required if
        any source is not already callable.
    log_input : bool
        Whether the interpolants return log-likelihood (RIFT convention) —
        forwarded to the adapter.

    Returns
    -------
    tuple of :class:`gwkokab.inference.event_likelihoods.RIFTMarginalLikelihood`
    """
    from gwkokab.inference.event_likelihoods import (  # lazy fork import
        RIFTMarginalLikelihood,
    )

    evaluators = []
    for i, src in enumerate(sources):
        if callable(src):
            interp = src
        elif builder is not None:
            interp = builder(src)
        else:
            raise TypeError(
                f"event {i}: source {type(src).__name__!r} is not callable and "
                "no builder was supplied — pass the RIFT-side "
                "builder(handle) -> interpolant constructor."
            )
        evaluators.append(RIFTMarginalLikelihood(interp, log_input=log_input))
    return tuple(evaluators)


def build_sampled_likelihood(
    *,
    event_log_likelihood_fns: Sequence[Callable[[Array], Array]],
    v_esc: float,
    m_min: float,
    m_max: float,
    kernel_a: float = 0.0,
    n_seed: int = 2000,
    depletion_fraction: float = 0.10,
    gate: str = "soft",
    n_steps_mult: float = 3.0,
    k_ratio: float = 1.5,
    mc_jax_key: Any = None,
    pdet_fn: Optional[Callable[[Array], Array]] = None,
    prior_factories: Optional[Dict[str, Callable[[], Any]]] = None,
    variance_cut_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    r"""One-call sampled-mode wiring — the dual of ``build_stage1_likelihood``.

    Same six Stage-1 free parameters, same priors, same ``sorted``-order
    ``variables_index`` contract; the only swaps are ``dist_fn → sampler_fn``
    and (PE samples + ``ln_offsets``) → per-event likelihood evaluators, per
    the fork's API.  Flat selection (``pdet_fn=None``) is the Stage-1
    convention; pass an SNR-surrogate ``pdet_fn`` for Stage 2.

    Returns
    -------
    dict with keys ``sampler_fn``, ``variables``, ``variables_index``,
    ``priors``, ``log_likelihood_fn``.  The numpyro model body
    (``log_likelihood_fn``) takes a single argument ``T_obs``.

    Lazily pulls gwkokab (priors + the fork's sampled wrapper); in a
    lightweight sandbox compose :func:`make_coagulation_sampler_fn` +
    ``sampled_poisson_likelihood_fn`` directly.
    """
    from gwkokab.inference.numpyro_sampled_poisson_likelihood import (
        numpyro_sampled_poisson_likelihood,
    )

    from .stage1_wiring import build_stage1_priors, build_stage1_variables

    sampler_fn = make_coagulation_sampler_fn(
        v_esc=v_esc,
        m_min=m_min,
        m_max=m_max,
        kernel_a=kernel_a,
        n_seed=n_seed,
        depletion_fraction=depletion_fraction,
        gate=gate,
        n_steps_mult=n_steps_mult,
        k_ratio=k_ratio,
        mc_jax_key=mc_jax_key,
    )
    variables, variables_index = build_stage1_variables(prior_factories)
    priors = build_stage1_priors(variables)

    log_likelihood_fn = numpyro_sampled_poisson_likelihood(
        sampler_fn,
        priors,
        variables,
        {},  # constant_params — all constants are baked into sampler_fn
        variables_index,
        tuple(event_log_likelihood_fns),
        pdet_fn,
        variance_cut_threshold,
    )
    return {
        "sampler_fn": sampler_fn,
        "variables": variables,
        "variables_index": variables_index,
        "priors": priors,
        "log_likelihood_fn": log_likelihood_fn,
    }
