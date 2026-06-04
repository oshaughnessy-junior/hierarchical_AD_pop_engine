r"""gwkokab bridge — expose the coagulation engine as a gwkokab population model.

gwkokab's analytical-Poisson likelihood
(:func:`gwkokab.inference.numpyro_analytical_poisson_likelihood`) consumes a
population model as **two** pieces:

1. a ``numpyro`` :class:`~numpyro.distributions.Distribution` whose
   ``log_prob(x)`` returns the **log rate-density (intensity)** at a binary
   parameter point ``x = (m1, m2, χ1, χ2)`` — NOT a normalised pdf.  This is
   exactly the gwkokab ``ScaledMixture`` convention
   (``log_prob = logsumexp(log_scales + component_log_probs)`` = log of the
   rate-scaled intensity) and exactly what the engine's
   :meth:`CoagulationResponse.log_rate_density` already returns
   (``log[½ R_env K(m1,m2) N(m1,χ1) N(m2,χ2)]``).
2. a ``poisson_mean_estimator(model, **pmean_kwargs) -> (μ, var)`` giving the
   expected detected count ``μ = ∫ VT(x)·intensity(x) dx`` (and its variance).

The analytical likelihood then forms (see
``gwkokab.inference.poissonlikelihood_utils.analytical_poisson_likelihood_fn``)

    log L = Σ_n logsumexp_k[ log_prob(ω_{n,k}) + ln_offset_{n,k} ]
            + n_events·log(T_obs) − μ

— i.e. the inhomogeneous-Poisson population likelihood, where the inner sum
over ``k`` is the Monte-Carlo marginalisation over each event's posterior
samples reweighted by the reference prior (``ln_offsets``).

This bridge supplies both pieces around a :class:`CoagulationResponse` built
with ``solver="mc_jax"`` (AD-traceable end-to-end, gate ∈ {bernoulli, soft}).
With a single posterior sample per event and ``ln_offsets = 0``, ``T_obs = 1``
it reduces to the hand-rolled Stage-1 likelihood the M4 integration test uses
(``Σ log_rate_density(event) − total_rate``) — so this is a faithful,
framework-standard generalisation, not a new estimator.

**Selection.**  This first increment uses **flat selection** (``VT ≡ 1`` —
the detected-rate convention of ``demo/PLAN_gwkokab_fit.md`` Stage 1, with the
detection efficiency folded into ``log_R_env``).  A ``pdet(x)``-weighted
``poisson_mean_estimator`` (Stage-2 SNR surrogate) is a follow-up — pass a
custom estimator to :func:`numpyro_analytical_poisson_likelihood`.

Hard rules honoured:
* ``[[feedback-no-fake-physics-stand-ins]]`` — no new physics; this only
  re-routes the engine's existing intensity + rate through gwkokab's API.
* ``[[feedback-perturbative-unreliable-implicit-production]]`` — additive; a
  new ``inference/`` subpackage, nothing in ``response/`` changes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from numpyro.distributions import Distribution, constraints

# NOTE: ``build_gaussian_coagulation`` is imported LAZILY inside
# :func:`make_coagulation_dist_fn` rather than at module load.  The engine's
# ``response`` package transitively imports ``gwkokab`` (via
# ``natal.mass``), which pulls gwkokab's full dependency tree (flowMC, RIFT,
# …).  Keeping the import lazy lets this module — and the pure-numpyro
# :class:`CoagulationPopulation` / :func:`coagulation_poisson_mean_estimator`
# — be imported and unit-tested with only jax + numpyro present (the
# in-sandbox fast-loop), deferring the heavy engine+gwkokab import to the
# moment a response is actually built.


__all__ = [
    "CoagulationPopulation",
    "coagulation_poisson_mean_estimator",
    "make_coagulation_dist_fn",
]


class CoagulationPopulation(Distribution):
    r"""numpyro ``Distribution`` adapter over a built :class:`CoagulationResponse`.

    ``event_shape = (4,)`` with the event ordered ``(m1, m2, χ1, χ2)``.
    ``log_prob`` returns the **log rate-density / intensity** (gwkokab
    ``ScaledMixture`` convention), delegating to
    :meth:`CoagulationResponse.log_rate_density`.

    Built fresh inside the numpyro model body each likelihood call (via
    :func:`make_coagulation_dist_fn`), so it is consumed within the same
    trace and does not need bespoke pytree registration — the AD path runs
    through the closed-over ``response``'s dynamic fields.

    Parameters
    ----------
    response : CoagulationResponse
        A response built with a solver whose ``log_rate_density`` is
        AD-traceable (``"mc_jax"`` for SVI/NUTS).
    env : dict
        Passed straight to ``log_rate_density`` (``{"v_esc": ...}``; the
        mc / mc_jax paths baked ``v_esc`` at construction time and ignore it).
    total_rate_fn : callable
        ``total_rate_fn(**pmean_kwargs) -> Array`` giving the (unselected)
        expected count ``∫ intensity dx``.  Supplied by the factory because
        the accessor differs per solver (``response.mc_jax_response.total_rate``
        for ``mc_jax``).
    m_min, m_max : float
        Box-support bounds.  Masses are allowed up to ``m_max_factor·m_max``
        (remnants exceed the natal cap); χ ∈ [0, 1].
    m_max_factor : float, default 2.5
        Head-room above ``m_max`` for retained hierarchical remnants.
    """

    arg_constraints: Dict[str, Any] = {}
    # Not a sampling distribution; reparametrized=False is irrelevant here.

    def __init__(
        self,
        response: Any,
        env: Dict[str, Any],
        total_rate_fn: Callable[..., Array],
        *,
        m_min: float,
        m_max: float,
        m_max_factor: float = 2.5,
        validate_args: Optional[bool] = None,
    ):
        self._response = response
        self._env = env
        self._total_rate_fn = total_rate_fn
        hi_m = float(m_max) * float(m_max_factor)
        self._support = constraints.independent(
            constraints.interval(
                jnp.asarray([float(m_min), float(m_min), 0.0, 0.0]),
                jnp.asarray([hi_m, hi_m, 1.0, 1.0]),
            ),
            reinterpreted_batch_ndims=1,
        )
        super().__init__(
            batch_shape=(),
            event_shape=(4,),
            validate_args=validate_args,
        )

    @property
    def response(self):  # noqa: D401
        """The underlying :class:`CoagulationResponse` (read-only accessor).

        Exposes the engine response built by ``make_coagulation_dist_fn`` so
        callers can inspect its configuration — e.g. ``pop.response.orientation``
        to confirm the Fisher-κ / Isotropic model threaded through, or
        ``pop.response.grid.chi`` to read the (clustered) χ nodes — without
        reaching into the private field.
        """
        return self._response

    @constraints.dependent_property(is_discrete=False, event_dim=1)
    def support(self):  # noqa: D401
        return self._support

    def log_prob(self, value: Array) -> Array:
        r"""``log[½ R_env K(m1,m2) N(m1,χ1) N(m2,χ2)]`` at ``value=(...,4)``.

        Broadcasts over any leading axes (gwkokab vmaps this over the event
        axis; the engine vectorises over the per-event sample axis).
        """
        value = jnp.asarray(value)
        m1 = value[..., 0]
        m2 = value[..., 1]
        chi1 = value[..., 2]
        chi2 = value[..., 3]
        return self._response.log_rate_density(m1, m2, chi1, chi2, self._env)

    def total_rate(self, **pmean_kwargs: Any) -> Array:
        r"""Unselected expected count ``∫ intensity dx`` (delegates to factory fn)."""
        return self._total_rate_fn(**pmean_kwargs)

    def sample(self, key, sample_shape=()):  # pragma: no cover - not needed
        raise NotImplementedError(
            "CoagulationPopulation is an intensity model for the analytical "
            "Poisson likelihood; it does not implement .sample.  Generate mock "
            "catalogues via the demo's grid inverse-CDF sampler instead."
        )


def coagulation_poisson_mean_estimator(
    model: CoagulationPopulation,
    *,
    T_obs: float = 1.0,
    n_quad: int = 64,
    **_ignored: Any,
) -> Tuple[Array, Array]:
    r"""``(μ, var)`` for gwkokab's analytical Poisson likelihood.

    ``μ = T_obs · ∫ VT(x) intensity(x) dx`` with **flat selection** ``VT ≡ 1``
    (detected-rate convention; detection efficiency folded into
    ``log_R_env``).  Variance is returned as ``0`` — the MC ``total_rate`` has
    finite-sample noise, but the analytical-likelihood ``variance_cut_threshold``
    is ``None`` in the flat-selection setup, so the variance term is unused.
    (Estimating the MC ``total_rate`` variance for a variance taper is a
    follow-up; see ``[[project-mc-jax-total-rate-bias]]`` — the metric itself
    reads low at small ``n_seed``, so do NOT enable a variance cut until that
    is fixed.)

    The signature matches gwkokab's ``poisson_mean_estimator`` contract
    (``Callable[[ScaledMixture, ...], tuple[Array, Array]]``); ``model`` is
    duck-typed (any object exposing ``.total_rate(**kw)``).
    """
    mu = jnp.asarray(T_obs) * model.total_rate(n_quad=n_quad)
    var = jnp.zeros_like(mu)
    return mu, var


def make_coagulation_dist_fn(
    *,
    v_esc: float,
    m_min: float,
    m_max: float,
    kernel_a: float = 0.0,
    k_max: int = 4,
    n_seed: int = 2000,
    depletion_fraction: float = 0.10,
    gate: str = "soft",
    n_steps_mult: float = 3.0,
    mc_jax_key: Any = None,
    grid: Any = None,
    m_max_factor: float = 2.5,
    natal_family: str = "gaussian",
    float_kernel_a: bool = False,
    kde_adaptive: bool = False,
    solver: str = "mc_jax",
    epsilon: float = 1.0,
    orientation: Any = None,
    retention_kind: str = "sigmoid",
    w_over_v_esc: float = 1.0,
) -> Callable[..., CoagulationPopulation]:
    r"""Build the ``dist_fn`` gwkokab calls as ``dist_fn(**params) -> Distribution``.

    Returns a closure taking the **free** population parameters as keyword
    arguments — ``mu_natal, sigma_natal, natal_spin_loc, natal_spin_scale,
    kernel_b, log_rate`` (the Stage-1 6-param model) — and producing a
    :class:`CoagulationPopulation` around a freshly-built ``solver="mc_jax"``
    response.  Everything else (``v_esc, m_min, m_max, kernel_a, k_max,
    n_seed, depletion_fraction, gate, key``) is captured here as the
    ``constant_params`` of the fit.

    Wire into gwkokab::

        dist_fn = make_coagulation_dist_fn(v_esc=..., m_min=..., m_max=...,
                                           gate="soft", mc_jax_key=key)
        log_lik_fn = numpyro_analytical_poisson_likelihood(
            dist_fn=dist_fn,
            priors=...,                      # JointDistribution over the 6 params
            variables=...,                   # {name: prior_dist}
            constant_params={},              # (already captured in dist_fn)
            variables_index=...,             # {name: i}
            poisson_mean_estimator=coagulation_poisson_mean_estimator,
            variance_cut_threshold=None,
        )

    Default ``gate="soft"`` (B.2) — the cleaner branch; pass
    ``gate="bernoulli"`` for the B.1 cross-check.

    Two-pulse model
    ---------------
    ``natal_family="two_gaussian"`` switches the returned ``dist_fn`` to the
    two-pulse natal IMF (``build_two_pulse_coagulation``;
    INSTRUCTIONS_gwkokab_stage1 §5.1).  The closure then consumes the
    expanded keyword set ``mu_natal, sigma_natal, mu_natal_2, sigma_natal_2,
    lambda_natal, natal_spin_loc, natal_spin_scale, kernel_b, log_rate`` —
    and ``kernel_a`` too when ``float_kernel_a=True`` (the §5.1 "kernel_a
    floated → 10 free params" setup).  ``(mu_natal, sigma_natal)`` is pulse
    1, ``(mu_natal_2, sigma_natal_2)`` pulse 2, ``lambda_natal`` the weight
    of pulse 2; the ``mu_natal < mu_natal_2`` label-switch convention is
    enforced fit-side by the priors / reparametrisation in
    :mod:`stage1_wiring`, not here.

    Solver
    ------
    ``solver`` (default ``"mc_jax"``) selects the density backend the
    likelihood reads through.  ``"implicit"`` routes both pieces through
    the grid Picard fixed point (``response/implicit.py``, AD via its
    ``@jax.custom_vjp`` IFT): ``log_prob`` falls through
    :meth:`CoagulationResponse.log_rate_density`'s grid path and
    ``total_rate`` integrates the converged ``N*`` on the engine's
    trapezoid measure (``observables.total_rate``).  This is the cure for
    the two-pulse λ/σ₂ bias: the implicit solver carries the
    mass-dependent ``N_natal − loss[N]`` term, so the surviving 1G shape
    is reshaped correctly where the ``mc_jax`` hybrid's scalar ``W_1g ×
    raw-natal`` factorisation cannot (see
    ``demo/HANDOFF_two_pulse_recovery.md``).  ``epsilon`` (coupling×time,
    absolute physical scale) is forwarded to the implicit solver and
    ignored by the MC backends.

    Spin orientation (Fisher-κ)
    ---------------------------
    ``orientation`` selects the spin-orientation model marginalised over at
    each merger event (``spin_orientation.Aligned`` / ``Isotropic`` /
    ``Fisher``).  ``None`` (the default) keeps the bit-for-bit v1 Aligned
    path.  Pass ``Fisher(kappa=…)`` or ``Isotropic()`` to engage the vector
    Barausse-Rezzolla / Lousto-Zlochower remnant physics — the engine
    promotes the scalar progenitor spins to vectors with tilts drawn from
    ``P(θ̂; κ)``, runs the full vector remnant + kick, and marginalises back
    to ``(|χ_rem|, v_kick)``.  This is what makes the cascade spin
    distribution physical (and κ-tunable) rather than perfectly aligned.

    The model is **fixed per run** (captured in the closure, same as
    ``v_esc``), but the wiring is inference-ready: under
    ``solver="implicit"`` the IFT ``@jax.custom_vjp`` already flows
    ``∂/∂κ`` through the orientation marginal (and recovers a nonzero
    ``∂/∂v_esc`` once a non-Aligned model + the smooth ``sigmoid``
    retention are in play — Aligned/Heaviside gives identically zero).  To
    make κ a *fit parameter* later, lift it into the ``dist_fn`` keyword
    set and rebuild ``Fisher(kappa=…)`` inside the closure; no engine
    change is needed.

    ``retention_kind`` (default ``"sigmoid"``) and ``w_over_v_esc`` are
    forwarded alongside ``orientation``; they configure the per-orientation
    kick-retention and are ignored under Aligned.
    """
    if mc_jax_key is None:
        mc_jax_key = jax.random.PRNGKey(0)
    if natal_family not in ("gaussian", "two_gaussian"):
        raise ValueError(
            "natal_family must be 'gaussian' or 'two_gaussian'; got "
            f"{natal_family!r}"
        )
    if solver not in ("mc_jax", "mc", "implicit", "perturbative", "basis"):
        raise ValueError(
            "solver must be one of {'mc_jax','mc','implicit','perturbative',"
            f"'basis'}}; got {solver!r}"
        )

    # Lazy import — pulls the engine response package (→ gwkokab) only when
    # a fit actually builds responses, not at bridge-module import time.
    from ..response.coagulation import (
        build_gaussian_coagulation,
        build_two_pulse_coagulation,
    )

    def _wrap(response) -> CoagulationPopulation:
        env = {"v_esc": jnp.asarray(v_esc)}
        if solver == "mc_jax":
            total_rate_fn = (
                lambda **kw: response.mc_jax_response.total_rate(**kw)
            )
        elif solver == "mc":
            total_rate_fn = (
                lambda **kw: response.mc_response.total_rate(**kw)
            )
        else:
            # Grid solvers (implicit / perturbative / basis): integrate the
            # engine's single-BH density on its own trapezoid measure — the
            # SAME convention :meth:`CoagulationResponse.log_rate_density`
            # reads through ``_log_total_density`` (so the gwkokab shape term
            # and the −μ term are mutually consistent).  AD-traceable: the
            # implicit solve carries its IFT ``@jax.custom_vjp``; the
            # quadrature after it is pure.  ``n_quad`` (passed by the
            # estimator) is irrelevant on a grid and is harmlessly ignored.
            from ..observables import total_rate as _grid_total_rate

            def total_rate_fn(**_kw):
                log_N = response._log_total_density(env["v_esc"])
                N = jnp.exp(log_N)
                return _grid_total_rate(
                    N, response.grid, response.kernel_params,
                    response.log_R_env,
                )
        return CoagulationPopulation(
            response, env, total_rate_fn,
            m_min=m_min, m_max=m_max, m_max_factor=m_max_factor,
        )

    if natal_family == "gaussian":
        def dist_fn(
            *,
            mu_natal,
            sigma_natal,
            natal_spin_loc,
            natal_spin_scale,
            kernel_b,
            log_rate,
            kernel_a=kernel_a,
            **_extra,
        ) -> CoagulationPopulation:
            response = build_gaussian_coagulation(
                mu=mu_natal,
                sigma=sigma_natal,
                m_min=m_min,
                m_max=m_max,
                natal_spin_loc=natal_spin_loc,
                natal_spin_scale=natal_spin_scale,
                kernel_a=kernel_a,
                kernel_b=kernel_b,
                log_R_env=log_rate,
                k_max=k_max,
                grid=grid,
                solver=solver,
                epsilon=epsilon,
                orientation=orientation,
                retention_kind=retention_kind,
                w_over_v_esc=w_over_v_esc,
                mc_v_esc=v_esc,
                mc_n_seed=n_seed,
                mc_depletion_fraction=depletion_fraction,
                mc_jax_key=mc_jax_key,
                mc_jax_gate=gate,
                mc_jax_n_steps_mult=n_steps_mult,
            )
            return _wrap(response)
    else:  # two_gaussian
        def dist_fn(
            *,
            mu_natal,
            sigma_natal,
            mu_natal_2,
            sigma_natal_2,
            lambda_natal,
            natal_spin_loc,
            natal_spin_scale,
            kernel_b,
            log_rate,
            kernel_a=kernel_a,
            **_extra,
        ) -> CoagulationPopulation:
            response = build_two_pulse_coagulation(
                mu_1=mu_natal,
                sigma_1=sigma_natal,
                mu_2=mu_natal_2,
                sigma_2=sigma_natal_2,
                lambda_2=lambda_natal,
                m_min=m_min,
                m_max=m_max,
                natal_spin_loc=natal_spin_loc,
                natal_spin_scale=natal_spin_scale,
                kernel_a=kernel_a,
                kernel_b=kernel_b,
                log_R_env=log_rate,
                k_max=k_max,
                grid=grid,
                solver=solver,
                epsilon=epsilon,
                orientation=orientation,
                retention_kind=retention_kind,
                w_over_v_esc=w_over_v_esc,
                mc_v_esc=v_esc,
                mc_n_seed=n_seed,
                mc_depletion_fraction=depletion_fraction,
                mc_jax_key=mc_jax_key,
                mc_jax_gate=gate,
                mc_jax_n_steps_mult=n_steps_mult,
                mc_jax_kde_adaptive=kde_adaptive,
            )
            return _wrap(response)

    return dist_fn
