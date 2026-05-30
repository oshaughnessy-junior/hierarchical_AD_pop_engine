r"""Stage-1 gwkokab fit wiring — priors / variables / variables_index + likelihood.

This is the glue that turns the :mod:`gwkokab_bridge` ``dist_fn`` into a
*runnable* gwkokab population fit (``PLAN_gwkokab_fit.md`` Stage 1, the
6-parameter single-Gaussian / flat-selection model).  It supplies the three
objects ``gwkokab.inference.numpyro_analytical_poisson_likelihood`` consumes
alongside the ``dist_fn`` and the Poisson-mean estimator:

* ``variables``        — ``{name: numpyro.Distribution}`` (the per-parameter priors),
* ``variables_index``  — ``{name: i}`` where ``i`` indexes the **alphabetically
  sorted** variable list (this IS gwkokab's contract; see below),
* ``priors``           — a gwkokab ``JointDistribution`` over the variables in
  that same sorted order.

**The sorted-order contract (load-bearing, easy to get wrong).**  Inside
``numpyro_analytical_poisson_likelihood`` the per-parameter samples are drawn as

    sorted_variables = sorted(variables.items(), key=lambda kv: kv[0])
    variables_samples = [numpyro.sample(name, prior) for name, prior in sorted_variables]
    mapped_params = {name: variables_samples[i] for name, i in variables_index.items()}
    model_instance = dist_fn(**constant_params, **mapped_params)

so ``variables_index[name]`` must be the position of ``name`` in
``sorted(variables)`` — exactly how gwkokab's own
``analysis.core.guru.classify_model_parameters`` builds it
(``{key: i for i, key in enumerate(sorted(variables.keys()))}``).  If the
index map and the prior tuple disagree on ordering, every parameter is fed to
the wrong slot of ``dist_fn`` and the fit silently recovers garbage.  We
replicate gwkokab's construction verbatim and pin it with an in-sandbox test
(``test_gwkokab_stage1_wiring.py``) that mirrors the ``mapped_params`` logic.

**Imports.**  ``numpyro`` is a hard dep (cheap).  ``gwkokab`` is imported
**lazily** inside the two functions that actually need it
(:func:`build_stage1_priors` → ``JointDistribution``, and
:func:`build_coagulation_likelihood` → ``numpyro_analytical_poisson_likelihood``),
because ``import gwkokab`` transitively pulls RIFT/lalsuite (see the bridge
module note).  Everything that does NOT need gwkokab — the priors dict, the
variables, the index map — is pure numpyro and runs in the lightweight
sandbox fast-loop.

Hard rules honoured:
* additive — a new module under ``inference/``; nothing in ``response/`` or
  the existing bridge changes;
* no fudge factors — the priors are exactly ``PLAN_gwkokab_fit.md`` Stage 1;
* flat selection stays the bridge default
  (``coagulation_poisson_mean_estimator``, ``VT ≡ 1``); a ``pdet``-weighted
  estimator is a Stage-2 follow-up passed in via ``poisson_mean_estimator``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpyro.distributions as dist
from numpyro.distributions import Distribution

from .gwkokab_bridge import (
    coagulation_poisson_mean_estimator,
    make_coagulation_dist_fn,
)

__all__ = [
    "STAGE1_PARAM_NAMES",
    "stage1_prior_factories",
    "build_stage1_variables",
    "build_stage1_priors",
    "build_coagulation_likelihood",
    "build_stage1_likelihood",
]


# The 6 free Stage-1 parameters.  Order here is the *model* order (matches the
# ``dist_fn`` keyword signature); the gwkokab sample order is alphabetical and
# is derived from this set in :func:`build_stage1_variables`.
STAGE1_PARAM_NAMES: Tuple[str, ...] = (
    "mu_natal",
    "sigma_natal",
    "natal_spin_loc",
    "natal_spin_scale",
    "kernel_b",
    "log_rate",
)


def stage1_prior_factories() -> Dict[str, Callable[[], Distribution]]:
    r"""Canonical Stage-1 priors as zero-arg factories (``PLAN_gwkokab_fit.md``).

    Factories (not instances) so each call to :func:`build_stage1_variables`
    gets fresh, independent ``Distribution`` objects — numpyro distributions
    are lightweight pytrees but reusing one instance across two ``variables``
    dicts can entangle traced state, so we mint per build.

    +---------------------+----------------------------+
    | name                | prior                      |
    +=====================+============================+
    | ``mu_natal``        | ``Uniform(5, 50)`` M⊙      |
    | ``sigma_natal``     | ``LogUniform(0.5, 10)`` M⊙ |
    | ``natal_spin_loc``  | ``Uniform(0, 0.3)``        |
    | ``natal_spin_scale``| ``LogUniform(0.02, 0.3)``  |
    | ``kernel_b``        | ``Uniform(-2, 6)``         |
    | ``log_rate``        | ``Uniform(-2, 6)``         |
    +---------------------+----------------------------+
    """
    return {
        "mu_natal": lambda: dist.Uniform(5.0, 50.0),
        "sigma_natal": lambda: dist.LogUniform(0.5, 10.0),
        "natal_spin_loc": lambda: dist.Uniform(0.0, 0.3),
        "natal_spin_scale": lambda: dist.LogUniform(0.02, 0.3),
        "kernel_b": lambda: dist.Uniform(-2.0, 6.0),
        "log_rate": lambda: dist.Uniform(-2.0, 6.0),
    }


def build_stage1_variables(
    prior_factories: Optional[Dict[str, Callable[[], Distribution]]] = None,
) -> Tuple[Dict[str, Distribution], Dict[str, int]]:
    r"""``(variables, variables_index)`` for the Stage-1 fit.

    ``variables`` maps each parameter name to a fresh prior ``Distribution``.
    ``variables_index`` maps each name to its position in the **alphabetically
    sorted** variable list — gwkokab's exact convention
    (``{key: i for i, key in enumerate(sorted(variables.keys()))}``), so the
    samples drawn in sorted order land in the right ``dist_fn`` keyword.

    Parameters
    ----------
    prior_factories
        Override the default :func:`stage1_prior_factories` (e.g. to tighten a
        prior for a diagnostic).  Keys must be exactly
        :data:`STAGE1_PARAM_NAMES`.
    """
    factories = prior_factories if prior_factories is not None else stage1_prior_factories()
    missing = set(STAGE1_PARAM_NAMES) - set(factories)
    extra = set(factories) - set(STAGE1_PARAM_NAMES)
    if missing or extra:
        raise ValueError(
            f"prior_factories keys must be exactly {STAGE1_PARAM_NAMES}; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    variables: Dict[str, Distribution] = {name: factories[name]() for name in STAGE1_PARAM_NAMES}
    variables_index: Dict[str, int] = {
        name: i for i, name in enumerate(sorted(variables.keys()))
    }
    return variables, variables_index


def build_stage1_priors(
    variables: Optional[Dict[str, Distribution]] = None,
    *,
    validate_args: bool = True,
) -> Any:
    r"""gwkokab ``JointDistribution`` over the Stage-1 priors (sorted order).

    Mirrors ``classify_model_parameters``:
    ``JointDistribution(*[variables[k] for k in sorted(variables)])``.  The
    marginal order MUST match the sorted sample order in the likelihood, hence
    the same ``sorted(variables)`` key sequence used by
    :func:`build_stage1_variables` for ``variables_index``.

    Lazily imports ``gwkokab`` (→ RIFT/lalsuite); call only where the full
    stack is installed (cardassia).  In-sandbox, validate the *ordering*
    contract via :func:`build_stage1_variables` + the inline-replica test
    instead of constructing the real joint.
    """
    if variables is None:
        variables, _ = build_stage1_variables()
    from gwkokab.models.utils import JointDistribution  # lazy: pulls RIFT/lalsuite

    sorted_keys = sorted(variables.keys())
    return JointDistribution(
        *[variables[k] for k in sorted_keys],
        validate_args=validate_args,
    )


def build_coagulation_likelihood(
    *,
    dist_fn: Callable[..., Any],
    priors: Any,
    variables: Dict[str, Distribution],
    variables_index: Dict[str, int],
    poisson_mean_estimator: Callable[..., Any] = coagulation_poisson_mean_estimator,
    constant_params: Optional[Dict[str, Any]] = None,
    variance_cut_threshold: Optional[float] = None,
) -> Callable[..., None]:
    r"""Wrap the bridge in gwkokab's ``numpyro_analytical_poisson_likelihood``.

    Returns the numpyro likelihood callable
    ``log_likelihood_fn(samples_stack, ln_offsets, pmean_kwargs)`` ready to be
    plugged into SVI / NUTS.  Thin, faithful pass-through — no reshaping or
    reweighting happens here; the bridge's ``CoagulationPopulation.log_prob``
    (= ``log_rate_density``) and ``coagulation_poisson_mean_estimator`` carry
    the physics, gwkokab carries the Poisson bookkeeping.

    ``constant_params`` defaults to ``{}`` because
    :func:`make_coagulation_dist_fn` already closes over the fixed model
    settings (``v_esc, m_min, m_max, kernel_a, k_max, n_seed, …``); the only
    things gwkokab feeds ``dist_fn`` are the 6 sampled ``mapped_params``.

    ``variance_cut_threshold`` stays ``None`` under flat selection — the MC
    ``total_rate`` variance is unused there, and the metric itself reads low at
    small ``n_seed`` (``[[project-mc-jax-total-rate-bias]]``), so a variance
    taper must NOT be enabled until that is fixed.

    Lazily imports ``gwkokab.inference`` (→ RIFT/lalsuite).
    """
    from gwkokab.inference import (  # lazy: pulls RIFT/lalsuite
        numpyro_analytical_poisson_likelihood,
    )

    return numpyro_analytical_poisson_likelihood(
        dist_fn=dist_fn,
        priors=priors,
        variables=variables,
        constant_params={} if constant_params is None else constant_params,
        variables_index=variables_index,
        poisson_mean_estimator=poisson_mean_estimator,
        variance_cut_threshold=variance_cut_threshold,
    )


def build_stage1_likelihood(
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
    poisson_mean_estimator: Callable[..., Any] = coagulation_poisson_mean_estimator,
    prior_factories: Optional[Dict[str, Callable[[], Distribution]]] = None,
    variance_cut_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    r"""One-call Stage-1 wiring: everything a gwkokab fit needs, in a dict.

    Composes :func:`make_coagulation_dist_fn`, :func:`build_stage1_variables`,
    :func:`build_stage1_priors`, and :func:`build_coagulation_likelihood`.

    Returns
    -------
    dict with keys ``dist_fn``, ``variables``, ``variables_index``,
    ``priors``, ``poisson_mean_estimator``, ``log_likelihood_fn``.  Hand the
    last straight to ``numpyro.infer.SVI`` / ``NUTS`` (it is the numpyro model
    body); the rest are returned for inspection / mock-catalogue generation.

    Lazily pulls ``gwkokab`` (the ``priors`` build + the likelihood wrap), so
    this convenience runs only where the full stack is installed.  In the
    sandbox, compose the pure-numpyro pieces
    (:func:`make_coagulation_dist_fn`, :func:`build_stage1_variables`)
    directly and skip :func:`build_stage1_priors`.
    """
    dist_fn = make_coagulation_dist_fn(
        v_esc=v_esc,
        m_min=m_min,
        m_max=m_max,
        kernel_a=kernel_a,
        k_max=k_max,
        n_seed=n_seed,
        depletion_fraction=depletion_fraction,
        gate=gate,
        n_steps_mult=n_steps_mult,
        mc_jax_key=mc_jax_key,
        grid=grid,
        m_max_factor=m_max_factor,
    )
    variables, variables_index = build_stage1_variables(prior_factories)
    priors = build_stage1_priors(variables)
    log_likelihood_fn = build_coagulation_likelihood(
        dist_fn=dist_fn,
        priors=priors,
        variables=variables,
        variables_index=variables_index,
        poisson_mean_estimator=poisson_mean_estimator,
        variance_cut_threshold=variance_cut_threshold,
    )
    return {
        "dist_fn": dist_fn,
        "variables": variables,
        "variables_index": variables_index,
        "priors": priors,
        "poisson_mean_estimator": poisson_mean_estimator,
        "log_likelihood_fn": log_likelihood_fn,
    }
