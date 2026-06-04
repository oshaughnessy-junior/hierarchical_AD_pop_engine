r"""Inference bridges — wire the coagulation engine into population-inference
frameworks (currently gwkokab).

See :mod:`hierarchical_ad_pop_engine.inference.gwkokab_bridge`.
"""

from .gwkokab_bridge import (
    CoagulationPopulation,
    coagulation_poisson_mean_estimator,
    make_coagulation_dist_fn,
)
from .sampled_wiring import (
    build_sampled_likelihood,
    gaussian_event_log_likelihoods_from_posteriors,
    load_rift_event_log_likelihoods,
    make_coagulation_sampler_fn,
)

__all__ = [
    "CoagulationPopulation",
    "coagulation_poisson_mean_estimator",
    "make_coagulation_dist_fn",
    "build_sampled_likelihood",
    "gaussian_event_log_likelihoods_from_posteriors",
    "load_rift_event_log_likelihoods",
    "make_coagulation_sampler_fn",
]
