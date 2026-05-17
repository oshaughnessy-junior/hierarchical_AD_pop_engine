"""Redshift rate profile — re-export of gwkokab's redshift models.

The redshift dependence enters our pipeline at the **formation-rate** layer
(``hierarchical_ad_pop_engine.formation``), not the response function — under
the v1 quasi-instantaneous-coagulation approximation, a single environment's
mergers are localised in cosmic time. We expose gwkokab's
``MadauDickinsonRedshift`` and ``PowerlawRedshift`` here for convenient
construction by formation models.

``log_prob`` on these returns ``log[dN / dV_c dt dz]`` (rate density, includes
``dV_c/dz`` and the ``1/(1+z)`` time-dilation factor), matching gwkokab's
convention for compatibility with ``numpyro_poisson_likelihood``.
"""

from __future__ import annotations

from gwkokab.models.redshift import MadauDickinsonRedshift, PowerlawRedshift

__all__ = ["MadauDickinsonRedshift", "PowerlawRedshift"]
