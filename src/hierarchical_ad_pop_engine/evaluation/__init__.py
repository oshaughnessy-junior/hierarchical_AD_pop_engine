r"""Arbitrary-point density / observable evaluation API.

See ``demo/PLAN_arbitrary_point_evaluation.md`` for the motivation and
the full protocol contract.

The :class:`RateDensityEvaluator` protocol decouples *plot/query
resolution* from *solver internal resolution*: each solver result type
exposes an ``.evaluator(response, ...)`` constructor returning a thin
wrapper that answers point queries (``N(m, χ)``, ``½ R_env K N N``,
1-D / 2-D marginals, total rate) at arbitrary continuous points
without re-running the solver.

This sub-package is **additive** — it does not replace the grid-
quadrature path in :mod:`hierarchical_ad_pop_engine.observables`,
which remains the fast, AD-friendly bulk integral used by inference.
The new API is for cheap one-off plot / export queries and for
swapping evaluators inside gwkokab without call-site changes.

Sequence step 1 shipped the protocol and the implicit-solver
evaluator.  Step 2 (this commit) adds the basis-solver evaluator.
The MC evaluator follows in step 3 of ``PLAN_arbitrary_point_evaluation.md``,
living in ``red_team/`` because the particle MC itself is a red-team
backstop.
"""

from .protocol import RateDensityEvaluator
from .implicit import ImplicitEvaluator
from .basis import BasisEvaluator

__all__ = ["RateDensityEvaluator", "ImplicitEvaluator", "BasisEvaluator"]
