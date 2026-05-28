r"""Re-export shim — :class:`MCEvaluator` lives in
:mod:`hierarchical_ad_pop_engine.evaluation.mc` since Session M2 of
``demo/PLAN_mc_first_reprioritization.md``.

Originally shipped here as the particle-MC validation backstop
(Step 3 of ``demo/PLAN_arbitrary_point_evaluation.md``;
``[[project-evaluation-step3-done]]``).  M2 promoted the file to
``src/hierarchical_ad_pop_engine/evaluation/mc.py`` because the same
class is now the inference-side ``log_prob`` backbone behind
:class:`hierarchical_ad_pop_engine.response.mc.MCResponse`.

Existing red-team callers (``tests/test_evaluation_mc.py``,
``red_team/mc_time_calibration.py``, ``red_team/observable_metrics.py``)
keep working through this shim — no import-path churn at the call
sites that the user owns.

New code should import from
``hierarchical_ad_pop_engine.evaluation.mc`` (or
``hierarchical_ad_pop_engine.evaluation`` re-export) directly.
"""

from __future__ import annotations

# Re-export everything the original module exposed.  All symbols below
# (``MCEvaluator``, the internal grid helpers, and the bilinear interp)
# are imported with their original names so dotted-attribute access
# like ``red_team.evaluation_mc._log_pairing_kernel_numpy`` continues
# to resolve — :mod:`tests.test_evaluation_mc` exercises a couple of
# these private helpers directly, and breaking that surface is the
# kind of dispatch-callgraph leak ``[[feedback-dispatch-callgraph-audit]]``
# explicitly flags.
from hierarchical_ad_pop_engine.evaluation.mc import (  # noqa: F401
    MCEvaluator,
    _bilinear_interp_2d,
    _internal_marginal_2d,
    _knn_balloon_2d,
    _log_pairing_kernel_numpy,
    _trapezoid_weights_linear,
    _trapezoid_weights_log,
)

__all__ = ["MCEvaluator"]
