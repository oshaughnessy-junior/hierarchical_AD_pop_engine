r"""Misalignment Session D — slow workstation pytest gating MC ↔ engine
parity at the observable layer under three orientations.

Parametrises the Phase C parity battery
(``tests/test_redteam_observables_parity.py``) over
``[Aligned, Isotropic, Fisher(κ=5)]``.

Per orientation:

* ``dN/dm_1``                   1-D JS-vs-noise gate.
* ``dN/dχ_eff``                 1-D JS-vs-noise gate.  **Headline** —
  the direct observable view of the cascade-compression-breaks-under-
  iso physical claim from ``[[project-misalignment-mode-planned]]``.
* ``dN/(dm_1 dχ_eff)``          2-D rel-L1-vs-noise gate.
* ``dN/(dm_1 dq)``              2-D rel-L1-vs-noise gate.

All assertions are MAGNITUDES (ratio-vs-noise, integrated ratios)
rather than orientation-vs-orientation DIRECTIONS, after the lesson
from ``[[project-misalignment-session-c-done]]`` rounds 3-4 (cascade
+ total_rate inequalities both turned out fixture-dependent and
needed magnitude reformulation).  See
``[[feedback-no-frozen-test-references]]``.

Bulk-mask threshold ``tail_frac=0.01`` (same as Phase C); we do NOT
tune per orientation per ``[[feedback-no-fudge-factors]]``.  Each
parity assertion uses the SAME multiplier across orientations — if a
multiplier needs relaxing for non-Aligned, that's a finding worth a
docstring paragraph identifying which physical / MC-noise term grew,
NOT a per-orientation knob.

Stringency budget (set conservatively above Phase C's already-loose
v1 bar; tighten if workstation results show headroom):

* ``JS_MULT_1D = 50.0`` — matches Phase C's mass-axis multiplier.
* ``RL1_MULT_2D = 10.0`` — ratio-of-rel-L1 across MC seeds.  Phase C
  used absolute ``rel_L1 < 0.5`` because the Aligned regime has small
  MC variance; under Iso/Fisher the orientation-quadrature variance
  grows so the ratio-vs-noise gate is more honest.
* ``TOTAL_RATE_TOLERANCE_1D = (0.85, 1.15)`` — Phase C convention.
* ``TOTAL_RATE_TOLERANCE_2D = (0.75, 1.25)`` — Phase C convention for
  the q-restricted (q ≥ 0.05) total-rate window.
* ``MIN_BULK_COVERAGE = 0.85`` — engine bulk-mask coverage sanity.

All tests carry ``@pytest.mark.slow``.  Opt in with::

    pytest tests/test_misalignment_observable_parity.py -m slow -v

Fixture cost on workstation: three implicit solves + six MC runs
(2 seeds × 3 orientations) at 64×20 — well under one minute total
post-2026-05-27 (the spin_orientation default quadrature drop +
sigmoid default cut per-solve cost ~8×).

Sigmoid-vs-Heaviside-MC subtlety
--------------------------------
Post-2026-05-27 the engine defaults to ``retention_kind="sigmoid"``
which makes the orientation integrand smooth (so the cheap (4, 8) =
128-point quadrature converges spectrally) and recovers AD-on-
``v_esc``.  The MC sampler, by contrast, always does a per-event
Heaviside (you cannot accept-half-an-event).  Under any non-Aligned
orientation this introduces a systematic ``O(w_over_v_esc)`` bias
between engine and MC — at the default ``w_over_v_esc = 1.0`` it is
a few percent.  This is well within the ``50× / 10× MC-noise``
gates, but if a future regime makes the bias important you can opt
back into heaviside via the engine kwargs (the Aligned path is
unaffected — Aligned dispatch shortcuts retention_kind in both
engine and MC).
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.spin_orientation import (
    Aligned, Fisher, Isotropic,
)

# The Session D demo lives in ``demo/`` (sibling of ``tests/``).  ``demo/``
# is not a package — we add its parent to sys.path so the import resolves
# as a flat module, mirroring the pattern in
# ``tests/test_integration_stage1_recovery.py``.
_DEMO_DIR = (Path(__file__).resolve().parent.parent / "demo").resolve()
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))
import misalignment_observable_parity as _mop  # noqa: E402

THETA_STAR = _mop.THETA_STAR
compute_metrics = _mop.compute_metrics
solve_and_sample = _mop.solve_and_sample


# ---------------------------------------------------------------------------
# Per-test budget — held identical across orientations on purpose.
# ---------------------------------------------------------------------------
_RESOLUTION = (64, 20)
_EPSILON = 1.0
_N_MC = 20_000      # same as Phase C; keeps fixture under a few minutes total
_SEED = 20260527
_TAIL_FRAC = 0.01

JS_MULT_1D = 50.0
RL1_MULT_2D = 10.0
TOTAL_RATE_TOLERANCE_1D = (0.85, 1.15)
TOTAL_RATE_TOLERANCE_2D = (0.75, 1.25)
MIN_BULK_COVERAGE = 0.85

KAPPA = 5.0

ORIENTATIONS = [
    ("aligned",          Aligned()),
    ("isotropic",        Isotropic.default()),
    (f"fisher_k{KAPPA:g}", Fisher.default(kappa=KAPPA)),
]


# ---------------------------------------------------------------------------
# Module-scoped fixture: one payload per orientation; parametrised so the
# pytest test_id carries the orientation tag for diagnostic readability.
# ---------------------------------------------------------------------------
@pytest.fixture(
    scope="module",
    params=ORIENTATIONS,
    ids=[t for t, _ in ORIENTATIONS],
)
def parity_payload(request):
    r"""Run engine + 2 MC seeds for one orientation; compute metrics."""
    tag, orientation = request.param
    payload = solve_and_sample(
        theta=dict(THETA_STAR),
        n_m=_RESOLUTION[0],
        n_chi=_RESOLUTION[1],
        epsilon=_EPSILON,
        n_mc=_N_MC,
        seed=_SEED,
        orientation=orientation,
    )
    metrics = compute_metrics(payload, tail_frac=_TAIL_FRAC)
    return dict(tag=tag, orientation=orientation, payload=payload, metrics=metrics)


# ---------------------------------------------------------------------------
# Shared assertion helpers.
# ---------------------------------------------------------------------------
def _assert_finite(name: str, value: float, tag: str) -> None:
    assert np.isfinite(value), (
        f"[{tag}] {name} not finite (={value!r}); fixture likely produced "
        f"empty MC or zero engine support."
    )


def _assert_total_rate_window(
    obs_name: str, ratio: float, tag: str, window: tuple[float, float],
) -> None:
    lo, hi = window
    _assert_finite(f"{obs_name} total_rate_ratio", ratio, tag)
    assert lo <= ratio <= hi, (
        f"[{tag}] {obs_name} total_rate_ratio MC/engine = {ratio:.3f} "
        f"outside [{lo}, {hi}] tolerance at n_seed = {_N_MC}."
    )


def _assert_bulk_coverage(obs_name: str, cov: float, tag: str) -> None:
    _assert_finite(f"{obs_name} bulk_coverage", cov, tag)
    assert cov >= MIN_BULK_COVERAGE, (
        f"[{tag}] {obs_name} engine-derived bulk-mask covers only "
        f"{cov:.2%} of engine rate at tail_frac={_TAIL_FRAC}; mask too "
        f"aggressive for this orientation?  "
        f"Per [[feedback-no-fudge-factors]] do NOT widen tail_frac per "
        f"orientation — investigate the orientation engine output first."
    )


# ---------------------------------------------------------------------------
# Parity tests — one per observable; each parametrised over orientation
# via the fixture's params.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_dN_dm1_parity(parity_payload):
    r"""1-D ``dN/dm_1``: bulk JS ≤ ``JS_MULT_1D × MC-noise JS``.

    Holds across all three orientations.  Promotes the Phase C
    ``test_dN_dm1_parity`` to a parametrised-over-orientation invariant.
    """
    tag = parity_payload["tag"]
    m = parity_payload["metrics"]["dN_dm1"]
    _assert_finite("dN_dm1 js_imp_mc", m["js_imp_mc"], tag)
    _assert_finite("dN_dm1 js_noise",  m["js_noise"],  tag)
    assert m["js_imp_mc"] < JS_MULT_1D * m["js_noise"], (
        f"[{tag}] dN/dm_1: bulk JS engine-vs-MC {m['js_imp_mc']:.4g} > "
        f"{JS_MULT_1D}× MC-noise JS {m['js_noise']:.4g}  (ratio "
        f"{m['ratio_imp_to_noise']:.1f}×).  Engine bias has grown "
        f"larger than the MC-noise budget × multiplier; investigate the "
        f"orientation pipeline before relaxing this gate."
    )
    _assert_total_rate_window(
        "dN_dm1", m["total_rate_ratio"], tag, TOTAL_RATE_TOLERANCE_1D,
    )
    _assert_bulk_coverage("dN_dm1", m["bulk_coverage"], tag)


@pytest.mark.slow
def test_dN_dchi_eff_parity(parity_payload):
    r"""1-D ``dN/dχ_eff``: bulk JS ≤ ``JS_MULT_1D × MC-noise JS``.

    **Session D headline observable.**  This is the cleanest direct
    view of the cascade-compression-breaks-under-iso physical claim
    from ``[[project-misalignment-mode-planned]]``: under Aligned the
    χ_eff distribution piles up at narrow cascade peaks; under
    Isotropic / Fisher the orientation broadening washes those peaks
    out into a smooth distribution.  Both regimes must agree
    engine-vs-MC within the JS-noise budget.
    """
    tag = parity_payload["tag"]
    m = parity_payload["metrics"]["dN_dchi_eff"]
    _assert_finite("dN_dchi_eff js_imp_mc", m["js_imp_mc"], tag)
    _assert_finite("dN_dchi_eff js_noise",  m["js_noise"],  tag)
    assert m["js_imp_mc"] < JS_MULT_1D * m["js_noise"], (
        f"[{tag}] dN/dχ_eff: bulk JS engine-vs-MC {m['js_imp_mc']:.4g} "
        f"> {JS_MULT_1D}× MC-noise JS {m['js_noise']:.4g}  (ratio "
        f"{m['ratio_imp_to_noise']:.1f}×).  This is the headline "
        f"misalignment observable; bias here points at a real engine "
        f"or MC physics mismatch (not a tolerance issue)."
    )
    _assert_total_rate_window(
        "dN_dchi_eff", m["total_rate_ratio"], tag, TOTAL_RATE_TOLERANCE_1D,
    )
    _assert_bulk_coverage("dN_dchi_eff", m["bulk_coverage"], tag)


@pytest.mark.slow
def test_dN_dm1_dchi_eff_parity(parity_payload):
    r"""2-D ``dN/(dm_1 dχ_eff)``: bulk rel-L1 ≤ ``RL1_MULT_2D × MC-noise rel-L1``.

    Phase C uses absolute ``rel_L1 < 0.5`` because the Aligned regime
    has small MC variance.  Under Iso/Fisher the orientation-quadrature
    variance grows, so the ratio-vs-noise gate (which auto-scales) is
    the right form for the parametrised test.
    """
    tag = parity_payload["tag"]
    m = parity_payload["metrics"]["dN_dm1_dchi_eff"]
    _assert_finite("dN_dm1_dchi_eff rel_l1_imp_mc", m["rel_l1_imp_mc"], tag)
    _assert_finite("dN_dm1_dchi_eff rel_l1_noise",  m["rel_l1_noise"],  tag)
    assert m["rel_l1_imp_mc"] < RL1_MULT_2D * m["rel_l1_noise"], (
        f"[{tag}] dN/(dm_1 dχ_eff): bulk rel-L1 engine-vs-MC "
        f"{m['rel_l1_imp_mc']:.4g} > {RL1_MULT_2D}× MC-noise rel-L1 "
        f"{m['rel_l1_noise']:.4g}  (ratio {m['ratio_imp_to_noise']:.1f}×)."
    )
    _assert_total_rate_window(
        "dN_dm1_dchi_eff", m["total_rate_ratio"], tag,
        TOTAL_RATE_TOLERANCE_1D,
    )
    _assert_bulk_coverage("dN_dm1_dchi_eff", m["bulk_coverage"], tag)


@pytest.mark.slow
def test_dN_dm1_dq_parity(parity_payload):
    r"""2-D ``dN/(dm_1 dq)``: bulk rel-L1 ≤ ``RL1_MULT_2D × MC-noise rel-L1``.

    ``q`` is clipped at q ∈ [0.05, 1.0] in both engine and MC (matches
    Phase C); the q < 0.05 tail is statistically thin enough on both
    sides that engine grid + MC sample combine make a noisy comparison
    pointless.  Total-rate window loosens to ±25 % because the q-tail
    drop differs slightly between engine and MC.
    """
    tag = parity_payload["tag"]
    m = parity_payload["metrics"]["dN_dm1_dq"]
    _assert_finite("dN_dm1_dq rel_l1_imp_mc", m["rel_l1_imp_mc"], tag)
    _assert_finite("dN_dm1_dq rel_l1_noise",  m["rel_l1_noise"],  tag)
    assert m["rel_l1_imp_mc"] < RL1_MULT_2D * m["rel_l1_noise"], (
        f"[{tag}] dN/(dm_1 dq): bulk rel-L1 engine-vs-MC "
        f"{m['rel_l1_imp_mc']:.4g} > {RL1_MULT_2D}× MC-noise rel-L1 "
        f"{m['rel_l1_noise']:.4g}  (ratio {m['ratio_imp_to_noise']:.1f}×)."
    )
    _assert_total_rate_window(
        "dN_dm1_dq", m["total_rate_ratio"], tag, TOTAL_RATE_TOLERANCE_2D,
    )
    _assert_bulk_coverage("dN_dm1_dq", m["bulk_coverage"], tag)


# ---------------------------------------------------------------------------
# Dispatch wire-up sanity: Aligned() via the parametrised pipeline must
# produce the same payload as Aligned-by-default (orientation=None).
# Guards against the Session C round-1-style dispatch leak where a kwarg
# silently dropped at an internal call site.  See
# [[feedback-dispatch-callgraph-audit]].
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_aligned_explicit_equals_default_bitwise():
    r"""``orientation=Aligned()`` ≡ ``orientation=None`` literally equal.

    Solver-level Session-C-style guard re-asserted at the observable
    layer.  Catches the failure mode where a kwarg added to the helper
    surface gets silently dropped on the path to ``MCConfig`` or
    ``CoagulationResponse``.
    """
    payload_explicit = solve_and_sample(
        theta=dict(THETA_STAR),
        n_m=32, n_chi=12,    # smaller — this is a bit-for-bit sanity, not a parity gate
        epsilon=_EPSILON,
        n_mc=2_000,
        seed=_SEED,
        orientation=Aligned(),
    )
    payload_default = solve_and_sample(
        theta=dict(THETA_STAR),
        n_m=32, n_chi=12,
        epsilon=_EPSILON,
        n_mc=2_000,
        seed=_SEED,
        orientation=None,
    )
    np.testing.assert_array_equal(
        payload_explicit["eng_dN_dm1"], payload_default["eng_dN_dm1"],
        err_msg="Engine dN/dm_1 differs between orientation=Aligned() and "
                "orientation=None — dispatch wire-up leak in helper surface "
                "(see [[feedback-dispatch-callgraph-audit]]).",
    )
    np.testing.assert_array_equal(
        payload_explicit["eng_m1_chi_eff"], payload_default["eng_m1_chi_eff"],
        err_msg="Engine dN/(dm_1 dχ_eff) differs between orientation=Aligned() "
                "and orientation=None — dispatch wire-up leak.",
    )
    np.testing.assert_array_equal(
        payload_explicit["mc1_dN_dm1"], payload_default["mc1_dN_dm1"],
        err_msg="MC dN/dm_1 differs between orientation=Aligned() and "
                "orientation=None — MCConfig.orientation Aligned shortcut leak.",
    )


# ---------------------------------------------------------------------------
# Cross-orientation reporting sanity (NOT a direction gate).
# Asserts only that the metrics fixture produced finite numbers for every
# orientation × every observable — guards against silently-skipped tests.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_all_metrics_finite(parity_payload):
    r"""Sanity: every metric returned by ``compute_metrics`` is finite.

    Per ``[[feedback-no-frozen-test-references]]`` we do not pin
    cross-orientation directions; this just guards against an
    orientation pipeline silently degenerating to empty MC or zero
    engine support (the failure-mode that would otherwise present as
    skipped or NaN assertions rather than a real test failure).
    """
    tag = parity_payload["tag"]
    for obs_name, m in parity_payload["metrics"].items():
        for k, v in m.items():
            if k in ("kind", "metric", "peak_engine", "peak_mc"):
                continue
            assert np.isfinite(v), (
                f"[{tag}] {obs_name}.{k} = {v!r} (expected finite)."
            )
