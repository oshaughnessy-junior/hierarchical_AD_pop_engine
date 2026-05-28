"""Tests for the MC ↔ engine time calibration (PLAN M1).

Two tiers:

* **Sandbox tier** (default): algebraic identities of the closed-form
  helpers in :mod:`red_team.mc_time_calibration`.  No MC run, no JAX,
  no scipy — runs in the Cowork sandbox.
* **Workstation tier** (``-m slow``): full MC sweep at the configured
  ``f_dep`` grid.  Gates the rel-L1 between MC ``N_active`` at each
  stop and the closed-form Smoluchowski ``N(t)`` at the calibrated
  time.

All gates are invariants vs the analytic Smoluchowski solution, never
against pinned ``.npz`` outputs
(`[[feedback-no-frozen-test-references]]`).  The 5 % rel-L1 budget
is the PLAN M1 headline; tightening or loosening it should be backed
by an MC-noise convergence argument
(`[[feedback-no-fudge-factors]]`).
"""

from __future__ import annotations

import numpy as np
import pytest

from red_team.mc_time_calibration import (
    CalibrationConfig,
    closed_form_f_dep,
    closed_form_total_number,
    epsilon_from_f_dep,
    epsilon_from_mc_state,
    t_mc_from_f_dep,
)


# ---------------------------------------------------------------------------
# Sandbox tier — closed-form algebraic identities
# ---------------------------------------------------------------------------


class TestClosedFormIdentities:
    """The closed-form helpers must satisfy the derivation in the module
    docstring as machine-precision algebra.
    """

    @pytest.mark.parametrize("f_dep", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9])
    def test_eq5_equiv_eq1(self, f_dep: float) -> None:
        """Equation (5) and (1) of the module docstring must agree.

        ``ε = 2·n_mergers/(K·n_active²)`` and
        ``ε = 2·f_dep/(K·N_natal·(1-f_dep)²)`` are algebraically
        equivalent under retention = 1.
        """
        K = 1.7
        N = 10_000
        n_mergers = N * f_dep
        n_active = N * (1.0 - f_dep)
        eps_5 = epsilon_from_mc_state(n_mergers, n_active, K)
        eps_1 = epsilon_from_f_dep(f_dep, K, N)
        assert eps_5 == pytest.approx(eps_1, rel=1e-12)

    @pytest.mark.parametrize("f_dep", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9])
    def test_t_mc_consistent_with_closed_form_f_dep(self, f_dep: float) -> None:
        """``t_mc_from_f_dep`` inverted by ``closed_form_f_dep`` must
        return the input ``f_dep`` to machine precision.
        """
        K = 0.42
        N = 8_000
        t = t_mc_from_f_dep(f_dep, K, N)
        f_recovered = closed_form_f_dep(t, K, N)
        assert f_recovered == pytest.approx(f_dep, rel=1e-12)

    @pytest.mark.parametrize("f_dep", [0.1, 0.5, 0.9])
    def test_epsilon_to_t_ratio_is_1_over_1_minus_f_dep(
        self, f_dep: float
    ) -> None:
        """Equation (3): ``ε / T = 1 / (1 - f_dep)``.

        The headline calibration insight: at large depletion the engine's
        non-depleting-source ε runs much faster than MC's depleting
        physical time.
        """
        K = 1.0
        N = 5_000
        eps = epsilon_from_f_dep(f_dep, K, N)
        t = t_mc_from_f_dep(f_dep, K, N)
        ratio = eps / t
        assert ratio == pytest.approx(1.0 / (1.0 - f_dep), rel=1e-12)

    @pytest.mark.parametrize("f_dep", [0.01, 0.001, 0.0001])
    def test_small_f_dep_limit_epsilon_equals_t(self, f_dep: float) -> None:
        """At small ``f_dep`` the non-depleting correction → 1; ε ≈ T.

        Lim_{f_dep → 0} (ε - T)/T = f_dep + O(f_dep²).  This pins the
        physical interpretation that engine and MC agree when the cascade
        is in its early stages.
        """
        K = 1.0
        N = 5_000
        eps = epsilon_from_f_dep(f_dep, K, N)
        t = t_mc_from_f_dep(f_dep, K, N)
        # Leading-order: ε/T - 1 = f_dep + O(f_dep²).  Allow 2× the
        # leading order as the gate.
        assert abs(eps / t - 1.0) < 2.0 * f_dep

    def test_full_depletion_raises(self) -> None:
        """The closed form diverges at f_dep = 1; the helper must guard."""
        with pytest.raises(ValueError):
            epsilon_from_f_dep(1.0, K_eff=1.0, n_natal=1000.0)
        with pytest.raises(ValueError):
            epsilon_from_mc_state(n_mergers=1000, n_active=0.0, K_eff=1.0)

    def test_zero_mergers_gives_zero_epsilon(self) -> None:
        """No mergers ⇒ no evolution.  Sanity."""
        eps = epsilon_from_mc_state(n_mergers=0.0, n_active=1000.0, K_eff=1.0)
        assert eps == 0.0

    def test_closed_form_n_monotone_decreasing(self) -> None:
        """``N(t)`` from the analytic solution must be strictly
        decreasing in t — the simplest invariant of the Smoluchowski
        equation.
        """
        K, N = 1.0, 5_000.0
        ts = np.linspace(0.0, 10.0, 50)
        Ns = np.asarray([closed_form_total_number(t, K, N) for t in ts])
        assert np.all(np.diff(Ns) < 0.0)
        assert Ns[0] == N  # initial condition
        assert Ns[-1] > 0.0  # never reaches zero in finite time

    def test_input_validation(self) -> None:
        """Negative or zero inputs are programmer errors."""
        with pytest.raises(ValueError):
            epsilon_from_mc_state(-1, 100, 1.0)
        with pytest.raises(ValueError):
            epsilon_from_mc_state(10, 100, 0.0)
        with pytest.raises(ValueError):
            epsilon_from_f_dep(-0.1, 1.0, 100.0)
        with pytest.raises(ValueError):
            t_mc_from_f_dep(0.5, K_eff=0.0, n_natal=100.0)


# ---------------------------------------------------------------------------
# Workstation tier — full MC sweep
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCalibrationSweepWorkstation:
    """Live MC + closed-form parity at the configured ``f_dep`` grid.

    Runs the particle MC once at the constant-K + no-retention limit,
    extracts state at each target ``f_dep``, and checks that the MC's
    ``N_active`` at each stop matches the closed-form Smoluchowski
    ``N(t)`` evaluated at the calibrated time.

    Gate: 5 % rel-L1 per `[[feedback-no-fudge-factors]]` — the PLAN M1
    headline.  This budget is set by MC shot noise at
    ``n_natal = 5000`` particles, not tuned.
    """

    @pytest.fixture(scope="class")
    def sweep(self) -> dict:
        from red_team.mc_time_calibration import run_calibration_sweep

        cfg = CalibrationConfig()  # defaults are the M1 configuration
        return run_calibration_sweep(cfg)

    def test_all_stops_within_gate(self, sweep: dict) -> None:
        """Headline invariant: rel-L1 < 5 % at every ``f_dep`` stop."""
        rel_l1 = sweep["rel_l1_total_number"]
        gate = CalibrationConfig().rel_l1_gate
        bad = np.flatnonzero(rel_l1 > gate)
        if len(bad) > 0:
            details = "\n".join(
                f"  f_dep={sweep['f_dep_mc'][i]:.3f}  rel_l1={rel_l1[i]:.4f}"
                for i in bad
            )
            pytest.fail(
                f"MC vs Smoluchowski deviated above {gate} at:\n{details}"
            )

    def test_eq1_eq5_agree_across_sweep(self, sweep: dict) -> None:
        """Bit-stability invariant: ``epsilon_from_mc_state`` (5) and
        ``epsilon_from_f_dep`` (1) must agree across all MC stops to
        machine precision.  No physics here — pure algebra.
        """
        np.testing.assert_allclose(
            sweep["epsilon_from_eq5"],
            sweep["epsilon_from_eq1"],
            rtol=1e-12,
        )

    def test_epsilon_monotone_increasing(self, sweep: dict) -> None:
        """``ε(f_dep)`` is strictly monotone increasing — guarantees the
        calibration is invertible.  Falls out of the closed form but
        worth pinning so a future regression on the helpers doesn't
        silently destroy invertibility.
        """
        eps = sweep["epsilon_from_eq5"]
        assert np.all(np.diff(eps) > 0.0)

    def test_n_active_matches_closed_form_at_calibrated_t(
        self, sweep: dict
    ) -> None:
        """Per-stop invariant: MC's ``n_active`` at each f_dep target
        equals the closed-form ``N(t_MC)`` to within the rel-L1 gate.

        This is the per-row form of the headline; ``test_all_stops_within_gate``
        is the conjunction.
        """
        for i in range(len(sweep["f_dep_target"])):
            mc = sweep["n_active_at_stop"][i]
            cf = sweep["closed_form_n_at_t"][i]
            rel = abs(mc - cf) / max(cf, 1.0)
            assert rel < CalibrationConfig().rel_l1_gate, (
                f"Row {i}: f_dep={sweep['f_dep_mc'][i]:.3f}  "
                f"mc_n_active={mc:.1f}  closed_form={cf:.1f}  rel={rel:.4f}"
            )
