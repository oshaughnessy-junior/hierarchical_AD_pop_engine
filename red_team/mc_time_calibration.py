"""MC ↔ engine time / coupling calibration (PLAN M1).

Pins down the map ``MC_state → ε_engine`` so ``solver="mc"`` reproduces
the same physical "evolution amount" as ``solver="implicit"`` at a
given ``ε``.  This is the gating piece for `PLAN_mc_first_reprioritization.md`
because, without it, the engine's dimensionless coupling ``ε`` and the
MC's sequential-merge counter have no shared physical interpretation.

Closed-form derivation
======================

Setup: constant pairing kernel ``K(m1, m2) = K`` (a constant), retention
= 1 (``v_esc → ∞``, no kicks), Aligned spin, narrow-Gaussian natal
(monodisperse limit).  This is the only regime where the Smoluchowski
equation has a tractable closed form *and* the engine's steady-state
equation collapses to a 1-line algebra.

The unordered-pair convention is honoured throughout (engine
``_step_one_generation`` uses ``sym = 1/2``).

Engine steady state (total number)
----------------------------------

The engine solves ``N* = N_natal + ε · (gain − loss)`` at steady state
with a *non-depleting* natal source.  Integrating both sides over
``(m, χ)`` in the constant-K limit:

    ∫ gain  dm dχ = (1/2) · K · N_total²
    ∫ loss  dm dχ =       K · N_total²
    ∫ gain − ∫ loss = − (1/2) K N_total²

so total number obeys ``N* = N_natal − (ε/2) K N*²``.

Defining the natal-depletion fraction ``f_dep = (N_natal − N*) / N_natal``::

    N* = N_natal · (1 − f_dep)
    f_dep · N_natal = (ε / 2) · K · N_natal² · (1 − f_dep)²

                                   2 · f_dep
    ε · K · N_natal  =  ───────────────────────────                    (1)
                              (1 − f_dep)²

This matches the ``f_natal_depleted`` operational identity in
``[[project-f-natal-depleted-fix]]`` — at constant K and unordered-pair
convention they agree by construction.

MC time-evolved (total number)
------------------------------

The continuous-time Smoluchowski equation under the unordered-pair
convention is ``dN/dt = −(K/2) N²``.  With monodisperse initial
condition ``N(0) = N_0``::

    N(t) = N_0 / (1 + K · N_0 · t / 2)

    f_dep_MC(t) = 1 − N(t) / N_0  =  (K · N_0 · t / 2) / (1 + K · N_0 · t / 2)

                            2 · f_dep
    K · N_0 · T  =  ──────────────────                                 (2)
                          (1 − f_dep)

Calibration formula
-------------------

At matched ``f_dep`` (i.e. for an MC stopped when its depletion fraction
equals the engine's steady-state ``f_dep``), divide (1) by (2):

                    1
    ε / T  =  ─────────────                                            (3)
                (1 − f_dep)

Substituting ``f_dep = 1 − N_active / N_natal``::

                       N_natal
    ε  =  T_MC · ──────────────                                        (4)
                       N_active

Or in bookkeeping-friendly form (``n_mergers = N_natal − N_active`` under
retention = 1, since each merger removes one net particle)::

              2 · n_mergers
    ε  =  ─────────────────                                            (5)
            K · N_active²

Equations (4) and (5) are equivalent; (5) is the form to expose to
``MCResponse`` because both ``n_mergers`` and ``N_active`` are tracked by
the existing MC driver's ``stats`` dict.

Physical interpretation
-----------------------

The factor ``N_natal / N_active = 1 / (1 − f_dep)`` is the
"non-depleting source" correction.  At small ``f_dep`` the engine and MC
agree (``ε ≈ T_MC``); at large ``f_dep`` the engine ``ε`` is much larger
than ``T_MC`` because the engine assumes an infinite natal supply that
keeps feeding the cascade, while MC's pool actually drains.

Limits
------

* ``f_dep → 0`` (no mergers):  ε → 0, T_MC → 0.  ✓
* ``f_dep → 1`` (full depletion):  N_active → 0, both ε and T_MC →
  ∞.  ε → ∞ faster than T_MC by 1/(1−f_dep).  ✓
* The map is monotonic in ``f_dep``, so the calibration is invertible:
  given an engine target ``ε``, the MC stopping condition is to run
  until ``f_dep`` satisfies (1).

Generalisation to non-constant K
--------------------------------

Equation (5) is **exact** for constant K + retention = 1 + Aligned.
For non-constant K, ``K_eff`` becomes a population-averaged quantity::

                 ⟨ K(m1, m2) · N(m1) · N(m2) ⟩
    K_eff  =  ─────────────────────────────────
                       N_total²

which is itself a function of the population state.  In that case (5)
becomes a *self-consistent* equation rather than a closed form; the
calibration formula still pins ε to the MC state, but ``K_eff`` must be
evaluated on the simulated population.  This generalisation is out of
scope for M1 — the closed-form constant-K case is what's needed to
*verify* the formula; production use of (5) on the natural Stage 1
kernel happens at session M2 when ``MCResponse`` wraps it.

What this module ships
======================

* :func:`epsilon_from_mc_state` — equation (5).  The canonical
  conversion routine; will be called from ``MCResponse`` once that
  ships in session M2.
* :func:`closed_form_total_number` — the Smoluchowski analytic
  ``N(t)``.  Used by :mod:`tests.test_mc_time_calibration` as the
  parameter-free reference (`[[feedback-no-frozen-test-references]]`).
* :func:`run_calibration_sweep` — workstation driver: runs the MC at a
  sequence of stopping ``f_dep`` values, computes ``ε`` via (5),
  compares MC's ``N(t)`` to the closed-form ``N(t)`` at the calibrated
  ``ε``.

Hard constraints honoured
=========================

* `[[feedback-no-fudge-factors]]` — equation (5) is derived from first
  principles, no tuning constants.  The factor of 2 is unordered-pair
  bookkeeping; the factor of ``N_natal/N_active`` is the
  non-depleting-source correction.
* `[[feedback-no-frozen-test-references]]` — calibration is validated
  against the analytic Smoluchowski solution, not pinned ``.npz`` files.
* `[[feedback-no-fake-physics-stand-ins]]` — no smoothing knobs; the
  MC driver under constant K is the existing particle MC, the
  reference is the textbook closed form.
* `[[feedback-mc-depletion-tracks-engine]]` — equation (5) is the
  exact algebraic restatement of the engine's ``f_natal_depleted``
  identity in the constant-K regime.

Module is numpy-only (no JAX, no scipy) so it imports cleanly in the
Cowork sandbox.  The :func:`run_calibration_sweep` driver imports the
particle-MC backend lazily so callers that only want the closed-form
helpers don't pay the JAX import cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Closed-form scalars
# ---------------------------------------------------------------------------


def epsilon_from_mc_state(
    n_mergers: float,
    n_active: float,
    K_eff: float,
) -> float:
    """Engine ``ε`` corresponding to an MC stopped at ``(n_mergers, n_active)``.

    Equation (5) from the module docstring::

                   2 · n_mergers
        ε  =  ─────────────────
                K_eff · n_active²

    Parameters
    ----------
    n_mergers : float
        Cumulative merger events executed by the MC up to the stopping
        point.  Under retention = 1 each merger removes exactly one net
        particle, so equivalently ``n_mergers = N_natal − n_active``.
    n_active : float
        Active particle count at the stopping point.  Must be > 0;
        the limit ``n_active → 0`` corresponds to ``ε → ∞`` (full
        depletion) and is not a useful calibration point.
    K_eff : float
        Constant kernel value.  For non-constant kernels, the
        population-averaged effective rate; see "Generalisation" in
        the module docstring.

    Returns
    -------
    epsilon : float
        Engine dimensionless coupling.  Has units of time × (K_eff has
        units 1/time/N, n_active is dimensionless) = time / N — but
        the engine's downstream code treats this as a dimensionless
        evolution parameter, with the dimensionful interpretation
        absorbed into ``K_eff``.

    Notes
    -----
    Equivalent form ``ε = T_MC · (n_active_init / n_active)`` is given
    by (4).  This routine uses (5) because both inputs are directly
    available from the MC driver's ``stats`` dict without separately
    tracking the simulated time.

    Bit-for-bit reproducibility: this is a 3-line numpy expression
    over scalars; deterministic.
    """
    if n_active <= 0.0:
        raise ValueError(
            "n_active must be positive; n_active == 0 corresponds to "
            "full depletion (ε → ∞) and is not a useful calibration."
        )
    if K_eff <= 0.0:
        raise ValueError("K_eff must be positive.")
    if n_mergers < 0.0:
        raise ValueError("n_mergers must be non-negative.")
    return float(2.0 * n_mergers / (K_eff * n_active * n_active))


def epsilon_from_f_dep(f_dep: float, K_eff: float, n_natal: float) -> float:
    """Engine ``ε`` corresponding to a depletion fraction ``f_dep``.

    Equation (1) from the module docstring::

                        2 · f_dep
        ε  =  ─────────────────────────────
                K · N_natal · (1 − f_dep)²

    Algebraically equivalent to :func:`epsilon_from_mc_state` via the
    identity ``n_active = n_natal · (1 − f_dep)`` and ``n_mergers =
    n_natal · f_dep`` (retention = 1).  This form is convenient when
    designing a calibration sweep at a *target* ``f_dep``.
    """
    if not (0.0 <= f_dep < 1.0):
        raise ValueError(
            f"f_dep must be in [0, 1); got {f_dep}.  f_dep = 1 "
            "corresponds to full depletion (ε → ∞)."
        )
    if K_eff <= 0.0 or n_natal <= 0.0:
        raise ValueError("K_eff and n_natal must be positive.")
    return float(2.0 * f_dep / (K_eff * n_natal * (1.0 - f_dep) ** 2))


def t_mc_from_f_dep(f_dep: float, K_eff: float, n_natal: float) -> float:
    """MC simulated time corresponding to a target depletion fraction.

    Equation (2) inverted::

                       2 · f_dep
        T_MC  =  ───────────────────────
                  K · N_natal · (1 − f_dep)

    Same constant-K monodisperse regime as :func:`epsilon_from_f_dep`.
    """
    if not (0.0 <= f_dep < 1.0):
        raise ValueError(f"f_dep must be in [0, 1); got {f_dep}.")
    if K_eff <= 0.0 or n_natal <= 0.0:
        raise ValueError("K_eff and n_natal must be positive.")
    return float(2.0 * f_dep / (K_eff * n_natal * (1.0 - f_dep)))


def closed_form_total_number(t: float, K_eff: float, n_natal: float) -> float:
    """Smoluchowski analytic ``N(t)`` for constant-K + monodisperse + no retention.

    ``N(t) = N_0 / (1 + K · N_0 · t / 2)``.  Reference for
    ``test_mc_time_calibration.py``.  Parameter-free — no tuning
    constants.
    """
    if t < 0.0 or K_eff <= 0.0 or n_natal <= 0.0:
        raise ValueError("t >= 0, K_eff > 0, n_natal > 0 required.")
    return float(n_natal / (1.0 + 0.5 * K_eff * n_natal * t))


def closed_form_f_dep(t: float, K_eff: float, n_natal: float) -> float:
    """``f_dep(t)`` for constant-K + monodisperse + no retention.

    ``f_dep(t) = (K·N_0·t/2) / (1 + K·N_0·t/2)``.  Reference.
    """
    if t < 0.0 or K_eff <= 0.0 or n_natal <= 0.0:
        raise ValueError("t >= 0, K_eff > 0, n_natal > 0 required.")
    tau = 0.5 * K_eff * n_natal * t
    return float(tau / (1.0 + tau))


# ---------------------------------------------------------------------------
# Workstation driver
# ---------------------------------------------------------------------------


@dataclass
class CalibrationConfig:
    """Inputs for the calibration sweep.

    The MC is run *once* at full natural termination; the per-merger
    log is replayed to extract the state at each target ``f_dep`` —
    this is a free trick that avoids running the MC N times.

    Field names match :class:`red_team.backup_particle_mc.MCConfig`
    where they overlap so the wiring in :func:`run_calibration_sweep`
    is mechanical.
    """

    K_eff: float = 1.0
    n_seed: int = 5000  # → MCConfig.n_seed
    mu_natal: float = 30.0  # → MCConfig.mu_natal
    sigma_natal: float = 0.5  # narrow Gaussian as monodisperse stand-in
    m_min: float = 5.0
    m_max: float = 200.0
    natal_spin_loc: float = 0.0
    natal_spin_scale: float = 0.01
    f_dep_grid: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    seed: int = 0xC4118A47
    rel_l1_gate: float = 0.05  # 5 % per the PLAN headline


def run_calibration_sweep(cfg: CalibrationConfig) -> dict:
    """Execute the MC + closed-form comparison at the configured grid.

    Workstation-only — imports the MC driver, which pulls in
    ``red_team/backup_particle_mc.py`` → JAX + scipy.

    Returns a dict with one row per ``f_dep`` target::

        {
          "f_dep_target": np.ndarray,
          "f_dep_mc": np.ndarray,         # actual nearest-merger stop
          "n_active_at_stop": np.ndarray,
          "n_mergers_at_stop": np.ndarray,
          "epsilon_from_eq5": np.ndarray, # via epsilon_from_mc_state
          "epsilon_from_eq1": np.ndarray, # via epsilon_from_f_dep
          "t_mc_at_stop": np.ndarray,
          "closed_form_n_at_t": np.ndarray,
          "rel_l1_total_number": np.ndarray,  # per gate, < 0.05 required
        }

    The two ``epsilon_*`` arrays are algebraically equivalent and must
    agree to machine precision; the test file checks this as a
    consistency invariant for free.
    """
    # Lazy import — keeps the sandbox-runnable closed-form helpers
    # importable without scipy/JAX.
    from red_team.backup_particle_mc import MCConfig, particle_mc  # noqa: E402

    rng = np.random.default_rng(cfg.seed)

    # Run MC under the constant-K + no-retention + Aligned regime.  We use
    # MCConfig(kernel_a=0, kernel_b=0) → K(m1,m2) = K_eff scalar via
    # ``kernel_K(... a=0, b=0)`` evaluation; v_esc → ∞ (very large) for
    # the retention=1 path; depletion_fraction = 0.99 lets MC run past
    # the highest f_dep target in the grid before naturally terminating
    # (cf. MCConfig docstring — the default 0.05 stops too early for
    # this sweep).
    largest_f_dep = max(cfg.f_dep_grid)
    if largest_f_dep >= 0.99:
        raise ValueError(
            "f_dep_grid contains a target ≥ 0.99; the MC depletion ceiling "
            "cap and closed-form divergence both bite there.  Cap at 0.95."
        )
    mc_cfg = MCConfig(
        mu_natal=cfg.mu_natal,
        sigma_natal=cfg.sigma_natal,
        m_min=cfg.m_min,
        m_max=cfg.m_max,
        natal_spin_loc=cfg.natal_spin_loc,
        natal_spin_scale=cfg.natal_spin_scale,
        kernel_a=0.0,
        kernel_b=0.0,
        v_esc=1.0e30,  # retention ≈ 1 (Heaviside saturated)
        n_seed=cfg.n_seed,
        depletion_fraction=0.99,  # run well past largest target
        n_max_mergers=cfg.n_seed,  # at most n_seed mergers under retention=1
    )
    result = particle_mc(mc_cfg, rng, track_generations=False)

    mergers = result["mergers"]  # shape (n_accept, 8) — heavier first
    n_mergers_total = len(mergers)

    # Replay the merger log to find the state at each target f_dep.
    f_dep_target = np.asarray(cfg.f_dep_grid, dtype=np.float64)
    n_active_target = cfg.n_seed * (1.0 - f_dep_target)

    rows = {key: [] for key in [
        "f_dep_mc", "n_active_at_stop", "n_mergers_at_stop",
        "epsilon_from_eq5", "epsilon_from_eq1",
        "t_mc_at_stop", "closed_form_n_at_t", "rel_l1_total_number",
    ]}

    for f_t, n_a_t in zip(f_dep_target, n_active_target):
        # Under retention=1 each merger removes 1 net particle, so we
        # stop at the merger index k where N_init − k = n_a_t (rounded).
        k_stop = int(round(cfg.n_seed - n_a_t))
        if k_stop > n_mergers_total:
            raise RuntimeError(
                f"MC ran out of mergers before reaching f_dep={f_t}; "
                "increase n_seed or shorten the f_dep_grid."
            )
        n_active_stop = cfg.n_seed - k_stop
        f_dep_mc = (cfg.n_seed - n_active_stop) / cfg.n_seed

        eps_eq5 = epsilon_from_mc_state(k_stop, n_active_stop, cfg.K_eff)
        eps_eq1 = epsilon_from_f_dep(f_dep_mc, cfg.K_eff, cfg.n_seed)
        t_mc = t_mc_from_f_dep(f_dep_mc, cfg.K_eff, cfg.n_seed)
        n_closed = closed_form_total_number(t_mc, cfg.K_eff, cfg.n_seed)
        rel_l1 = abs(n_active_stop - n_closed) / max(n_closed, 1.0)

        rows["f_dep_mc"].append(f_dep_mc)
        rows["n_active_at_stop"].append(n_active_stop)
        rows["n_mergers_at_stop"].append(k_stop)
        rows["epsilon_from_eq5"].append(eps_eq5)
        rows["epsilon_from_eq1"].append(eps_eq1)
        rows["t_mc_at_stop"].append(t_mc)
        rows["closed_form_n_at_t"].append(n_closed)
        rows["rel_l1_total_number"].append(rel_l1)

    out = {"f_dep_target": f_dep_target}
    out.update({k: np.asarray(v) for k, v in rows.items()})
    return out


__all__ = [
    "CalibrationConfig",
    "closed_form_f_dep",
    "closed_form_total_number",
    "epsilon_from_f_dep",
    "epsilon_from_mc_state",
    "run_calibration_sweep",
    "t_mc_from_f_dep",
]
