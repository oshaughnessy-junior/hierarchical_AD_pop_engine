# PLAN — Monte Carlo as a first-class JAX-wired likelihood path

Written 2026-05-27 in response to the user-reported timing + memory
collapse of the implicit solver under angular-marginalisation (see
`[[project-session-handoff-20260527-late]]` and this PLAN's sibling
`PLAN_implicit_footprint_refactor.md`).

## Why MC is being promoted

The implicit Picard solver works at production resolution under
**Aligned** orientation, but under any non-Aligned model the
orientation quadrature multiplies cost on both sides of every Picard
iteration. The 8× compute speedup landed in
`[[project-orientation-quadrature-defaults-flip]]` shaved the constant
but did not change the asymptotic — even at `(n_θ, n_Δφ) = (4, 8)` a
single solve still blows past the workstation memory ceiling for the
production grid the user actually wants to run for gwkokab inference.

The particle MC backend `red_team/backup_particle_mc.py` (Session A/C
vintage; `[[project-misalignment-session-a-done]]`,
`[[project-misalignment-session-c-done]]`) was written as a red-team
validation backstop, **not** an inference path. Promoting it to a
first-class JAX-wired likelihood requires resolving four obstacles the
user named, in priority order:

| Obstacle | Severity | Fix surface |
|----------|----------|-------------|
| (c) Single-event likelihoods in gwkokab — MC gives samples; `gwkokab.PoissonLikelihood` wants `log_prob(θ_event)` per detected event | High — gates the inference wire-up | Wrap `red_team/evaluation_mc.MCEvaluator` in a `Distribution` shim paralleling `CoagulationPopulation` |
| (b) "Time" meaning — the MC sequential-merge process needs explicit time scaling so `ε` corresponds to a definite simulated time | High — gates physical interpretability of any MC fit | One-time calibration sweep against closed-form Smoluchowski (constant kernel, no retention) — establish `ε(MC_state)` map |
| (a) Cascade resolution — narrow 1G peaks under-resolved at finite particle count, especially the natal-χ peak | Medium — controllable via N_particles | Stratified sampling on the natal distribution; importance-weighted cascade |
| (d) Narrow-posterior hit-rate — chirp-mass observations have narrow posteriors; blind MC sampling won't deposit enough particles in the narrow region of (m1, m2, χ_eff) | Low if scope is restricted to broad-posterior BBHs (user-confirmed restriction); High at full scope | Phase 2 only: importance-sampled MC conditioned on the per-event posterior |

(d) is by far the hardest to do well; (a) and the broad-posterior
subset of (d) share a fix surface (importance sampling).

## Hard constraints honoured

* `[[feedback-no-fake-physics-stand-ins]]` — MC continues to run real
  vector BR + LZ + per-event Heaviside retention. No smoothing knobs
  dressed as physics.
* `[[feedback-no-fudge-factors]]` — every bandwidth / particle-count
  / IS-proposal width must be discovered from data or set by analytic
  convergence argument, never tuned to a target.
* `[[feedback-mc-depletion-tracks-engine]]` — MC `depletion_fraction`
  continues to be read from the engine's `f_natal_depleted`, never
  hardcoded.
* `[[feedback-no-frozen-test-references]]` — every parity gate is an
  invariant or engine-vs-MC relationship, no pinned `.npz` baselines.
* `[[feedback-perturbative-unreliable-implicit-production]]` —
  implicit remains the default solver until the MC path passes parity
  on the full Stage 1 fixture. MC ships **additively** as a second
  `solver=` value on `CoagulationResponse`, not in-place.
* `[[feedback-dispatch-callgraph-audit]]` — every new kwarg on
  `coagulation_model` / `CoagulationPopulation` is followed by a grep
  through all callers; positional-arg leaks are the documented Session
  C failure mode.
* `[[feedback-mise-kernel-not-the-lever]]` — when MCEvaluator's
  bandwidth misses its rel-L1 gate, change the bias-variance regime
  (adaptive k, stratification), not the kernel.

## Architecture

### Current state (start of plan)

```
gwkokab.PoissonLikelihood
    ↓ log_prob(θ_event)
distributions.CoagulationPopulation
    ↓ log_rate_density(...)
universe.SyntheticUniverse
    ↓ response × formation × convolution
response/coagulation.CoagulationResponse  ← solver="implicit" (production)
                                            solver="perturbative" (legacy)
```

`red_team/backup_particle_mc.py:particle_mc()` is a numpy driver with
JAX leaves at the physics fits. It returns a `mergers` 8-column ndarray
+ `v_kicks` 1-D array. **It does not satisfy the `ResponseFunction`
protocol**, has no `log_rate_density` method, and is not pytree.

### Target state

```
gwkokab.PoissonLikelihood
    ↓ log_prob(θ_event)
distributions.CoagulationPopulation        ← unchanged surface
    ↓ log_rate_density(...)
universe.SyntheticUniverse                 ← unchanged
    ↓
response/coagulation.CoagulationResponse   ← solver="implicit" | "basis" | "mc"
                                              ALL three return a Response
                                              that exposes the same evaluator
                                              protocol
                            ↓
                       evaluation/mc.MCEvaluator (promoted from red_team/)
                            ↓ rate_density_4d, total_rate, etc.
```

The MCResponse owns:
* an inner driver that runs the particle MC at a chosen "time"
  (Step 2 below) and a chosen particle count (Step 4),
* a stash of the merger record array,
* an `evaluator(self)` factory that returns the existing
  `MCEvaluator` (already shipped — see
  `[[project-evaluation-step3-done]]`).

The MCResponse's `log_rate_density` query path goes:

  `(m1, m2, χ1, χ2, z) → MCEvaluator.rate_density_4d(...) → log`

i.e. exactly the protocol that `CoagulationPopulation.log_prob` already
calls into via `universe.log_rate_density`. **No change** to the
gwkokab-facing API.

## Sessions

### Session M1 — Time calibration (obstacle b)

Goal: pin down the map `MC_state → ε` so `solver="mc"` reproduces the
same physical "evolution amount" as `solver="implicit"` at a given `ε`.

Approach: constant pairing kernel `K = const`, retention = 1
(`v_esc = ∞`), no spin, single mass bin → Smoluchowski-with-constant-
kernel has a closed-form solution (Scott 1968; cluster-mass moment
recursion). MC is run at a sequence of stopping fractions
`f_dep ∈ {0.1, 0.2, ..., 0.9}` and compared to the closed-form
`N(t)` at the corresponding ``ε(t)``. The result is a one-line
formula `ε ↔ (1 − N_active / N_init)` valid for any kernel, derived
once.

Deliverables:
* `red_team/mc_time_calibration.py` — driver + closed-form comparison.
* `demo/figures_mc_time/` — calibration curve.
* `tests/test_mc_time_calibration.py` — invariant: at every
  `f_dep` from the calibration grid, MC `N(m)` matches closed-form
  Smoluchowski to within 5% rel L1.
* Single equation `ε = ε(f_dep, K_eff, T_obs)` added to the
  `MCResponse` class docstring.

Effort: 1 session.

### Session M2 — MCResponse + Distribution shim (obstacle c)

Goal: ship `response.mc.MCResponse` that exposes the same
`evaluator()` factory pattern as `BasisSolveResult` /
`ImplicitSolveResult`. Wire `coagulation_model(..., solver="mc")` and
`coagulation_model_delta_formation(..., solver="mc")` to route through
it. Confirm `CoagulationPopulation.log_prob` lights up under
`solver="mc"`.

Deliverables:
* `src/hierarchical_ad_pop_engine/response/mc.py` — new `MCResponse`
  eqx.Module wrapping the MC driver, with `.evaluator()` returning
  the existing `MCEvaluator` from `red_team/evaluation_mc.py`
  (promote that file to `src/.../evaluation/mc.py` at this point —
  Evaluation Step 3 had it red-team-namespaced because it was a
  validation backstop; promotion is the canonical move).
* `distributions.coagulation_model` and
  `distributions.coagulation_model_delta_formation` gain a
  `solver: Literal["implicit", "basis", "mc"]` kwarg routed through
  to `build_default_coagulation` / `CoagulationResponse`.
* `CoagulationResponse` gains optional MC fields plumbed at
  construction; `solver="mc"` stores the merger array on the
  response.
* `tests/test_mc_distribution_log_prob.py` — invariant: under the
  default Stage 1 θ\*, `solver="mc"` and `solver="implicit"`
  agree on `log_prob` at a small set of `(m1, m2, χ1, χ2, z)`
  query points to within `MCEvaluator`'s headline 10 % rel-L1
  budget (`[[project-evaluation-step3-done]]`). One bitwise-Aligned
  check guards the dispatch.

**Hard gotcha**: MCResponse is **not** jit-traceable under standard
patterns because the MC driver is numpy + scipy.truncnorm. The
gradient path goes through the `MCEvaluator`'s k-NN density (also
numpy) — *only* `log_prob` queries are JIT/grad-traceable
post-construction. We document explicitly that `solver="mc"` is for
inference where the population parameters that change *between*
likelihood evaluations are the natal / kernel parameters (which
re-trigger the MC driver via Python-time eqx control flow), while
event-level `(m1, m2, χ1, χ2, z)` queries inside one likelihood
evaluation are traceable.

For MCMC: every new θ-sample retriggers an MC driver run, then the
likelihood loop over events is fast. This is the same usage pattern
as e.g. NRSur waveforms — a Python-time switch on the parameter
update, traceable inside one likelihood evaluation.

Effort: 1 session (medium — straightforward plumbing + one new test
file + one big architectural ASCII diagram in the new module's
docstring).

### Session M3 — Cascade resolution (obstacle a)

Goal: address narrow-natal-peak under-resolution by stratified
natal-sampling.

Approach: instead of `truncnorm.rvs(size=N_init)` and accepting
whatever (m, χ) distribution that gives, use an importance-sampled
draw with stratification on `(m, χ)` cells of the engine grid. Each
particle carries a weight `w_i = p_natal(m_i, χ_i) / q_proposal(m_i,
χ_i)`. The merger driver propagates weights forward; descendant
particles inherit the appropriate weight product. At density-
estimation time the `MCEvaluator` weights its k-NN kernel by `w_i`.

This is well-trodden statistical territory (the k-NN density on
unweighted samples in `red_team/evaluation_mc.py` extends to the
weighted case in one line; cf. Loftsgaarden-Quesenberry 1965
weighted variant).

Deliverables:
* `red_team/backup_particle_mc.py` gains `MCConfig.proposal` and
  `MCConfig.stratify` arguments. Default `proposal=None` preserves
  v1 behaviour bit-for-bit.
* `red_team/evaluation_mc.py::MCEvaluator.__init__` gains a `weights`
  array; default `None` preserves Step 3 behaviour bit-for-bit
  (`[[project-evaluation-step3-done]]`).
* `tests/test_mc_stratified_natal.py` — invariant: under a stratified
  natal proposal with `N_init = 10^4` particles, the MCEvaluator's
  recovery of a narrow natal peak (σ_χ = 0.02) is at most 2× worse
  than the implicit solver's recovery, measured by rel-L1. This is
  the gate. The number 2× is *not* tuned — it's the worst case under
  Monte-Carlo SE at this `N_init`.

Effort: 1 session.

### Session M4 — Observable-layer parity under MC inference (validation)

Goal: confirm MC-based inference recovers Stage 1 θ\* (in the sense
of `tests/test_integration_stage1_recovery.py`) within the same
priors and tolerances as the implicit-based inference.

Deliverable: `tests/test_integration_stage1_recovery_mc.py` — exact
parallel of the existing integration test, with
`solver="mc"`. Runtime budget: same as the implicit test (truth-in-CI
gate; xfail if MC noise pushes beyond the existing σ_natal / kernel_b
tolerances and document the regime where MC is preferred to
implicit).

Effort: 1 session, possibly two if recovery fails and obstacle (a)
needs additional N_init scaling.

### Session M5 (deferred — obstacle d, narrow posteriors)

**Out of scope for v1 MC promotion.** Plan but do not implement.
Importance-sampled MC conditioned on per-event posteriors —
each detected GW event's narrow `(M_chirp, q, χ_eff)` posterior
becomes an additional importance distribution; the MC driver
draws progenitors enriched in cells likely to land in that
posterior. Returns to the standard MC sample under "broad"
posteriors automatically (importance ratio → 1).

Required only when the inference dataset has many narrow-posterior
events. User-noted: focusing on broad-posterior BHs makes this
manageable; full LVK catalogue would need it.

## Cross-cutting decisions to lock at Session M1

* **MC driver lives in `red_team/` or `src/`?** Plan calls for
  promotion in M2. Until then, leave it red-teamed.
* **Single MC driver run per likelihood call?** Yes — MCMC's
  population-parameter step retriggers; event-level loop is fast.
* **Bandwidth selection — global k or adaptive k for MCEvaluator?**
  Step 3 landed on global k = 500 with biweight (10 % headline rel
  L1; `[[project-evaluation-step3-done]]`). Adaptive k is the
  natural next move if M4's recovery test fails; deferred to M3 if
  needed.

## Estimated calendar

| Session | Topic                                         | Risk |
|---------|-----------------------------------------------|------|
| M1      | Time calibration vs Smoluchowski              | Low  |
| M2      | MCResponse + Distribution shim                | Med  |
| M3      | Stratified natal + weighted MCEvaluator       | Med  |
| M4      | Stage 1 integration test under solver="mc"    | Med  |
| M5      | Narrow-posterior IS                            | Deferred |

Four sessions to a production-grade MC-based inference path.

## Why not give up on implicit and ship MC only

Two reasons:

1. **AD on natal / kernel parameters is cleaner under implicit.**
   The implicit Picard solver has a clean custom-VJP via IFT
   (`response/implicit.py:_make_picard_solve_fn`). The MC path
   gradient route is through the k-NN MCEvaluator + the parameter
   re-trigger pattern — workable but lossier in HMC.
2. **Aligned implicit is fast and shipped.** No reason to deprecate.
   The cost problem is specifically the orientation-quadrature inner
   loop under non-Aligned (`PLAN_implicit_footprint_refactor.md`).

The two streams cover orthogonal regimes: implicit for Aligned +
sharp-AD-on-natal; MC for misaligned + large N + observable-layer
flexibility.

## Related

* `[[project-session-handoff-20260527-late]]` — the timing /
  memory pain that motivated this prioritisation.
* `[[project-evaluation-step3-done]]` — the MCEvaluator k-NN
  density that becomes MC's `log_prob` backbone.
* `[[project-misalignment-session-c-done]]` — the existing MC
  misaligned wire-up; this PLAN takes that as starting point.
* `[[project-orientation-quadrature-defaults-flip]]` — the
  8× implicit speedup that was insufficient.
* `[[feedback-perturbative-unreliable-implicit-production]]` —
  the "additive, not in-place" rule applied to MC promotion.
* `PLAN_implicit_footprint_refactor.md` — the companion plan
  for the implicit side.
