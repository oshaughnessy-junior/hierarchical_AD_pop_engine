# Plan — synthetic-data integration test (round-trip recovery)

A pytest gate for the end-to-end inference pipeline. Bridges the
unit-test world (the bug-finding scaffolding built out by the Phase
A/B/C red-team work) and the end-to-end story already running
informally inside `demo/demo_gwkokab_fit_stage1.py`.

## Why this exists

The unit + parity tests answer "is each piece correct?".  This test
answers "do they fit together into a working inference pipeline?".
Specifically: given a synthetic catalogue drawn from a known ``θ*``,
does the SVI inference recover ``θ*`` inside its credible interval?

The demo answers the same question every time someone runs it by hand
— but there's no gate, no regression alarm if a numpyro upgrade
breaks `make_inference_model`, or a refactor to `build_response`
silently shifts the rate normalisation.  The integration test makes
the recovery story automatically reproducible.

## Scope (v1)

* **Stage 1 model only.**  Single Gaussian natal pulse, flat selection,
  6 free parameters: ``mu_natal``, ``sigma_natal``,
  ``natal_spin_loc``, ``natal_spin_scale``, ``kernel_b``,
  ``log_rate``.  Matches `PLAN_gwkokab_fit.md` §Stage 1 and reuses the
  demo's helpers verbatim.
* **Engine-vs-engine consistency.**  The synthetic catalogue is drawn
  from the engine's *own* forward model (not the particle MC).  This
  pins that the inference pipeline is internally consistent — given
  a correctly-implemented forward model, can SVI recover the
  injected ``θ*``?  Cross-algorithm inference (MC-synthesised data,
  engine recovery) waits for the v3 basis refactor to close
  ``project_implicit_vs_mc_systematic_bias`` — until then the open
  systematic would dominate the recovery diagnostic.
* **SVI only.**  NUTS is too slow for routine CI (minutes per chain,
  multi-chain by design) and its value is diagnostic richness, not
  gate strength.  The demo retains the NUTS path; the test gates on
  SVI only.

## What this plan deliberately does NOT do

* No NUTS.  Demo keeps it; test does not.
* No MC-synthesised data.  Waits on v3 systematic closure.
* No Stage 2 / Stage 3 recovery.  Those are separate tests once
  `PLAN_gwkokab_fit.md` stages 2 and 3 actually ship.
* No selection-function inference.  Flat selection per Stage 1.
* No frozen posterior reference (per
  `feedback_no_frozen_test_references`).  The acceptance gate is the
  invariant "truth inside credible interval", not pinned numbers.
* No multi-seed averaging.  A single deterministic JAX PRNG key for
  the mock catalogue and a single deterministic key for SVI.  Flaky
  recovery under a single seed is a real signal — multi-seed
  smoothing would mask it.

## File layout

```
demo/PLAN_integration_test_synthetic.md          # this file
demo/demo_gwkokab_fit_stage1.py                  # unchanged — imported by tests
tests/test_integration_stage1_recovery.py        # NEW — slow-tier
```

The demo file is *not* modified.  The test imports
`build_response`, `expected_event_count`, `sample_mock_catalog`,
`make_inference_model`, `run_svi`, and the parameter / config
constants directly from the demo via `sys.path` manipulation (the
demo lives in `demo/`, not in `src/`).  Demo refactoring is
explicitly out of scope — if it turns out the demo grows in a way
that breaks reuse, the test gets pinned to its own copy of the
helpers at that point.

## Acceptance

The test runs once per session under `pytest -m slow` and asserts:

1. **All 6 truth values inside the SVI 95 % credible interval.**
   For each parameter, compute the empirical 2.5 % and 97.5 %
   quantiles from the SVI posterior draws (1000 draws from the
   fitted ``AutoNormal`` guide).  Assert
   ``q_2.5 ≤ θ*_param ≤ q_97.5``.  The 95 % CI is looser than the
   demo's 90 % because SVI's mean-field approximation can
   underestimate posterior width; 95 % gives headroom against that
   while still failing on any parameter that's truly mis-recovered.
   *Hard gate — every param must pass.*
2. **SVI ELBO improves over training.**  Compare the median ELBO of
   the first 10 % of steps against the last 10 %.  Final must be at
   least 1 nat below initial.  Catches "SVI didn't converge / got
   stuck at init" failure mode.  *Hard gate.*
3. **Expected count at truth matches sample size.**
   ``expected_event_count(θ_truth, v_esc) ≈ N_OBS`` within
   ``3 · sqrt(N_OBS)`` Poisson noise (default ``N_OBS=100`` ⇒
   tolerance 30).  Catches rate-normalisation drift in
   `build_truth_model_and_calibrate_rate`.  *Hard gate.*

## Stringency conventions

* **Wall-clock budget**: ≤ 3 min on a workstation CPU.  Knobs:
  ``N_OBS=100``, ``SVI_STEPS=3000``, ``N_M_FIT=64``, ``N_CHI_FIT=20``,
  ``K_MAX_FIT=4``.  Matches the demo defaults.
* **Determinism**: one fixed JAX PRNG key for mock catalogue, one
  fixed JAX PRNG key for SVI initialisation.  Bit-reproducible
  across runs on the same machine (modulo JAX backend re-tracing
  noise, which is below the 95 % CI tolerance).
* **Per the no-fudge-factors rule**: the CI percentages (2.5, 97.5)
  and the ELBO-improvement minimum (1 nat) are set once and never
  tuned per run.  A flaky test under these defaults is a real signal
  about the inference pipeline, not an excuse to widen the window.

## Decisions log (2026-05-21)

1. **Engine-vs-engine for v1.**  MC-vs-engine inference waits for v3
   systematic closure.
2. **SVI only.**  NUTS optional in demo; not gated in CI.
3. **N_OBS = 100, SVI_STEPS = 3000.**  Same as demo defaults; the
   test is a thin wrapper around the demo's existing entry points.
4. **95 % CI per param** instead of demo's 90 % — headroom against
   SVI mean-field width underestimate.
5. **Single-seed determinism.**  No averaging across seeds; flaky
   recovery is a real failure signal.
6. **No frozen posterior references.**  Per
   `feedback_no_frozen_test_references`.  Acceptance is invariant
   (truth-in-CI), not number-pinned.

## Open follow-ups (not v1)

* **MC-synthesised inference.**  Once the v3 basis closes the
  implicit-vs-MC systematic, add `tests/test_integration_stage1_mc_recovery.py`
  with the same model but synthetic data from the particle MC.
* **Stage 2 + Stage 3 recovery tests.**  Once Stage 2 (two pulses +
  SNR(m) selection) and Stage 3 (full PL+peak + redshift) demos
  ship, lift each into its own integration test using the same
  scaffolding.
* **NUTS gate.**  If SVI mean-field starts under-covering after
  Stage 2 lands (two-pulse posteriors are bimodal; AutoNormal can't
  represent that), add a separate slow-tier NUTS test with a tighter
  wall-clock budget (single chain, ~200 warmup, ~500 samples).

## What the test catches that unit tests don't

The full pipeline is "natal IMF + retention + kernel + cascade +
likelihood + SVI optimiser".  Each piece has unit-test coverage; the
integration test is what catches:

* numpyro guide / SVI API changes that break the optimiser.
* Silent rate-normalisation drift between `build_response` and
  `expected_event_count` (the two pieces share `log_R_env` plumbing;
  a refactor to one without the other surfaces here).
* AD breakage at the integration point (a non-trivial gradient path
  that doesn't show up in finite-difference unit tests because the
  path runs through `numpyro.factor`).
* Likelihood / prior-bound interactions (an over-tight prior would
  clip the posterior and fail the CI gate; the gate then makes the
  diagnostic obvious).

## Effort estimate

~0.5 session.  All the heavy lifting (truth build, mock catalogue,
inference model, SVI driver) is already in
`demo/demo_gwkokab_fit_stage1.py`; the test is the assertion harness
and the deterministic-seed plumbing on top.
