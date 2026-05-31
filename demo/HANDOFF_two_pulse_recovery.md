# HANDOFF — two-pulse recovery arc (hybrid density → nan fixes → adaptive KDE)

**Date:** 2026-05-31.  **Branch:** `observables`.  **Status:** NOT closed — a
clean checkpoint.  All code py-compiles; JAX-numeric tests H1–H11 pass on the
user's workstation (the Cowork VM has no working JAX install — disk-bound).

## The arc in one paragraph

Started from `demo/PLAN_hybrid_analytic_1g_density.md`: the `mc_jax` two-pulse
fit couldn't recover pulse widths because the J3 fixed-bandwidth Gaussian KDE
oversmooths the natal pulses (`demo/DIAGNOSIS_two_pulse_kde_floor.md`).  Built
the **hybrid analytic-1G + KDE-(2G+) density** (gen==1 evaluated with the exact
parametric natal model, only gen≥2 cascade remnants KDE'd).  Then, to develop on
small machines, added **memory-bounded query chunking** to the KDE.  Then ran the
fit and chased two failures to ground: a **nan in the MAP** (two distinct
root causes, both fixed) and a **σ₂/λ recovery bias** that we proved is the
density estimator, not measurement scatter — and that the **adaptive KDE did
NOT fix**, pointing at implicit as the real cure.

## What shipped (all additive, default-off → legacy paths bit-for-bit)

### 1. Hybrid analytic-1G density — `evaluation/mc_jax.py`, `response/mc_jax.py`
`MCEvaluatorJax` gains `hybrid_1g`: gen==1 weight → exact
`W_1g·f_natal_mass·f_natal_spin`; gen≥2 weight → its own Silverman-bandwidth KDE
(`_f_hat_weighted` + `_analytic_1g_density`).  `∫∫N = W_1g + W_kde = W_total`, so
convention preserved.  `build_mc_jax_response(hybrid_1g=None)` auto-enables for
`natal_mass_kind=="two_gaussian"`.  Tests H1–H5.

### 2. Memory-bounded chunked KDE — `evaluation/mc_jax.py`
`density()` dispatches to `_density_flat` (full `Q×S`) or `_density_chunked`
(`lax.scan` of `query_chunk`-sized blocks, each `_density_flat` wrapped in
`jax.checkpoint` so the BACKWARD pass rematerialises instead of stacking →
peak mem ~`chunk·S` fwd+bwd).  Field `query_chunk` (static), default 512 via
`build_mc_jax_response`.  Two equinox gotchas fixed: (a) can't `jax.checkpoint`
a bound Module method → `eqx.partition`/`combine` so only static fields hash;
(b) chunked is round-off-equal, not bit-identical (XLA reassociates) → H6 uses
`allclose(rtol=1e-11)`.  Tests H6 (invariance), H7 (grad matches flat).

### 3. nan fix #1 — analytic out-of-support grad — `evaluation/mc_jax.py`
`_analytic_1g_density` clamps m/χ to the interior before the truncnorm logpdfs,
then `jnp.where(inside, val, 0)` — `-inf` logpdf has nan grad, which poisoned
reverse-mode for any out-of-support query (high-mass remnant, scattered event).
**Boundary subtlety (caught via H2 failure):** use INCLUSIVE `>=`/`<=` — the
spin truncnorm (loc=0) PEAKS at χ=0 and the mock clips events there; masking the
boundary under-integrated ~6% and drove `natal_spin_scale` to its rail.  Tests
H8; this fix made the spin recover sanely.

### 4. nan fix #2 — log-floor 1/N blow-up — `response/mc_jax.py`
The real MAP-killer (localised with `demo/localize_two_pulse_nan.py`): events in
the KDE-sparse inter-pulse void get N≈0; bare `log(maximum(N,1e-300))` routes
`1/N`→inf into `logsumexp` → `0·inf = nan`, killing the grad for all 9
cascade-coupled params (spares log_rate, additive).  Fix: `_safe_log_floor(x,
floor) = where(x>floor, log(max(x,floor)), log(floor))` (zero grad on the
floored branch); field `density_floor=1e-12`.  Test H9.  After this **the MAP
climbs** (−logL 550→132).

### 5. Adaptive gen≥2 KDE (Abramson) — `evaluation/mc_jax.py` — the science bet
`_pilot_lambda`: pilot fixed-bw density at each gen≥2 sample → per-sample
`λ_i=(f̃_i/G)^{-1/2}` → bandwidth `h·λ_i` (sharp in dense cascade peaks, wide in
tails), mass-preserving (per-sample norm `1/(2π h_log_m h_chi λ_i²)`).  Fields
`lam_kde`, static `kde_adaptive`; threaded through
`build_mc_jax_response`→`_finalize_response_solver`→`build_two_pulse_coagulation`
(`mc_jax_kde_adaptive`)→`make_coagulation_dist_fn` (`kde_adaptive`, two_gaussian
branch only)→`--kde-adaptive` on fit + diagnostic.  Tests H10 (mass-preserving +
sharper peak), H11 (off≡default).

## Diagnostic verdicts (the science state)

- **nan:** root-caused + fixed (both #3 and #4).  MAP now climbs.
- **bias is the DENSITY ESTIMATOR, not scatter:** `diagnose_two_pulse_bias.py
  --sigma-lnm 0 --sigma-chi 0` (delta posteriors) STILL biased — σ₂ shape-peak
  5.48 vs truth 2.0, λ 0.10 vs 0.40.  Deconvolution excluded.
- **bias is n_seed-invariant:** D3 λ shape-peak stuck at 0.170 for n_seed
  400/800/1600 (fixed-bw Silverman floor; `feedback-mise-kernel-not-the-lever`).
- **adaptive KDE did NOT move the peak** (user's last run).  ⚠️ CONFIRM the run
  used `--kde-adaptive` (the D3 helper only enables it via that flag) — if it
  did and the peak still didn't move, the gen≥2 pool is genuinely EMPTY in the
  pulse-2 region (a variable bandwidth widens into a void, can't manufacture
  structure), which is the empirical case to **push implicit forward** (the
  opening breadcrumb's plan) rather than polish the MC estimator further.

## Files this arc

Engine: `src/hierarchical_ad_pop_engine/evaluation/mc_jax.py`,
`response/mc_jax.py`, `response/coagulation.py`, `inference/gwkokab_bridge.py`.
Demos: `demo/{fit_two_pulse_gwkokab,diagnose_two_pulse_bias,
localize_two_pulse_nan,validate_hybrid_1g_density}.py`.
Tests: `tests/test_hybrid_1g_density.py` (H1–H11).
Plans/diagnosis: `demo/{PLAN_hybrid_analytic_1g_density,
DIAGNOSIS_two_pulse_kde_floor}.md`.

## Next session (decision point, not more building)

1. `pytest tests/test_hybrid_1g_density.py -q` — confirm H1–H11 still green.
2. `diagnose_two_pulse_bias.py --kde-adaptive` (±`--sigma-lnm 0`) — does the
   σ₂/λ shape-peak move toward truth vs the fixed-bw baseline?  This is the
   go/no-go for the MC adaptive estimator.
3. If NO (expected from the n_seed-invariance): **pivot to implicit density**
   for the two-pulse fit — the engine's implicit solver already produces a
   grid density with no KDE sparsity; the gwkokab bridge is solver-agnostic
   (`CoagulationResponse.log_rate_density` dispatches), so the main work is
   making the implicit path AD-traceable through the fit (custom-VJP /
   `[[project-engine-v2-production]]`), not new physics.
4. If YES: full `fit_two_pulse_gwkokab.py --kde-adaptive` recovery run.

Memory: `project-two-pulse-fit-bias`, `project-two-pulse-kde-adaptive-recovery`,
`project-hybrid-1g-density-implemented`.
