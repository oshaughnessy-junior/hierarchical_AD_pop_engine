# HANDOFF (mid-session) — implicit pivot wired + sampler-bias root-caused

**Date:** 2026-06-01. **Branch:** `observables`. **Status:** mid-session
checkpoint; all code py-compiles, workstation-validated where noted. Follows
`HANDOFF_two_pulse_recovery.md`.

## The arc in one paragraph

Pivoted the two-pulse fit off the `mc_jax` hybrid KDE onto the **implicit grid
solver** (the KDE oversmoothing was unfixable; `HANDOFF_two_pulse_recovery.md`).
Wired `solver="implicit"` through the gwkokab bridge, then chased the residual
λ/σ "bias" to ground — and it turned out **not to be the implicit solver at
all**. It was the self-consistent **mock sampler** broadening events with
coarse-grid jitter: invisible against the smooth `mc_jax` KDE, fatal against the
sharp implicit density. Fixing the sampler vindicated implicit. Conclusion:
go **cross-method** — inject with MC (no grid-resolution bias), recover with
implicit (no KDE smoothing). Plan written.

## What shipped this session (all additive; `mc_jax` default preserved)

### 1. Implicit through the gwkokab bridge — `inference/gwkokab_bridge.py`
`make_coagulation_dist_fn` gains `solver="mc_jax"` (default, back-compat) +
`epsilon=1.0`, threaded into both gaussian / two_gaussian builder calls.
`_wrap`'s `total_rate_fn` is now solver-aware: MC paths use
`mc_*_response.total_rate`; grid solvers (implicit/perturbative/basis) integrate
`exp(_log_total_density(v_esc))` via `observables.total_rate` — the SAME
trapezoid measure `log_rate_density`'s grid path reads, so the gwkokab shape
term and −μ term stay consistent. `log_prob` for implicit rides the grid path,
AD via the solver's existing `@jax.custom_vjp` IFT (engine-side Steps 1–3 were
already done in `response/implicit.py`). The only gap was the bridge hardcoding
`mc_jax`.

### 2. Diagnostics + demos made solver/grid/scatter/sampler-aware
* `demo/diagnose_two_pulse_bias.py`: `--solver {mc_jax,implicit,perturbative}`,
  `--epsilon`, `--n-m/--n-chi` (grid sweep), `--full` (profile the UNBIASED
  shape−μ objective, not shape-only). D3 now prints the grid it ran on.
* `demo/fit_stage1_gwkokab.py` (single-pulse): `--solver --epsilon --n-m
  --n-chi --sigma-log-m --sigma-chi --n-sm --n-sc`. The self-consistent sampler
  now **defaults to 2× the fit grid** (faithful) — the fix for the root cause.

### 3. Tests — `tests/test_gwkokab_bridge_implicit.py`
I1–I2 in-sandbox (grid `total_rate` numerics + AD via real `observables`);
I3–I6 full-stack implicit two-pulse, I6 = FD-vs-AD on `lambda_natal`/
`sigma_natal_2` through the Picard fixed point. Workstation: **PASS**.

### 4. Plan — `demo/PLAN_cross_method_injection_recovery.md`
MC-inject → implicit-recover, with the four calibration hazards (density
convention, ε↔depletion mapping, shared physics primitives, recovery spin
resolution) and the `compare_all` JS + M1 ε-mapping gating strategy.

## Diagnostic verdicts (the science state)

Single-pulse, self-consistent (`--gen self`), workstation 2026-06-01:

| run | σ_natal recovered (truth 1.0) | verdict |
|---|---|---|
| implicit, default scatter | 1.33 | biased |
| implicit, **delta** (σ=0) | 1.33 (unchanged) | NOT scatter |
| mc_jax, same delta pipeline | ~1.0 (peak 1.125, MAP 1.004) | clean |
| implicit, delta, **faithful sampler** (n_sm48/n_sc24) | 1.125; spin_scale 0.103✓ | ~recovered |

The decisive control was mc_jax recovering on the *exact same* delta pipeline
that implicit failed → the bias is the **sampler vs eval-density sharpness
mismatch**, localised to `sample_from_model` (coarse n_sm=20/n_sc=8 over
geomspace [4,200], Δlog_m≈0.2, uniform within-cell jitter ~0.5 M⊙). Ruled out
along the way, each cheaply: resolution (λ grid-invariant at 0.240 across
(64,20)→(128,20)→(64,48)), the shape-only objective (`--full` invariant), and
measurement scatter (delta invariant). The implicit forward model is **not**
shown buggy.

## Next session

1. Re-run single-pulse `--gen self` (faithful sampler is now default) at a few
   fit resolutions — confirm residual σ/spin bias shrinks with n_m/n_chi.
2. Begin `PLAN_cross_method_injection_recovery.md` Step 1 (the `--gen mc`
   inject-MC harness) + Step 3 (promote the inject-θ `compare_all` JS check to a
   pytest gate) — gate before fit.
3. Two-pulse λ under cross-method (the headline experiment).

Memory: `project_implicit_pivot_bridge_wired`.

## Caveat (unchanged)

Cowork VM rootfs is 100% full (OS image) — **no JAX installable in-VM**. All
numeric runs are workstation; in-VM is py-compile + logic only. Commits are
Cowork-local; push from the workstation (no github network in the VM).
