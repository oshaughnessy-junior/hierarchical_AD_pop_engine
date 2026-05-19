# Plan — merger-rate observable plots (parallel-branch handoff)

## Purpose

Add merger-rate observable diagnostics — `dC/dm1`, `dC/dm1 dm2`,
`dC/dm1 dχ_eff`, `dC/dm_f dχ_f` — alongside the existing single-BH
`N(m)`, `N(χ)` plots in the red-team comparison and the Stage 1 demo.

This is a **diagnostic / visualisation layer** only. No physics is added,
no solver is changed. Every plot is a deterministic integral over an
existing public API (`log_rate_density`, `gain_density`).

## Why

The single-BH density `N(m, χ)` is **not the inference observable**. GW
catalogs are pairwise merger events; the response function's
inference-facing object is `dN_R / dt dm1 dm2 dχ1 dχ2`. The current
open systematic (`project_implicit_vs_mc_systematic_bias.md` — 8–24× JS
above shot noise between implicit Picard and particle MC on `dN/dm`
and `dN/dχ`) is measured on `N(x)`; we do not yet know how it projects
into observable space. The 2G "ringing" in `N(χ)` may largely vanish
under K·N² integration; the growing-with-mass `N(m)` bias may be
amplified in `dC/dm1`. These plots are the instrument that tells us
which artefacts matter for science.

This branch runs in parallel with the v3 basis refactor
(`PLAN_engine_v3_basis.md`). It must not block or be blocked by it.

## Branching and conflict avoidance

* Base off the same commit as `engine_fem` is cut from (current `main`).
* Branch name suggestion: `observables`.
* **Do not touch** the following files / directories (these are
  load-bearing for the v3 refactor and the validated v2 solver):
  - `src/hierarchical_ad_pop_engine/response/implicit.py`
  - `src/hierarchical_ad_pop_engine/response/coagulation.py`
  - `src/hierarchical_ad_pop_engine/response/protocol.py`
  - `src/hierarchical_ad_pop_engine/remnants/`
  - `red_team/engine_v2_implicit.py`
  - `red_team/backup_particle_mc.py`
* **Do modify / add** (all additive, no editing of validated cores):
  - New: `red_team/observables.py` — pure diagnostic functions.
  - New: `red_team/test_observables.py` — pytest unit tests.
  - Extend: `red_team/compare_implicit_vs_mc.py` — add observable
    plots and JS metrics alongside the existing `N(m)`, `N(χ)` plots.
    Do not delete or change existing plots; this is purely additive.
  - Extend: `demo/demo_response_doctor2020.py` (or sibling) — add
    observable-plot calls; keep existing output unchanged.
* Output figures go to `red_team/figures_<hash>/` and
  `demo/figures_<hash>/` per `feedback_demo_output_conventions` (one
  dir per parametrised run so parallel sessions don't clobber).

When this branch lands, conflicts with `engine_fem` should be limited
to `compare_implicit_vs_mc.py` — both branches extend it. Coordinate
the merge there; the engine_fem changes will add a third path
(basis-Picard) and the observable changes will add new plot calls.
The merge is mechanical if the diffs touch different code regions.

## Observables to compute

All four are integrals of the per-environment differential merger rate
`dN_R / dt dm1 dm2 dχ1 dχ2 = ½ R_env K(m1,m2) N(m1,χ1) N(m2,χ2) P_retain`
already returned by `CoagulationResponse.log_rate_density`.

### 1. `dC/dm1` — primary-mass merger rate (1D)

The single most important inference observable. The LVK headline plot.

  dC/dm1 = ∫ dm2 dχ1 dχ2 [½ K(m1,m2) N(m1,χ1) N(m2,χ2) P_retain]
           · 2 · Θ(m1 − m2)

The factor of 2 · Θ converts the symmetric ½-integral to the ordered
m1 ≥ m2 convention.

Plot: log-y, log-x; line for each path (implicit, MC seed 1, MC seed 2;
later also basis-Picard).

### 2. `dC/dm1 dm2` — 2D mass plane (m1 ≥ m2)

  dC/dm1 dm2 = ∫ dχ1 dχ2 [K(m1,m2) N(m1,χ1) N(m2,χ2) P_retain]
               · Θ(m1 − m2)

Plot: pcolormesh on log-log axes; one panel per path; difference
panel between paths.

### 3. `dC/dm1 dχ_eff` — primary mass vs χ_eff (2D)

χ_eff = (m1 χ1 + m2 χ2) / (m1 + m2) — at Stage 1 (aligned only, χ ≥ 0)
this lives in [0, 1]. Caption MUST say "aligned-only χ_eff" — do not
mirror to negative values to fake the LVK convention
(`feedback_no_fake_physics_stand_ins`).

Easiest implementation: sample the 4D rate density on the existing
(m, χ) grid (which is what `log_rate_density` is built to evaluate
pointwise), compute χ_eff per cell, weight-histogram into a 2D
(m1 axis × χ_eff bin) grid using the cell weights.

### 4. `dC/dm_f dχ_f` — remnant population (2D)

Closest visible probe of the hierarchical cascade. The 2G peak at
(m_f ≈ 60 M☉, χ_f ≈ 0.687) shows up cleanly here.

  dC/dm_f dχ_f at the converged fixed point is precisely
  `gain_density(N*, log_K_mm, response, v_esc)` — which is already
  computed inside `solve_implicit_fixed_point`.

**Implementation**: re-run `gain_density` on the returned `N*` from
`ImplicitSolveResult.N`. No new physics, no new map. Just an extra
post-solve diagnostic call.

For the particle MC: each accepted merger has known
(M_rem, χ_rem) — histogram them into the same (m, χ) grid bins. The
MC already records this internally (per `project_engine_jl_doublecount_fix`
era code review).

## Module sketch — `red_team/observables.py`

```python
"""Merger-rate observable diagnostics.

Pure functions over already-computed response solutions. No physics
added; every output is an integral / histogram of existing public
quantities."""

from __future__ import annotations
import jax.numpy as jnp
from hierarchical_ad_pop_engine.response.coagulation import CoagulationResponse
from hierarchical_ad_pop_engine.response.implicit import (
    ImplicitSolveResult, gain_density, _dm_log_weights, _dchi_weights,
)
from hierarchical_ad_pop_engine.kernels import log_pairing_kernel_grid
from hierarchical_ad_pop_engine.remnants import P_retain

# All public functions return numpy arrays + axis arrays; pure for plotting.

def merger_rate_4d(
    N_star: Array, response: CoagulationResponse, v_esc: float,
) -> Array:
    """``½ K(m1,m2) N(m1,χ1) N(m2,χ2) P_retain`` on the 4D grid.

    Returns (N_m, N_chi, N_m, N_chi). Common precursor for plots 1–3.
    """

def dC_dm1(rate_4d: Array, grid) -> Array:
    """Integrate over (m2, χ1, χ2) with m1 ≥ m2 ordering. Returns (N_m,)."""

def dC_dm1_dm2(rate_4d: Array, grid) -> Array:
    """Integrate over (χ1, χ2); apply m1 ≥ m2 mask. Returns (N_m, N_m)."""

def dC_dm1_dchi_eff(
    rate_4d: Array, grid, *, n_chi_eff_bins: int = 32,
) -> tuple[Array, Array]:
    """Histogram into (N_m, n_chi_eff_bins). Returns (heatmap, chi_eff_edges)."""

def dC_dm_rem_dchi_rem(
    result: ImplicitSolveResult, response: CoagulationResponse, v_esc: float,
) -> Array:
    """gain_density(N*, …) at the fixed point. Returns (N_m, N_chi)."""

# Histogram helpers for the particle MC (input: arrays of accepted-merger props)
def mc_dC_dm1(m1_arr, m2_arr, m_bins) -> Array: ...
def mc_dC_dm1_dm2(m1_arr, m2_arr, m_bins) -> Array: ...
def mc_dC_dm1_dchi_eff(m1_arr, m2_arr, chi1_arr, chi2_arr, m_bins, chi_eff_bins) -> tuple: ...
def mc_dC_dm_rem_dchi_rem(m_rem_arr, chi_rem_arr, m_bins, chi_bins) -> Array: ...
```

Keep the file small; resist adding "convenience wrappers" that re-wrap
`CoagulationResponse` — that's the v3 refactor's territory.

## Comparison-script changes

`red_team/compare_implicit_vs_mc.py`: keep the existing `N(m)` and
`N(χ)` plots and JS reporting **unchanged**. After the existing output,
add:

1. Compute the 4D rate density for the implicit path (once, cached).
2. Compute the four observable plots for both paths (implicit + MC,
   both seeds).
3. JS metrics on `dC/dm1` at the grid's own m-resolution, reported with
   the same `phys/noise` and `phys/natal` floors as the existing
   `N(m)` metric.
4. Save figures into the existing `red_team/figures_<hash>/`
   directory with names `obs_dC_dm1.png`, `obs_dC_dm1_dm2.png`,
   `obs_dC_dm1_chi_eff.png`, `obs_dC_dm_rem_chi_rem.png`.
5. Print interpretation header explaining `phys/noise` ≫ 1 in
   *observable space* is the physics signal worth chasing (vs the
   N(x)-space JS, which mixes representation noise with physics).

CLI flag suggestion: `--observables-only` to skip the existing
`N(x)`-space plots (faster diagnostic loop).

## Pytest

`red_team/test_observables.py` — unit tests, fast (< 5 s total):

a. `test_dC_dm1_normalisation` — total event count integral
   `∫ dC/dm1 dm1 = ½ ∫∫∫∫ K N N P_retain` (modulo the m1≥m2 doubling)
   matches a direct 4D quadrature to relative 1e-10. Catches Heaviside-
   masking bugs.
b. `test_dC_dm_rem_dchi_rem_matches_gain_density` — bit-equal parity
   between `dC_dm_rem_dchi_rem(result, response, v_esc)` and a direct
   call to `gain_density(result.N, log_K_mm, response, v_esc)`. This
   guards against accidental re-implementation drift.
c. `test_mc_histograms_match_engine_at_high_N` — at n_seed = 100k and
   ε = 0.1 (the regime where the MC shot noise is small and the open
   systematic is cleanest), `mc_dC_dm1` and engine `dC_dm1` agree to
   JS < 0.05 on the 1G+2G mass range. **Gate threshold is
   deliberately loose** — this branch's job is to *expose* the
   systematic in observable space, not to certify agreement. The v3
   refactor (and any subsequent fix) is where the threshold tightens.
d. `test_aligned_chi_eff_range` — assert χ_eff ∈ [0, 1] for all
   accepted MC mergers and for the engine's 4D rate density. Catches
   sign-convention bugs.

Existing red-team integration test (planned in PLAN_engine_v2_production
Step 4) gets one extra assertion: `JS(implicit, MC) on dC/dm1 < 0.1`
on the bulk mass range. Same gating philosophy — catastrophic-failure
only at this stage.

## What this plan deliberately does NOT do

* Does not redesign the demo Stage 1 mock generator. The existing mock
  catalog is already in event space (it consumes `log_rate_density`).
* Does not introduce a `MergerObservable` class hierarchy in `src/`.
  Premature abstraction; observable plots are diagnostics, not API.
* Does not touch the particle MC's internal data structures. It only
  reads the existing accepted-merger arrays.
* Does not unify the figure pipeline with `compare_all.py`. That
  script is in the deprecation list (see PLAN_engine_v2_production
  §Step 7); writing new code into it is wasted effort.
* Does not implement χ_eff with mirroring to negative values to
  approximate non-aligned mergers — Stage 1 is aligned-only, label it
  so (`feedback_no_fake_physics_stand_ins`).
* Does not propose a new validation algorithm. The validation pair
  stays at implicit Picard + particle MC
  (`feedback_validation_downselect_two_algorithms`). These observables
  are diagnostics on the existing pair.

## Effort and acceptance

Estimated: 1–2 sessions for a focused agent.

Acceptance criteria:
1. `red_team/observables.py` exists, public functions documented, all
   four pytest tests pass.
2. `compare_implicit_vs_mc.py` produces eight figures per run (four
   existing N(x) plots, four new observable plots), and reports a
   `dC/dm1` JS metric alongside the existing `dN/dm` metric.
3. At Stage 1 θ*, ε = 1, n_seed = 20k: the new figures clearly show
   the 1G+2G structure in observable space, with the implicit and MC
   paths visually overlapping in the bulk and shot-noise-divergent in
   the 3G+ tail (same qualitative pattern as the N(m) plots).
4. At ε = 0.1, n_seed = 200k: the `dC/dm1` `phys/noise` ratio is
   reported. **This number is the headline diagnostic** the v3 basis
   work needs to monitor. Record it in a short note in the figures
   directory (`figures_<hash>/README.md`) so the engine_fem branch can
   compare against pre-refactor baseline.

The handoff agent should not chase the systematic itself — that's the
v3 refactor's job. The acceptance bar is "the new diagnostic exists,
runs cleanly, and produces interpretable plots."
