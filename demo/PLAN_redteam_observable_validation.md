# Plan — red-team observable validation: implicit Picard vs particle MC

Two-algorithm validation, observable-driven. Closes the gap between the
existing single-BH `N(x)` red-team scripts (research-grade, ad-hoc) and
the package's pytest gate, and extends the comparison from `N(x)` to
the binary merger-rate observables (`dN_mgr/dt dm1`, `dN_mgr/dt dm1 dχ_eff`,
`dN_mgr/dt dm1 dq`) that LVK inference actually cares about.

## Why this plan exists

The validation rationale (per `feedback_validation_downselect_two_algorithms`):
we maintain **two** independent physics algorithms — the production
implicit Picard solver on the engine grid and a pure particle Monte Carlo
in continuous (m, χ) — and use their cross-agreement as the correctness
oracle. Internal pytest agreement alone is insufficient
(`feedback_red_team_validation`).

Two things are currently weak about that validation:

1. **It's research-grade, not test-grade.** The N(m)/N(χ) comparison
   lives in `red_team/compare_implicit_vs_mc.py`, a script you run by
   hand. Acceptance is interpreted from JS numbers in a printout. No
   pytest, no CI gate, no fail signal when somebody changes
   `_step_one_generation` and the systematic doubles.

2. **It's wrong-axis.** `N(m, χ)` is not the inference observable — GW
   catalogues contain pairwise merger events parameterised by
   `(m1, m2, χ1, χ2, z)`. The 8–24× JS systematic in `N(m)` between
   implicit Picard and particle MC at ε=0.1
   (`project_implicit_vs_mc_systematic_bias`) may partially or wholly
   wash out under the `K·N²` integration into observable space, or it
   may be amplified. Until we measure it in observable space we don't
   know which artefacts matter for science.

This plan promotes the N(x) prototypes into pytest tests (with the
chi_rem m1≥m2 fix and j≠l double-count fix integrated) **and** builds
the observable-driven comparison tests on top of the new
`src/hierarchical_ad_pop_engine/observables.py` module.

## What's a "trivial marginal" for the particle MC

For continuous (grid / basis) representations the marginals are 4D
quadrature contractions of `½ R_env K N N` — the work
`src/observables.py` already does. For the particle MC the marginal is
**a weighted histogram of accepted merger events**:

```
m1_arr, m2_arr, chi1_arr, chi2_arr, ...  ←  particle_mc(...).mergers
```

Each accepted row is one physical merger event (per
`backup_particle_mc.py` line 15). The MC's `dN_mgr/dt dm1` is
literally `np.histogram(m1_arr, bins=m_bin_edges) · rate_normalisation`;
2-D marginals are `np.histogram2d` on the derived axis pair. The MC
side is one line per marginal — the comparison-side complexity all
lives in (i) matching axis conventions between paths and (ii) choosing
sensible bin schemes that keep MC shot noise from dominating the
metric in the tails.

## Stringency convention — tail tolerance

Two regimes, picked based on what each path resolves cleanly:

- **Bulk** (1G + 2G, where the bulk of the rate lives): both paths
  resolve cleanly. Acceptance: JS-on-grid below a tight threshold
  (~0.05) for 1-D; relative-L1 ≤ 0.15 inside a bulk mask for 2-D.
- **Tail** (3G+, equal-mass-ratio extremes, χ_eff endpoints): MC shot
  noise dominates per-bin counts; engine grid resolution is coarse.
  Acceptance: pattern-only (peak location within a bin, sign of
  residual matches expectation), no scalar metric.

The bulk mask is constructed once per test, defined by the **engine
density** (not the MC density — using the MC would introduce a
seed-dependent mask that could mask a real bias). Specifically:
`bulk = {x : engine_density(x) ≥ tail_frac · max(engine_density)}`
with `tail_frac = 0.01` default. Cells outside the mask are excluded
from the JS / L1 metric but are still plotted, so the tail behaviour
is visible by eye.

For the 2-D tests this matters most. With 100k MC mergers spread over
~1000 bins in bulk, bins carry ~10–100 events; tail bins drop to 0–1
events. Comparing per-bin densities in the tail is comparing noise to
the engine's truncation-level artefacts, not measuring physics.

Per `feedback_no_fudge_factors`, the bulk mask is **not** a knob — it's
derived from the engine density at a fixed `tail_frac` written once per
test and never tuned per run. If a test fails inside the bulk, the
remedy is fixing the physics or the solver, not raising the threshold.

## Phase A — promote the `N(x)` prototypes

### A.1 Inventory of existing comparisons

`red_team/compare_implicit_vs_mc.py` currently produces:
- `dN/dm` and `dN/dχ` for implicit Picard, particle MC seed 1, particle
  MC seed 2 — at both `ε=1` (production) and `ε=ε_natal=1e-4` (noise
  floor).
- JS-on-grid metrics: `js_full_imp_mc`, `js_spin_imp_mc`,
  `js_full_noise`, `js_spin_noise`, and the natal-baseline variants.
- Mass-range split (1G-only mask vs full range).
- Plot output to `red_team/figures_<hash>/`.

These are the measurements; what's missing is a **gate**.

### A.2 New pytest target

`tests/test_redteam_Nx_parity.py` — slow tier (mark `@pytest.mark.slow`),
opt-in via `pytest -m slow` so the fast CI doesn't pay the MC's cost.

The test fixture:
- Reuses `_STAGE1` from `tests/test_implicit_parity.py` for natal +
  kernel parameters (single source of truth for the comparison
  configuration).
- Runs the package implicit solver and two MC seeds (small, ≈20k
  initial particles to keep test wall-clock under ~60 s).
- Builds the engine `N(m, χ)` and the MC density estimate on the
  engine grid (KDE on (log m, χ), then evaluated at `grid.m × grid.chi`,
  same convention as `compare_implicit_vs_mc.py`'s `mc_marginals_on_grid`).

The acceptance assertions — all four part of v1:

1. **Cross-method noise gate (dN/dm).**
   `js(implicit, mc_seed1) on dN/dm in bulk < 5 · js(mc_seed1, mc_seed2) on dN/dm in bulk`
   — the cross-method disagreement must be within 5× of the MC-to-MC
   noise on the same statistic. (Same logic as `compare_implicit_vs_mc.py`
   already uses informally.) The `5×` is loose deliberately at this
   stage; tightening waits for the v3 basis work to close the
   grid-quadrature systematic.

2. **Cross-method noise gate (dN/dχ).** Analogous to (1) on the spin
   axis.

3. **Symmetry-fix invariant guard.** The forward rate integrand
   `½ R_env K(m1, m2) N(m1, χ1) N(m2, χ2)` is symmetric under the
   `(m1, χ1) ↔ (m2, χ2)` swap of progenitor labels. Once the
   chi_rem m1≥m2 canonicalisation is in place at every call site
   (`project_chi_rem_asymmetry_finding`), the engine's pre-scatter 4-D
   progenitor-weight tensor (the integrand of
   `_step_one_generation` *before* bilinear scatter) is also
   bit-symmetric under the same swap. Test: build the tensor for a
   non-degenerate `(n_j, n_l)` with `n_j ≠ n_l`, take its transpose
   under the joint `(m1, m2)` and `(χ1, χ2)` axis swap, assert
   `max |T - T_swapped| / max |T| ≤ 1e-12`. The numerical value of
   each entry shifts as natal / kernel parameters change; the
   invariant `T == T_swapped` holds in any configuration where the
   physics is implemented correctly.

4. **j≠l double-count invariant guard.** For unordered progenitor
   pairs `(j, l)` with `j ≠ l`, the contribution to `n_k` is a single
   `_step_one_generation(n_j, n_l, …, sym=1.0)` call, not the
   symmetrised sum `_step_one_generation(n_j, n_l, …) +
   _step_one_generation(n_l, n_j, …)` that the original bug
   produced (`project_engine_jl_doublecount_fix`). The cleanest
   invariant is a head-to-head against a hand-rolled "obviously
   correct" reference: a small `tests/redteam_reference_solve.py`
   module that re-implements the generation accumulator as a single
   nested loop over **unordered** `(j, l)` pairs, with no swap-and-
   add. By inspection this reference cannot double-count.  Assert
   `max |n_k_engine - n_k_reference| / max |n_k_engine| ≤ 1e-12` for
   all `k` up to `k_max = 4`. The reference is itself parameter-free
   (it consumes the same `_step_one_generation`, the same `K`, the
   same `n_j` inputs as the engine), so any physics shift moves both
   numbers together — the equality is a structural property of the
   accumulator, not a frozen number.

**Why these tests don't need a frozen reference file.** Both (3) and
(4) are *equality* assertions between two quantities computed on the
same engine state. If the natal IMF, kernel exponents, retention
sharpness, etc. change, both sides shift identically and the equality
holds. There is no `.npz` file checked in, no maintenance burden when
physics shifts, and no fix-the-test-not-the-bug failure mode — a
broken engine produces inequality, full stop. Per the
2026-05-21 decision: no frozen numerical results from unit tests; all
regression guards are invariant-based.

The hand-rolled reference in (4) deliberately *reuses*
`_step_one_generation` as the per-pair primitive. That is the symbol
under test — but the loop structure around it (which calls happen,
with what arguments, summed in what order) is the actual bug surface,
not the primitive. Re-using the primitive isolates the test to the
accumulator logic; if we wanted a stronger statement we could re-
implement the bilinear scatter too, but that drifts toward "test the
implementation by re-implementing it," which is its own anti-pattern.
The cleanest layering: invariant (3) pins the primitive's symmetry;
invariant (4) pins the accumulator's pairing rule given the primitive.

### A.3 Refactoring touchpoints

- `red_team/compare_implicit_vs_mc.py` keeps existing measurement /
  plot behaviour. The pytest imports a small number of helpers from it
  (`mc_run`, `mc_marginals_on_grid`, `js_subrange`); promote those to a
  thin `red_team/comparison_helpers.py` to keep the test from
  depending on the script's `main()` argument-parsing machinery.
- No physics edits to `red_team/backup_particle_mc.py`,
  `red_team/engine_v2_implicit.py`, `response/coagulation.py`, or
  `response/implicit.py` are needed — those already carry the
  chi_rem and j≠l fixes per the project memories.
- A short "regen reference" script lives in `tests/data/`; pyproject
  excludes `tests/data/` from packaging.

Acceptance for Phase A: `pytest -m slow -k redteam_Nx` runs in ≤ 90 s
and passes deterministically across three independent invocations (MC
seeds are fixed in the fixture). The frozen references file is the
only checked-in numerical artefact.

## Phase B — observable comparison infrastructure

### B.1 Particle-MC histogram adapters

New file: `red_team/observables_mc.py`. Pure numpy. One small public
function per observable; each consumes the structured `mergers` array
from `particle_mc(...).mergers` and returns a histogram on user-
supplied bin edges. No solver-specific logic, no AD requirement (this
is the MC side; we never differentiate through the MC).

```python
def dN_dm1_mc(mergers, m_bin_edges, rate_scale): ...
def dN_dm1_dchi_eff_mc(mergers, m_bin_edges, chi_eff_bin_edges, rate_scale): ...
def dN_dm1_dq_mc(mergers, m_bin_edges, q_bin_edges, rate_scale): ...
```

`rate_scale` converts particle counts into rate density (`n_init · T_simulated`
on the same scale the engine uses for `R_env`). The conversion is
explicit, not a guess — same convention as
`compare_implicit_vs_mc.py` uses for the existing `N(m)` comparison.

The MC always counts *all accepted* mergers, retained or not — same as
the engine's forward rate, which carries no `P_retain` (per the
observables module decision). Tests assert this in B.4 below.

### B.2 Engine integration into χ_eff and q axes

`src/hierarchical_ad_pop_engine/observables.py` already provides 1-D
and 2-D marginals on the **axis-aligned** binary parameter axes
(`m1, m2, χ1, χ2`). Two new observables are derived quantities:

- `χ_eff = (m1 χ1 + m2 χ2) / (m1 + m2)`, range `[0, 1]` at Stage 1 (aligned only).
- `q = m_light / m_heavy`, range `(0, 1]`.

These are not separable on the (m, χ) tensor, so they require either
(i) explicit 4D quadrature with histogram binning, or (ii) Monte-Carlo
quadrature with stratified sampling. Path (i) is the natural fit
for the engine grid (the 4D tensor already fits in ~75 MB) and is
fully deterministic and AD-safe via the same bilinear-scatter pattern
used in `response.coagulation._bilinear_scatter`.

Add to `observables.py`:

```python
def dN_dm1_dchi_eff(N_mchi, grid, kernel_params, log_R_env, *,
                    chi_eff_bin_edges, ordered=True): ...

def dN_dm1_dq(N_mchi, grid, kernel_params, log_R_env, *,
              q_bin_edges, ordered=True): ...
```

Both build the (N_m, N_chi, N_m, N_chi) integrand
`½ R_env K(m1, m2) N(m1, χ1) N(m2, χ2)`, compute the derived axis
value per (m1, χ1, m2, χ2) cell, and bilinearly scatter the
cell-weighted integrand into the (m1, χ_eff) or (m1, q) histogram.
The `ordered=True` flag enforces `m1 ≥ m2` (LVK convention) — flipping
labels per cell so χ_eff and q match LVK definitions cell-by-cell.

These two functions live in `observables.py` (not a separate module)
because they share infrastructure with `rate_density_4d`. Bilinear-
scatter helper goes alongside.

Memory check: at (96, 32) grid the 4-D integrand is 75 MB and the
bilinear-scatter into a (96, 32) output histogram is comfortable. If
later we want larger grids, lazily compute the 4-D integrand row-by-row
in `m1` via `jax.lax.scan`.

### B.3 Comparison metric — JS-on-grid + bulk mask

New file: `red_team/observable_metrics.py`. Pure functions over (engine
histogram, MC histogram, axis arrays) pairs:

```python
def js_on_grid_bulk(engine_dens, mc_dens, axis_arrays, *, tail_frac=0.01): ...
def total_rate_ratio(engine_dens, mc_dens, axis_arrays): ...
def peak_location(dens, axis_arrays): ...
```

The bulk mask is computed from the **engine** density (B.3 stringency
section above) and applied to both densities before JS / L1. Tests
assert that the chosen `tail_frac=0.01` mask covers ≥ 90% of the
engine's total rate — same idempotency check that the existing
`compare_implicit_vs_mc.py` uses for its mass-range split.

JS divergence uses the natural-log base (`log(2)`-normalised so JS ∈
[0, 1]); same convention as `phase1_overlay.js_divergence`.

### B.4 Sanity tests for B.1–B.3

`tests/test_redteam_observables_helpers.py` (fast tier):

- `test_dN_dm1_mc_normalisation` — at `n_init = 10k` and uniform
  natal mass distribution, the MC histogram's total integrated rate
  matches the analytic engine rate to within MC shot noise.
- `test_dN_dm1_dchi_eff_engine_range` — `χ_eff` per cell stays in
  `[0, 1]` at Stage 1 (aligned only). Catches sign-convention bugs in
  the χ_eff computation.
- `test_dN_dm1_dq_q_range` — `q ∈ (0, 1]` after the `ordered=True`
  mask. Catches m1≥m2 labelling bugs.
- `test_observables_marginals_match_axis_aligned` — integrating
  `dN_dm1_dchi_eff` over `χ_eff` reproduces `marginal_1d(axis='m1')`
  to ≤ 1e-12 relative. Same logic for `dN_dm1_dq` integrated over `q`.
  Pins consistency between the new derived-axis observables and the
  existing axis-aligned ones.

Acceptance for Phase B: tests above pass; the new observable functions
are AD-safe (a single `jax.grad` sanity check, same pattern as the
existing `tests/test_observables.py::test_marginal_2d_grad_through_N`).

## Phase C — three observable comparison tests

All three live in `tests/test_redteam_observables_parity.py`, slow
tier, opt-in via `pytest -m slow`.

### C.1 `dN_mgr/dt dm1` parity

Fixture: same `_STAGE1` config and MC seeds as Phase A.

```
engine: marginal_1d(N_implicit*, grid, kernel_params, log_R_env, axis='m1')
MC:     dN_dm1_mc(mergers, m_bin_edges=grid.m, rate_scale=...)
```

Acceptance:
1. **Bulk JS:** `js_on_grid_bulk(engine, mc1, m, tail_frac=0.01) < 5 ·
   js_on_grid_bulk(mc1, mc2, m, tail_frac=0.01)`.
2. **Total rate:** `total_rate_ratio(engine, mc1) in [0.85, 1.15]`.
   The 15% tolerance reflects MC shot-noise on ~20k mergers; the
   slow-tier MC budget keeps test runtime reasonable.
3. **2G peak location:** `peak_location(engine, m)` agrees with
   `peak_location(mc1, m)` within ±1 mass bin in the 2G mass range
   (`m ∈ [m_2G_lo, m_2G_hi]`, defined per fixture).

### C.2 `dN_mgr/dt dm1 dχ_eff` parity

Fixture extends C.1's by setting `chi_eff_bin_edges = linspace(0, 1, 17)`
(16 bins; coarse on the noisier axis is the right call for MC budgets).

```
engine: dN_dm1_dchi_eff(N_implicit*, grid, kernel_params, log_R_env,
                       chi_eff_bin_edges=..., ordered=True)
MC:     dN_dm1_dchi_eff_mc(mergers, grid.m, chi_eff_bin_edges, rate_scale=...)
```

Acceptance:
1. **Bulk relative-L1:**
   `sum(|engine - mc1| * bulk_mask) / sum(engine * bulk_mask) < 0.5`.
   Looser than the 1-D case because per-bin counts are ~10× smaller.
2. **Total rate within bulk:** as in C.1.
3. **2G χ_eff peak location:** the 2G mass slice
   `m ∈ [m_2G_lo, m_2G_hi]` has its χ_eff distribution peaked at the
   Lousto-Zlochower predicted value within ±2 χ_eff bins. (Equal-mass
   1G+1G → χ_rem ≈ 0.687 → χ_eff ≈ 0.687 when both progenitors are
   themselves spinless 1G blocks.)
4. **Tail tolerance (pattern only):** assert qualitative monotonicity
   in the 1G mass range — `dN/dχ_eff` at fixed 1G `m1` decreases
   monotonically with `χ_eff` (natal spin distribution is narrow and
   peaked at low χ; 1G+1G mergers inherit a χ_eff close to the natal
   peak, so the distribution falls off with χ_eff). Cheap, robust,
   surfaces gross sign / labelling bugs.

### C.3 `dN_mgr/dt dm1 dq` parity

Fixture: `q_bin_edges = linspace(0.05, 1.0, 20)` (19 bins; we clip
the very-extreme-q tail at 0.05 — both methods are noisy there).

```
engine: dN_dm1_dq(N_implicit*, grid, kernel_params, log_R_env,
                  q_bin_edges=..., ordered=True)
MC:     dN_dm1_dq_mc(mergers, grid.m, q_bin_edges, rate_scale=...)
```

Acceptance: same three thresholds as C.2 (bulk-L1 < 0.5, total rate
within 15%, q-axis pattern monotonicity / equal-mass bias as predicted
by the chosen kernel exponent `b`). The pattern check is the
diagnostic that surfaces the kernel implementation's mass-ratio bias
working as intended: with `kernel_b > 0` the engine and MC must both
show the `dN/dq` distribution rising toward `q = 1`.

## File layout summary

```
demo/PLAN_redteam_observable_validation.md          # this file
red_team/
├── comparison_helpers.py        # NEW — pulled out of compare_implicit_vs_mc.py
├── observables_mc.py            # NEW — MC histogram adapters (B.1)
├── observable_metrics.py        # NEW — JS/L1 + bulk mask (B.3)
├── compare_implicit_vs_mc.py    # unchanged (still ad-hoc research tool)
└── ...
src/hierarchical_ad_pop_engine/
└── observables.py               # EXTEND — add dN_dm1_dchi_eff, dN_dm1_dq (B.2)
tests/
├── redteam_reference_solve.py                   # NEW — hand-rolled accumulator (A.2 invariant 4)
├── test_redteam_Nx_parity.py                    # NEW — Phase A
├── test_redteam_observables_helpers.py          # NEW — Phase B sanity
└── test_redteam_observables_parity.py           # NEW — Phase C
```

`tests/redteam_reference_solve.py` is a small (≲ 80 line) module
implementing the "obviously correct" generation accumulator used by
invariant (4) in Phase A.2 — a single nested loop over unordered
`(j, l)` progenitor pairs with no swap-and-add. It is **not** a frozen
numerical artefact; it is parameter-free Python that recomputes the
same arithmetic the engine performs, structured to make the pairing
rule manifest by inspection. Lives in `tests/` (not `red_team/`)
because it has no purpose outside the test suite.

No `.npz` / frozen-reference files anywhere in this plan.

## Effort estimate

- Phase A: 0.5 session. Refactoring the existing
  `compare_implicit_vs_mc.py` into pytest fixtures, plus the
  ~80-line hand-rolled reference accumulator for invariant (4).
- Phase B: 1 session. The two new engine observables
  (`dN_dm1_dchi_eff`, `dN_dm1_dq`) are the most novel pieces; the MC
  histogram adapters are one-liners.
- Phase C: 0.5 session. Three tests, each ~50 lines. Most of the
  effort is wall-clock waiting for the MC seeds to run during test
  development, not coding.

Total: ~2 sessions for a focused agent.

## What this plan deliberately does NOT do

- Does not redesign the particle MC's stopping criterion. Per
  `project_f_natal_depleted_fix`, the existing `f_natal_depleted`
  driven by the implicit solver is the right convention.
- Does not introduce a third independent algorithm. Per
  `feedback_validation_downselect_two_algorithms`, the validation pair
  stays at implicit Picard + particle MC.
- Does not extend to the v3 basis solver. Once Phase A+B+C land the
  same observable tests can be lifted onto the basis solver in a
  follow-up (the marginal infrastructure in
  `observables.density_from_basis` already accepts basis results),
  but that comparison is in `engine_fem`'s scope, not here.
- Does not mirror χ_eff to negative values to mimic LVK's non-aligned
  convention. Stage 1 is aligned-only; the χ_eff axis is `[0, 1]` per
  `feedback_no_fake_physics_stand_ins`. Plot captions say so.
- Does not test redshift dependence. Stage 1 single-environment
  hypothesis; redshift is the formation-rate layer's responsibility,
  not the response function's.

## Decisions log (2026-05-21)

1. **MC budget per test: 20k.** Approved.

2. **No frozen numerical results from unit tests.** The symmetry-fix
   and j≠l double-count guards are **core for v1** — they would have
   caught both bugs immediately in the previous red-team cycle — but
   the mechanism is *invariant testing*, not a checked-in `.npz` of
   pinned numbers. Each guard is an equality assertion between the
   engine and a structurally-obvious reference computed on the same
   physics inputs: both sides shift identically under any legitimate
   parameter change, so the equality is a property of the
   implementation, not a property of the numbers. See Phase A.2
   assertions (3) and (4) for the concrete shape.

3. **χ_eff: aligned-only is correct for v1.** No χ_p. The misalignment
   / tilt axis is not implemented — adding it would lift `(m, χ)` to
   `(m, χ, tilt)` and inflate several solver costs by a large power.
   This is a known follow-up (necessary at some point) but the draft
   v1 stays aligned-only.

## Follow-ups parked for later (not part of v1)

- **Tilt / misalignment.** Lift the binary parameter space from
  `(m, χ)` to `(m, χ, tilt)`. χ_p observable becomes meaningful at
  that point. This is the v2-era upgrade alluded to in `DESIGN.md`
  §8.5; redoing the observable comparison in `(m, χ_eff, χ_p)` space
  is straightforward once the tilt axis exists.
- **Basis solver in the observable comparison.** Once the v3 basis
  refactor (`engine_fem` branch) is green on Step 5 (observable JS vs
  MC), drop the basis path into the same `test_redteam_observables_parity.py`
  via `observables.density_from_basis`. The plan's stringency
  thresholds should hold by construction if the basis is doing its
  job; if it doesn't hold we have a real diagnostic instead of a hand-
  inspected plot.
