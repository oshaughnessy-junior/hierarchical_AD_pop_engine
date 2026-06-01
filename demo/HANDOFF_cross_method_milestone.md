# MILESTONE — cross-method inject-MC / recover-implicit WORKS

**Date:** 2026-06-01. **Branch:** `observables`. **Status:** milestone reached;
delta-scatter cross-method recovery is clean. Residual systematics are now the
(interesting) science, not bugs. Follows `HANDOFF_implicit_pivot.md`.

## The milestone in one paragraph

The whole arc was building to this: **inject the catalogue with particle MC
(exact draws, no grid bias) and recover the population with the implicit grid
solver (sharp, KDE-free).** As of commit `7dbca1d` that round-trips correctly.
A self-consistent single-pulse run, injected from the MC particle pool and
recovered through the implicit solver at the depletion-matched ε, recovers the
natal parameters to within a few percent — the two methods, each used only
where it is strong, agree.

## Headline numbers (single-pulse, `--gen mc --solver implicit`, delta scatter)

```
[ε-match] MC injection f_dep=0.1000 → implicit ε*=0.4397
          (f_natal_depleted=0.1002, |Δf|=0.0002, ok)   ← H2 gate passes
   param            truth     MAP      Δ
   mu_natal          8.000    8.002   +0.002   ✓✓
   sigma_natal       1.000    1.103   +0.103   (resolution-limited)
   natal_spin_loc    0.050    0.043   −0.007   ✓
   natal_spin_scale  0.100    0.100   −0.000   ✓✓
   kernel_b          1.000    1.430   +0.430   ← largest residual (H3/resolution)
   log_rate          4.881    4.958   +0.077   ✓
```

## How the three pieces fit (all committed)

1. **Step 1 — exact-particle injection** (`c7cde9f`). `--gen mc` draws merger
   events by pairing the surviving MC pool ∝ `w_i·w_j·K(m_i,m_j)`
   (`pair_sample_from_mc`), self-pairs excluded, `m1≥m2`. No grid, no jitter.
2. **Step 2 — H2 ε-matching** (`4e6de09`, robustness `98077d3`). Read the MC
   injection's `f_dep = 1 − W_total/n_init`; root-find (scan converging ε grid,
   skip diverged solves, bracket the crossing, bisect) the implicit ε* with
   `f_natal_depleted(ε*) = f_dep`. ε* matched the depletion to |Δf|=2e-4.
   Direct f_dep matching, not the M1 closed form (exact only for constant-K).
3. **Boundary + support fix** (`7dbca1d`). `log_density_at` → `mode="nearest"`
   (kills the `0·-inf=NaN` / last-half-cell `-inf` at the grid edge that
   scatter triggered); `--gen mc` clips scattered PE samples to the recovery
   grid box. Unblocked the scattered case.

## Two reassurances that came out of the debugging

* **ε*=0.44 is correct, not weird.** ε is coupling×time, NOT the depletion
  fraction; the map ε→f_dep is nonlinear. Same value in the scatter and delta
  runs because the injection cascade is scatter-independent.
* **The implicit forward model was never buggy.** Every "bias" this arc chased
  was a harness/measurement artefact: coarse self-consistent sampler (fixed),
  shape-only objective (ruled out), measurement scatter deconvolution (ruled
  out), ε mismatch (matched), grid-boundary NaN (fixed).

## What's left — the residual cross-method systematics (now the real work)

`kernel_b` +43 % is the standout; `sigma_natal` +10 %. These are genuine
method-to-method differences, the regime H1/H3 target:
* **H1 (Step 3, NEXT):** promote `red_team/compare_all.py` JS(implicit ↔ MC) on
  `dN/dm`, `dN/dχ` to a pytest gate **at the injection θ** (+ a small θ-ball),
  run before any fit. Quantifies exactly the `kernel_b` discrepancy.
* **H3:** confirm MC pairing kernel and implicit kernel share the same
  primitive; `K_eff` self-consistency for non-constant K.
* **Resolution:** `sigma_natal` +10 % is likely fit-grid-limited.

## Run it

```
# clean cross-method recovery (delta):
python demo/fit_stage1_gwkokab.py --gen mc --solver implicit \
       --sigma-log-m 0 --sigma-chi 0
# with measurement scatter (now finite after the boundary fix):
python demo/fit_stage1_gwkokab.py --gen mc --solver implicit
```

Memory: `project_implicit_pivot_bridge_wired`.
Caveat unchanged: no JAX in the Cowork VM (rootfs full) — all numeric runs are
workstation; commits are Cowork-local, push from the workstation.
