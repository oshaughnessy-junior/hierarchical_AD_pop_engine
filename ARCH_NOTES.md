# Architecture notes — flagged for future iterations

Living scratchpad of architectural concerns the team has raised but
chosen not to address in the current pass. Each entry: motivation,
sketch of the proposed change, and rough effort estimate.

---

## Implicit / matrix-resolvent coagulation solve

**Concern.** The current engine sums the coagulation cascade over
`k = 1..k_max` *explicitly* (a perturbative truncation of the merger
process). For massive natal seeds (e.g. μ ~ 8-30 M⊙) and modest
retention this converges in ~4 generations and the truncation is fine.

**Failure mode.** For *light* seeds (μ ~ 1 M⊙ or below) or *very high*
retention, the cascade is effectively infinite. The current code
silently drops the tail beyond `k_max`, biasing the high-mass / high-
spin end of the predicted population.

**Proposed remedy.** Recast the coagulation operator as a linear map on
the discretised single-BH density and solve the steady-state implicitly:

    N = N_natal + ε · A · N
    ⇒  N = (I − ε A)⁻¹ N_natal

where

- `N` is the single-BH density vector on the (m, χ) grid, flattened to
  shape `(N_m · N_chi,)`.
- `N_natal` is the natal density on the same grid.
- `A` is a square matrix of shape `(N_m·N_chi, N_m·N_chi)` encoding the
  "two singles merge → one remnant" bilinear-scatter operator
  *evaluated against* the current `N`.
- `ε ∈ (0, 1]` is a per-pair efficiency scaling — pulls retention out
  of the matrix so it can be a single scalar knob during inference.

The bilinear nature of the merger step (it is `N · N`, not just `A · N`)
needs care: either iterate the linear solve to fixed point with `A`
re-evaluated at each step (cheap Picard / Newton iteration; AD-friendly
via implicit differentiation), or expand and solve at first order in
`ε`. Both fit on the existing grid.

**Algorithmic note.** Matrix form is natural because the coagulation
step *is* a bilinear-scatter operator on the discretised `(m, χ)` grid;
the existing `_step_one_generation` already builds the necessary
weights. JAX has `jax.scipy.sparse.linalg.gmres` and dense `solve`.
For `N_m = 96, N_chi = 32` the matrix is `3072 × 3072` — comfortably
dense.

**Effort estimate.** ~1.5 sessions. Half a session to refactor
`_step_one_generation` into an `A`-builder, half a session for the
implicit-solve wrapper with custom JVP / implicit diff, half a session
for unit tests against the explicit `k_max = 10` reference.

**Triggering event.** ~~Don't pull this forward until either~~ **Triggered.**
Red-team Phase 1 found that the engine matches Backup 2 (Doctor implicit
Picard) *only* at the unphysically small effective coupling ε ≈ 0.025.
For natural hierarchical-merger physics (visible 2G/3G peaks in GW
catalogues), the effective coupling is order unity, which the current
perturbative truncation cannot reach. See
`demo/PLAN_implicit_engine.md` for the refactor plan and
`red_team/engine_v2_implicit.py` for a working 2D prototype.

**Status (2026-05-17).**

a. ✅ Red-team Phase 1 fired the trigger — `compare_all.py` shows
   engine vs Backup 2 implicit agree only at ε ≈ 0.025; engine vs
   Backup 1 (pure particle MC) agree only at depletion fraction ≈ 0.01.
b. ✅ 2D `(m, χ)` implicit Picard prototype shipped at
   `red_team/engine_v2_implicit.py`. Reuses engine grid +
   `_step_one_generation` for gain; adds 4D loss term
   `⟨K · N · P_retain⟩(m, χ)`. Damped Picard, ~100 iterations to
   converge for the Stage 1 configuration.
c. ⏳ Refactor `CoagulationResponse._solve` to call the new solver, or
   add `_solve_implicit` as a parallel method — pending validation
   pass and AD-via-implicit-function-theorem custom VJP work. See
   `demo/PLAN_implicit_engine.md`.
