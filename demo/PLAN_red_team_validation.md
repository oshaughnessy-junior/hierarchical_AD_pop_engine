# Red-team validation plan — discrete-realization parallel implementation

## Why

The coagulation engine represents populations as **continuous densities on
a grid** and aggregates them via 4D quadrature, bilinear scatter, and a
per-generation accumulator. A bug anywhere in that machinery would still
produce a smooth-looking ``dN/dm`` that *looks plausible*. Internal unit
tests check pieces of the engine against themselves — that is not the same
as checking it against an independent re-derivation.

The right red/blue-team check is to re-implement the **same physics** with
a **completely independent algorithm**: a discrete forward simulator that
samples natal BHs as a particle list, pairs them stochastically, and
records mergers. We then compare the simulator's outputs to the engine
both at the distribution level and at the posterior level.

## Architectural separation rule (non-negotiable)

The simulator imports **only the lowest-level physics primitives** that
both sides explicitly agree on:

- `M_rem(m1, m2, χ1, χ2)`              — Lousto-Zlochower remnant mass.
- `chi_rem(m1, m2, χ1, χ2)`            — Barausse-Rezzolla remnant spin.
- `P_retain(m1, m2, χ1, χ2, v_esc)`    — aligned-only retention Heaviside.

It does **not** import:

- `hierarchical_ad_pop_engine.response.*`  (the thing under test).
- `hierarchical_ad_pop_engine.grids.*`     (no grid representation).
- `jax` or `equinox`                       (no shared autodiff path).

Mass-sampling and spin-sampling primitives are re-implemented from
`scipy.stats` distributions so that even the random-number plumbing is
independent. Independence is the entire point.

## Phase 0 — independent forward simulator

File: `red_team/sim_coagulation.py`

Inputs: same θ as the engine — natal IMF (μ, σ), natal spin (loc, scale),
kernel (a, b), v_esc, k_max, N_seed.

Algorithm:

1. **Natal sample.** Draw `N_seed` natal BHs from `f_IMF(m) × g_spin(χ)`
   using `scipy.stats.truncnorm.rvs`. Annotate each with `generation = 1`.
2. **Pair selection.** For each generation:
   a. Form an active list of unpaired BHs of the current generation.
   b. Draw a random pair (i, j) uniformly from the active list.
   c. Accept with probability `K(m_i, m_j) / K_max(active)` (rejection
      sampling on the unnormalised kernel).
   d. If accepted, remove i and j from the active list. If rejected,
      put them back and try another pair.
   e. Loop until no further pairs can be drawn (budget exhausted).
3. **Retention.** For each accepted pair draw `r ~ Uniform(0, 1)` and
   retain iff `r < P_retain(m_i, m_j, χ_i, χ_j, v_esc)`. For pure
   aligned-only `P_retain ∈ {0, 1}`, so this is a hard cut.
4. **Remnant.** For retained pairs compute `M_rem, chi_rem`. Annotate
   the remnant with `generation = max(g_i, g_j) + 1`.
5. **Record merger event.** Save `(m_i, m_j, χ_i, χ_j, g_i, g_j)` to
   the merger catalog. Enforce `m_i ≥ m_j` by post-swap.
6. **Next generation.** Insert the retained remnants into the seed
   pool. Increment generation counter. Repeat (2-5).
7. Stop when `generation > k_max` or the active list is empty.

Output: numpy structured array of merger events, shape `(N_mergers, 6)`:
`(m1, m2, χ1, χ2, g1, g2)`. Float64 throughout. No JAX.

Acceptance: simulator runs end-to-end in seconds for
`N_seed = 10^4, k_max = 4`. No JAX, no grid, no imports from
`response.*`.

## Phase 1 — distribution-level engine ↔ simulator check

At a fixed θ* (the Stage 1 truth point), run both implementations and
compare:

1. Engine: compute `dN/d(m1, m2, χ1, χ2)` on the binary grid by
   evaluating `log_rate_density` at grid points and exp-ing.
2. Simulator: draw ~10⁵ mergers (large enough to histogram cleanly).
3. Compare 1D marginals — `dN/dm1`, `dN/dm2`, `dN/dχ1`, `dN/dχ2`,
   `dN/dq` where `q = m2/m1`:
   - Engine: marginalise the grid (trapezoid).
   - Simulator: histogram with adaptive bin widths.
   - Metrics:
     - 1-sample KS distance (`scipy.stats.ks_1samp` against the engine's
       CDF).
     - Total variation (½ ∫ |p_eng − p_sim| dx).
     - Integrated absolute error normalised by the truth integral.

Acceptance: KS < 0.03, total variation < 0.05 on each 1D marginal.
Disagreement above this flags a bug in *either* implementation (the
framework's red/blue check).

Plot deliverables, all log-y so low-rate hierarchical echoes are visible:

```
red_team/figures/phase1_dN_dm1.png
red_team/figures/phase1_dN_dm2.png
red_team/figures/phase1_dN_dchi1.png
red_team/figures/phase1_dN_dq.png
```

## Phase 2 — generation-resolved comparison

Both implementations know the generation of each merger (engine via
`_solve` returning `n_k` per k; simulator via the `(g1, g2)`
annotations). For each `k = 1..k_max` separately, repeat Phase 1's 1D
marginal checks with the events whose `max(g1, g2) == k`. This isolates
*which* generation, if any, the engine and simulator disagree on, which
narrows the bug hunt drastically.

Acceptance: per-generation KS < 0.05. Per-generation event-count ratio
within 10 %.

Plot deliverables:

```
red_team/figures/phase2_n_k_mass.png      # n_k(m) overlay, k = 1..k_max
red_team/figures/phase2_n_k_spin.png      # n_k(χ) overlay
red_team/figures/phase2_count_ratio.png   # N_k^sim / N_k^eng per k
```

## Phase 3 — parallel inference (the more important next step)

**Approach: ABC rejection.** The simulator slots into a likelihood-free
inference loop:

1. Sample `θ_i ~ prior` (10⁵ samples, same prior as Stage 1 numpyro
   model).
2. For each `θ_i`, simulate ~`N_obs` events via `sim_coagulation`.
   Normalise the seed budget so the catalog size matches the observed
   catalog within ±10 %.
3. Compute summary statistics on the catalog:
   - `median(m1)`, `std(m1)`, `quantile(m1, 0.9)`
   - `median(m2/m1)`, `std(m2/m1)`     (q-distribution)
   - `median(χ1)`, `std(χ1)`, `mean(χ1²)`
   - `len(catalog)`
4. Compute distance `d(S_i, S_obs)` in the whitened summary space
   (divide each summary by its prior-predictive std).
5. Accept the top 1 % by distance → ABC posterior of size 10³.

Compare the ABC posterior to the engine's Stage 1 SVI / NUTS posterior:

- Per-parameter 1D marginal medians within 1 σ_engine.
- 90 % credible-interval overlap > 80 %.

Plot deliverables:

```
red_team/figures/phase3_corner_overlay.png   # ABC (red) vs engine (blue)
red_team/artifacts/phase3_abc_posterior.npz
```

Acceptance: posteriors visually overlap at the corner-plot level and
metric thresholds met. Disagreement narrows to one of:

a. The engine's likelihood has a bug.
b. The simulator's physics differs in some pathway.
c. The summary statistics are insufficient.

Iterate on (c) first; if posteriors still disagree, the bug is real.

## Anti-goals

- Don't share *any* code path with the engine beyond `M_rem`, `chi_rem`,
  `P_retain`. The whole point is independence.
- Don't expand to real LIGO data. Mock-only red-teaming.
- Don't run Phase 3 if Phase 1 fails — fix the bug first.
- Don't substitute "I added a unit test" for the red-team check. The
  red-team check is a parallel implementation, not a test of the engine
  against itself.

## Effort estimate

| Phase | What                          | Effort       |
|-------|-------------------------------|--------------|
| 0     | Independent simulator         | 0.5 session  |
| 1     | Distribution check            | 0.5 session  |
| 2     | Per-generation check          | 0.3 session  |
| 3     | ABC parallel inference        | 1.0 session  |
|       | Total                         | ~2-3 sessions |

## Open questions

- Best summary statistics for Phase 3? Start with the list above; if
  posteriors are too wide, add `quantile(m1, 0.9)`, `mean(χ1²)`, and
  per-q-bin mean spin.
- How does pair selection's rejection sampling behave at extreme
  kernel exponents (`kernel_b > 4`)? Acceptance rate could plummet —
  fall back to weighted random choice on the discrete pair list.
- Two-pulse (Stage 2) needs more summary statistics to discriminate; if
  ABC is too lossy, pivot to neural-likelihood SBI (`sbi` package).

## Reproducibility

Random seed: `np.random.default_rng(20260518)`. Engine seed:
`jax.random.PRNGKey(20260517)` (same as Stage 1 demo). Phase 3 ABC
sweep uses a *separate* fresh seed `20260519` so the ABC and engine
draws are independent.

## Repository layout (proposed)

```
red_team/
  __init__.py
  sim_coagulation.py             # Phase 0: discrete simulator
  compare_distributions.py       # Phase 1: 1D marginal checks
  compare_per_generation.py      # Phase 2: per-k checks
  abc_inference.py               # Phase 3: ABC pipeline
  figures/                       # outputs, with run-hash suffix per run
  artifacts/                     # ditto
```

Living separately from `demo/` makes the role explicit: this is
adversarial validation, not user-facing demonstration.
