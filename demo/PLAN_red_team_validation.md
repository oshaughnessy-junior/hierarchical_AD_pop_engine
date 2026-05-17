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

## Architectural separation rule

The independence we want is **at the algorithm level**: discrete particle
Monte Carlo vs continuous-density grid quadrature. The library choice is
secondary. Re-implementing physics-fit formulas (`M_rem`, `chi_rem`,
`v_kick`) in a parallel pure-numpy stack risks introducing *new*
formula-typing bugs that defeat the validation purpose.

So the rule is:

- **May import:** engine physics primitives that both sides explicitly
  agree on — `M_rem`, `chi_rem`, `retention_aligned`, `v_kick_aligned`,
  natal-mass / natal-spin distribution samplers. JAX is allowed
  transitively (these primitives are jax-backed); the simulator wraps
  call sites with `np.asarray` so the outer driver is numpy-native.
- **May not import:** `hierarchical_ad_pop_engine.response.*` — that's
  the thing under test, including `_solve`, `_log_total_density`,
  `log_rate_density`, the `_step_one_generation` bilinear-scatter.
  The simulator does not use the grid representation as its working
  state (it tracks particles); it may *output* on the engine's
  `BinaryGrid` for hot-swap convenience.

This is a deliberate relaxation of an earlier draft of this plan. We
keep the validation power (independent algorithm + independent code
path through quadrature / scatter / accumulator) while not duplicating
formulas that are external truth.

## Hot-swap design goal

The simulator's outputs should plug into the same likelihood path the
engine plugs into, so the gold-standard red-team check is a *posterior
swap*: same prior, same mock catalog, same likelihood interface, two
forward models. Concretely:

- **Phase 0 output:** discrete catalog of merger events
  `(m1, m2, χ1, χ2, g1, g2)`. Sufficient for distribution-level checks.
- **Phase 1+ output:** also produce `log_n_k(m, χ)` on a
  `BinaryGrid` by histogramming the per-generation populations.
  This matches the shape `CoagulationResponse._solve()` returns and can
  be substituted for it in any downstream Poisson-likelihood call.
- **Phase 3 hot-swap:** wrap the simulator behind the same numpyro
  Distribution shim the engine will eventually wear (Stage 2 of
  `PLAN_gwkokab_fit.md`). NUTS / SVI then run unchanged on the
  simulator-backed model; posterior comparison is the gold-standard
  red-team check.

## Phase 0 — independent forward simulator

File: `red_team/sim_coagulation.py`

Inputs: same θ as the engine — natal IMF (μ, σ), natal spin (loc, scale),
kernel (a, b), v_esc, k_max, N_seed.

Algorithm (matches the engine's accumulator structure, but as discrete
particle MC rather than 4D quadrature):

1. **Natal sample.** Draw `N_seed` natal BHs from
   `f_IMF(m) × g_spin(χ)` using `scipy.stats.truncnorm.rvs`. Annotate
   each with `generation = 1`. Store as `populations[1] = (m, χ)`.
2. **Per generation `k = 2..k_max`:** the k-th generation is built from
   pairs (j, l) with `max(j, l) = k − 1`, mirroring `_solve`'s
   accumulator.
   For each such pair-of-generations (j, l):
   a. **Sample N pairs** of particles (one from `populations[j]`, one
      from `populations[l]`) with probability ∝ `K(m_a, m_b)` via
      batched rejection on a kernel upper bound.
   b. **Retention.** Apply `retention_aligned(m_a, m_b, χ_a, χ_b, v_esc)`
      — keeps a pair iff `v_kick < v_esc`. Pure Heaviside in the
      aligned-only physics.
   c. **Remnant.** Compute `M_rem`, `chi_rem` for retained pairs.
      Annotate remnant with `generation = k`.
   d. **Record merger event.** `(m1, m2, χ1, χ2, g1, g2)` with
      `m1 ≥ m2` enforced. Both retained and non-retained pairs are
      logged with a `retained: bool` flag so the simulator's retention
      fraction can be sanity-checked against the engine's `P_retain`
      integrand.
3. Stop when `k > k_max` or no pairs can be sampled (empty pool).

Note on with- vs without-replacement pairing: the engine treats `n_1`
as a *non-depleting source* — `n_2` is computed from pairs of `n_1`'s
without subtracting consumed natal BHs. The simulator matches this by
**sampling pairs with replacement** from each generation's pool. A
without-replacement variant (more physically realistic for a finite
cluster) is a separate experiment; the with-replacement variant is the
right comparison target for Phase 1.

Output:

- `populations`: dict `{1: (m_1, χ_1), 2: (m_2, χ_2), ...}` of retained
  per-generation populations. `len(populations[k])` is the discrete
  analog of `∫ n_k(m, χ) dm dχ`.
- `merger_events`: numpy structured array, shape `(N_events, 7)`:
  `(m1, m2, χ1, χ2, g1, g2, retained)`. Float64; `retained` is bool.

Acceptance: simulator runs end-to-end in seconds for
`N_seed = 10^4, k_max = 4`, vectorised pair-sampling via batched
rejection. No imports from `response.*`. JAX is invoked transitively
through the imported physics primitives only.

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
