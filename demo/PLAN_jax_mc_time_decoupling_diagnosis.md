# Diagnosis — JAX MC "interaction time" decoupled from pool depletion

Written 2026-05-29 in response to the workstation-round-3 handoff
breadcrumb: "interaction count is different in the two methods (!) —
obvious in right panel — so 'time' is different. (Lower in JAX than np)".

This memo is **diagnosis + proposed fix**, no code edits.  Awaiting
green-light before touching `red_team/backup_particle_mc.py` or
`src/.../response/mc_jax.py`.

Related:
* `[[project-jax-mc-workstation-round2]]` — the round that fixed the
  `n_init²` total_rate rescaling.
* `[[project-jax-mc-j2-done]]` — the J2 cascade design (K-rejection-
  as-weighting) whose semantics this memo claims are subtly off.

---

## 1. The user's diagnosis, verified by code reading

Empirical numbers from the handoff:

```
Demo (20k particles, depletion_fraction ≈ 1.0):
  n_jax (scan steps recorded) = 13 328
  Σw_mergers                  = 10 743
  numpy n_accepted            = 19 988   (per [[project-jax-mc-j2-done]])

Stage-1 MC-JAX integration test (n_init=2000, depletion=0.10):
  E[N_obs](θ*) = 23.75   vs   target N_OBS = 100   (7.6 σ deficit, factor 4.2)
```

User's framing: **the two methods are at different cascade
"times"**, so they're apples of a different vintage; recalibrate the
interaction time and agreement should return.  Code reading
confirms this for the demo line; I think it **also** explains the
integration-test failure, but through a different mechanism than a
flat time rescale.  See § 4.

---

## 2. Root cause (algorithmic asymmetry)

The numpy and JAX cascades use the same K-rejection physics on
different **boundaries between rate-weighting and pool-depletion**.

### Numpy `particle_mc` (rejection-as-rejection)

Each iteration of the `while n_accepted < n_max_mergers` loop:

1. Draw a candidate pair `(i, j)` from the active pool.
2. Roll `u ~ Uniform(0,1)`; **accept** the pair iff `u < K_ij / K_max`.
3. **Only on acceptance:** advance retention/remnant physics, mutate
   `active[]` (light slot off; heavy off too if not retained), append
   the merger record with weight ≡ 1.

Consequences:
* Pool depletion is **K-weighted**: only K-accepted pairs touch the
  pool, so heavy / equal-mass particles preferentially merge.
* `n_accepted` and `Σw_mergers` are the *same quantity* (every
  record has weight 1 under v1 uniform-weight path).
* Stopping criterion `n_accepted ≥ depletion_fraction · n_init` is in
  the same units as the pool-depletion target.

### JAX `particle_mc_jax` (rejection-as-weighting)

Each `jax.lax.scan` step (cannot use a data-dependent stop, so
`n_steps` is fixed):

1. Draw a candidate pair `(i, j)` from the active pool.
2. Compute `w_event = K_ij / K_max` and **record** the merger every
   step with that fractional weight.
3. Advance retention/remnant physics **every step**.
4. Mutate `active[]` **every step**: `idx_light` always deactivated;
   `idx_heavy` deactivated iff `~retain`.

Two things this design conflates:

> **(a)** Pool depletion happens **unconditionally** at every step,
>   regardless of `w_event`.
>
> **(b)** The rate-equivalent "amount of cascade" per step is only
>   `E[w_event] = ⟨K⟩ / K_max < 1`, not 1.

The intent of K-rejection-as-weighting was to keep the AD trace smooth
by avoiding the discrete Bernoulli rejection.  That intent is fine
for the **weight bookkeeping** (`Σw_mergers`, `w_active`, density
estimation) but it accidentally also drove the **dynamical state**
(pool depletion, remnant injection) — which still happens at full
strength regardless of weight.  The two should not share the same
loop step.

---

## 3. Quantitative match — the demo (Σw_mergers ≈ n_accepted)

The handoff's "factor-of-2-ish" gap in the demo follows directly.

```
Σw_mergers   ≈ n_steps · E[K_ij / K_max]   (every step records w_event)
n_accepted   ≈ depletion · n_init          (numpy stops at this cap)
n_steps      = int(depletion · n_init / 1.5)   ← JAX's heuristic
```

For the demo (depletion ≈ 1.0, n_init=20 000):

```
n_steps   = 13 328
Σw_mergers = n_steps · E[K/K_max] = 10 743   ⇒   E[K/K_max] ≈ 0.806
n_accepted = 19 988                ⇒   matches depletion · n_init ≈ 20k
ratio Σw_mergers / n_accepted ≈ 0.54  (= 13 328 / 19 988 · 0.806)
```

The deficit is exactly the factor `(1.5 · E[K/K_max])` you lose by
treating "scan step" as "accepted merger" in the same heuristic that
worked for numpy.

---

## 4. The integration-test 4.2× deficit — same root cause, shape-side

The integration-test failure (23.75 vs 100, 7.6 σ) **does not** come
from the `n_init²` total_rate rescaling — that fix from round 2 is
intact (`total_rate(log_R=0) = 0.412` is sane).  The arithmetic:

```
n_exp(θ*)         = 23.75
total_rate(log_R=0) = 0.412        (round-2 fix, sane O(1))
⇒ exp(log_rate_star) ≈ 57.6
⇒ implicit's total_rate(log_R=0) ≈ 100 / 57.6 ≈ 1.737
JAX / implicit ratio at log_R=0  ≈ 0.412 / 1.737  ≈ 0.24
```

So at the same θ* and the same `log_R_env=0`, JAX MC under-predicts
the rate integral by ~4×.

### Why pool-depletion symmetry matters here

`total_rate = ½ exp(log_R) ∫∫ K(m₁,m₂) N̄(m₁) N̄(m₂) dm₁ dm₂` with
`N̄(m) = ∫ N(m,χ) dχ`.

For the implicit Picard fixed point, the integral
`∫N dm dχ = (1 − f_dep)`, and — critically — **the shape of N(m,·) is
K-weighted**: heavy / equal-mass particles preferentially merge in
the engine's loss term, so by f_dep = 0.10 the surviving N has a
characteristic depletion in the high-K region and enhancement in the
remnant-mass region.  `∫∫ K · N · N` picks that structure up.

For JAX MC at the same target `mc_depletion_fraction = 0.10`:
* `W_total/n_init` ≈ `1 − f_dep` ≈ 0.87 — **scale ~correct**.
* But each scan step deactivates `idx_light` regardless of `K_ij`, so
  **light-slot removal is uniform random over the active pool**, not
  K-weighted.  The (m,χ) distribution of the surviving pool stays
  close to the natal Gaussian instead of developing the
  cascade-product structure.

The KDE then gives an `N̄(m)` that's essentially `(W_total/n_init) ·
gaussian(mu_natal, σ_natal)` — broader than implicit's K-sculpted
post-cascade marginal and with no equal-mass enhancement.  Plug that
into the K-weighted double integral and the factor of ~4× is
plausible (sharply-peaked kernel × shape mismatch ≫ (1−f_dep)²
scale mismatch).

**Same root cause as the demo line.**  Recouple K to pool dynamics
and both numbers move together.

---

## 5. Proposed fix (three options, ordered by surgical reach)

### Option A — Time recalibration only (smallest change)

Change `n_steps` so `Σw_mergers` matches the same "cascade evolution"
as numpy's `n_accepted`:

```python
# Old:
n_steps = max(1, int(cfg.depletion_fraction * n_init / 1.5))
# New (analytic estimate of K_max / ⟨K⟩ from natal moments):
K_max_over_Kmean = estimate_K_ratio(cfg)       # see § 5.1
n_steps = max(1, int(cfg.depletion_fraction * n_init / 1.5 * K_max_over_Kmean))
```

**Pros:** one-line change, fully AD-traceable (constants).
**Cons:** Σw_mergers will now match n_accepted, but **pool depletion
in the JAX cascade will overshoot** (more steps → more
unconditional light-slot kills).  The (m,χ) shape will be even
flatter than today.  Likely fixes the demo line but **makes the
integration test worse**.

Verdict: not enough — addresses (b) but worsens (a).

### Option B — Decouple K from pool dynamics (the principled fix)

Inside `step_fn`, gate the **state mutations** by `w_event`
multiplicatively rather than always-on.  Two flavours:

**B.1 — Stochastic gate (Bernoulli, AD-traceable via straight-through).**
```python
u = jax.random.uniform(key_u)
accept_event = u < w_event           # mirror numpy's K-rejection
do_retain    = step_active_now & retain & accept_event
do_drop_both = step_active_now & (~retain) & accept_event
# light slot only deactivated if accept_event
idx_light_update = jnp.where(step_active_now & accept_event, False, ...)
# w_event recording for density estimation:
# under stochastic gating, accepted events have unit weight; Σw_mergers
# = n_accepted exactly, matching numpy semantics.
w_desc = 0.5 * (w_i + w_j)
```

Gradients through the Bernoulli use either (i) a straight-through
estimator (zero-bias for the K-weighted shape, biased on the
hard-accept boundary), or (ii) gumbel-softmax with low temperature.
J2 picked rejection-as-weighting precisely to avoid this — the
trade is "smooth AD vs. faithful dynamics".

**B.2 — Soft gate (deterministic K-fractional pool dynamics).**
Instead of fully deactivating the light slot, **scale its weight** by
`(1 − w_event)`.  After many steps this is mathematically
equivalent to "a w_event fraction of the slot was consumed".  The
remnant slot enters with weight `w_event · 0.5·(w_i+w_j)`, **and the
parent slots' weights are reduced by w_event** (light → 0,
heavy → w_heavy·(1−w_event) + remnant_path).

**Pros:** fully deterministic + AD-smooth (no Bernoulli); makes pool
depletion K-weighted matching numpy in expectation.
**Cons:** the "fractional particle" abstraction blurs the
particle-MC interpretation; needs careful re-derivation of the
W_total invariant.  Tests touch `w_active` semantics.

Verdict: B.1 with straight-through is the closest mirror of numpy and
is likely the minimum-surprise fix; B.2 is the right answer if you
care about smooth gradients more than fidelity to particle-MC
semantics.

### Option C — Importance-weighted pool with K-aware step count

Keep the current always-on pool dynamics, but:
1. Multiply `n_steps` by `K_max/⟨K⟩` (Option A's number) so Σw_mergers
   matches in expectation.
2. **Reweight each step's pool-depletion contribution** by `w_event`
   so the cumulative pool depletion only advances at the rate-
   weighted pace.  Concretely: track an array
   `active_weight ∈ [0,1]` that decays by `w_event` per touch rather
   than being binary-flipped.

This is essentially B.2 with the K_max/⟨K⟩ step-count rescale baked
in.  AD-smooth, faithful, but a bigger rewrite.

### § 5.1 — Estimating `K_max / ⟨K⟩` for Option A or C

`K(m₁,m₂) = (m₁+m₂)^a q^b`.  At step start, `K_max =
upper_bound(m_min_active, m_max_active, a, b)`.  `⟨K⟩` over uniform
pair sampling from the active pool is a moment of the natal mass
distribution that can be evaluated analytically for the Gaussian
natal (∫∫ K(m₁,m₂) φ(m₁) φ(m₂) dm₁ dm₂).

For the live cascade `⟨K⟩` drifts as the pool shape changes, but the
demo's 0.806 value tells us the ratio is O(1), so a single
factory-time estimate is good to ~10 %.

---

## 6. Predicted effect on the failing surfaces

Assuming Option B.1 (Bernoulli K-gate, straight-through gradients):

| Surface | Today | After fix | Mechanism |
|---|---|---|---|
| Demo `Σw_mergers` vs numpy `n_accepted` | 10 743 vs 19 988 (ratio 0.54) | ≈ matches in expectation (Poisson noise only) | accepted events now have unit weight, fully analogous to numpy |
| KS-raw `p` (m_rem retained) | 0.478 | ↑ (bias removed: no more low-K over-rep) | accepted-event histogram matches numpy's |
| KS-weighted-resample `p` | 0.17 | similar or ↑ | already apples-to-apples; gets a free boost from larger effective N |
| Integration test `n_exp(θ*)` | 23.75 (7.6 σ low) | → ~100 (within 5 σ) | K-weighted depletion produces the right N̄(m) shape; `∫∫ K · N̄²` recovers implicit's value |
| SVI gradient through K-rejection | smooth (current) | smooth (straight-through) or smooth (Option B.2) | straight-through preserves AD; B.2 is fully smooth |

### Risks
1. **Straight-through bias** on the K-rejection boundary may introduce
   a small SVI bias on `kernel_a`, `kernel_b`.  Likely smaller than the
   KDE-bandwidth bias already documented in
   `[[feedback-gaussian-kde-over-smooths-cascade-peaks]]`.  Sandbox
   gate: compare θ̂ under rejection-as-rejection (numpy) vs
   straight-through JAX over a 20-seed batch.
2. **Stop-criterion drift.**  With K-gating, the *expected* number of
   accepted events in `n_steps` is `n_steps · E[K/K_max]`, so to hit
   `depletion · n_init` accepted events the JAX path also needs the
   `K_max/⟨K⟩` step-count rescale from § 5.1.  Without it, the JAX
   cascade undershoots `depletion` in expected accepted count.
3. **B.2 only:** `W_total` no longer equals `Σ_{active slots} w_i`
   under the simple definition; needs `W_total = Σ active_weight_i`,
   which trickles into the KDE bandwidth's `n_eff` definition.

---

## 7. What I want to do, pending your green-light

1. Confirm the diagnosis empirically — run the existing
   `tests/test_mc_distribution_log_prob.py` and `test_mc_particle_mc_jax.py`
   sandbox tiers with a printf-style instrumentation patch that
   reports `(n_init, n_steps, Σw_mergers, n_active_final, W_total)`
   for both numpy and JAX at matched config.  No commit.
2. Estimate `K_max/⟨K⟩` analytically for the Stage 1 θ* and
   cross-check it against the demo's empirical 0.806.
3. Implement **Option B.1** (Bernoulli K-gate with
   straight-through gradients) as the minimum-surprise fix.  Land
   the K_max/⟨K⟩ step-count rescale at the same time.
4. Sandbox + slow-tier gates:
   * `tests/test_mc_particle_mc_jax.py` — bit-stability against
     numpy on retention rate and merger count distribution.
   * `tests/test_mc_time_calibration.py` — re-check ε ↔ MC-state map
     hasn't drifted.
   * `tests/test_integration_stage1_recovery_mc_jax.py::
     test_expected_count_at_truth_matches_n_obs` — should pass at 5 σ.
5. Re-run `demo/forward_modeling_mc_vs_mc.py --n-seed 20000` and
   report the KS-raw / KS-weighted / Σw_mergers numbers.
6. Update `[[project-jax-mc-j2-done]]` + write
   `[[project-jax-mc-workstation-round3]]` memory.

---

## 8. B.2 feasibility — memory + AD check (added round-3 in-flight)

User asked: is B.2 (soft K-gate) memory-feasible before we commit
to B.1?  Answer: yes, with caveats.

**The architectural issue.**  In a faithful soft K-gate, each scan
step can produce a fractional remnant at a `(m_remnant, χ_remnant)`
that differs from both parent slots.  Collapsing this back to the
heavy slot via a weighted blend (`m_new = (1 − w_event)·m_heavy +
w_event·m_remnant`) gives a "half-merged" mass that has no physical
interpretation.  The principled path is option (b) from § 5:
**pre-allocate `n_init + n_steps_max` slot arrays** and write each
step's fractional remnant into the tail.

**Memory footprint estimate.**

| Config | slots today | slots under B.2-(b) | Δ memory |
|---|---|---|---|
| SVI integration test (n_init=2000, depletion=0.10) | 2 000 | ≈ 2 250 | +12 % |
| Demo (n_init=20 000, depletion=1.0) | 20 000 | ≈ 40 000 | +100 % |
| Hypothetical production (n_init=100 000, depletion=0.5) | 100 000 | ≈ 150 000 | +50 % |

At 32 bytes/slot (m, χ, w, gen as float32) the worst case is ~6 MB
for production — **not a memory explosion**.

**Runtime.**  KDE is O(n_samples × n_queries).  Doubling slot count
doubles per-density-eval cost.  SVI throughput halves at the worst
end.  Tolerable.

**AD.**  B.2 is fully smooth (no straight-through, no Bernoulli),
which is the J2 design's original aspiration.  Wins over B.1 on
gradient quality near the `K_ij ≈ K_max` boundary.

**The cost.**  `jax.lax.scan` needs fixed-shape carries.  Growing
arrays mid-scan requires pre-allocating the remnant tail upfront and
using a write-pointer index.  That's a moderate rewrite of
`particle_mc_jax`'s scan body and `MCResponseJax.build` (the
`active_mask`-based weight-zeroing trick wants a refresh — remnant
tail slots are "born" at step k with weight `w_event` rather than
"deactivated" later).

### Verdict
B.2-(b) is the right long-term answer.  Memory is fine; runtime is
acceptable; AD is cleaner than B.1's straight-through.  But the
rewrite is bigger than B.1 — slot allocation, scan-body
restructure, KDE bandwidth re-derivation (`n_eff` over the augmented
sample), test surface.  Plan: **ship B.1 + the K_max/⟨K⟩ rescale
first** as the minimal change that fixes the failing tests, then
follow up with B.2-(b) as a separate session once we've confirmed
the fix unblocks the M4 gate.

Logged as `[[project-jax-mc-b2-soft-gate-followup]]` for the
follow-up.

---

## 9. Open questions for the user

Q1. **Option B.1 vs B.2 — your call.**  B.1 mirrors numpy
exactly (modulo stochastic vs. weighted recording).  B.2 keeps the
"every step is smooth" J2 design philosophy but blurs the
particle-MC interpretation.  Default to B.1 unless you want to
preserve full AD smoothness for downstream SVI.

Q2. The integration test's `_MC_DEPLETION = 0.10` was set with the
comment "At Stage 1 θ* the implicit's f_natal_depleted ≈ 0.10 by
construction".  Worth confirming this value experimentally — if
implicit's actual f_dep at θ* differs (say 0.20 or 0.05), part of
the 4.2× could be a depletion mismatch independent of the K-gating
issue.  Will check as part of the empirical confirmation in § 7.1.

Q3. Should the K_max/⟨K⟩ rescale be (a) a single factory-time
estimate, (b) re-estimated every step from the live `K_max` and a
running mean of `K_ij`, or (c) `Σw_mergers / n_steps` style
calibration with a post-hoc step-count adjustment?  (a) is simplest,
(c) is most faithful to the demo's empirical 0.806.
