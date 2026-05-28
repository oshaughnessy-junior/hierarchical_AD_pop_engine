# PLAN — implicit-mode footprint refactor (separable angular surrogate)

Written 2026-05-27 in response to user-reported timing + memory
collapse of the implicit solver under angular marginalisation
(`[[project-session-handoff-20260527-late]]`). The 8× quadrature
speedup landed in `[[project-orientation-quadrature-defaults-flip]]`
removed a constant; this PLAN attacks the asymptotic.

## What the implicit hot loop currently does

Per Picard iteration, both sides of the fixed-point map
`N → N_natal + ε·(gain[N,N] − loss[N])` scan the orientation
quadrature internally. See `response/coagulation.py:
_step_one_generation_marginalised` (gain side) and
`remnants/marginal.py:P_retain_marginalised` (loss side, after
the Session C 19cc0b6 lax.scan refactor —
`[[project-misalignment-session-c-done]]`).

Inside the scan body the integrand is a 4-D `(N_m, N_χ, N_m, N_χ)`
progenitor mesh that bilinear-scatters onto the 2-D `(N_m, N_χ)`
output grid via a vector remnant evaluation. At the new default
`(n_θ, n_Δφ) = (4, 8) = 128` points × Picard ~20 iters, the compute
is `O(128 × N_m² × N_χ² × n_iter)` per solve, and the autodiff
traces a tape proportional to that. With κ as a free parameter, AD
also unrolls a parallel tape for `∂/∂κ`.

For Aligned, this all short-circuits to a single bit-stable point —
the v1 path is preserved. The pain is **only** under non-Aligned.

## Key observation that unlocks the refactor

The progenitor density `N(m, |χ|)` carries **no orientation
information** under the misalignment convention
(`[[project-misalignment-session-b-done]]` — under non-Aligned, the
engine χ-grid is interpreted as `|χ|`; the orientation distribution
carries the tilt). Orientation is a **pair-level kinematic
quantity** that enters only via `chi_rem_vec` and `v_kick_vec`.

That means: the orientation integral

  `⟨ chi_rem(m1, m2, |χ1|, |χ2|, θ̂1, θ̂2) ⟩_{P(θ̂)}`
  `⟨ retention(v_kick(m1, m2, |χ1|, |χ2|, θ̂1, θ̂2), v_esc) ⟩_{P(θ̂)}`

are functions of **only the progenitor scalars + v_esc + the
orientation-distribution hyperparameter κ** (Aligned: trivial;
Isotropic: no hyperparameter; Fisher: κ).

Crucially: the inner Picard loop iterates over `N`, not over θ. The
orientation marginalisation **factors out of the iteration**. It can
be precomputed once per solve (cached on the response), reused at
every Picard iter. AD-on-κ is a one-shot pullback through the
precomputed tensor; AD-on-(m, χ, v_esc) is a pullback through a
4-D-scalar surrogate.

## Hard constraints honoured

* `[[feedback-no-fake-physics-stand-ins]]` — the surrogate is a
  controlled-error spectral expansion of the *true* orientation
  integral, not a free smoothing knob. Convergence is provable.
* `[[feedback-no-fudge-factors]]` — the surrogate's rank is set by
  a convergence criterion against the v1 (4, 8)-point quadrature,
  not tuned against a target observable.
* Aligned bit-for-bit preserved at every entry point — the dispatch
  shortcut already in place at every call site
  (`[[project-misalignment-session-b-done]]`) is untouched. Guard
  test: `test_aligned_explicit_equals_default_bitwise`.
* `[[feedback-perturbative-unreliable-implicit-production]]` — this
  PLAN **adds** a faster path additively; the existing scan-based
  marginalised path remains as the validation backstop.
* `[[feedback-dispatch-callgraph-audit]]` — every new internal
  surface gets a grep across the codebase before flipping defaults.
* `[[feedback-no-frozen-test-references]]` — all parity gates are
  scan-vs-surrogate invariants.

## Refactor strategies, scored

Four possibilities; the PLAN sequences A → B → (C/D conditional on
A/B insufficiency).

### Strategy A — Precompute, don't surrogate

The orientation-marginalised quantities

  `chi_rem_marg[i1, j1, i2, j2; κ]      ← 4-D tensor in progenitor cells`
  `P_retain_marg[i1, j1, i2, j2, v_esc; κ] ← 5-D scalar`

are precomputed once at `CoagulationResponse.__init__` (or in a
cached method called by the solver entry point), then **reused at
every Picard iter**. AD-on-κ is a single pullback through the
precomputed tensor.

Memory budget at production `(N_m, N_χ) = (64, 20)`:
`chi_rem_marg`: 64² × 20² × 8 B = ~13 MB.
`P_retain_marg`: same × #v_esc (= 1 in Stage 1) = ~13 MB.

**This is the dominant fix.** Cost drops from
`O(128 × N_m² × N_χ²)` per Picard iter to `O(N_m² × N_χ²)`. The 128
factor reappears once at construction, not 20 times per solve.

But: `chi_rem_marg` is a **scalar** mean. The gain term needs the
*distribution* of χ_out under orientation marginalisation, not just
the mean — because a single progenitor (m1, m2, χ1, χ2) cell
scatters into many χ_out cells via the orientation-induced spread.
Storing the full scatter as a 6-D tensor
`G[i1, j1, i2, j2; i_out, j_out]` is
`64² × 20² × 64 × 20 ≈ 2 × 10^9` floats = 16 GB. **Not viable.**

Strategy A is therefore a partial fix — **complete for the loss
side** (retention is a scalar), but only the *mean* of the gain
side. A separate surrogate is needed for the gain-side χ-out
distribution.

### Strategy B — Spherical-harmonic surrogate for the gain-side χ-out distribution

Decompose the orientation-induced χ_out distribution

  `q(χ_out | m1, m2, |χ1|, |χ2|; κ) = ∫ δ(χ_out − χ_rem(..., θ̂)) P(θ̂; κ) dθ̂`

in a small orthonormal basis on χ_out. Two viable bases:

* **Legendre / Chebyshev** on the χ_out grid range (clean for a
  fixed grid); ~5–10 basis functions resolve any orientation-
  spread that the (4, 8) quadrature can resolve.
* **Hermite functions** centred at the aligned chi_rem value with
  an orientation-spread-dependent width; the leading term recovers
  Strategy A's mean.

Either way: store `c_r[i1, j1, i2, j2; κ]` for `r = 0, ..., R−1`
(typically R ≤ 10). Total memory:
`R × N_m² × N_χ² × 8 B = 10 × 64² × 20² × 8 ≈ 130 MB` — viable.

The gain kernel scatter becomes
`gain(m_out, χ_out) = Σ_r c_r[i1,j1,i2,j2; κ] × φ_r(χ_out)`
contracted against `K · N · N` over (i1, j1, i2, j2). The basis
functions `φ_r(χ_out)` are precomputed on the χ_out grid (a 2-D
tensor `R × N_χ`). The contraction is one matrix product per Picard
iter on top of Strategy A's mean.

For Fisher: κ enters only through the orientation distribution
weights, not the quadrature nodes (Session A architectural choice;
`[[project-misalignment-session-a-done]]`). The basis coefficients
are linear in `P(θ̂; κ)`, so `c_r(κ)` is itself a linear combination
of κ-independent moments. **AD on κ is a single jvp** through that
linear map.

**Convergence target**: the surrogate's rel-L1 error on
`solve_implicit_fixed_point` output, vs the existing scan path,
must be `< 0.5%` at the (4, 8) quadrature equivalent. R is set by
the smallest value that meets this gate — see Session F1 below.

### Strategy C — Spherical-harmonic decomposition of the orientation distribution itself

Expand `P(θ̂1, θ̂2; κ) = Σ_{ℓm,ℓ'm'} a_{ℓmℓ'm'}(κ) Y_{ℓm}(θ̂1) Y_{ℓ'm'}(θ̂2)`.
The integral `⟨ remnant_quantity ⟩` becomes a sum over harmonic
moments of the remnant quantity (κ-independent) weighted by `a(κ)`.

Pros:
* Fisher von Mises-Fisher has a *closed-form* spherical-harmonic
  expansion in terms of modified Bessel functions
  (`I_{ℓ+½}(κ)`). The decomposition is analytic — no quadrature
  error at all under Fisher.
* AD on κ is now AD through `I_{ℓ+½}(κ)`, which JAX has via
  `jax.scipy.special.ive`.
* The pair distribution `P(θ̂1, θ̂2)` factorises when the two spins
  are i.i.d.; the harmonic indices decouple. Even when they don't,
  the cross-moments are still finite-dimensional.

Cons:
* The remnant quantities `chi_rem_vec`, `v_kick_vec` are *not*
  low-degree polynomials in `θ̂_i`. Barausse-Rezzolla has terms
  like `(σ_1 + σ_2 · q²)/(1+q²)²` where σ_i are the spin
  projections — quadratic in `cos θ_i` overall, OK. But the
  Lousto-Zlochower kick has `v_⊥ = K · cos(φ_Δ + φ_offset)` —
  degree-1 in azimuthal harmonics. Then the retention sigmoid /
  Heaviside `1 − P_retain` is **non-polynomial in θ̂_i** so its
  harmonic expansion is infinite. Truncation at some ℓ_max is
  controlled but error-prone.
* The book-keeping is intricate; the implementation effort is
  ~2-3× Strategy B for a similar speedup.

Strategy C is preferred *if* Strategy B turns out to be too slow or
too memory-heavy at higher χ-resolution, or if Fisher with extreme
κ (≫ 50) is the production regime — at high κ the orientation
integrand is concentrated, and Strategy B's polynomial basis on
χ_out needs many functions to resolve a narrow peak, while
Strategy C's analytic Fisher expansion handles it natively.

**Defer Strategy C until Strategy A + B are benchmarked.**

### Strategy D — Surrogate net (last resort)

If A + B + C still don't hit the production memory/time budget, a
small MLP regression on the precomputed 4-D tensor would crush the
storage cost. This is a fake-physics-adjacent move (the surrogate's
extrapolation error is unbounded) and only acceptable as a
controlled-residual model: trained against the true quadrature, with
residual gate against the v1 path on a held-out (m1, m2, χ1, χ2)
fold. Defer until A+B+C have been measured insufficient.

## Sessions

### Session F1 — Strategy A (precomputed retention + mean chi_rem)

Goal: separate the orientation marginalisation out of the Picard
loop entirely, for the scalar quantities (retention,
mean chi_rem).

Deliverables:
* New module `src/hierarchical_ad_pop_engine/orientation_precompute.py`:
  * `precompute_retention_marginal(grid, orientation, retention_kind,
    v_esc, w_over_v_esc) -> Array[N_m, N_χ, N_m, N_χ]`
  * `precompute_chi_rem_mean(grid, orientation) -> Array[N_m, N_χ,
    N_m, N_χ]`
  * Internally uses the existing `P_retain_marginalised` and
    `chi_rem_at_orientation` scan path so it cannot drift — it's
    just **eager evaluation + caching** of the same surfaces.
* `CoagulationResponse` gains optional cached fields
  `_retention_marg_cache`, `_chi_rem_mean_cache` (eqx static fields,
  set in factory `build_default_coagulation` when `orientation` is
  not Aligned).
* `response/implicit.py:gain_density` and `loss_density` route
  through the precomputed tensors when present; fall back to the
  scan path when absent (preserves existing test coverage).
* `tests/test_orientation_precompute.py`:
  * Bit-for-bit Aligned shortcut (the cache is None under Aligned).
  * Precomputed loss-side equals scan-side to rel `< 1e-10` —
    these are literally the same integral.
  * AD-on-κ flows through the precomputed tensor.
  * Memory budget assertion: `P_retain_marg` array `nbytes` is
    within 2× the analytic bound `N_m² × N_χ² × 8 B`.

Expected speedup: the **loss side** scan disappears entirely. The
**gain side** still scans because the mean alone is insufficient —
Session F2 closes that.

Effort: 1 session.

### Session F2 — Strategy B (gain-side χ-out distribution surrogate)

Goal: collapse the gain-side orientation scan via a low-rank
spectral surrogate.

Deliverables:
* `src/hierarchical_ad_pop_engine/orientation_precompute.py` gains
  `precompute_chi_out_basis_coeffs(grid, orientation; R) ->
  Array[R, N_m, N_χ, N_m, N_χ]` and
  `precompute_chi_out_basis_functions(grid, R) ->
  Array[R, N_χ_out]`.
  * Default basis: shifted Chebyshev on [0, 1] (the χ grid range).
* `_step_one_generation_marginalised` gains a fast path: if
  the response carries a cached basis, replace the orientation scan
  with the contraction
  `gain(m_out, χ_out) ← Σ_r c_r[...] × φ_r(χ_out)`.
* `tests/test_chi_out_basis_surrogate.py`:
  * Convergence: rel-L1 of `solve_implicit_fixed_point` vs scan
    path falls monotonically with R; gates at R such that error
    < 0.5 %.
  * Aligned shortcut still bit-for-bit (basis is None under Aligned).
  * AD on κ flows through the basis coefficients linearly.

R-selection sub-task: at the start of Session F2 run a one-off
convergence sweep at Stage 1 θ\*; pin R as a class-level constant
documented by the convergence figure.

Effort: 1 session.

### Session F3 — Production-resolution benchmark + comparison_helpers wire-up

Goal: confirm the refactor crosses the user's production budget.

Deliverables:
* `red_team/benchmark_orientation_refactor.py` — measures wall time
  and peak memory of:
  1. The Session C scan-path (baseline).
  2. Session F1 (loss precomputed, gain still scanning).
  3. Session F1 + F2 (full surrogate).

  At three resolutions: `(N_m, N_χ) = (32, 16), (64, 20), (96, 32)`.

* `demo/figures_orient_refactor/` — speedup curve + memory curve.
* Sandbox-verified before commit; live workstation run measures the
  numbers that go in the figure.
* User-decision gate: if the production resolution `(96, 32)` is
  now solvable in `< 30 s` and `< 4 GB`, declare the implicit-mode
  rescue successful. If not, escalate to Strategy C (Session F4) or
  drop the implicit path for production and ride MC alone.

Effort: 1 session.

### Session F4 (conditional — only if F3 misses budget)

Strategy C: spherical-harmonic decomposition of `P(θ̂1, θ̂2; κ)`
combined with analytic Fisher moments. Larger surface, ~2-3
sessions, deferred until F3's measurement says it's needed.

## Cross-cutting decisions to lock at Session F1

* **Where do the precomputed tensors live?**
  Option (a): on `CoagulationResponse` as eqx static fields (jitted
  through, never traced). Pro: clean. Con: changing κ requires
  rebuilding the response.
  Option (b): on a separate `OrientationCache` pytree threaded
  through the solve. Pro: κ is a leaf, AD on κ flows naturally.
  Con: more plumbing.
  **Recommendation**: (b). κ is a model knob the user will want to
  fit.
* **Is the precompute itself jit-traced?** No — it runs once at
  construction, the cost is amortised. Jit only the inner Picard
  loop.
* **What happens under Aligned?** The cache is `None` (Python-time
  attribute), the Picard inner loop branches at trace time on
  `isinstance(response.orientation, Aligned)` and never touches the
  cache surface. Existing Aligned tests are bit-stable.

## Estimated calendar

| Session | Topic                                              | Risk |
|---------|----------------------------------------------------|------|
| F1      | Strategy A precompute (retention + mean chi_rem)   | Low  |
| F2      | Strategy B χ-out basis surrogate                   | Med  |
| F3      | Benchmark + gate decision                          | Low  |
| F4      | Strategy C — only if F3 misses budget              | Defer |

Three sessions to a viable implicit path under non-Aligned at
production resolution, with one optional bail-out session.

## Why not give up on implicit entirely

`[[project-orientation-quadrature-defaults-flip]]` already cashed
8× for free. Strategy A alone will eliminate the loss-side scan
contribution (~half the work). Strategy B eliminates the gain-side
scan contribution (the other half). The two together collapse the
128-point orientation overhead to **construction time**, which is
amortised over the entire MCMC chain.

Expected speedup at the gain side from Session F1 + F2 together vs
the current scan path: ~`(128 / R)` per Picard iter, with R ~ 10.
~13× on top of the 8× already cashed = ~100× faster than the pre-
quadrature-flip baseline. That is what gets the production
resolution `(96, 32)` running.

If the speedup is still insufficient, the user has a clean
fallback: ship MC-only inference for misaligned populations
(`PLAN_mc_first_reprioritization.md`), keep implicit for Aligned
where it already works.

## Related

* `[[project-session-handoff-20260527-late]]` — the timing pain
  motivating this PLAN.
* `[[project-orientation-quadrature-defaults-flip]]` — the
  in-place 8× speedup; this PLAN is the asymptotic complement.
* `[[project-misalignment-session-c-done]]` — the existing loss-
  side `lax.scan` over orientations (19cc0b6) that the precompute
  replaces.
* `[[project-misalignment-session-b-done]]` — the marginal-layer
  surfaces (`chi_rem_at_orientation`, `v_kick_at_orientation`,
  `P_retain_marginalised`) the precompute is built on.
* `[[project-engine-v3-step3-architecture]]` — the basis-decoupled
  density/solver pattern that Strategy B mirrors.
* `[[feedback-no-fake-physics-stand-ins]]` — surrogate is
  controlled-error spectral, not a free knob.
* `PLAN_mc_first_reprioritization.md` — the companion plan for the
  MC side.
