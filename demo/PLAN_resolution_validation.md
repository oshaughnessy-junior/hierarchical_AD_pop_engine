# PLAN — Resolution & control validation study

Written 2026-05-21.  Flagged from
`tests/test_integration_stage1_recovery.py` (the headline 95 % CI
gate is marked `xfail` pending the work in this plan).

## Why this exists

The synthetic-data integration test (`PLAN_integration_test_synthetic.md`)
runs the full Stage 1 pipeline (natal → cascade → likelihood → SVI)
on a single deterministic mock catalogue (N_OBS=100) and checks that
the demo's six truth parameters land inside the 95 % SVI credible
interval.  After alignment of (truth grid, fit grid, JAX precision,
PRNG seed) to the demo's known-good config, four of six parameters
recover inside CI; **`sigma_natal` and `kernel_b` remain biased by
≈25–30 %**, with the demo's seed in `artifacts_99cb35/` showing the
same residual bias when looked at carefully.

This is not a bug in any single primitive — it is a *combined*
limit of:

1. **`AutoNormal` mean-field guide width-underestimate.**  The
   guide is a product of independent Gaussians (in unconstrained
   space) and cannot represent the known degeneracy
   ``sigma_natal × kernel_b × log_rate``.  Mean-field finds a
   compromise location, not the joint mode.
2. **Coarse-grid quadrature.**  The fit grid is 64 × 20 on
   `jnp.geomspace(m_min, m_max)` × `jnp.linspace(0, 1)`.  Trapezoid
   quadrature on this grid does not reproduce the analytic
   `expected_event_count` integral to the precision the likelihood
   demands when constraining the *width* of the natal Gaussian.
3. **Boundary parameters.**  `natal_spin_loc = 0.0` sits on the
   `Uniform(0.0, 0.3)` lower edge; `AutoNormal` in unconstrained
   space cannot place posterior mass at the boundary.  The
   integration test now uses `natal_spin_loc = 0.05` to remove this
   specific structural failure, but the systematic study should
   verify that other prior-boundary effects are not lurking.

The integration test cannot resolve any of these by itself — they
are properties of the inference and quadrature setup, not the
plumbing the integration test gates.  Trying to "fix" them inside
the test would either be (a) fudging the CI width or (b) tuning the
prior boundaries — both empirical knob-twiddling we have explicitly
forbidden (see `feedback-no-fudge-factors`).

## Scope (this plan)

Reconcile observed inference behaviour against the physics-vs-resolution
limits *systematically*, by varying one control axis at a time and
characterising how the recovery moves.

### Control axes

| Axis | Knob | Range to study | Hypothesis |
|------|------|----------------|------------|
| Fit grid resolution | `(N_M_FIT, N_CHI_FIT)` | (32, 10), (64, 20), (128, 40), (256, 80) | Bias on `sigma_natal` shrinks like O(1/N_M) — confirms trapezoid quadrature is the limiting factor |
| Truth grid resolution | `BinaryGrid.default` size | as above, plus full v3 Chebyshev basis | Truth-vs-fit grid mismatch alone is a small effect; truth refinement alone won't fix the bias |
| Cascade depth | `K_MAX_TRUTH = K_MAX_FIT` | 2, 3, 4, 5 | Truth at higher `k_max` than fit should bias `kernel_b` upward (cascade pile-up looks like steeper pairing); test by sweeping |
| Sample size | `N_OBS` | 30, 100, 300, 1000, 3000 | All biases should shrink as `1/sqrt(N_OBS)` if the limit is purely statistical; if a floor persists, it's structural (guide or quadrature) |
| Guide family | `AutoNormal` vs `AutoMultivariateNormal` vs `AutoLowRankMultivariateNormal` | all three | If `AutoMultivariateNormal` recovers truth where `AutoNormal` does not, the bias is mean-field guide misspecification |
| Inference engine | SVI vs NUTS | both | NUTS does not assume mean-field; if NUTS recovers where SVI does not, the bias is the guide |
| JAX precision | float32 vs float64 | both | Currently float32 by demo default; should not matter at the bias level we observe, but worth confirming |
| Prior choice | `LogUniform(0.5, 10.0)` vs `LogNormal(0, 1.0)` for `sigma_natal` | both | Posterior shape under different priors |
| Initialisation | `init_loc_fn` for `AutoNormal` | default vs `init_to_value(truth)` | Local-minimum diagnosis |

### Acceptance for the study

The study is *not* a test — it is characterisation.  Deliverable:
a single notebook or markdown report
(`demo/REPORT_resolution_validation.md`) with:

* Table of recovery error vs each axis (median, 1-σ, 95 % CI width).
* Identification of the *dominant* limiting factor.
* Recommendation for the integration test's gate: either tighter
  config that recovers cleanly, or loosened gate with documented
  justification (e.g. "fit grid 64 × 20 cannot identify
  `sigma_natal` below 20 % — gate ≥ 25 %").

### What this plan deliberately does NOT do

* Add new physics.  This is purely characterisation of the
  existing pipeline's resolution behaviour.
* Optimise the engine for speed.  If we conclude that 256 × 80 is
  needed, that's the answer — speed work is downstream.
* Multi-seed averaging.  Each (axis, value) cell uses ONE
  deterministic mock catalogue.  Seed-induced variance is part of
  what we are characterising.

## How this lands in the test suite

When the study is done and we know which axis is responsible:

* If a tighter grid / different guide / larger `N_OBS` recovers
  cleanly: bump the integration test to that config and flip
  `xfail` to passing.
* If the recovery floor is genuinely > 25 % at N_OBS=100: loosen
  the gate explicitly (e.g. require median within 30 % of truth)
  with a documented reason that *traces back to this study*.
* Either way, the integration test stops being a placeholder.

## Connections

* `PLAN_integration_test_synthetic.md` — what gates we tried to
  put in place; this plan reconciles why some of them don't pass.
* `PLAN_engine_v3_basis.md` — v3 Chebyshev-Lobatto basis directly
  attacks axis (1) above; if v3 converges before this study runs,
  the study should start from v3 quadrature, not v2 grid.
* `[[feedback-no-fudge-factors]]` — the study replaces
  empirical-tolerance fudging with a measured characterisation.
* `[[feedback-no-frozen-test-references]]` — the study's
  deliverable is a *report*, not a checked-in numerical reference.

## Effort estimate

~1–2 sessions.  Most of the cost is wall-clock for the sweeps,
not engineering — each cell of the study is a re-parameterised
SVI run that reuses the existing helpers.
