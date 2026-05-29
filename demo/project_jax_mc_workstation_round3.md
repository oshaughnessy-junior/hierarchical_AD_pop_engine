# [[project-jax-mc-workstation-round3]]

**Date:** 2026-05-29  
**Branch:** `observables` (continues from round 2's `f2b7b21`)  
**Predecessors:** `[[project-jax-mc-workstation-round2]]`,
`[[project-jax-mc-j2-done]]`, `[[demo/PLAN_jax_mc]]`  
**This-session detail memo:** `[[demo/PLAN_jax_mc_time_decoupling_diagnosis]]`

---

## Headline

Two failure modes traced to a single algorithmic asymmetry between
`particle_mc` (numpy) and `particle_mc_jax` (JAX):

1. Demo `Σw_mergers (10 743)` vs numpy `n_accepted (19 988)` — factor
   ~0.54, was "factor-of-2-ish".
2. `test_expected_count_at_truth_matches_n_obs` failing at 23.75 vs
   N_OBS = 100 (7.6 σ low).

Root cause: **the J2 design (rejection-as-weighting) decoupled
K-weighting from pool depletion.**  Every scan step deactivated the
light slot regardless of `K_ij/K_max`, while numpy's K-rejection
guards the pool entirely.  Consequences: (a) `Σw_mergers` is short
by `⟨K⟩/K_max` per step, and (b) the (m, χ) shape of the surviving
pool stays close to natal Gaussian instead of developing the
K-sculpted (heavy / equal-mass) structure that the implicit Picard
fixed point has, so the ∫∫ K·N·N rate integral is structurally
under-counted.

Round-3 fix: **B.1 — replace rejection-as-weighting with Bernoulli
K-rejection + straight-through estimator for AD**, plus the
`K_max/⟨K⟩ → k_ratio` step-count rescale so the expected accepted-
event count matches numpy's `n_max_mergers`.

## Numbers (sandbox-verified this session)

| Surface | Pre-fix | Post-fix |
|---|---|---|
| Stage-1 SVI config (n_init=2000, depletion=0.10) | — | Σw=191 / numpy n_accepted=200 → **0.96** (target 1.0) |
| Pool depletion at Stage 1 | — | JAX 9.60 % / numpy 10.00 % |
| `W_total/n_init` vs `(1 − f_dep)` | — | 0.904 / 0.900 |
| `Σw_mergers / numpy n_accepted` at Stage 1 | 0.54 | 0.96 |
| Sandbox tests (`test_mc_particle_mc_jax`, `test_mc_time_calibration`) | — | **34/34 green** |
| AD `∂Σw_mergers/∂kernel_b` vs soft-surrogate FD (INV-7) | exact (smooth) | within 22 % (straight-through) |

`test_expected_count_at_truth_matches_n_obs` wasn't directly
runnable here (gwkokab not in sandbox), but the workstation
re-check the user ran after the edit lands the JAX cascade where
the implicit's Picard sits, so the K-weighted rate integral should
now recover N_OBS = 100 within the 5 σ tolerance.  **Confirm on
workstation before closing M4.**

## Why this matters

`[[project-jax-mc-j2-done]]` shipped rejection-as-weighting
explicitly because the J2 PR's stated goal was "smooth gradients
through kernel_b for SVI."  That goal is preserved in round 3 —
straight-through routes `∂(K/K_max)/∂kernel_{a,b}` through the K-gate
even though the forward pass is now hard Bernoulli — but the unstated
cost of the old design (pool dynamics blind to K) was paying for the
smoothness in shape distortion of the surviving (m, χ) distribution.
B.1 is the minimum-surprise fix that pays the smoothness cost
explicitly (a small straight-through bias on `kernel_{a,b}`) instead
of paying it implicitly in the rate integral.

## The k_ratio "magic factor" risk — explicit warning

`k_ratio = K_max / ⟨K⟩` over the natal pool sizes `n_steps` so the
expected accepted-event count matches numpy's
`n_max_mergers = depletion · n_init`.  **It is regime-dependent**.
Empirical values measured this session for the canonical Gaussian
natal (μ=20, σ=3) × power-law kernel (a=0, b=2):

| Regime | k_ratio needed for Σw_jax / n_accepted_numpy ≈ 1.0 |
|---|---|
| Stage-1 SVI (depletion=0.10) | ≈ 1.5 (sandbox: 0.96 at default 1.5) |
| Demo full cascade (depletion=1.0) | ≈ 3.5 (sandbox sweep: 1.001 at 3.5) |

**Default in code: `k_ratio = 1.5`.**  This is *correct for Stage 1
SVI* and *under-budgets the demo*.  If the setup changes — different
natal shape, different kernel, different depletion target — the
default will drift.  The drift mode is: too-low k_ratio → too few
accepted events → JAX cascade undersamples → expected_count
under-predicts.

### How to set k_ratio for a new regime

Run a single eager probe (no AD needed — k_ratio is a control-flow
constant):

```python
# at SVI init, with concrete-θ defaults
import numpy as np
from red_team.backup_particle_mc import (
    MCConfig, kernel_K, _kernel_upper_bound,
)

n_probe = 4096
rng = np.random.default_rng(0)
# truncated-normal natal draws from the prior mean
# (... your natal sampler ...)
m1, m2 = natal_pool[:n_probe], natal_pool[n_probe:2*n_probe]
K_vals = kernel_K(m1, m2, a, b)
K_max  = _kernel_upper_bound(natal_pool, a, b)
k_ratio_est = K_max / float(np.mean(K_vals))
# pass into the factory:
truth = build_gaussian_coagulation(..., mc_jax_k_ratio=k_ratio_est)
```

At depletion → 1, the running cascade pool's `K_max/⟨K⟩` drifts
upward (heavy / equal-mass survives, lightest die first); the
natal-time estimate underestimates the cascade-averaged ratio.  At
depletion ≪ 1 the natal-time estimate is good.  For SVI under low-
depletion targets, the natal-time estimate suffices; for forward-
modelling demos at higher depletion, add a small safety multiplier
(empirically ~2× the natal estimate to land in numpy parity).

### Why not auto-derive inside the trace

Under SVI, `mu_natal` etc. are JAX tracers — can't be cast to a
Python int for `jax.lax.scan(..., length=n_steps)` without forcing
a retrace per gradient step.  k_ratio is therefore *deliberately
static* at factory time.  This is the pragmatic shortcut, not the
principled answer.  Principled answer = **B.2** (next item).

## Long-term follow-up — B.2 (logged, not yet started)

`[[project-jax-mc-b2-soft-gate-followup]]`

B.2 = soft K-gate with deterministic fractional pool dynamics, option
(b) pre-allocate `n_init + n_steps_max` slot arrays with
write-pointer.  Per `[[demo/PLAN_jax_mc_time_decoupling_diagnosis]]`
§ 8: memory footprint ≤ 2× current (≈ 6 MB at n_init=100 k);
runtime ≤ 2× per density-eval; AD fully smooth (no straight-through
bias on `kernel_{a,b}`).  Removes the k_ratio magic factor entirely:
under B.2 the cascade's pool-depletion-per-step is already K-weighted
deterministically, so `n_steps` only needs to be sized for "how much
cascade evolution do I want", not "what's K_max/⟨K⟩".

**Recommendation:** ship round 3 (this session, B.1 + k_ratio) to
unblock the M4 gate now; commit B.2 in a separate session once
forward-modelling demos confirm round-3 numbers are robust across
the natal-IMF / kernel sweep.

## Other open questions

Q1. **Is `_MC_DEPLETION = 0.10` actually matching implicit's
`f_natal_depleted` at θ*?**  The integration-test comment claims it
does; I couldn't verify in this session (no gwkokab in the
sandbox).  Worth confirming explicitly — if implicit's f_dep at θ*
is actually 0.20 or 0.05, part of the round-2 4.2× deficit was a
depletion-mismatch separate from the K-decoupling.  One-line check:

```python
truth_impl = demo.build_truth_model_and_calibrate_rate(N_OBS, grid=fit_grid)
print("implicit f_natal_depleted at θ*:", float(truth_impl.f_natal_depleted))
```

If this is far from 0.10, retune `_MC_DEPLETION` to match before
relying on the round-3 numbers.

Q2. **The user's "interaction time" framing extends to other
solvers.**  The Smoluchowski "ε" coupling (red_team/mc_time_calibration)
maps engine-ε ↔ MC-state via the depleting-vs-non-depleting argument.
Under B.1 the MC's pool depletion is now K-weighted (like
the engine), so the ε ↔ f_dep map should agree more tightly.  The
sandbox `test_mc_time_calibration::TestClosedFormIdentities` all
still pass — closed-form identities aren't affected.  But the
slow-tier `TestCalibrationSweepWorkstation` rerun on the
workstation is worth doing to verify the MC sweep stays inside the
5 % rel-L1 budget after the algorithmic change.

## What changed in code

| File | Change |
|---|---|
| `red_team/backup_particle_mc.py` | `particle_mc_jax`: Bernoulli K-gate via uniform draw, `st_accept` straight-through estimator, gated state mutations on K-accept, removed `w_event` multiplier from `w_desc`; new `n_steps` default `= int(depletion · n_init · k_ratio)`; added `n_accepted` to stats |
| `src/hierarchical_ad_pop_engine/response/mc_jax.py` | `build_mc_jax_response` accepts `k_ratio` (default 1.5), threads to `particle_mc_jax` |
| `src/hierarchical_ad_pop_engine/response/coagulation.py` | `build_gaussian_coagulation` accepts `mc_jax_k_ratio` (default 1.5), threads through |
| `tests/test_mc_particle_mc_jax.py` | INV-1 expected shape `(13, 8) → (30, 8)`; INV-3 rewritten for binary `w_mergers ∈ {0, 0.5(w_i+w_j)}` semantics; INV-7 rewritten to compare AD vs **soft-surrogate FD** (since straight-through targets the smooth gradient, not the hard one) |
| `demo/PLAN_jax_mc_time_decoupling_diagnosis.md` | Diagnosis memo with root cause, fix options A / B.1 / B.2, predicted effects table, B.2 feasibility analysis |
| `demo/project_jax_mc_workstation_round3.md` | This memory file |

## Workstation re-run sequence

```
# 1. Sandbox first (any workstation will run these in < 30 s)
pytest tests/test_mc_particle_mc_jax.py tests/test_mc_time_calibration.py -v

# 2. The failing integration test (the headline acceptance gate)
pytest -m slow -k "integration_stage1_recovery_mc_jax" -v

# 3. Demo with the higher k_ratio for depletion=1.0 (numpy parity check)
python demo/forward_modeling_mc_vs_mc.py --n-seed 20000
# (if forward_modeling has a knob for k_ratio, set it to 3.5 there; if
# not, the default 1.5 will give Σw ≈ 0.68 × n_accepted_numpy — expected
# at full depletion per the round-3 sweep)
```

Expected M4 SVI integration test outcome under round 3:

* `test_expected_count_at_truth_matches_n_obs` **PASS** (was 7.6 σ
  fail; round 3 puts JAX cascade in implicit's f_dep regime).
* `test_svi_elbo_improves_over_training` **PASS** (no change —
  unaffected by the cascade-time fix).
* `test_all_truth_params_in_svi_credible_interval` **XFAIL** (still
  inherits the implicit's resolution-limited bias + KDE-oversmooth
  story per
  `[[feedback-gaussian-kde-over-smooths-cascade-peaks]]`).
* `test_posterior_samples_are_finite` **PASS**.

## Bottom line for handoff

Round 3 ships B.1 + k_ratio rescale as the minimum-surprise fix.
Both failing surfaces should be addressed.  The **`k_ratio = 1.5`
default** is the one magic factor introduced — *correct for low-
depletion SVI fits, under-budgets at high depletion* — and is
flagged here so the next session catches it if the setup moves.
B.2 (no magic factor, smoother AD) is logged as
`[[project-jax-mc-b2-soft-gate-followup]]` for a separate session.
