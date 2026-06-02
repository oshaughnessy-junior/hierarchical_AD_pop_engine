# PLAN — proper measurement model (reuse gwkokab; don't reinvent)

**Date:** 2026-06-01. **Why:** the cross-method harness scattered events with NO
measurement model (`scatter_to_posteriors` + `ln_offsets=0`, samples around
TRUTH) → biased natal widths. Fixed by defaulting to EXACT injection (f9c4856).
This plan wires a CORRECT synthetic-PE measurement model when we want one, using
gwkokab's existing machinery. **Paths below are in the sibling repo
`../gwkokab/src/gwkokab/` (source is checked out alongside the engine).**

## The two pieces a correct measurement model needs

1. **Injection (generate PE):** for each exact event, draw a single per-event
   peak offset (the NOISY DATUM `d = ω_true + obs_noise`) THEN per-sample PE
   spread about `d`. My broken `scatter_to_posteriors` had only one noise draw
   (samples around truth) and NO peak offset — that's the bug the user flagged
   ("a noise offset for the peak that's self-consistent").
2. **Recovery (deconvolve):** the population term must be
   `Σ_n log[ (1/K) Σ_k λ(ω_{n,k}) / π_ref(ω_{n,k}) ]`, i.e. **`ln_offsets =
   −log π_ref`** (the reference prior the samples were drawn under). With the
   noisy-datum structure + correct ln_offsets the convolution cancels between
   data and model → TRUE widths recovered.

## gwkokab already implements BOTH — reuse it

### Injection side — `../gwkokab/src/gwkokab/errors.py`
* `truncated_normal_error(x, size, key, *, scale, estimates, rho, low, high)` —
  `samples = x + scale·(r0 + r)·(12/rho)` where **`r0` = the single per-event
  peak offset (the noisy datum), `r` = the K per-sample PE spread**, SNR-scaled
  (`12/rho`), with reflection at `low/high`. THIS is the self-consistent
  delta→gaussian. (errors.py:99-167)
* `banana_error(...)` — realistic correlated (Mc, eta) error (Fishbach/Farah).
  (errors.py:24)
* `mock_spin_error(chi_eff, eta, ...)` — SNR-scaled chi_eff error via the
  ψ-reparametrisation, TruncatedNormal in ψ then back to chi_eff. (errors.py:214)

### Injection orchestration — `../gwkokab/src/gwkokab/analysis/core/synthetic_pe.py`
* `SyntheticDiscretePE` — full pipeline: delta injections → SNR-scaled PE
  samples → per-event HDF5 (`injection_data` + `posterior_samples` + `rho`).
  `error_function_registry` maps params → error fns (banana for (Mc,eta),
  mock_spin for chi_eff, generic truncated-normal for the rest).
  `generate_parameter_estimates()` is the entry. (synthetic_pe.py:36-422)
* `SyntheticAnalyticalPE` (585) — analytical-PE variant.
* `calculate_js_metrics` / `js_divergence_from_histograms` (504-583) — reusable
  JS, could feed our H1 gate.

### Recovery side (deconvolution ALREADY there) — `../gwkokab/src/gwkokab/inference/`
* `poissonlikelihood_utils.py:85`: `log_prob = batch_model_log_prob −
  batched_log_ref_priors` — the sage/discrete path SUBTRACTS log π_ref.
* `poissonlikelihood_utils.py:120-140` `analytical_poisson_likelihood_fn`: adds
  `ln_offsets` inside the logsumexp (`:136`) ⇒ `ln_offsets = −log π_ref`.
* `analysis/core/monk.py:259`: `ln_offsets =
  data_loader.sample_transformer.log_abs_det_jacobian(...)` — gwkokab derives
  ln_offsets from the reparametrisation Jacobian + reference prior. STUDY THIS
  to see exactly what π_ref it assumes and how the transform is built.

## THE CATCH — coordinate systems (this is the real work)

gwkokab errors live in **(chirp mass Mc, symmetric mass ratio η, χ_eff)** +
spin-component params. OUR engine + bridge events are **(m1, m2, χ1, χ2)**. So:
* need `(m1,m2) ↔ (Mc, η)` and `(χ1,χ2) ↔ χ_eff` transforms (and back), and
* the **Jacobian** of that transform enters `ln_offsets` (the π_ref in the
  sampled coords vs the engine's evaluation coords). monk.py's
  `sample_transformer.log_abs_det_jacobian` is the pattern.
This coordinate + Jacobian layer is the bulk of the effort — the error physics
and the deconvolution are done.

## Wiring plan (next session, after exact-injection re-runs are the baseline)

1. **Decide scope.** Simplest correct version: a 1-coordinate-consistent
   measurement model in the engine's own (m1,m2,χ1,χ2) coords using
   `truncated_normal_error` per axis (gets the r0 peak-offset right), with
   `ln_offsets` = the log-Jacobian/π_ref for whatever sampling prior we choose
   (flat-in-(m1,m2,χ) ⇒ ln_offsets from that prior only). Avoids the Mc/η detour
   for a first proof. Full realism = use `SyntheticDiscretePE` (banana + mock
   spin) + the Mc/η transform + Jacobian.
2. **Generate PE** from the exact MC-injected events (`pair_sample_from_mc`
   output) via the chosen error model; pick an SNR `rho` (fixed, e.g. 12-20, or
   drawn from a detection-weighted SNR distribution).
3. **Thread `ln_offsets`** (≠ 0) into `build_logL` / the bridge call
   (`fit_two_pulse_gwkokab` / `fit_stage1_gwkokab` currently pass
   `ln_offsets = zeros`).
4. **VERIFY (the gate):** a test that the synthetic-PE catalogue RECOVERS the
   true natal widths (σ₁, σ₂, spin_scale) to within MC error — i.e. the
   deconvolution works. THIS is the acceptance criterion; without it we're back
   to the bug. Cross-check: exact-injection vs measurement-model recovery should
   agree on the widths (the measurement model must not bias them).

## Anti-goals
* Don't scatter without `ln_offsets = −log π_ref` (the bug we just fixed).
* Don't hand-roll the error physics — `gwkokab.errors` has banana / mock-spin /
  truncated-normal with SNR scaling already.
* Don't conflate the measurement model with the forward-model resolution study
  (H1-JS) — keep them separate (run exact-injection for forward-model).

## Pointers
Engine side to change: `demo/fit_two_pulse_gwkokab.py`,
`demo/fit_stage1_gwkokab.py` (the `ln_offsets = jnp.zeros(...)` lines + the
scatter step), `src/.../inference/gwkokab_bridge.py` (already solver-agnostic).
gwkokab side to reuse: `errors.py`, `analysis/core/synthetic_pe.py`,
`inference/poissonlikelihood_utils.py`, `analysis/core/monk.py` (ln_offsets).
Memory: `project_implicit_pivot_bridge_wired`.
