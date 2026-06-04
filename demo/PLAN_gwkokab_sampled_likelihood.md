# PLAN — gwkokab sample-based (likelihood-mode) inference for MC populations

**Date:** 2026-06-04 (ROS direction, post chi-resolution rerun). **Status:**
DEVELOPMENT TASK SPEC — intended for a **separate agent** to develop on a
gwkokab fork, then thread back into the engine bridge. Not started.

## Motivation (why promote MC again)

The implicit Picard solver is hitting structural limits on the spin axis:

* **Ringing at n_chi=64** (workstation rerun 2026-06-04): visible
  oscillation at high spin as the grid model struggles to replicate the
  *completely empty* gap between the natal spins and the 1g remnant peak.
  With hundreds of sources in the catalog (current scale), such ringing
  could imprint artificial features on the recovered population —
  contamination risk at exactly the science target.
* **Deep-cascade limitation** (paper methods 2.B.3): implicit is
  effectively low-generation only; cost grows as (n_m·n_chi)^2 while
  features sharpen per generation.

The particle MC has neither problem — it produces **exact draws from the
model space naturally**, at any cascade depth, with no grid and no empty-
gap representation issue. What blocks using MC as the *recoverer* today is
that gwkokab's likelihood evaluates the population as a DENSITY:
reconstructing one from MC samples means KDE, the rabbit hole we
deliberately avoid (oversmoothing of narrow cascade peaks; see
`feedback_gaussian_kde_over_smooths_cascade_peaks`).

The cure is the dual formulation: **evaluate per-event likelihoods at
samples drawn from the model**, instead of evaluating the model density at
samples drawn from the events.

## The math (both directions of the same importance-sampling identity)

Per-event term in the inhomogeneous-Poisson likelihood:

    I_i(θ) = ∫ L_i(x) · p(x|θ) dx

* **Current gwkokab path (density mode):** draw x_k ~ PE posterior of
  event i, divide out the PE prior, evaluate `model.log_prob(x_k)`:
  I_i ≈ logsumexp[ log p(x_k|θ) + ln_offsets_k ]. Requires a pointwise
  density p(x|θ) → grid/implicit solver (or KDE on MC).
* **Proposed path (sample mode):** draw x_j ~ p(·|θ) (the MC does this
  natively, with weights w_j), evaluate the event's likelihood:
  I_i ≈ Σ_j w_j · L_i(x_j) / Σ_j w_j. Requires a per-event likelihood
  EVALUATOR L_i(x) — for us, RIFT-style marginal-likelihood
  interpolants/fits per event ("gwkokab can/could run in a
  likelihood-based mode for single events") — and NO model density at all.
* Selection/Poisson mean: μ(θ) = Σ_j w_j · pdet(x_j) — also sample-based,
  also natural for MC.

AD: the engine's JAX MC (J1–J4 + B.2 soft gate) is already reparameterized
— gradients flow through the sampler (∂/∂μ_natal etc. validated), so the
sample-mode likelihood is differentiable end-to-end with the SAME variance
caveats already documented in `feedback_single_key_fd_not_ad_reference`.

## What exists in gwkokab today (checked 2026-06-04, mounted copy)

`src/gwkokab/inference/`:
* `poissonlikelihood_utils.py::analytical_poisson_likelihood_fn` (l.120) —
  vmaps `model_instance.log_prob(samples_stack)` over event PE samples +
  `ln_offsets`; **density mode**.
* `poissonlikelihood_utils.py::discrete_poisson_likelihood_fn` (l.51) —
  same `model_instance.log_prob(safe_data)` core with masks/variance;
  **density mode**.
* numpyro/flowMC wrappers of both; `factory.py`.
* **NO mode consumes model-space samples.** Nothing accepts per-event
  likelihood evaluators. Confirmed by grep: every per-event term goes
  through `model_instance.log_prob`.

## The development task (fork deliverable)

1. `sampled_poisson_likelihood_fn(event_likelihood_fns, model_samples,
   model_log_weights, pdet_fn, T_obs, ...)` in `poissonlikelihood_utils`,
   mirroring the analytical fn's conventions (masks, variance estimate,
   logsumexp hygiene). Per-event evaluators duck-typed callables
   `L_i: (n_samples, n_dim) -> (n_samples,)` (log-likelihood values).
2. numpyro + flowMC wrappers (`numpyro_sampled_poisson_likelihood`, ...)
   matching the existing factory/`variables_index` plumbing, except
   `dist_fn` is replaced by `sampler_fn(**params) -> (samples, log_w)`.
3. A reference event-likelihood adapter: RIFT marginal-likelihood
   interpolant (per-event fit) → callable; plus a trivial Gaussian test
   likelihood for CI.
4. **Acceptance gates:** (i) on a population where both modes are exact
   (analytic Gaussian population + Gaussian event likelihoods), sampled
   mode reproduces the analytical-Poisson posterior within MC error;
   (ii) jax.grad through `sampler_fn` finite + matches many-key FD-mean
   (NOT single-key FD); (iii) variance/ESS diagnostic per event exposed
   (low-ESS warning when the model puts few samples under an event).
5. Thread-back (engine side, AFTER the fork lands): bridge increment
   exposing `particle_mc_jax`/`particle_mc_jax_soft` actives + weights as
   `sampler_fn`; per-event RIFT likelihood loading; promote to a Results-C
   capable path.

## Notes / risks for the implementing agent

* ESS collapse: events in the tail of p(x|θ) get few model samples —
  the variance term (gate iii) is not optional. Stratified/weighted natal
  proposals (M3a/M3b machinery) can target underpopulated regions.
* Keep the public API additive — do not break the two density modes; the
  engine uses `numpyro_analytical_poisson_likelihood` in production today.
* gwkokab is PUBLISHED (PRD 113, 103003); coordinate the fork → upstream
  PR with the gwkokab authors (shared author list with the paper).

## Cross-refs

Engine: `demo/PLAN_jax_mc.md` (J-series), `demo/HANDOFF_PORTABLE.md` §4.2
(bridge increments), `src/.../inference/gwkokab_bridge.py`. Memory:
`project_clustered_chi_and_fisher_bridge_done` (ringing breadcrumb),
`project_mc_jax_total_rate_bias` (open MC metric thread — relevant to μ(θ)).
