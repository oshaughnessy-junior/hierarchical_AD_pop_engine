# PLAN — gwkokab sample-based (likelihood-mode) inference for MC populations

**Date:** 2026-06-04 (ROS direction, post chi-resolution rerun). **Status:**
**FORK DELIVERABLE STARTED 2026-06-04** — items 1–4 (core fn, numpyro+flowMC
wrappers, factory wiring, Gaussian + RIFT-interpolant adapters, TP0–TP4
acceptance gates) landed on gwkokab branch
`feature/sampled-poisson-likelihood`, commit `41b1c7e` (local-only in the
Cowork sandbox — **push to the `junior` remote, not `origin/main`, from a
networked clone**). 7/7 test-problem gates green in-VM under the RIFT import
stub. See `examples/sampled_likelihood_method.md` and
`examples/sampled_likelihood_demo.py` in the fork.

**ITEM 5 (thread-back) CLOSED 2026-06-04 (same day, later session):**
`src/hierarchical_ad_pop_engine/inference/sampled_wiring.py` —
`make_coagulation_sampler_fn` (particle_mc_jax/_soft actives+weights as
`sampler_fn`, Σw = e^{log_rate}, NaN-slot hygiene + floored weights),
`load_rift_event_log_likelihoods` (RIFT interpolant loading via the fork's
adapter; builder-injected so the RIFT stack stays workstation-side) +
`gaussian_event_log_likelihoods_from_posteriors` (genie-style fallback),
`build_sampled_likelihood` (one-call wiring, Stage-1 priors/sorted-index
contract) — the Results-C capable path. Gates S1–S8 in
`tests/test_gwkokab_sampled_bridge.py`, 8/8 green in-VM. Realism evidence:
`internal_tests/gw_banana_sampled_vs_density.py` (uncommitted, top-level) —
banana-error GW catalogues, sampled vs density NUTS posteriors agree to
dz≤0.09; density mode fails outright on events whose PE cloud exits the
support while sampled mode degrades gracefully (low-ESS flag). Production
NUTS runs remain workstation jobs (sandbox 45 s cap).

> **UPDATE 2026-06-04 (late): ENGINE-SIDE TOY GREEN — item 5 de-risked.**
> `demo/inference/toy_single_pulse_sampled_likelihood.py` exercises the
> FULL path end-to-end in-sandbox: single-pulse truth →
> `particle_mc_jax_soft` merger traces (`mergers[:, :4]` =
> (m1,m2,χ1,χ2) + `w_mergers·step_active` weights; dead NaN slots
> sanitized via where-select; weight normalisation `Σw = e^{log_rate}`)
> → genie-style Gaussian event likelihoods
> (`gaussian_event_log_likelihoods`) → `sampled_poisson_likelihood_fn`
> check + μ profile scan → `numpyro_sampled_poisson_likelihood` NUTS:
> (μ, σ, logR) all recovered within ~1.5σ at toy scale (1.5k seeds,
> 12 events). Per-event ESS healthy broad (min 58 @ 445 live), COLLAPSES
> under `--narrow` (all events ESS 1–6) — the low-mass worry is real and
> now demonstrable. Toy gotcha worth keeping: with a fixed-size catalog
> the only self-consistent rate truth is log(n_obs/T_obs) — a wrong rate
> truth shows up as a cleanly-converged-but-biased log_rate.
> REMAINING for item 5 proper: bridge-level `sampler_fn` export
> (production config, not the toy's local MCConfig), per-event RIFT
> interpolant loading, selection (pdet_fn) wiring.

> **UPDATE 2026-06-04 (later): PRODUCTION-SCALE TEST SHIPPED (05f1c2d)**
> on top of the wiring (71be29b) + its workstation NUTS gate (2b1c88e):
> `demo/inference/production_plpeak_sampled_recovery.py` — GWTC-3-like
> smoothed-PL + 30 M⊙ pulse injection, Mc^{15/6} selection, genie
> (Mc, η, χ_eff)-only measurements (NO Jacobian needed in sample mode —
> "THE CATCH" sidestepped), 6 free params. PL+Peak natal via
> fixed-mixture proposal + θ-smooth `weight_array` (no engine surgery;
> **proposal quality is load-bearing** — log-uniform collapsed all ESS
> to ~2, matched mixture → median 15 @ 4k seeds). α weakly constrained
> from above under Mc-selection (physics, not artifact); μ_peak sharp.
> Workstation: `--mode nuts --n-seed 50000 --n-obs 100`. Follow-ups:
> promote `natal_family="pl_peak"` into `sampled_wiring`; swap pdet_fn
> to neural/injection VT when files exist; real-data RIFT interpolants.

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
