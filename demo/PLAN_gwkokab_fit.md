# Demo 3 — end-to-end coagulation fit (iterative plan)

We grow the demo in **three** stages of increasing realism. Each stage is a
self-contained, runnable script that ends in a corner + posterior-predictive
plot. The structure is the same at every stage; only the population model
and the selection function get richer.

The point of the staging is to keep the bite size small and let us *see*
each new ingredient pay off (or not) before piling on the next.

---

## Stage 1 — single natal Gaussian pulse, flat selection (this session)

**Population.** A single Gaussian natal IMF (μ, σ) on `[m_min, m_max]`,
narrow natal-spin truncated normal (loc, scale), single environment with
fixed `v_esc`, full coagulation cascade with `k_max = 4`.

**Free parameters (6 total).**

| Name                 | Meaning                                | Prior                          |
|----------------------|----------------------------------------|--------------------------------|
| `mu_natal`           | natal-IMF mean                         | Uniform(5, 50) M⊙              |
| `sigma_natal`        | natal-IMF width                        | LogUniform(0.5, 10) M⊙         |
| `natal_spin_loc`     | natal-spin mean                        | Uniform(0, 0.3)                |
| `natal_spin_scale`   | natal-spin width                       | LogUniform(0.02, 0.3)          |
| `kernel_b`           | pairing mass-ratio exponent            | Uniform(-2, 6)                 |
| `log_rate`           | overall log rate (T_obs absorbed)      | Uniform(-2, 6)                 |

**Fixed at known.** `m_min`, `m_max`, `v_esc`, `kernel_a` (=0), `k_max`, the
natal-mass family (Gaussian, not PL+peak), and no redshift dependence.

**Selection.** Flat. Detection efficiency is taken constant and folded into
`log_rate` — we are fitting the **detected** rate. No `pdet(θ)` callable.

**Likelihood.** Inhomogeneous-Poisson on the binary parameter space, rolled
by hand against `CoagulationResponse.log_rate_density` plus a grid quadrature
for the expected number. We deliberately do **not** route through
`gwkokab.inference.numpyro_analytical_poisson_likelihood` yet — that path
needs the model wrapped as a numpyro `Distribution` / `ScaledMixture`, which
is a Stage 2+ concern. Stage 1 is the smallest-possible end-to-end fit.

**Mock catalog.** Factor `p(m1,m2,χ1,χ2)` via
`p(m1,m2) ∝ K · N_m(m1) · N_m(m2)` and `p(χ|m) ∝ N(m,χ)/N_m(m)`, then draw
via grid inverse-CDF. `N_obs ≈ 100` events.

**Inference.** SVI (`AutoNormal` + Adam) for the smoke test, NUTS for the
gold-standard posterior. Both compile once per chain.

**Deliverables.**

```
demo/demo_gwkokab_fit_stage1.py
demo/figures/stage1_truth_dN_dm.png
demo/figures/stage1_truth_dN_dchi.png
demo/figures/stage1_mock_catalog.png
demo/figures/stage1_svi_loss.png
demo/figures/stage1_corner.png
demo/figures/stage1_ppc_mass.png
demo/figures/stage1_ppc_spin.png
demo/artifacts/stage1_theta_star.json
demo/artifacts/stage1_mock_catalog.npz
demo/artifacts/stage1_posterior_svi.npz
demo/artifacts/stage1_posterior_nuts.npz
```

**Acceptance.** All six truth values inside the 90 % credible interval; SVI
ELBO plateau visible; NUTS r̂ < 1.01 for every param. Posterior-predictive
band on `dN/dm` and `dN/dχ` straddles the truth almost everywhere.

---

## Stage 2 — two pulses + lazy SNR(m) selection (next pass)

**Population.** Two-pulse natal IMF: a weighted mixture
`(1-λ) N(μ₁,σ₁) + λ N(μ₂,σ₂)` on `[m_min, m_max]`. Same natal spin family,
same single environment with fixed `v_esc`.

**Free parameters (10 total).** Stage 1's six **plus** `mu_natal_2`,
`sigma_natal_2`, `lambda_natal` (peak weight in [0,1]); and we now also let
`kernel_a` float (Uniform(-2,3)) since two pulses make total-mass scaling
weakly identifiable.

**Selection.** Lazy SNR(m) interpolation. Two equally fine options:

a) **Cheap analytic surrogate.** `SNR ≈ (m_chirp / m_ref)^{5/6}` evaluated
   at a fiducial luminosity distance, thresholded at SNR > 8 →
   `pdet(m1,m2)` smoothed by `expit`.
b) **gwkokab utility.** Look for `gwkokab.utils.popfactor` or similar; if
   present, use it for IMRPhenomD optimal SNR against the H1-O3 PSD.

Start with (a). Bump to (b) only if (a) biases the recovery.

**Likelihood.** Now we want the gwkokab analytical likelihood path so the
selection integral is uniform across the team's tooling. Wrap
`CoagulationResponse` as a numpyro `Distribution` (a thin shim exposing
`log_prob` from `log_rate_density` and a Monte-Carlo / grid
`poisson_mean_estimator`). Resolve the open question about
`ScaledMixture.sample` while we are at it.

**Acceptance.** Two-pulse recovery: truth inside the 90 % credible interval
for all 10 params, including the (potentially degenerate) `μ₁,σ₁ ↔ μ₂,σ₂`
labelling resolved by ordering the centres (`μ₁ < μ₂`).

---

## Stage 3 — two pulses + non-hierarchical background (next-next pass)

**Population.** Stage 2 model **plus** a non-hierarchical background: a
fraction `f_bg ∈ [0, 1]` of events come from a wide, smooth PL+peak that
does **not** undergo coagulation (`k_max = 1` for that component). The
detected catalog is then a two-component additive mixture:

`λ_total = (1 - f_bg) λ_coag + f_bg λ_bg`

with `λ_bg` the existing Talbot-Thrane PL+peak from `log_natal_mass`. The
point of Stage 3 is to test that the coagulation signal survives
contamination by a non-hierarchical population.

**Free parameters (~14).** Stage 2's ten plus `f_bg`, `alpha_bg`, `mu_bg`,
`sigma_bg`.

**Selection.** Same lazy SNR(m) as Stage 2.

**Acceptance.** Hierarchical signal recovered (`mu_natal_1`, `kernel_b`,
`log_rate` all within the 90 % credible interval) even at `f_bg` up to
≈ 0.5. Beyond that, document the breakdown point as a finding.

---

## Open questions to resolve in each stage

**Stage 1.**

- How long does NUTS warmup take with `k_max = 4`? If the JIT compile of
  `_solve` blows up the warmup wall clock, drop `k_max` to 3 for the demo
  and document the tradeoff.

**Stage 2.**

- Does `ScaledMixture.sample` exist for the coagulation mixture, or do we
  always inverse-CDF? Check
  `gwkokab/src/gwkokab/models/utils.py::ScaledMixture` first.
- Cleanest API for `pdet(θ)`: callable, injection set, or both? Skim
  `gwkokab/src/gwkokab/inference/` and `gwkokab/src/gwkokab/utils/`.

**Stage 3.**

- Does label switching between `μ₁ ↔ μ₂` (Stage 2 problem) get worse once
  we add the smooth `λ_bg`? If so, parameterise as `μ₂ = μ₁ + Δμ` with
  `Δμ > 0` prior.

---

## Anti-goals (do not do in any stage of this demo)

- Don't recover all 17 of the full model's parameters at once.
- Don't add the formation-rate × redshift convolution. That belongs in a
  separate demo. Stages 1-3 fit only `coagulation_model_delta_formation`-
  style detected-rate models.
- Don't fit real LIGO data. Mock first; real data is a separate deliverable.
- Don't introduce new physics. Natal spin distribution and remnant fits
  already in the package are what we fit with.

---

## Reproducibility

- Random seed: `PRNGKey(20260517)`.
- All figures regeneratable by running the stage script start-to-finish.
- Stage scripts are independent — no shared imports between stages.

---

## Effort estimate

| Stage | Mock catalog | Numpyro model | SVI+NUTS | Figures | Total       |
|-------|--------------|---------------|----------|---------|-------------|
| 1     | 0.3 session  | 0.3 session   | 0.3      | 0.1     | ~1 session  |
| 2     | 0.5          | 0.7           | 0.3      | 0.1     | ~1.5        |
| 3     | 0.3          | 0.5           | 0.4      | 0.1     | ~1.3        |

The hard parts are Stage 2 likelihood (wrapping `CoagulationResponse` as a
numpyro Distribution that plugs into the gwkokab analytical likelihood) and
Stage 3 label switching.
