# Demo 3 — end-to-end gwkokab fit (PLAN)

Status: **planned**, not yet implemented. This document is the step-by-step
roadmap for the next session — pick up at any step.

## Goal

Show that the coagulation engine plugs into `gwkokab.inference` the way a
stock `PowerlawPeak` does, and that AD makes the fit cheap. Concretely:

> Inject a known parameter set θ*, generate a mock catalog from
> `coagulation_model(θ*)`, hand it to `gwkokab.inference.numpyro_poisson_likelihood`,
> and recover θ* with **SVI + NUTS**. Produce a corner plot of the posterior
> overlaid on θ*, and a posterior-predictive plot of the response function
> R(m, χ) with credible bands.

The fit must run in **minutes** on a laptop (CPU JAX) for a 100-event mock
catalog. If it doesn't, that's a finding, not a failure.

## Files this demo will produce

```
demo/demo_gwkokab_fit.py                 # entry point
demo/figures/gwkokab_fit_corner.png
demo/figures/gwkokab_fit_ppc_mass.png
demo/figures/gwkokab_fit_ppc_spin.png
demo/figures/gwkokab_fit_traces.png      # optional: SVI loss curve, NUTS traces
demo/artifacts/mock_catalog.npz          # injected θ*, N events, sampled (m1, m2, χ1, χ2, z)
demo/artifacts/posterior.npz             # samples, log-prob, diagnostics
```

## Steps

### Step 1 — fix the injected parameter set θ*

Use a single dense-environment AGN-disk-like point so the response function
is rich enough that all the per-parameter directions are identifiable. Start
with the `build_default_coagulation` defaults (`alpha=3, mu_peak=33,
sigma_peak=3, kernel_b=2, v_esc=300 km/s, ...`) and snapshot the dict to a
JSON / npz at `demo/artifacts/theta_star.json`.

Acceptance: a self-contained dict in the script, plus a printed table of
ground-truth values.

### Step 2 — sample a mock catalog from `coagulation_model(θ*)`

The mixture returned by `coagulation_model` is a
`gwkokab.models.utils.ScaledMixture` — a numpyro `Distribution`. Two ways
to sample:

a) **Direct** — `mix.sample(key, sample_shape=(N_obs,))`. Cleanest.
b) **Grid inverse-CDF** — if `sample` is not implemented (which is plausible
   for the coagulation mixture; check first), evaluate `log_prob` on a fine
   `(m1, m2, χ1, χ2, z)` grid, build a multivariate CDF column by column,
   and draw via inverse-CDF.

Use N_obs = 100 to start; bump to 300 once the pipeline works.

Acceptance: an `(N_obs, 5)` array of `[m1, m2, χ1, χ2, z]` samples whose
1-D marginals visually match the analytic `log_prob` slices. Save to
`mock_catalog.npz`.

### Step 3 — selection function

Two options, in order of effort:

a) **No selection** (idealised). The likelihood reduces to a pure unweighted
   Poisson over events. Good for the smoke test.
b) **A simple SNR cutoff**. Compute the IMRPhenomD optimal SNR for each
   event at a fiducial luminosity-distance / detector PSD via gwkokab's
   selection utilities (look in `gwkokab.selection` / `gwkokab.utils`), keep
   events with SNR > 8.

Start with (a). Add (b) only after the (a) fit recovers θ*.

Acceptance: a `selected_events` array + a `pdet(θ)` callable suitable for
the gwkokab likelihood. For (a), `pdet ≡ 1` and there's nothing to do.

### Step 4 — build the numpyro inference model

Wrap `coagulation_model` in a numpyro model with priors on the free
parameters. Start by freezing the slow / less-identifiable ones (z_max,
kappa, gamma, z_peak, m_min, m_max) at θ* and sampling only the
**5 directions we most care about**:

- `alpha` (natal IMF slope)
- `mu_peak`, `sigma_peak`, `lambda_peak` (peak)
- `kernel_b` (pairing-kernel exponent)
- `natal_spin_loc`, `natal_spin_scale` (natal spin)
- `log_mu_v_esc` (environment escape velocity)
- `log_rate` (overall rate)

Total ~9 free parameters. Reasonable priors:

| Parameter             | Prior                       |
|-----------------------|-----------------------------|
| `alpha`               | Uniform(1, 6)               |
| `mu_peak`             | Uniform(20, 50)             |
| `sigma_peak`          | LogUniform(1, 10)           |
| `lambda_peak`         | Uniform(0, 0.5)             |
| `kernel_b`            | Uniform(-2, 6)              |
| `natal_spin_loc`      | Uniform(0, 0.5)             |
| `natal_spin_scale`    | LogUniform(0.05, 0.5)       |
| `log_mu_v_esc`        | Uniform(log 50, log 1500)   |
| `log_rate`            | Uniform(-2, 6)              |

Likelihood: `gwkokab.inference.numpyro_poisson_likelihood(mix, events, pdet, ...)`.

Acceptance: `numpyro.infer.util.log_density` returns a finite scalar at θ*
and gradients flow.

### Step 5 — SVI first (fast smoke test)

```python
from numpyro.infer import SVI, autoguide, Trace_ELBO
guide = autoguide.AutoNormal(model)
svi = SVI(model, guide, optax.adam(1e-2), Trace_ELBO())
state = svi.init(key, events)
state, losses = jax.lax.scan(lambda s, _: svi.update(s, events), state, jnp.arange(2000))
```

Plot the ELBO loss; it should drop and plateau within ~1-2 k steps for a
problem this small. Sample 1000 posterior draws from the AutoNormal and
make a quick corner plot.

Acceptance: SVI converges, posterior modes within 1-2 σ of θ*. Save samples
to `posterior_svi.npz`.

### Step 6 — NUTS (gold standard)

```python
from numpyro.infer import MCMC, NUTS
kernel = NUTS(model, target_accept_prob=0.85)
mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=2,
            chain_method="parallel")
mcmc.run(key, events)
samples = mcmc.get_samples()
```

Diagnostics: r̂ < 1.01 for all params, ESS > 200.

Acceptance: ground-truth θ* sits inside the 90 % HDI for every recovered
parameter. Save to `posterior_nuts.npz`.

### Step 7 — corner plot + truth lines

Use `corner.corner(samples, truths=theta_star)`. One figure,
`figures/gwkokab_fit_corner.png`.

### Step 8 — posterior-predictive R(m, χ)

For each of K=200 posterior draws θ_k:

1. Build `CoagulationResponse` at θ_k.
2. Solve, marginalise to `dN/dm`, `dN/dχ`.
3. Stack across k → median + 90 % credible band.

Plot the band against the ground-truth R(m, χ) curve. Two figures:
`figures/gwkokab_fit_ppc_mass.png` and `figures/gwkokab_fit_ppc_spin.png`.

Acceptance: the band straddles the truth curve almost everywhere. Pockets
of disagreement at the natal-IMF tails or near m_min are expected and
non-pathological — flag them in the figure caption.

### Step 9 — SBC-lite coverage check (optional, stretch)

Sample 20 θ* draws from the prior, run a fast SVI fit per draw, record
the fraction of 90 % credible intervals that contain the truth. Hits
should be 17-20 out of 20. Anything below 14 indicates miscalibration —
inspect the prior / likelihood.

### Step 10 — writeup

A short `demo/README_gwkokab_fit.md` with the four figures and a paragraph
of interpretation: "we inject θ*, recover it, AD made this take N minutes
on a laptop." Suitable to include in the AFI Year-1 deliverable.

## Open questions to resolve in the next session

1. **Does `coagulation_model(...)` -> ScaledMixture support `sample`?** If
   not, Step 2 needs the grid inverse-CDF fallback. Check
   `gwkokab/src/gwkokab/models/utils.py::ScaledMixture` first.

2. **Selection-function plumbing.** Find the cleanest API in gwkokab —
   is it a `pdet(θ)` callable, an injection set, or both? Skim
   `gwkokab/src/gwkokab/inference/` and `gwkokab/src/gwkokab/selection/`.

3. **Recompile cost.** `_solve` traces the K-loop with `k_max=4`; if NUTS
   inflates compile time, vmap the leapfrog or drop `k_max` to 3 for the
   demo (mention as a v2 cleanup).

4. **Random seeds.** Pick `key = jax.random.PRNGKey(20260517)` everywhere
   so the figures are reproducible.

## Anti-goals (do not do in this demo)

- Don't try to recover all 17 parameters at once. Start frozen.
- Don't add the formation-rate convolution (env distribution × redshift) —
  that's Demo 4. Use `coagulation_model_delta_formation` for this fit.
- Don't fit real LIGO data. Mock first; real data is a separate deliverable.
- Don't introduce new physics. The natal spin distribution and remnant
  fits already in the package are what we fit with.

## Effort estimate

- Step 1-2 (theta, mock catalog): 1 session.
- Step 3-4 (selection, numpyro model): 1 session.
- Step 5-6 (SVI + NUTS): 1 session.
- Step 7-8 (figures): 0.5 session.
- Step 9-10 (SBC, writeup): 0.5-1 session.

Total: ~4 sessions. The hard part is Step 4 — wiring priors into the
gwkokab likelihood — and Step 6 if NUTS compile time is bad.
