# demo/inference — full-pipeline inference demos (sampled-likelihood path)

End-to-end inference exercises that couple the engine's particle-MC
sampler to **gwkokab's sample-mode (likelihood-mode) Poisson population
likelihood** — the path that lets MC be the *recoverer* with no grid and
no KDE. Requires the gwkokab fork branch
**`feature/sampled-poisson-likelihood`** (commit `41b1c7e`+) on
`PYTHONPATH` (see `demo/PLAN_gwkokab_sampled_likelihood.md` for the task
spec and `examples/sampled_likelihood_method.md` in the fork for the
method note).

## Why this exists (strategy)

The implicit Picard solver was always the *backstop* recoverer — adopted
because we had no continuous per-event likelihoods to integrate against
MC samples. With the sampled-likelihood mode and genie-style per-event
likelihood models, we can attempt **full-MC recovery**: MC draws are
exact at any cascade depth, immune to the grid pathologies now
documented on the spin axis (n_chi=64 ringing against the empty
natal–1g gap; deep-cascade node-budget blowup).

**The known hard limit** (ROS 2026-06-04): the model's samples must land
*on every event* with enough statistics. Broad BBH posteriors are
forgiving; narrow (e.g. low-mass) posteriors will starve — watch the
per-event ESS diagnostic, which every script here prints. The `--narrow`
flag on the toy demonstrates the collapse deliberately.

## Scripts

* `toy_single_pulse_sampled_likelihood.py` — the first full-path toy:
  single-pulse truth → MC merger traces (`particle_mc_jax_soft`, AD-safe
  weights) → genie-style Gaussian event likelihoods at scattered observed
  centers → `sampled_poisson_likelihood_fn` sanity check + per-event ESS
  → 1-D profile scan over `mu_natal` → full
  `numpyro_sampled_poisson_likelihood` NUTS recovery.

  Sandbox smoke (seconds):

      python demo/inference/toy_single_pulse_sampled_likelihood.py --mode check
      python demo/inference/toy_single_pulse_sampled_likelihood.py --mode scan

  Workstation (minutes):

      python demo/inference/toy_single_pulse_sampled_likelihood.py --mode nuts \
          --n-seed 20000 --n-obs 60 --num-warmup 300 --num-samples 500

  ESS-collapse demonstrator (the low-mass worry):

      python demo/inference/toy_single_pulse_sampled_likelihood.py --mode check --narrow

## Conventions

* Model samples are merger events `(m1, m2, chi1, chi2)` (heavier-first),
  i.e. columns 0–3 of the engine MC's 8-column `mergers` schema.
* Weight normalisation: `log_w = log(w_mergers·step_active) −
  logsumexp(...) + log_rate`, so `Σ w_j = e^{log_rate}` = the total
  merger rate per unit observing time. This keeps `log_rate` cleanly
  identifiable through the Poisson mean and deliberately sidesteps the
  open MC count-normalisation thread
  (`project_mc_jax_total_rate_bias`) — revisit when that closes.
* The sampler key is FIXED inside `sampler_fn` (deterministic objective,
  same convention as the analytical bridge); the catalog is drawn with an
  independent key.
