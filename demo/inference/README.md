# demo/inference — full-pipeline inference demos (sampled-likelihood path)

End-to-end inference exercises that couple the engine's particle-MC
sampler to **gwkokab's sample-mode (likelihood-mode) Poisson population
likelihood** — the path that lets MC be the *recoverer* with no grid and
no KDE. Requires the gwkokab fork branch
**`feature/sampled-poisson-likelihood`** (commit `41b1c7e`+) on
`PYTHONPATH` (see `demo/PLAN_gwkokab_sampled_likelihood.md` for the task
spec and `examples/sampled_likelihood_method.md` in the fork for the
method note).

> **PRODUCTION WIRING (PLAN item 5, 2026-06-04):** the toy's
> `make_sampler_fn` is promoted to
> `src/hierarchical_ad_pop_engine/inference/sampled_wiring.py` —
> `make_coagulation_sampler_fn` (constant/free split mirroring
> `make_coagulation_dist_fn`), `load_rift_event_log_likelihoods` /
> `gaussian_event_log_likelihoods_from_posteriors` (per-event likelihood
> loading), and one-call `build_sampled_likelihood` (Stage-1 priors +
> sorted-order `variables_index` + the fork's numpyro wrapper). S1–S8
> gates in `tests/test_gwkokab_sampled_bridge.py`. NUTS at production
> scale stays a workstation job — run `run_sampled_wiring_nuts.py` below.

* `run_sampled_wiring_nuts.py` — **THE workstation gate for the
  production wiring**: 6-parameter Stage-1 NUTS recovery through
  `build_sampled_likelihood`, with PASS/FAIL gates G1 (ESS), G2 (r_hat),
  G3 (|z|<3 recovery), G4 (informativeness) and a marginals figure.

      python -m pytest tests/test_gwkokab_sampled_bridge.py -q   # S1-S8 first
      python demo/inference/run_sampled_wiring_nuts.py            # ~minutes
      python demo/inference/run_sampled_wiring_nuts.py --smoke    # 1-min wiring check

  Sandbox status 2026-06-04: S1–S8 8/8 green; `--smoke` verified through
  G1 (catalog + logL + ESS table) before the 45 s cap cut NUTS warmup.

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

* `production_plpeak_sampled_recovery.py` — the production-scale test
  (ROS spec 2026-06-04): GWTC-3-like **smoothed PL + 30 M⊙ pulse**
  injection (small spin, mild cascade), **Mc^{15/6} selection** (swap
  `pdet_fn` for gwkokab's neural/injection VT when trained-network /
  injection files are available), genie measurements in
  **(Mc, η, χ_eff) only** — the evaluator maps model samples to the
  measured coordinates *inside* L_i, so sample mode needs **no
  coordinate Jacobian** (sidesteps PLAN_measurement_model "THE CATCH").
  Target ~100 detected events; 6 free params (α, λ_peak, μ_peak,
  σ_peak, spin scale, log rate).

  The PL+Peak natal enters WITHOUT engine surgery: fixed-mixture
  proposal + θ-smooth `cfg.weight_array` IS pattern. **Proposal quality
  is load-bearing**: a lazy log-uniform proposal collapsed every event's
  ESS to ~2; the matched mixture (PL(α₀)+peak masses, truncnorm+uniform
  spins) restores median ESS ~15 at 4k seeds (scales ~linearly with
  `--n-seed`; high-Mc tail events are the stragglers). Known physics,
  not a bug: α is weakly constrained from ABOVE under chirp-mass
  selection (profile plateaus above truth — selection kills the low-Mc
  events that measure the slope); μ_peak is sharply identified.
  Promoting `natal_family="pl_peak"` into `sampled_wiring` is the
  follow-up once this pattern has a green workstation NUTS run.

      # sandbox smoke (~30 s each)
      python demo/inference/production_plpeak_sampled_recovery.py --mode check
      python demo/inference/production_plpeak_sampled_recovery.py --mode scan
      # workstation production
      python demo/inference/production_plpeak_sampled_recovery.py --mode nuts \
          --n-seed 50000 --n-obs 100 --num-warmup 500 --num-samples 1000

* `compare_recovery_two_methods.py` — the paper's Fig-5 upgrade: TWO
  equivalent inference calculations on ONE single-pulse catalog,
  posteriors overlaid (MC sample-mode in blue; implicit density-mode as
  black shading + contours; truth red). The implicit arm runs at the
  H2-calibrated ε* (geometric bisection to `f_natal_depleted == F_DEP`;
  sandbox-verified: ε* = 2.139 → f_dep = 0.1000 exactly). Arms are
  npz-cached and assemble incrementally via `--arm plot`.

  Sandbox status 2026-06-04: **MC arm green end-to-end** (reduced knobs);
  **implicit arm wiring verified through ε-calibration + model build**,
  but its NUTS compile (IFT/GMRES backward through the Picard
  `while_loop`, vmapped over PE samples) exceeds the 45 s command cap —
  run it on the workstation:

      python demo/inference/compare_recovery_two_methods.py --arm both \
          --n-seed 20000 --n-obs 60 --n-pe 512 \
          --num-warmup 500 --num-samples 1000

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
