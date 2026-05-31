# Stage-1 gwkokab fit — runbook + look-ahead (Richard)

A practical guide to the single-pulse coagulation→gwkokab fit demos built
2026-05-30, and the path to the two-pulse / Fisher-κ methods-paper runs.

Everything routes the coagulation engine (`solver="mc_jax"`, AD-traceable)
through the **real** gwkokab analytical-Poisson likelihood via the new
`inference/stage1_wiring.py` + `inference/selection.py`.  The demos all ran in
the Cowork VM under a RIFT stub; on cardassia they run against the real stack.

---

## 0. Environment

**cardassia / workstation (real stack):**
```bash
cd hierarchical_AD_pop_engine
pip install -e .            # engine + gwkokab + RIFT/lalsuite already present
# or the robust full-stack env:  pixi run tests-bridge
export PYTHONPATH=src:.:demo:../gwkokab/src
```

**Cowork VM / any ephemeral CPU box (no lalsuite):** the inference + engine MC
path imports without lalsuite if you stub the one RIFT symbol gwkokab loads at
import (never called on this path).  Full recipe is in
`HANDOFF_PORTABLE.md` §4.2 ("Running the REAL gwkokab … in the Cowork VM").
Short form:
```bash
SITE=<scratch>/.pylibs; STUB=<scratch>/.stubs
pip install --target=$SITE jax jaxlib equinox numpyro jaxtyping scipy pytest \
    quadax interpax loguru h5py optax matplotlib pandas
mkdir -p $STUB/RIFT; : > $STUB/RIFT/__init__.py
printf 'MsunInSec = 4.925491025543576e-06\n' > $STUB/RIFT/lalsimutils.py
export PYTHONPATH=$SITE:$STUB:<repo>/gwkokab/src:<repo>/src:<repo>:<repo>/demo
export JAX_PLATFORMS=cpu
```

---

## 1. What's built (4 scripts + 1 module)

| file | does |
|---|---|
| `demo/mock_catalog.py` | multi-sample `(N,S,4)` posteriors + `ln_offsets` (measurement blur; `scatter_to_posteriors` is engine-free + unit-tested) |
| `demo/fit_stage1_gwkokab.py` | **identifiability**: 1-D logL profiles through θ* + AD-MAP; `sample_from_model` (self-consistent inverse-CDF draws, optional `vt_fn`) |
| `demo/posterior_stage1_gwkokab.py` | **posterior**: SVI (+ NUTS) → medians, 90% CI, coverage, corner |
| `demo/ppc_stage1_gwkokab.py` | **observables**: dN/dm & dN/dχ PPD bands + event scatter over predicted density |
| `src/.../inference/stage1_wiring.py`, `selection.py` | priors/variables/variables_index + likelihood factory; VT(θ) estimator |

## 2. How to run each

```bash
# (a) identifiability — fast, robust; the "is the origin unique?" check
python3 demo/fit_stage1_gwkokab.py --gen self --n-obs 40 --n-samples 5 \
    --n-seed 300 --k-max 3 --map-steps 120
#   → demo/figures/stage1_identifiability_profiles.png
#   --gen implicit   : cross-solver recovery (implicit oracle) instead of self-consistent
#   --selection vt   : M_chirp^(15/6) detection-weighted μ instead of flat

# (b) posterior — SVI (+ NUTS)
python3 demo/posterior_stage1_gwkokab.py --svi-steps 2500 --n-obs 40 \
    --n-seed 300 --k-max 3 --nuts --nuts-warmup 150 --nuts-samples 400
#   → demo/figures/stage1_posterior_{svi,nuts}.png

# (c) observables / PPC
python3 demo/ppc_stage1_gwkokab.py --detection none --n-draws 80 --n-pred 2000 \
    --svi-steps 2000 --n-obs 40 --n-seed 300 --k-max 3
#   → demo/figures/stage1_ppd_mass_spin_none.png, stage1_ppc_scatter_none.png
#   --detection vt --gate bernoulli   : detection-weighted view (see gotchas)
```

## 3. Results so far (single-pulse, self-consistent, n_obs=40, n_seed=300)

* **Identifiability:** all 6 params identified, unique origin; profile peaks on
  truth (mu_natal 8.00→7.92, log_rate exact, kernel_b 1.0→0.5).  `kernel_b` and
  `natal_spin_loc` are the weakest (broad but peaked).
* **Posterior:** SVI **6/6** truths in the 90% CI; NUTS **5/6** cross-check
  (mu_natal just outside its tighter CI — a ~0.3 mc_jax/KDE peak bias surfaces
  as the CI tightens).  `kernel_b` widest.
* **Observables:** intrinsic (flat) PPD bands straddle the observed mass & spin
  histograms — good fit; the event scatter sits inside the predicted density.

## 4. Gotchas (these will bite on cardassia too)

* **SVI + soft gate can NaN.**  The soft-gate cascade is `nan_to_num`-guarded in
  the forward pass but NOT in the backward; an aggressive/wandering guide hits a
  backward NaN → −ELBO→nan, all-NaN posterior.  Mitigations (all in the
  scripts): `AutoNormal(init_scale=0.1)` + `optax.clip_by_global_norm` + lr 0.01.
  If it still NaNs (e.g. the detection-weighted, higher-mass catalogue did),
  use **`--gate bernoulli`** (B.1) — more SVI-robust — or a different `--seed`.
* **NUTS OOMs the 3.9 GB VM at n_seed=300** → full-resolution NUTS is a
  workstation task (cardassia has the RAM).  The VM ran NUTS only at
  n_obs=30/n_seed=200.
* **Forward-model loop must be jitted ONCE.**  Per-draw eager engine builds
  accumulate device buffers and OOM; `ppc_stage1_gwkokab.forward_one` jits the
  whole per-draw map (build + sample + JAX histogram).  Pattern to reuse.
* **Forward sampling is grid-free particle resampling** (`fit_stage1_gwkokab.
  sample_pairs_from_particles`): it draws binary pairs straight from the
  engine's mc_jax particle pool ∝ `K·w_i·w_j·VT`, so the discrete 2G high-χ
  remnant spins are exact.  Do NOT reintroduce a density grid / KDE sampler for
  observables — it under-resolves the χ≈0.68 2G peak (that was the original
  spin-PPD artifact).  (The identifiability/posterior scripts still use the
  older grid `sample_from_model` only for the self-consistent *event positions*
  — the likelihood itself never samples — but switching them to
  `sample_pairs_from_particles` too is a clean unification when convenient.)
* **`n_seed=300` is the known-stable SVI point** at these settings.  Lower
  n_seed has diverged.

---

## 5. Next: two-pulse + full-resolution NUTS (the fun run)

### 5.0 Two-pulse natal IMF — LANDED 2026-05-30 (`natal_family="two_gaussian"`)

The two-pulse stack is **built and sandbox-validated**; only the
full-stack run (gwkokab analytical-Poisson) is workstation-tier.  What
shipped, bottom-up:

| layer | what |
|---|---|
| `natal/mass.py` | `log_natal_mass_two_gaussian(m, mu_1, sigma_1, mu_2, sigma_2, lambda_2, m_min, m_max)` — normalised mix of two truncnorms (each independently normalised ⇒ mix is too). |
| `red_team/backup_particle_mc.py` | `MCConfig` + `mu_natal_2`/`sigma_natal_2`/`lambda_natal` (all `None` ⇒ single-pulse, bit-for-bit). `sample_natals_jax` grows an **IS-weight two-pulse branch**: masses drawn from a *fixed 50/50* mixture proposal, the true λ-mixture carried through smooth per-particle weights `w = f_mix(m;λ)/(½c₁+½c₂)`. **λ enters only the weight numerator ⇒ its gradient flows** (hard component assignment would zero it — see the chosen design). |
| `response/coagulation.py` | `_natal_log_density` dispatch on `natal_mass_kind=="two_gaussian"`; new `build_two_pulse_coagulation(...)` (twin of `build_gaussian_coagulation`, shares the solver-attach tail via `_finalize_response_solver`). |
| `response/mc_jax.py` | accepts `natal_mass_kind ∈ {gaussian, two_gaussian}`; threads pulse-2 params into `MCConfig`. |
| `inference/gwkokab_bridge.py` | `make_coagulation_dist_fn(..., natal_family="two_gaussian", float_kernel_a=True)` → 9/10-keyword `dist_fn`. |
| `inference/stage1_wiring.py` | `TWO_PULSE_PARAM_NAMES` (10), `two_pulse_prior_factories()`, `build_two_pulse_variables()`. Label switch broken **structurally**: disjoint mass windows `μ₁∈(5,20)`, `μ₂∈(20,50)` (gwkokab's `JointDistribution` takes independent marginals, so no `OrderedTransform`). |
| `demo/fit_two_pulse_gwkokab.py` | identifiability profiles + AD-MAP + `--label-switch-probe` (MAP from a swapped start, confirm truth ordering) + a **natal-IMF reconstruction figure** (`two_pulse_natal_recovery.png`) — the methods-paper headline. |
| `tests/test_two_pulse_natal.py` | 7 sandbox invariants (density norm, IS unbiasedness, λ/μ₁/μ₂ AD-vs-FD, single-pulse regression, **full-cascade AD in λ**, both-pulses-seed-pool). 7/7 green; existing single-pulse suite 24/24 unaffected. |

**Run the two-pulse identifiability/recovery (cardassia or VM+RIFT-stub):**
```bash
python3 demo/fit_two_pulse_gwkokab.py --n-obs 60 --n-samples 5 \
    --n-seed 600 --k-max 3 --map-steps 200 --label-switch-probe
#   → demo/figures/two_pulse_identifiability_profiles.png
#   → demo/figures/two_pulse_natal_recovery.png   (PROMOTE to paper)
```

Sandbox AD payoff measured: `d⟨m⟩/dλ` AD-vs-FD rel 1.6e-8 through the
pool, and through the **full `particle_mc_jax_soft` cascade** rel 1.6e-8 —
so SVI/NUTS recovery of λ is unblocked end-to-end.

**Still workstation-tier:** the SVI/NUTS posterior on all 10 params, and
the full-resolution corner.  Mirror `posterior_stage1_gwkokab.py` with the
10-param `build_two_pulse_variables`; reuse the §4 gotchas verbatim.

### 5.1 Full-resolution NUTS

**Full-resolution NUTS** (n_seed≥2000, n_obs≥100) with the `--gate bernoulli`
robustness; report r̂ and the corner.  The μ₁↔μ₂ label switch is already
broken by the disjoint-window priors above; if a future experiment needs
*overlapping* pulses, switch to a `μ₂=μ₁+Δμ, Δμ>0` reparam in the `dist_fn`.

## 6. Methods-paper experiments (and their prerequisites)

The three scenarios you flagged, in dependency order:

* **(easiest — NOW UNBLOCKED) Two-component, small initial spins — inject at
  10 & 30 M⊙.**  The two-pulse IMF landed (§5.0); `demo/fit_two_pulse_gwkokab.py`
  is the recovery + label-switch demo.  Remaining: run it full-stack on
  cardassia + the 10-param SVI/NUTS posterior.
* **Natal alignment — one-pulse, vary Fisher κ, recover κ.**  Requires **J2b**
  (non-Aligned vector physics in the mc_jax soft gate: per-event orientation
  sampling + vector Barausse-Rezzolla spin / Lousto-Zlochower kick + retention).
  The misalignment machinery EXISTS for the implicit path
  (`spin_orientation.py` Fisher, `remnants/{spin_vector,kick_vector,marginal}.py`)
  but the MC path is **Aligned-only** today (HANDOFF §4.4).  Wire κ as a free
  param once J2b lands; `∂/∂v_esc` recovers for free under misalignment.
* **Exact alignment, fit with Fisher family (model mismatch).**  Also needs
  J2b (a Fisher-family likelihood vs Aligned-truth data).  A clean
  model-misspecification study once the Fisher MC path exists.

So: **two-pulse small-spin** is reachable with just the two-pulse IMF (DONE,
§5.0); both **Fisher-κ** experiments are gated on **J2b** (the next engine
milestone).  **J2b is now scaffolded** — see `demo/PLAN_j2b_vector_mc.md`: the
primitive audit + AD de-risk is done (Fisher.sample is κ-reparametrised; the
`*_at_orientation` vector remnants are AD-clean; sandbox κ→χ_rem AD-vs-FD rel
1.8e-7, v_esc→retain rel 1.4e-6), so the milestone is pure wiring of an
orientation-sampled + soft-retention remnant step into `particle_mc_jax_soft`
(exact insertion site `red_team/backup_particle_mc.py:1353-1356`).

---

## 7. Memory / handoff pointers

`HANDOFF_PORTABLE.md` §4.2 carries the full state + the RIFT-stub recipe.
Assistant memories (per-machine, do not travel): `gwkokab-stage1-wiring-done`,
`gwkokab-stage2-selection-done`, `gwkokab-stage1-identifiability-done`,
`gwkokab-stage1-posterior-done`, `gwkokab-stage1-ppc-done`.  Commits this
stream: `b213325`→`d6b43bb` on `observables` (Cowork-local — push from a
networked machine).
