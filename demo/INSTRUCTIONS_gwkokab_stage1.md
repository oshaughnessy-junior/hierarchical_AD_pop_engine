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

On cardassia, the immediately-doable escalation:

1. **Two-pulse natal IMF.**  Needs a `build_two_pulse_coagulation` (or a
   `natal_family="two_gaussian"` arm): natal density
   `(1−λ)·N(μ₁,σ₁) + λ·N(μ₂,σ₂)`, +1 weight + 2 location/scale params, and
   `kernel_a` floated → 10 free params.  Extend `stage1_wiring`'s
   `STAGE1_PARAM_NAMES`/priors and the `dist_fn` keyword set; everything
   downstream (likelihood, selection, PPC) is param-count-agnostic.
2. **Full-resolution NUTS** (n_seed≥2000, n_obs≥100) with the `--gate bernoulli`
   robustness; report r̂ and the corner.  Watch the μ₁↔μ₂ label switch — order
   the centres (`μ₁<μ₂`) or parameterise `μ₂=μ₁+Δμ, Δμ>0`.

## 6. Methods-paper experiments (and their prerequisites)

The three scenarios you flagged, in dependency order:

* **(easiest) Two-component, small initial spins — inject at 10 & 30 M⊙.**
  The cleanest two-pulse demo: two near-zero-spin natal pulses at 10 and 30 M⊙.
  Only needs the two-pulse IMF (§5.1); spins stay the existing narrow natal
  truncnorm.  Good first two-pulse recovery + label-switch test.
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

So: **two-pulse small-spin** is reachable with just the two-pulse IMF; both
**Fisher-κ** experiments are gated on **J2b** (the next engine milestone after
the two-pulse fit).

---

## 7. Memory / handoff pointers

`HANDOFF_PORTABLE.md` §4.2 carries the full state + the RIFT-stub recipe.
Assistant memories (per-machine, do not travel): `gwkokab-stage1-wiring-done`,
`gwkokab-stage2-selection-done`, `gwkokab-stage1-identifiability-done`,
`gwkokab-stage1-posterior-done`, `gwkokab-stage1-ppc-done`.  Commits this
stream: `b213325`→`d6b43bb` on `observables` (Cowork-local — push from a
networked machine).
