# PORTABLE HANDOFF — hierarchical_AD_pop_engine (2026-05-29)

**Read this first on a new machine (e.g. cardassia).**  It is
self-contained: it carries the project state, the next big steps, and
the distilled "memory" (hard rules + AD lessons) that otherwise live in
a per-machine assistant memory store and do **not** travel.  Everything
here is in-repo, so `git` carries it; or scp this single file.

> One-line status: **B.2 soft K-gate is DONE and green** (sandbox tier +
> M4 SVI on BOTH gate branches).  The JAX-MC population engine
> (`solver="mc_jax"`) is end-to-end AD-traceable and SVI-recovers Stage-1
> truth.  The next big step is **wiring to gwkokab** (Stage 2:
> `CoagulationResponse` → numpyro `Distribution` → gwkokab analytical
> Poisson likelihood + selection).

---

## 0. What this project is

A JAX, fully-AD coagulation engine for hierarchical compact-binary
formation.  Black holes merge hierarchically (`x1 + x2 → x3` with
radiation loss + remnant spin/kick physics), so the merger-rate density
`dN/dVdtdx` correlates the binary parameters `x` across generations.  The
engine forward-models that cascade and exposes
`log_rate_density(m1,m2,χ1,χ2)` + `total_rate`, differentiable in the
population hyperparameters, to drive SVI/NUTS population inference.  End
goal: wire into **gwkokab** (the team's GW population-inference
framework) as a population model.

Three solver backends, all behind `solver=`:
* `"implicit"` — grid Picard fixed point (production-quality, "beautiful";
  revisit deferred, see §6).
* `"basis"` — Chebyshev/Legendre-Lobatto Galerkin (engine v3).
* `"mc"` / `"mc_jax"` — particle Monte-Carlo; **`mc_jax` is the
  AD-traceable JAX path this work-stream built** (J-series → M4).

Red-team philosophy: numpy `particle_mc` is the parallel
reference implementation; nothing in the JAX path may modify it.

---

## 1. Current state (2026-05-29)

* Git branch: **`observables`**.  Last *committed* HEAD: `4be91a9`
  (round-3 B.1 + an xfail).  **Everything below is UNCOMMITTED** in the
  working tree (Cowork sessions commit locally only; you push from a
  networked machine).
* B.2 soft K-gate implemented, sandbox-validated, and **M4 green on both
  gate branches** (B.1 Bernoulli and B.2 soft).
* JAX/pytest do NOT run in the Cowork VM that authored this; all JAX
  validation here was done on the workstation by the user.  numpy
  sandboxes run anywhere.

### Uncommitted files (the B.2 work)

```
MODIFIED:
  red_team/backup_particle_mc.py                         # +particle_mc_jax_soft (B.2)
  src/hierarchical_ad_pop_engine/response/mc_jax.py      # gate={bernoulli,soft} arm
  src/hierarchical_ad_pop_engine/response/coagulation.py # mc_jax_gate kwarg thread
  tests/test_integration_stage1_recovery_mc_jax.py       # M4 parametrised over both gates
NEW:
  tests/test_mc_particle_mc_jax_soft.py                  # B.2 sandbox tier S1–S8
  demo/forward_modeling_mc_parallel.py                   # numpy vs B.1 vs B.2 (JAX)
  demo/b2_soft_gate_derivation.py                        # numpy Q1/Q2/Q3 (no JAX)
  demo/b2_jax_mirror_check.py                            # numpy mirror of JAX step (no JAX)
  demo/b2_histogram_headtohead.py                        # numpy discrete-vs-soft hist (no JAX)
  demo/b2_parallel_preview.py                            # numpy preview of the parallel demo
  demo/project_jax_mc_b2_soft_gate.md                    # B.2 design memo
  demo/HANDOFF_PORTABLE.md                               # THIS FILE
```

### Suggested commit plan (atomic)

1. `particle_mc_jax_soft` + the `gate` wiring (`backup_particle_mc.py`,
   `mc_jax.py`, `coagulation.py`).
2. B.2 tests + M4 parametrisation
   (`test_mc_particle_mc_jax_soft.py`, `test_integration_stage1_recovery_mc_jax.py`).
3. Demos + sandboxes + memos (`demo/*`).

(FUSE git quirk on the Cowork VM only: `.git/*.lock` can be create-OK /
rm-deny; rename idiom `mv .git/index.lock .git/index.lock.dead` if a
commit fails "Operation not permitted".  Irrelevant on cardassia.)

---

## 2. What B.2 is and why it exists

**B.1 (round 3, committed `596bb53`)** — `particle_mc_jax`: a Bernoulli
K-gate cascade.  Two costs: (1) a `k_ratio ≈ K_max/⟨K⟩` magic factor to
size `n_steps`; (2) a straight-through estimator (small AD bias at the
`K_ij≈K_max` boundary; the hard forward gives a zero single-key gradient).

**B.2 (this work)** — `particle_mc_jax_soft`: a **deterministic soft
occupancy-flow K-gate** with slot pre-allocation.  Each slot has a
continuous occupancy `o ∈ [0,1]`.  Per `lax.scan` step: draw a pair `∝ o`,
consume `δ = (K_ij/K_max)·min(o_i,o_j)` from both parents, write the
retained remnant into tail slot `n_init+k` (the scan counter IS the write
index — no dynamic pointer).  A cumulative **depletion budget** caps δ so
total net removal hits `depletion·n_init` exactly.

B.2 removes BOTH B.1 costs:
* **`k_ratio` eliminated**: the budget pins depletion regardless of
  `n_steps`, so `n_steps` is just a generous upper bound (`n_steps_mult`,
  default 3.0).  numpy-verified: depletion 0.10000, std 0.0000 for any
  `n_steps ≥ ~1.5·D·n_init`.
* **No straight-through**: the forward `δ` is smooth in the kernel params,
  so the kernel gradient flows directly.

Validation (numpy sandboxes + workstation pytest):
* Surviving `(m,χ)` parity vs numpy `particle_mc`: KS(m,χ) ≈ 0.009–0.015
  ≪ KS_crit 0.037; `W_total = (1−D)·n_init` exact.
* JAX-step numpy mirror: layout-invariance KS = 0.0000 (one-tail-slot-per-
  step ≡ packed pointer; dead occupancy-0 slots contribute nothing).
* M4 SVI: **both gates pass** ELBO-improves + finite-posterior.

Wiring: `build_mc_jax_response(gate="bernoulli"|"soft", n_steps_mult=3.0)`
and `build_gaussian_coagulation(mc_jax_gate=..., mc_jax_n_steps_mult=...)`.
Default stays `"bernoulli"` (additive; nothing changes unless you opt in).
Under `"soft"` the evaluator weight is `w_active · occupancy` (continuous,
AD-traceable) — the smooth replacement for B.1's binary `active_mask`.
`MCEvaluatorJax` is unchanged.

---

## 3. Test runbook (run on cardassia)

> **Ephemeral-sandbox fast loop** (verified 2026-05-29): an ephemeral
> CPU box (incl. the Cowork sandbox) can run the whole JAX path locally —
> `source demo/bootstrap_jax_sandbox.sh` (pip-installs jax/jaxlib/equinox/
> numpyro/pytest/jaxtyping/scipy from PyPI + exports `PYTHONPATH=src:.`),
> jax 0.6.2 CPU, B.2 tier 8/8 in ~5s.  Re-source per session (installs
> don't persist); re-export `PYTHONPATH` per shell call; CPU-only; long
> SVI may hit a ~45s/command cap → background or reduce steps.  On a
> persistent dev box (cardassia) just `pip install -e .` once.

```bash
cd hierarchical_AD_pop_engine

# B.2 sandbox tier (NEW) — S1–S8.  S2/S3/S4 are exact gates; S6/S7 are AD.
pytest tests/test_mc_particle_mc_jax_soft.py -v

# B.1 + evaluator + response regression (must stay green; B.2 is additive).
pytest tests/test_mc_particle_mc_jax.py tests/test_evaluation_mc_jax.py \
       tests/test_response_mc_jax.py tests/test_mc_sample_natals_jax.py -v

# M4 head-to-head (slow) — parametrised over BOTH gates (8 instances).
# Hard gates (ELBO improves, posterior finite) pass on both branches.
# Non-strict xfail: expected_count (gate-independent metric bias, §5),
# truth-in-CI (resolution-limited).
pytest -m slow -k integration_stage1_recovery_mc_jax -v

# numpy sandboxes (no JAX needed — run anywhere for intuition):
python demo/b2_soft_gate_derivation.py     # Q1 k_ratio / Q2 shape / Q3 smoothness
python demo/b2_jax_mirror_check.py         # JAX-step logic mirror
python demo/b2_parallel_preview.py         # discrete vs soft histograms (PNG)

# Visual head-to-heads (need JAX):
python demo/forward_modeling_mc_parallel.py          # numpy vs B.1 vs B.2 overlay+resid
python demo/forward_modeling_mc_vs_mc.py --gate soft # numpy vs B.2
```

---

## 4. NEXT BIG STEPS (priority order)

### 4.1 Commit + push, decide the default gate
Commit the B.2 work (§1), push from a networked machine.  Then DECIDE: if
`gate="soft"` M4 converges as well or better than `gate="bernoulli"`
(check the ELBO curves + posterior CIs side by side), **promote
`gate="soft"` to the default** and demote B.1 to opt-in.  B.2 is the
cleaner design (no magic factor, no straight-through bias); B.1 stays as
a validation cross-check.

### 4.2 ►gwkokab wiring — the headline (PLAN_gwkokab_fit.md Stage 2)
Stage 1 (single Gaussian, flat selection, hand-rolled Poisson likelihood)
is DONE and is what M4 exercises.

**DONE (2026-05-29): the bridge.**  `src/hierarchical_ad_pop_engine/
inference/gwkokab_bridge.py` — `CoagulationPopulation(numpyro Distribution)`
whose `log_prob(x=(m1,m2,χ1,χ2))` returns the engine's
`log_rate_density` (= the gwkokab `ScaledMixture` log-INTENSITY
convention, confirmed by reading `_scaledmixture.py`), a box `support`,
`coagulation_poisson_mean_estimator(model,T_obs,n_quad)->(μ,var)` giving
`T_obs·total_rate` (flat selection VT≡1, var=0), and
`make_coagulation_dist_fn(...)` returning the `dist_fn(**params)` gwkokab
calls.  With 1 sample/event + ln_offsets=0 + T_obs=1 it reduces to M4's
hand-rolled `Σ log_rate_density − total_rate`, so it's a faithful
generalisation.  Heavy engine import is LAZY (module imports under
jax+numpyro alone).  Tests: `tests/test_gwkokab_bridge.py` — G1–G5
(bridge glue + an inline replica of gwkokab's
`analytical_poisson_likelihood_fn`, AD-differentiable) PASS in-sandbox
against a fake response; G6–G7 (real engine + real gwkokab) SKIP in the
lightweight sandbox and run here on cardassia.

**► RUN G6–G7 ON CARDASSIA FIRST** (they're the real-integration gates):
```
export PYTHONPATH=src:.:../gwkokab/src   # or pip install -e . and -e ../gwkokab
python3 -m pytest tests/test_gwkokab_bridge.py -q   # all 7 should run (no skips)
```
If G6/G7 pass, the bridge is real-validated; then proceed to the fit.

**NEXT sub-steps to a running gwkokab fit:**
* **DONE (2026-05-30): priors/variables/variables_index wiring.**
  `src/hierarchical_ad_pop_engine/inference/stage1_wiring.py` —
  `build_stage1_variables()` (the 6 Stage-1 numpyro priors +
  `variables_index` built the gwkokab way, `{k:i for i,k in
  enumerate(sorted(variables))}`), `build_stage1_priors()`
  (`JointDistribution` over `sorted(variables)`, lazy gwkokab import),
  `build_coagulation_likelihood()` (thin wrap of the REAL
  `numpyro_analytical_poisson_likelihood`, `constant_params={}`,
  `variance_cut_threshold=None`), and a one-call `build_stage1_likelihood()`
  returning `{dist_fn, variables, variables_index, priors,
  poisson_mean_estimator, log_likelihood_fn}`.  The load-bearing detail is
  the **sorted-order contract**: the likelihood draws samples in
  `sorted(variables)` order and `variables_index[name]` must index into THAT
  list, else every param feeds the wrong `dist_fn` slot silently.  Tests:
  `tests/test_gwkokab_stage1_wiring.py` — W1–W6 (priors, bounds, the
  sorted-index contract, an inline `mapped_params` routing replica, and AD
  through the assembled wiring) PASS in the lightweight sandbox; W7–W9 (REAL
  gwkokab `JointDistribution` marginal-order check + REAL
  `numpyro_analytical_poisson_likelihood` run + `jax.grad` via
  `numpyro.infer.util.log_density`) are `importorskip`-gated like G6/G7.
  **All nine (incl. W7–W9) were exercised GREEN in this Cowork VM** against
  real gwkokab by stubbing only `RIFT.lalsimutils` (a single import-time
  symbol the wiring never calls) — see §3 "Running real gwkokab in the VM".
  cardassia re-runs them against the genuine RIFT stack (no stub) for the
  final gate.
* Mock catalogue with **multi-sample-per-event** posteriors + matching
  `ln_offsets` (reference-prior log-weights) — the bridge already handles
  the `(n_events, n_samples, 4)` shape via vmap; M4 used 1 sample/event.
* **Add `pdet(θ)` selection** (Stage 2): analytic SNR surrogate
  `SNR ∝ m_chirp^{5/6}`, threshold 8, `expit`-smoothed → fold into a
  `pdet`-weighted `poisson_mean_estimator` (Monte-Carlo the VT integral).
* **Two-pulse natal IMF** + float `kernel_a` (10 params).
* Open questions (PLAN_gwkokab_fit.md): `ScaledMixture.sample` vs
  inverse-CDF for mock gen; cleanest `pdet` API (callable vs injection
  set).  Acceptance: 10-param recovery, truth in 90% CI, `μ₁<μ₂` ordering.

This is where `mc_jax` AD-traceability pays off: SVI/NUTS gradients flow
through the whole cascade into gwkokab's likelihood (proven: the inline
G5 contract replica differentiates the analytical-Poisson logL w.r.t. the
population params).

**gwkokab version drift (2026-05-29).** The user replaced gwkokab with the
latest `main` (old tree preserved as the sibling `gwkokab_orig/`).  Verified
by diff that the bridge needs **NO API rewire**:
`numpyro_analytical_poisson_likelihood`, `poissonlikelihood_utils.
analytical_poisson_likelihood_fn`, and `models/utils/ScaledMixture` are
**byte-identical** old↔new, (only addition: `inference/factory.py::get_likelihood_fn`, an additive
selector returning the same `numpyro_analytical_poisson_likelihood`).  BUT
the ENGINE had a real break: gwkokab main renamed
`MadauDickinsonRedshift`->`MadauDickinsonRedshiftModel` and
`PowerlawRedshift`->`PowerlawRedshiftModel`; `natal/redshift.py` re-exported
the old names -> ImportError -> the G6/G7 (and M4) skip.  **Fixed
(commit 93d2161):** version-tolerant import (try `*Model`, fall back to old
names), aliased to the engine's stable unsuffixed names; constructors
unchanged (diff-verified), no call-site edits.  (A prior substring grep gave
a false "resolves" — exact-name checks confirm all OTHER engine-imported
gwkokab symbols [`doubly_truncated_power_law_log_norm_constant`,
`log_planck_taper_window`, `mass_sandwich`, `ScaledMixture`] are unchanged.)
`test_gwkokab_bridge.py` was hardened: G6/G7 now skip ONLY on
import/OSError (env) and FAIL loudly on any other exception (a real future
API drift), so a genuine break can never again hide as a silent skip.

**Dependency / local-operation note (important for the dev loop):** the
engine's `response` package transitively imports gwkokab (`natal.mass` →
`gwkokab.models.utils`), and `import gwkokab` pulls `errors.py` →
`RIFT.lalsimutils` → **lalsuite**.  So the full `response/coagulation`
path (M4, bridge G6/G7) needs the whole gwkokab+RIFT+lalsuite stack.

* **Lightweight pip sandbox** (`demo/bootstrap_jax_sandbox.sh`): covers the
  MC PRIMITIVES path only (`red_team/backup_particle_mc.py` → `remnants.*`,
  e.g. the B.2 tier + bridge G1–G5), which does NOT import
  `response/__init__` → no gwkokab needed.
* **Full stack**: use **`pixi.toml`** (repo root) — pulls `lalsuite` from
  conda-forge (prebuilt; avoids the slow/fragile RIFT+lalsuite pip-sdist
  build that does not converge on the near-full ephemeral sandbox) + the
  JAX/numpyro stack + editable `gwkokab` and engine.  `pixi run
  tests-bridge` / `tests-m4`.  (FYI: `lalsuite` DOES publish
  `manylinux_2_28_aarch64` + `osx-arm64` wheels, so a pure-pip env is also
  possible given enough disk + build time; pixi is just the robust path.)
  This session could NOT bring the full stack up in the ephemeral sandbox
  (disk ~1.8 GB free, RIFT sdist build stalled) — full-stack runs remain a
  workstation/pixi task.

**Running the REAL gwkokab inference/models layer in the Cowork VM
(2026-05-30, for the wiring W7–W9 tier).**  The `inference` /
`models.utils` code (`JointDistribution`,
`numpyro_analytical_poisson_likelihood`, `analytical_poisson_likelihood_fn`,
`ScaledMixture`) is pure jax/numpyro and never *calls* RIFT — but
`import gwkokab` pulls `errors.py → import RIFT.lalsimutils` at module load.
So you can exercise the real inference layer in the VM without lalsuite by
**stubbing that one import**:
```bash
# 1. JAX + the lightweight gwkokab deps, installed to a PERSISTENT mnt cache
#    (the VM's own / is ~99% full; /dev/shm resets every bash call, so neither
#    works as a target).  ~700 MB total.
SITE=<outputs-mnt>/.pylibs            # e.g. .../mnt/outputs/.pylibs
pip install --target=$SITE --no-cache-dir \
    jax jaxlib equinox numpyro jaxtyping scipy pytest \
    quadax interpax loguru h5py optax matplotlib pandas
# 2. stub the single RIFT symbol gwkokab.errors imports at load (never called
#    on the inference path):
STUB=<outputs-mnt>/.stubs; mkdir -p $STUB/RIFT
: > $STUB/RIFT/__init__.py
printf 'MsunInSec = 4.925491025543576e-06\n' > $STUB/RIFT/lalsimutils.py
# 3. run (ABSOLUTE PYTHONPATH entries — a relative gwkokab/src breaks once
#    pytest cds into the engine dir):
export PYTHONPATH=$SITE:$STUB:<repo>/gwkokab/src:<repo>/hierarchical_AD_pop_engine/src:<repo>/hierarchical_AD_pop_engine
export JAX_PLATFORMS=cpu TMPDIR=/dev/shm/tmp
python3 -m pytest tests/test_gwkokab_stage1_wiring.py -v   # all 9 incl. W7–W9
```
This is a DEV-LOOP convenience only — it is NOT a substitute for the real
RIFT run on cardassia (where the stub is absent and lalsuite is present).
The committed tests stay `importorskip`-gated so CI without gwkokab skips
W7–W9 rather than passing against a stub.  (Same Mc·v_PN code path that needs
`MsunInSec` is in `errors.error_factory*`, which neither the wiring nor the
bridge touches.)

### 4.3 Fix the `total_rate` metric bias (gates a real rate)
`MCResponseJax.total_rate` reads ~5× low at the M4 working point
(`n_seed=2000`, depletion=0.10): round-2 23.75, round-3 20.22, target 100.
**Gate-independent** — it lives in `MCEvaluatorJax.total_rate` (J3
Gaussian-KDE oversmoothing of the natal peak + quadrature span
`[min(m_active),max(m_active)]` clipping the natal Gaussian tails), fed
identically by both cascade gates.  Currently `expected_count` is
non-strict xfail.  Diagnostic plan: n_quad sweep (domain vs resolution)
+ n_seed sweep (KDE bandwidth `n^(-1/6)`).  Candidate fixes: widen
quadrature span to the natal analytic support `mu±5σ`; or a k-NN-balloon
density inside JAX.  Needed before quoting an absolute detected rate
(per-event SHAPE is already fine).

### 4.4 J2b — non-Aligned vector physics in the soft gate
`particle_mc_jax_soft` is Aligned-only (raises on non-Aligned), same as
B.1.  J2b adds per-event orientation sampling inside the scan (extra key
split per step) + vector Barausse-Rezzolla remnant spin / Lousto-Zlochower
kick + per-event retention.  Under misalignment + a soft-sigmoid
retention, `∂/∂v_esc` recovers (it's currently 0 under Aligned Heaviside).
The misalignment machinery already exists for the implicit path
(`spin_orientation.py`, `remnants/spin_vector.py`, `remnants/kick_vector.py`,
`remnants/marginal.py`) — J2b ports it into the JAX MC scan.

### 4.5 Revisit implicit/Picard (USER: only AFTER gwkokab/M4 ships)
The implicit grid-Picard solver is the production-quality path.  The user
wants to revisit it now that the MC normalization (density convention,
`1/n_init` rescaling) is well understood — but explicitly **after** the
MC→gwkokab path ships.  Don't preempt it.

---

## 5. Distilled memory — HARD RULES (these do not travel; honor them)

These are standing directives from the user.  Violating them has burned
sessions before.

* **No fudge factors.**  Empirical rescalings are forbidden; `ε`/rates have
  absolute meaning.  Disagreement-by-a-constant = a units/measure bug to
  find, not a knob to turn.  (B.2's `n_steps_mult` is a loose upper bound,
  NOT a precision knob — depletion is pinned by the budget;
  `w_event = K_ij/K_max` exactly.)
* **No fake physics stand-ins.**  If a model is Aligned-only, implement
  Aligned-only and RAISE on the rest — don't smear a scalar to fake a
  vector treatment.  (B.2's fractional occupancy is a legitimate
  importance-weighted particle-MC reinterpretation, not a smoothing hack;
  physics fits are the shared scalar BR/LZ/Heaviside triple.)
* **Red-team validation.**  numpy `particle_mc` is the parallel reference;
  the JAX path must KS-match it and must never modify it.  Gold-standard
  validation = an independent re-implementation sharing only the lowest-
  level physics primitives.
* **Implicit is production; perturbative is unreliable.**  New deliverables
  default to robust solvers; JAX-MC ships ADDITIVELY (default behaviour
  unchanged).
* **No frozen numerical results in unit tests.**  Use invariants /
  parameter-free references (equality to an analytic identity), not pinned
  `.npz` snapshots.
* **Audit the call graph after a dispatch refactor.**  Surface-level kwarg
  additions silently leak at internal call sites that pass positionally.
  (B.2's `gate` kwarg was threaded through the one real caller chain
  coagulation→factory→cascade; the test mirror calls the cascade directly.)
* **MC depletion tracks the engine.**  `mc_depletion_fraction` is set from
  the engine's `f_natal_depleted`, never hardcoded — the coupling is
  physics, not a knob.
* **Histograms, not KDEs, for human-facing plots.**  KDE smoothing erases
  the deep-cascade peaks (3G+ at discrete `M_rem` locations, ~100
  counts/bin).  `forward_modeling_*` + the B.2 demos are histogram-only;
  KDE (`MCEvaluatorJax`) is for the AD likelihood surface, not diagnostics.

## 5b. Distilled memory — AD LESSONS (hard-won this stream)

* **`jnp.where`/`max(x,eps)` guards the FORWARD, not the BACKWARD.**
  Dividing by a magnitude that can →0 is a latent NaN-in-grad: `d(a/x)/dx
  = −a/x²` explodes even when the forward is clamped.  This caused B.2's
  M4 SVI ELBO→nan (1000/1000 posterior non-finite) while the forward demo
  looked perfect — a backward-only NaN, unique to the soft gate's
  depletion clip.  **Fixed** with a division-free cap:
  `delta = min(delta_raw, max(budget_left,0)/mult)`, `mult∈{1,2}` (bounded
  gradient; bit-identical forward).  Tell: forward fine + one branch
  NaNs-under-SVI ⇒ hunt a branch-unique divide-by-vanishing.  Prefer
  `minimum`/`clip` caps over ratio rescales in AD code.
* **A single-key FD is NOT an AD reference for a θ-dependent stochastic
  sampler.**  `jax.grad` holds `jax.random.choice` indices fixed (it does
  not differentiate which pair was drawn); a single-key FD lets the
  selection shift with θ → they disagree by orders of magnitude (B.2 S7:
  AD 1.6e-3 vs single-key FD 0.55, all selection-shift noise).  Validate
  via fixed-selection-replay FD, many-key FD-mean, or just assert
  finite+non-zero.  Don't gate on single-key `|AD−FD|`.
* **A depletion/normalization budget pins the TOTAL; the parameter signal
  is in the SHAPE.**  `Σ w_mergers = Σδ` is pinned to `D·n_init` by the
  budget ⇒ its kernel_b gradient is correctly ~0.  Test kernel gradients
  on a shape observable (e.g. δ-weighted merger `⟨q⟩`), never the pinned
  total.
* **Gaussian-KDE oversmooths cascade χ peaks** vs a k-NN balloon
  (bandwidth, not kernel shape, is the lever).  This is the load-bearing
  suspect for the §4.3 `total_rate` bias and the M4 truth-in-CI xfail.

---

## 6. Open threads / breadcrumbs (one-line status)

* **B.2 soft gate** — DONE, green both gates, uncommitted.  (this file)
* **total_rate metric bias** — OPEN; gate-independent KDE+quadrature; §4.3.
* **gwkokab Stage 2** — IN PROGRESS; priors/variables/variables_index wiring
  DONE + green (incl. real-gwkokab W7–W9 via the RIFT stub), §4.2.  Remaining:
  multi-sample mock catalogue + `ln_offsets`, `pdet(θ)` selection, the 2-pulse
  10-param model, and a runnable end-to-end SVI/NUTS fit on cardassia.
* **J2b non-Aligned MC** — NOT STARTED; §4.4.
* **implicit/Picard revisit** — DEFERRED to after gwkokab ships (user).
* **aligned-χ cascade under-resolution / dN/dχ_eff engine-MC bias** —
  OPEN, orthogonal; aligned-only pathology; engine `N_g(m,χ)` vs MC
  effective `N_g` mechanism still unexplained.  Not on the gwkokab path.
* **misalignment Sessions A–D** — DONE for the implicit path; J2b is the
  MC analog.

---

## 7. Glossary (the shorthand in commit msgs / memos)

* **J-series** — the JAX-MC rewrite: J1 natal sampler, J2 cascade
  (`particle_mc_jax`), J3 `MCEvaluatorJax` (Gaussian KDE), J4
  `MCResponseJax` + `solver="mc_jax"`.  J2b = non-Aligned MC (open).
* **M-series** — milestones: M3a/M3b weighted MCEvaluator, M4 = the SVI
  recovery integration test (`test_integration_stage1_recovery_mc_jax.py`).
* **B.1 / B.2** — the two cascade K-gate implementations: B.1 Bernoulli
  (round 3, `particle_mc_jax`), B.2 soft occupancy flow
  (`particle_mc_jax_soft`, this work).
* **F-series** — engine performance refactors (orientation precompute,
  χ-out surrogate) on the implicit/basis path.
* **θ\*** — the Stage-1 truth point: `mu_natal=8, sigma_natal=1,
  m∈[4,80], natal_spin_loc≈0.05, natal_spin_scale=0.10, kernel_a=0,
  kernel_b=1, v_esc=1000`.
* **K(m1,m2) = (m1+m2)^a · q^b** — pairing kernel; `q=m_min/m_max`.
  With a=0, kernel_b alone controls mass-ratio preference.
* **occupancy** — B.2's continuous per-slot surviving fraction; the KDE
  weight is `w_active · occupancy`.

---

## 8. Repo orientation (key files)

```
src/hierarchical_ad_pop_engine/
  response/coagulation.py     # CoagulationResponse + build_gaussian_coagulation (solver dispatch)
  response/mc_jax.py          # MCResponseJax + build_mc_jax_response (gate dispatch)
  response/mc.py              # numpy MCResponse (solver="mc")
  evaluation/mc_jax.py        # MCEvaluatorJax (J3 Gaussian KDE) — total_rate bias lives here
  remnants/{mass,spin,kick,marginal,spin_vector,kick_vector}.py  # BR/LZ physics fits
  spin_orientation.py         # Aligned / Isotropic / Fisher orientation models
red_team/
  backup_particle_mc.py       # numpy particle_mc (oracle) + particle_mc_jax (B.1) + particle_mc_jax_soft (B.2)
  sim_coagulation.py          # numpy Doctor-perturbative backup
tests/
  test_mc_particle_mc_jax_soft.py            # B.2 tier
  test_integration_stage1_recovery_mc_jax.py # M4 (both gates)
demo/
  demo_gwkokab_fit_stage1.py  # Stage-1 fit harness (M4 reuses its helpers)
  PLAN_gwkokab_fit.md         # the gwkokab staging plan (§4.2 source)
  project_jax_mc_b2_soft_gate.md  # B.2 design memo
../gwkokab/                   # sibling repo: target inference framework
```
```bash
# get oriented fast on a new machine:
git -C hierarchical_AD_pop_engine log --oneline -8
git -C hierarchical_AD_pop_engine status -s
sed -n '1,60p' hierarchical_AD_pop_engine/demo/project_jax_mc_b2_soft_gate.md
```
```
