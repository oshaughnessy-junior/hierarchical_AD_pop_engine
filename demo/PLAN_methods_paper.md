# PLAN — methods paper (proof-of-concept) structure & dependencies

**Date:** 2026-06-01 (objectives set by user for next session). **Status:**
planning scaffold. Next session: user will point to the **gwkokab paper** to
scaffold the results-section format.

> **UPDATE 2026-06-04 (LATE — workstation high-res rerun findings, ROS):**
> (a) **MC size**: push the sweep benchmark BEYOND 1e6 seeds — the
> secondary (2g+) peaks are still under-populated in the histogram.
> Numpy MC is ~0.5 ms/seed single-core → 1e7 ≈ 90 min; use seed-split
> chunked parallel runs with merged histograms (raw hists merge exactly;
> seed via independent `default_rng` streams — see
> `demo/forward_modeling_mc_parallel.py`) or the JAX MC. Bump
> `RS_MC_N_SEED` once timed.
> (b) **n_chi=64 RINGING at high spin**: the grid model oscillates trying
> to represent the *completely empty* gap between the natal spins and the
> 1g remnant peak. RISK: with hundreds of sources now, ringing could
> imprint artificial features on recovered populations. First diagnostic:
> re-render with `--render hat` — if node VALUES oscillate it's the
> solver (bilinear deposit of a sub-cell-width peak + positivity-clipped
> Picard around zero), not the pchip-log rendering. Candidate
> mitigations, in order: clustered grid w/ FEWER nodes in the empty gap
> (less ringing DOF), deposit-side mild diffusion/limiter (TVD-style,
> mass-conserving), damping/tol tuning; physically, the next-paper
> SPINDOWN mechanism fills the gap and likely removes the pathology at
> the source.
> (c) **Path forward — may need to promote MC again.** gwkokab's two
> likelihood modes are both density-mode (`model.log_prob` at event PE
> samples — verified in `inference/poissonlikelihood_utils.py`); a
> SAMPLE-mode (per-event likelihood evaluators × samples from the model,
> which MC produces naturally — no KDE) is NOT implemented. Logged as a
> standalone development task for a separate agent on a gwkokab fork:
> **`demo/PLAN_gwkokab_sampled_likelihood.md`** (math, code evidence, API
> sketch, acceptance gates, thread-back plan).

> **UPDATE 2026-06-04 (END OF SESSION — chi-spin work CLOSED, 4 rounds):**
> Rounds 2–4 (on top of round 1 below): χ→1 tail artifact fixed
> (`max_gap_factor` cap, MC-confirmed), MC raw-hist overlays everywhere
> (no KDE), pchip-log vs hat rendering, continuous rel-L1 resolution
> sweep emitting a machine-generated LaTeX table (relL1 + t_solve) +
> standalone recon figure for the paper, mass-grid investigation
> (gen-clustered log-m 1.3–1.9× better; geomspace stays default),
> methods 2.B.3 full prose (ε/f_dep intros → calibration ¶ → deep-cascade
> limitation + spindown next-paper flag), Results 3.A compressed to one
> 4-paragraph section. Engine commits abbe247→ed195a7; paper repo
> 6f7a022→cc9ed91 + HANDOFF eighth pass. ALL Cowork-local — push from
> workstation, then `make figures-chi-spin` at paper res + recommit.
> Memory: `project_clustered_chi_and_fisher_bridge_done`. NOTE: a
> concurrent agent is fixing a circular import in NuclearMatter-Backend —
> check for foreign commits before committing in shared repos
> (`project_concurrent_agent_nuclearmatter_warning`).

> **UPDATE 2026-06-04 (TECHNICAL ROUND #1 DONE — believable + interesting spin):**
> Both spin deliverables landed and are sandbox-validated.
>
> **(1) Nonuniform χ grid ("believable") — DONE.** Priority-1 item below.
> Implemented as a *node retrofit*, NOT a basis rewrite: the whole χ
> machinery (gain-side bilinear scatter, loss-side trapezoid, the F2
> bilinear-hat surrogate) already reaches the grid only through
> `jnp.interp(·, grid.chi, ·)` and `jnp.diff(grid.chi)`, so it is exact on
> any monotone χ grid — the only thing hardcoding uniformity was
> `grids.BinaryGrid.default`'s `linspace`. New `grids.clustered_chi_nodes()`
> (smooth inverse-CDF equidistribution: uniform floor + Gaussian bumps at
> the natal peak χ≈0.05 and the cascade peak χ≈0.6865 = engine
> `chi_rem(30,30,0,0)`) + `BinaryGrid.clustered_chi(...)` classmethod
> (drop-in for `.default`, plugs straight into the bridge's `grid=`).
> Validated: bilinear-hat partition-of-unity exact to machine-0 on the
> nonuniform nodes; trapezoid exact on linear; **clustered beats uniform
> 1.8× rel-L1** on a sharp two-peak test density at matched n_χ=32; AD
> through the implicit solve still finite+nonzero. Tests:
> `tests/test_clustered_chi_grid.py` (10, gwkokab-free, runs in any
> sandbox).
>
> **(2) Fisher-κ through the bridge ("interesting") — DONE.** Impl-support
> item 1 below. Added `orientation` / `retention_kind` / `w_over_v_esc`
> params to `make_coagulation_dist_fn` (gwkokab_bridge.py), forwarded to
> both `build_gaussian_coagulation` and `build_two_pulse_coagulation`
> (which already accepted them). Fixed-per-run, inference-ready: the IFT
> custom-VJP already flows `∂/∂κ` (verified `∂ΣN*/∂κ ≈ 0.11`, finite+nonzero)
> AND restores a nonzero `∂/∂v_esc` under Fisher+sigmoid (≈ −1.2e-3; it is
> identically 0 under the Aligned+Heaviside default). Lifting κ to a *fit*
> parameter later is a one-line closure change, no engine work. Added a
> read-only `CoagulationPopulation.response` accessor. Tests:
> `tests/test_fisher_bridge_clustered.py` (4, engine tier, skips if gwkokab
> absent — same convention as the G6–G7 full-stack tier).
>
> **Demo:** `demo/forward_modeling_chi_grid_kappa.py` — two-panel `dN/dχ`
> figure (LEFT clustered-vs-uniform resolution of the cascade peak; RIGHT
> the κ sweep Isotropic→aligned). Reduced-res defaults fit the sandbox;
> production figure is a workstation rerun (`--n-m 64 --n-chi 24`).
> Output → `demo/figures_chi_grid_kappa/chi_grid_kappa.png`.
>
> **Sandbox recipe addendum** (beyond `demo/bootstrap_jax_sandbox.sh`): the
> full engine import additionally needs `pip install quadax h5py` and the
> 2-line `RIFT.lalsimutils` shim (`MsunInSec = 4.925491025543576e-06`) on
> PYTHONPATH, plus `gwkokab/src` — then `solver="implicit"` runs in-VM. All
> new tests + G1–G7 + orientation_models GREEN in-sandbox this session.
>
> **Still open from the priority list below:** (2) full *spectral* basis
> rendering for `dN/dχ` (the clustered grid + bilinear hats already
> de-blocks the trace; the engine_fem Chebyshev path remains the
> higher-order option) and (3) the paper-demo reruns at matched ε +
> production resolution + the joint resolution sweep — both workstation.

> **UPDATE 2026-06-03:** the paper repo `../../gwk-hierarchical-initial` is now
> scaffolded — `analyses/ demos/ paper/` + a compiling revtex4-2 LaTeX skeleton
> (intro 4 paras, methods 1–5, results parts 1–2) with `% TODO/SOURCE/FIG/CITE`
> anchors mapping every section to this PLAN, the cross-method HANDOFF, DESIGN.md,
> and the proposal. `refs.bib` is a stub flagged `% RECONCILE` (bibcodes pending
> the BibtexArchives — not mounted that session). Build-checked via
> `paper/_buildcheck/` (article class; VM has no RevTeX). See
> `../../gwk-hierarchical-initial/HANDOFF.md`. Next: bib reconcile, deep-research
> the background paragraph, run `analyses/` for numbers/figures.

> **UPDATE 2026-06-03 (end of paper-narrative round):** methods 1–5 are PROSE
> (bib reconciled against BibtexArchives; ZeeMalagon O4a defaults adopted for
> sensitivity/mixture/real-data); placeholder figures deployed for Results A.1
> (stage1 set) and 3.A.2 (fmobs panel crops); Results B figure DESIGN specced +
> implemented in `../../gwk-hierarchical-initial/demos/forward_model_examples/
> fit_two_pulse_gwkokab.py --paper-figs` (blocked corners mass/spin/couplings +
> natal-PPD band — run on workstation, images pending). Full state:
> `../../gwk-hierarchical-initial/HANDOFF.md`.
>
> **NEXT ROUND = TECHNICAL (engine), not paper: NONUNIFORM χ QUADRATURE.**
> The paper round surfaced the priority order:
> 1. **Nonuniform χ quadrature** in the implicit solver — clustered nodes at
>    the natal peak (χ≈0, scale~0.05) and the cascade remnant peak (χ≈0.69);
>    this is H4 of PLAN_cross_method_injection_recovery.md and the root of the
>    implicit-spin weakness acknowledged in the 3.A.2 figure. Seeds:
>    engine_fem branch (Chebyshev–Lobatto basis, PLAN_engine_v3_basis.md) or a
>    direct nonuniform-node retrofit of the uniform grid.
> 2. **Basis-function rendering for dN/dχ** — plots must evaluate the
>    low-level basis representation built into the quadrature scheme, not the
>    blocky grid trace (paper figure TODOs depend on this).
> 3. Then rerun the paper demos (demos/forward_model_examples/) at matched ε
>    (H2) + production resolution, and the joint resolution sweep (the
>    headline robustness figure).

## Paper structure

* **Methods section.** The hierarchical-coagulation population model + the
  AD/implicit machinery: response function (single-env) × formation rate;
  implicit Picard fixed point with IFT custom-VJP; the gwkokab analytical-Poisson
  likelihood bridge. (Most of this is built + validated this session.)
* **ONE results section, sketched** (scaffold from the gwkokab paper format —
  get the pointer next session). Lead with the cleanest recovery result.

## Results A — single pulse (the core demo)

Three recovery configurations, increasing realism:
1. **Basic**: delta posteriors in CODE coordinates `(m1,m2,χ1,χ2)`. = the
   EXACT-injection path we defaulted to (f9c4856). Clean forward-model recovery.
2. **Advanced**: synthetic data in 'REAL' coordinates `(Mc, η, χ_eff)` with a
   proper measurement model (`PLAN_measurement_model.md` — gwkokab errors +
   ln_offsets). Recovery through the coord transform + Jacobian glue.
3. **(if possible) Fisher-κ** as a 3rd option (spin-orientation concentration).

Orientation: **isotropic case** is the baseline for A. (Fisher-κ = the stretch
goal 3rd option.)

## Results B — two pulse

Same recovery, two-pulse natal IMF (the λ result). Cross-method + exact injection
already working (`HANDOFF_cross_method_session.md`); re-baseline with exact
injection, then add the measurement-model (real-coords) variant.

## Results C — real data comparison

Fit real catalog data with a **MIXTURE of the hierarchical (coagulation) model +
a NON-hierarchical (standard population) component**, to IDENTIFY/extract the
hierarchical (cascade) sub-population. (New modelling: a mixture likelihood where
one component is the coagulation response and the other a vanilla population;
infer the mixing fraction = "is there a hierarchical component?".)

## Implementation support (the glue these results need)

1. **Fisher-κ orientation.** ENGINE-READY (`spin_orientation.py`:
   `Fisher(κ)`/`Isotropic()`/`Aligned()`; `build_two_pulse_coagulation` accepts
   `orientation=`), but **NOT wired through the bridge/fit demos**. Task: thread
   `orientation=Fisher(kappa=…)` (and `Isotropic()`) through
   `make_coagulation_dist_fn` → `build_*_coagulation`; decide if κ is a fit
   parameter or fixed per-run. Needed for A.3 and a realistic spin treatment.
2. **Coordinates & Jacobian glue.** `(m1,m2)↔(Mc,η)`, `(χ1,χ2)↔χ_eff` transforms
   + the log-Jacobian into `ln_offsets`. The bulk of the measurement-model work
   (`PLAN_measurement_model.md` "THE CATCH"). Needed for A.2, B-advanced, C.

## Dependency order (suggested)

1. Re-baseline A.1 / B with EXACT injection (already coded — just run; figures).
2. Coords+Jacobian glue → A.2 (measurement-model recovery) + the verification
   gate (recover true widths from synthetic PE).
3. Wire Fisher-κ through the bridge → A.3 (+ isotropic baseline already there).
4. Results C mixture model (new likelihood component) — largest new build.
5. Methods text + the one results-section scaffold (after the gwkokab pointer).

## Cross-refs
`PLAN_measurement_model.md` (coords/Jacobian, synthetic PE), `PLAN_cross_method_
injection_recovery.md` (H1/H2/H3 gates), `HANDOFF_cross_method_session.md`
(current state + paper framing), `PLAN_misalignment_mode.md` (orientation
physics). Memory: `project_implicit_pivot_bridge_wired`.
