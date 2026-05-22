# PLAN — Forward-modeling observables (head-to-head cross-method)

Written 2026-05-21.  Handoff for the next session.

## Why this exists

Step away from inference-side tests (now parked behind
`PLAN_resolution_validation.md`) and return to the *physics*:
at a single θ\*, do the engine's predicted observables agree
with an independent particle-MC realisation of the same model?
Same fundamental question as the red-team work
(`PLAN_redteam_observable_validation.md`), with two deliberate
differences:

1. **Production solver only.**  Particle MC vs the *implicit*
   coagulation solver (`CoagulationResponse(..., solver="implicit")`).
   No perturbative.  User direction 2026-05-21: "the perturbative
   isn't reliable; if we were shipping that, we shipped the wrong
   thing."  Retooling the rest of the demo / integration-test
   stack off perturbative is a separate, additive session.
2. **Observable-first, human-readable visualisation.**  The output
   is a figure a LIGO PE reader could read at a glance, not a raw
   `N(m, χ)` dump.  Source-frame native marginals (sanity check),
   plus LIGO-derived axes (m1, q, χ\_eff, M\_chirp), plus 2D joints
   where correlations live.

Secondary intent: *showing* the observables across a resolution
sweep lets us *see* where the resolution limits diagnosed in
`PLAN_resolution_validation.md` actually bite the physics.  That
visual is worth more than the same information in a table.

## Scope (v1)

### Methods compared

* **Particle MC** via `red_team/observables_mc.py` (shipped on
  `observables` branch).
* **Implicit solver** via `CoagulationResponse(..., solver="implicit")`
  + `src/hierarchical_ad_pop_engine/observables.py` (1D/2D
  source-frame) + the red-team derived-axis adapters for LIGO
  observables.

Explicitly NOT in scope: perturbative, engine v3, SVI / inference.

### Observables (in figure order, columns of the multi-panel)

| Layer | Axis | Engine path | MC path |
|---|---|---|---|
| Source 1D | `dN/dm`        | `observables.py` 1D marginal | MC histogram on m   |
| Source 1D | `dN/dχ`        | `observables.py` 1D marginal | MC histogram on χ   |
| Astro 1D  | `dN/dm₁`       | red-team derived-axis adapter | MC histogram on m₁ |
| Astro 1D  | `dN/dq`        | red-team derived-axis adapter | MC histogram on q  |
| Astro 1D  | `dN/dχ_eff`    | red-team derived-axis adapter | MC histogram on χ_eff |
| Astro 1D  | `dN/dM_chirp`  | red-team derived-axis adapter | MC histogram on M_chirp |
| Astro 2D  | `dN/(dm₁ dq)`  | `observables.py` 2D joint    | MC 2D histogram |
| Astro 2D  | `dN/(dm₁ dχ_eff)` | derived-axis 2D            | MC 2D histogram |

User note: source-frame `dN/dm`, `dN/dχ` are "lazy / unphysical"
but useful as a baseline sanity check on the engine's native
quadrature.  Keep them.

### Resolution sweep

* CLI flag: `--resolutions 32x10,64x20,128x40,256x80` (comma-
  separated `N_m × N_χ` pairs).  Default: a single baseline pair.
* Each resolution becomes one *row* of panels in the figure.
* Every figure label, legend entry, output filename, and JSON
  metadata field includes `(N_m, N_χ)` explicitly so post-hoc
  comparison across runs is unambiguous.
* Internal smoke survey: baseline-only.  The actual sweep is the
  user's exercise.

## Non-goals

* No inference / SVI / NUTS.
* No perturbative solver — use implicit, full stop.
* No engine v3 (parallel track on `engine_fem`).
* No automated resolution-convergence assertions.  The script
  *visualises*; it does not gate.
* No new physics primitives — everything plumbed through existing
  `src/` and `red_team/` interfaces.

## File layout

`demo/forward_modeling_observables.py`
* Demo-style script (not a pytest, not a notebook).
* CLI flags:
  * `--resolutions`   — comma-separated `N_m × N_χ` pairs.
  * `--n-mc`          — particle count for MC realisation.
  * `--seed`          — deterministic JAX PRNG root.
  * `--epsilon`       — implicit-solver depletion strength.
  * `--v-esc`         — environment retention velocity.
  * `--out-hash`      — output-dir suffix (default: random hex,
    per [[feedback-demo-output-conventions]]).
* Outputs:
  * Figures → `demo/figures_fmobs_<hash>/`
  * Data artifacts (npz per (resolution, observable) + JSON
    metadata) → `demo/artifacts_fmobs_<hash>/`

## Implementation sketch

1. Build truth response at θ\* with `solver="implicit"`.
2. Predict engine observables (source 1D, astro 1D, astro 2D)
   via `src/observables.py` + red-team derived-axis adapters.
3. Run particle MC at the same θ\* with the existing
   `red_team/observables_mc.py` machinery.
4. Compute MC histograms on the same axes as the engine outputs.
5. For each resolution in `--resolutions`, repeat steps 1–4 at
   that `(N_m, N_χ)`.
6. Render a multi-row figure: rows = resolutions, columns =
   observables.  Engine + MC overlaid in each panel; resolution
   in panel title.
7. Save figure + per-cell `.npz` artifacts + one consolidated
   `metadata.json` (θ\*, FIXED, resolutions, MC seed, MC count,
   git rev if available).

Sequence for next session:
1. Read this plan + `PLAN_redteam_observable_validation.md` to
   reload context (5 min).
2. Verify the four prerequisites below.
3. Scaffold `demo/forward_modeling_observables.py` end-to-end at
   one resolution (no sweep yet).  Smoke-test.
4. Add the resolution-sweep loop.  Smoke-test at two settings.
5. Hand back a baseline figure for inspection.

## Prerequisites — verify at session start

* `CoagulationResponse(..., solver="implicit")` accepts current
  natal/kernel param dicts and converges at the resolutions we
  want.  Per [[project-engine-implicit-refactor]] this should be
  fine for 1G/2G; higher-K behaviour to be confirmed.
* `src/hierarchical_ad_pop_engine/observables.py` 2D joint
  marginals are functional (1D is shipped; per
  [[project-observables-marginals-done]] 2D should be too —
  spot-check).
* `red_team/observables_mc.py` produces derived-axis MC
  histograms for `m1, q, χ_eff, M_chirp`.  These were shipped on
  `observables` branch; the files are currently untracked but
  functional.
* `BinaryGrid.default(m_min, m_max, n_m=..., n_chi=...)` accepts
  arbitrary `(n_m, n_chi)`.  Yes (verified during integration-test
  work).

Any failure on the above blocks this plan.  Fix first, then
proceed.

## Definition of done

* Single CLI invocation produces:
  * One PNG figure with rows = resolutions, columns = observables;
    engine + MC overlaid in each panel.
  * One JSON metadata file capturing θ\*, FIXED params,
    resolutions used, MC seed, MC sample count.
  * One `.npz` per (resolution, observable) cell, so the user
    can re-plot without re-running.
* All filenames, panel titles, and legend entries include the
  resolution `(N_m, N_χ)` unambiguously.
* Baseline smoke run produces sensible figures by visual
  inspection (no automated tolerance gate).
* User can swap `--resolutions` to drive their own sweep with
  no code changes.

## Open questions for the next session

* Baseline resolution to start at — suggest 64×20 to match the
  existing Stage 1 demo, with 128×40 as a "high-res" reference.
  Resolve at session start based on implicit-solver compile time.
* MC sample count — start at `N_mc = 5e4`; bump to 1e5 if MC
  noise visibly dominates the comparison.
* Whether to plot perturbative as a labelled-deprecated reference
  line so the user can *see* why perturbative is the wrong ship.
  Default: no for v1.  Revisit if the implicit-vs-MC comparison
  looks clean and we want the contrast.

## Related

* [[feedback-perturbative-unreliable-implicit-production]] — the
  user directive driving "implicit only".
* [[project-redteam-observable-tests-done]] — the red-team MC
  + derived-axis paths this demo reuses.
* [[project-observables-marginals-done]] — the engine observable
  module this demo reuses.
* [[project-engine-implicit-refactor]] — implicit solver status.
* `demo/PLAN_resolution_validation.md` — the systematic
  resolution / control study this visualisation is meant to
  inform but not replace.
* [[feedback-no-fake-physics-stand-ins]] — implicit is the
  production solver; do not smear perturbative results to
  pretend otherwise.
* [[feedback-demo-output-conventions]] — `figures_<hash>/`,
  `artifacts_<hash>/`, m1 ≥ m2, log-y on `dN/d*` plots.
