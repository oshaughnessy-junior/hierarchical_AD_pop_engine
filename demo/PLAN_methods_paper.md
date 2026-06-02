# PLAN — methods paper (proof-of-concept) structure & dependencies

**Date:** 2026-06-01 (objectives set by user for next session). **Status:**
planning scaffold. Next session: user will point to the **gwkokab paper** to
scaffold the results-section format.

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
