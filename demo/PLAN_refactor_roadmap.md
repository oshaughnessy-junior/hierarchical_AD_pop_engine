# Refactor roadmap — 2026-05-27 planning session

Written 2026-05-27 after a planning conversation with the user about
the implicit-solver footprint problems exposed by Session D and
production-resolution non-Aligned runs.

This is a **top-level sequencing note** binding the two PLAN
documents written this session:

* `PLAN_mc_first_reprioritization.md` — promote particle MC to a
  first-class JAX-wired likelihood path (4 sessions M1-M4 + optional
  M5).
* `PLAN_implicit_footprint_refactor.md` — separable angular
  surrogate for the implicit solver to collapse the 128-point
  orientation overhead to construction time (3 sessions F1-F3 +
  optional F4).

## The framing

The two streams cover **orthogonal regimes**, not competitors:

| Regime                              | Production solver | Why |
|-------------------------------------|-------------------|-----|
| Aligned, sharp AD on natal/kernel   | Implicit          | Already shipped, fast, clean IFT VJP |
| Misaligned, broad-posterior events  | MC (post-promo)   | Avoids angular-quadrature blow-up entirely |
| Misaligned, narrow-posterior events | Implicit (post-F1+F2) OR importance-sampled MC (M5) | Open question — measure at F3 |
| Aligned, fast misalignment-knob fit | Implicit          | κ as AD-leaf is cleaner under implicit |

The "both" is doing real work. There is no decision to ship one and
drop the other.

## Recommended interleave

The user runs ~one session per evening. Recommended order:

| # | Session | Stream | Rationale |
|---|---------|--------|-----------|
| 1 | M1 (time calibration) | MC | Unblocks the rest of MC; doesn't touch JAX-fragile surfaces; cheap |
| 2 | F1 (precompute retention + mean chi_rem) | Implicit | Largest single speedup from a tightly-scoped diff; preserves Aligned bit-stably |
| 3 | M2 (MCResponse + Distribution shim) | MC | Light up `solver="mc"` end-to-end through gwkokab |
| 4 | F2 (χ-out basis surrogate) | Implicit | Closes the gain side; gates with an R-convergence sweep |
| 5 | F3 (benchmark + decision) | Implicit | The gate: does implicit cross the production budget? |
| 6 | M3 / M4 (stratified natal / Stage 1 recovery) | MC | Independent of F3 outcome |

Interleaving the streams keeps a single workstation environment
warm, and the user can revisit each stream's PLAN without a context
swap between sessions.

## Decision points

* **After F3**: if production `(96, 32)` solves in `< 30 s` and `<
  4 GB`, implicit is rescued and the misaligned-implicit path is
  the default. Otherwise: drop misaligned-implicit for production,
  use Aligned-implicit only, ride MC for misalignment.
* **After M4**: if Stage 1 recovery under `solver="mc"` fits within
  the existing σ_natal / kernel_b tolerances, MC is production-
  ready for broad-posterior BBHs. Otherwise: defer MC promotion,
  rely on the implicit path (assuming F3 went well).
* **If both F3 and M4 miss**: escalate to F4 (Strategy C —
  spherical-harmonic decomposition) and M5 (narrow-posterior IS).
  Pessimistic 2-3 extra sessions.

## What stays bit-stable through both PLANs

* Aligned implicit — already shipped. Every PLAN session has a
  guard test that confirms `orientation=Aligned()` ≡
  `orientation=None` bit-for-bit through the new code path.
* Existing perturbative solver — already legacy
  (`[[feedback-perturbative-unreliable-implicit-production]]`),
  not touched.
* Existing observable / evaluator surfaces — both PLANs go through
  the existing protocols without changing their signature.

## What the user owns end-to-end

* Live workstation runs at every Session's pytest + demo gate.
* Final git commits — Cowork VM can't reach the macOS `.git`.
* Decision gates at F3 and M4.

## Related

* `PLAN_mc_first_reprioritization.md`
* `PLAN_implicit_footprint_refactor.md`
* `[[project-session-handoff-20260527-late]]` — the cumulative
  state going into this planning session.
* `[[project-session-handoff-20260527-planning]]` — the breadcrumb
  for this session itself.
