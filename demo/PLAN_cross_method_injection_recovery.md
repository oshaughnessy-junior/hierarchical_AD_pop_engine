# Plan — cross-method injection / recovery (MC-inject → implicit-recover)

**Date:** 2026-06-01. **Branch:** `observables`. **Status:** planning — the
forward models are validated; this is the calibration + harness work to combine
them safely.

## Motivation

The two-pulse / single-pulse recovery arc converged on a clean division of
labour, each method used where it is strong and never where it is weak:

* **Inject with particle MC.** Exact particle draws — no grid-cell
  quantisation, no within-cell jitter broadening (the `sample_from_model`
  coarse-grid artefact that inflated implicit-recovered σ on 2026-06-01). MC
  generates sharp natal features (near-δ spin at χ≈0) and the cascade
  natively.
* **Recover with implicit.** A sharp, KDE-free grid density. No Silverman
  oversmoothing of the bimodal natal IMF, no k-NN balloon noise — the failure
  modes that drove the `mc_jax` two-pulse λ bias
  (`DIAGNOSIS_two_pulse_kde_floor.md`).

This is a **cross-method injection/recovery**: the injection density and the
recovery density are produced by *different* numerical methods. That is
powerful (plays to both strengths) and dangerous (a method-to-method systematic
masquerades as a population feature). The danger is entirely calibration; this
plan is about making the calibration *continuously validated*, never assumed.

The same-method baselines stay as controls:
* `mc → mc` (inject MC, recover mc_jax KDE): the current working path; clean
  recovery but KDE-limited width fidelity.
* `implicit → implicit` (self-consistent grid): now faithful after the 2×-fit-
  grid sampler fix; validates the implicit forward model in isolation.

## The four calibration hazards (and the existing tooling for each)

### H1 — density-convention parity (θ-dependent drift)
MC carries a count→probability rescale (`/n_init²`, Session M2 / J-series);
implicit is a trapezoid-measure grid density (`observables.total_rate`,
`log_density_at`). A *constant* offset is absorbed by `log_rate` at
calibration. The killer is a **θ-dependent** convention drift — it biases the
recovered shape, not just the rate.

* **Gate:** `red_team/compare_all.py` JS(engine_v2 implicit ↔ particle MC) must
  hold **at the injection θ**, not only at one fiducial. Sweep a small θ-ball
  around the injection point and assert JS < threshold on `dN/dm` AND `dN/dχ`.
* **Test:** a slow pytest that runs the inject-θ JS check; fails on regression.

### H2 — ε ↔ depletion-fraction / MC-time mapping (the sharpest one)
MC stops at a depletion fraction `f`; implicit evolves to ε (coupling×time). If
`f` and ε do not map to the **same physical evolution time**, the cascade depth
differs → a 2G+ offset that looks exactly like a real hierarchical feature.

* **Calibration (DONE 2026-06-01, `--gen mc --solver implicit`):** the M1
  closed form `ε = 2·n_mergers/(K_eff·N_active²)` is **exact only for
  constant-K / monodisperse**; for the Stage-1 kernel `K_eff` is a
  state-averaged self-consistent quantity (see M1 module "Generalisation").
  Rather than trust the approximate closed form, the harness matches the
  **dimensionless depletion fraction directly**: read the MC injection's
  `f_dep = 1 − W_total/n_init`, then root-find (geometric bisection,
  `_bisect_geometric`) the implicit ε such that
  `solve_implicit_fixed_point(ε).f_natal_depleted == f_dep`. This is
  convention-safe (no weight-normalisation or `n_mergers`-padding ambiguity —
  `f_dep` is a pure ratio defined identically by both methods) and makes the
  H2 parity **exact by construction**. The M1 closed form remains the sanity
  reference (`test_mc_time_calibration.py`); the bisection is its
  convention-robust production replacement for non-constant K.
* **Test:** `test_mc_pair_inject.py` pins the bisection against the saturating
  analogue `f(ε)=ε/(1+ε)` (closed-form inverse). The harness prints `|Δf|`
  between MC `f_dep` and implicit `f_natal_depleted` at ε* every run — the live
  H2 gate.

### H3 — shared low-level physics primitives only
Both paths must call the *same* remnant mass/spin (Barausse–Rezzolla aligned),
retention (`P_retain` / soft-sigmoid), and pairing kernel. Per the red-team
rule (`feedback_red_team_validation`), the two methods should share **nothing
above the lowest-level physics** — but they MUST share that. A remnant-formula
divergence becomes a fake recovery bias.

* **Check:** audit that MC (`response/mc_jax.py`) and implicit
  (`response/implicit.py`) import remnant/kernel/retention from the same
  `remnants/`, `kernels/` modules. Add a test that pins identical remnant
  `(m_rem, χ_rem)` from both call paths at sampled `(m1,m2,χ1,χ2)`.

### H4 — recovery spin resolution at the boundary
Implicit recovery must resolve χ≈0 (natal, scale≈0.05) AND χ≈0.69 (cascade
remnant) simultaneously. A **uniform** χ grid does this expensively (the
n_chi≳64 wall that originally pushed the project MC-ward). MC injection
sidesteps it, but implicit recovery cannot — so this is on the critical path.

* **Fix (production):** clustered / non-uniform χ nodes concentrated at the two
  features — the `engine_fem` branch (Chebyshev–Lobatto Galerkin,
  `project_engine_v3_basis_planned`) is the seed. Consider an analytic-1G +
  basis-only-cascade split so the near-δ natal spin source is never threaded
  through a grid at all.
* **Interim:** uniform grid at the resolution the single-pulse residual sweep
  shows is sufficient; document the cost.

## Steps

1. **Inject-MC harness.** A `--gen mc` mode in `fit_stage1_gwkokab.py` (and the
   two-pulse fit) that draws the catalogue from a particle-MC response
   (`solver="mc"` / `mc_jax` particle draws) instead of the grid sampler, then
   recovers with `--solver implicit`. The MC draws are the exact particles — no
   `sample_from_model` grid/jitter.
2. **ε* wiring (H2).** Compute ε* from the MC stopping state via the M1 closed
   form; build the implicit recovery response at ε*. Assert `f_natal_depleted`
   parity.
3. **Cross-method gate (H1).** Promote the inject-θ `compare_all` JS check to a
   pytest gate; run it as the first thing the harness does, before any fit.
4. **Physics-primitive audit + pin test (H3).**
5. **Single-pulse cross-method recovery.** Inject MC, recover implicit; confirm
   `(mu_natal, sigma_natal, natal_spin_*)` recovery is at least as good as the
   faithful `implicit→implicit` self-consistent baseline. Differences here are
   pure cross-method systematics — investigate, don't paper over.
6. **Two-pulse cross-method recovery.** The headline: does λ recover at truth
   when injected with MC and recovered with implicit? This is the experiment the
   whole arc has been building toward.
7. **Resolution pass (H4).** Only after 1–6 are green: clustered-χ / basis if
   the uniform-grid recovery cost is unacceptable at production resolution.

## Anti-goals

* **No hand-tuned ε.** ε* comes from the M1 mapping or the harness is invalid
  (`feedback_no_fudge_factors`).
* **No silent convention patches.** If MC and implicit disagree at the
  injection θ, find the measure/Jacobian bug (the cell-area precedent) — do not
  add a θ-dependent fudge to force agreement.
* **Don't drop the same-method controls.** `mc→mc` and `implicit→implicit` stay
  as the bracketing baselines; cross-method must sit between them sensibly.
* **Don't ship cross-method without the H1 gate live.** An ungated cross-method
  fit is a systematic-bias generator with a credible-looking corner plot.

## Open questions

* Does the MC injection need importance weights to match the implicit recovery
  measure exactly, or is the particle empirical measure sufficient once H1
  holds? (Likely sufficient; confirm via the single-pulse cross-method step.)
* Is ε* stable across the θ-ball, or does it need re-computing per θ-sample
  during inference? (Recovery θ moves; injection θ is fixed — so ε* is computed
  once at injection, but the *recovery* implicit ε is a fixed model constant.
  Confirm these are the same ε or document why they differ.)
