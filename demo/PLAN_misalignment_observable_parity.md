# PLAN — Misalignment Session D: observable-space MC ↔ engine parity

**Status**: planned 2026-05-27.  Scope locked with the user 2026-05-27:

* **Orientation grid**: ``Aligned`` + ``Isotropic`` + one representative
  ``Fisher(κ=5)``.  Not a full κ sweep — one κ value demonstrates the
  Fisher knob without paying for the sweep.
* **Deliverable**: comparison figures + unified JSON metrics summary
  + ``slow``-marked workstation pytest gating the per-orientation
  parity invariants.
* **Demo shape**: NEW ``demo/misalignment_observable_parity.py``
  layered on top of the existing ``demo/forward_modeling_observables.py``
  building blocks.  The existing demo's behaviour and outputs are
  bit-stable — Session D does not edit it.

## Background

The misalignment work landed across three sessions:

* Session A (`[[project-misalignment-session-a-done]]`) — vector
  remnant primitives + canonical BR 2009 cross-term fix.
* Session B (`[[project-misalignment-session-b-done]]`) —
  `remnants/marginal.py` + Aligned-dispatch wiring at every engine
  surface (retention, `_step_one_generation`, `build_remnant_tensor`,
  `build_loss_tensor`).
* Session C (`[[project-misalignment-session-c-done]]`) — MC
  orientation field + `--orientation` CLI flag + demo-level
  ``∂total_rate/∂v_esc`` cash-in + solver-level invariant pytest +
  loss-side `lax.scan` refactor.  All 7 invariants green on
  workstation round 4.

Session C's "What the tests do NOT cover (deferred)" section called
out a missing piece:

> **MC ↔ engine parity on the misaligned path.**  Session B has it at
> the marginal layer (``tests/test_remnant_marginal.py``).  The
> solver-level analogue — observable-level JS between MC and engine
> under matching ``--orientation isotropic`` — is captured by
> ``demo/forward_modeling_observables.py`` not pytest, since it
> requires production-resolution running.  This is a workstation job,
> not a unit test.

Session D **closes that gap** by adding the workstation parity
deliverable: figures + JSON + a slow-marked pytest invariant.  It
extends the Phase C parity pattern (
``[[project-redteam-observable-tests-done]]``,
`tests/test_redteam_observables_parity.py`) from Aligned-only to
``Aligned + Isotropic + Fisher(κ=5)``.

## Hard constraints (carried forward, will be honoured)

* **No fake physics** ([[feedback-no-fake-physics-stand-ins]]) — the
  MC and engine both run the real vector BR + LZ + per-orientation
  retention; no smoothing operators dressed as physics.
* **No frozen numerical references** ([[feedback-no-frozen-test-references]])
  — every test assertion is an engine-vs-MC or MC-vs-MC relationship.
  No pinned ``.npz`` baselines.
* **No hardcoded physical scales** ([[feedback-no-hardcoded-physical-scales]])
  — bandwidth-style choices stay data-driven; bulk mask is engine-
  derived at fixed ``tail_frac=0.01`` (the Phase C convention).
* **MC depletion tracks engine f_natal_depleted**
  ([[feedback-mc-depletion-tracks-engine]]) — `run_mc` reads it from
  the implicit solver's converged result, never hardcoded.
* **No fudge factors** ([[feedback-no-fudge-factors]]) — `tail_frac`
  set once at 0.01, never tuned per orientation.
* **Implicit solver is production** ([[feedback-perturbative-unreliable-implicit-production]])
  — every parity check routes through `solve_implicit_fixed_point`.
* **Dispatch call-graph audit** ([[feedback-dispatch-callgraph-audit]])
  — when `comparison_helpers.run_mc` and `build_response_for_theta`
  gain `orientation`-family kwargs, all existing callers continue to
  see Aligned bit-for-bit (kwargs default to `None` / `0` / "heaviside" /
  `1.0`).

## Direction-vs-magnitude lesson from Session C

Session C's round-3 round-4 reformulations taught us that orientation-
broadened observables can have configuration-dependent inequality
directions vs Aligned:

* `total_rate(Iso+sigmoid)` can be *higher* than `total_rate(Aligned+
  Heaviside)` at intermediate `v_esc` (some Aligned cells eject
  cleanly while Iso retains a fraction).
* Peak-to-mean concentration of the cascade can be *higher* under Iso
  at narrow natal Gaussian.

Lesson baked into Session D: every parity assertion will be a
**MAGNITUDE** invariant (rel-L1, JS, total-rate ratio in a window),
**NOT** a direction inequality.  Cross-orientation comparisons go in
the figure/metrics report for the human reader, not the test gate.

## Deliverables

### 1. `demo/PLAN_misalignment_observable_parity.md` — this file.

### 2. `demo/misalignment_observable_parity.py` — new layered demo.

CLI:

```
python demo/misalignment_observable_parity.py \
    [--kappa 5.0] \
    [--n-mc 50000] \
    [--resolution 64x20] \
    [--epsilon 1.0] \
    [--seed 20260527] \
    [--out-hash <hex>] \
    [--tail-frac 0.01]
```

Behaviour:

1. For each orientation in `[Aligned(), Isotropic.default(),
   Fisher.default(kappa)]`:

   a. Run the production implicit solver at the chosen `(N_m, N_χ)`
      (only one resolution per invocation — no row-per-resolution
      sweep here; resolutions add cost ×3 for the orientation grid).

   b. Run the particle MC at the engine's converged
      `f_natal_depleted` with matching `orientation`.

   c. Compute these observables on matching axes for engine + MC:
      * `dN/dm_1` (1-D) — engine `marginal_1d(AXIS_M1)`, MC
        `dN_dm1_mc`.
      * `dN/d(m_1 χ_eff)` (2-D, ordered) — engine
        `dN_dm1_dchi_eff`, MC `dN_dm1_dchi_eff_mc`.
      * `dN/d(m_1 q)` (2-D, ordered, q ∈ [0.05, 1.0]) — engine
        `dN_dm1_dq`, MC `dN_dm1_dq_mc`.
      * `dN/dχ_eff` (1-D) — collapsed from the 2-D `(m_1, χ_eff)`
        engine and MC density via mass-marginal sum with engine
        trapezoid weights.

      The 1-D χ_eff axis is the SESSION-D-CRITICAL observable: it
      is the most direct view of the cascade-compression-breaks-
      under-iso physical claim from `[[project-misalignment-mode-planned]]`.

   d. Compute per-observable metrics via
      `red_team/observable_metrics.py`:
      * `js_on_grid_bulk` for the 1-D observables (engine-vs-MC and
        MC-seed-1 vs MC-seed-2 for the noise gauge).
      * `relative_L1_bulk` for the 2-D observables.
      * `total_rate_ratio` for all observables.
      * `bulk_mask_coverage_fraction` as a sanity check.
      * `peak_location` (1-D) / `peak_location_2d` (2-D) for
        cross-orientation reporting.

2. Write outputs:

   * `demo/figures_mobs_<hash>/comparison_<obs>.png` — one figure
     per observable, three orientations side-by-side (engine solid,
     MC overlay, axes shared so the orientation effect is visually
     direct).
   * `demo/figures_mobs_<hash>/all_observables.png` — single
     summary figure (rows = orientations, columns = observables).
   * `demo/artifacts_mobs_<hash>/metrics.json` — unified summary:

         {
           "config": {"theta_star", "kappa", "n_mc", "resolution", "tail_frac", ...},
           "orientations": {
             "aligned":      {"engine_total_rate", "mc_n_accepted",
                              "f_natal_depleted",
                              "metrics": {<obs>: {"js_imp_mc", "js_noise",
                                                  "rel_l1_imp_mc", "rel_l1_noise",
                                                  "total_rate_ratio",
                                                  "bulk_coverage",
                                                  "peak_engine", "peak_mc"}}, ... },
             "isotropic":    {...},
             "fisher_k5.0":  {...}
           }
         }

   * `demo/artifacts_mobs_<hash>/orientations_<o_tag>.npz` — per-
     orientation raw arrays (engine + MC) for any axis, so the
     figures can be re-rendered post-hoc.

3. Per-orientation output naming follows the demo convention
   ([[feedback-demo-output-conventions]]) — one `<hash>_<o_tag>`
   suffix so parallel runs don't clobber each other.

### 3. `tests/test_misalignment_observable_parity.py` — new slow test.

Mirrors `tests/test_redteam_observables_parity.py` but parametrised
over `[Aligned(), Isotropic.default(), Fisher.default(kappa=5.0)]`.

Fixture: one engine solve + two MC seeds **per orientation**, all at
the same resolution (`64x20` baseline — same as Phase C).  Total
runtime budget ~ 3× Phase C ≈ a few minutes on workstation.

Acceptance invariants (per orientation):

* **Per-observable bulk metric ≤ `JS_MULT × noise`** where
  `JS_MULT = 50` matches the Phase C convention.  Noise is `MC1 vs
  MC2` on the same observable + same orientation.  This is the right
  encoding because:
  1. It doesn't pin an absolute number ([[feedback-no-frozen-test-references]]).
  2. It scales naturally with the per-orientation MC variance: Iso /
     Fisher orient-quadrature ADDS variance compared to Aligned, so
     the same multiplier on the noise baseline is the right scaled
     threshold.
  3. It gates engine bias, not MC noise.

* **Total-rate ratio in [0.85, 1.15]** for 1-D, [0.75, 1.25] for the
  q-restricted 2-D (matches Phase C).

* **Bulk-mask coverage ≥ 85%** as a sanity check that `tail_frac=0.01`
  is not too aggressive at this orientation.

* **Aligned baseline already passes Phase C** — the test asserts the
  same Phase C numbers re-derive when `Aligned()` is the dispatch.
  This is a redundant guard, but cheap, and it locks `Aligned()`
  through the parametrised harness as bit-for-bit equivalent to
  `None` / `Aligned`-by-default — keeps `[[feedback-dispatch-callgraph-audit]]`
  honest.

* **NO direction-inequality assertions** between orientations.  The
  per-orientation rel-L1 / JS numbers are *recorded* by the demo for
  the human; the test only gates each orientation's parity against
  its own MC noise.  Lesson from
  `[[project-misalignment-session-c-done]]` 743f830 / addc023.

Tolerance discovery pattern: the test reads `JS_MULT` and total-rate
window from constants at the top of the file (same values as Phase C
at the start, lifted if the workstation results show iso / fisher
need looser).  ANY relaxation must be justified in a docstring
paragraph naming the source of the residual disagreement (e.g.
"orientation quadrature adds 1.4× MC noise from the additional
random axes per event") — same discipline as
`[[project-evaluation-step3-done]]`'s 5% → 10% relaxation.

### 4. `red_team/comparison_helpers.py` — backward-compatible kwarg
extension.

Add `orientation`, `orientation_key_seed`, `retention_kind`,
`w_over_v_esc` to:

* `build_response_for_theta` — forwards to
  `build_gaussian_coagulation` (already takes them per Session C).
* `run_mc` — forwards to `MCConfig` (already has them per Session C
  a76fa8f).

Defaults: `None` / `0` / `"heaviside"` / `1.0` — Aligned bit-for-bit.
All existing callers (Phase C test + `test_redteam_Nx_parity.py`)
get Aligned by omission; behaviour preserved.  This is the
[[feedback-dispatch-callgraph-audit]] response — extending the
helper layer so all comparison-test callers can opt into orientation.

## Out of scope for Session D

* **Multi-resolution sweep under orientation** — the existing forward-
  modeling demo already does it for one orientation.  Three orientations
  × multiple resolutions × ~225 MB per implicit solve at production
  is too much for a single demo run.  If needed later, the layered
  script accepts a single resolution and the user batches.
* **Per-generation 3G invariants** — Session C deferred item still
  open.  Adds k_max=4 and ~2× iso time per invariant; outside Session
  D scope, was its own AskUserQuestion option.
* **κ sweep** — user picked a single representative κ=5.  Adding 3-4
  κ values would have been the other AskUserQuestion branch; this
  PLAN's layered design makes it a trivial future extension (loop
  over a κ list, same metrics).
* **Healy-Lousto 2018 v_kick polynomial** — its own AskUserQuestion
  option; not Session D.
* **MixtureOrientation** — same.
* **Loss-side memory at production resolution** — already DONE in
  Session C 19cc0b6 (lax.scan refactor brought peak from ~75 GB to
  ~225 MB at `(96, 32) × n_quad=1024`).  Session D runs at `64x20`
  which is well within budget.

## Sequence

Order of work (revised order based on tasks created):

1. Write this PLAN — done.
2. Extend `red_team/comparison_helpers.py` (small, backward-compatible).
3. Write `demo/misalignment_observable_parity.py` — the layered demo.
4. Write `tests/test_misalignment_observable_parity.py` — the slow
   pytest gating the workstation parity.
5. Sandbox-verify what is sandbox-runnable (`py_compile`, AST checks,
   numpy-only paths).  Full JAX-dependent run is workstation-only —
   same pattern as every prior misalignment session.
6. Memory + handoff.

## Workstation validation expected after landing

The user runs:

```
# Headline demo
python demo/misalignment_observable_parity.py --kappa 5.0 \
    --n-mc 50000 --resolution 64x20

# Slow pytest
pytest tests/test_misalignment_observable_parity.py -m slow -v
```

Expected:

* Three orientations all pass the per-orientation JS / rel-L1 / total-
  rate gates.
* Figures show the cascade-compression-breaks-under-iso physical
  claim from `[[project-misalignment-mode-planned]]`: the iso χ_eff
  marginal is visibly broader than Aligned, with Fisher(κ=5) sitting
  between the two limits.
* JSON metrics file is the machine-readable record for future
  regression diffing.

## Related

* `[[project-misalignment-session-c-done]]` — closes the open deferred
  item that motivated Session D.
* `[[project-misalignment-session-b-done]]` — marginal-layer MC↔engine
  parity (Session D's analogue at the observable layer).
* `[[project-redteam-observable-tests-done]]` — the Aligned-only
  Phase C parity battery that Session D extends.
* `[[project-evaluation-step3-done]]` — companion validation backstop
  (MC k-NN evaluator); orthogonal axis.
* `[[project-aligned-chi-undersampled]]` — the cascade pathology
  whose misalignment-mode fix Session D demonstrates at the observable
  level.
* `demo/PLAN_misalignment_mode.md` — original PLAN (Sessions A→C).
* `demo/PLAN_redteam_observable_validation.md` — the Phase C plan
  Session D parallels structurally.
* `[[feedback-no-frozen-test-references]]`, `[[feedback-no-fudge-factors]]`,
  `[[feedback-no-fake-physics-stand-ins]]`, `[[feedback-mc-depletion-tracks-engine]]`,
  `[[feedback-perturbative-unreliable-implicit-production]]`,
  `[[feedback-dispatch-callgraph-audit]]` — invariant constraints
  carried forward.
