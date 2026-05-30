# B.2 soft K-gate — implementation memo (2026-05-29)

Implements the deterministic soft K-gate scoped in
`PLAN_jax_mc_time_decoupling_diagnosis.md` § 5.B.2 + § 8 and the
`project-jax-mc-b2-soft-gate-followup` breadcrumb.  B.2 is the principled
successor to round-3's B.1 Bernoulli K-gate (`particle_mc_jax`); it
removes B.1's two costs:

1. the `k_ratio` magic factor (a `K_max/⟨K⟩` estimate used to size
   `n_steps` so expected accepts hit the depletion target), and
2. the straight-through estimator bias on `∂/∂kernel_{a,b}`.

**Status: sandbox-validated, workstation-pending.**  This Cowork VM has
no JAX runtime, so the JAX path is verified only through a faithful numpy
mirror (below).  First workstation run must execute the pytest tier and
calibrate the two AD tolerances.

---

## Mechanism (occupancy flow + slot pre-allocation)

Pre-allocate `S = n_init + n_steps` slots: `[0:n_init]` natal parents
(occupancy `o = 1`), `[n_init:S]` remnant tail (`o = 0`, born as the
cascade runs).  Each `lax.scan` step `k`:

* draw a pair `(i, j)` with probability `∝ o` (the soft analog of numpy's
  uniform-over-active draw),
* `w_event = K_ij / K_max`,
* `δ = w_event · min(o_i, o_j)` — the consumed amount, smooth in the
  kernel params, bounded by the smaller partner so neither slot goes
  negative,
* `o_i −= δ;  o_j −= δ`,
* remnant born into tail slot `n_init + k` with occupancy `δ` (if
  retained).  **The scan counter is the static write index** — no dynamic
  write-pointer, which keeps the scan body simple and `jit`-clean.

Net occupancy removed per step: `δ` if retained (2 consumed − 1 born),
`2δ` if not.  A cumulative **depletion budget** clips the last
contributing step so total net removal lands exactly on
`depletion_fraction · n_init`; once exhausted, `δ = 0` (cheap no-op
steps).  This is what eliminates `k_ratio`: `n_steps` only has to be a
generous UPPER BOUND (sized by `n_steps_mult`, default 3.0), never a
tuned count.

Aligned-only first pass, exactly like B.1: physics fits are called in
pair-selection order (matching numpy `particle_mc` / B.1 bit-for-bit);
non-Aligned raises.  Retention stays Heaviside under Aligned, so `∂/∂v_esc`
is still zero here — B.2 fixes the KERNEL-gradient quality and the
depletion-control knob, NOT the v_esc gradient (that recovers under
misalignment + soft-sigmoid retention, the J2b follow-up).

---

## Numpy-sandbox verification (runnable, this session)

`demo/b2_soft_gate_derivation.py` — compares the soft gate against a
discrete K-rejection reference (numpy `particle_mc` semantics) under
IDENTICAL simplified physics, isolating the GATE mechanism:

* **Q1 — k_ratio elimination.**  Depletion = 0.1000 (std 0.0000) for
  every `n_steps ≥ ~1.5·D·n_init` (= `K_max/⟨K⟩·D·n_init`; the natal pool
  here has `K_max/⟨K⟩ = 1.21`).  Beyond that it is flat — the budget pins
  it; `n_steps` is a free upper bound.
* **Q2 — shape parity.**  Depletion and `W_total` match exactly
  (`W_total = (1−D)·n_init = 2700`); `KS(m) = 0.009/0.013`,
  `KS(χ) = 0.009/0.012`, both well under `KS_crit(0.05) = 0.037`.
* **Q3 — smoothness.**  Holding the pair sequence fixed (the AD-faithful
  condition — JAX `random.choice` indices are constants w.r.t. θ), the
  soft observable is C0-smooth in `kernel_b` (FD −0.41, 2nd-diff 3e-6);
  the discrete mirror is piecewise-constant with 20 accept-flip jumps over
  `b ∈ [0.8, 1.2]` and zero gradient between them — exactly the
  discontinuity B.1's straight-through estimator papers over.

`demo/b2_jax_mirror_check.py` — ports the EXACT JAX `step_fn` indexing
(one tail slot `n_init+k` per step, budget clip, occupancy-weighted draw)
to numpy and confirms:

* depletion pinned to 0.10000 (std ~1e-17), `W_total = 2700`,
* `KS(m) = 0.012`, `KS(χ) = 0.015` vs the discrete reference,
* **layout invariance**: `KS = 0.0000` vs the packed-write-pointer
  prototype — the one-tail-slot-per-step layout gives bit-identical
  observables (dead occupancy-0 tail slots contribute nothing).

---

## Code surface

* `red_team/backup_particle_mc.py` — **NEW** `particle_mc_jax_soft(cfg,
  key, *, n_steps=None, n_steps_mult=3.0, occ_tol=1e-9)`.  Additive;
  numpy `particle_mc` and B.1 `particle_mc_jax` untouched.  Returns the
  B.1 schema plus a continuous `occupancy` field.
* `src/.../response/mc_jax.py` — `build_mc_jax_response` gains
  `gate={"bernoulli","soft"}` (default `"bernoulli"`, unchanged) and
  `n_steps_mult`.  Under `gate="soft"` the evaluator weights are
  `w_active · occupancy` (continuous) — the smooth replacement for B.1's
  binary `active_mask`, so the KDE carries kernel/natal AD tangents
  through the surviving-pool *fraction*.  `MCEvaluatorJax` needs NO change
  (already consumes arbitrary positive weights; dead slots have weight 0
  and drop out of `n_eff`/bandwidth).
* `src/.../response/coagulation.py` — `build_gaussian_coagulation` gains
  `mc_jax_gate` + `mc_jax_n_steps_mult`, threaded to the factory.
  Call-graph audited per `feedback-dispatch-callgraph-audit`: the only
  caller of `build_mc_jax_response` is this site (keyword args); the
  test mirror in `test_response_mc_jax.py` calls `particle_mc_jax`
  directly and is unaffected.
* `tests/test_mc_particle_mc_jax_soft.py` — **NEW** 8-invariant tier
  (S1–S8).  S2/S3/S4 are exact gates (verified bit-stably in the mirror);
  S6/S7 carry conservative, flagged AD tolerances to calibrate on the
  first workstation run.

---

## Workstation runbook (next session)

```bash
# 1. Sandbox tier — B.2 + all existing MC-JAX backstops.
pytest tests/test_mc_particle_mc_jax_soft.py -v          # NEW (B.2)
pytest tests/test_mc_particle_mc_jax.py \
       tests/test_evaluation_mc_jax.py \
       tests/test_response_mc_jax.py \
       tests/test_mc_sample_natals_jax.py -v             # B.1 regress

# 2. Calibrate S6/S7 tolerances if they fail only on threshold (the
#    forward logic is mirror-validated; selection-shift FD noise is the
#    only unknown).  Tighten/loosen rel in the two AD tests as warranted.

# 3. Compare B.2 vs B.1 on the M4 working point.
pytest -m slow -k integration_stage1_recovery_mc_jax -v
#    Build the M4 response with gate="soft" (mc_jax_gate="soft") and
#    re-run.  NOTE: B.2 is orthogonal to the total_rate metric-bias
#    xfail (project-mc-jax-total-rate-bias) — that is a KDE-oversmoothing
#    + quadrature-span issue in MCEvaluatorJax.total_rate, NOT a cascade-
#    dynamics issue, so B.2 will NOT by itself un-xfail
#    test_expected_count_at_truth_matches_n_obs.  What B.2 SHOULD improve
#    is gradient quality (no straight-through bias) and remove the
#    k_ratio knob.

# 4. Optional: re-run the numpy sandboxes for the record.
python demo/b2_soft_gate_derivation.py
python demo/b2_jax_mirror_check.py
```

Decision after the workstation run: if B.2's gradients + depletion
control are clean and B.1 offers nothing extra, make `gate="soft"` the
default and demote B.1 to a `gate="bernoulli"` opt-in.  Until then B.2 is
additive and opt-in.

---

## Related

* `PLAN_jax_mc_time_decoupling_diagnosis.md` § 5.B.2 + § 8 — design spec.
* `project-jax-mc-workstation-round3` — B.1 ship; the parent.
* `project-mc-jax-total-rate-bias` — orthogonal metric-bias thread; B.2
  does not address it.
* `feedback-no-fudge-factors` — `n_steps_mult` is a loose upper bound, not
  a tuning knob; depletion is pinned by the budget, `w_event = K_ij/K_max`
  exactly.
* `feedback-no-fake-physics-stand-ins` — fractional occupancy is a clean
  importance-weighted particle-MC reinterpretation (each fractional slot
  is the IS weight of a fully-realised particle's contribution), not a
  fake-physics smoothing.  Physics fits are the shared scalar BR/LZ/
  Heaviside triple, identical to numpy / B.1.
* `feedback-red-team-validation` — numpy `particle_mc` untouched; B.2
  KS-green against it in both sandboxes.
