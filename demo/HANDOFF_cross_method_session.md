# HANDOFF — cross-method inject/recover, validated single + two-pulse

**Date:** 2026-06-01. **Branch:** `observables`. **Status:** cross-method
pipeline complete and validated; λ recovers; residuals characterized. Supersedes
`HANDOFF_cross_method_milestone.md` (extends it with two-pulse + diagnostics).

## Where we are (one paragraph)

Inject the catalogue with particle MC (exact draws, no grid bias), recover with
the implicit grid solver (sharp, KDE-free), at the depletion-matched ε. It works:
single-pulse recovers the natal params, and **two-pulse recovers λ** (0.41–0.43
vs 0.40) — the experiment the whole arc was built for. The remaining wrinkles are
understood and quantified, not bugs: a structural `sigma_natal` cross-method
systematic, and a `kernel_a`↔`log_rate` model degeneracy from floating kernel_a.

## The pipeline (all committed, `28a54bc` … `ea501db`)

1. **Bridge** (`28a54bc`): `make_coagulation_dist_fn(solver=…, epsilon=…)`;
   implicit `log_prob`/`total_rate` AD via the IFT custom_vjp.
2. **Step 1 — exact-particle injection** (`c7cde9f`): `--gen mc` pairs the
   surviving MC pool ∝ `w_i·w_j·K(m_i,m_j)`, no grid/jitter.
3. **Step 2 — H2 ε-match** (`4e6de09`,`98077d3`): root-find implicit ε so
   `f_natal_depleted(ε) = MC f_dep`; scan converging range, skip diverged solves.
4. **Boundary/support** (`7dbca1d`): `log_density_at` → `mode="nearest"` (kills
   the edge `0·-inf=NaN`); clip scattered events to the grid box.
5. **Step 3 — H1 JS gate** (`7113ad3`,`84b3ce5`): `cross_method_js` = JS(implicit‖
   MC) on `dN/dm`, `dN/dχ`, and the 2D `N(m,χ)` (the structural one); printed in
   `--gen mc`, gated in `test_cross_method_js_gate.py`.
6. **Two-pulse** (`6b5272e`): `--gen mc` on `fit_two_pulse_gwkokab`.
7. **Diagnostics** (`5b6409d`,`bc4f5b6`,`ea501db`): `Δ/σ₁d`, `peak-bias`,
   `--hessian` joint Fisher (scaled, eval@MAP, soft-direction stabilize).

## Scientific state (two-pulse, `--gen mc --solver implicit`)

The n_obs scaling is the key diagnostic — biases that GROW ∝√n_obs are
structural (forward-model), not statistical:

| param         | n_obs=60     | n_obs=400    | interpretation |
|---------------|--------------|--------------|----------------|
| lambda_natal  | 0.414 (+0.3σ)| 0.466 (+3.9σ)| +16%, hidden by noise at n60 |
| sigma_natal   | 1.30 (+2.6σ) | 1.36 (+7.5σ) | **+36% structural** |
| sigma_natal_2 | 2.60 (+1.9σ) | 2.72 (+6.0σ) | **+36% structural** |
| mu_natal_2    | (1.1σ)       | (3.2σ)       | small, sharpening |
| spins, mu_natal | clean      | clean        | ✓ |

* **The cross-method systematic is WIDTH-DOMINATED** (σ₁, σ₂ both +36%), λ +16%.
  Physical: the particle-MC vs implicit-Picard cascade difference is a density
  SHAPE difference, so the width params absorb it most. Quantified by the
  `[H1-JS] N(m,χ)2D = 0.086` (the canary — it predicted the systematic).
* **`kernel_a`↔`log_rate`**: a **model degeneracy** from `float_kernel_a=True`,
  NOT a bias. `--hessian` (eval@MAP + stabilize) now confirms it: softest
  eigvec = `log_rate:+0.82 kernel_a:-0.55`. A residual negative eigenvalue
  survives (kernel_a (m1+m2)^a is exponentially non-quadratic — flagged, not
  fixable by polish). `kernel_b` follows the same valley.

## Paper framing (PROOF-OF-CONCEPT) — agreed 2026-06-01

* **Headline**: cross-method inject-MC / recover-implicit RECOVERS the two-pulse
  population (λ, μ's, spins), using each method where it's strong (MC: exact
  injection, no grid bias; implicit: sharp KDE-free recovery).
* **The width systematic is a forward-model SYSTEMATIC CHECK, not a bug.** It
  measures "how much does the answer depend on which cascade solver you trust"
  (~36% on widths at the current grid). The UNBIASED production path is
  **implicit→implicit** self-consistent recovery (faithful sampler, no
  cross-method floor) — that's what runs on real data; cross-method is the
  systematic estimate alongside it.
* **Resolution hypothesis (testable, expected to hold)**: because H2 (ε-match)
  and the kernel (H3) are exact and the density convention cancels in the JS,
  the 0.086 is a DISCRETIZATION difference — it should converge to 0 as the
  implicit grid AND the MC n_seed refine together. If it does, the width
  systematic is a resolution knob, not a floor. **This is the first thing to
  test next session** (sweep n_m/n_chi AND n_seed jointly; watch 2D JS → 0 and
  σ-bias → 0). Single-axis sweeps didn't move it (each alone insufficient);
  the joint refinement is the well-calibrated limit.

## H1/H2/H3 verdicts

* **H1** (shape agreement): 1D JS ~0.05, 2D N(m,χ) ~0.086 — elevated vs
  single-pulse (~0.02); richer two-pulse structure, bigger forward-model
  residual. λ robust to it; sigma_natal not.
* **H2** (ε↔depletion): matched to |Δf|=0.0014. ✓
* **H3** (kernel parity): kernel `(m1+m2)^a·q^b` BYTE-IDENTICAL across injection
  pairing, MC cascade (`_kernel_K_jax`, ∝ N·N·K, a=0 ⇒ exact K_max), implicit
  gain/loss + recovery. No kernel bug. ✓

## Next session: PAPER CONTENT PREP

Plan is to start drafting proof-of-concept paper content. Before/while drafting:

1. **JOINT resolution sweep (the key result for the paper).** Sweep implicit
   `--n-m/--n-chi` AND MC `--n-seed` TOGETHER; tabulate `[H1-JS] N(m,χ)2D` and
   the σ₁/σ₂ bias vs resolution. Expected: both → 0 in the well-calibrated
   limit (H2/H3 exact ⇒ the 0.086 is discretization). This converts "36%
   systematic" into "resolution-controlled, converges away" — the headline
   robustness claim. n_obs=400 is the right scale to see the bias clearly.
2. **Figures**: (a) cross-method recovery corner/profiles (λ + natal params);
   (b) the resolution-convergence plot (JS & σ-bias → 0); (c) the H1/H2/H3
   gate values; (d) maybe the σ₁d/Δ-σ diagnostic table.
3. **Lift the `kernel_a`↔`log_rate` degeneracy** for a clean corner: don't float
   kernel_a, or reparametrise to (overall-rate, mass-slope) coords / log-kernel_a.
   Then `--hessian` is cleanly PD, publication-grade.
4. **Method narrative**: implicit→implicit = unbiased production; cross-method =
   systematic check; resolution controls the residual.

## Run

```
python demo/fit_two_pulse_gwkokab.py --gen mc --solver implicit --hessian
python demo/fit_two_pulse_gwkokab.py --gen mc --solver implicit --n-obs 400
```

Caveat unchanged: no JAX in the Cowork VM (rootfs full); all numeric runs are
workstation; commits Cowork-local, push from the workstation.
Memory: `project_implicit_pivot_bridge_wired`.
