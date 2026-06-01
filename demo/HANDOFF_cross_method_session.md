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

## Scientific state (two-pulse, `--gen mc --solver implicit`, n_obs=60)

* **λ recovers**: MAP 0.414 (+0.3σ). Headline result. ✓
* **mu_natal, mu_natal_2, spins, log_rate-shape**: recover well. ✓
* **`sigma_natal`**: peak 1.33, MAP 1.30 — a **structural** cross-method bias,
  2.8σ at n_obs=60 and ~6.9σ at n_obs=400 (it SHARPENS with data → real
  forward-model floor, the 0.086 2D JS, ~30%). The thing to reduce next.
* **`sigma_natal_2`**: +1.7–1.9σ (looser — fewer pulse-2 events).
* **`kernel_a`↔`log_rate`**: 1D-peak at truth, joint MAP slides together
  (kernel_a −10σ, log_rate +12σ in CONDITIONAL σ). A **model degeneracy** from
  `float_kernel_a=True`, NOT a bias — the `--hessian` marginal σ_marg/σ₁d
  inflation + flat eigvec show this. `kernel_b` follows the same valley.

## H1/H2/H3 verdicts

* **H1** (shape agreement): 1D JS ~0.05, 2D N(m,χ) ~0.086 — elevated vs
  single-pulse (~0.02); richer two-pulse structure, bigger forward-model
  residual. λ robust to it; sigma_natal not.
* **H2** (ε↔depletion): matched to |Δf|=0.0014. ✓
* **H3** (kernel parity): kernel `(m1+m2)^a·q^b` BYTE-IDENTICAL across injection
  pairing, MC cascade (`_kernel_K_jax`, ∝ N·N·K, a=0 ⇒ exact K_max), implicit
  gain/loss + recovery. No kernel bug. ✓

## Open threads (next session)

1. **Reduce the `sigma_natal` floor.** It's the MC-vs-Picard forward-model
   difference. Localize with `red_team/compare_all.py` per-generation marginals
   (gen-1 should agree; if gen-2+ diverges, that's the cascade-placement
   difference driving σ). n_seed↑ shrinks JS but NOT the bias (structural).
2. **Lift the `kernel_a`↔`log_rate` degeneracy** for a clean corner: either
   don't float kernel_a, or reparametrise to (overall-rate, mass-slope)
   orthogonal coords / log-kernel_a. Then `--hessian` is cleanly PD and the
   Fisher is publication-grade.
3. **n_obs scaling study** (user running): confirm sigma_natal bias ∝√n_obs
   (structural) vs the constrained params staying ~truth.
4. **Two-pulse SVI/NUTS** corner once the degeneracy is lifted.

## Run

```
python demo/fit_two_pulse_gwkokab.py --gen mc --solver implicit --hessian
python demo/fit_two_pulse_gwkokab.py --gen mc --solver implicit --n-obs 400
```

Caveat unchanged: no JAX in the Cowork VM (rootfs full); all numeric runs are
workstation; commits Cowork-local, push from the workstation.
Memory: `project_implicit_pivot_bridge_wired`.
