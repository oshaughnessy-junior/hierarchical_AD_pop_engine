# PLAN — Hybrid analytic-1G + KDE-(2G+) density (fixes two-pulse width floor)

**Why:** the `mc_jax` likelihood reads the population density through the J3
Gaussian KDE, whose global Silverman bandwidth oversmooths the natal pulses
(2–3× too wide) and is bimodal-inflated + `n_obs`-independent → widths
unrecoverable regardless of data. Full diagnosis: `demo/DIAGNOSIS_two_pulse_kde_floor.md`.
User picked this fix 2026-05-30. Honours `[[feedback-no-fake-physics-stand-ins]]`
(evaluate the parametric component parametrically) + `[[feedback-no-hardcoded-physical-scales]]`.

## The idea (convention-preserving)

Split `N(m,χ) = n₁(m,χ) + n_{≥2}(m,χ)` by generation:

* **n₁ ANALYTIC:** `N₁ = W_1G · f_natal_mass(m) · f_natal_spin(χ)`, where
  `W_1G = Σ wᵢ·(genᵢ==1)` (surviving-1G total weight, a differentiable scalar)
  and `f_natal_*` are the **normalised parametric** natal distributions
  (`log_natal_mass_two_gaussian`/`_gaussian` × natal-spin truncnorm). Exact,
  differentiable in μ₁,σ₁,μ₂,σ₂,λ,spin_loc,spin_scale ⇒ **widths/locations/λ
  recovered exactly.** `∫∫ N₁ = W_1G` (f's are normalised).
* **n_{≥2} KDE:** the existing Gaussian KDE but built from **gen≥2 particles
  only** — zero the 1G weights out of the KDE (`w_kde = w·(gen≥2)`), recompute
  the Silverman bandwidth from `w_kde` (so the cascade remnants get their own,
  un-bimodal-inflated bandwidth — an independent win). `∫∫ N₂ = W_2G`.
* Total `∫∫ N = W_1G + W_2G = W_total` ✓ — density convention preserved, so
  `total_rate` / `rate_density_4d` / gwkokab wiring are untouched downstream.

Approximation (documented, NOT fake physics): treats surviving-1G **shape** as
∝ the natal input shape (ignores mass-dependent 1G depletion). Excellent for
narrow pulses + weak depletion (the methods-paper regime). The 2G+ stays fully
nonparametric.

## Where to change (all in `src/.../evaluation/mc_jax.py` + 1 wiring line)

`MCEvaluatorJax` — additive, default-off so non-hybrid is bit-for-bit:

1. New fields (all optional/None when off; `hybrid_1g: bool` static):
   `gen_active: Array`, `W_1g`, `W_kde`, `h_kde_log_m`, `h_kde_chi`,
   `natal_mass_kind` (static str), `natal_mass_params` (dynamic — splat dict;
   eqx needs it flattened or stored as a small pytree), `natal_spin_params`.
2. `from_run(..., gen_active=None, natal_mass_kind=None, natal_mass_params=None,
   natal_spin_params=None, hybrid_1g=False)`: when hybrid, compute
   `w_kde = weights * (gen_active>=2)`, `W_1g = Σ weights*(gen==1)`,
   `W_kde = Σ w_kde`, and the Silverman bandwidths from `w_kde` (reuse
   `_weighted_mean_var` + `_silverman_bandwidth_1d`, just on `w_kde`).
3. `density(m,χ)`: if `hybrid_1g`, `return _kde_density(...) + _analytic_1g_density(...)`
   (with the existing valid-mask). Else the current path verbatim.
   * `_kde_density`: clone of current `density`/`_f_hat_log_m_chi` but using
     `w_kde`, `W_kde`, `h_kde_*` (so 1G particles contribute 0 to the KDE).
   * `_analytic_1g_density(m,χ)`: `W_1g * exp(log_f_natal_mass(m;params)) *
     exp(log_natal_spin(spin_dist,χ))`. Import `log_natal_mass_two_gaussian` /
     `log_natal_mass_gaussian` + the natal-spin logpdf. Dispatch on
     `natal_mass_kind`. Out-of-`[m_min,m_max]`/`[0,1]` → 0 (the truncnorms
     already give that).
4. Wiring: `build_mc_jax_response` (`response/mc_jax.py:403`) passes
   `gen_active=out["gen_active"]`, the natal params from `response`, and a new
   `hybrid_1g` knob plumbed from `build_*_coagulation` (e.g.
   `mc_jax_hybrid_1g: bool=False` → through to `build_mc_jax_response`). Turn
   ON by default for `build_two_pulse_coagulation`; keep OFF elsewhere until
   validated.

## Validation (sandbox, JAX — `source demo/bootstrap_jax_sandbox.sh`)

* **H1 width recovery:** build two-pulse pool, hybrid `density` dN/dm vs truth
  → pulse FWHM within ~10% (vs 2–3× for KDE). Compare to
  `two_pulse_kde_bandwidth_diagnosis.png`.
* **H2 convention:** `∫∫ N dm dχ ≈ W_total` (hybrid) to ~1%.
* **H3 AD:** `d(dN/dm at peak)/dσ₁` and `/dλ` AD-vs-FD agree (the recovery
  gradient SVI needs).
* **H4 regression:** `hybrid_1g=False` → `density` bit-for-bit unchanged
  (existing `tests/test_evaluation_mc_jax.py` green).
* **H5 total_rate:** hybrid `total_rate` finite + sensible (may also help the
  `[[project-mc-jax-total-rate-bias]]` natal-peak-clipping, since 1G is now
  analytic — check the M4 working point).

## Then

Re-run `demo/fit_two_pulse_gwkokab.py` (workstation) with hybrid on; expect
width recovery + the label-switch test to pass (sharp `(μ₁,μ₂)` basin returns).
Add `tests/test_hybrid_1g_density.py`. Breadcrumb back into INSTRUCTIONS §5.0.

## Pointers / open

* The `gen_active` field already exists on the soft-cascade output
  (`out["gen_active"]`, int32) — no cascade change needed.
* Watch the eqx pytree for `natal_mass_params` (dict of arrays): store as
  flat fields or a frozen mapping so `tree_at`/grad work; mirror how
  `kernel_a`/`kernel_b` are stored as flat scalars rather than a dict.
* `[[project-two-pulse-natal-done]]` (the stack), `[[project-mc-jax-total-rate-bias]]`
  (the sibling KDE thread this may also relieve), `[[feedback-mise-kernel-not-the-lever]]`.
