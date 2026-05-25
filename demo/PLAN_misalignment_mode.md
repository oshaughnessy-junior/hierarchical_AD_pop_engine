# PLAN — prescribed-tilt misalignment mode (physical promotion)

Written 2026-05-25 in response to the aligned-only χ cascade pathology
(`[[project-aligned-chi-undersampled]]`).  Implements a prescribed,
not self-consistently evolved, spin-orientation model that promotes
the engine's scalar progenitor χ to vector spins at merger time, runs
the **full vector Barausse-Rezzolla / Lousto-Zlochower** remnant
physics, and integrates over a prescribed orientation distribution
`P(θ̂)` to recover scalar χ_rem.  The χ_rem distribution at the output
is the *consequence* of real physics + the orientation model — **not**
a chosen broadening kernel.

**This PLAN supersedes an earlier sketch that proposed an ad-hoc
χ-axis broadening operator.  That was fake physics and is rejected.
See `[[feedback-no-fake-physics-stand-ins]]`.**

## Physical framing

For each merger event with input scalars `(m1, m2, |χ1|, |χ2|)`:

1. **Promote** `|χ1| → χ1_vec`, `|χ2| → χ2_vec` by drawing tilt angles
   `(θ_i, φ_i)` from a prescribed orientation distribution `P(θ̂)`
   defined relative to the orbital angular momentum `L̂`.
2. **Compute** the remnant vector spin `χ_rem_vec(m1, m2, χ1_vec,
   χ2_vec)` via the *full* Barausse-Rezzolla 2009 formula (not the
   aligned-only simplification currently in `remnants/spin.py`).
3. **Compute** the recoil `v_kick(m1, m2, χ1_vec, χ2_vec)` via the
   *full* Lousto-Zlochower 2008/2011 formula including the in-plane
   "superkick" term `v_⊥` (not the aligned-only
   `v_kick_aligned` currently in `remnants/kick.py`).
4. **Marginalise** over the prescribed `P(θ̂)`:
   * for **basis / implicit (deterministic)**: spherical quadrature
     on the orientation axes.  The output is a *distribution* over
     `(|χ_rem|, v_kick)` that scatters across multiple output χ
     cells.
   * for **particle MC**: a single sample of `(θ_i, φ_i)` per merger
     event; the resulting `|χ_rem|` is the new particle's spin
     magnitude, the resulting `v_kick` enters the retention sigmoid.

The orientation distribution is the **model knob**, ranging from
`Aligned()` (δ at `θ̂ = L̂`; current behaviour) through
`PartialAligned(κ)` (Fisher distribution on the sphere, concentration
`κ`) to `Isotropic()` (uniform on the sphere).  This is *not
self-consistently evolved*: every merger draws fresh from the same
`P(θ̂)`, regardless of the progenitor's generation or merger history.
The "magic extra astrophysics" framing is the user's: whatever
physical process (Lidov-Kozai, post-merger torque, cluster encounter)
re-randomises tilts between merger events is treated as an external
prescription, not a tracked field.

## Why this is not a broadening operator

The earlier (rejected) PLAN proposed an operator `R[n](χ_out) = ∫
n(χ_in)/χ_in dχ_in` applied to the gain term.  That was *exactly* the
fake-physics failure mode `[[feedback-no-fake-physics-stand-ins]]`
warns against: ad-hoc smoothing dressed as misalignment with no
honest connection to spin orientation.

The current PLAN doesn't choose any smoothing.  The output χ_rem
distribution comes out of `χ_rem_BR(m1, m2, χ1_vec, χ2_vec)` evaluated
on quadrature points of `P(θ̂)`, which is the actual physics.
Different `P(θ̂)` choices give *different* output distributions, and
the differences are physical.

## Retention is not separable from misalignment

Under aligned-only physics, `v_kick_aligned` is small (the `v_⊥`
"superkick" branch is absent) and `P_retain ≈ 1` for Stage 1's
`v_esc = 300 km/s`.  Under isotropic-tilt physics, the superkick term
reaches `O(few × 10³) km/s` for maximally-spinning misaligned
mergers — easily exceeding any cluster `v_esc`.  Retention drops
sharply.

This means **misalignment + ejection-driven retention are one story
and land together**.  The earlier PLAN's "Phase 1A keeps retention
as-is; Phase 2 wires kicks" decomposition was wrong: it would have
produced a non-physical regime where remnant spins broaden but no
ejection happens.

A useful consequence: under any non-trivial orientation distribution,
`P_retain` becomes a smooth function of `v_esc` (integral of Heaviside
over a continuous distribution is a smooth CDF).  This **recovers
`∂/∂v_esc` gradients for free** — the AD limitation flagged in
`remnants/kick.py`'s module docstring goes away as soon as we land
this work.  Two-for-one.

## File layout

### New

* `src/hierarchical_ad_pop_engine/spin_orientation.py` —
  the orientation-distribution abstraction.
  * `class OrientationModel(eqx.Module)`: abstract; exposes
    `quadrature_3d() -> Array[(n_orient, 4)]` returning the flat
    array of (θ1, θ2, Δφ, weight) tuples for the post-azimuthal-
    symmetry-reduced 3-D quadrature, and `sample(key, n_events) ->
    (theta_1, phi_1, theta_2, phi_2)` for MC.  Flat-quadrature API
    deliberate — feeds directly into the inner `jax.lax.scan` on
    the basis / implicit sides (Decision 3).
  * `Aligned()` — single quadrature point `(0, 0, 0, 1)`.  Used
    only as the κ→∞ limit reference; in production the
    `isinstance(orientation, Aligned)` shortcut bypasses
    marginalisation entirely (Decision 2).
  * `Isotropic()` — product Gauss-Legendre on `cos θ_1` × `cos θ_2`
    × uniform-trapezoidal on `Δφ`, default `(n_θ, n_θ, n_Δφ) = (8,
    8, 16)`.  Uniform `sample`.  The κ=0 limit of `Fisher`.
  * `Fisher(kappa)` — von Mises-Fisher on the 2-sphere concentrated
    along `L̂ = ẑ` with concentration `κ`.  Quadrature points are
    the same as `Isotropic()`; quadrature weights are the Fisher
    weight evaluated at each point (smooth, differentiable in `κ`).
    `κ → ∞` recovers `Aligned()` numerically (asymptote, not
    bit-equivalent — see Decision 2).
  * `MixtureOrientation([(w_i, model_i), ...])` — weighted union of
    component quadratures.  Lands later, once `Fisher` is stable;
    needed for LIGO-literature parameterisations
    (`Isotropic + GaussianCosTheta(σ)` and `Isotropic + Aligned`).
    `GaussianCosTheta(sigma)` is the only other concrete model
    eventually needed.

* `src/hierarchical_ad_pop_engine/remnants/spin_vector.py` —
  full vector-spin Barausse-Rezzolla 2009.
  * `chi_rem_vec(m1, m2, chi1_vec, chi2_vec) -> chi_rem_vec`.
  * Reduces to the existing scalar `chi_rem(m1, m2, χ1, χ2)` exactly
    when both input vectors are along `L̂ = ẑ`.  Pinned by a
    machine-precision regression test.

* `src/hierarchical_ad_pop_engine/remnants/kick_vector.py` —
  full vector-spin Lousto-Zlochower 2008/2011 including the `v_⊥`
  superkick term.
  * `v_kick_vec(m1, m2, chi1_vec, chi2_vec) -> v_kick_scalar`.
  * Reduces to the existing `v_kick_aligned` when both spins are
    along `L̂`.  Pinned regression test.

* `src/hierarchical_ad_pop_engine/remnants/marginal.py` —
  tilt-marginalised wrappers.  Pure quadrature/sampling — no new
  physics, just integrates the above over an `OrientationModel`.
  * `chi_rem_distribution(m1, m2, chi1_mag, chi2_mag, model)
    -> (chi_rem_samples, weights)` for the basis/implicit path.
  * `chi_rem_sample(key, m1, m2, chi1_mag, chi2_mag, model)
    -> chi_rem_scalar` for MC.
  * Similarly for kick and retention.
  * `P_retain_marginalised(m1, m2, chi1_mag, chi2_mag, v_esc, model)
    -> P_scalar` — integral of `1[v_kick < v_esc]` over `P(θ̂)`.
    Smooth in `v_esc` for any non-Aligned model.

* `tests/test_orientation_models.py` — quadrature accuracy,
  isotropy invariants (`∫ dΩ = 4π`, second-moment recovery).

* `tests/test_remnant_vector.py` — vector formulas reduce to
  scalar formulas at `Aligned()`; published-case spot checks
  against Barausse-Rezzolla 2009 Tables / Lousto-Zlochower 2011.

* `tests/test_misalignment_invariants.py` — solver-level
  invariants:
  * `Aligned()` reproduces every existing test to machine precision
    (NOT bit-for-bit — different code paths through equivalent
    algebra).
  * `Isotropic()` makes 2G/3G χ peaks broader than at `Aligned()`
    at the *same* nodal count, with simultaneously smaller
    `total_rate` (ejection branch active).
  * `∂P_retain/∂v_esc` is finite and non-zero for any non-Aligned
    model.
  * `Isotropic()` chi_rem-distribution moments match a numpy
    Monte Carlo backstop at large `n_orient`.

### Modified

* `src/hierarchical_ad_pop_engine/retention.py` —
  `P_retain(..., orientation: OrientationModel = Aligned())`.
  Dispatches at trace time:

  ```python
  if isinstance(orientation, Aligned):
      return retention_aligned(m1, m2, chi1, chi2, v_esc)  # bit-for-bit existing path
  return P_retain_marginalised(m1, m2, chi1, chi2, v_esc, orientation)
  ```

  Existing call sites pass `Aligned()` explicitly until they're
  updated, preserving bit-for-bit behaviour.

* `src/hierarchical_ad_pop_engine/response/coagulation.py` —
  `CoagulationResponse` gains an `orientation: OrientationModel =
  Aligned()` field.  `_step_one_generation` dispatches on
  `isinstance(response.orientation, Aligned)`:
  * Aligned → existing scalar `chi_rem`-based bilinear scatter,
    unchanged.
  * Marginalised → `jax.lax.scan` over orientation quadrature
    points.  Inside each scan iteration: build vector spins from
    `(χ1_mag, χ2_mag, θ1, θ2, Δφ)`, call `chi_rem_vec` /
    `v_kick_vec` (scalar-valued for given orientation), do the
    same bilinear scatter on the resulting scalar χ_rem,
    accumulate weighted into the gain accumulator.  **No
    orientation-indexed intermediate is ever realised** — the
    accumulator has the same shape as today's gain
    `(N_m, N_χ)`; only the current scan iteration's contribution
    is alive at a time.

* `src/hierarchical_ad_pop_engine/basis/remnant_tensor.py` —
  `build_remnant_tensor(theta, basis_solver, orientation,
  n_quad_factor)`.  Dispatches on `isinstance(orientation, Aligned)`:
  * Aligned → existing T-build, unchanged.
  * Marginalised → `jax.lax.scan` over orientation quadrature
    points from `orientation.quadrature_3d()`.  Inside each
    iteration: existing T-build inner code runs with `chi_rem` and
    `v_kick` replaced by their per-orientation evaluation; the
    contribution to T is a single `(n_basis, n_basis, n_basis)`
    tensor; accumulator is the same shape; nothing is materialised
    in angle.  Build cost is `n_orient × current` (≈ minutes at
    `n_orient ≈ 1024`); build memory is `2× T_size` peak (the
    accumulator plus the current contribution), unchanged from
    aligned.

* `src/hierarchical_ad_pop_engine/basis/loss_tensor.py` —
  same dispatch + same scan pattern as remnant_tensor.

  **Critical for the implicit side:** `response/implicit.py::loss_density`
  currently materialises a `(N_m, N_χ, N_m, N_χ)` ≈ 75 MB 4-D
  integrand.  Naively extending with orientation axes would
  multiply that by `n_orient ≈ 1024` → 37 GB.  The scan-over-
  orientations pattern is *required* on the implicit side, not
  optional: each scan iteration computes the existing 4-D
  integrand at one orientation, contracts to 2-D via the existing
  einsum, accumulates.  Peak memory stays at ~75 MB.

* `red_team/particle_mc.py` — at each merger event, sample
  `(θ1, φ1, θ2, φ2)` from `model.sample(key, n_events=1)`, compute
  the vector spins, run `chi_rem_vec` and `v_kick_vec` per event.
  Per-event Heaviside retention against the per-event `v_kick` —
  no marginalisation needed on the MC side.  Aligned shortcut is
  also a dispatch here, for fairness.

* `demo/forward_modeling_observables.py` — CLI `--orientation
  {aligned,isotropic,fisher:κ}` selects the model; default
  `aligned`.

## Acceptance invariants

* **Aligned-default reproduces current behaviour** to machine
  precision across the entire existing test suite (not bit-for-bit
  — different code paths, equivalent algebra).
* **Cascade compression breaks** under `Isotropic()`: 2G/3G χ peaks
  visibly broader at the same nodal count and the cascade tail does
  not collapse into delta functions.
* **Retention activates correctly**: under `Isotropic()`,
  `total_rate` is meaningfully smaller than under `Aligned()` at the
  same `v_esc` — high-spin misaligned remnants are getting kicked
  out and the engine sees it.
* **AD on `v_esc` recovers**: `jax.grad` of `total_rate` w.r.t.
  `v_esc` is finite and non-zero for any non-Aligned model.  This is
  the v2-module promise from `remnants/kick.py` cashed in.
* **No frozen numerical references** — every invariant is
  engine-vs-parameter-free-reference equality
  (`[[feedback-no-frozen-test-references]]`).

## Quadrature accuracy on the sphere

The orientation integration is a 3-D integral on `(θ1, θ2, Δφ)` after
azimuthal-symmetry reduction.  **Product Gauss-Legendre on `cos θ_1`
× `cos θ_2` × uniform-trapezoidal on `Δφ`** — simple, exact for
polynomials, well-understood error bounds.  Default `(n_θ, n_θ,
n_Δφ) = (8, 8, 16) = 1024` quadrature points per (m1, m2, χ1, χ2)
cell.  Verify against a 4× refinement before believing any
production number.

Lebedev sphere quadrature is *not* needed: under Decision 3 the
orientation integration is a flat scan over a 1-D list of quadrature
points, so any reduction in *count* is the only thing that matters,
and Gauss-Legendre × Gauss-Legendre × trapezoidal is asymptotically
optimal for the smooth weight functions involved (the BR / LZ
formulas are polynomial in vector-spin components).

## Decisions captured 2026-05-25

1. **Orientation parameterisation: Fisher only for initial development.**
   `PartialAligned(κ)` via Fisher centred on `L̂ = ẑ` is the working
   model.  `Aligned()` (the κ→∞ limit) and `Isotropic()` (the κ=0
   limit) ship as concrete classes too since they are clean physical
   reference points.  *Eventually* — flagged for follow-up — we need
   `Isotropic + GaussianCosTheta(σ)` and `Isotropic + Aligned`
   mixtures because that's what the LIGO population-inference
   literature uses (see e.g. Talbot & Thrane 2017 / 2018, Vitale et
   al. 2017).  This pushes us toward a **`MixtureOrientation([(w1,
   m1), (w2, m2), …])` composition wrapper** rather than special-
   casing each parameterisation: any concrete orientation model can
   be mixed with any other by weight-combining their quadratures.
   PLAN lands Fisher first; mixture wrapper follows once Fisher is
   stable.

2. **Aligned path: shortcut.**  When `OrientationModel == Aligned()`,
   `retention.P_retain`, `_step_one_generation`, and
   `build_remnant_tensor` dispatch to the *existing* scalar
   `chi_rem` / `v_kick_aligned` / Heaviside-`retention_aligned` code
   path.  Bit-for-bit aligned preservation is required, not just
   machine-precision equivalence — the user wants the aligned
   pathological cascade features intact for future study, not
   reconstructed by a different code path that agrees only to float
   epsilon.  Dispatch is via `isinstance(orientation, Aligned)` at
   trace time — Python-time branch, jit-stable.  Two code paths,
   both jit'd, no runtime dispatch cost.

3. **Quadrature-cost budget: no intermediate realisation in angle.**
   The basis T-tensor build and the implicit-side loss/gain *must
   not* materialise the orientation axes as extra leading
   dimensions of an intermediate tensor.  Orientation integration is
   an inner `jax.lax.scan` over a flat array of quadrature points
   `(θ1, θ2, Δφ, w_orient)`, accumulating one (`n_basis,
   n_basis, n_basis`) T-contribution per scan iteration — only one
   contribution is alive at a time, peak memory unchanged from the
   current aligned build.  Same for the implicit-side 4D loss
   integrand (~75 MB) — a scan over orientations would multiply that
   by n_orient → 37 GB if naïvely vmapped, so scan is *required*,
   not optional.  Build *cost* is n_orient × current; build
   *memory* is unchanged.

4. **κ differentiable: yes.**  `jax.grad` flows through Fisher
   quadrature weights cleanly; gwkokab will see κ as a fittable
   parameter.  No implementation cost.

## Sequence for next session

Estimated effort: **2–3 sessions**, not one.  The math is shallow but
the surface area is larger than the evaluation API.  Decisions above
resolved 2026-05-25 — coding can start immediately on Session A.

**Session A — physics primitives**

1. Land `spin_orientation.py` with `Aligned`, `Isotropic`,
   `Fisher(κ)`.  `quadrature_3d()` flat-array API; sample method
   for MC.  `test_orientation_models.py` covers: `Fisher(κ→∞)`
   asymptote to `Aligned()` quadrature, `Fisher(0) == Isotropic`,
   `∫ dΩ` normalisation, second-moment recovery, κ-grad finite.
2. Land `remnants/spin_vector.py` (full BR vector formula).
   `test_remnant_vector.py` with the scalar-reduction invariant
   (both vectors along `ẑ` → existing `chi_rem` to machine
   precision) plus published-case spot checks.
3. Land `remnants/kick_vector.py` (full LZ with `v_⊥` superkick
   term).  `test_remnant_vector.py` extensions: scalar reduction
   to `v_kick_aligned` at aligned input; high-spin misaligned
   peak values match Lousto-Zlochower 2011 Table II within a few %.

**Session B — marginalisation + solver wiring**

4. Land `remnants/marginal.py` — `chi_rem_distribution`,
   `chi_rem_sample`, `P_retain_marginalised`.  Pure
   quadrature/sampling wrappers — no physics — over
   `OrientationModel.quadrature_3d()` / `.sample()`.
   `test_remnant_marginal.py` against numpy-MC backstop at large
   `n_orient`.
5. Wire `retention.P_retain` with `isinstance(orientation,
   Aligned)` shortcut dispatch.  Verify every existing test still
   green with `Aligned()`.
6. Wire `CoagulationResponse.orientation` field and
   `_step_one_generation` aligned-shortcut + scan-over-orientations
   marginalised path.  Re-verify every existing test still green
   with `Aligned()` — bit-for-bit, since the shortcut routes to
   the unchanged code path.
7. Wire `basis/remnant_tensor.py` and `basis/loss_tensor.py` —
   aligned shortcut + scan-over-orientations marginalised path
   per Decision 3.  Confirm peak memory unchanged from aligned
   build under scan.

**Session C — MC + demo + acceptance**

8. Wire `red_team/particle_mc.py` orientation sampling.
9. Run `forward_modeling_observables.py --orientation isotropic`
   and `--orientation fisher:κ` for a few κ.  Confirm cascade
   compression breaks visibly under `isotropic` and is recovered
   under `fisher:κ` as κ→∞.
10. Confirm `jax.grad(total_rate)(v_esc)` finite and non-zero
    under `isotropic` / `fisher` (this is the AD-on-`v_esc`
    two-for-one win).
11. Write `tests/test_misalignment_invariants.py`.

Sessions can be re-ordered if blockers arise; A → B → C is the
dependency line.  `MixtureOrientation([(w_i, model_i), ...])` and
`GaussianCosTheta(σ)` land as a Session D follow-up once Fisher
is stable — needed for LIGO-literature parameterisation
comparisons but not blocking the cascade-compression fix.

## Related

* `[[project-aligned-chi-undersampled]]` — the pathology this addresses.
* `[[feedback-no-fake-physics-stand-ins]]` — the rule the earlier
  PLAN draft violated.  This PLAN does NOT introduce a smoothing
  operator; the χ_rem broadening is the consequence of real BR
  evaluated on real misaligned vector spins.
* `[[feedback-no-hardcoded-physical-scales]]` — `κ` is a model
  parameter not a hardcoded smoothing scale.  Fisher concentration
  has dimensional meaning (it's a rate per square radian on the
  sphere) but it's an explicit knob, not buried.
* `[[feedback-no-fudge-factors]]` — `κ`'s endpoints (0 = isotropic,
  ∞ = aligned) are clean physical limits.  Intermediate values are
  honest partial-alignment fractions.
* `[[feedback-no-frozen-test-references]]` — every invariant in
  this PLAN is engine-vs-reference equality.
* `[[feedback-dont-straightjacket-basis]]` — the basis-side
  implementation extends T-build quadrature axes without
  special-casing.
* `[[project-chi-rem-asymmetry-finding]]` — the `m1 ≥ m2`
  canonicalisation convention.  Vector BR has the same
  asymmetry and the same canonicalisation must be done at the call
  site (matches the existing `chi_rem` / `v_kick_aligned` pattern).
* `remnants/kick.py` module docstring — explicitly says
  "tilt-marginalised kicks ... are a separate v2 module".  This
  PLAN is that v2 module.
* `retention.py` module docstring — "Future tilt-marginalised
  variants will plug in here behind the same function name."  This
  PLAN delivers them.
