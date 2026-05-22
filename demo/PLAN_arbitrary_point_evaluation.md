# PLAN — arbitrary-point density / observable evaluation API

Written 2026-05-22 after the forward-modelling demo surfaced a
conflation between *internal solver resolution* and *plot/query
accuracy*.  This plan defines the long-term architecture; the demo
brute-forces a placeholder (fixed-bandwidth KDE) pending its
implementation.

## Why this exists

The forward-modelling demo had a sweep over engine `(N_m, N_χ)` and
showed:

* **Engine LIGO 1-D plots improved with resolution.**  Expected —
  the engine bin-scatters the 4-D rate density onto histogram edges
  set by the engine grid, so finer engine = finer engine output.
* **MC LIGO 1-D plots ALSO improved with engine resolution.**  Bug.
  The MC algorithm has nothing to do with the engine grid; the only
  reason this happened was that I was histogramming MC samples into
  engine-grid bins.  Coupling MC plot quality to engine resolution
  is the cleanest possible expression of "we conflated internal
  resolution with evaluation accuracy".
* **MC tails (3G+) look "fainter than implicit" even at well-
  resolved engine grids.**  The discriminator the demo needs — "is
  this MC shot noise, MC over-smoothing, or genuine cross-method
  miscalibration?" — is unanswerable until the *evaluator* on each
  method is well-defined and decoupled from the solver's internal
  representation.

Beyond the demo: gwkokab and any downstream inference layer wants
to **query the rate density at arbitrary continuous points** (e.g.,
at observed-event posterior samples for a Poisson likelihood).
Today this is hand-implemented per call site by bilinear-
interpolating the implicit solver's grid (`log_density_at`); the
basis solver has its own evaluator; the MC has none.  We need a
common interface.

## Goals

1. Define a `RateDensityEvaluator` protocol that every solver
   (implicit, basis, MC) implements.
2. Decouple **plot/query resolution** from **solver resolution**.
3. **No hardcoded physical scales.**  The MC adaptive bandwidth
   must discover local widths from particles, NOT from
   prior knowledge of natal-Gaussian width or cascade locations.
   This is the user's stated criterion:
   > "generically we can't hardcode in scales for natal mass peaks/etc !
   >  — must discover from particles — and lots of substructure"
4. AD compatibility on the inference-side paths (implicit, basis);
   MC may be non-AD since it's a validation backstop.
5. Refactor the existing forward-modelling demo to use the new API
   once it's available; remove the placeholder KDE / `--bw-scale`
   plumbing in the process.

## Non-goals

* Replacing `observables.py` quadrature contractions on the engine
  grid — those stay as the *fast, AD-friendly, batched* path for
  inference-time marginal computation.  The new API is **additive**:
  it adds an "evaluate at these specific points" capability without
  removing the existing "give me the full grid marginal" capability.
* Speculative performance optimisation of the adaptive KDE.  A
  ``N · M`` scatter is fine at the scales we care about
  (``N ≈ 5×10⁵`` particles, ``M ≈ 400`` eval points); revisit only
  if the API gets called inside an inner inference loop.

## Scope

### Queries the API must answer

| query | implicit | basis | MC |
|---|---|---|---|
| ``N(m, χ)`` at one or many (m, χ) points | bilinear interp on (log m, χ) grid (exists as `log_density_at`) | basis expansion (exists as `TensorProductBasis.evaluate`) | **adaptive KDE in (log m, χ)** |
| ``½ R_env K(m₁,m₂) N(m₁,χ₁) N(m₂,χ₂)`` at one or many (m₁, m₂, χ₁, χ₂) | product of two ``N`` evals × analytic K | same | KDE-evaluated N at each requested point; K is shared |
| ``dN/dx_i(x_i)`` at one or many query x_i for derived axis (m₁, q, χ_eff, M_chirp) | constrained integral of the 4-D rate density via adaptive quadrature | same, but in basis-coefficient space | KDE in x_i directly from the per-event derived axis (m₁, q, χ_eff, M_chirp samples are O(N) to compute from the mergers array) |
| ``dN/(dx_i dx_j)`` at query (x_i, x_j) | 2-D constrained integral | same | 2-D adaptive KDE on the derived-axis pair |

### Out of API scope

* Anything that requires the *full* density tensor — those calls
  use the existing grid path.  The new API is point-query / sparse
  vectorised; the existing observables.py is bulk integral.
* Evaluation of the engine outside its support.  Each evaluator
  returns 0 outside its representable region (grid bounds for
  implicit, basis support for basis, sample range for MC); callers
  must clamp / mask if they want physical interpretation outside.

## Protocol

```python
# src/hierarchical_ad_pop_engine/evaluation/protocol.py
from typing import Protocol, Tuple
from jax import Array

class RateDensityEvaluator(Protocol):
    """Evaluate the forward-rate density at arbitrary binary points.

    All methods accept JAX arrays of any broadcast-compatible shape;
    output shape matches the input under standard broadcasting.

    AD: implicit and basis evaluators are JAX-traceable end-to-end;
    the MC evaluator is numpy-only (validation backstop, not in the
    inference AD path).
    """

    def density(self, m: Array, chi: Array) -> Array:
        """Single-BH density N(m, χ).  Non-negative."""
        ...

    def rate_density_4d(
        self, m1: Array, m2: Array, chi1: Array, chi2: Array
    ) -> Array:
        """½ R_env K(m₁,m₂) N(m₁,χ₁) N(m₂,χ₂)."""
        ...

    def marginal_1d(self, axis: str, x: Array) -> Array:
        """dN/dx at query points x.  axis ∈ {'m1','m2','chi1','chi2',
        'q','chi_eff','M_chirp','m_heavy','m_light'}."""
        ...

    def marginal_2d(
        self, axes: Tuple[str, str], x_a: Array, x_b: Array,
    ) -> Array:
        """dN/(dxa dxb) at query points."""
        ...

    def total_rate(self) -> Array:
        """Scalar total rate ∫∫∫∫ ½ R_env K N N."""
        ...
```

Construction sites:

```python
# Each solver's result type gains a `.evaluator()` method.

result_implicit: ImplicitSolveResult = solve_implicit_fixed_point(...)
ev: RateDensityEvaluator = result_implicit.evaluator(response)

result_basis: BasisSolveResult = solve_basis_fixed_point(...)
ev = result_basis.evaluator(response, basis_solver)

mc_out = particle_mc(...)
ev = MCEvaluator(mc_out["mergers"], response.log_R_env, ...)
```

## Per-implementation evaluator design

### Implicit (grid Picard)

* Backing data: ``ImplicitSolveResult.N`` on the engine ``BinaryGrid``.
* `density(m, χ)` = `log_density_at(grid.log_m, grid.chi, log_N, m, χ)`
  (bilinear interp in (log m, χ), already implemented).
* `marginal_1d(axis, x)` for derived axes: compute the constraint
  integral by **adaptive quadrature** (e.g., `scipy.integrate.quad`
  in numpy mode; `jax.experimental.ode`-free approach in AD mode).
  For source axes (`m1`, `chi1`): just integrate the 4-D rate
  density along the kept axis via grid quadrature, then bilinear-
  interp at the query points (this preserves the existing
  `observables.marginal_1d` answer when queried at grid points).
* AD: must remain jit-friendly.  Adaptive quadrature can be tricky
  to AD; alternative is to over-resolve a fixed quadrature grid
  internal to the evaluator (e.g., 4× engine grid) and bilinear-
  interp the result.

### Basis (v3 Galerkin)

* Backing data: ``BasisSolveResult.coeffs`` on a `TensorProductBasis`.
* `density(m, χ)` = `basis_solver.evaluate(coeffs, log m, χ)`
  (already exists).
* Marginal evaluations: leverage basis-quadrature identities where
  possible (the basis Galerkin tensors `T_{γαβ}`, `L_{γα}` already
  encode the integrand structure).  For arbitrary-point queries,
  fall back to: evaluate the density at a dense grid in the
  unobserved axes, do trapezoidal quadrature, bilinear-interp the
  result at query points.
* AD: end-to-end jax via the existing basis evaluator.

### MC (particle backstop)

* Backing data: the mergers array shape ``(N_events, 8)``
  ``(m1, m2, χ1, χ2, g1, g2, retained, m_remnant)``.
* `density(m, χ)` = adaptive KDE in (log m, χ) over the surviving
  *single-BH* population (NOT the mergers array — that's the rate
  density, see below).  Actually subtle: the single-BH density and
  the rate density are different objects; the MC has direct access
  to the surviving population (`m_active`, `chi_active` from
  `particle_mc(...)`), which is the natural backing for `density()`.
* `rate_density_4d(m1, m2, χ1, χ2)` = adaptive KDE on the
  4-D mergers array directly.  Or: `½ R_env K · density(m1,χ1) ·
  density(m2,χ2)` via the single-BH evaluator (more parsimonious;
  same approximation as the engine's rate-density definition).
  Pick one and pin in the protocol — probably **the latter** for
  consistency with how implicit/basis define rate_density_4d.
* `marginal_1d(axis, x)` for derived axes: KDE on the per-event
  derived-axis values (e.g., `chi_eff = (m1·χ1 + m2·χ2)/(m1+m2)`
  per row of mergers).  This is what the placeholder in the demo
  does — but with adaptive bandwidth.
* `marginal_2d`: 2-D adaptive KDE on the derived-axis pair.

#### Adaptive KDE specifically

**Required:** location- and density-dependent bandwidth, discovered
from the particles.  No hardcoded physical scales.

Two well-established candidates:

**(A) Abramson's adaptive KDE** (1982).  Two-pass:
1. Fit a pilot density ``f̂_pilot`` with global Silverman bandwidth.
2. Per-sample bandwidth ``h_i = h_0 · (f̂_pilot(x_i) / G)^(-α)``
   with ``α = 1/2`` (Abramson's choice — gives ``O(h^4)`` bias).
   ``G`` is the geometric mean of ``f̂_pilot`` at the sample points.
3. Evaluate the variable-kernel KDE at query points.

Pros: principled, well-known asymptotic rates, single hyperparam
(initial ``h_0``, often chosen by cross-validation).
Cons: depends on the pilot — bad pilot → bad bandwidth at high
mass tails.  Mitigations: log-transform mass axis before piloting;
use a robust pilot (e.g., diffusion KDE) if Abramson over-smooths
tails.

**(B) k-NN bandwidth** (Loftsgaarden-Quesenberry / "balloon
estimator"):
Per *eval-point* bandwidth ``h(x) = distance to k-th nearest
sample``.  Simpler than Abramson, no pilot needed, but bias
properties are worse in the multimodal case.

Recommendation: **start with Abramson** in log-space for mass and
linear space for spins/q.  Provide k-NN as a fallback (toggle).

#### Boundary handling

* Mass axes (log m): reflect at ``log m_min``.  Above ``log m_max·k``
  for large k, density is naturally zero — no reflection needed.
  Actually we may want NO reflection at all for log mass — the natal
  distribution decays smoothly; reflection at ``m_min`` would create
  a spurious double-peak.  Try both and pin empirically.
* χ axes: reflect at both 0 and 1 (hard boundaries).
* q axis: reflect at 1 (hard); 0 is asymptotic.
* χ_eff: reflect at 0 and ±1 (Stage 1 aligned-only → [0, 1]; future
  precessing → [-1, 1]).
* M_chirp (log): no reflection — natural support; reflection would
  bias the bulk.

### MC validation tests

* `density()` integrated over engine grid via trapezoid ≈
  ``N_active / total environment-time normalisation``.  Match to
  implicit `result.N` integral within shot-noise envelope.
* Compare adaptive-bandwidth KDE marginal to a *very fine raw
  histogram* of the same samples.  KDE should be a smoothed version
  of the histogram, NOT systematically shifted or biased in
  amplitude.  Concretely: integrate the KDE against the histogram
  edges; ratio should be within 1% bin-by-bin in the bulk and
  within shot-noise envelope in the tails.

## Composition with `observables.py`

* `observables.py` keeps its current grid-quadrature path
  unchanged.  That is the AD-friendly bulk path used by inference.
* A new module `evaluation.py` (or a sub-package
  `evaluation/{implicit, basis, mc}.py`) provides the
  arbitrary-point query layer.
* `marginal_1d(axis, x_query)` for grid-resident solvers can be
  implemented in two layers:
  1. Compute the full grid marginal via `observables.marginal_1d`
     (returns a (N_kept,) array on the engine grid).
  2. Bilinear-interpolate at `x_query`.
  This guarantees the new API agrees with the old API at grid
  points — useful regression invariant in tests.

## gwkokab integration

* gwkokab population fitting expects a `log_rate_density(m1, m2,
  χ1, χ2)` evaluator that supports broadcast and is differentiable
  in θ.  Today `CoagulationResponse.log_rate_density` provides this
  via the implicit solver + bilinear interp; the new protocol
  formalises it and lets the basis solver substitute in
  transparently.
* Effective signature inside gwkokab:
  ```python
  ev = response.evaluator()      # under the hood: solver-specific
  log_rate = jnp.log(jnp.maximum(
      ev.rate_density_4d(m1, m2, χ1, χ2), 1e-300
  ))
  ```
  No call-site change needed when swapping solvers.

## File layout

```
src/hierarchical_ad_pop_engine/
  evaluation/
    __init__.py
    protocol.py         # RateDensityEvaluator protocol
    implicit.py         # ImplicitEvaluator (wraps ImplicitSolveResult)
    basis.py            # BasisEvaluator (wraps BasisSolveResult + basis_solver)
  observables.py        # unchanged

red_team/
  evaluation_mc.py      # MCEvaluator (adaptive KDE; lives in red_team because
                        # the particle MC itself is red-team)
```

Each `*.py` exports a single class implementing `RateDensityEvaluator`.

## Test/validation strategy

1. **Equivalence at grid points.**  ImplicitEvaluator.density at the
   engine grid points exactly reproduces `result.N`.  marginal_1d at
   the engine grid points exactly reproduces `observables.marginal_1d`.
2. **Adaptive KDE on synthetic samples.**  Sample 10⁵ points from
   a known 2-Gaussian mixture in log m × χ; the MC evaluator must
   recover the mixture density to ≤ 5 % integrated L1 error on a
   fine eval grid.  Same test but with the modes at very different
   widths and very different sample counts — the *raison d'être* of
   the adaptive bandwidth.
3. **Cross-method invariant at θ\*.**  Implicit, basis, and MC
   evaluators all called at the same 400-point eval grid; the three
   marginals must agree on the 1G/2G bulk to JS ≤ noise floor.
4. **gwkokab smoke test.**  A simplest-possible inference call
   that uses the new `evaluator()` method instead of the existing
   `log_rate_density`; expected to give the same posterior to
   numerical precision.

## Open-question resolutions (user direction 2026-05-22)

* **Adaptive KDE bandwidth selector.**  **k-NN first**, Abramson
  *not* used.  Reason: aligned-only Stage 1 produces very narrow
  spin substructure at high gen (cascade of compressing scales);
  Abramson's pilot-density bandwidth oversmooths these.  k-NN's
  per-eval-point bandwidth adapts to local density and resolves
  small islands without smearing.
  > "MC bandwidth — issue is very small mass/spin 'islands';
  >  Abramson isn't capturing these well, probably want k-NN.
  >  Massive oversmoothing obvious in spin.  Research project —
  >  note aligned spins is an exceptional/unusual problem which
  >  surfaces many small substructures we wouldn't normally have."

  Note this is an aligned-only pathology; misaligned spin
  (production case) carries much broader features.

* **AD on the implicit evaluator's `marginal_1d` for derived axes.**
  **No AD required.**  The evaluator is for plots and ad-hoc
  export queries, not for the inference AD path (inference uses
  the engine grid + `observables.py` directly).  This unblocks
  using scipy adaptive quadrature, numpy histograms, etc.
  > "AD marginals?: no, this is just for plots/export, not fits."

* **Single-BH density vs rate density for the MC evaluator.**
  Confirmed: `MCEvaluator.density(m, χ)` is backed by the surviving
  pool ``m_active`` / ``chi_active`` (the natural `N(m, χ)` analogue
  in the MC), NOT by the mergers array.  The mergers array still
  backs the derived-axis marginals (``dN/dq``, ``dN/dχ_eff``, etc.)
  since those count merger *events*, not surviving BHs.

* **Where does the basis Galerkin tensor route through the
  evaluator?**  **Recompute K analytically; do not reuse the basis
  ``T_{γαβ}``/``L_{γα}`` tensors.**  Reason: evaluator queries are
  cheap one-off plot/export calls — reusing the precomputed Galerkin
  tensors would propagate basis quadrature systematics into the
  query path, while a fresh analytic ``K(m₁, m₂) · N(m₁,χ₁) ·
  N(m₂,χ₂)`` evaluation has only the basis density evaluator's
  systematic.  Cleaner separation of concerns.
  > "Basis reuse: Evaluation points are cheap plotting / query —
  >  just recompute analytically, otherwise we're coding in basis
  >  systematics."

## Known engine pathology — aligned-spin cascade (open)

Surfaced 2026-05-22 by running the forward-modelling demo at
``32×240`` (hand-tuned, low-mass / high-spin):

> "Implicit is wildly wrong in spin past 2nd gen — lots of narrow
>  spiky features in high-gen mergers for physics reasons.  Even
>  at 32×240 spin is still under-resolved with implicit.  Physics
>  means cascade of narrow scales — very hard, but unlikely problem
>  in real situations (with misalignment)."

What's happening: under aligned-only kicks (Stage 1 retention
model), successive generations land at increasingly narrow χ
values — 1G near χ ≈ 0, 2G piles at χ ≈ 0.69, 3G/4G compress into
even narrower features near 0.69.  A *uniform* χ grid resolves the
2G peak but blurs the 3G/4G islands.

User assessment: this is an aligned-only artifact; misaligned
spin (the realistic case) dilutes the substructure.  We may not
need to fix it for production.  Documented here so the next
session can decide.

### Mitigation options (not committed)

1. **Concentrated χ grid.**  Replace ``BinaryGrid.default``'s
   linear χ with a non-uniform array clustering points near
   ``χ_rem(1, 0) ≈ 0.69``.  The trapezoid quadrature in
   ``_dchi_weights`` and ``log_density_at`` already supports
   arbitrary monotone χ arrays — purely a constructor change.
   Could be exposed as ``BinaryGrid.concentrated_chi(peak=0.69,
   ...)``.

2. **χ reparametrisation.**  Map χ → some transformed variable in
   which cascade peaks have ~constant width.  Possible candidates:
   ``log(1 - χ)`` or ``arctanh(2χ - 1)``.  Requires changing all
   χ-axis quadrature consistently — much larger refactor.

3. **Higher-order interpolation.**  The bilinear scatter in
   ``_step_one_generation`` is O(h²).  A cubic scheme would
   help narrow features at the same nodal count.  Adds AD
   complexity.

4. **Track peaks separately.**  Decompose ``N(m, χ) = N_smooth +
   Σ_k A_k · δ_χ(χ - χ_k)`` with a small set of analytic deltas
   tracking the cascade-peak locations.  Most accurate, most
   complex.  Defer.

For the demo's purpose (illustrating cross-method agreement), the
existing under-resolution is acceptable provided it is flagged.
Add a note to the figure / panel titles when χ-axis disagreement
is suspected to come from this rather than from the MC or the
evaluator placeholder.

## Known MC ↔ implicit residual (open)

Forward-modelling demo at ``--n-mc 400000`` shows residual mass-
axis disagreement that doesn't close with the k-NN placeholder
or engine resolution:

> "MC vs implicit: still quite a lot of disagreement in mass.
>  Spin features are very narrow for higher generations, getting
>  compressed for good physics reasons."

Diagnosis sequence (deferred until the proper MC evaluator lands):

1. Switch from the placeholder k-NN to the production evaluator
   (whatever the new `red_team/evaluation_mc.py` ships with) and
   re-run.  Settles the smoothing axis.
2. Bump ``--n-mc`` by 4-10× at fixed engine resolution and watch
   the raw-histogram tails settle.  Settles the shot-noise axis.
3. If both above are stable and disagreement remains, that's
   genuine cross-method miscalibration — needs an engine vs MC
   *per-generation* breakdown to localise (which generation is
   carrying the disagreement?).

## Sequence for next session

1. Read this plan + `observables.py` + the existing
   `log_density_at` and `TensorProductBasis.evaluate` to reload
   context.  (Open-question resolutions already captured above.)
2. Land `evaluation/protocol.py` (just the Protocol class) +
   `evaluation/implicit.py` (wraps existing `log_density_at`).
   No new physics; should be ~50 LOC.  Pytest: grid-point
   equivalence to `result.N` and `observables.marginal_1d`.
3. Land `evaluation/basis.py` similarly (wraps
   `TensorProductBasis.evaluate`).  Pytest: grid-point equivalence
   to `density_from_basis`.  Recompute ``K`` analytically per the
   user's direction; do NOT route through the basis Galerkin
   tensors.
4. Land `red_team/evaluation_mc.py` with **k-NN balloon
   estimator** (not Abramson).  Pytest: 2-Gaussian mixture with
   modes of dissimilar widths and dissimilar sample counts; both
   modes must be recovered without smearing the narrow one or
   inventing structure in the wide one.
5. Refactor `demo/forward_modeling_observables.py` to use the
   new evaluators.  Delete the placeholder ``KNN_K_DEFAULT``,
   ``_gaussian_smooth_1d``, and ``--knn-k`` plumbing; the
   evaluator owns the bandwidth choice and any smoothing.
6. Re-run the diagnostic sequence in "Known MC ↔ implicit
   residual" with the new evaluators to settle whether the
   remaining mass disagreement is smoothing, statistics, or
   physics.

## Related

* `demo/PLAN_forward_modeling_observables.md` — the demo this plan
  was triggered by.
* [[project-forward-modeling-demo-scaffolded]] — current demo
  state, including the placeholder KDE.
* [[feedback-no-fake-physics-stand-ins]] — directly applicable:
  fixed-bandwidth KDE smearing hides cascade structure, that's a
  fake stand-in for the proper adaptive method.
* [[project-engine-v3-step4-done]] — basis solver, whose evaluator
  will plug into the new protocol.
* [[feedback-dont-straightjacket-basis]] — same plumbing principle
  applied to evaluators rather than solvers.
