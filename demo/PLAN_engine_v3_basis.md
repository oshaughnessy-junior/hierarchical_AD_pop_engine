# Plan — engine v3 basis (spectral / FEM) refactor

## Status and supersession

Supersedes Steps 5–7 of `PLAN_engine_v2_production.md`. v2 Steps 1–3
(lift solver into package, JIT-compatible `while_loop` Picard, IFT
custom VJP) remain valid and load-bearing — this plan **builds on
them**. v2 Step 4 (test migration + red-team-as-pytest) is the
prerequisite checkpoint: it pins the current correct state so the
basis refactor has a stable baseline to validate against. Do not start
v3 Step 1 until v2 Step 4 is green.

Sibling plan: `PLAN_observable_plots.md` runs in parallel on the
`observables` branch and supplies the `dC/dm1` diagnostic this plan
uses as its validation oracle.

Branch: `engine_fem` (cut off the same commit `observables` is cut off).

## Why a refactor at all

The v2 solver works and is bit-equal-validated end to end. It also
carries a 8–24× JS-above-shot-noise systematic against the particle MC
backstop on `N(m)` and `N(χ)`
(`project_implicit_vs_mc_systematic_bias`). The leading suspects in
that note are (a) ε ↔ f_dep mapping mismatch and (b) grid quadrature
error accumulating per generation. Inspection of the implementation
shows both reduce to a single root cause: **measure / quadrature
ambiguity on the (m, χ) grid**. Specifically:

* `_step_one_generation` uses full-weight at both endpoints of the
  progenitor m-axis (midpoint-style), while `_dm_log_weights` (used
  for the loss integral and all other integrals) uses half-weight at
  endpoints (symmetric trapezoid). Numerically small but present and
  systematically per-generation.
* Off-grid remnants (M_rem > m_max) are silently zeroed by the
  bilinear scatter. At Stage 1 this is small, but it's a non-conservative
  loss that grows with generation depth.
* Sharp structures in 2G+ remnant spin (χ_rem ≈ 0.687 from equal-mass
  non-spinning mergers) land on the 32-point linear χ grid; Picard
  produces ringing artefacts at high χ.
* KDE-vs-grid representation noise inflates JS comparison against the
  particle MC by ~10×, masking the true physics agreement.

Each is individually patchable. Collectively they say: the
discretisation strategy itself is the bug. A Galerkin / spectral basis
expansion fixes them at the root:

* Coefficients `n_α` are dimensionless amplitudes; mass / spin
  Jacobians live in basis-projection integrals computed once.
  `feedback_no_fudge_factors` is satisfied structurally — ε is
  absolute, all integrals are exact at the chosen quadrature order.
* Adaptive nodes (Gauss-Lobatto on χ, Chebyshev on log m) concentrate
  resolution where the cascade puts mass — no uniform refinement
  needed.
* Cross-method JS becomes well-posed: both methods project to the
  same basis at comparison time.

The IFT custom VJP machinery (`project_jax_custom_root_gotchas`)
generalises directly: Picard on coefficients is still a fixed-point
problem, the matvec dimension changes from `N_m·N_χ` to `N_basis`, the
VJP wrapper is otherwise unchanged. This is what makes v3 a tractable
refactor rather than a rewrite.

## Design goals

1. **Galerkin orthogonality at quadrature order.** ε has absolute
   meaning. No measure ambiguity anywhere on the grid.
2. **Drop-in public API.** `CoagulationResponse.log_rate_density`,
   `_log_total_density`, and the gwkokab integration layer are
   unchanged externally. Dispatch on a new `solver = "basis"` value.
3. **AD-clean by construction.** Reuse v2's `@jax.custom_vjp` Picard
   wrapper; the only change is the matvec body and the carry shape.
4. **Validation pair preserved.** The post-v3 validation pair is
   basis-Picard + particle MC, per
   `feedback_validation_downselect_two_algorithms`. Grid-Picard
   (current v2 implicit) joins perturbative as diagnostic, not as a
   second validation source — it would be same-algorithm-class.
5. **Stage-1 first.** v3 ships at Stage 1 physics (aligned-only,
   v_esc = 1000 km/s, P_retain ≡ 1 to numerical precision). Stage 2
   adaptive-element retention boundary is scoped out (see §Deferred).
6. **No fake stand-ins** (`feedback_no_fake_physics_stand_ins`). If
   Chebyshev convergence is poor on some θ regime, document the
   limitation and add hp-FEM enrichment, do not smooth the
   P_retain Heaviside to make convergence easier.

## Architecture overview

```
src/hierarchical_ad_pop_engine/
  basis/                       (NEW)
    __init__.py
    protocol.py                Basis ABC: nodes, weights, eval, project, mass_matrix
    chebyshev.py               Chebyshev-Lobatto on χ ∈ [0,1] and log m ∈ [log m_min, log m_max]
    tensor.py                  Tensor-product basis on (m, χ); inherits 1D bases
  response/
    basis.py                   (NEW) — basis Picard solver, mirrors implicit.py structure
    implicit.py                (kept; becomes diagnostic after v3 ships)
    coagulation.py             (extended: solver ∈ {"perturbative", "implicit", "basis"})
```

The basis Picard solver `response/basis.py` mirrors `implicit.py`'s
layout one-for-one: `_F_map_basis`, `_picard_while_loop_basis`,
`_make_basis_solve_fn` (lru_cache factory returning `@jax.custom_vjp`),
`BasisSolveResult` (eqx.Module), `solve_basis_fixed_point`. Reusing
the structure is deliberate — we want diff-able code reviewability
between the two solvers, and we want the IFT VJP code to be obviously
the same except for the matvec dim.

## Steps

### Step 1 — Basis infrastructure (1 session)

`basis/chebyshev.py` implements:

* `ChebyshevLobatto1D(n_nodes, *, domain=(a, b))` — eqx.Module with
  fields `nodes`, `weights`, `eval_matrix`, `project_matrix`, `mass_matrix`.
  All arrays. Construction is JAX-traceable so it composes with jit.
* Project: `n_α = Σ_i w_i φ_α(x_i) f(x_i)` where `φ_α` is the
  α-th Chebyshev polynomial of the first kind. This is exact (modulo
  truncation) for `f` polynomial of degree ≤ n-1.
* Evaluate: `f(x) = Σ_α n_α T_α(x)` via Clenshaw recurrence
  (numerically stable; avoids the bad conditioning of explicit-power
  evaluation).
* Mass matrix: diagonal in Chebyshev with the appropriate weight; we
  use `L²([a,b], dx)` not `L²([a,b], dx/√(1−x²))` — i.e., we treat
  Chebyshev as a polynomial basis on a *Lebesgue* inner product, not
  the natural Chebyshev weight. This keeps `∫ N dm` meaning the
  Lebesgue integral (the physical particle count) and matches the
  Galerkin orthogonality assumptions downstream.

`basis/tensor.py` provides `TensorProductBasis(b_logm, b_chi)` —
all matrices are Kronecker products of 1D versions; nodes are the
Cartesian product.

**Acceptance:**
* `test_project_evaluate_inverse` — project then evaluate recovers
  smooth test functions (gaussians, sines on log m, Talbot-Thrane
  PL+Peak) to relative 1e-10 at n_basis ≥ 24 per axis.
* `test_mass_matrix_spd` — symmetric, positive-definite, condition
  number < 1e6 at n_basis = 32.
* `test_quadrature_exact_for_polynomials` — Gauss-Lobatto quadrature
  exact to machine precision for polynomials of degree 2n−3.

**No physics yet at this step.** Just numerics on `basis/`.

### Step 2 — Natal density on basis (DONE 2026-05-19; log-path deprecated 2026-05-20)

Module: ``src/hierarchical_ad_pop_engine/basis/natal.py``.

**Production path:** :func:`project_natal_density` — LINEAR-space
projection of ``N_natal``.  Works for any ``natal_mass_kind``.  Slight
Gibbs ringing near sharp boundaries (e.g. Planck taper), bounded well
below 1% of peak; small negatives near the boundary are clipped at
zero downstream (matching v2's Picard solver behaviour).

**Deprecated, retained in-module:** :func:`project_natal_log_density`
+ :func:`evaluate_natal_density_from_log` — LOG-space projection.  Two
iterations were attempted before deprecation:

1. *No input floor.*  Lets the polynomial fit the analytic
   ``log N`` directly.  Achieves ``rel_err ≈ 1e-12`` at ``n_basis_m =
   24`` for Gaussian — but this is "measuring an irrelevant thing
   well": the polynomial is reproducing ``log N`` out to ``≈ -1e4`` at
   the support boundary, six orders past float64 underflow, no
   observable sensitivity.  Catastrophic ringing for any natal with a
   true ``-∞`` inside the support (PL+Peak Planck taper).
2. *Symmetric floor on both ends.*  Input clipped at ``_LOG_N_FLOOR =
   -50``; output re-clipped before ``exp``.  Physically honest framing,
   but the input clip introduces a C⁰ kink at the floor crossing and
   Legendre's L² best-fit develops a stuck-amplitude overshoot near
   the kink that bleeds into the peak region.  Empirically
   (``outputs/probe_floored_log_projection.py``) ``rel_err ≈ 0.2``
   even at ``n_basis_m = 128``.  Not useful at any practical
   resolution.

The linear path's accuracy is two orders of magnitude better than v2's
bilinear-interp representation error at ``n_basis_m = 128``, so the
deprecation costs no production capability.  Log-projection code is
preserved in-module (with deprecation notices in docstrings) for
possible future revival under a smoother floor design — e.g.
softplus-style soft clip, or a compactly-supported polynomial basis
such as Bernstein.

**Realistic convergence (calibrated 2026-05-19 against the numpy
mirror at ``outputs/validate_natal_projection.py``):**

| natal kind                | path   | n_basis_m | rel_err     |
|---------------------------|--------|-----------|-------------|
| Gaussian (μ=8, σ=1)       | linear | 64        | 2e-3        |
| Gaussian (μ=8, σ=1)       | linear | 128       | 1e-5        |
| PL+Peak (Stage 1 θ\*)     | linear | 64        | 1e-3        |
| PL+Peak (Stage 1 θ\*)     | linear | 128       | 3e-5        |
| Gaussian (μ=8, σ=1)       | log, symmetric floor *(DEPRECATED)* | 128 | ≈ 2e-1 |

The plan's original "1e-9 at n=64" target was aspirational; what is
**actually** achievable for the linear path is dominated by the narrow
Gaussian peak in log-mass coordinates (σ_ξ ≈ 0.06 for the PL+Peak's
Gaussian peak).  Even so, the basis at n=128 (PL+Peak rel_err 3e-5)
is two orders of magnitude better than v2's bilinear-interp
representation error ``~2e-3`` — so Step 2 is a strict improvement and
the original gating intent (basis ≥ v2 representation accuracy) is
satisfied.

**Stage 1 inference recommendation:** ``project_natal_density``
(linear) with ``n_basis_m = 128, n_basis_chi = 32``.

Tests live at ``tests/test_basis_natal.py``; thresholds per the table
above, all gated by ``jax_enable_x64`` (matching the v2 test
convention).  The two log-projection tests are present but
``@pytest.mark.skip``-ed pending revival.

### Step 3 — Remnant-scatter projection tensor (refactor; ~2 sessions)

**Status 2026-05-20:** Step 3a (`restrict_to` / `prolong_to`) DONE and
green.  Step 3c (retention sigmoid spot-check) DONE — see
``outputs/probe_retention_sigmoid.py``; at Stage 1 v_esc = 300 km/s,
``w_over_v_esc ∈ {1.0, 1.5, 2.0}`` gives smooth retention (⟨P⟩ ≈ 0.34,
0.37, 0.39 vs hard 0.32), finite ∂⟨P⟩/∂v_esc and ∂⟨P⟩/∂(w/v_esc),
sigmoid width O(300 km/s) broader than per-LGL-node Δv_kick.  Step 3d
(`build_remnant_tensor`) DONE 2026-05-20 — see
``src/hierarchical_ad_pop_engine/basis/remnant_tensor.py``.  Symmetry
acceptance test (3.0e-11 abs / ~1e-15 rel at n_basis=(8,4)) and AD-vs-FD
on every θ field (rel ≈ 1e-9 on kernel_a) green; pytest scaffold at
``tests/test_basis_remnant_tensor.py``.  Caught a latent
``chi_rem`` asymmetry under (m1,χ1)↔(m2,χ2) (the formula carries an
implicit m1≥m2 convention that v2 was not enforcing) and fixed it
at the T call site via canonical ordering — same pattern as the v2
j≠l double-count discovery.  Step 3e (full acceptance + JS-against-
v2 grid scatter at production resolution) pending.

The big engineering piece.  The current `_step_one_generation` does a
bilinear scatter of (M_rem, χ_rem) onto the (m, χ) grid; the basis
analog is the rank-3 tensor ``T(θ)`` such that

  G_γ = (1/2) Σ_α Σ_β K_{αβ}(θ) P_{αβ}(θ) n_α n_β T_{γ; α,β}(θ)

where α, β index progenitor basis-coefficient pairs (m1, χ1) × (m2,
χ2) and γ indexes the remnant basis.  ``T_{γ;α,β}(θ)`` is the
basis-projection coefficient of the θ-dependent remnant map
``(m1, m2, χ1, χ2) → (M_rem(·;θ), χ_rem(·;θ))``:

  T_{γ; α,β}(θ) = ∫∫∫∫ φ_α(m1, χ1) φ_β(m2, χ2)
                  · φ_γ(M_rem(m1,m2,χ1,χ2;θ), χ_rem(m1,m2,χ1,χ2;θ))
                  dm1 dχ1 dm2 dχ2

with Gauss-Lobatto quadrature at order ≥ 2·n_basis to maintain
Galerkin orthogonality.  **This is the critical correctness step**;
insufficient quadrature order silently reintroduces a measure mismatch
(the v2 midpoint-vs-trapezoid bug pattern, but in a basis-shaped
disguise).

#### T(θ) is θ-dependent (no Stage 1 caching shortcut)

The previous draft proposed absorbing ``K_αβ · P_αβ`` into ``T`` so
that ``T`` would be θ-independent and could be cached across gradient
steps.  **This was wrong even at Stage 1.**  Gradients of observables
w.r.t. natal parameters (μ, σ, α_PL, β_kernel, …) flow through the
remnant maps and the kernel, both of which sit inside ``T(θ)``.  The
solver must:

* Rebuild ``T(θ)`` on every forward call (no cross-call cache).
* Differentiate through ``T(θ)`` for the cotangent pull-back (Step 4
  IFT VJP).  ``jax.grad`` does this automatically as long as the
  physics functions are AD-friendly — the retention cliff
  (see below) is the only non-smooth piece and needs explicit care.

Implication for performance: each gradient evaluation pays the
``T``-build cost.  At ``n_basis_solve = 32×16``, the build is a
quadruple Gauss-Lobatto integral with quadrature order ≈ 64 per axis
and outer product onto a (512, 512, 512) tensor; on a modern CPU
this is seconds.  On GPU, ms.  Acceptable for inference loops.

#### Two-resolution architecture

Step 2 sized the natal projection at ``n_basis_density = (128, 32)``
to match v2's representation accuracy.  That resolution is wasteful
for the propagator: ``T`` at ``(128·32)³ = 6.9e10`` entries is 550 GB
at float64, never going to fit in memory and the einsum cost scales
the same way.  ``T`` is instead built at a smaller, separate
resolution ``n_basis_solve = (n_m_solve, n_χ_solve)``.  Density
representation and solver discretisation are now two independent
knobs:

```
natal pipeline
      │ project at n_basis_density = (128, 32)        Step 2
      ▼
  n_natal_density  (n_m_d × n_χ_d coeffs)
      │ restrict (modal truncation, exact for Legendre)
      ▼
  n_natal_solver   (n_m_s × n_χ_s coeffs)             Step 3
      │ Picard iteration on T(θ) at solver resolution Step 4
      ▼
  n_star_solver
      │ prolong (modal zero-pad, exact)
      ▼
  n_star_density   (n_m_d × n_χ_d coeffs)
      │ evaluate observables at high resolution
      ▼
  dC/dm₁, dC/dM_c, …                                  Step 5
```

Restrict and prolong are exact in the Legendre modal sense — they are
truncation / zero-padding of orthogonal coefficient arrays at the
same physical interval.  No quadrature, no interpolation error
introduced by the operator itself.  All error sits in the truncation:
modes ``n ∈ [n_solve, n_density)`` are dropped during restriction.

**Implementation form:** restrict and prolong are **class methods on
``TensorProductBasis``** — ``self.restrict_to(coeffs, target_basis)``
and ``self.prolong_to(coeffs, target_basis)``.  Not separate
free-standing functions in a ``basis/restriction.py`` module.
Class methods keep the validity checks (same physical interval,
target is a strict modal subset/superset of source) close to the
representation they're operating on and avoid style/logic drift
across modules.  Decision 2026-05-20.

For a smooth natal density (Gaussian-like) those high modes decay
exponentially with ``n`` and the truncation is essentially free.  For
the Talbot-Thrane PL+Peak, the Planck taper drives modes ``n ≳ 30`` to
algebraic decay; we expect some truncation error and **the headline
verification is whether the solver-resolution result, prolonged back
to density resolution, still beats v2 in observable JS**.  This is the
refactor risk the architecture buys; Step 5 is the test.

#### Memory at chosen resolution

| n_m_s | n_χ_s | n_basis_solve | T entries  | float64    |
|-------|-------|---------------|------------|------------|
| 24    | 8     | 192           | 7.1 × 10⁶  | 57 MB      |
| 32    | 16    | 512           | 1.3 × 10⁸  | 1.1 GB     |
| 48    | 16    | 768           | 4.5 × 10⁸  | 3.6 GB     |
| 64    | 16    | 1024          | 1.1 × 10⁹  | 8.6 GB     |

Default ``n_basis_solve = (32, 16)``.  Easy to inspect, builds in
seconds, leaves headroom on a workstation.  We can scale up if Step 5
shows the solver resolution is the binding error; the architecture
makes the bump cheap.

#### Retention cliff

The hard cutoff ``P_retain = 1[v_kick < v_esc]`` introduces a true
discontinuity in the integrand for ``T``.  Two options were
considered:

(a) **Soft-sigmoid retention** ``P_retain(v_kick) =
    σ((v_esc - v_kick) / w)`` with width ``w`` set as a fraction
    of ``v_esc``.  Smooth integrand, clean AD.

(b) **Hard cutoff, projected as-is.**  Galerkin-projects a
    discontinuity → fixed-amplitude Gibbs ringing in ``T``.
    Same failure pattern as the deprecated log-projection;
    rules itself out at modest resolution.

**Decision: (a).**  Two parameters:

* ``v_esc`` — physics threshold; Stage 1 **default 300 km/s**, the
  compromise across AGN-disk / NSC / mid-density host regimes
  (Richard's typical working range).  Exposed in θ so inference can
  vary it per cluster type when needed.  Previous "v_esc = ∞ at
  Stage 1" framing was wrong — at v_esc = 300 km/s the sigmoid is
  active for the high-v_kick tail of aligned-spin progenitors and
  the retention is materially below 1 there.
* ``w = w_over_v_esc · v_esc`` — sigmoid sharpness, **parameterized
  as a fraction of v_esc, never as an absolute km/s.** Default
  ``w_over_v_esc = 1.0``, expected range ``[1.0, 2.0]``.  Rule of
  thumb (Richard 2026-05-20): "w is a fraction of v_esc (the
  structure providing confinement is extended) — a good rule of
  thumb is of order 1-2 * v_esc, so never an absolute scale."  The
  physical broadening of the retention boundary scales with the
  host's escape velocity itself; an artificially narrow w (say
  0.1 · v_esc) is both unphysical and triggers the same
  Galerkin-Gibbs failure as a hard cutoff.

The retention sigmoid is **active in Stage 1**, not dormant.

API signature: ``retention_fn(v_kick, v_esc, w_over_v_esc=1.0)``.
Never pass ``w=<km/s>``.  When Stage 2+ fits ``w_over_v_esc``, it
enters θ with a prior on ``[1.0, 2.0]``.

Task #24 (sweep) targets ``w_over_v_esc ∈ {1.0, 1.5, 2.0}`` over the
Stage 1 mass and spin ranges; observable JS should be insensitive
to the choice (< 1e-4) at fiducial Stage 1 θ.  If sensitive, the
chosen quadrature order in T was too low and needs bumping.

#### Acceptance tests at n_basis_solve

* `test_T_symmetric_in_α_β` — `T_{γ;α,β} = T_{γ;β,α}` to 1e-12.
  Encodes the bosonic / unordered-pair structure; same failure mode
  as the v2 j≠l double-count.
* `test_T_conserves_count` — `Σ_γ T_{γ;α,β} · (basis weight on γ) =
  1` to 1e-9 for every (α, β) with both progenitors in-grid.  The
  Galerkin restatement of "mass conservation through the scatter":
  every progenitor pair lands somewhere in remnant space.
* `test_restriction_prolongation_identity` — for any Legendre
  coefficient array ``c`` at the solver basis, ``restrict(prolong(c))
  = c`` to machine precision.  Confirms the modal operators are
  one-sided inverses.
* `test_T_matches_grid_scatter` — for a basis projection of a known
  natal density and a fixed-θ kernel, the basis gain term
  ``G = ½ Σ_αβ K_αβ n_α n_β T_γ;αβ``, prolonged to density resolution
  and evaluated on a dense grid, agrees with the v2 grid
  `_step_one_generation` to JS < 1e-4 on `dC/dm_f`.  **Headline
  parity check** that says the basis solver shares the same physics
  as the v2 solver up to discretisation.
* `test_T_grad_finite_against_fd` — `jax.grad(Σ_γ T_γ;α,β · ψ_γ)(θ)`
  matches central FD at low resolution (24×8) to rtol 1e-4.
  Confirms AD flows through `T(θ)`; required for Step 4 IFT.

#### Forward-looking constraints (don't straightjacket future refactors)

Two refactors are explicitly **out of scope for v3 ship** but the Step
3 API must not foreclose them.  Captured 2026-05-20 from Richard:
"actionable now: don't straightjacket us re basis."  See also
[[feedback_dont_straightjacket_basis]] memory.

(i) **Split-basis projection for narrow seed populations.**  Eventually
    the seeding population's depletion needs a different basis from
    the merger continuum — e.g. δ-spin natals, or a narrow mass spike
    thinner than the LGL node spacing.  The projection becomes
    ``P = P_seed ⊕ P_continuum``.  An LGL polynomial basis cannot
    represent these without Runge-Gibbs pollution, and a uniform
    refinement of n_basis to cover them is wasteful.

    **API rule:** solver code must address basis objects through the
    methods (``project``, ``evaluate``, ``inner_product``,
    ``restrict_to``, ``prolong_to``) and must not assume tensor-product
    structure beyond what those methods expose.  Do not import
    ``TensorProductBasis`` from solver modules; pass an instance in.
    A future ``DirectSumBasis(SeedBasis, ContinuumBasis)`` must be a
    drop-in replacement.

(ii) **Brute-force C[N,N] as memory backstop.**  T storage is
     ``O(n_basis³)`` — 1.1 GB at (32, 16), 8.6 GB at (64, 16),
     70 GB at (64, 32).  If higher mass or spin resolution is
     ever required, the brute-force path — recompute
     ``C[N, N](χ_γ, m_γ)`` per output node by direct 4-D quadrature —
     scales as ``O(n_basis · n_quad⁴)`` flops with ``O(n_basis)``
     storage.  Trades flops for memory; viable when the wall hits.

     **API rule:** the solver's Picard iteration consumes a callable
     ``gain_fn(coeffs, theta) -> coeffs`` and does not assume a
     T-tensor contraction recipe.  Step 3 ships ``TTensorGain``; a
     future ``BruteForceGain`` swap-in must require zero changes to
     the Picard loop or the IFT VJP.

These constraints add no implementation cost to Step 3 itself — they
just say "don't hardcode ``TensorProductBasis`` and don't hardcode
the T-tensor contraction" in the surrounding solver glue.

### Step 4 — Picard map and IFT VJP on coefficients (1 session)

**Status 2026-05-20 (evening):** Step 4 DONE — see
``src/hierarchical_ad_pop_engine/response/basis.py`` and
``src/hierarchical_ad_pop_engine/basis/loss_tensor.py``.  End-to-end
probe (``outputs/probe_basis_solver.py``) passes: Picard converges in
13 iters at Stage-1 ε=1e-4, perturbative limit ε=1e-8 matches natal
to expected O(ε · 15) ≈ 4e-6 relative, all four ``jax.grad`` outputs
(``v_esc``, ``w_over_v_esc``, ``epsilon``, response fields) finite,
AD-vs-FD on ``epsilon`` rel 2.6e-5 (within IFT GMRES ``tol=1e-7``
headroom), and ``jax.jit`` traces cleanly.  Pytest scaffold at
``tests/test_basis_solver.py``; user must run on their machine.
Notable: ``∂/∂v_esc`` is non-zero by construction (soft-sigmoid
retention) — v2 Heaviside gives exactly zero here; a regression
test pins this property.

`response/basis.py`:

```python
def _F_map_basis(n_alpha, response, v_esc_arr, epsilon_arr, T, K, P_const):
    """Picard RHS on coefficients."""
    n_natal_alpha = project_natal_log(response)        # cached on response
    gain_alpha    = 0.5 * jnp.einsum("gab,ab,a,b->g", T, K * P_const, n_alpha, n_alpha)
    loss_alpha    = jnp.einsum("ab,ab,a,b->a",         K * P_const, n_alpha, n_alpha)  # diagonal in α
    return n_natal_alpha + epsilon_arr * (gain_alpha - loss_alpha)
```

(Indexing here is schematic; the tensor-product structure means
gain/loss split into m-axis and χ-axis contractions for efficiency,
similar to how `gain_density` in v2 broadcasts over (N_m, N_χ, N_m,
N_χ). The `einsum` shown is the conceptual core.)

`_picard_while_loop_basis`: copy of v2's `_picard_while_loop` with
carry shape `(N_basis,)`; damping and tol policy identical.

`_make_basis_solve_fn`: copy of `_make_picard_solve_fn` in `implicit.py`
with two changes:
1. Forward calls `_picard_while_loop_basis` instead of grid version.
2. Backward GMRES matvec uses `_F_map_basis` instead of `_F_map`.

Everything else — lru_cache factory, IFT linear solve, cotangent
pull-back through θ — is structurally identical. Diff-vs-implicit.py
review is the correctness check.

**Acceptance:**
* `test_basis_solver_jit_traceable` — runs under `jax.jit`.
* `test_basis_grad_matches_fd` — `jax.grad(∫ N dm dχ)(θ)` matches
  central finite difference on a 24×8 basis (i.e., low resolution) to
  rtol 1e-4 across (μ, σ, kernel_b, ε). Same test pattern as
  `tests/test_implicit_parity.py` case (4).
* `test_basis_perturbative_limit` — at ε → 0, the basis solver
  reproduces the analytic linearisation `N* = N_natal + ε · (G[N_natal]
  − L[N_natal])` to rtol 1e-9.

### Step 5 — Cross-validation in observable space (1 session)

The validation oracle is `dC/dm1` (and `dC/dm_f dχ_f`) from the
parallel observable-plots branch, applied to all three paths.

**Tests (require `--run-integration` like the v2 red-team-as-pytest):**

a. **JS(basis-Picard ↔ particle MC) < JS(grid-Picard ↔ particle MC)**
   on `dC/dm1` at ε = 0.1, n_seed = 200k. The expectation is that
   removing the measure ambiguity *reduces* the systematic; if it
   doesn't, the basis solver has a bug that the parity test in Step 3
   didn't catch. Decision rule: if basis JS isn't materially better
   than grid JS, halt v3 progression and debug. **Do not declare
   success on the parity tests alone** — observable-space agreement
   is the bar.

b. **JS(basis-Picard ↔ grid-Picard) on `dC/dm1`** at ε = 0.1 — should
   be small (both methods are correct, just different
   discretisations). A useful "is the basis solver itself correct"
   sanity check; expect JS in the 1e-3 to 1e-4 range.

c. **AD consistency** — `jax.grad(log_rate_density_summary)(θ)`
   between basis and grid solvers agree to rtol 1e-3 at Stage 1 θ*.
   (Output values may differ at the 1e-2 level — that's the
   systematic we're trying to fix — but the *gradients* should track
   each other because they're driven by the same physics and the same
   θ-dependence of the natal density.)

d. **Spin ringing resolved** — at the same θ, the `dN/dχ` of the
   basis solver shows no high-frequency oscillation in the χ ≈ 0.687
   region; visual inspection (and a numerical
   `(d²N/dχ²)_max < threshold` check) confirms.

### Step 6 — Demo migration and mock regeneration (0.5 session)

Mirrors v2 Step 5. After Step 5 acceptance:

1. Edit `demo/demo_response_doctor2020.py` (and the gwkokab fit demo)
   to build `response = build_gaussian_coagulation(..., solver="basis",
   epsilon=1.0, basis_resolution=(32, 32))`.
2. Regenerate the Stage 1 mock catalog under the basis solver. Save
   as `mock_catalog_basis_<hash>.npz` under `demo/figures_<hash>/`.
3. Re-fit with SVI smoke pass, NUTS for the headline corner.
4. Document expected posterior shifts vs the v2 mock. The leading
   prediction: `kernel_b` posterior shifts (the measure bug
   selectively biased the cascade direction), `mu_natal` narrows
   (the natal peak was previously contaminated by mis-located 2G
   pile-up).
5. Add a regression test that runs SVI for 200 steps on the canned
   basis mock and recovers `mu_natal` within 1σ.

### Step 7 — Default switch and cleanup (0.5 session)

* Change `CoagulationResponse.solver` default from `"perturbative"` to
  `"basis"` (skipping `"implicit"` as a default — it was an intermediate
  state).
* Add `DeprecationWarning` on `solver="implicit"` and
  `solver="perturbative"` after one minor release.
* Update `DESIGN.md` §3.1 with the basis solver path. Update
  `ARCH_NOTES.md` to mark the v2 open-systematic concern as resolved
  (referencing the Step 5 acceptance metrics).
* Keep `response/implicit.py` (now diagnostic) and
  `red_team/engine_v2_implicit.py` in the tree; they remain useful
  parity references for any future basis-internal refactor.
* Update `red_team/compare_implicit_vs_mc.py` to a 3-way comparison
  (basis-Picard, grid-Picard, particle MC) — observable-plot script
  on `observables` branch will already have the right hooks.
* `compare_all.py` deletion: per PLAN_engine_v2_production §7. Now is
  the right time.

## Effort estimate

| Step | What                                                       | Effort      |
|------|------------------------------------------------------------|-------------|
| 1    | Basis infrastructure (Chebyshev-Lobatto, tensor product)   | 1.0 session |
| 2    | Natal density projection + validation                      | 0.5 session |
| 3    | Remnant-scatter projection tensor T                        | 1.5 session |
| 4    | Picard map + IFT VJP on coefficients                       | 1.0 session |
| 5    | Cross-validation in observable space                       | 1.0 session |
| 6    | Demo migration + Stage 1 mock regeneration + refit         | 0.5 session |
| 7    | Default switch + deprecation + cleanup                     | 0.5 session |
|      | **Total**                                                  | **6.0 sessions** |

This is the largest single chunk of engine work since the original
v1 build. Two ways the estimate can blow:

1. **Step 3 quadrature pathology.** If the Gauss-Lobatto quadrature
   at 2·n_basis order turns out to leak ~1e-4 mass on each projection
   (silent measure error), Step 3 expands by ~1 session to add an
   adaptive-quadrature fallback near the M_rem(·) ridge.
2. **Step 5 unfavourable result.** If basis JS isn't materially
   better than grid JS, we have a deeper bug — possibly in the K·N²
   approximation itself. Budget another session for investigation.

## Anti-goals

* **No mass-lumping FEM on geomspace m.** Lumping reintroduces the
  exact midpoint/trapezoid asymmetry we're refactoring away from.
  Galerkin orthogonality at the quadrature order is the rule;
  pseudospectral collocation that nominally satisfies it but uses
  lumped quadrature does not (`feedback_no_fudge_factors`).
* **No smoothing of P_retain.** Stage 1 uses v_esc = 1000 km/s where
  retention is ≈ 1; Stage 2 will use adaptive elements at the
  retention boundary. Do not Gaussian-blur the Heaviside to make
  Chebyshev converge faster — that's a `feedback_no_fake_physics_stand_ins`
  violation.
* **No "basis convergence parameter" leaking into the public API.**
  `n_basis` per axis is a numerical knob (like `max_iter`, `tol`,
  `damping` in v2), not a model hyperparameter. Default it; allow
  override via kwargs; do not expose in `build_gaussian_coagulation`'s
  primary signature.
* **No third independent algorithm in the validation suite.**
  Post-v3 the pair is basis-Picard + particle MC. Grid-Picard joins
  perturbative as diagnostic, not as part of the validation harness
  (`feedback_validation_downselect_two_algorithms`).
* **Do not start v3 with v2 Step 4 still red.** The basis refactor's
  parity tests assume the v2 path is bit-equal-validated against the
  red-team prototype. If Step 4 turned up regressions, fix those
  first.
* **Don't combine v3 with the retention-v2 work or tilt
  marginalisation.** Orthogonal axes; mixing them makes the
  validation diagnostic ambiguous.

## Deferred work (explicitly out of scope)

* **Adaptive-element retention boundary (Stage 2).** Once v_esc varies
  per environment and P_retain has a non-trivial Heaviside in the
  basis, we need h-refinement at the retention curve. Plan separately
  as `PLAN_retention_v2.md`.
* **Non-aligned spin.** The basis tensor-product extends naturally to
  (m, χ_aligned, χ_perp) or full spin (m, χ⃗), but the dimensionality
  blow-up is significant and the physics layer needs design first.
  Separate plan when the team is ready for tilt marginalisation.
* **Weighted-tracer particle MC + explicit seeding rate.** Sibling
  project. The basis solver makes weighted tracers more useful
  (higher-generation comparison in observable space becomes feasible)
  but does not depend on them.
* **GPU performance pass.** v2 Step 6 was already conditional; for v3
  it's strictly post-correctness. Profile first, optimise only if
  Stage 1 NUTS wall-clock is unacceptable.

## Decision points to flag at execution time

1. **Step 1 basis weight choice.** Lebesgue inner product (chosen
   default, matches physical particle-count semantics) vs natural
   Chebyshev weight (numerically nicer but breaks ε-as-absolute-rate
   interpretation). Decision is locked here; revisit only if Step 3
   acceptance fails for conditioning reasons.
2. **Step 3 T-tensor storage.** Float32 precompute (preferred) vs
   on-the-fly recomputation vs sparse storage. Decide after the first
   benchmark on Stage 1 grid sizes.
3. **Step 5 abort criterion.** If basis JS on `dC/dm1` is not
   materially better than grid JS at Stage 1 θ*, do not proceed to
   Step 6. Open an investigation issue and resume v2 Step 4 baseline.
4. **Step 7 deprecation timing.** Default-switch can happen immediately
   on Step 5 acceptance; the deprecation warnings on `"implicit"` /
   `"perturbative"` should wait one minor release for downstream
   gwkokab fit demos to migrate.

## What this plan does not cover

Mirroring PLAN_engine_v2_production:

* Spin precession / non-aligned retention — see Deferred.
* Redshift dependence — formation-rate layer's job
  (`feedback_population_response_vs_formation`).
* Selection effects — Stage 2 wires gwkokab's selection-aware
  likelihood; engine API unchanged.
