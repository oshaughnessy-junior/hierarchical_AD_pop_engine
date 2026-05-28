# PLAN — M3 stratified natal + weighted MCEvaluator

Written 2026-05-28.  Locks the design for Session M3 of
``demo/PLAN_mc_first_reprioritization.md``: tighten M2's 20 % parity
gate back to the PLAN's headline 10 % by giving the MC path importance-
weighted natal sampling and a weighted k-NN balloon estimator.

Per-session scope discipline (matching F1 / F2 / M1 / M2): this
session lands the *evaluator* side cohesively + sandbox tests + a
PLAN-locked design for the driver side.  A follow-up **Session M3b**
ships the driver-side weighting + the workstation integration test
that joins them.  The split keeps each session's commit set
sandbox-verifiable end-to-end on the Cowork VM where the engine /
JAX dependency chain is unavailable.

## Why split M3 into M3a + M3b

* **M3a** (this session) — weighted `MCEvaluator` only.  Pure numpy,
  sandbox-verifiable in full, no JAX dependency.  Bit-for-bit invariant:
  ``weights=None`` reproduces Step-3 behaviour exactly.
* **M3b** (follow-up) — `MCConfig.proposal` + `MCConfig.stratify` +
  weight propagation rule in the merger loop + workstation parity
  test.  Requires JAX (for `_per_event_remnant_misaligned`) + scipy
  truncnorm + the engine for the rel-L1 comparison gate; not
  sandbox-verifiable in the same way.

Each half stays under one session's complexity budget and the M3a
deliverable de-risks M3b by pinning the weighted-density estimator
in code before the driver-side bookkeeping lands.

## Hard constraints (inherited from
[[PLAN_mc_first_reprioritization]])

* `[[feedback-no-fake-physics-stand-ins]]` — weights live in
  density-estimation bookkeeping, not in physics primitives; the
  per-event vector BR + LZ retention path is untouched.
* `[[feedback-no-fudge-factors]]` — the M3b acceptance gate
  (2× implicit's rel-L1) is the asymptotic Monte-Carlo SE at the
  chosen ``N_init``, not a tuned threshold.  Bandwidth selection
  (still k-NN balloon per `[[feedback-mise-kernel-not-the-lever]]`)
  uses the same Silverman √n rule as Step 3 — adapted to use
  effective sample size ``n_eff = (Σ w)² / Σ w²``.
* `[[feedback-mc-depletion-tracks-engine]]` — depletion fraction
  continues to track engine ``f_natal_depleted``; the stratified
  proposal scope is *natal* sampling, not stopping criterion.
* `[[feedback-no-frozen-test-references]]` — every parity gate is
  a weighted-vs-uniform invariant or engine-vs-MC rel-L1; no
  pinned `.npz` baselines.
* `[[feedback-dispatch-callgraph-audit]]` — every new kwarg gets a
  grep across the codebase before defaults flip.

## M3a — weighted MCEvaluator (this session)

### Algebra

Loftsgaarden-Quesenberry weighted variant (cf. Silverman 1986 §5.4
Eq. 5.16 generalisation):

  ``f̂(x) = (1 / (W · h(x)^d)) · Σ_{i ∈ N_k(x)} w_i · K_b(d_i / h(x))``

with

* ``W = Σ_i w_i`` — total weight,
* ``h(x)`` = distance to the k-th nearest neighbour by sample index
  (NOT by cumulative weight — count-based k preserves the Step-3
  bandwidth-selection asymptotics, and the weighted normalisation
  inside the ball does the heavy lifting),
* ``K_b(u) = (3/π)(1 − u²)²`` — biweight kernel (unchanged).

Unweighted reduction: with all ``w_i = c`` constant, ``W = n·c``
and the ``c`` cancels in the ratio.  So ``weights=None`` ⇒
``c = 1`` ⇒ bit-for-bit Step-3 output.

### Boundary handling

Reflection-augmentation of the sample set (Step-3) carries over
unchanged: each reflected copy inherits the same weight as the
original.  The `cKDTree` is built over the augmented (sample +
reflection) set; the weight array passed to the kernel sum is
broadcast across the three copies.

### Effective sample count for the default-k heuristic

Step-3 picks ``k = max(4, round(√n))``.  Under non-uniform weights
the relevant scale is the effective sample size:

  ``n_eff = (Σ w_i)² / Σ w_i²``

(Kish 1965 design effect; coincides with ``n`` for uniform weights).
Default ``k = max(4, round(√n_eff))``.  Explicit ``k`` overrides
unchanged.

### Surface

```python
class MCEvaluator:
    def __init__(
        self,
        m_active, chi_active, mergers,
        *,
        log_R_env, kernel_params,
        k: int | None = None,
        reflect_chi: bool = True,
        weights: ArrayLike | None = None,    # NEW
    ) -> None:
        ...
```

* ``weights=None`` ⇒ ``np.ones(n_active)`` internally ⇒ Step-3
  bit-for-bit.
* ``weights.shape == (n_active,)`` required; ValueError otherwise
  (matching the existing ``m_active`` / ``chi_active`` shape check).
* All weights must be ``> 0`` (importance weights are positive by
  construction).  Validation in `__init__`.
* Stored on the instance as ``self.weights``; broadcast to the
  reflection-augmented sample set inside
  ``_build_density_tree`` and passed to ``_knn_balloon_2d``.

```python
def _knn_balloon_2d(
    tree, n_samples, query_pts, k,
    *,
    sample_weights: np.ndarray | None = None,    # NEW
) -> np.ndarray:
    ...
```

* ``sample_weights=None`` ⇒ uniform (Step-3 bit-for-bit).  Stored
  on the tree's data order so kernel-sum lookups are
  ``sample_weights[idx]`` for each query's k-NN indices.
* The biweight kernel sum becomes ``Σ w_i · (1−u²)²``; the
  denominator becomes ``W · h(x)^d`` (with ``W = Σ w_i`` total —
  passed by the caller).
* Returns same shape ``(Q,)``.

### Sandbox-verifiable invariants

1. **Bit-for-bit on uniform weights.**  Call ``MCEvaluator`` twice
   with identical samples, ``weights=None`` vs
   ``weights=np.ones(n)``; ``density(m, χ)`` outputs equal at
   machine precision.
2. **Constant-weight invariance.**  ``weights=np.full(n, 7.0)``
   gives identical density to ``weights=None``.
3. **Importance-recovery sanity.**  Draw samples from
   ``q_proposal`` (e.g. uniform), set ``w_i = p_target(x_i) /
   q_proposal(x_i)``, evaluate density on a grid — recovered
   density approximates the *target* density even though the
   sampling was from the proposal.  Gate: rel-L1 vs analytic
   target ≤ Monte-Carlo SE at the chosen N.
4. **Effective-sample-count default-k.**  For ``weights = [1, 1,
   ..., 1, 1000]`` (one outlier), ``n_eff`` ≈ 1, so default
   ``k = max(4, round(√1)) = 4``.  Verify ``self.k = 4`` from
   the constructor.
5. **Weight broadcasting through reflection.**  ``weights[i]`` is
   passed at every k-NN query containing sample ``i`` or its
   reflections.  Verify by computing density at a boundary-near
   χ — confirm the boundary density matches the unweighted-with-
   same-samples case to within float roundoff (since uniform weights
   trivially broadcast).
6. **W normalisation.**  ``W = Σ w_i`` (NOT weighted to include
   reflections); the ``(3/π) · Σ w_i (1−u²)² / (W · r²)``
   asymptotics match the unweighted ``(3/π) · k / (n · r²)`` for
   uniform weights.

### Test file

``tests/test_evaluation_mc_weighted.py`` — new sandbox-tier file.
Six invariants above as ``pytest`` functions; no engine, no JAX,
``@pytest.mark.fast``.  Mirrors the structure of
``tests/test_evaluation_mc.py`` (Step 3).

### Out-of-scope for M3a (deferred to M3b)

* MC-driver weight initialization (``MCConfig.proposal``).
* Per-merger weight propagation rule.
* Engine-vs-weighted-MC rel-L1 gate (the headline acceptance).
* Workstation-only pytest using the engine.

## M3b — driver-side weighting + workstation gate (follow-up)

Spec'd here so M3a's design intent is clear; implementation
deferred.

### Weight propagation rule (LOCKED)

Each particle ``i`` carries weight ``w_i``.  On accepted merger of
``(i, j)``:

* **Heavier slot inherits descendant weight** (cosmetic m_1 ≥ m_2
  preservation matches the existing m/χ update at lines 414-417 of
  ``red_team/backup_particle_mc.py``).
* **Descendant weight** ``w_desc = (w_i + w_j) / 2`` — arithmetic
  mean of the parent weights.

Why arithmetic mean and not minimum-consumption (Babovsky 1984
style)?

* Bit-for-bit invariant under uniform weights: ``w_i = w_j = 1`` ⇒
  ``w_desc = 1`` ⇒ all subsequent dynamics match the unweighted v1
  cascade exactly.
* Total weight conservation per merger: 2 progenitors of total
  ``w_i + w_j`` → 1 descendant of weight ``w_desc + ??``.  This is
  the catch — strict total-weight conservation needs
  ``w_desc = w_i + w_j``.  The arithmetic-mean choice instead
  conserves *per-particle weight scale* (i.e. each particle
  continues to represent "one bin's worth" of probability mass) at
  the cost of total weight decay.
* Practical implication: the *effective* sample size after K
  mergers is ``n_eff ≈ n_init - K`` (not the partial decay you'd
  get with min-consumption).  This is the same accounting as the
  unweighted MC and gives the cleanest M3a / M3b unit-test bridge.

### MCConfig surface

```python
@dataclass
class MCConfig:
    ...                                         # existing fields
    proposal: Callable | None = None            # NEW: q(m, χ) or None
    stratify: tuple[int, int] | None = None     # NEW: (n_m_strata, n_chi_strata)
                                                 # for stratified sampling
    weight_array: np.ndarray | None = None      # NEW: explicit override
```

* ``proposal=None`` ⇒ existing ``sample_natals`` (truncnorm),
  ``weight_array=None`` ⇒ ``np.ones(n_seed)`` ⇒ v1 bit-for-bit.
* ``proposal`` is a callable returning samples + log-pdf at those
  samples; the driver computes ``w_i = p_natal(x_i) /
  q_proposal(x_i)`` automatically.
* ``stratify=(n_m, n_chi)`` activates a stratified-sampler
  proposal: divide ``(log m, χ)`` into ``n_m × n_chi`` cells, draw
  ``n_seed / (n_m × n_chi)`` particles per cell, weights set
  automatically from the per-cell area Jacobian.

### Workstation gate (test_mc_stratified_natal.py)

* Fixture: implicit solve at ``θ*`` + MC at ``N_init = 10^4`` with
  stratified ``(8, 16)`` cells over the engine grid range +
  unweighted MC at the same N_init.
* Gate: ``rel_L1(MCEvaluator_weighted, engine) ≤
  2 · rel_L1(MCEvaluator_unweighted, engine)`` — stratified must
  not be more than 2× worse than uniform.
* Headline gate: at ``N_init = 10^4`` and σ_χ = 0.02 (the narrow-
  natal-peak fixture from `[[PLAN_mc_first_reprioritization]]`
  Session M3), stratified rel-L1 vs engine ≤ 10 % (PLAN's headline
  target).

### Hard gotchas already known

* **Weighted-sample tree.**  ``scipy.spatial.cKDTree`` doesn't carry
  per-point payloads, so M3a stores weights as a separate
  ``self.weights`` array indexed by the tree's insertion order.
  The reflection-augmented tree has 3 × n_active entries; the
  weight array indexed in the same order.
* **Stratified proposal at the engine grid edges.**  ``log m``
  near ``m_min`` and ``m_max`` requires the proposal to have
  support in the truncated region.  Simplest: stratification cells
  cover ``[log(m_min), log(m_max)] × [0, 1]`` exactly; proposal
  density per cell is uniform on the cell.
* **PRNG sequence preservation.**  When ``proposal=None`` the v1
  ``rng.truncnorm`` draws stay in the same sequence position so the
  pair-rejection loop continues bit-for-bit.

## Sequencing this session (M3a)

1. Write this PLAN doc.
2. Add ``weights`` parameter to ``MCEvaluator`` + private
   ``_knn_balloon_2d`` helper.  Bit-for-bit on uniform-weight
   defaults; explicit weight validation in ``__init__``.
3. Write ``tests/test_evaluation_mc_weighted.py`` covering the six
   sandbox invariants above.
4. Sandbox-verify the test file runs ``@pytest.mark.fast``-marked
   tier green from the Cowork VM (numpy only).
5. Commit M3a as a single atomic commit on ``observables``.
6. Update memory with session handoff + M3b scoping.

## Related

* `[[PLAN_mc_first_reprioritization]]` — parent PLAN for the
  MC-as-inference workstream; this is its Session M3.
* `[[project-evaluation-step3-done]]` — Step-3 MCEvaluator that M3a
  generalises.
* `[[project-engine-m2-done]]` — M2 moved MCEvaluator to
  ``src/.../evaluation/mc.py``; this PLAN edits that module.
* `[[feedback-mise-kernel-not-the-lever]]` — why we stay on
  biweight + k-NN bandwidth and change the *weighting regime*
  to fix the cascade-resolution shortfall.
* `[[feedback-no-fake-physics-stand-ins]]` /
  `[[feedback-no-fudge-factors]]` /
  `[[feedback-mc-depletion-tracks-engine]]` /
  `[[feedback-no-frozen-test-references]]` /
  `[[feedback-dispatch-callgraph-audit]]` — the cross-cutting
  constraints honoured throughout.
