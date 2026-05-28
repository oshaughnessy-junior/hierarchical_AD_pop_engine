# PLAN — JAX-pure MC rewrite for AD-traceable inference

Written 2026-05-28 night in response to the user's M4 reality-check
(``[[project-session-handoff-20260528]]`` evening Q&A): SVI under
``solver="mc"`` does not work because the numpy/scipy particle MC is
not JAX-traceable, so gradients w.r.t. ``θ_pop`` cannot flow through
the MC retrigger.  Fix is to rebuild the MC in JAX so AD is available
end-to-end.

This PLAN locks the design and stages the rewrite into atomic
J-sessions (``J1`` → ``J4``) so each is a single-session commit with
its own sandbox tier.  Closes out the M3 work cleanly and unblocks
M4 once J4 lands.

## Hard rule — numpy MC stays as a parallel implementation

**The existing numpy/scipy MC (``red_team/backup_particle_mc.py`` +
``src/.../evaluation/mc.py`` + ``src/.../response/mc.py``) is NOT
deleted or replaced in-place.**  It remains alongside the JAX
rewrite, in three roles:

1. **Validation backstop.**  Every J-session ships a sandbox-tier
   parity test that the JAX path agrees with the numpy path on
   distributions (KS test for sample paths, rel-L1 on summary
   statistics).  Per ``[[feedback-red-team-validation]]``, the
   parallel implementation IS the validation — independent code
   paths sharing only the lowest-level physics primitives.
2. **Bit-for-bit-on-defaults regression guard.**  The numpy
   ``sample_natals`` / ``particle_mc`` / ``MCEvaluator`` continue
   to satisfy the existing M3a/M3b tests bit-for-bit.  J-series
   commits CANNOT touch the numpy code paths in ways that break
   those tests.
3. **Engineering backstop.**  If a J-session reveals a design
   flaw in the JAX path (e.g. ``J2``'s reparametrised K-rejection
   has unacceptable bias), the numpy path is still production-
   ready for non-AD use cases (red-team validation,
   forward-modelling demos, the Session D parity battery).

Per ``[[feedback-perturbative-unreliable-implicit-production]]``
(promoted-by-analogy to MC here): JAX MC ships **additively**, not
in-place.  The default ``solver="mc"`` path keeps using the numpy
MC for now; a future commit can flip the default once the JAX
path passes the full M4 integration gate.

## Hard constraints honoured

Carry-forward from
``[[demo/PLAN_mc_first_reprioritization]]`` and
``[[demo/PLAN_m3_stratified_natal]]``:

* `[[feedback-no-fake-physics-stand-ins]]` — JAX MC runs the same
  vector BR + LZ + per-event retention as numpy; no smoothed
  surrogates outside the K-rejection (see § K-rejection design).
* `[[feedback-no-fudge-factors]]` — every JAX-side knob
  (bandwidth, soft-K temperature if any) is set from data or
  analytic convergence, not tuned to a target.
* `[[feedback-red-team-validation]]` — keeping the numpy MC as
  the validation parallel is non-negotiable.
* `[[feedback-dispatch-callgraph-audit]]` — every new function
  surface gets grepped through callers before any default flip.
* `[[feedback-no-frozen-test-references]]` — all parity gates are
  numpy-vs-JAX relationships, not pinned ``.npz`` baselines.

## End-state architecture

```
gwkokab.PoissonLikelihood
    ↓ log_prob(θ_event)
distributions.CoagulationPopulation
    ↓ log_rate_density(...)
universe.SyntheticUniverse
    ↓
response/coagulation.CoagulationResponse  ← solver="implicit" | "basis" | "mc_numpy" | "mc_jax"
                                              The two MC paths produce equivalent
                                              MCResponses; "mc_numpy" routes through
                                              the existing static-numpy stash;
                                              "mc_jax" routes through JAX arrays
                                              all the way to the evaluator.
                            ↓
                       evaluation/mc.MCEvaluator (numpy) — Step-3/M3a
                       evaluation/mc_jax.MCEvaluatorJax (NEW) — J3
                            ↓ density / rate_density_4d / total_rate / log_rate_density
```

The two paths share the M3a weighting API (``weights`` kwarg) at
both the driver and evaluator levels; the only difference is the
backing array type and the bandwidth-selector implementation
(fixed-bandwidth Gaussian KDE in JAX vs k-NN balloon in numpy —
see § J3 § "Bandwidth selector trade").

## Sessions

### J1 — JAX sample_natals + helpers

Goal: rewrite ``sample_natals`` + ``_natal_log_pdf`` +
``_stratified_natal`` in JAX, **adding** them alongside the numpy
versions (do not touch the numpy code path).

Surface (in ``red_team/backup_particle_mc.py``):

```python
def sample_natals_jax(cfg: MCConfig, key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JAX version of sample_natals.  Same MCConfig surface (proposal,
    stratify, weight_array all honoured).  Returns JAX arrays.
    Differentiable in cfg.mu_natal, sigma_natal, natal_spin_loc,
    natal_spin_scale via the truncnorm reparametrisation
    (jax.random.truncated_normal samples are reparametrised by
    construction)."""
```

Reparametrisation: ``jax.random.truncated_normal`` returns standard-
truncated samples that we shift+scale by ``μ`` and ``σ``.  This is
the standard score-function reparametrisation — gradients in ``μ``
and ``σ`` flow directly through the affine transform.

Stratified-uniform path: JAX equivalent via
``jax.random.uniform`` per cell.  No stratification gradient (the
cells themselves are static).

Proposal path: caller-supplied ``q(key) → (m, χ, log_q)`` —
identical signature shape to numpy proposal, just JAX rngs.

Weights: same logic as numpy — ``w = exp(log p_natal - log q)``,
with the zero-weight filter at the sampler boundary.  Filtered set
has the same M3a-strictly-positive contract.

Tests (``tests/test_mc_sample_natals_jax.py``, sandbox tier):

* **Distributional equivalence with numpy** — KS test on independent
  draws of ``m`` and ``χ`` from ``sample_natals`` vs
  ``sample_natals_jax`` at the same θ; both should have the same
  truncnorm CDFs.
* **Bit-for-bit on weight contract** — JAX uniform-path weights all
  equal 1.0.
* **Stratified-IS recovery** — ``Σ w_jax ≈ n_drawn`` and ``⟨m⟩``
  matches truncnorm mean within MC SE (parallel of the numpy
  INV-3).
* **Validation errors** — negative ``weight_array`` and non-finite
  proposal ``log_q`` raise (matching numpy validation).
* **AD-through-natal-params** —
  ``jax.grad(lambda mu: jnp.sum(sample_natals_jax(cfg.replace(mu_natal=mu), key)[0]))(8.0)``
  is finite and matches central finite-differences on the same
  function.

Out of scope for J1: the ``particle_mc`` loop itself.

Effort: 1 session.

### J2 — JAX particle_mc (the hot loop)

Goal: rewrite ``particle_mc`` as a JAX-traceable function using
``jax.lax.while_loop`` over a functional state pytree.

**K-rejection design (LOCKED — option (c) from the
2026-05-28-evening AskUserQuestion):**

Replace the discrete accept/reject Bernoulli with an
**importance-weighted continuous accept**.  Each candidate pair
``(i, j)`` gets a weight increment proportional to
``K(m_i, m_j) / K_max`` rather than being accepted with that
probability.  Mergers happen at every step (no rejection); the
weight bookkeeping ensures the *effective* merger rate matches the
discrete K-rejection version in expectation.

Why this beats Gumbel-softmax:

* **No relaxation bias.**  Gumbel-softmax introduces a temperature
  knob that biases the dynamics; importance weighting is exact in
  expectation.
* **Plays naturally with M3a/M3b weights.**  The M3 weight
  infrastructure already handles per-particle weights;
  K-rejection-as-weighting is the natural extension (the descendant
  weight becomes ``w_desc · K(m_i, m_j) / K_max`` instead of
  ``0.5(w_i + w_j)``).
* **AD-friendly.**  Smooth gradient w.r.t. ``K_max`` and the kernel
  parameters; no discrete switching.

Open question (TBD in J2): does this preserve the existing
sample-path statistics or does it require a corrective
normalisation?  Sandbox-side derivation + KS test vs numpy MC at
the same effective ε.

Other JAX-loop concerns:

* **Variable-length cascade**: replace the
  ``while_loop`` with a fixed step budget (e.g. ``2 * n_init``
  inner iterations) and mask out post-termination steps.  Cleaner
  for JIT than a true while-loop predicate.
* **Active-mask state**: use a boolean array carried through
  ``lax.scan`` state.  Slot replacement becomes
  ``jnp.where(swap_mask, new_value, old_value)``.
* **Random key splitting**: ``jax.random.split`` per inner step;
  master key carried through the scan state.
* **Per-event vector physics** (already JAX in the numpy code via
  ``_per_event_remnant_misaligned``): wraps unchanged.

Tests (``tests/test_mc_particle_mc_jax.py``, sandbox tier):

* **Distributional parity with numpy MC** — at the same effective ε,
  the JAX MC's surviving-pool ``(m_active, χ_active)`` distribution
  matches numpy MC's within MC SE (KS test on independent draws).
* **Weight propagation** — under uniform initial weights, the
  weight bookkeeping degenerates to the unweighted case AND the
  K-rejection-as-weighting reduces to the discrete case in
  expectation.
* **AD-through-kernel** — gradient of ``jnp.sum(m_active)`` w.r.t.
  ``kernel_b`` is finite and non-zero (the cascade depends on the
  pairing kernel).

Effort: 1-2 sessions.  J2 is the highest-risk piece because the
K-rejection-as-weighting design needs derivation + numerical
validation.

### J3 — JAX MCEvaluator

Goal: build ``MCEvaluatorJax`` in
``src/hierarchical_ad_pop_engine/evaluation/mc_jax.py`` alongside
the existing numpy ``MCEvaluator``.

**Bandwidth selector trade (LOCKED):**

Replace the k-NN balloon estimator (which requires
``scipy.spatial.cKDTree`` — no JAX equivalent) with a
**fixed-bandwidth Gaussian KDE**:

```
f̂(x) = (1/W) Σ_i w_i · K_h(x − x_i)
K_h(u) = (2π h²)^(-d/2) exp(-‖u‖² / (2h²))
```

with ``h`` set by Silverman's rule of thumb adapted for weighted
samples:

```
h = σ̂_w · n_eff^(-1/(d+4))
σ̂_w² = (Σ w_i · ‖x_i - x̄_w‖²) / W
n_eff = (Σ w)² / (Σ w²)
```

Trade vs k-NN: Gaussian KDE has heavier tails and more bias near
sharp features; k-NN adapts locally and resolves narrow cascade
peaks better.  But Gaussian KDE is differentiable in both the
evaluation point AND the bandwidth (which itself depends on the
sample weights, hence on θ_pop).

For the M4 integration test (the eventual gate this J3 unlocks),
Gaussian KDE's smoother gradient is more important than k-NN's
sharper feature recovery — the SVI optimizer needs a smooth
likelihood surface.  k-NN MCEvaluator (numpy) remains for the
Session D parity battery where sharpness matters.

Tests (``tests/test_evaluation_mc_jax.py``, sandbox tier):

* **Distributional parity with numpy MCEvaluator** — rel-L1
  between the two density estimators on a shared sample +
  evaluation grid is within (say) 30 % at n=1000 (Gaussian KDE is
  smoother so won't match exactly).
* **AD-through-bandwidth** — gradient of ``density(m, χ)`` w.r.t.
  ``h`` (or w.r.t. sample weights, which determine ``h`` via
  Silverman) is finite and matches finite-differences.
* **Uniform-weights bit-for-bit** — JAX MCEvaluator with
  ``weights=None`` matches itself at the bit level across two
  builds with the same data.  (Cannot bit-match numpy MCEvaluator
  because they use different bandwidth selectors.)

Effort: 1 session.

### J4 — JAX MCResponse + M4 integration test

Goal: wire ``solver="mc_jax"`` (new) through
``build_gaussian_coagulation`` and the population layer.  Then
write ``tests/test_integration_stage1_recovery_mc_jax.py`` —
**this is M4 proper**, finally feasible because the full chain is
AD-traceable.

Surface changes:

* ``CoagulationResponse.solver`` field grows the
  ``"mc_jax"`` literal.
* ``build_gaussian_coagulation(..., solver="mc_jax", ...)`` calls a
  new ``build_mc_jax_response`` that runs the J2 ``particle_mc`` and
  builds a J3 ``MCEvaluatorJax``.
* ``MCResponse`` gains a sibling class
  ``MCResponseJax`` (or the existing class grows JAX-array
  alternatives to the static numpy fields, controlled by a flag).

The M4 integration test parallels
``tests/test_integration_stage1_recovery.py`` exactly with
``solver="mc_jax"``.  Expected outcome: posterior 95 % CIs from
SVI under mc_jax contain θ* on the parameters the implicit test
recovers (mu_natal, natal_spin_loc, natal_spin_scale, log_rate).
The implicit test's xfail-strict on sigma_natal and kernel_b
(resolution-limited bias, see
``[[project-integration-test-in-flight]]``) carries over here.

Effort: 1 session (the work has been front-loaded into J1-J3).

## Sequencing decision

| Session | Topic                          | Risk | Sandbox-verifiable |
|---------|--------------------------------|------|---------------------|
| J1      | sample_natals_jax              | Low  | Yes (full)          |
| J2      | particle_mc JAX                | High | Mostly (parity vs numpy) |
| J3      | MCEvaluator JAX (Gaussian KDE) | Med  | Yes (full)          |
| J4      | MCResponse JAX + M4 integration test | Med | Workstation-only for M4 |

Total: **4 sessions** to land the JAX-MC stack and the M4
integration test.

J1 is the natural first session: smallest blast radius, fully
sandbox-verifiable, unblocks J2 by establishing the JAX-array
plumbing pattern (key splitting, MCConfig field handling).

## Why not just keep numpy and use SBI / NUTS-without-gradient

Two reasons:

1. **AD on θ_pop is the user's stated requirement.**  SBI sidesteps
   AD but requires a separate density estimator training step; not
   the integration path the rest of the project (engine SVI, basis
   SVI) uses.  NUTS-without-gradient is a step backward from the
   implicit path's HMC and has well-known convergence pathologies.
2. **The JAX rewrite buys more than M4.**  Once the MC is JAX-pure,
   it composes with the existing `eqx.Module` response stack
   uniformly with implicit and basis; ``solver="mc_jax"`` becomes a
   drop-in replacement in any code path that uses the response
   surface today.

## What this PLAN does NOT cover

* Engine-side fixes for the
  ``[[project-dchi-eff-engine-mc-systematic-bias]]`` finding —
  separate per-generation diagnosis track.
* SBI / neural-posterior estimation — out of scope for v1.
* M5 (narrow-posterior IS) from
  ``[[demo/PLAN_mc_first_reprioritization]]`` — composes naturally
  with the JAX-MC stack once J4 lands but not gated on this PLAN.

## Related

* `[[demo/PLAN_mc_first_reprioritization]]` — parent PLAN; this is
  its M4-unblocking dependency.
* `[[demo/PLAN_m3_stratified_natal]]` — M3 PLAN; M3a/M3b
  infrastructure (weights, propagation rule) carries forward to the
  JAX rewrite unchanged.
* `[[project-engine-m3b-done]]` — M3 closeout, immediately preceding
  this PLAN.
* `[[project-evaluation-step3-done]]` — numpy MCEvaluator that
  J3 mirrors.
* `[[project-mc-time-calibration-m1-done]]` — M1; ε ↔ depletion
  calibration applies to JAX MC unchanged (the cascade dynamics are
  the same in expectation regardless of implementation language).
* `[[feedback-red-team-validation]]` — load-bearing argument for
  keeping numpy MC alongside JAX MC permanently.
* `[[feedback-perturbative-unreliable-implicit-production]]` —
  "additive, not in-place" rule applied to MC re-implementations.
