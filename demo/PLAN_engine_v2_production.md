# Plan — engine v2 production migration

## Status snapshot (2026-05-18)

Supersedes `PLAN_implicit_engine.md`. That plan was written when the
implicit prototype was un-validated. We now have:

- **Validated prototype.** `red_team/engine_v2_implicit.solve_implicit`
  (2D damped Picard on (m, χ)) agrees with the 1D Doctor port
  `backup_perturbative.implicit_picard` to **JS = 0.003** on `dN/dm`
  at ε = 1 (Stage 1 θ*). Residual large-mass decoherence is a real
  physics effect (1D has `m_rem = m1 + m2`; 2D uses Lousto-Zlochower
  `M_rem` with non-zero `E_rad` from aligned spins).
- **Validated cross-method.** Pure particle-MC backstop
  (`backup_particle_mc`) matches the two implicit solvers to
  JS ≈ 0.007 at ε = 1 when stopped at the **self-consistent**
  depletion fraction
  `f = ε · ∫L[N*] / ∫N_natal ≈ 0.25`. That mapping is now wired into
  `compare_all.py` as the default — no empirical knob.
- **Cell-area Jacobian fix landed.** `_step_one_generation` returns
  count-per-cell; both the new implicit solver and the legacy `_solve`
  now divide by `dm_out · dχ_out` so all `n_k` are on the same density
  measure. ε is now interpretable as evolution time × kernel-rate-
  constant on an absolute scale.
- **Red-team test harness exists.** `red_team/compare_all.py` is the
  living validation: any future regression in the production solver
  must hold its JS against backups 1 and 2.

The remaining gap — `engine` (perturbative, k_max = 4) vs the implicit
methods — is **structural**, not a bug: `_solve` accumulates only the
gain term; the loss term (mass leaving the natal pool) and the
all-orders-summed Volterra series are missing. The fix is to ship the
implicit solver as the production code path. This document is the plan
for that migration.

## Design goals

1. **Right physics by default.** A user who calls
   `build_gaussian_coagulation(...)` and then evaluates `log_rate_density`
   should get the implicit, all-orders, density-consistent result — not
   the perturbative truncation.
2. **AD-clean.** Gradients of inference posteriors through the
   Picard solver must be exact (not unrolled-loop noise) and fast
   enough for NUTS / SVI on production-resolution grids.
3. **Perturbative path kept as diagnostic.** `_solve` (perturbative)
   stays as a comparison surface and for the existing test suite, but
   becomes opt-in via a `solver` field.
4. **No empirical tuning knobs.** ε has absolute meaning. The closed-
   system MC backstop's stopping fraction is computed from the
   implicit fixed point, not chosen by hand
   (`feedback_no_fudge_factors.md`).
5. **Validation is a callable test.** The red-team comparison must be
   runnable from `pytest` as a slow integration check.

## Step 1 — Lift solver into the package

Move `solve_implicit`, `gain_density`, `loss_density`, `_dm_log`,
`_dchi` from `red_team/engine_v2_implicit.py` into a new module
`src/hierarchical_ad_pop_engine/response/implicit.py`. Public API:

```python
def solve_implicit_fixed_point(
    response: CoagulationResponse,
    v_esc: Array,
    *, epsilon: float = 1.0,
    max_iter: int = 200, tol: float = 1e-7, damping: float = 0.5,
) -> ImplicitSolveResult: ...
```

`ImplicitSolveResult` is an `eqx.Module` carrying `N`, `log_N`,
`n_iter`, `final_diff`, `loss_integral`, `natal_integral`,
`f_natal_depleted`, `epsilon`. The diagnostics fields are part of the
public interface — inference code can sanity-check
`f_natal_depleted ∈ (0, 1)` and flag pathological θ samples (Picard
divergence ⇒ `n_iter == max_iter` ⇒ raise a JAX-traceable warning).

Add a `solver` field on `CoagulationResponse`:

```python
class CoagulationResponse(eqx.Module):
    ...
    solver: str = eqx.field(static=True, default="perturbative")
    epsilon: Array = jnp.asarray(1.0)  # used only when solver == "implicit"

    def _log_total_density(self, v_esc):
        if self.solver == "implicit":
            return solve_implicit_fixed_point(self, v_esc, epsilon=self.epsilon).log_N
        log_n_k = self._solve(v_esc)
        return _logsumexp_axis(log_n_k, axis=0)
```

`log_rate_density` and the gwkokab integration layer are unchanged —
both consume `_log_total_density`.

`build_gaussian_coagulation(..., solver="implicit", epsilon=1.0)` is the
new opt-in constructor; the default stays `"perturbative"` for one
minor release so existing demos don't change semantics under us.

**Effort:** 0.5 session. **Tests:** smoke-import + a parity check that
`solver="implicit"` on the package matches `red_team/engine_v2_implicit`
output bitwise.

## Step 2 — JIT-compatible Picard loop

The prototype's Picard loop is a Python `for` with `float(...)` casts
on the convergence trace. This works but won't `jax.jit` (the
Python-side `if final_diff < tol: break` is not traceable). Rewrite
as `jax.lax.while_loop` with the carry
`(N, n_iter, final_diff)`, terminating on
`(final_diff < tol) | (n_iter >= max_iter)`. Drop the per-iter
history (or accumulate via `jax.lax.scan` if we want it for
diagnostics; default to off for inference).

This is a precondition for Step 3 — `custom_vjp` needs the forward to
be traceable.

**Effort:** 0.5 session. **Risk:** none beyond the standard `while_loop`
ergonomics. **Tests:** parity vs Step 1's Python loop within rtol 1e-6.

## Step 3 — AD via implicit-function theorem (custom VJP)

The Picard loop is **not** AD-safe by default: JAX would unroll the
loop and back-prop through every iteration, producing gradient noise
that scales with the number of iterations and consumes ~100× the
forward memory.

Standard fix: register a custom VJP at the fixed point. For
`N = F(N, θ)` with `F = N_natal(θ) + ε · (G[N, θ] − L[N, θ])`,
implicit-function theorem gives

```
dN/dθ = (I − ∂F/∂N)⁻¹ · ∂F/∂θ
```

evaluated at the converged `N*`. Two implementations to evaluate:

- **`jax.lax.custom_root`** (preferred): documented, well-tested,
  handles the linear-solve internally via `gmres`. Signature:
  `custom_root(F_minus_N, init, solve, has_aux=False)`.
- **`jaxopt.implicit_diff.custom_root`**: same idea, slightly more
  flexible API for the linear solve. Pull `jaxopt` only if `jax.lax`
  version is awkward for our grid sizes.

Memory check: `∂F/∂N` is a `(N_m·N_χ, N_m·N_χ)` linear operator —
3072×3072 for (96, 32). Dense LU is fine; matvec-only `gmres` is
better for AD scaling. We never need to form `∂F/∂N` explicitly; only
its action on a vector, which is one extra `gain_density` +
`loss_density` evaluation per gmres step.

**Effort:** 1 session. **Tests:** finite-difference vs AD on a 6-θ
perturbation; agreement within ~1e-4 relative confirms. Run on a tiny
(N_m = 24, N_χ = 8) grid for speed.

**Open question:** the loss term contains `P_retain(m1, m2, χ1, χ2,
v_esc)` — a 4D Heaviside. `∂P/∂v_esc` is a delta-function on the
escape boundary, evaluating to zero almost everywhere in float arithmetic.
This is the *same* AD constraint the current engine already has, so
nothing new is introduced. Document in the response-function
docstring; flag as a known limitation in `DESIGN.md`.

## Step 4 — Test migration

The existing 20-test smoke suite was written against the buggy
perturbative path. With the cell-area Jacobian fix already landed in
`_solve`, baselines may need updating. Approach:

a. **Re-baseline perturbative tests.** Run with `solver="perturbative"`,
   capture new expected values, commit baseline diffs separately with
   a comment that points to `project_engine_cell_area_fix.md`.
b. **Add implicit tests** (~6 new): convergence on Stage 1 θ*;
   matches `backup_perturbative.implicit_picard` to JS < 0.01 on
   `dN/dm`; matches `backup_particle_mc` at self-consistent f to
   JS < 0.05; finite-difference vs AD gradient on a 6-θ perturbation;
   pathological-θ handling (Picard non-convergence raises or returns a
   diagnostic).
c. **Promote `compare_all.py` to a pytest integration test.** Slow
   (~30 s on CPU), gated behind `--run-integration`. Fails if any of
   `engine_v2 ↔ backup1`, `engine_v2 ↔ backup2`, `backup1 ↔ backup2`
   JS regresses beyond a documented threshold. This is the gold-
   standard cross-method check; without it we have no defence
   against future regressions in the production solver.

**Effort:** 0.8 session. **Anti-goal:** don't delete tests just
because they go red — investigate whether the new value reflects a
correctness fix (it almost always does after the cell-area patch).

## Step 5 — Stage 1 demo + mock regeneration + refit

The existing Stage 1 mock catalog (`demo/demo_gwkokab_fit_stage1.py`)
was generated by the buggy engine. Posteriors recovered from it are
self-consistent (mock-buggy ↔ fit-buggy) but mean nothing physically
because they encode the cell-area bias and the missing loss term.

Procedure:
1. Edit the demo to build `response = build_gaussian_coagulation(...,
   solver="implicit", epsilon=1.0)`.
2. Run `generate_mock` once → new mock catalog
   `mock_catalog_implicit_<hash>.npz`. Save under
   `figures_<hash>/`.
3. Re-fit with SVI as the smoke pass, NUTS for the headline corner.
4. Document expected vs observed posterior shifts:
   - `mu_natal`, `sigma_natal`: narrower (the implicit engine sees
     through the perturbative biasing of natal peak by 2G+ pile-up).
   - `kernel_b`: shifts more than the natal parameters — kernel
     directly controls cascade depth.
   - `log_rate`: shifts by `log(some O(1) factor)` — the engine's
     normalisation changes when log_n[k] units become consistent.
5. Add a regression test that runs SVI for 200 steps on the canned
   mock and checks recovery of `mu_natal` within 1σ.

If recovery on the new mock looks worse than on the old, **that's a
feature**: the old mock had matched bias.

**Effort:** 0.5 session.

## Step 6 — Performance pass

Picard at (96, 32): ~75 MB / iter, ~80–120 iters at ε = 1 ⇒
~6–10 GB array ops per solve. NUTS: 1500 solves total ≈ 10–15 TB.
GPU-tractable, CPU painful.

Profile first; optimise only if real wallclock is unacceptable
(target: <30 min CPU for the Stage 1 NUTS smoke).

Options ranked by leverage:
- **Anderson acceleration.** 3–5× fewer Picard iterations for the
  same tolerance. Single-page change wrapping the iterate update with
  the Anderson mixing block. Try first.
- **Warm-start.** Cache the converged `N*` across Picard solves and
  warm-start from the previous θ-sample's `N*`. Most NUTS warmup
  transitions are small θ moves; warm-start could cut iters by 5×.
- **Lower grid for fit, full grid for analysis.** `N_m = 48,
  N_χ = 16` gives a quartic speedup; physics validates at low grid
  on smooth populations.
- **Mixed precision.** Picard converges fine at float32 for ε ≤ 2;
  the linear solve in the VJP may need float64 for conditioning.

**Effort:** 0.5–1.0 session, conditional on profiling.

## Step 7 — Cleanup + deprecation

- Add a `DeprecationWarning` on `solver="perturbative"` after the
  implicit path has shipped for 1 minor release.
- Keep `red_team/` as the gold-standard validation suite. Update
  `red_team/engine_v2_implicit.py` to be a thin wrapper around the
  package solver (so the comparison driver continues to work but
  isn't a duplicate codebase).
- Update `DESIGN.md` §3.1 with the new solver path. Update
  `ARCH_NOTES.md` to mark the implicit-solver concern as resolved.
- Delete the `--v2-epsilon` decoupling flag in `compare_all.py` once
  the engine path is also implicit (it exists today as a separation
  knob between the prototype and the bug-prone engine path).

**Effort:** 0.3 session.

## Effort estimate

| Step | What                                                  | Effort         |
|------|-------------------------------------------------------|----------------|
| 1    | Lift solver into package; add `solver` field          | 0.5 session    |
| 2    | JIT-compatible `lax.while_loop` Picard                | 0.5 session    |
| 3    | Implicit-function-theorem custom VJP                  | 1.0 session    |
| 4    | Test migration + red-team-as-pytest                   | 0.8 session    |
| 5    | Stage 1 demo switch + mock regen + refit              | 0.5 session    |
| 6    | Performance pass (conditional)                        | 0.5–1.0 session |
| 7    | Cleanup + deprecation                                 | 0.3 session    |
|      | **Total**                                             | **~4 sessions** |

## Anti-goals

- **No empirical fudge factors.** ε is physical. If something doesn't
  agree at the same ε, find the measure / Jacobian bug. The cell-area
  story (`project_engine_cell_area_fix.md`) is the precedent.
- **Don't delete `_solve`.** It stays as the diagnostic path and as the
  reference against which we measure how much the perturbative
  truncation costs.
- **Don't widen the public surface area.** No `k_max_implicit`, no
  `solver="anderson"` mode-switch. The implicit solver has one
  hyperparameter — ε — and a small set of numerical knobs (max_iter,
  tol, damping) that are tuning, not modelling.
- **Don't ship without the AD path.** A non-AD-safe implicit solver
  is worse than the perturbative truncation — it's *also* slow under
  NUTS and *also* wrong, just for a different reason.
- **Don't half-migrate.** Step 5's mock regeneration is non-optional:
  shipping the implicit solver while leaving the demo on a stale
  mock guarantees confusion at the next inference-review meeting.

## What this plan does not cover

- **Spin precession / non-aligned retention.** Engine v2 keeps
  aligned-only `P_retain`. Generalising to a tilted-spin retention
  surface is a separate AD-friendly retention v2 task — orthogonal
  to the solver migration.
- **Redshift dependence.** Stage 1 is fixed-`z`. The implicit solver
  is per-environment; redshift integration is the formation-rate
  layer's job.
- **Selection effects.** Stage 1 has flat selection. Stage 2 wires
  gwkokab's selection-aware likelihood; the engine API is unchanged.

## Decision points to flag at execution time

1. **Step 3 implementation choice.** `jax.lax.custom_root` vs
   `jaxopt.implicit_diff.custom_root`. Default: `jax.lax`. Bring in
   `jaxopt` only if the linear-solve API forces it.
2. **Step 4 baseline updating.** Some existing tests will go red after
   the cell-area fix lands fully. Decide per-test whether to update
   the baseline or whether the test was testing the bug.
3. **Step 6 trigger.** Skip the performance pass entirely if Stage 1
   NUTS fits in 30 min on the dev laptop. Only profile if a real
   wallclock problem surfaces.
