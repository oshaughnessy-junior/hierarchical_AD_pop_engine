# J2b — non-Aligned vector physics in the `mc_jax` soft cascade

**Status:** scaffolded 2026-05-30 (primitive audit + AD de-risk done; wiring
not yet landed).  This is the next engine milestone after the two-pulse IMF
(§5.0 of `INSTRUCTIONS_gwkokab_stage1.md`).  It gates **both** Fisher-κ
methods-paper experiments (§6):

* **Natal alignment — recover κ.**  One-pulse, vary Fisher κ, fit κ.
* **Exact alignment, fit with Fisher family (model mismatch).**

Today the `mc_jax` path is **Aligned-only** (`particle_mc_jax_soft` raises on
non-Aligned at `red_team/backup_particle_mc.py:1276`).  J2b lifts that.

---

## 1. Audit — every primitive already exists and is AD-clean

Verified in the JAX sandbox 2026-05-30 (`jax_enable_x64`, CPU):

| primitive | location | AD status |
|---|---|---|
| `Fisher(κ).sample(key, n)` → `(θ₁,φ₁,θ₂,φ₂)` | `spin_orientation.py:365` | **reparametrised inverse-CDF** `cos θ = 1 + (1/κ)·log(u+(1−u)e^{−2κ})`, κ a leaf ⇒ AD in κ flows; clean κ→0 (Isotropic) limit |
| `chi_rem_at_orientation(m1,m2,χ1,χ2,ct1,ct2,Δφ)` → `|χ_rem|` | `remnants/marginal.py:118` | smooth in all inputs (wraps `spin_vector.chi_rem_vec`) |
| `v_kick_at_orientation(...)` → `v_kick` | `remnants/marginal.py:143` | smooth (wraps `kick_vector.v_kick_vec`) |
| `M_rem(m1,m2,χ_z1,χ_z2)` | `remnants/mass.py` | already in the cascade |

**De-risk numbers (one synthetic cascade step, 2000 pairs, Fisher κ):**

```
κ → mean retained |χ_rem|   AD vs FD rel 1.8e-7
v_esc → retain fraction     AD vs FD rel 1.4e-6   (soft sigmoid retention)
κ = 2000 → mean|χ_rem| 0.7725  (→ Aligned asymptote)
```

So **κ-recovery and `∂/∂v_esc` are both differentiable end-to-end** through a
Fisher-sampled vector-remnant step.  The milestone is now *pure wiring* — no
new physics, no new AD machinery.  (Honours `[[feedback-no-fake-physics-stand-ins]]`:
the vector path IS the real BR spin / LZ kick, not a broadening surrogate.)

The eager **numpy** misaligned path already encodes the intended convention
(`_per_event_remnant_misaligned`, `:221`): `φ₁=0, φ₂=Δφ` (azimuthal-symmetry
collapse — `|χ_rem|` & `v_kick` invariant under rigid rotation about `L̂`);
parallel projection `χ_z = χ·cos θ` feeds `M_rem`'s `chi_eff`; per-event
retention.  J2b is the **JAX-traceable** mirror of that, soft-gated.

---

## 2. Insertion site (exact)

`particle_mc_jax_soft.step_fn`, `red_team/backup_particle_mc.py:1353-1356`:

```python
m_new   = _M_rem_jax(m_i, m_j, chi_i, chi_j)            # Aligned scalar
chi_new = _chi_rem_jax(m_i, m_j, chi_i, chi_j)          # signed χ (v1)
v_kick_evt = _v_kick_aligned_jax(m_i, m_j, chi_i, chi_j)
retain  = _retention_aligned_jax(m_i, m_j, chi_i, chi_j, v_esc) > 0.5  # HARD
```

(The B.1 `particle_mc_jax` has the twin block at `:1033-1036`.)

Two changes:

1. **Dispatch the remnant triple** on `cfg.orientation`.  Aligned keeps the
   bit-for-bit block above.  Non-Aligned samples ONE orientation for this
   step's pair and routes through the `*_at_orientation` primitives:

   ```python
   key_o = jax.random.fold_in(orientation_key, k)         # per-step, indep RNG
   th1, ph1, th2, ph2 = cfg.orientation.sample(key_o, 1)
   ct1, ct2, dphi = jnp.cos(th1[0]), jnp.cos(th2[0]), ph2[0]-ph1[0]
   chi_z_i, chi_z_j = chi_i*ct1, chi_j*ct2                # parallel proj → M_rem
   m_new   = _M_rem_jax(m_i, m_j, chi_z_i, chi_z_j)
   chi_new = _chi_rem_at_orientation_jax(m_i, m_j, chi_i, chi_j, ct1, ct2, dphi)  # |χ|
   v_kick_evt = _v_kick_at_orientation_jax(m_i, m_j, chi_i, chi_j, ct1, ct2, dphi)
   ```

   `orientation_key = jax.random.PRNGKey(cfg.orientation_key_seed)` is split by
   step index `k` so it is **independent** of the selection RNG (preserves the
   Aligned-path numerics bit-for-bit — same invariant as the numpy path).

2. **Soft retention** (replaces the `>0.5` Heaviside, so κ / v_esc gradients
   flow): `retain_soft = sigmoid((v_esc − v_kick_evt)/τ)`, `τ` a fixed scale
   (≈ 50 km/s; reuse the `w_over_v_esc`/`retention_kind="sigmoid"` convention
   the implicit path already uses).  The occupancy-flow accounting then uses
   the **continuous** `retain_soft` in place of the bool:

   * net occupancy `mult = 2 − retain_soft` (was `where(retain,1,2)`);
   * `tail_occ = retain_soft · delta` (was `where(retain, delta, 0)`);
   * the heavy slot carries `chi_new` with weight `w_desc` scaled by
     `retain_soft`, the drop fraction `(1−retain_soft)` flows to depletion.

   This is the **same continuous-occupancy philosophy as the B.2 soft K-gate**
   (`[[project-jax-mc-b2-soft-gate-followup]]`), so it composes cleanly with
   the existing budget clip.  Under Aligned we keep the hard bool (bit-for-bit).

**χ sign semantics** (already the documented convention, `:38-41`): the heavy
slot stores **signed χ** under Aligned (v1) and **|χ|** under non-Aligned (the
orientation distribution carries the tilt/sign).  Downstream observables read
χ ∈ [0,1] either way.

---

## 3. Staging

* **J2b-a — vector remnant step + soft retention + dispatch.**  The two edits
  above in `particle_mc_jax_soft` (and optionally the B.1 twin).  Drop the
  `:1276` guard for the soft gate.  Sandbox tier:
  - AD-vs-FD on κ and v_esc through a `total_rate`-like scalar of the run;
  - **Aligned-limit parity**: `Fisher(κ=large)` step observables → the
    existing Aligned block (the 0.7725 asymptote check, tightened);
  - **Isotropic sanity**: `κ=0` ≡ `Isotropic`, ⟨cos θ⟩=0, retention drops
    (kicks larger) relative to Aligned;
  - occupancy invariants (Σ occ·w, depletion budget) unchanged from B.2.
* **J2b-b — κ as a free fit parameter.**  `build_*_coagulation(orientation=
  Fisher(κ))` already plumbs through; expose κ in `make_coagulation_dist_fn`
  (a `natal_orientation="fisher"` arm) + a `STAGE1+κ` / `TWO_PULSE+κ` param
  set in `stage1_wiring`.  Then `demo/fit_fisher_kappa_gwkokab.py`: inject at
  fixed κ*, recover κ (the §6 alignment experiment).  `∂/∂v_esc` recovers for
  free here.
* **J2b-c — model-mismatch study.**  Generate data at `Aligned`, fit with the
  `Fisher(κ)` family; watch κ run to its large-κ rail + the bias it injects in
  the natal/kernel params.  Pure demo once J2b-b exists.

## 4. Risks / watch-items

* **Per-step single-pair vs batched orientation.**  The soft scan processes
  one heavy/light merge per step; one `orientation.sample(key, 1)` per step is
  correct but adds a sphere-sample to every step — cheap, but confirm it does
  not perturb the jit step-time budget materially.
* **Soft-retention NaN-in-backward.**  Follow `[[feedback-where-guards-forward-not-backward]]`:
  use a division-free sigmoid (no `v_kick/clip` ratios) so the budget-clip
  backward stays finite under SVI.
* **Single-key FD is not an AD reference** for the orientation draw
  (`[[feedback-single-key-fd-not-ad-reference]]`): validate κ-AD via
  fixed-`u`-replay FD or many-key FD-mean, not a single-key `|AD−FD|`.
* **`v_kicks` return semantics** already switch to per-event under non-Aligned
  (`:35`) — keep that.

## 5. Pointers

Aligned dispatch precedent end-to-end: the **implicit** path already does all
of this (`spin_orientation.Fisher`, `remnants/{spin_vector,kick_vector,marginal}`,
`P_retain_marginalised`); J2b is the MC mirror.  Misalignment history:
`[[project-misalignment-session-c-done]]` (implicit), `demo/PLAN_jax_mc.md` §J2
(the original J-series scoping that deferred J2b).
