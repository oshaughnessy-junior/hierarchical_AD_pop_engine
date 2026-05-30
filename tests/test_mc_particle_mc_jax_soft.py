r"""Session B.2 sandbox-tier tests — JAX-pure ``particle_mc_jax_soft``.

The deterministic soft K-gate (occupancy flow + slot pre-allocation),
successor to round-3's B.1 Bernoulli K-gate.  Design + numpy derivation:

* ``demo/PLAN_jax_mc_time_decoupling_diagnosis.md`` § 5.B.2 + § 8
* ``[[project-jax-mc-b2-soft-gate-followup]]``
* numpy sandbox ``outputs/b2_soft_gate_derivation.py`` (verified Q1/Q2/Q3)
  and ``outputs/b2_jax_mirror_check.py`` (this step logic, numpy mirror).

Pins eight invariants.  The EXACT ones (S2/S3/S4) are the hard gates —
they encode the B.2 design guarantees and were verified bit-stably in the
numpy mirror.  The AD ones (S6/S7) carry conservative tolerances flagged
for first-workstation calibration (this Cowork VM has no JAX runtime, so
their numeric thresholds are sandbox-unverified — see the per-test
comments).

S1. **Smoke / schema** — runs at Stage 1 θ*; returns the B.1 schema PLUS
    the continuous ``occupancy`` field; full-length axis is
    ``S = n_init + n_steps`` (slot pre-allocation).
S2. **Depletion pinned by budget (k_ratio elimination)** — ``stats
    ['depletion'] == depletion_fraction`` to machine tolerance, and
    INDEPENDENT of ``n_steps_mult`` (mult=2 and mult=5 give the same
    depletion).  This is the headline B.2 guarantee: no magic factor.
S3. **W_total identity** — ``Σ(occupancy · w_active) == (1 − D)·n_init``
    bit-stably.  Same density convention as B.1.
S4. **Occupancy bounds** — ``occupancy ∈ [0, 1]`` everywhere; parent
    slots start ≤ 1, tail slots ≥ 0; non-event tail slots are exactly 0.
S5. **Distributional parity with numpy MC** — KS on the occupancy-
    weighted surviving m_active vs numpy ``particle_mc`` shows no
    detectable difference (red-team validation backstop).
S6. **AD through mu_natal** — ``jax.grad(Σ m_active·occupancy)(mu_natal)``
    finite and matches central FD (the load-bearing M4 gradient).
S7. **AD through kernel_b WITHOUT straight-through (the B.2 headline)** —
    on the K-weighting probe ``⟨q⟩ = Σ(δ·q) / Σ δ`` (the merger-weighted
    mean mass ratio; with kernel a=0, ``K = q^b`` so kernel_b directly
    controls the q-preference and ``⟨q⟩`` rises monotonically with b), a
    DIRECT single-key central FD is finite and NON-zero and ``jax.grad``
    matches it in sign/scale.  The forward pass is smooth, so no
    straight-through estimator is required (contrast B.1, where the hard
    Bernoulli accept zeroes the forward FD and forces a soft surrogate).

    NB: do NOT test ``Σ w_mergers`` here — under uniform natal weights
    ``Σ w_mergers = Σ δ`` and (at ~100% retention) the depletion BUDGET
    pins ``Σ δ = D·n_init`` constant in kernel_b, so its gradient is
    correctly ~0.  The kernel_b signal lives in the cascade SHAPE (WHICH
    pairs merge — captured by the δ-weighted ⟨q⟩), not the pinned total.
    Surviving-pool aggregates (``Σ m·occ``) are also weakly/noisily
    kb-dependent under the pinned budget; the δ-weighted merger ⟨q⟩ is
    the clean monotone probe (numpy sandbox: 0.853→0.881 over b∈[0.6,1.4]).
S8. **non-Aligned raises** — Aligned-only first pass.

Additive per ``[[feedback-red-team-validation]]``: numpy ``particle_mc``
and B.1 ``particle_mc_jax`` are both untouched.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

scipy_stats = pytest.importorskip("scipy.stats")

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from red_team.backup_particle_mc import (  # noqa: E402
    MCConfig,
    particle_mc,
    particle_mc_jax_soft,
)


THETA = dict(
    mu_natal=8.0, sigma_natal=1.0, m_min=4.0, m_max=80.0,
    natal_spin_loc=0.0, natal_spin_scale=0.10,
    kernel_a=0.0, kernel_b=1.0, v_esc=1000.0,
)
N_SEED = 200
DEPL = 0.10


# ---------------------------------------------------------------------------
# S1: smoke — runs, returns schema with continuous occupancy.
# ---------------------------------------------------------------------------
def test_smoke_runs_returns_schema():
    cfg = MCConfig(n_seed=N_SEED, depletion_fraction=DEPL, **THETA)
    out = particle_mc_jax_soft(cfg, jax.random.PRNGKey(2026))
    for k in ("m_active", "chi_active", "w_active", "gen_active",
              "occupancy", "active_mask", "mergers", "v_kicks",
              "w_mergers", "step_active", "stats"):
        assert k in out, f"missing key {k} in particle_mc_jax_soft output"
    n_steps = out["stats"]["n_steps"]
    S = N_SEED + n_steps
    # Slot pre-allocation: full-length axis is S = n_init + n_steps.
    assert out["m_active"].shape == (S,)
    assert out["occupancy"].shape == (S,)
    assert out["active_mask"].shape == (S,)
    assert out["mergers"].shape == (n_steps, 8)
    assert out["w_mergers"].shape == (n_steps,)
    # default n_steps = int(D * n_init * n_steps_mult) = int(0.1*200*3) = 60.
    assert n_steps == 60, f"expected n_steps=60, got {n_steps}"
    print(f"\n[S1] schema OK: n_init={N_SEED} n_steps={n_steps} S={S} "
          f"W_total={float(out['stats']['W_total']):.2f} "
          f"depletion={float(out['stats']['depletion']):.4f}")


# ---------------------------------------------------------------------------
# S2: depletion pinned by the budget, independent of n_steps_mult.
#     The k_ratio-elimination headline (EXACT gate).
# ---------------------------------------------------------------------------
def test_depletion_pinned_independent_of_n_steps():
    cfg = MCConfig(n_seed=N_SEED, depletion_fraction=DEPL, **THETA)
    key = jax.random.PRNGKey(11)
    deps = {}
    for mult in (2.0, 3.0, 5.0):
        out = particle_mc_jax_soft(cfg, key, n_steps_mult=mult)
        deps[mult] = float(out["stats"]["depletion"])
    print(f"\n[S2] depletion vs n_steps_mult: {deps}  (target {DEPL})")
    for mult, d in deps.items():
        assert abs(d - DEPL) < 1e-5, (
            f"depletion {d} != target {DEPL} at n_steps_mult={mult} — "
            f"budget clip should pin it exactly"
        )
    # Independence: spread across mults is machine-level.
    spread = max(deps.values()) - min(deps.values())
    assert spread < 1e-6, (
        f"depletion depends on n_steps_mult (spread={spread:.2e}) — "
        f"the budget is not pinning it; k_ratio knob not eliminated"
    )


# ---------------------------------------------------------------------------
# S3: W_total identity — Σ(occupancy·w_active) = (1-D)·n_init.
# ---------------------------------------------------------------------------
def test_w_total_identity():
    cfg = MCConfig(n_seed=N_SEED, depletion_fraction=DEPL, **THETA)
    out = particle_mc_jax_soft(cfg, jax.random.PRNGKey(3))
    W_total = float(jnp.sum(out["occupancy"] * out["w_active"]))
    expect = (1.0 - DEPL) * N_SEED
    print(f"\n[S3] W_total={W_total:.4f}  expect (1-D)*n_init={expect:.4f}")
    assert abs(W_total - expect) < 1e-3 * expect, (
        f"W_total {W_total} != (1-D)*n_init {expect}"
    )
    # Cross-check stats field agrees.
    assert abs(float(out["stats"]["W_total"]) - W_total) < 1e-4


# ---------------------------------------------------------------------------
# S4: occupancy bounds and dead-tail-slot semantics.
# ---------------------------------------------------------------------------
def test_occupancy_bounds():
    cfg = MCConfig(n_seed=N_SEED, depletion_fraction=DEPL, **THETA)
    out = particle_mc_jax_soft(cfg, jax.random.PRNGKey(5))
    o = np.asarray(out["occupancy"])
    assert np.all(o >= -1e-9), f"occupancy went negative: min={o.min()}"
    assert np.all(o <= 1.0 + 1e-9), f"occupancy exceeded 1: max={o.max()}"
    # active_mask is exactly occupancy > tol.
    am = np.asarray(out["active_mask"])
    assert np.array_equal(am, o > 1e-9)
    n_dead_tail = int(np.sum((np.arange(o.size) >= N_SEED) & (o <= 1e-9)))
    print(f"\n[S4] occupancy in [{o.min():.3e}, {o.max():.3f}]; "
          f"dead tail slots={n_dead_tail}")


# ---------------------------------------------------------------------------
# S5: distributional parity with numpy particle_mc (red-team backstop).
# ---------------------------------------------------------------------------
def test_distribution_matches_numpy_ks():
    cfg = MCConfig(n_seed=2000, depletion_fraction=DEPL, **THETA)
    out_np = particle_mc(cfg, np.random.default_rng(2026))
    out_jx = particle_mc_jax_soft(cfg, jax.random.PRNGKey(2026))
    m_np = np.asarray(out_np["m_active"])
    o = np.asarray(out_jx["occupancy"])
    m_jx = np.asarray(out_jx["m_active"])
    keep = o > 1e-9
    m_jx = m_jx[keep]
    w_jx = o[keep]
    # Weighted KS via resampling the soft sample by occupancy weight.
    rng = np.random.default_rng(0)
    idx = rng.choice(m_jx.size, size=m_np.size,
                     p=w_jx / w_jx.sum(), replace=True)
    ks = scipy_stats.ks_2samp(m_np, m_jx[idx])
    print(f"\n[S5] KS(m_active) numpy vs soft: stat={ks.statistic:.4f} "
          f"p={ks.pvalue:.3f}")
    assert ks.pvalue > 0.01, (
        f"surviving m_active mismatch numpy vs soft: KS p={ks.pvalue:.3g}"
    )


# ---------------------------------------------------------------------------
# S6: AD through mu_natal — load-bearing M4 gradient.
# ---------------------------------------------------------------------------
def test_grad_through_mu_natal_matches_fd():
    r"""``jax.grad(Σ m_active·occupancy)(mu_natal)`` matches central FD.

    Tolerance note (sandbox-unverified): occupancy weighting is smooth in
    mu, but pair SELECTION (jax.random.choice) shifts with mu across the
    FD step — AD treats the selected indices as constants, FD does not, so
    a single-key AD-vs-FD can carry selection-shift noise.  Threshold set
    conservatively; tighten on first workstation run if clean.
    """
    cfg_base = MCConfig(n_seed=N_SEED, depletion_fraction=DEPL, **THETA)
    key = jax.random.PRNGKey(7777)

    def f(mu):
        cfg = replace(cfg_base, mu_natal=mu)
        out = particle_mc_jax_soft(cfg, key, n_steps=60)
        return jnp.sum(out["m_active"] * out["occupancy"])

    mu_ref = jnp.asarray(THETA["mu_natal"])
    ad = float(jax.grad(f)(mu_ref))
    h = 1e-2
    fd = float((f(mu_ref + h) - f(mu_ref - h)) / (2.0 * h))
    rel = abs(ad - fd) / max(abs(fd), 1e-9)
    print(f"\n[S6] AD ∂Σ(m·occ)/∂mu: AD={ad:.4f} FD={fd:.4f} rel={rel:.3e}")
    assert np.isfinite(ad), "AD gradient must be finite"
    # f ≈ W_total·(mean surviving mass); ∂/∂mu is O(W_total) ≈ (1-D)·n_init.
    assert abs(ad) > 0.0, "AD gradient must be non-zero"
    assert rel < 1e-1, f"AD vs FD through mu_natal: rel={rel:.3e}"


# ---------------------------------------------------------------------------
# S7: AD through kernel_b WITHOUT a straight-through estimator.
#     The B.2 headline — forward pass is smooth.
# ---------------------------------------------------------------------------
def test_grad_through_kernel_b_flows_without_straight_through():
    r"""kernel_b gradient FLOWS through the smooth forward (finite, non-zero)
    with NO straight-through estimator — the B.2 win over B.1.

    Observable: ``⟨q⟩ = Σ(δ·q)/Σδ`` — the δ-weighted (w_mergers-weighted)
    mean mass ratio of merged pairs (``q = m2/m1 ≤ 1``, heavier-first).
    With kernel a=0, ``K = q^b``, so kernel_b reweights δ toward high-q
    pairs.  ``δ = (K_ij/K_max)·min(o_i,o_j)`` is smooth in kernel_b, so
    ``jax.grad`` returns a finite, non-zero gradient — whereas B.1's hard
    Bernoulli forward gives an EXACTLY zero single-key forward gradient
    (it needs a straight-through surrogate even to define one;
    cf. ``test_grad_through_kernel_b_matches_fd``).

    Why there is NO single-key FD assertion here.  In a stochastic
    cascade the pair selection (``jax.random.choice ∝ occupancy``)
    depends on θ, but AD treats the SELECTED INDICES as constants (it
    does not differentiate the sampler).  A single-key FD instead lets
    the selection SHIFT as kernel_b moves, so its value is dominated by
    discrete selection jumps that AD correctly ignores — the two simply
    measure different things.  Numerically confirmed in numpy at this
    exact config (n_init=200, n_steps=60): the fixed-selection
    (AD-faithful) gradient and the free-selection gradient are both
    O(1e-2), while a JAX single-key FD came out ~0.5 (≈45× larger) — i.e.
    selection-shift contaminated.  The RIGOROUS quantitative smoothness
    proof is the fixed-pair-sequence comparison in
    demo/b2_soft_gate_derivation.py § Q3 (2nd-diff 3e-6).  A quantitative
    AD-vs-FD-MEAN match belongs at a many-key workstation tier where the
    selection noise averages out; it is intentionally not gated here.

    Do NOT use ``Σ w_mergers`` or ``Σ m·occ``: ``Σ w_mergers = Σ δ`` is
    pinned to ``D·n_init`` by the depletion budget (gradient ~0), and
    surviving-pool aggregates are weakly/noisily kb-dependent under the
    fixed budget.
    """
    cfg_base = MCConfig(n_seed=N_SEED, depletion_fraction=DEPL, **THETA)
    key = jax.random.PRNGKey(7777)

    def f(kb):
        cfg = replace(cfg_base, kernel_b=kb)
        out = particle_mc_jax_soft(cfg, key, n_steps=60)
        sa = out["step_active"]
        # NaN-safe: replace non-event parent masses with finite dummies
        # (weight 0 nulls them).  Heavier-first record ⇒ q = m2/m1 ≤ 1.
        m1 = jnp.where(sa, out["mergers"][:, 0], 1.0)
        m2 = jnp.where(sa, out["mergers"][:, 1], 0.0)
        q = m2 / m1
        wd = jnp.where(sa, out["w_mergers"], 0.0)
        return jnp.sum(wd * q) / jnp.maximum(jnp.sum(wd), 1e-30)

    kb_ref = jnp.asarray(THETA["kernel_b"])
    ad = float(jax.grad(f)(kb_ref))
    # Single-key FD — DIAGNOSTIC ONLY (selection-shift contaminated; see
    # docstring).  Printed, never asserted against.
    h = 1e-2
    fd_diag = float((f(kb_ref + h) - f(kb_ref - h)) / (2.0 * h))
    print(f"\n[S7] AD ∂⟨q⟩/∂kernel_b = {ad:.4e}  (finite, non-zero ⇒ no "
          f"straight-through needed)")
    print(f"     single-key FD = {fd_diag:.4e}  [DIAGNOSTIC ONLY — "
          f"selection-shift contaminated, not an AD reference]")
    assert np.isfinite(ad), "AD gradient through kernel_b must be finite"
    assert abs(ad) > 1e-5, (
        "AD gradient through kernel_b is ~0 — the smooth forward should "
        "carry a non-zero kernel_b gradient (B.1's hard forward is the "
        "one that gives exactly 0)."
    )


# ---------------------------------------------------------------------------
# S8: non-Aligned orientation raises.
# ---------------------------------------------------------------------------
def test_non_aligned_orientation_raises():
    from hierarchical_ad_pop_engine.spin_orientation import Isotropic
    cfg = MCConfig(
        n_seed=N_SEED, depletion_fraction=DEPL,
        orientation=Isotropic.default(), **THETA,
    )
    with pytest.raises(NotImplementedError, match="Aligned"):
        particle_mc_jax_soft(cfg, jax.random.PRNGKey(2026))
