r"""Faithful numpy mirror of the JAX particle_mc_jax_soft step_fn.

Replicates the EXACT indexing of the JAX implementation (S = n_init +
n_steps pre-allocated slots; one tail slot n_init+k owned per step;
occupancy-weighted pair draw; depletion-budget clip) so we can validate
the JAX code path's logic without a JAX runtime.  Cross-checks against
the already-validated cascade_soft / cascade_discrete in
b2_soft_gate_derivation.py.

Verifies:
  M1. depletion pinned to target (budget clip works with per-step tail).
  M2. W_total = (1 - D) * n_init  (occupancy*importance sum).
  M3. KS(m), KS(chi) vs discrete reference under the KS_crit floor.
  M4. tail-slot layout (occupancy-0 dead slots) contributes nothing —
      identical observables to the packed-write-pointer prototype.
"""
from __future__ import annotations

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from b2_soft_gate_derivation import (
    CFG, A, B, V_ESC, sample_natal, kernel_K, kernel_upper_bound,
    remnant, cascade_discrete, cascade_soft, weighted_ks,
)


def cascade_soft_jaxlike(rng, m0, chi0, a, b, v_esc, depletion, n_steps,
                         occ_tol=1e-9):
    """Numpy port that mirrors particle_mc_jax_soft.step_fn exactly."""
    n_init = m0.size
    S = n_init + n_steps
    m = np.zeros(S); chi = np.zeros(S); w = np.zeros(S)
    occ = np.zeros(S); gen = np.zeros(S, dtype=int)
    m[:n_init] = m0; chi[:n_init] = chi0; w[:n_init] = 1.0
    occ[:n_init] = 1.0; gen[:n_init] = 1
    R_target = depletion * n_init
    net_removed = 0.0
    w_mergers = []

    for k in range(n_steps):
        occ_active = occ > occ_tol
        n_active = occ_active.sum()
        budget_left = R_target - net_removed
        step_active = (n_active >= 2) and (budget_left > occ_tol)

        m_for_min = np.where(occ_active, m, np.inf)
        m_for_max = np.where(occ_active, m, -np.inf)
        K_max = kernel_upper_bound(
            m[occ_active] if occ_active.any() else np.array([1.0, 1.0]), a, b
        )
        if not step_active:
            K_max = 1.0

        # occupancy-weighted pair draw (j != i)
        p_occ = np.where(occ_active, occ, 0.0)
        s = p_occ.sum()
        if s <= occ_tol:
            continue
        p_i = p_occ / s
        i = int(rng.choice(S, p=p_i))
        p_j = p_occ.copy(); p_j[i] = 0.0
        sj = p_j.sum()
        if sj <= occ_tol:
            continue
        p_j = p_j / sj
        j = int(rng.choice(S, p=p_j))

        K_ij = kernel_K(m[i], m[j], a, b)
        w_event = K_ij / K_max
        delta_raw = w_event * min(occ[i], occ[j])

        m_new, chi_new, v_kick = remnant(m[i], m[j], chi[i], chi[j])
        retain = v_kick < v_esc
        # Division-free, gradient-safe depletion cap (mirrors the JAX fix):
        # delta_cap = max(budget_left,0)/mult, delta = min(delta_raw, cap).
        mult = 1.0 if retain else 2.0
        delta_cap = max(budget_left, 0.0) / mult
        delta = min(delta_raw, delta_cap)
        if not step_active:
            delta = 0.0
        net_this = mult * delta

        occ[i] -= delta
        occ[j] -= delta
        tail = n_init + k
        m[tail] = m_new; chi[tail] = chi_new
        w[tail] = 0.5 * (w[i] + w[j])
        occ[tail] = delta if retain else 0.0
        gen[tail] = max(gen[i], gen[j]) + 1
        net_removed += net_this
        if delta > occ_tol:
            w_mergers.append(0.5 * (w[i] + w[j]) * delta)

    occ_final = occ > occ_tol
    W_total = float(np.sum(occ[occ_final] * w[occ_final]))
    return dict(
        m_active=m[occ_final], chi_active=chi[occ_final],
        occ_active=occ[occ_final], w_active=w[occ_final],
        depletion=net_removed / n_init, W_total=W_total, n_init=n_init,
        net_removed=net_removed, n_dead_tail=int(np.sum(
            (np.arange(S) >= n_init) & (occ <= occ_tol))),
        w_mergers=np.asarray(w_mergers),
    )


def main(depletion=0.10, n_init=3000, seeds=12):
    print("=" * 72)
    print("JAX-mirror check — particle_mc_jax_soft step logic in numpy")
    print("=" * 72)
    n_steps = int(3.0 * depletion * n_init)
    ks_m, ks_chi, dep, wtot, pkt, dead = [], [], [], [], [], []
    for sd in range(seeds):
        rng_d = np.random.default_rng(31000 + sd)
        m0, chi0 = sample_natal(rng_d, n_init, **CFG)
        d = cascade_discrete(rng_d, m0, chi0, A, B, V_ESC, depletion)

        rng_j = np.random.default_rng(31000 + sd)
        m0j, chi0j = sample_natal(rng_j, n_init, **CFG)
        jx = cascade_soft_jaxlike(rng_j, m0j, chi0j, A, B, V_ESC,
                                  depletion, n_steps)

        # also the packed-write-pointer prototype for M4 (layout invariance)
        rng_p = np.random.default_rng(31000 + sd)
        m0p, chi0p = sample_natal(rng_p, n_init, **CFG)
        pk = cascade_soft(rng_p, m0p, chi0p, A, B, V_ESC, depletion, n_steps)

        wd = np.ones(d["m_active"].size)
        wj = jx["occ_active"] * jx["w_active"]
        ks_m.append(weighted_ks(d["m_active"], wd, jx["m_active"], wj))
        ks_chi.append(weighted_ks(d["chi_active"], wd, jx["chi_active"], wj))
        dep.append(jx["depletion"]); wtot.append(jx["W_total"])
        dead.append(jx["n_dead_tail"])
        # layout invariance: jaxlike vs packed prototype, same seed
        wp = pk["occ_active"] * pk["w_active"]
        pkt.append(weighted_ks(jx["m_active"], wj, pk["m_active"], wp))

    Neff = (1 - depletion) * n_init
    ks_crit = 1.36 / np.sqrt(Neff / 2)
    print(f"  n_init={n_init}  n_steps={n_steps}  depletion target={depletion}")
    print()
    print(f"  M1 depletion (pinned):  mean={np.mean(dep):.5f}  "
          f"std={np.std(dep):.2e}   target={depletion}")
    print(f"  M2 W_total:             mean={np.mean(wtot):.1f}   "
          f"expect (1-D)*n_init={(1-depletion)*n_init:.0f}")
    print(f"  M3 KS(m) vs discrete:   mean={np.mean(ks_m):.4f}  "
          f"max={np.max(ks_m):.4f}   KS_crit(.05)={ks_crit:.4f}")
    print(f"     KS(chi) vs discrete: mean={np.mean(ks_chi):.4f}  "
          f"max={np.max(ks_chi):.4f}")
    print(f"  M4 KS(jaxlike vs packed prototype): mean={np.mean(pkt):.4f} "
          f"max={np.max(pkt):.4f}  (layout invariance -> ~0)")
    print(f"     mean dead tail slots: {np.mean(dead):.0f} of {n_steps} "
          f"(occupancy-0, contribute nothing)")
    print()
    ok = (abs(np.mean(dep) - depletion) < 1e-3
          and np.max(ks_m) < ks_crit and np.max(ks_chi) < ks_crit
          and np.max(pkt) < ks_crit)
    print("  VERDICT:", "PASS — JAX step logic validated in numpy mirror"
          if ok else "FAIL — investigate")
    print("=" * 72)


if __name__ == "__main__":
    main()
