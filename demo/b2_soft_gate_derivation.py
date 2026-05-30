r"""B.2 soft-gate derivation — numpy sandbox (no JAX needed).

Purpose
-------
Settle the B.2 design BEFORE touching the JAX `particle_mc_jax`.  Three
open questions from `project-jax-mc-b2-soft-gate-followup` /
`PLAN_jax_mc_time_decoupling_diagnosis.md`:

  Q1.  Does the `k_ratio = K_max/<K>` magic factor actually VANISH under
       a deterministic soft K-gate, or does it merely transmute?
  Q2.  Does the soft gate's surviving (m, chi) distribution match the
       discrete K-rejection reference (numpy `particle_mc` semantics)?
  Q3.  Is the soft gate smooth in the kernel parameters (so AD has no
       straight-through bias), unlike the discrete Bernoulli gate?

Method
------
Two self-contained numpy cascades sharing IDENTICAL simplified physics
(so the ONLY thing under comparison is the GATE mechanism, not the
remnant fits — those are shared by both production paths and are
orthogonal to the gate question):

  * `cascade_discrete`  — mirrors numpy `particle_mc`: draw a pair,
    Bernoulli-accept with prob K_ij/K_max, full merge on accept, stop
    at n_max_mergers = D * n_init accepted events.

  * `cascade_soft`      — the B.2 candidate: deterministic occupancy
    flow.  Pre-allocated slots (n_init parents + n_steps remnant tail).
    Each step consumes delta = w_event * min(o_i, o_j) of BOTH parents
    and (if retained) is born into a tail slot with occupancy delta.
    A cumulative net-removal BUDGET pins total depletion to D * n_init
    EXACTLY (clip the last contributing step), so n_steps only needs to
    be a generous UPPER BOUND, not a tuned count.  <-- the k_ratio claim.

The simplified physics (identical on both sides):
  * kernel_K = (m1+m2)^a q^b   -- the REAL kernel (drives K-weighting).
  * remnant mass m_new = (m_i+m_j)*(1 - eps_rad)   (eps_rad=0.05).
  * remnant spin chi_new = clip(0.7 - 0.1*(1-q), 0, 0.99) -- bounded, q-dep.
  * retention: deterministic Heaviside  retain = (v_kick_proxy < v_esc),
    v_kick_proxy = V0 * (1-q)   (asymmetric pairs kick harder).  This is
    the SAME deterministic-given-pair structure as retention_aligned.

We compare the GATE, holding physics fixed.  This is the same sandbox
methodology used in j2_pool_scaling_v2.py.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Shared simplified physics (identical on both gate paths).
# ---------------------------------------------------------------------------
def kernel_K(m1, m2, a, b):
    m_big = np.maximum(m1, m2)
    m_sml = np.minimum(m1, m2)
    q = m_sml / m_big
    return (m1 + m2) ** a * q ** b


def kernel_upper_bound(m_active, a, b):
    mlo = float(np.min(m_active))
    mhi = float(np.max(m_active))
    cands = [
        kernel_K(2 * mhi, 2 * mhi, a, b),
        kernel_K(2 * mlo, 2 * mlo, a, b),
        kernel_K(mhi + mlo, mhi + mlo, a, b),
    ]
    return float(max(cands)) * 1.05


EPS_RAD = 0.05
V0_KICK = 400.0


def remnant(m_i, m_j, chi_i, chi_j):
    """Deterministic simplified remnant (m_new, chi_new, v_kick_proxy)."""
    m_big = max(m_i, m_j)
    m_sml = min(m_i, m_j)
    q = m_sml / m_big
    m_new = (m_i + m_j) * (1.0 - EPS_RAD)
    chi_new = min(max(0.7 - 0.1 * (1.0 - q), 0.0), 0.99)
    v_kick = V0_KICK * (1.0 - q)
    return m_new, chi_new, v_kick


# ---------------------------------------------------------------------------
# Natal sampler (shared).
# ---------------------------------------------------------------------------
def sample_natal(rng, n_init, mu, sigma, m_min, m_max, chi_loc, chi_scale):
    m = rng.normal(mu, sigma, size=n_init)
    m = np.clip(m, m_min, m_max)
    chi = np.clip(rng.normal(chi_loc, chi_scale, size=n_init), 0.0, 1.0)
    return m, chi


# ---------------------------------------------------------------------------
# Reference: discrete K-rejection (numpy particle_mc semantics).
# ---------------------------------------------------------------------------
def cascade_discrete(rng, m0, chi0, a, b, v_esc, depletion):
    n_init = m0.size
    m = m0.copy()
    chi = chi0.copy()
    gen = np.ones(n_init, dtype=int)
    active = np.ones(n_init, dtype=bool)
    n_max_mergers = max(1, int(depletion * n_init))
    n_accepted = 0
    n_attempts = 0
    w_mergers = []
    max_attempts = 200 * n_max_mergers + 10000
    while n_accepted < n_max_mergers and n_attempts < max_attempts:
        idx = np.flatnonzero(active)
        if idx.size < 2:
            break
        K_max = kernel_upper_bound(m[idx], a, b)
        i, j = rng.choice(idx, size=2, replace=False)
        n_attempts += 1
        K_ij = kernel_K(m[i], m[j], a, b)
        if rng.random() >= K_ij / K_max:
            continue
        # accepted
        n_accepted += 1
        m_new, chi_new, v_kick = remnant(m[i], m[j], chi[i], chi[j])
        retain = v_kick < v_esc
        heavy, light = (i, j) if m[i] >= m[j] else (j, i)
        w_mergers.append(0.5 * (1.0 + 1.0))  # uniform natal weights
        if retain:
            m[heavy] = m_new
            chi[heavy] = chi_new
            gen[heavy] = max(gen[i], gen[j]) + 1
            active[light] = False
        else:
            active[i] = False
            active[j] = False
    surv = np.flatnonzero(active)
    return dict(
        m_active=m[surv], chi_active=chi[surv], gen_active=gen[surv],
        occ_active=np.ones(surv.size),
        n_accepted=n_accepted, n_attempts=n_attempts,
        depletion=1.0 - surv.size / n_init,
        W_total=float(surv.size), n_init=n_init,
        w_mergers=np.asarray(w_mergers),
    )


# ---------------------------------------------------------------------------
# Candidate: deterministic soft occupancy-flow gate (B.2).
# ---------------------------------------------------------------------------
def cascade_soft(rng, m0, chi0, a, b, v_esc, depletion, n_steps,
                 budget_clip=True, pair_seq=None, record_pairs=False):
    """Deterministic soft K-gate with slot pre-allocation + depletion budget.

    Slots: [0:n_init] natal parents (occupancy 1), [n_init:n_init+n_steps]
    remnant tail (occupancy 0, filled as the cascade runs).

    Per step:
      * draw pair (i, j) from occupied slots with prob proportional to
        occupancy o (the soft analog of numpy's uniform-over-active draw).
      * w_event = K_ij / K_max.
      * delta = w_event * min(o_i, o_j).
      * if budget_clip: clip delta so cumulative NET removal never exceeds
        the target R_target = depletion * n_init; once exhausted delta=0.
      * o_i -= delta;  o_j -= delta.
      * retain (deterministic): tail slot born with occupancy delta at the
        remnant (m_new, chi_new); net removal += delta  (2 consumed-1 born).
      * not retain: no tail; net removal += 2*delta.
    """
    n_init = m0.size
    S = n_init + n_steps
    m = np.zeros(S)
    chi = np.zeros(S)
    o = np.zeros(S)
    gen = np.zeros(S, dtype=int)
    w = np.zeros(S)
    m[:n_init] = m0
    chi[:n_init] = chi0
    o[:n_init] = 1.0
    gen[:n_init] = 1
    w[:n_init] = 1.0

    R_target = depletion * n_init
    net_removed = 0.0
    tail_ptr = n_init
    w_mergers = []
    recorded = []
    tiny = 1e-12

    for step in range(n_steps):
        occ_mask = o > tiny
        if occ_mask.sum() < 2 or (budget_clip and net_removed >= R_target - tiny):
            if record_pairs:
                recorded.append((-1, -1))
            continue
        occ_idx = np.flatnonzero(occ_mask)
        m_active = m[occ_idx]
        K_max = kernel_upper_bound(m_active, a, b)
        if pair_seq is not None:
            # REPLAY mode: forced indices (the AD-faithful proxy — JAX
            # random.choice indices are constants w.r.t. theta).  An index
            # that has been depleted contributes delta->0 via min(o_i,o_j).
            i, j = pair_seq[step]
            if i < 0 or j < 0:
                continue
        else:
            # occupancy-weighted pair draw
            p = o[occ_idx] / o[occ_idx].sum()
            i = int(occ_idx[rng.choice(occ_idx.size, p=p)])
            p2 = p.copy()
            p2[occ_idx == i] = 0.0
            if p2.sum() <= 0:
                if record_pairs:
                    recorded.append((-1, -1))
                continue
            p2 = p2 / p2.sum()
            j = int(occ_idx[rng.choice(occ_idx.size, p=p2)])
        if record_pairs:
            recorded.append((i, j))

        K_ij = kernel_K(m[i], m[j], a, b)
        w_event = K_ij / K_max
        delta = w_event * min(o[i], o[j])
        if delta <= tiny:
            continue

        m_new, chi_new, v_kick = remnant(m[i], m[j], chi[i], chi[j])
        retain = v_kick < v_esc
        net_this = delta if retain else 2.0 * delta

        if budget_clip and net_removed + net_this > R_target:
            # clip delta so net removal lands exactly on target
            scale = (R_target - net_removed) / net_this
            delta *= scale
            net_this *= scale

        o[i] -= delta
        o[j] -= delta
        w_mergers.append(0.5 * (w[i] + w[j]) * delta)  # occupancy-scaled
        if retain and delta > tiny:
            m[tail_ptr] = m_new
            chi[tail_ptr] = chi_new
            o[tail_ptr] = delta
            w[tail_ptr] = 0.5 * (w[i] + w[j])
            gen[tail_ptr] = max(gen[i], gen[j]) + 1
            tail_ptr += 1
        net_removed += net_this

    occ_final = o > tiny
    W_total = float(np.sum(o[occ_final] * w[occ_final]))
    return dict(
        m_active=m[occ_final], chi_active=chi[occ_final],
        gen_active=gen[occ_final], occ_active=o[occ_final],
        w_active=w[occ_final],
        net_removed=net_removed, R_target=R_target,
        depletion=net_removed / n_init, n_init=n_init,
        W_total=W_total, tail_used=tail_ptr - n_init,
        w_mergers=np.asarray(w_mergers),
        recorded=recorded,
    )


def cascade_discrete_replay(m0, chi0, a, b, v_esc, pair_seq, accept_u):
    """Discrete K-rejection replayed on a FIXED (i,j) sequence + fixed
    accept uniforms.  This is the discrete mirror of the soft replay:
    the only b-dependence is through accept = (u < K_ij(b)/K_max(b)),
    which is a STEP function of b -> the observable jumps as accepts flip.
    Returns (occupancy-weighted) mean surviving mass and the accept vector."""
    n_init = m0.size
    m = m0.copy()
    chi = chi0.copy()
    active = np.ones(n_init, dtype=bool)
    accepts = np.zeros(len(pair_seq), dtype=bool)
    for step, (i, j) in enumerate(pair_seq):
        # Skip tail-slot (2G+) indices the soft cascade produced — the
        # discrete reference keeps remnants in the heavier parent's slot,
        # so it has no tail index space.  The 1G subset suffices for the
        # accept-flip discreteness demonstration.
        if i < 0 or j < 0 or i >= n_init or j >= n_init:
            continue
        if not (active[i] and active[j]):
            continue
        idx = np.flatnonzero(active)
        K_max = kernel_upper_bound(m[idx], a, b)
        K_ij = kernel_K(m[i], m[j], a, b)
        if accept_u[step] >= K_ij / K_max:
            continue
        accepts[step] = True
        m_new, chi_new, v_kick = remnant(m[i], m[j], chi[i], chi[j])
        retain = v_kick < v_esc
        heavy, light = (i, j) if m[i] >= m[j] else (j, i)
        if retain:
            m[heavy] = m_new
            chi[heavy] = chi_new
            active[light] = False
        else:
            active[i] = False
            active[j] = False
    surv = np.flatnonzero(active)
    return float(np.mean(m[surv])), accepts


# ---------------------------------------------------------------------------
# Weighted empirical CDF / KS distance between two weighted samples.
# ---------------------------------------------------------------------------
def weighted_ks(x1, w1, x2, w2):
    order = np.argsort(np.concatenate([x1, x2]))
    allx = np.concatenate([x1, x2])[order]
    c1 = np.zeros_like(allx)
    c2 = np.zeros_like(allx)
    # build weighted CDFs on the pooled support
    s1 = np.argsort(x1); xs1 = x1[s1]; ws1 = w1[s1] / w1.sum()
    s2 = np.argsort(x2); xs2 = x2[s2]; ws2 = w2[s2] / w2.sum()
    cdf1 = np.searchsorted(xs1, allx, side="right")
    cdf2 = np.searchsorted(xs2, allx, side="right")
    F1 = np.cumsum(ws1)[np.clip(cdf1 - 1, 0, len(ws1) - 1)]
    F1 = np.where(cdf1 == 0, 0.0, F1)
    F2 = np.cumsum(ws2)[np.clip(cdf2 - 1, 0, len(ws2) - 1)]
    F2 = np.where(cdf2 == 0, 0.0, F2)
    return float(np.max(np.abs(F1 - F2)))


# ---------------------------------------------------------------------------
# Stage-1-like config.
# ---------------------------------------------------------------------------
CFG = dict(mu=8.0, sigma=1.0, m_min=4.0, m_max=80.0,
           chi_loc=0.0, chi_scale=0.10)
A, B, V_ESC = 0.0, 1.0, 1000.0


def run_Q1_kratio(depletion=0.10, n_init=2000, seeds=8):
    """Q1: does target depletion track R_target independent of n_steps,
    once n_steps is a generous upper bound?  If yes, k_ratio is gone."""
    print("=" * 72)
    print("Q1 — does the depletion BUDGET remove the k_ratio magic factor?")
    print("=" * 72)
    # First measure the discrete reference's empirical K_max/<K> so we know
    # what the OLD k_ratio would have to have been.
    rng = np.random.default_rng(0)
    m0, chi0 = sample_natal(rng, n_init, **CFG)
    K_max = kernel_upper_bound(m0, A, B)
    # <K> over uniform pairs
    ii = rng.integers(0, n_init, 20000)
    jj = rng.integers(0, n_init, 20000)
    Kmean = float(np.mean(kernel_K(m0[ii], m0[jj], A, B)))
    print(f"  natal pool: K_max={K_max:.3f}, <K>={Kmean:.3f}, "
          f"K_max/<K> = {K_max / Kmean:.2f}   (this is the OLD k_ratio)")
    print()
    print(f"  target depletion D = {depletion}  ->  R_target = "
          f"{depletion * n_init:.0f} net removals")
    print()
    print(f"  {'n_steps':>9} {'n_steps/(D*n_init)':>20} "
          f"{'mean_depletion':>15} {'std':>8} {'mean_tail_used':>15}")
    for mult in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        n_steps = int(mult * depletion * n_init)
        deps, tails = [], []
        for s in range(seeds):
            rng = np.random.default_rng(100 + s)
            m0, chi0 = sample_natal(rng, n_init, **CFG)
            out = cascade_soft(rng, m0, chi0, A, B, V_ESC, depletion, n_steps)
            deps.append(out["depletion"])
            tails.append(out["tail_used"])
        print(f"  {n_steps:>9} {mult:>20.1f} {np.mean(deps):>15.4f} "
              f"{np.std(deps):>8.4f} {np.mean(tails):>15.1f}")
    print()
    print("  READ: if mean_depletion -> D and flattens once n_steps exceeds")
    print("  ~K_max/<K> * D*n_init, the budget clip pins depletion EXACTLY")
    print("  and n_steps is just a 'big enough' upper bound (no precision")
    print("  knob).  That is the k_ratio-elimination claim.")
    print()


def run_Q2_shape(depletion=0.10, n_init=3000, seeds=12):
    """Q2: surviving (m, chi) marginal parity soft vs discrete."""
    print("=" * 72)
    print("Q2 — surviving (m, chi) marginal parity: soft vs discrete")
    print("=" * 72)
    ks_m, ks_chi, dep_d, dep_s, wtot_d, wtot_s = [], [], [], [], [], []
    n_steps = int(3.0 * depletion * n_init)  # generous upper bound
    for s in range(seeds):
        rng_d = np.random.default_rng(7000 + s)
        m0, chi0 = sample_natal(rng_d, n_init, **CFG)
        d = cascade_discrete(rng_d, m0, chi0, A, B, V_ESC, depletion)
        rng_s = np.random.default_rng(7000 + s)
        m0s, chi0s = sample_natal(rng_s, n_init, **CFG)
        sft = cascade_soft(rng_s, m0s, chi0s, A, B, V_ESC, depletion, n_steps)
        wd = np.ones(d["m_active"].size)
        ws = sft["occ_active"] * sft["w_active"]
        ks_m.append(weighted_ks(d["m_active"], wd, sft["m_active"], ws))
        ks_chi.append(weighted_ks(d["chi_active"], wd, sft["chi_active"], ws))
        dep_d.append(d["depletion"]); dep_s.append(sft["depletion"])
        wtot_d.append(d["W_total"]); wtot_s.append(sft["W_total"])
    print(f"  depletion:  discrete={np.mean(dep_d):.4f}  "
          f"soft={np.mean(dep_s):.4f}")
    print(f"  W_total:    discrete={np.mean(wtot_d):.1f}  "
          f"soft={np.mean(wtot_s):.1f}  "
          f"(expect ~ (1-D)*n_init = {(1-depletion)*n_init:.0f})")
    print(f"  KS(m)   mean={np.mean(ks_m):.4f}  max={np.max(ks_m):.4f}")
    print(f"  KS(chi) mean={np.mean(ks_chi):.4f}  max={np.max(ks_chi):.4f}")
    print("  (KS ~ 1/sqrt(N_eff) floor; both samples are finite MC draws.)")
    # crude two-sample KS critical value at alpha=0.05
    Neff = (1 - depletion) * n_init
    print(f"  KS_crit(0.05) ~ 1.36/sqrt(N_eff/2) = "
          f"{1.36 / np.sqrt(Neff / 2):.4f}")
    print()


def run_Q3_smoothness(depletion=0.10, n_init=2000, db=2e-3):
    """Q3: AD-faithful smoothness in kernel_b.

    The honest AD proxy holds the SELECTED pair sequence fixed across b
    (JAX random.choice indices are constants w.r.t. theta — gradients do
    NOT flow through which pair is chosen, only through the values given
    the pair).  We:
      1. record the soft cascade's (i,j) sequence at b0,
      2. pre-draw fixed accept uniforms u for the discrete mirror,
      3. sweep b holding BOTH fixed, and compare the observable's
         smoothness.
    Soft: delta = (K_ij/K_max)*min(o_i,o_j) is C0-smooth in b -> clean FD.
    Discrete: accept = 1{u < K_ij/K_max} is a STEP in b -> jumps as
    accepts flip; this is exactly the discontinuity the B.1 straight-
    through estimator papers over.
    """
    print("=" * 72)
    print("Q3 — AD-faithful smoothness in kernel_b (fixed pair sequence)")
    print("=" * 72)
    n_steps = int(3.0 * depletion * n_init)
    b0 = 1.0

    # 1. reference pair sequence at b0 (no budget clip -> isolate value
    #    smoothness; clip adds a piecewise-smooth term, tested separately).
    rng = np.random.default_rng(424242)
    m0, chi0 = sample_natal(rng, n_init, **CFG)
    ref = cascade_soft(rng, m0, chi0, A, b0, V_ESC, depletion, n_steps,
                       budget_clip=False, record_pairs=True)
    pair_seq = ref["recorded"]
    rng_u = np.random.default_rng(999)
    accept_u = rng_u.random(len(pair_seq))

    def soft_obs(b):
        out = cascade_soft(None, m0, chi0, A, b, V_ESC, depletion, n_steps,
                           budget_clip=False, pair_seq=pair_seq)
        wgt = out["occ_active"] * out["w_active"]
        return float(np.sum(out["m_active"] * wgt) / np.sum(wgt))

    def disc_obs(b):
        val, _ = cascade_discrete_replay(m0, chi0, A, b, V_ESC,
                                         pair_seq, accept_u)
        return val

    bs = [b0 + k * db for k in (-2, -1, 0, 1, 2)]
    sv = [soft_obs(b) for b in bs]
    dv = [disc_obs(b) for b in bs]
    print(f"  {'b':>8} {'soft_obs':>14} {'disc_obs':>14}")
    for b, s, d in zip(bs, sv, dv):
        print(f"  {b:>8.4f} {s:>14.8f} {d:>14.8f}")
    fd_soft = (sv[3] - sv[1]) / (2 * db)
    fd_disc = (dv[3] - dv[1]) / (2 * db)
    soft_curv = abs(sv[0] - 2 * sv[2] + sv[4])
    disc_curv = abs(dv[0] - 2 * dv[2] + dv[4])
    # count accept flips across a fine b-sweep -> the discreteness measure
    _, acc_lo = cascade_discrete_replay(m0, chi0, A, b0 - 5 * db, V_ESC,
                                        pair_seq, accept_u)
    _, acc_hi = cascade_discrete_replay(m0, chi0, A, b0 + 5 * db, V_ESC,
                                        pair_seq, accept_u)
    n_flips = int(np.sum(acc_lo != acc_hi))
    print()
    print(f"  FD d<obs>/db   soft = {fd_soft:+.5f}   discrete = {fd_disc:+.5f}")
    print(f"  |2nd diff| (smaller=smoother): soft={soft_curv:.3e}  "
          f"discrete={disc_curv:.3e}")
    print(f"  discrete accept-flips over b in [b0-0.01, b0+0.01]: {n_flips}")
    # coarse sweep: show the discrete observable STEPS over a wider range
    coarse = [0.80, 0.90, 1.00, 1.10, 1.20]
    cs = [soft_obs(b) for b in coarse]
    cd = [disc_obs(b) for b in coarse]
    _, acc0 = cascade_discrete_replay(m0, chi0, A, 0.80, V_ESC, pair_seq, accept_u)
    _, acc1 = cascade_discrete_replay(m0, chi0, A, 1.20, V_ESC, pair_seq, accept_u)
    print(f"  coarse sweep b in [0.8,1.2]: discrete accept-flips = "
          f"{int(np.sum(acc0 != acc1))}")
    print(f"  {'b':>6} {'soft':>12} {'discrete':>12}")
    for b, s, d in zip(coarse, cs, cd):
        print(f"  {b:>6.2f} {s:>12.6f} {d:>12.6f}")
    print("  READ: with the pair sequence held fixed (the AD-faithful")
    print("  condition), the soft observable is a SMOOTH function of b")
    print("  (small 2nd diff); the discrete mirror is piecewise-constant and")
    print("  jumps at each accept flip -> its gradient is 0 a.e. and the")
    print("  flips are exactly where B.1's straight-through estimator is")
    print("  biased.  B.2's forward pass is already smooth: no estimator")
    print("  needed.")
    print()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    run_Q1_kratio()
    run_Q2_shape()
    run_Q3_smoothness()
    print("=" * 72)
    print("DONE — see READ notes under each block for the verdict.")
    print("=" * 72)
