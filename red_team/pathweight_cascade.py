"""Path-weighted soft cascade: exact output population under a fixed natal proposal.

Design: runs/BS_weights/DESIGN.md (2026-11-28).

The soft cascade (:func:`backup_particle_mc.particle_mc_jax_soft`) draws pairs by
occupancy of its own slots. With natal draws from a proposal q and a target p, the
committed merger weight 0.5(w_i+w_j) is exact only for q = p. Here the cascade is
run ONCE on the q draws (the "shadow"; theta-independent, traced), and each slot
carries a path importance weight W_s: the ratio of the target-world to the
shadow-world probability of that slot's trajectory, in the large-N (mean-field)
limit where each slot meets an environment given by the weighted population.

Per shadow step k (pair i, j; intruder kk when exchange is on), with copy-level
occupancy totals Zq (shadow) and Zp (target, IS estimate) and the shadow clock
rescaled by Zp/Zq so that per-copy selection hazards agree:

    pair ratio      r_k = W_i W_j (Zq - o_i) / (Zp - o_i)
    intruder ratio  x_k = W_kk Zmq_ex / Zmp_ex          (exchange mode only)
    event weight    m_k = r_k x_k

Every slot whose state changes at step k (i, j, and kk on an exchange) and the
newborn remnant get W = m_k. The merger record weight is m_k * delta_k, capped
so that the target's own depletion budget f_dep * sum(W_natal) is met exactly.
In exchange mode the per-copy hazard of being exchanged in as an intruder
differs between the worlds; slots not involved at step k get the survival factor

    log W_s -= o_s iw_s p_exch(m_s / m_light_k) (r_k / Zmp_ex - 1 / Zmq_ex),

accumulated lazily on a log-mass grid (iw = intruder mass weight).

With W = 1 for every natal slot (proposal = target) all ratios are exactly 1, the
survival term is 0 and the records equal the committed cascade's.
Natal spins are reparameterised from frozen uniforms in the caller; the shadow
runs at a fixed reference spin scale, so spin enters the dynamics only through
the remnant mass, which this module recomputes for the records but not for the
pair sequence.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import red_team.backup_particle_mc as bpm


def check_supported():
    """the shadow trace covers the soft-gate options used by the B-series fits."""
    bad = []
    if bpm.EXCHANGE_MODE not in (0, 1, 2):
        bad.append(f"EXCHANGE_MODE={bpm.EXCHANGE_MODE}")
    if bpm.SEGREGATION_N != 0.0:
        bad.append("SEGREGATION_N")
    if bpm.KERNEL_2G_ON:
        bad.append("KERNEL_2G_ON")
    if bpm.SUMK_ON:
        bad.append("SUMK_ON")
    if bpm.DRIVEN_MODE:
        bad.append("DRIVEN_MODE")
    if bad:
        raise NotImplementedError("pathweight cascade does not support: " + ", ".join(bad))


def run_shadow(cfg, key, budget_scale=1.5, steps_scale=1.0, n_steps_mult=3.0):
    """theta-independent shadow cascade on the proposal draws, with its trace.

    budget_scale extends the shadow's depletion budget and steps_scale its scan
    length, so the target world (which stops at its own budget or after the
    committed number of steps on its own clock) can run past the shadow's stop."""
    check_supported()
    if cfg.orientation is not None and not bpm._orientation_is_aligned(cfg.orientation):
        raise NotImplementedError("pathweight cascade: Aligned orientation only")
    n_init = int(cfg.n_seed)
    n_target = max(1, int(cfg.depletion_fraction * n_init * float(n_steps_mult)))
    n_sh = n_target if steps_scale == 1.0 else int(round(n_target * float(steps_scale)))
    out = bpm.particle_mc_jax_soft(cfg, key, n_steps=n_sh, trace=True, budget_scale=float(budget_scale))
    if int(out["stats"]["n_init"]) != n_init:
        raise RuntimeError("proposal length differs from cfg.n_seed")
    tr = {k: np.asarray(v) for k, v in out["trace"].items()}
    st = out["stats"]
    return dict(tr=tr, m_slot=np.asarray(out["m_active"]), n_init=int(st["n_init"]),
                n_steps=int(st["n_steps"]), S=int(st["n_slots"]), n_steps_target=n_target,
                f_dep=float(cfg.depletion_fraction), budget_scale=float(budget_scale),
                exchange_mode=int(bpm.EXCHANGE_MODE), exchange_c=float(bpm.EXCHANGE_C),
                intr_n=float(bpm.EXCHANGE_INTRUDER_N), mref=float(bpm.SEGREGATION_MREF),
                shadow_depletion=float(st["depletion"]))


def _p_exch_np(m_k, m_light, mode, c):
    m_light = np.maximum(m_light, 1e-30)
    if mode == 2:
        return np.where(m_k > m_light, min(c, 1.0), 0.0)
    r72 = c * (m_k / m_light) ** 3.5
    return r72 / (1.0 + r72)


def make_evaluator(sh, n_grid=192):
    """returns evaluate(logw_natal, chi_natal) -> (records (T,4), weights (T,), diag).

    logw_natal: natal log IS weights log p/q (any normalisation); chi_natal: natal
    spins (reparameterised by the caller). Records are NaN where the shadow step
    was not an event, matching the committed cascade's ``mergers[:, :4]``."""
    tr, S, n_init, T = sh["tr"], sh["S"], sh["n_init"], sh["n_steps"]
    n_target = float(sh["n_steps_target"])
    exch = sh["exchange_mode"] != 0
    f_dep = sh["f_dep"]
    m_slot = sh["m_slot"]
    if sh["intr_n"] != 0.0:
        iw_np = (np.maximum(m_slot, 0.0) / sh["mref"]) ** sh["intr_n"]
    else:
        iw_np = np.ones_like(m_slot)
    f64 = lambda a: jnp.asarray(np.asarray(a, dtype=np.float64))
    i32 = lambda a: jnp.asarray(np.asarray(a, dtype=np.int32))
    xs = dict(t=i32(np.arange(T)), i=i32(tr["i"]), j=i32(tr["j"]), a=i32(tr["a"]), b=i32(tr["b"]),
              oi=f64(tr["o_i"]), oj=f64(tr["o_j"]), dl=f64(tr["delta"]), mult=f64(tr["mult"]),
              ret=f64(tr["retain"]), hia=jnp.asarray(tr["heavier_is_a"].astype(bool)),
              ev=jnp.asarray(tr["is_event"].astype(bool)), iwn=f64(iw_np[n_init + np.arange(T)]))
    iw = f64(iw_np)
    if exch:
        mpos = m_slot[m_slot > 0]
        lg = np.linspace(np.log(mpos.min()) - 1e-9, np.log(mpos.max()) + 1e-9, n_grid)
        fi = np.interp(np.log(np.maximum(m_slot, mpos.min())), lg, np.arange(n_grid, dtype=float))
        gi = np.clip(np.floor(fi).astype(np.int32), 0, n_grid - 2)
        gf = fi - gi
        gi_j, gf_j = jnp.asarray(gi), f64(gf)
        xs.update(kk=i32(tr["kk"]), ok=f64(tr["o_kk"]), isx=f64(tr["is_exch"].astype(float)),
                  trow=f64(_p_exch_np(np.exp(lg)[None, :], tr["m_light"][:, None],
                                      sh["exchange_mode"], sh["exchange_c"])))
    m0 = f64(m_slot)
    M_rem, chi_rem = bpm._M_rem_jax, bpm._chi_rem_jax

    def evaluate(logw_natal, chi_natal):
        lw = logw_natal - (logsumexp(logw_natal) - jnp.log(float(n_init)))
        logW = jnp.zeros(S, dtype=lw.dtype).at[:n_init].set(lw)
        W0 = jnp.exp(lw)
        Zp0 = jnp.sum(W0)
        Zq0 = jnp.sum(jnp.ones(n_init, dtype=lw.dtype))
        Zmq0 = jnp.sum(iw[:n_init]); Zmp0 = jnp.sum(W0 * iw[:n_init])
        chi = jnp.zeros(S, dtype=lw.dtype).at[:n_init].set(chi_natal)
        budget = jnp.asarray(f_dep) * Zp0
        phil = jnp.zeros(S, dtype=lw.dtype)
        Phi = jnp.zeros(n_grid if exch else 1, dtype=lw.dtype)

        def phi_at(Ph, s):
            return Ph[gi_j[s]] * (1.0 - gf_j[s]) + Ph[gi_j[s] + 1] * gf_j[s]

        def step(c, x):
            logW, phil, m, chi, Zq, Zp, Zmq, Zmp, Rp, Ph, lwmax, tau = c
            i, j, a, b = x["i"], x["j"], x["a"], x["b"]
            oi, oj, dl, mult = x["oi"], x["oj"], x["dl"], x["mult"]
            act = dl > 0.0
            if exch:
                kk, ok, isx = x["kk"], x["ok"], x["isx"]
                lWi = logW[i] - oi * iw[i] * (phi_at(Ph, i) - phil[i])
                lWj = logW[j] - oj * iw[j] * (phi_at(Ph, j) - phil[j])
                lWk = logW[kk] - ok * iw[kk] * (phi_at(Ph, kk) - phil[kk])
            else:
                lWi, lWj = logW[i], logW[j]
            Wi = jnp.exp(lWi)
            log_r = lWi + lWj + jnp.log(Zq - oi) - jnp.log(Zp - oi)
            if exch:
                Zmq_ex = Zmq - (oi * iw[i] + oj * iw[j])
                Zmp_ex = Zmp - (oi * iw[i] + oj * iw[j])
                log_m = log_r + lWk + jnp.log(Zmq_ex) - jnp.log(Zmp_ex)
                d = jnp.where(act, jnp.exp(log_r) / Zmp_ex - 1.0 / Zmq_ex, 0.0)
            else:
                log_m = log_r
            mk = jnp.exp(log_m)
            # target clock: Zp/Zq target steps per shadow step; the target stops
            # after the committed n_steps on its own clock (fraction at the crossing)
            rate = Zp / Zq
            fs = jnp.clip((n_target - tau) / rate, 0.0, 1.0)
            tau = tau + rate
            # record weight, capped at the target's own depletion budget
            drec = jnp.minimum(mk * dl * fs, jnp.maximum(budget - Rp, 0.0) / mult)
            Rp = Rp + mult * drec
            wrec = jnp.where(x["ev"], drec, 0.0)
            # merger record (merging pair a, b in the remnant-physics order)
            ma, mb, ca, cb = m[a], m[b], chi[a], chi[b]
            hia = x["hia"]
            rec = jnp.stack([jnp.where(hia, ma, mb), jnp.where(hia, mb, ma),
                             jnp.where(hia, ca, cb), jnp.where(hia, cb, ca)])
            rec = jnp.where(x["ev"], rec, jnp.full_like(rec, jnp.nan))
            r = n_init + x["t"]
            m = m.at[r].set(M_rem(ma, mb, ca, cb))
            chi = chi.at[r].set(chi_rem(ma, mb, ca, cb))
            # occupancy totals (shadow exact, target by the same event weights)
            Zq = Zq - mult * dl
            Zp = Zp - mk * mult * dl
            if exch:
                dZm = dl * (iw[i] + iw[j] + isx * iw[kk]) - x["ret"] * dl * x["iwn"]
                Zmq = Zmq - dZm
                Zmp = Zmp - mk * dZm
                Ph = Ph + x["trow"] * d
            # path weights of every slot whose state changed, and the remnant
            lnew_i = jnp.where(act, log_m, logW[i])
            lnew_j = jnp.where(act, log_m, logW[j])
            logW = logW.at[i].set(lnew_i).at[j].set(lnew_j).at[r].set(log_m)
            if exch:
                phil = phil.at[i].set(jnp.where(act, phi_at(Ph, i), phil[i]))
                phil = phil.at[j].set(jnp.where(act, phi_at(Ph, j), phil[j]))
                hit_k = act & (isx > 0.5)
                logW = logW.at[kk].set(jnp.where(hit_k, log_m, logW[kk]))
                phil = phil.at[kk].set(jnp.where(hit_k, phi_at(Ph, kk), phil[kk]))
                phil = phil.at[r].set(phi_at(Ph, r))
            lwmax = jnp.maximum(lwmax, jnp.where(act, log_m, -jnp.inf))
            return (logW, phil, m, chi, Zq, Zp, Zmq, Zmp, Rp, Ph, lwmax, tau), (rec, wrec)

        c0 = (logW, phil, m0, chi, Zq0, Zp0, Zmq0, Zmp0, jnp.zeros((), lw.dtype), Phi,
              jnp.asarray(-jnp.inf, lw.dtype), jnp.zeros((), lw.dtype))
        cf, (rec, wrec) = jax.lax.scan(step, c0, xs)
        diag = dict(budget_frac=cf[8] / budget, log_m_max=cf[10], Zp_end=cf[5], Zq_end=cf[4],
                    clock_frac=cf[11] / n_target)
        return rec, wrec, diag

    return evaluate
