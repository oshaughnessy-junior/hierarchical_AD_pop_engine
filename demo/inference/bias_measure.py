"""Cascade-weighting bias at the headline posterior (reassessment 2026-11-20).

The soft cascade draws pairs by occupancy (i.e. from the frozen natal PROPOSAL)
and gives a merger the weight 0.5(w_i + w_j).  That is exact only where the natal
importance weights are uniform (proposal == target).  This script compares, at
the same theta and sampler key:
  committed : the frozen wide proposal used by the headline fit (M.make_proposal)
  exact     : natal draws from the target itself (PL_PROP_REF recentring)
and records channel-A cascade diagnostics, detected-count predictions and the
3D-NAL log-likelihood (events and selection separately).
"""
import argparse, json, math, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import hier_map_laplace as H

WS = os.path.expanduser("~/20261120-hier-reassess")
COLS = ['alpha_A', 'm_max_A', 'spin_chi', 'mu_B', 'sig_B', 'mu_chiB', 'sig_chiB',
        'log_R_A', 'log_R_B', 'f_peakB', 'alpha_B']
M1_EDGES = np.array([5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 120, 200.])
HIST_EDGES = np.geomspace(5, 200, 41)


def thetas(post_path, n_draws):
    z = np.load(post_path, allow_pickle=True)
    out = [("mean", {k: float(np.asarray(z[k]).mean()) for k in COLS})]
    n = len(z["m_max_A"])
    for i in np.linspace(0, n - 1, n_draws).astype(int):
        out.append(("post_%03d" % i, {k: float(np.asarray(z[k])[i]) for k in COLS}))
    return out


def wq(x, w, qs):
    o = np.argsort(x); x = x[o]; c = np.cumsum(w[o]); c /= c[-1]
    return [float(np.interp(q, c, x)) for q in qs]


def runner(pl, mode, th, key, n_seed, n_b):
    """configure + build channel A (full output) and channel B for this mode/key."""
    prop = dict(alpha=0.0, m_max=th["m_max_A"], spin=th["spin_chi"]) if mode == "exact" else None
    H.configure(pl, "REFRC" if mode == "exact" else "REF", 0.7, key, prop)
    if mode == "committed":
        pl.PL_PROP_REF = None
    M = pl.M
    if pl.PL_PROP_REF is None:
        m_prop, chi_prop, log_q = M.make_proposal(n_seed)
    else:
        m_prop, chi_prop, log_q = pl._pl_ref_proposal(n_seed)

    def cfg(weight_array):
        return M.MCConfig(
            m_min=jnp.asarray(M.M_MIN), m_max=jnp.asarray(pl.M_MAX_PROP),
            natal_spin_loc=jnp.asarray(M.CFG_SPIN_LOC0),
            natal_spin_scale=jnp.asarray(M.CFG_SPIN_SCALE0),
            kernel_a=jnp.asarray(M.KERNEL_A), kernel_b=jnp.asarray(M.KERNEL_B),
            v_esc=jnp.asarray(1000.0), n_seed=int(n_seed),
            depletion_fraction=float(M.F_DEP),
            mu_natal=jnp.asarray(M.CFG_MU0), sigma_natal=jnp.asarray(M.CFG_SIGMA0),
            proposal=lambda _k: (m_prop, chi_prop, log_q), weight_array=weight_array)
    log_p_cfg = M._natal_log_pdf_jax(m_prop, chi_prop, cfg(None))
    log_p_t = pl._log_p_mass_pl_cutoff(m_prop, th["alpha_A"], th["m_max_A"]) + M.log_p_spin(chi_prop, th["spin_chi"])
    wa = jnp.exp(log_p_t - log_p_cfg)
    natal_w = np.asarray(jnp.exp(log_p_t - log_q))       # p_target / q
    out = M.particle_mc_jax_soft(cfg(wa), M.SAMPLER_KEY)
    # taint run: natal weight = 1[m > m_max_A]; dynamics do not depend on weights
    taint_w = jnp.where(m_prop > th["m_max_A"], 1.0, 0.0) * jnp.exp(log_q - log_p_cfg)
    out_t = M.particle_mc_jax_soft(cfg(taint_w), M.SAMPLER_KEY)
    sampB = pl.make_channelB_mix_sampler(n_b)
    sB, lwB = sampB(th["mu_B"], th["sig_B"], th["mu_chiB"], th["sig_chiB"], th["log_R_B"], th["f_peakB"], th["alpha_B"])
    return M, np.asarray(m_prop), natal_w, out, out_t, np.asarray(sB), np.asarray(lwB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", default="mean")
    ap.add_argument("--n-draws", type=int, default=4)
    ap.add_argument("--modes", default="committed,exact")
    ap.add_argument("--keys", default="11,202,3003,40004")
    ap.add_argument("--n-seed", type=int, default=8000)
    ap.add_argument("--n-b", type=int, default=6000)
    ap.add_argument("--post", default=f"{WS}/inputs/headline_combined_vesc1000_post.npz")
    ap.add_argument("--catalog", default=f"{WS}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz")
    ap.add_argument("--pdet-grid", default=f"{WS}/inputs/mass_pdet_grid_kappa2.7.npz")
    ap.add_argument("--outdir", default=f"{WS}/runs/C_bias")
    a = ap.parse_args()
    th_all = dict(thetas(a.post, a.n_draws))
    th = dict(th_all[a.theta]); th["alpha_A"] = 0.0
    pl = H.load_pl()
    from mass_pdet_interp import make_real_pdet_fn
    pdet = make_real_pdet_fn(a.pdet_grid)
    cat = np.load(a.catalog, allow_pickle=True)
    ce = jnp.asarray(cat["centers"]); cov = jnp.asarray(cat["covs"])
    prec = jnp.linalg.inv(0.5 * (cov + jnp.transpose(cov, (0, 2, 1))))
    _, logdet = jnp.linalg.slogdet(cov)
    names = [str(n) for n in cat["names"]]
    n_obs = len(names)

    for mode in a.modes.split(","):
        for key in [int(k) for k in a.keys.split(",")]:
            tag = f"{a.theta}_{mode}_k{key}"
            fn = os.path.join(a.outdir, tag + ".npz")
            if os.path.exists(fn):
                print("skip", tag, flush=True); continue
            t0 = time.time()
            M, m_prop, natal_w, out, out_t, sB, lwB = runner(pl, mode, th, key, a.n_seed, a.n_b)
            rec = np.asarray(out["mergers"]); act = np.asarray(out["step_active"]).astype(bool)
            wA = np.asarray(out["w_mergers"]) * act
            wT = np.asarray(out_t["w_mergers"]) * np.asarray(out_t["step_active"]).astype(bool)
            same_dyn = bool(np.array_equal(np.nan_to_num(rec), np.nan_to_num(np.asarray(out_t["mergers"]))))
            ok = act & np.isfinite(rec[:, 0]) & (wA > 0)
            m1, m2, c1, c2, g1, g2 = (rec[ok, i] for i in range(6))
            w = wA[ok] / wA[ok].sum(); tw = wT[ok]
            # isotropic projection exactly as the sampler does it
            sA = np.where(np.isfinite(rec[:, :4]), rec[:, :4], np.array([10., 8., 0., 0.]))
            cost = np.asarray(jax.random.uniform(M.SPIN_ORIENT_KEY, (sA.shape[0], 2), minval=-1.0, maxval=1.0))
            sA = sA.copy(); sA[:, 2] *= cost[:, 0]; sA[:, 3] *= cost[:, 1]
            lwA = np.log(np.maximum(wA, 1e-300)); lwA = lwA - logsumexp(lwA) + th["log_R_A"]
            ch = (sA[ok, 0] * sA[ok, 2] + sA[ok, 1] * sA[ok, 3]) / (sA[ok, 0] + sA[ok, 1])
            M1 = np.maximum(m1, m2); M2 = np.minimum(m1, m2); q = M2 / M1
            gmax = np.maximum(g1, g2); gmin = np.minimum(g1, g2)
            met = dict(
                n_mergers=int(ok.sum()),
                ess_merger=float(1.0 / np.sum(w ** 2)),
                natal_w_cv=float(np.std(natal_w) / np.mean(natal_w)),
                natal_frac_above_mmax_prop=float(np.mean(m_prop > th["m_max_A"])),
                same_dynamics_taint_run=same_dyn,
                frac_m1_gt={str(x): float(w[M1 > x].sum()) for x in (40, 60, 80)},
                gen_mix=dict(g1g1=float(w[gmax == 1].sum()), g1gN=float(w[(gmin == 1) & (gmax >= 2)].sum()),
                             gNgN=float(w[gmin >= 2].sum())),
                q_above40=dict(frac_q_gt_0p8=float(w[(M1 > 40) & (q > 0.8)].sum() / max(w[M1 > 40].sum(), 1e-300)),
                               quantiles=wq(q[M1 > 40], w[M1 > 40], [0.1, 0.5, 0.9]) if (M1 > 40).any() else None),
                taint_frac=float(w[tw > 0].sum()),
                taint_mean=float(np.sum(w * np.clip(tw / np.maximum(wA[ok] + tw, 1e-300), 0, 1))),
            )
            cw = []
            for lo, hi in zip(M1_EDGES[:-1], M1_EDGES[1:]):
                s = (M1 >= lo) & (M1 < hi)
                if w[s].sum() > 0:
                    mu_ = np.sum(w[s] * ch[s]) / w[s].sum()
                    cw.append([float(lo), float(hi), float(w[s].sum()), float(np.sqrt(np.sum(w[s] * (ch[s] - mu_) ** 2) / w[s].sum()))])
                else:
                    cw.append([float(lo), float(hi), 0.0, float("nan")])
            met["chieff_width_vs_m1"] = cw
            hist = np.histogram(M1, bins=HIST_EDGES, weights=w)[0] / np.diff(HIST_EDGES)
            # likelihood on the old 3D NAL catalog, kappa2.7 selection
            s_all = np.concatenate([sA, sB]); lw_all = np.concatenate([lwA, lwB]); nA = sA.shape[0]
            pd = np.asarray(pdet(jnp.asarray(s_all)))
            dw = np.exp(lw_all) * pd
            mu = float(dw.sum()); muA = float(dw[:nA].sum())
            x = M.mc_eta_chieff(jnp.asarray(s_all))
            logI = []
            for e0 in range(0, n_obs, 40):
                dz = x[None, :, :] - ce[e0:e0 + 40, None, :]
                qf = jnp.einsum("eni,eij,enj->en", dz, prec[e0:e0 + 40], dz)
                lm = -0.5 * qf - 0.5 * logdet[e0:e0 + 40, None] - 1.5 * jnp.log(2 * jnp.pi)
                logI.append(np.asarray(logsumexp(jnp.asarray(lw_all)[None, :] + lm, axis=1)))
                ess_part = None
            logI = np.concatenate(logI)
            ev = float(logI.sum())
            prof = ev - n_obs * math.log(mu) + n_obs * math.log(n_obs) - n_obs
            Mall1 = np.maximum(s_all[:, 0], s_all[:, 1]); Mall2 = np.minimum(s_all[:, 0], s_all[:, 1])
            qall = Mall2 / Mall1
            box = lambda a1, b1, a2, b2: float(dw[(Mall1 >= a1) & (Mall1 < b1) & (Mall2 >= a2) & (Mall2 < b2)].sum())
            sel = (Mall1 >= 20) & (Mall1 < 40)
            det = dict(mu=mu, mu_A=muA, Ndet_gt={str(x): float(dw[Mall1 > x].sum()) for x in (40, 60, 80)},
                       Ndet_gt_A={str(x): float(dw[:nA][Mall1[:nA] > x].sum()) for x in (40, 60, 80)},
                       boxes=dict(lowlow=box(5, 20, 5, 20), eq20_40=box(20, 40, 20, 40),
                                  hi_x_mid=box(40, 100, 20, 40), hi_hi=box(40, 100, 40, 100)),
                       frac_q80_20_40=float(dw[sel & (qall > 0.8)].sum() / max(dw[sel].sum(), 1e-300)))
            lnl = dict(events=ev, mu=mu, fixed_rate=ev - mu, profiled=prof)
            res = dict(theta_label=a.theta, theta=th, mode=mode, key=key, n_seed=a.n_seed, n_b=a.n_b,
                       metrics=met, detected=det, lnL=lnl, seconds=time.time() - t0)
            np.savez_compressed(fn, result=json.dumps(res), hist=hist, logI=logI, names=np.array(names))
            print(tag, f"{time.time()-t0:.0f}s prof={prof:.2f} ev={ev:.2f} mu={mu:.1f} f40={met['frac_m1_gt']['40']:.4f} taint={met['taint_frac']:.3f}", flush=True)


if __name__ == "__main__":
    main()
