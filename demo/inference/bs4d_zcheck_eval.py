"""Redshift checks of the native4d path at the v3 kappa 2.7 optima of B_d and REFRC (fixed parameters).

For key 11, n_seed 8000 / n_b 6000, exact per-event integrals (no Mc windows):
  - predicted detected z, from the engine's selection grid (neural baseline + low-mass extension,
    bilinear in source masses as ZSelection.xi) and from the grid file's embedded native
    injection histogram (injection_pdet: sum over found GWTC-5 injections of 1/p_draw, per cell);
  - the detected-z weight above z = 1 by source primary mass;
  - per event, the population-informed z posterior (weights over draws x z nodes of the event
    integral), stacked over the 189 4D-fit events, and its median per event;
  - a kappa profile at fixed shape parameters with the rate profiled out,
    lnL_prof(kappa) = sum_i ln I_i(kappa) - N ln mu(kappa) + N ln N - N, for both selections.
Usage: bs4d_zcheck_eval.py OUTDIR
"""
import json, os, sys, time, argparse
import numpy as np, jax, jax.numpy as jnp
from jax.scipy.special import logsumexp
import hier_map_laplace as H
import native4d_population as NP
import native4d_likelihood as N4

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
WS = os.environ.get("HIER_WS", os.path.expanduser("~/20261120-hier-reassess"))
H.M_MAX_LO = 20.0; H.FREE_KAPPA = None; H.LOG_R_RANGE = None
PTS = {"BD": "BS4D_scan/BD_f0.5", "REFRC": "BS4D_scan/REFRC_f0.3_e1m"}
KAPPAS = [1.0, 2.0, 2.7, 3.5, 4.5, 5.5]
ZB = np.linspace(0, 2.5, 51)
grid = np.load(f"{WS}/inputs/combined_pdet_grid_mmin2_refinedz.npz", allow_pickle=True)
ME, ZE, IP = grid["injection_mass_edges"], grid["injection_z_edges"], grid["injection_pdet"]
ZC = 0.5 * (ZE[1:] + ZE[:-1])


def a_ns(f_dep):
    return argparse.Namespace(catalog=f"{WS}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz",
                              pdet_grid=f"{WS}/inputs/mass_pdet_grid_kappa2.7.npz",
                              pdet_z_grid=f"{WS}/inputs/combined_pdet_grid_mmin2_refinedz.npz",
                              likelihood="native4d", kappa=2.7, native4d_tables=f"{WS}/runs/A_tables/native4d_events.npz",
                              native4d_tiers="primary", native4d_nodes=32, native4d_window=False, f_dep=f_dep)


def pz_grid(sel, s):
    """(ndraw, nz) grid pdet at source (m1, m2), as ZSelection.xi interpolates."""
    a_, b_ = s[:, 0], s[:, 1]
    m1 = jnp.clip(jnp.maximum(a_, b_), sel.mmin, sel.mmax); m2 = jnp.clip(jnp.minimum(a_, b_), sel.mmin, sel.mmax)
    mg = jnp.exp(jnp.asarray(sel.lm)); n_ = mg.shape[0]
    i1 = jnp.clip(jnp.searchsorted(mg, m1, side="right") - 1, 0, n_ - 2); i2 = jnp.clip(jnp.searchsorted(mg, m2, side="right") - 1, 0, n_ - 2)
    t1 = jnp.clip((m1 - mg[i1]) / (mg[i1 + 1] - mg[i1]), 0, 1)[:, None]; t2 = jnp.clip((m2 - mg[i2]) / (mg[i2 + 1] - mg[i2]), 0, 1)[:, None]
    return np.asarray((1 - t1) * (1 - t2) * sel.P[i1, i2] + t1 * (1 - t2) * sel.P[i1 + 1, i2]
                      + (1 - t1) * t2 * sel.P[i1, i2 + 1] + t1 * t2 * sel.P[i1 + 1, i2 + 1])


def pz_inj(s):
    """(ndraw, nzcell) injection-histogram pdet, nearest cell (ordered masses, clamped to the edges)."""
    m1 = np.clip(np.maximum(s[:, 0], s[:, 1]), ME[0], ME[-1] - 1e-9); m2 = np.clip(np.minimum(s[:, 0], s[:, 1]), ME[0], ME[-1] - 1e-9)
    i1 = np.searchsorted(ME, m1, "right") - 1; i2 = np.searchsorted(ME, m2, "right") - 1
    return IP[i1, i2, :]


def pz_weights(kappa, zc, ze):
    """population p(z|kappa) mass in each injection z cell (normalized on (0, 2.5])."""
    x, w = np.polynomial.legendre.leggauss(16)
    out = []
    for a, b in zip(ze[:-1], ze[1:]):
        zz = (a + b) / 2 + (b - a) / 2 * x
        out.append(np.sum((b - a) / 2 * w * np.exp(np.asarray(NP.log_pz(jnp.asarray(zz), kappa)))))
    return np.array(out)


res = {}
for M, rel in PTS.items():
    t0 = time.time(); r = json.load(open(f"{WS}/runs/{rel}/result.json"))
    B = H.build(a_ns(r["f_dep"]), r["model"], 11, 8000, 6000, r.get("final_proposal"))
    th = {k: float(r["theta_map"][k]) for k in B["names"]}; full = dict(th); full.update(B["consts"])
    s, lw = B["comb"](**full); like = B["like"]; sel = like.sel
    s_np, w = np.asarray(s), np.exp(np.asarray(lw)); x = NP.mc_eta_chieff(s); mc = np.asarray(x[:, 0])
    m1s = np.maximum(s_np[:, 0], s_np[:, 1])
    Pg = pz_grid(sel, s); Pi = pz_inj(s_np)
    rec = dict(m1_max=float(m1s.max()), frac_w_m1_gt_120=float(np.sum(w[m1s > 120]) / np.sum(w)))
    # detected z (kappa 2.7) from both selections
    cg = np.asarray(sel.zweights(2.7)); ci = pz_weights(2.7, ZC, ZE)
    dg = (w[:, None] * Pg) * cg[None, :]; di = (w[:, None] * Pi) * ci[None, :]
    rec.update(z_grid=np.asarray(sel.zg), det_grid=dg.sum(0), z_inj=ZC, det_inj=di.sum(0),
               mu_grid=float(like.T_obs * dg.sum()), mu_inj=float(like.T_obs * di.sum()))
    mb = [0, 20, 40, 60, 80, 120, 1e9]
    hi_g = dg[:, np.asarray(sel.zg) > 1.0].sum(1); hi_i = di[:, ZC > 1.0].sum(1)
    rec["hiz_by_m1"] = [dict(m1_lo=lo, m1_hi=hi, frac_det_grid=float(hi_g[(m1s >= lo) & (m1s < hi)].sum() / dg.sum()),
                             frac_det_inj=float(hi_i[(m1s >= lo) & (m1s < hi)].sum() / di.sum())) for lo, hi in zip(mb[:-1], mb[1:])]
    # per-event population-informed z posteriors (4D events, exact)
    ln = NP.log_norm_pz(2.7)

    @jax.jit
    def ev_hist(P, ti, li):
        z = jnp.exp(ti)[None, :] / x[:, 0:1] - 1.0
        dc = N4.comoving_distance(jnp.clip(z, 0.0, 5.9))
        pts = jnp.concatenate([jnp.broadcast_to(x[:, None, :], z.shape + (3,)), z[..., None]], -1)
        ll = N4.log_likelihood(pts, P, dc); ll = jnp.where(jnp.isfinite(ll), ll, NP._MASKED)
        a = lw[:, None] + li[None, :] + ti[None, :] - jnp.log(x[:, 0:1]) + NP.log_pz(z, 2.7, ln, dc) + ll
        wt = jnp.exp(a - logsumexp(a)).ravel(); zz = z.ravel()
        h = jnp.histogram(zz, bins=jnp.asarray(ZB), weights=wt)[0]
        o = jnp.argsort(zz); cw = jnp.cumsum(wt[o]); zmed = zz[o][jnp.searchsorted(cw, 0.5)]
        return h, zmed
    hs, meds = [], []
    for i in range(like.n4):
        Pi_ = jax.tree_util.tree_map(lambda a: a[i], like.P)
        h, zm = ev_hist(Pi_, like.t[i], like.lwt[i]); hs.append(np.asarray(h)); meds.append(float(zm))
    rec.update(stack=np.mean(hs, 0), pop_medians=np.array(meds), names4=np.array(like.names4))
    # kappa profile, rate profiled, shape parameters fixed
    prof = []
    for k in KAPPAS:
        e4 = NP.event_log_integrals(like.P, like.t, like.lwt, x, lw, k)[:, 0]
        e3 = NP.nal3d_log_integrals(like.centers3, like.covs3, x, lw)[:, 0]
        S = float(jnp.sum(e4) + jnp.sum(e3)); N = like.n
        mug = float(like.T_obs * np.sum(w * np.asarray(sel.xi(s, k))))
        mui = float(like.T_obs * np.sum(w[:, None] * Pi * pz_weights(k, ZC, ZE)[None, :]))
        prof.append(dict(kappa=k, sum_lnI=S, mu_grid=mug, mu_inj=mui,
                         lnLprof_grid=S + N * np.log(like.T_obs) - N * np.log(mug) + N * np.log(N) - N,
                         lnLprof_inj=S + N * np.log(like.T_obs) - N * np.log(mui) + N * np.log(N) - N))
    rec["kappa_profile"] = prof
    res[M] = rec
    print(M, f"{time.time()-t0:.0f}s mu_grid={rec['mu_grid']:.2f} mu_inj={rec['mu_inj']:.2f} m1max={rec['m1_max']:.0f} frac_w(m1>120)={rec['frac_w_m1_gt_120']:.2e}", flush=True)
    for p in prof: print("   kappa %.1f lnLprof grid %.2f inj %.2f" % (p["kappa"], p["lnLprof_grid"], p["lnLprof_inj"]), flush=True)

flat = {}
for M, rec in res.items():
    for k, v in rec.items():
        flat[f"{M}|{k}"] = json.dumps(v) if isinstance(v, list) else v
np.savez(f"{OUT}/zcheck_key11.npz", **flat)
print("saved", flush=True)
