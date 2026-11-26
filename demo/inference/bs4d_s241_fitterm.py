"""S241011k's per-event term at fixed population points (no fitting).
For one result.json (an ALL168 fit; NO241011 fits are evaluated with the ALL168 event set) and configs of
(cascade key, field seed, draw multiplier): n_seed 24000, channel B draws n_b = 18000 * mult and channel C draws N_C = 2500 * mult.
Records for S241011k: lnI of the 4D fit term (kappa of the point), lnI of the fit's 3D projection (z weighted by dVc/dz/(1+z), i.e. kappa 0),
lnI of the 3D NAL Gaussian, the draw ESS of each, and the five draws carrying most of the 4D term (channel, Mc, eta, chi_eff).
Saves the draws (Mc_source, eta, chi_eff, log w) for the sample-based comparison, and the full lnL at mult 1.
Usage: bs4d_s241_fitterm.py OUT_NPZ RESULT_JSON"""
import argparse, json, os, sys, time
import numpy as np, jax, jax.numpy as jnp
from jax.scipy.special import logsumexp
import hier_map_laplace as H
import native4d_population as NP
import native4d_likelihood as N4

OUT, RP = sys.argv[1], sys.argv[2]; r = json.load(open(RP)); WS = os.environ["HIER_WS"]
H.M_MAX_LO = 20.0; H.FREE_KAPPA = (-1.0, 7.0); H.KAPPA_INIT = 2.7; H.LOG_R_RANGE = (-8.0, 12.0)
MULT = {"N_C": 1}
_orig = H.configure
def configure(pl, *a, **k):                       # raise channel C draws with the multiplier (shared code unchanged)
    out = _orig(pl, *a, **k); pl.N_C = 2500 * MULT["N_C"]; return out
H.configure = configure
args = argparse.Namespace(catalog=f"{WS}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz", pdet_grid=f"{WS}/inputs/mass_pdet_grid_kappa2.7.npz",
    pdet_z_grid=f"{WS}/inputs/combined_pdet_grid_mmin2_refinedz.npz", likelihood="native4d", kappa=2.7, native4d_tables=f"{WS}/runs/A_tables/native4d_events.npz",
    native4d_tiers="primary", native4d_nodes=32, native4d_window=False, selection="injections", injection_file=f"{WS}/inputs/derived/gwtc5_o4ab_found.npz",
    event_subset="o4", f_dep=r["f_dep"], exclude_events=None)
cat = np.load(args.catalog, allow_pickle=True); ci = [str(n) for n in cat["names"]].index("S241011k")
C3, V3 = cat["centers"][ci:ci + 1], cat["covs"][ci:ci + 1]
CONF = [(k, None, 1) for k in (50005, 60006, 70007)] + [(50005, s, 1) for s in (1, 2, 3)] + [(50005, s, 10) for s in (None, 1, 2, 3)]
res = dict(np.load(OUT, allow_pickle=True)) if os.path.exists(OUT) else {}
for key, fs, mult in CONF:
    nm = f"{key}|{fs}|{mult}"
    if f"{nm}|lnI4" in res: continue
    t0 = time.time(); H.FIELD_SEED = fs; MULT["N_C"] = mult
    B = H.build(args, r["model"], key, 24000, 18000 * mult, r.get("final_proposal"))
    names = B["names"]; th = {n: float(r["theta_map"][n]) for n in names}; full = dict(th); full.update(B["consts"]); kap = full.pop("kappa")
    s, lw = B["comb"](**full); like = B["like"]; x = NP.mc_eta_chieff(s); j = like.names4.index("S241011k")
    P = jax.tree_util.tree_map(lambda a: a[j], like.P); ti, li = like.t[j], like.lwt[j]
    def per_draw(kp):
        ln = NP.log_norm_pz(kp); z = jnp.exp(ti)[None, :] / x[:, 0:1] - 1.0; dc = N4.comoving_distance(jnp.clip(z, 0.0, 5.9))
        pts = jnp.concatenate([jnp.broadcast_to(x[:, None, :], z.shape + (3,)), z[..., None]], -1)
        ll = N4.log_likelihood(pts, P, dc); ll = jnp.where(jnp.isfinite(ll), ll, NP._MASKED)
        return logsumexp(li[None, :] + ti[None, :] - jnp.log(x[:, 0:1]) + NP.log_pz(z, kp, ln, dc) + ll, 1)
    c4 = np.asarray(lw + per_draw(kap)); c3 = np.asarray(lw + per_draw(0.0))
    nal = NP.nal3d_log_integrals(C3, V3, x, lw); prec = np.linalg.inv(0.5 * (V3[0] + V3[0].T)); dd = np.asarray(x) - C3[0]
    cn = np.asarray(lw) + (-0.5 * np.einsum("ni,ij,nj->n", dd, prec, dd) - 0.5 * np.linalg.slogdet(V3[0])[1] - 1.5 * np.log(2 * np.pi))
    ess = lambda c: float(np.exp(2 * np.logaddexp.reduce(c) - np.logaddexp.reduce(2 * c)))
    nA = B["nA"]; nB = 18000 * mult; lab = np.where(np.arange(len(c4)) < nA, "A", np.where(np.arange(len(c4)) < nA + nB, "B", "C"))
    top = np.argsort(-c4)[:5]; tot = float(np.logaddexp.reduce(c4)); X = np.asarray(x)
    res.update({f"{nm}|lnI4": tot, f"{nm}|ess4": ess(c4), f"{nm}|lnI3fit": float(np.logaddexp.reduce(c3)), f"{nm}|ess3fit": ess(c3),
                f"{nm}|lnI3nal": float(np.logaddexp.reduce(cn)), f"{nm}|ess3nal": ess(cn), f"{nm}|kappa": kap,
                f"{nm}|top": json.dumps([dict(i=int(i), ch=str(lab[i]), Mc=float(X[i, 0]), eta=float(X[i, 1]), chi=float(X[i, 2]), share=float(np.exp(c4[i] - tot))) for i in top]),
                f"{nm}|draws": np.column_stack([X, np.asarray(lw)]).astype(np.float32)})
    if mult == 1:
        L, d = like(s, lw, kap); res[f"{nm}|lnL"] = float(L)
    res["tag"] = os.path.basename(os.path.dirname(RP)); res["model"] = r["model"]
    np.savez(OUT + ".tmp.npz", **res); os.replace(OUT + ".tmp.npz", OUT)
    print(os.path.basename(OUT), nm, "lnI4 %.2f ess %.2f | lnI3fit %.2f ess %.2f | NAL %.2f ess %.1f | top %s (%.0fs)" % (tot, ess(c4), float(np.logaddexp.reduce(c3)), ess(c3), float(np.logaddexp.reduce(cn)), ess(cn), "/".join(f"{lab[i]}:{X[i,2]:.2f}" for i in top[:3]), time.time() - t0), flush=True)
