"""Fixed-parameter per-event evaluations for the 4D B-series diagnosis (no fitting).

For one sampler key, evaluate at n_seed 8000 / n_b 6000 and save per-event log integrals:
  4D:   B_d, REFRC, Bmix at their v3 kappa 2.7 optima (native4d, kappa 2.7, window on)
  3D:   the same three at their 3D union optima (3D NAL likelihood, mass-only pdet grid)
  X:    B_d and REFRC 3D union optima under native4d (log_R shifted by -0.6365)
For every 4D-likelihood point, the 3D NAL log integrals of all events at the same draws are also
saved (used to put validation-failing events back on their NAL). For key 11, the predicted detected
distributions of Mc, eta, chi_eff (draw weights w*xi) and z (on the selection z grid) are saved.
Usage: bs4d_diag_eval.py KEY OUTDIR   (DIAG_WINDOW=0: exact per-event integrals, no Mc windows;
DIAG_ONLY4D=1: skip the 3D points)
"""
import json, os, sys, time, argparse
import numpy as np, jax, jax.numpy as jnp
import hier_map_laplace as H
import native4d_population as NP

KEY, OUT = int(sys.argv[1]), sys.argv[2]
WS = os.environ.get("HIER_WS", os.path.expanduser("~/20261120-hier-reassess"))
SC = f"{WS}/runs"
SHIFT = -0.6365
P4 = {"BD": "BS4D_scan/BD_f0.5", "REFRC": "BS4D_scan/REFRC_f0.3_e1m", "BMIX": "BS4D_scan/BMIX_f0.5"}
P3 = {k: os.environ.get("DIAG3D_PREFIX", "") + v for k, v in
      {"BD": "BS_scan_mmax20/BD_f0.5", "REFRC": "BS_scan/REFRC_f0.4", "BMIX": "BS_scan_mmax20/BMIX_f0.5"}.items()}
WINDOW = os.environ.get("DIAG_WINDOW", "1") == "1"
SUFFIX = "" if WINDOW else "_nowin"
ONLY4D = os.environ.get("DIAG_ONLY4D", "0") == "1"
H.M_MAX_LO = 20.0; H.FREE_KAPPA = None; H.LOG_R_RANGE = None


def args_for(lik, f_dep):
    return argparse.Namespace(catalog=f"{WS}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz",
                              pdet_grid=f"{WS}/inputs/mass_pdet_grid_kappa2.7.npz",
                              pdet_z_grid=f"{WS}/inputs/combined_pdet_grid_mmin2_refinedz.npz",
                              likelihood=lik, kappa=2.7, native4d_tables=f"{WS}/runs/A_tables/native4d_events.npz",
                              native4d_tiers="primary", native4d_nodes=32, native4d_window=WINDOW, f_dep=f_dep)


cat = np.load(f"{WS}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz", allow_pickle=True)
C3, V3 = np.asarray(cat["centers"]), np.asarray(cat["covs"])
out = {}


def ev4(tag, rel, shift=0.0):
    t0 = time.time(); r = json.load(open(f"{SC}/{rel}/result.json"))
    a = args_for("native4d", r["f_dep"])
    B = H.build(a, r["model"], KEY, 8000, 6000, r.get("final_proposal"))
    th = {k: float(r["theta_map"][k]) + (shift if k.startswith("log_R_") else 0.0) for k in B["names"]}
    full = dict(th); full.update(B["consts"])
    s, lw = B["comb"](**full)
    like = B["like"]
    L, d = like(s, lw, 2.7)
    Lp = B["parts"](jnp.asarray([th[k] for k in B["names"]]))[0]
    x = NP.mc_eta_chieff(s)
    nal = np.asarray(NP.nal3d_log_integrals(C3, V3, x, lw))           # (n_events, 2): log I, log sum a^2
    rec = dict(lnL=float(L), lnL_parts=float(Lp), mu=float(d["mu"]), T=float(like.T_obs), n=like.n,
               names4=np.array(like.names4), names3=np.array(like.names3),
               logI4=np.asarray(d["log_I4"]), ess4=np.asarray(d["ess4"]),
               logI3=np.asarray(d["log_I3"]), ess3=np.asarray(d["ess3"]),
               nal_logI=nal[:, 0], nal_ess=np.exp(2 * nal[:, 0] - nal[:, 1]),
               overflow=int(d["window_overflow"]), f_dep=r["f_dep"], model=r["model"], src=rel)
    if KEY == 11 and shift == 0.0:
        sel = like.sel; xi = np.asarray(sel.xi(s, 2.7)); w = np.exp(np.asarray(lw))
        rec.update(det_w=w * xi, det_x=np.asarray(x))
        # detected z distribution on the selection grid: sum_j w_j pdet(m1_j, m2_j, z_k) c_k
        a_, b_ = s[:, 0], s[:, 1]
        m1 = jnp.clip(jnp.maximum(a_, b_), sel.mmin, sel.mmax); m2 = jnp.clip(jnp.minimum(a_, b_), sel.mmin, sel.mmax)
        mg = jnp.exp(jnp.asarray(sel.lm)); n_ = mg.shape[0]
        i1 = jnp.clip(jnp.searchsorted(mg, m1, side="right") - 1, 0, n_ - 2); i2 = jnp.clip(jnp.searchsorted(mg, m2, side="right") - 1, 0, n_ - 2)
        t1 = jnp.clip((m1 - mg[i1]) / (mg[i1 + 1] - mg[i1]), 0, 1)[:, None]; t2 = jnp.clip((m2 - mg[i2]) / (mg[i2 + 1] - mg[i2]), 0, 1)[:, None]
        Pz = ((1 - t1) * (1 - t2) * sel.P[i1, i2] + t1 * (1 - t2) * sel.P[i1 + 1, i2]
              + (1 - t1) * t2 * sel.P[i1, i2 + 1] + t1 * t2 * sel.P[i1 + 1, i2 + 1])          # (ndraw, nz)
        cz = np.asarray(sel.zweights(2.7))
        rec.update(det_z_grid=np.asarray(sel.zg), det_z_w=np.asarray(jnp.einsum("j,jk->k", jnp.asarray(w), Pz)) * cz)
    out[tag] = rec
    print(f"{tag:12s} key={KEY} lnL={float(L):.3f} (parts {float(Lp):.3f}) mu={float(d['mu']):.2f} overflow={rec['overflow']} {time.time()-t0:.0f}s", flush=True)


def ev3(tag, rel):
    t0 = time.time(); r = json.load(open(f"{SC}/{rel}/result.json"))
    a = args_for("nal3d", r["f_dep"])
    B = H.build(a, r["model"], KEY, 8000, 6000, r.get("final_proposal"))
    th = {k: float(r["theta_map"][k]) for k in B["names"]}
    full = dict(th); full.update(B["consts"])
    s, lw = B["comb"](**full)
    Lp = B["parts"](jnp.asarray([th[k] for k in B["names"]]))[0]
    mu = float(jnp.sum(jnp.exp(lw) * B["pdet"](s)))
    nal = np.asarray(NP.nal3d_log_integrals(C3, V3, B["M"].mc_eta_chieff(s), lw))
    L = float(np.sum(nal[:, 0]) - mu)
    out[tag] = dict(lnL=L, lnL_parts=float(Lp), mu=mu, logI=nal[:, 0], ess=np.exp(2 * nal[:, 0] - nal[:, 1]),
                    f_dep=r["f_dep"], model=r["model"], src=rel)
    print(f"{tag:12s} key={KEY} lnL={L:.3f} (parts {float(Lp):.3f}) mu={mu:.2f} {time.time()-t0:.0f}s", flush=True)


for M, rel in P4.items(): ev4(f"4D_{M}", rel)
if not ONLY4D:
    for M, rel in P3.items(): ev3(f"3D_{M}", rel)
for M in ("BD", "REFRC"): ev4(f"X_{M}", P3[M], SHIFT)
flat = {f"{t}|{k}": v for t, rec in out.items() for k, v in rec.items()}
os.makedirs(OUT, exist_ok=True)
np.savez(f"{OUT}/diag_key{KEY}{SUFFIX}.npz", **flat)
print("saved", f"{OUT}/diag_key{KEY}{SUFFIX}.npz", flush=True)
