"""Gate for the O4-only (168-event) refit with injection selection, at the v3 free-kappa optima of B_d and
REFRC: finite value and gradient, N_eff and MC variance of mu, split-half agreement of mu.
Usage: bs4d_o4_gate.py OUT_JSON"""
import json, math, os, sys, argparse
import numpy as np, jax, jax.numpy as jnp
import hier_map_laplace as H
import native4d_population as NP

WS = os.environ.get("HIER_WS", os.path.expanduser("~/20261120-hier-reassess"))
PTS = {"BD": "BS4D_scan/BD_f0.5_kw12", "REFRC": "BS4D_scan/REFRC_f0.3_kw12_e2p"}
H.M_MAX_LO = 20.0; H.FREE_KAPPA = (-1.0, 7.0); H.KAPPA_INIT = 2.7; H.LOG_R_RANGE = (-8.0, 12.0)
T_OLD = 3.0456693791669833
args = argparse.Namespace(catalog=f"{WS}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz",
                          pdet_grid=f"{WS}/inputs/mass_pdet_grid_kappa2.7.npz",
                          pdet_z_grid=f"{WS}/inputs/combined_pdet_grid_mmin2_refinedz.npz", likelihood="native4d", kappa=2.7,
                          native4d_tables=f"{WS}/runs/A_tables/native4d_events.npz", native4d_tiers="primary", native4d_nodes=32,
                          native4d_window=False, selection="injections", injection_file=f"{WS}/inputs/derived/gwtc5_o4ab_found.npz",
                          event_subset="o4", f_dep=0.5)
out = {}
for M, rel in PTS.items():
    r = json.load(open(f"{WS}/runs/{rel}/result.json")); args.f_dep = r["f_dep"]
    B0 = H.build(args, r["model"], 11, 3000, 3000, dry=True)
    T_new = float(np.load(args.injection_file)["T_obs_s"]) / (365.25 * 86400)
    shift = math.log(B0["n_obs"] / 236.0) + math.log(T_OLD / T_new)
    th = {k: float(r["theta_map"][k]) + (shift if k.startswith("log_R_") else 0.0) for k in B0["names"]}
    B = H.build(args, r["model"], 11, 3000, 3000, r.get("final_proposal"))
    t = jnp.asarray([th[k] for k in B["names"]])
    (L, (mu, muA)), g = jax.jit(jax.value_and_grad(B["parts"], has_aux=True))(t)
    g = np.asarray(g); diag = B["sel_diag"](t)
    # split-half mu (random half of the found injections each; each half estimates with N_gen/2)
    full = dict(th); full.update(B["consts"]); kap = full.pop("kappa"); s, lw = B["comb"](**full)
    sel = B["like"].injsel; lv = np.asarray(sel.log_v(s, lw, kap)); c = sel.T_obs_years / sel.NG
    rng = np.random.default_rng(12345); half = rng.permutation(len(lv)) < len(lv) // 2
    def est(mask):
        v = np.exp(lv[mask]); m = 2 * c * v.sum(); var = (2 * c) ** 2 * (v ** 2).sum() - m * m / (sel.NG / 2); return m, math.sqrt(max(var, 0))
    (m1, s1), (m2, s2) = est(half), est(~half)
    out[M] = dict(src=rel, n_obs=B["n_obs"], lnL=float(L), mu=float(mu), mu_A=float(muA), grad_finite=bool(np.all(np.isfinite(g))),
                  grad={k: float(x) for k, x in zip(B["names"], g)}, selection=diag,
                  split_half=dict(mu1=m1, sd1=s1, mu2=m2, sd2=s2, z=(m1 - m2) / math.sqrt(s1 ** 2 + s2 ** 2)),
                  start_theta=th, log_R_shift=shift)
    print(M, json.dumps({k: v for k, v in out[M].items() if k not in ("grad", "start_theta")}, default=float), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
