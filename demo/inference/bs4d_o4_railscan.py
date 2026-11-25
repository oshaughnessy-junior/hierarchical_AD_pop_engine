"""Scan lnL along a railed parameter p across its whole prior, the other parameters following the conditional mode of the
quadratic model at the fit (linear response, clipped to the prior), at n_seed 8000 / n_b 6000 and each key.
Saves p grid, lnL(p) per key and the conditional Hessian block. Post-processing integrates exp(lnL(p)) dp / width(p) with
the Gaussian volume of the remaining parameters.
Usage: bs4d_o4_railscan.py OUT_NPZ LAPDIR RESULT_JSON PARAM [RESULT_JSON PARAM ...]"""
import argparse, json, os, sys, time
import numpy as np, jax, jax.numpy as jnp
import hier_map_laplace as H

OUT, LAP = sys.argv[1], sys.argv[2]; items = sys.argv[3:]
WS = os.environ["HIER_WS"]; H.M_MAX_LO = 20.0; H.FREE_KAPPA = (-1.0, 7.0); H.KAPPA_INIT = 2.7; H.LOG_R_RANGE = (-8.0, 12.0)
KEYS, NS, NB = (11, 202, 3003), 8000, 6000


def ns(f_dep):
    return argparse.Namespace(catalog=f"{WS}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz", pdet_grid=f"{WS}/inputs/mass_pdet_grid_kappa2.7.npz",
                              pdet_z_grid=f"{WS}/inputs/combined_pdet_grid_mmin2_refinedz.npz", likelihood="native4d", kappa=2.7,
                              native4d_tables=f"{WS}/runs/A_tables/native4d_events.npz", native4d_tiers="primary", native4d_nodes=32,
                              native4d_window=False, selection="injections", injection_file=f"{WS}/inputs/derived/gwtc5_o4ab_found.npz",
                              event_subset="o4", f_dep=f_dep)


res = {}
for path, par in zip(items[::2], items[1::2]):
    r = json.load(open(path)); tag = os.path.basename(os.path.dirname(path)); names = r["names"]; ip = names.index(par)
    L0 = json.load(open(f"{LAP}/{tag}_n8000_k11.json")); A = -np.array(L0["hessian"]); th = np.array([r["theta_map"][n] for n in names])
    lo = np.array([r["bounds"][n][0] for n in names]); hi = np.array([r["bounds"][n][1] for n in names])
    rest = [i for i in range(len(names)) if i != ip]
    resp = -np.linalg.solve(A[np.ix_(rest, rest)], A[np.ix_(rest, [ip])]).ravel()      # d theta_rest / d p along the conditional mode
    grid = np.unique(np.concatenate([np.linspace(lo[ip], hi[ip], 17), [th[ip]]]))
    for key in KEYS:
        B = H.build(ns(r["f_dep"]), r["model"], key, NS, NB, r.get("final_proposal")); f = jax.jit(B["parts"]); t0 = time.time(); vals = []
        for g in grid:
            t = th.copy(); t[ip] = g; t[rest] = np.clip(th[rest] + resp * (g - th[ip]), lo[rest] + 1e-6, hi[rest] - 1e-6)
            vals.append(float(f(jnp.asarray(t))[0]))
        res[f"{tag}|{par}|{key}"] = np.array(vals); print(tag, par, key, "done", f"{time.time()-t0:.0f}s", np.round(np.array(vals) - max(vals), 1), flush=True)
    res[f"{tag}|{par}|grid"] = grid; res[f"{tag}|{par}|resp"] = resp; res[f"{tag}|{par}|A"] = A; res[f"{tag}|{par}|theta"] = th
    res[f"{tag}|{par}|lohi"] = np.array([lo[ip], hi[ip]])
    np.savez(OUT, **res)
