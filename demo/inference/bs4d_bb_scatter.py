"""Per-event ln I, ESS and lnL across sampler keys at fixed points (native4d, O4 subset, injection selection).
Usage: bs4d_bb_scatter.py OUT_NPZ N_SEED N_B KEY[,KEY...] RESULT_JSON [RESULT_JSON ...]"""
import argparse, json, math, os, sys, time
import numpy as np, jax, jax.numpy as jnp
import hier_map_laplace as H
import native4d_population as NP

OUT, NS, NB = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); KEYS = [int(k) for k in sys.argv[4].split(",")]
WS = os.environ["HIER_WS"]; H.M_MAX_LO = 20.0; H.FREE_KAPPA = (-1.0, 7.0); H.KAPPA_INIT = 2.7; H.LOG_R_RANGE = (-8.0, 12.0)


def ns(f_dep):
    return argparse.Namespace(catalog=f"{WS}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz",
                              pdet_grid=f"{WS}/inputs/mass_pdet_grid_kappa2.7.npz",
                              pdet_z_grid=f"{WS}/inputs/combined_pdet_grid_mmin2_refinedz.npz", likelihood="native4d", kappa=2.7,
                              native4d_tables=f"{WS}/runs/A_tables/native4d_events.npz", native4d_tiers="primary",
                              native4d_nodes=32, native4d_window=False, selection="injections",
                              injection_file=f"{WS}/inputs/derived/gwtc5_o4ab_found.npz", event_subset="o4", f_dep=f_dep)


res = {}
for path in sys.argv[5:]:
    r = json.load(open(path)); tag = os.path.basename(os.path.dirname(path))
    for key in KEYS:
        t0 = time.time()
        B = H.build(ns(r["f_dep"]), r["model"], key, NS, NB, r.get("final_proposal"))
        th = {k: float(r["theta_map"][k]) for k in B["names"]}; full = dict(th); full.update(B["consts"]); kap = full.pop("kappa")
        s, lw = B["comb"](**full); L, d = B["like"](s, lw, kap)
        p = f"{tag}|{NS}|{key}"
        res[f"{p}|lnL"] = float(L); res[f"{p}|mu"] = float(d["mu"]); res[f"{p}|lnI"] = np.asarray(d["log_I4"])
        res[f"{p}|ess"] = np.asarray(d["ess4"]); res[f"{tag}|names"] = np.array(B["like"].names4)
        print(f"{p} lnL {float(L):.3f} mu {float(d['mu']):.2f} min ESS {np.min(np.asarray(d['ess4'])):.2f} ({time.time()-t0:.0f}s)", flush=True)
        np.savez(OUT, **res)
