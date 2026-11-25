"""lnL at fixed converged points (free kappa, O4 injection selection): for n_seed 8000 / n_b 6000 and 24000 / 18000,
(a) cascade keys 50005/60006/70007 with the default static draws, and (b) field seeds 1/2/3 (re-seeded static draws) at key 50005.
None of these keys or field seeds were used in any fit. Per-event ln I and ESS are saved, per evaluation.
Usage: bs4d_ms_eval.py OUT_NPZ DATASET RESULT_JSON"""
import argparse, json, os, sys, time
import numpy as np, jax, jax.numpy as jnp
import hier_map_laplace as H

OUT, DS, RP = sys.argv[1], sys.argv[2], sys.argv[3]; r = json.load(open(RP)); WS = os.environ["HIER_WS"]
H.M_MAX_LO = 20.0; H.FREE_KAPPA = (-1.0, 7.0); H.KAPPA_INIT = 2.7; H.LOG_R_RANGE = (-8.0, 12.0)
args = argparse.Namespace(catalog=f"{WS}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz", pdet_grid=f"{WS}/inputs/mass_pdet_grid_kappa2.7.npz",
    pdet_z_grid=f"{WS}/inputs/combined_pdet_grid_mmin2_refinedz.npz", likelihood="native4d", kappa=2.7, native4d_tables=f"{WS}/runs/A_tables/native4d_events.npz",
    native4d_tiers="primary", native4d_nodes=32, native4d_window=False, selection="injections", injection_file=f"{WS}/inputs/derived/gwtc5_o4ab_found.npz",
    event_subset="o4", f_dep=r["f_dep"], exclude_events=["S241011k"] if DS == "NO241011" else None)
EV = [(k, None) for k in (50005, 60006, 70007)] + [(50005, s) for s in (1, 2, 3)]
res = dict(np.load(OUT, allow_pickle=True)) if os.path.exists(OUT) else {}
for ns, nb in ((8000, 6000), (24000, 18000)):
    for key, fs in EV:
        name = f"{ns}|{key}|{fs}"
        if f"{name}|lnL" in res: continue
        t0 = time.time(); H.FIELD_SEED = fs
        B = H.build(args, r["model"], key, ns, nb, r.get("final_proposal"))
        th = {n: float(r["theta_map"][n]) for n in B["names"]}; full = dict(th); full.update(B["consts"]); kap = full.pop("kappa"); s, lw = B["comb"](**full)
        L, d = B["like"](s, lw, kap)
        dg = B["sel_diag"](jnp.asarray([th[n] for n in B["names"]]))
        res.update({f"{name}|lnL": float(L), f"{name}|mu": float(d["mu"]), f"{name}|lnI": np.asarray(d["log_I4"]), f"{name}|ess": np.asarray(d["ess4"]),
                    f"{name}|neff": dg["neff"], f"{name}|mc_var_lnL": dg["mc_var_lnL"], "names": np.array(B["like"].names4), "tag": os.path.basename(os.path.dirname(RP)), "f_dep": r["f_dep"]})
        np.savez(OUT + ".tmp.npz", **res); os.replace(OUT + ".tmp.npz", OUT)
        print(os.path.basename(OUT), ns, key, fs, "lnL %.2f neff %.0f minESS %.2f (%.0fs)" % (float(L), dg["neff"], float(np.min(np.asarray(d["ess4"]))), time.time() - t0), flush=True)
