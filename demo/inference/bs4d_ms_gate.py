"""Gate for the multi-start campaign. Usage: gate_ms.py"""
import argparse, json, math, os, sys
import numpy as np, jax, jax.numpy as jnp
import hier_map_laplace as H
WS = os.environ["HIER_WS"]; H.M_MAX_LO = 20.0; H.FREE_KAPPA = (-1.0, 7.0); H.KAPPA_INIT = 2.7; H.LOG_R_RANGE = (-8.0, 12.0)
def ns(f, excl=None):
    return argparse.Namespace(catalog=f"{WS}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz", pdet_grid=f"{WS}/inputs/mass_pdet_grid_kappa2.7.npz",
        pdet_z_grid=f"{WS}/inputs/combined_pdet_grid_mmin2_refinedz.npz", likelihood="native4d", kappa=2.7, native4d_tables=f"{WS}/runs/A_tables/native4d_events.npz",
        native4d_tiers="primary", native4d_nodes=32, native4d_window=False, selection="injections", injection_file=f"{WS}/inputs/derived/gwtc5_o4ab_found.npz",
        event_subset="o4", f_dep=f, exclude_events=excl)
out = {}
SC = f"{WS}/runs/BS4D_O4/scan"
for tag in ("BMIX_f0.3_e1m", "BC_f0.3_e2m", "BB_f0.4", "BD_f0.3", "REFRC_f0.3", "NULLG_f0.2_e1m", "NULL_f0.3"):
    r = json.load(open(f"{SC}/{tag}/result.json"))
    for fs, excl in ((None, None), (1, None), (None, ["S241011k"])):
        H.FIELD_SEED = fs
        B = H.build(ns(r["f_dep"], excl), r["model"], 11, 3000, 3000, r["final_proposal"])
        th = jnp.asarray([r["theta_map"][k] for k in B["names"]])
        (L, (mu, _)), g = jax.jit(jax.value_and_grad(B["parts"], has_aux=True))(th)
        print(tag, "field_seed", fs, "excl", excl, "lnL %.6f mu %.3f n_obs %d grad finite %s" % (float(L), float(mu), B["n_obs"], bool(np.all(np.isfinite(np.asarray(g))))), flush=True)
        if fs is None and excl is None: print("   stored lnL %.6f -> match: %s" % (r["lnL"], abs(float(L) - r["lnL"]) < 1e-6), flush=True)
    H.FIELD_SEED = None
