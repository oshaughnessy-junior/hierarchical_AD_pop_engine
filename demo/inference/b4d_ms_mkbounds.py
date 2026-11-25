import argparse, json, os
import hier_map_laplace as H
WS = os.environ["HIER_WS"]; H.M_MAX_LO = 20.0; H.FREE_KAPPA = (-1.0, 7.0); H.KAPPA_INIT = 2.7; H.LOG_R_RANGE = (-8.0, 12.0)
a = argparse.Namespace(catalog=f"{WS}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz", pdet_grid="", pdet_z_grid="", likelihood="native4d", kappa=2.7,
    native4d_tables=f"{WS}/runs/A_tables/native4d_events.npz", native4d_tiers="primary", native4d_nodes=32, native4d_window=False, selection="injections",
    injection_file=f"{WS}/inputs/derived/gwtc5_o4ab_found.npz", event_subset="o4", f_dep=0.3, exclude_events=None)
R = f"{WS}/runs/BS4D_O4_ms"; os.makedirs(R, exist_ok=True)
for M in ("NULL", "BB", "BC", "BD", "BMIX", "NULLG", "REFRC"):
    b = H.build(a, M, 11, 3000, 3000, dry=True)["bounds"]; json.dump({k: [float(x) for x in v] for k, v in b.items()}, open(f"{R}/bounds_{M}.json", "w")); print(M, sorted(b))
