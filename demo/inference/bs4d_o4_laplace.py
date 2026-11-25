"""Laplace quantities of O4-only fits at fixed theta_map, per sampler key (no re-optimisation).
Saves lnL, mu, gradient, Hessian, eigenvalues and selection diagnostics per key to
<OUT>/<tag>_n<nseed>_k<key>.json (existing files are skipped). Volume terms are computed afterwards
from the saved Hessian and gradient (bs4d_o4_laplace_table.py).
Usage: bs4d_o4_laplace.py OUT N_SEED N_B KEY[,KEY...] RESULT_JSON [RESULT_JSON ...]"""
import argparse, json, os, sys, time
import numpy as np, jax, jax.numpy as jnp
import hier_map_laplace as H

OUT, NS, NB = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
KEYS = [int(k) for k in sys.argv[4].split(",")]; os.makedirs(OUT, exist_ok=True)
WS = os.environ["HIER_WS"]
H.M_MAX_LO = 20.0; H.FREE_KAPPA = (-1.0, 7.0); H.KAPPA_INIT = 2.7; H.LOG_R_RANGE = (-8.0, 12.0)


def ns(f_dep):
    return argparse.Namespace(catalog=f"{WS}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz",
                              pdet_grid=f"{WS}/inputs/mass_pdet_grid_kappa2.7.npz",
                              pdet_z_grid=f"{WS}/inputs/combined_pdet_grid_mmin2_refinedz.npz", likelihood="native4d", kappa=2.7,
                              native4d_tables=f"{WS}/runs/A_tables/native4d_events.npz", native4d_tiers="primary",
                              native4d_nodes=32, native4d_window=False, selection="injections",
                              injection_file=f"{WS}/inputs/derived/gwtc5_o4ab_found.npz", event_subset="o4", f_dep=f_dep)


for path in sys.argv[5:]:
    r = json.load(open(path)); tag = os.path.basename(os.path.dirname(path))
    for key in KEYS:
        dst = f"{OUT}/{tag}_n{NS}_k{key}.json"
        if os.path.exists(dst): continue
        t0 = time.time()
        B = H.build(ns(r["f_dep"]), r["model"], key, NS, NB, r.get("final_proposal"))
        assert list(B["names"]) == list(r["names"])
        th = np.array([r["theta_map"][n] for n in B["names"]])
        for n in B["names"]:                      # the fit's bounds must match this build's
            assert np.allclose(B["bounds"][n], r["bounds"][n]), (n, B["bounds"][n], r["bounds"][n])
        (L, (mu, muA)), g = jax.jit(jax.value_and_grad(B["parts"], has_aux=True))(jnp.asarray(th))
        lap = H.laplace(B, th, float(L))
        sel = B["sel_diag"](jnp.asarray(th))
        out = dict(tag=tag, model=r["model"], f_dep=r["f_dep"], key=key, n_seed=NS, n_b=NB, names=list(B["names"]),
                   theta=th.tolist(), bounds={k: list(v) for k, v in B["bounds"].items()}, lnL=float(L), mu=float(mu),
                   grad=np.asarray(g).tolist(), selection=sel, wall_s=time.time() - t0, **lap)
        json.dump(out, open(dst + ".tmp", "w"), indent=1); os.replace(dst + ".tmp", dst)
        print(f"{tag} n{NS} key {key}: lnL {float(L):.3f} lnZ_bdy {lap['lnZ_laplace_bdy']:.3f} pd {lap['negH_pd']} "
              f"eig0 {min(lap['negH_eigs']):.3g} |grad|max {np.max(np.abs(np.asarray(g))):.3g} N_eff {sel['neff']:.0f} "
              f"({out['wall_s']:.0f}s)", flush=True)
