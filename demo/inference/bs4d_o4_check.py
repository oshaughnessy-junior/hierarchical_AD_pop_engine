"""Checks at the best O4 points (fixed parameters, key 11, n_seed 8000 / n_b 6000, injection selection):
  - kappa profile 1..5 with shape parameters fixed and the rate profiled out, N_eff per kappa;
  - predicted detected z (injection weights) vs stacked population-informed event z posteriors;
  - predicted detected Mc and chi_eff (draw weight w_j * xi_j = d mu / d ln w_j) vs catalog centres;
  - the same point under --event-subset o4all (188 events, the 20 without fits on their 3D NAL, no thinning).
Usage: bs4d_o4_check.py SCAN_DIR TAG [TAG ...] OUT_NPZ"""
import json, math, os, sys, argparse
import numpy as np, jax, jax.numpy as jnp
from jax.scipy.special import logsumexp
import hier_map_laplace as H
import native4d_population as NP
import native4d_likelihood as N4

SC, TAGS, OUT = sys.argv[1], sys.argv[2:-1], sys.argv[-1]
WS = os.environ.get("HIER_WS", os.path.expanduser("~/20261120-hier-reassess"))
H.M_MAX_LO = 20.0; H.FREE_KAPPA = (-1.0, 7.0); H.KAPPA_INIT = 2.7; H.LOG_R_RANGE = (-8.0, 12.0)
ZB = np.linspace(0, 2.5, 51); KAP = [1.0, 2.0, 3.0, 4.0, 5.0]


def ns(subset, f_dep):
    return argparse.Namespace(catalog=f"{WS}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz",
                              pdet_grid=f"{WS}/inputs/mass_pdet_grid_kappa2.7.npz",
                              pdet_z_grid=f"{WS}/inputs/combined_pdet_grid_mmin2_refinedz.npz", likelihood="native4d", kappa=2.7,
                              native4d_tables=f"{WS}/runs/A_tables/native4d_events.npz", native4d_tiers="primary",
                              native4d_nodes=32, native4d_window=False, selection="injections",
                              injection_file=f"{WS}/inputs/derived/gwtc5_o4ab_found.npz", event_subset=subset, f_dep=f_dep)


out = {}
for tag in TAGS:
    r = json.load(open(f"{SC}/{tag}/result.json")); th0 = r["theta_map"]; rec = {}
    for subset in ("o4", "o4all"):
        B = H.build(ns(subset, r["f_dep"]), r["model"], 11, 8000, 6000, r.get("final_proposal"))
        t = jnp.asarray([th0[k] for k in B["names"]]); L, (mu, muA) = jax.jit(B["parts"])(t)
        rec[f"{subset}_lnL"] = float(L); rec[f"{subset}_mu"] = float(mu); rec[f"{subset}_n"] = B["n_obs"]
        rec[f"{subset}_sel"] = B["sel_diag"](t)
        if subset != "o4":
            continue
        like = B["like"]; sel = like.injsel; full = dict(th0); full.update(B["consts"]); kfit = full.pop("kappa")
        s, lw = B["comb"](**full); x = NP.mc_eta_chieff(s); N = like.n
        prof = []
        for k in KAP:
            e4 = NP.event_log_integrals(like.P, like.t, like.lwt, x, lw, k)[:, 0]
            S = float(jnp.sum(e4)); lmu = float(sel.log_mu(s, lw, k)); dg = sel.diagnostics(s, lw, k, N)
            prof.append(dict(kappa=k, sum_lnI=S, mu=math.exp(lmu), neff=float(dg["neff"]),
                             lnLprof=S + N * math.log(like.T_obs) + N * math.log(N) - N - N * lmu))
        rec["kappa_profile"] = prof; rec["kappa_fit"] = float(kfit)
        lv = np.asarray(sel.log_v(s, lw, kfit)); v = np.exp(lv - lv.max())
        rec["det_z_hist"] = np.histogram(np.asarray(sel.z), bins=ZB, weights=v)[0]
        dw = np.asarray(jax.grad(lambda l: jnp.exp(sel.log_mu(s, l, kfit)))(lw))   # = w_j xi_j
        rec["det_x"] = np.asarray(x); rec["det_w"] = dw
        ln = NP.log_norm_pz(kfit)

        @jax.jit
        def ev_hist(P, ti, li):
            z = jnp.exp(ti)[None, :] / x[:, 0:1] - 1.0
            dc = N4.comoving_distance(jnp.clip(z, 0.0, 5.9))
            pts = jnp.concatenate([jnp.broadcast_to(x[:, None, :], z.shape + (3,)), z[..., None]], -1)
            ll = N4.log_likelihood(pts, P, dc); ll = jnp.where(jnp.isfinite(ll), ll, NP._MASKED)
            a = lw[:, None] + li[None, :] + ti[None, :] - jnp.log(x[:, 0:1]) + NP.log_pz(z, kfit, ln, dc) + ll
            wt = jnp.exp(a - logsumexp(a)).ravel()
            return jnp.histogram(z.ravel(), bins=jnp.asarray(ZB), weights=wt)[0]
        rec["stack_z"] = np.mean([np.asarray(ev_hist(jax.tree_util.tree_map(lambda a: a[i], like.P), like.t[i], like.lwt[i]))
                                  for i in range(like.n4)], 0)
        cat, _ = H.load_catalog(ns("o4", r["f_dep"]))
        rec["cat_centres"] = np.asarray(cat["centers"])
    out[tag] = rec
    print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k.endswith(("_lnL", "_mu", "_n", "kappa_fit"))}, flush=True)
    for p in rec["kappa_profile"]: print("   ", {k: round(v, 2) for k, v in p.items()}, flush=True)
flat = {f"{t}|{k}": (json.dumps(v) if isinstance(v, (list, dict)) else v) for t, rec in out.items() for k, v in rec.items()}
np.savez(OUT, **flat); print("saved", OUT, flush=True)
