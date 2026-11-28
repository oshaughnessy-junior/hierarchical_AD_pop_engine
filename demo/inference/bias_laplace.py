"""Pinned-proposal bias at Laplace displacements from the MAP (2026-11-27).

For each MAP JSON and one cascade key: points at MAP + t * Sigma e_s / sqrt(Sigma_ss)
(t = -2, -1, 0, 1, 2; s = spin_chi; Mahalanobis distance |t|) and n_rand random points
at Mahalanobis 1 (Sigma = (-H)^-1 from the fit). Points are clipped into the prior box;
the distance is recomputed after clipping. At each point: lnL with the proposal pinned
at the MAP final_proposal, and lnL with the proposal recentred at the point (exact there),
both at this key. Run once per key; combine with bias_laplace_summary.py.
"""
import argparse, json, os, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import hier_map_laplace as hml
import hier_nuts as hn


def points(r, n_rand, seed):
    names = r["names"]
    th = np.array([r["theta_map"][k] for k in names])
    negH = -np.array(r["hessian"])
    S = np.linalg.inv(negH)
    lo = np.array([r["bounds"][k][0] for k in names]); hi = np.array([r["bounds"][k][1] for k in names])
    w = hi - lo
    s = names.index("spin_chi")
    d = S[:, s] / np.sqrt(S[s, s])
    pts = [(f"spin{t:+d}", th + t * d) for t in (-2, -1, 0, 1, 2)]
    L = np.linalg.cholesky(S)
    rng = np.random.default_rng(seed)
    for i in range(n_rand):
        z = rng.standard_normal(len(names)); z /= np.linalg.norm(z)
        pts.append((f"rand{i}", th + L @ z))
    out = []
    for lab, p in pts:
        q = np.clip(p, lo + 1e-6 * w, hi - 1e-6 * w)
        dq = q - th
        out.append(dict(label=lab, theta=q, mahal=float(np.sqrt(dq @ negH @ dq)),
                        mahal_unclipped=float(np.sqrt((p - th) @ negH @ (p - th)))))
    return names, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map-jsons", nargs="+", required=True)
    p.add_argument("--key", type=int, required=True)
    p.add_argument("--n-seed", type=int, default=8000)
    p.add_argument("--n-b", type=int, default=6000)
    p.add_argument("--n-rand", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--outdir", required=True)
    p.add_argument("--catalog", default=os.path.join(hml.WS, "inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz"))
    p.add_argument("--pdet-grid", default=os.path.join(hml.WS, "inputs/mass_pdet_grid_kappa2.7.npz"))
    p.add_argument("--likelihood", default="nal3d")
    a = p.parse_args()
    a.map_json = None
    os.makedirs(a.outdir, exist_ok=True)
    for mj in a.map_jsons:
        r = hn.load_map(mj)
        tag = os.path.basename(os.path.dirname(os.path.abspath(mj)))
        names, pts = points(r, a.n_rand, a.seed)
        t0 = time.time()
        Bp = hn.setup(a, r, key=a.key)
        fp = jax.jit(Bp["parts"])
        rows = []
        for pt in pts:
            th = pt["theta"]
            Lp, (mup, _) = fp(jnp.asarray(th))
            prop = hml.prop_ref_from(r["model"], dict(zip(names, th)))
            Be = hn.setup(a, r, key=a.key, prop=prop)
            Le, (mue, _) = jax.jit(Be["parts"])(jnp.asarray(th))
            rows.append(dict(label=pt["label"], mahal=pt["mahal"], mahal_unclipped=pt["mahal_unclipped"],
                             theta=dict(zip(names, th.tolist())), proposal=prop,
                             lnL_pinned=float(Lp), lnL_exact=float(Le), mu_pinned=float(mup), mu_exact=float(mue)))
            print(f"{tag} key {a.key} {pt['label']:7s} d={pt['mahal']:.2f} spin={th[names.index('spin_chi')]:.4f} "
                  f"mmax={th[names.index('m_max_A')]:.3f} pinned {float(Lp):.3f} exact {float(Le):.3f} "
                  f"diff {float(Lp)-float(Le):+.3f} ({time.time()-t0:.0f}s)", flush=True)
        with open(os.path.join(a.outdir, f"{tag}_k{a.key}.json"), "w") as f:
            json.dump(dict(map_json=os.path.abspath(mj), tag=tag, key=a.key, n_seed=a.n_seed, n_b=a.n_b,
                           pinned_proposal=r["final_proposal"], rows=rows), f, indent=1)


if __name__ == "__main__":
    raise SystemExit(main())
