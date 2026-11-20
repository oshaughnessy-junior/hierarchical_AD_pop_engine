"""PPC figure: best hierarchical (H*) fit vs REF at their MAPs.

Left: intrinsic dR/dm1 (per unit T_obs, arbitrary overall units shared by both
models) with the field component dashed. Middle: detected m1 distribution
(expected counts per bin) vs the catalog (m1 from the NAL (Mc, eta) centres).
Right: chi_eff vs m1, model detected-weighted median and 5-95% band per m1 bin,
data points with 1-sigma chi_eff bars.

  python hier_ppc.py --scan runs/B_scan [--ref TAG] [--hier TAG] [--n-seed 3000]
Default tags: the best-lnZ_bdy REF and the best-lnZ_bdy H* row.
"""
import argparse, glob, json, os, sys
import numpy as np
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hier_map_laplace as H


def best(scan, pred):
    rs = []
    for f in glob.glob(os.path.join(scan, "*/result.json")):
        r = json.load(open(f)); r["tag"] = os.path.basename(os.path.dirname(f))
        if pred(r):
            rs.append(r)
    return max(rs, key=lambda r: r["lnZ_heldout"]) if rs else None


def forward(r, n_seed, n_b, catalog, grid):
    ns = argparse.Namespace(f_dep=r["f_dep"], catalog=catalog, pdet_grid=grid)
    B = H.build(ns, r["model"], r["key"], n_seed, n_b, r.get("final_proposal"))
    th = {k: r["theta_map"][k] for k in B["names"]}; th.update(B["consts"])
    s, lw = B["comb"](**th)
    s = np.asarray(s); w = np.exp(np.asarray(lw)); pd = np.asarray(B["pdet"](jnp.asarray(s)))
    isA = np.zeros(len(s), bool); isA[:B["nA"]] = True
    m1 = np.maximum(s[:, 0], s[:, 1]); m2 = np.minimum(s[:, 0], s[:, 1])
    chie = (s[:, 0] * s[:, 2] + s[:, 1] * s[:, 3]) / (s[:, 0] + s[:, 1])
    return dict(m1=m1, chie=chie, w=w, wd=w * pd, isA=isA, tag=r["tag"], r=r)


def wq(x, w, qs):
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]
    return [x[o][min(np.searchsorted(c, q), len(x) - 1)] for q in qs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", required=True)
    p.add_argument("--ref", default=None)
    p.add_argument("--hier", default=None)
    p.add_argument("--n-seed", type=int, default=3000)
    p.add_argument("--n-b", type=int, default=3000)
    p.add_argument("--catalog", default=os.path.join(H.WS, "inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz"))
    p.add_argument("--pdet-grid", default=os.path.join(H.WS, "inputs/mass_pdet_grid_kappa2.7.npz"))
    a = p.parse_args()
    ld = lambda t: dict(json.load(open(os.path.join(a.scan, t, "result.json"))), tag=t)
    rref = ld(a.ref) if a.ref else best(a.scan, lambda r: r["model"] == "REF")
    rh = ld(a.hier) if a.hier else best(a.scan, lambda r: r["model"] != "REF")
    if rref is None or rh is None:
        raise SystemExit("need at least one finished REF and one H* fit")
    cat = np.load(a.catalog, allow_pickle=True)
    ce, cov = np.asarray(cat["centers"]), np.asarray(cat["covs"])
    mc, eta, ch = ce[:, 0], np.clip(ce[:, 1], 1e-3, 0.25), ce[:, 2]
    M_tot = mc * eta ** -0.6
    m1d = 0.5 * M_tot * (1 + np.sqrt(1 - 4 * eta))
    chs = np.sqrt(cov[:, 2, 2])
    models = [(forward(rref, a.n_seed, a.n_b, a.catalog, a.pdet_grid), "C0"),
              (forward(rh, a.n_seed, a.n_b, a.catalog, a.pdet_grid), "C3")]
    bins = np.geomspace(5, 150, 41); bc = np.sqrt(bins[1:] * bins[:-1]); dm = np.diff(bins)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for mdl, col in models:
        lab = f"{mdl['tag']} (held-out lnL {mdl['r']['lnL_heldout_mean']:.1f})"
        h, _ = np.histogram(mdl["m1"], bins, weights=mdl["w"])
        ax[0].step(bc, h / dm, where="mid", color=col, label=lab)
        hB, _ = np.histogram(mdl["m1"][~mdl["isA"]], bins, weights=mdl["w"][~mdl["isA"]])
        if hB.sum() > 0:
            ax[0].step(bc, hB / dm, where="mid", color=col, ls="--", lw=0.9, label=f"{mdl['tag']} field")
        hd, _ = np.histogram(mdl["m1"], bins, weights=mdl["wd"])
        ax[1].step(bc, hd, where="mid", color=col, label=lab)
    hdat, _ = np.histogram(m1d, bins)
    ax[1].errorbar(bc, hdat, yerr=np.sqrt(hdat), fmt="k.", ms=4, label="catalog (NAL centres)")
    ax[0].set(xscale="log", yscale="log", xlabel="m1 [Msun]", ylabel="dR/dm1 (model units)")
    ax[1].set(xscale="log", xlabel="m1 [Msun]", ylabel="detected events per bin")
    cb = np.geomspace(5, 120, 16)
    for mdl, col in models:
        med, lo, hi, xc = [], [], [], []
        for l, u in zip(cb[:-1], cb[1:]):
            sel = (mdl["m1"] >= l) & (mdl["m1"] < u) & (mdl["wd"] > 0)
            if mdl["wd"][sel].sum() <= 0 or sel.sum() < 5:
                continue
            q = wq(mdl["chie"][sel], mdl["wd"][sel], (0.05, 0.5, 0.95))
            xc.append(np.sqrt(l * u)); lo.append(q[0]); med.append(q[1]); hi.append(q[2])
        ax[2].plot(xc, med, color=col); ax[2].fill_between(xc, lo, hi, color=col, alpha=0.2)
    ax[2].errorbar(m1d, ch, yerr=chs, fmt="k.", ms=3, lw=0.5, alpha=0.6)
    ax[2].set(xscale="log", xlabel="m1 [Msun]", ylabel="chi_eff (detected, 5-95%)", ylim=(-0.6, 0.8))
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.suptitle(f"PPC at MAP: REF {rref['tag']} vs {rh['tag']} (n_seed {a.n_seed}, key {rref['key']})")
    fig.tight_layout()
    out = os.path.join(a.scan, f"ppc_{rh['tag']}_vs_{rref['tag']}.png")
    fig.savefig(out, dpi=130); fig.savefig(out.replace(".png", ".pdf"))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
