"""Honest PPC for the FREE-m_max corrected model (two_channel_pl_demo).
Mass: detected COUNTS per Mc bin (model mu-scaled vs observed point-estimate
histogram -- apples-to-apples, no measurement-width inflation). Spin: sigma(chi_eff)
per bin, model intrinsic + measurement vs observed raw.
"""
import argparse, importlib.util, os, numpy as np, jax, jax.numpy as jnp
from jax.scipy.special import logsumexp

_HERE = os.path.dirname(__file__)
PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
M = pl.M
from mass_pdet_interp import make_real_pdet_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--post", default="/home/richard.oshaughnessy/coag_container/figures_two_channel_plmmax/two_channel_pl_post.npz")
    p.add_argument("--catalog", default=os.path.join(_HERE, "../../../planning_o4_mixed/full_o3_o4a_o4b_catalog.npz"))
    p.add_argument("--f-dep", type=float, default=0.10)
    p.add_argument("--n-seed", type=int, default=8000)
    p.add_argument("--n-b", type=int, default=6000)
    p.add_argument("--n-draw", type=int, default=25)
    p.add_argument("--out", default="/home/richard.oshaughnessy/coag_container/figures_two_channel_plmmax/ppc_plmmax.png")
    args = p.parse_args()
    M.F_DEP = args.f_dep
    comb, nA = pl.make_combined_pl(args.n_seed, args.n_b)
    pdet = make_real_pdet_fn(); T = float(M.T_OBS)
    post = np.load(args.post)
    keys = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
    idx = np.linspace(0, len(post["m_max_A"]) - 1, min(args.n_draw, len(post["m_max_A"]))).astype(int)
    edges = np.linspace(4, 60, 29); cen = 0.5 * (edges[:-1] + edges[1:]); bw = edges[1] - edges[0]
    mc_b = np.array([5, 12, 18, 25, 32, 40, 55]); mc_c = 0.5 * (mc_b[:-1] + mc_b[1:])
    cAll, cA, cB, dvar = [], [], [], []
    for i in idx:
        th = {k: float(post[k][i]) for k in keys}
        s, lw = comb(**th); s = np.asarray(s); w = np.exp(np.asarray(lw)); pd = np.asarray(pdet(s))
        isA = np.zeros(len(s), bool); isA[:nA] = True
        mc = (s[:, 0] * s[:, 1]) ** 0.6 / (s[:, 0] + s[:, 1]) ** 0.2
        chie = (s[:, 0] * s[:, 2] + s[:, 1] * s[:, 3]) / (s[:, 0] + s[:, 1])
        dw = w * pd
        cAll.append(T * np.histogram(mc, edges, weights=dw)[0])
        cA.append(T * np.histogram(mc[isA], edges, weights=dw[isA])[0])
        cB.append(T * np.histogram(mc[~isA], edges, weights=dw[~isA])[0])
        d = []
        for lo, hi in zip(mc_b[:-1], mc_b[1:]):
            mm = (mc >= lo) & (mc < hi); ww = dw[mm]
            if ww.sum() > 1e-6:
                wn = ww / ww.sum(); mu = np.sum(wn * chie[mm]); d.append(np.sqrt(np.sum(wn * (chie[mm] - mu) ** 2)))
            else:
                d.append(np.nan)
        dvar.append(d)
    cAll, cA, cB, dvar = map(np.array, (cAll, cA, cB, dvar))
    cat = np.load(args.catalog, allow_pickle=True)
    mco, ceo, sco = cat["centers"][:, 0], cat["centers"][:, 2], cat["sigs"][:, 2]
    oc, _ = np.histogram(mco, edges)
    od, mv = [], []
    for lo, hi in zip(mc_b[:-1], mc_b[1:]):
        m = (mco >= lo) & (mco < hi)
        od.append(ceo[m].std() if m.sum() > 3 else np.nan); mv.append(np.mean(sco[m] ** 2) if m.sum() > 3 else np.nan)
    od, mv = np.array(od), np.array(mv)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    a0 = ax[0]
    a0.bar(cen, oc, width=bw * 0.9, color="k", alpha=0.22, label=f"observed (N={oc.sum()})")
    lo, md, hi = np.percentile(cAll, [5, 50, 95], 0)
    a0.fill_between(cen, lo, hi, step="mid", color="C0", alpha=0.25)
    a0.step(cen, md, where="mid", color="C0", lw=2.5, label=f"model total (mu={md.sum():.0f})")
    a0.step(cen, np.median(cA, 0), where="mid", color="C3", lw=1.5, ls="--", label="natal+cascade A")
    a0.step(cen, np.median(cB, 0), where="mid", color="C2", lw=1.8, ls=":", label="aligned low-mass B")
    a0.axvline(np.median([(post["m_max_A"][i]) for i in idx]) * 0.87, color="grey", ls=":", lw=1)
    a0.set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$"); a0.set_ylabel("detected events per bin")
    a0.set_title("Mass: detected COUNTS (apples-to-apples)"); a0.legend(fontsize=8)
    a1 = ax[1]
    disp = np.sqrt(dvar ** 2 + mv[None, :]); lo, md, hi = np.nanpercentile(disp, [5, 50, 95], 0)
    a1.fill_between(mc_c, lo, hi, alpha=0.3, color="C1", label="model+meas 90%")
    a1.plot(mc_c, md, "C1", lw=2, marker="o", ms=4)
    a1.plot(mc_c, od, "k", lw=2, marker="s", ms=5, label="observed (raw)")
    a1.set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$"); a1.set_ylabel(r"$\sigma(\chi_{\rm eff})$")
    a1.set_title("Spin: effective-spin dispersion"); a1.legend(fontsize=8); a1.set_ylim(0, None)
    fig.suptitle(f"Free-$m_{{\\max}}$ corrected model PPC  (m_max={post['m_max_A'].mean():.1f}, f_dep={args.f_dep}, 238 ev)", fontsize=11)
    fig.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight"); print("wrote", args.out)
    # numeric summary
    print("\nMc bin    model   obs")
    for lo, hi in zip(mc_b[:-1], mc_b[1:]):
        mm = (cen >= lo) & (cen < hi)
        print(f"  [{lo:2.0f},{hi:2.0f})  {np.median(cAll,0)[mm].sum():6.1f} {oc[mm].sum():5d}")
    print(f"  Mc>30:   model {np.median(cAll,0)[cen>30].sum():.1f}  obs {(mco>30).sum()}")


if __name__ == "__main__":
    raise SystemExit(main())
