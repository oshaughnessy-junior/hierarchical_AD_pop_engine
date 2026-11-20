"""Aggregate bias_measure.py outputs into summary.json + a 2-panel figure."""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/20261120-hier-reassess/runs/C_bias")
HIST_EDGES = np.geomspace(5, 200, 41)
rows = {}
for f in sorted(glob.glob(os.path.join(D, "*_k*.npz"))):
    z = np.load(f, allow_pickle=True); r = json.loads(str(z["result"]))
    r["hist"] = z["hist"]; r["logI"] = z["logI"]; r["names"] = [str(n) for n in z["names"]]
    rows.setdefault((r["theta_label"], r["mode"]), []).append(r)

def flat(r):
    m, d, l = r["metrics"], r["detected"], r["lnL"]
    o = {"f_m1_gt40": m["frac_m1_gt"]["40"], "f_m1_gt60": m["frac_m1_gt"]["60"], "f_m1_gt80": m["frac_m1_gt"]["80"],
         "gen_1g1g": m["gen_mix"]["g1g1"], "gen_1gNg": m["gen_mix"]["g1gN"], "gen_NgNg": m["gen_mix"]["gNgN"],
         "q_gt0p8_above40": m["q_above40"]["frac_q_gt_0p8"], "taint_frac": m["taint_frac"],
         "taint_mean": m["taint_mean"], "ess_merger": m["ess_merger"], "natal_w_cv": m["natal_w_cv"],
         "mu": d["mu"], "mu_A": d["mu_A"], "Ndet40": d["Ndet_gt"]["40"], "Ndet60": d["Ndet_gt"]["60"],
         "Ndet80": d["Ndet_gt"]["80"], "box_hi_hi": d["boxes"]["hi_hi"], "box_eq20_40": d["boxes"]["eq20_40"],
         "frac_q80_20_40": d["frac_q80_20_40"],
         "lnL_events": l["events"], "lnL_fixed_rate": l["fixed_rate"], "lnL_profiled": l["profiled"]}
    for lo, hi, wt, sd in m["chieff_width_vs_m1"]:
        o[f"chieff_sd_m1_{int(lo)}_{int(hi)}"] = sd
    return o

summ = {"note": "mean/sd over sampler keys; delta = exact - committed (key-paired)", "thetas": {}}
for (tl, mode), rs in sorted(rows.items()):
    keys = sorted(r["key"] for r in rs)
    F = [flat(r) for r in sorted(rs, key=lambda r: r["key"])]
    summ["thetas"].setdefault(tl, {"theta": rs[0]["theta"]})[mode] = {
        "keys": keys, **{k: [float(np.nanmean([f[k] for f in F])), float(np.nanstd([f[k] for f in F], ddof=1) if len(F) > 1 else np.nan)] for k in F[0]}}
for tl, d in summ["thetas"].items():
    if "committed" in d and "exact" in d:
        rc = {r["key"]: flat(r) for r in rows[(tl, "committed")]}
        rx = {r["key"]: flat(r) for r in rows[(tl, "exact")]}
        ks = sorted(set(rc) & set(rx))
        d["delta"] = {k: [float(np.nanmean([rx[s][k] - rc[s][k] for s in ks])),
                          float(np.nanstd([rx[s][k] - rc[s][k] for s in ks], ddof=1)) if len(ks) > 1 else np.nan]
                      for k in rc[ks[0]]}
        # per-event: biggest movers in ln I (key-averaged)
        names = rows[(tl, "committed")][0]["names"]
        Ic = np.mean([r["logI"] for r in rows[(tl, "committed")]], 0)
        Ix = np.mean([r["logI"] for r in rows[(tl, "exact")]], 0)
        sc = np.std([r["logI"] for r in rows[(tl, "committed")]], 0)
        o = np.argsort(-np.abs(Ix - Ic))[:12]
        d["event_movers"] = [[names[i], float(Ix[i] - Ic[i]), float(sc[i])] for i in o]
        for ev in ("S241011k", "GW241011_233834", "S241113p", "GW231123_135430", "S241225v", "S240413p"):
            if ev in names:
                i = names.index(ev); d.setdefault("priority_events", {})[ev] = [float(Ic[i]), float(Ix[i]), float(sc[i])]
json.dump(summ, open(os.path.join(D, "summary.json"), "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
cen = np.sqrt(HIST_EDGES[1:] * HIST_EDGES[:-1])
for mode, col in (("committed", "C3"), ("exact", "C0")):
    rs = rows.get(("mean", mode), [])
    if not rs: continue
    RA = np.exp(rs[0]["theta"]["log_R_A"])
    H = np.array([r["hist"] for r in rs]) * RA
    ax[0].fill_between(cen, H.min(0), H.max(0), color=col, alpha=0.3, step="mid")
    ax[0].step(cen, H.mean(0), where="mid", color=col, label=f"{mode} ({len(rs)} keys)")
    W = np.array([[c[3] for c in r["metrics"]["chieff_width_vs_m1"]] for r in rs])
    e = np.array([[c[0], c[1]] for c in rs[0]["metrics"]["chieff_width_vs_m1"]]); mc = np.sqrt(e[:, 0] * e[:, 1])
    ax[1].fill_between(mc, np.nanmin(W, 0), np.nanmax(W, 0), color=col, alpha=0.3)
    ax[1].plot(mc, np.nanmean(W, 0), "o-", color=col, label=mode)
ax[0].set(xscale="log", yscale="log", xlabel=r"$m_1\ [M_\odot]$", ylabel=r"channel A $dR/dm_1$ (arb. rate units)",
          title="headline mean, channel A")
ax[1].set(xscale="log", xlabel=r"$m_1\ [M_\odot]$", ylabel=r"std $\chi_{\rm eff}$ (channel A)", title="spin width vs mass")
for a in ax: a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(D, "bias_channelA.png"), dpi=130); fig.savefig(os.path.join(D, "bias_channelA.pdf"))
print("wrote", os.path.join(D, "summary.json"))
