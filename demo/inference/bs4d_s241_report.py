"""S241011k sensitivity tables (cardassia). Inputs: samples CSV (bs4d_s241_samples.py), fitterm npz files, the campaign's eval npz.
Writes: term_by_point.csv (each point and config: 4D fit term, fit 3D projection, sample-based 3D term, NAL 3D term, ESS, bootstrap sd),
sampling_depth.csv (mult 1 vs 10, four static-draw seeds), term_at_winners.csv (S241011k term per representation at each winner), ranking_sensitivity.csv (pair gaps at the winners with the S241011k term replaced, ALL168, or added, NO241011).
Gaps are reported per evaluation with the sd over 6 evaluations. Event constants differ between representations but are the same for every population point, so gaps between
points are compared; absolute lnL changes carry an unknown constant.
Usage: bs4d_s241_report.py SAMPLES_CSV WORKDIR EVALDIR OUTDIR"""
import csv, glob, itertools, json, math, os, sys
import numpy as np
SC, WK, EV, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
smp = {(r["tag"], r["config"]): r for r in csv.DictReader(open(SC))}
f_ = lambda v: f"{v:.3f}" if isinstance(v, float) else v
rows, depth = [], []
term = {}
for f in sorted(glob.glob(f"{WK}/fitterm_*.npz")):
    z = np.load(f, allow_pickle=True); tag = str(z["tag"])
    for k in sorted({k.rsplit("|", 1)[0] for k in z.files if k.endswith("|lnI4")}):
        s = smp.get((tag, k), {})
        d = dict(tag=tag, config=k, kappa=float(z[f"{k}|kappa"]), lnI4_fit=float(z[f"{k}|lnI4"]), ess4=float(z[f"{k}|ess4"]), lnI3_fit=float(z[f"{k}|lnI3fit"]),
                 lnI3_samp=float(s["lnI_samp"]) if s.get("lnI_samp") not in (None, "", "-inf") else float("nan"), samp_boot_sd=float(s["boot_sd"]) if s.get("boot_sd") else float("nan"),
                 samp_ess_draws=float(s["ess_draws"]) if s.get("ess_draws") else float("nan"), top_samples_under_kernel=s.get("top_samples_under_kernel", ""),
                 top_dist_h=s.get("top_dist_to_nearest_sample_h", ""), lnI3_nal=float(z[f"{k}|lnI3nal"]), ess_nal=float(z[f"{k}|ess3nal"]), top_draws=str(z[f"{k}|top"]))
        d["samp_minus_fit3D"] = d["lnI3_samp"] - d["lnI3_fit"]; d["nal_minus_fit3D"] = d["lnI3_nal"] - d["lnI3_fit"]
        rows.append(d); term[(tag, k)] = d
cols = list(rows[0])
with open(f"{OUT}/term_by_point.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, cols); w.writeheader(); [w.writerow({k: f_(v) for k, v in r.items()}) for r in rows]
for tag in sorted({r["tag"] for r in rows}):
    for mult in ("1", "10"):
        L = [term[(tag, f"50005|{fs}|{mult}")] for fs in ("None", "1", "2", "3") if (tag, f"50005|{fs}|{mult}") in term]
        if not L: continue
        a = np.array([x["lnI4_fit"] for x in L]); e = np.array([x["ess4"] for x in L]); sm = np.array([x["lnI3_samp"] for x in L]); fit3 = np.array([x["lnI3_fit"] for x in L])
        depth.append(dict(tag=tag, mult=int(mult), n_seeds=len(L), lnI4_mean=float(a.mean()), lnI4_sd_field=float(a.std(ddof=1)), ess_min=float(e.min()), ess_max=float(e.max()),
                          samp_minus_fit3D_mean=float(np.nanmean(sm - fit3)), samp_minus_fit3D_sd=float(np.nanstd(sm - fit3, ddof=1))))
with open(f"{OUT}/sampling_depth.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, list(depth[0])); w.writeheader(); [w.writerow({k: f_(v) for k, v in r.items()}) for r in depth]
# ranking sensitivity per evaluation (3 default keys + 3 field seeds at key 50005, n_seed 24000).
# ALL168: S241011k's committed 4D term is replaced. NO241011: the event is not in the likelihood, so its term is added.
SETS = {"ALL168": {"B_b": "BB_ALL168_b1_f0.3_xp", "Bmix": "BMIX_ALL168_b1_f0.3", "REFRC": "REFRC_ALL168_b1_f0.3", "B_d": "BD_ALL168_b1_f0.3"},
        "NO241011": {"B_b": "BB_NO241011_b3_f0.3", "B_d": "BD_NO241011_b2_f0.3"}}
EVK = [("50005", "None"), ("60006", "None"), ("70007", "None"), ("50005", "1"), ("50005", "2"), ("50005", "3")]
L, delta = {}, []
for ds, WIN in SETS.items():
    for m, t in WIN.items():
        if not os.path.exists(f"{WK}/fitterm_{t}.npz"): print("missing", t); continue
        z = np.load(f"{EV}/{t}.npz", allow_pickle=True); nm = [str(x) for x in z["names"]]
        for key, fs in EVK:
            e = f"24000|{key}|{fs}"; lnL = float(z[f"{e}|lnL"]); tf = term[(t, f"{key}|{fs}|1")]
            if "S241011k" in nm:
                Iev = float(np.asarray(z[f"{e}|lnI"])[nm.index("S241011k")])
                assert abs(Iev - tf["lnI4_fit"]) < 1e-3, (t, e, Iev, tf["lnI4_fit"])        # same draws, same term
                rest = lnL - tf["lnI4_fit"]
            else:
                rest = lnL
            v = dict(committed=lnL, fit4D=rest + tf["lnI4_fit"], samp=rest + tf["lnI3_samp"], nal=rest + tf["lnI3_nal"], without=rest)
            t10 = term.get((t, f"{key}|{fs}|10"))
            if t10: v.update(fit4D_x10=rest + t10["lnI4_fit"], samp_x10=rest + t10["lnI3_samp"], nal_x10=rest + t10["lnI3_nal"])
            L[(ds, m, key, fs)] = v
        for var in ("fit4D", "samp", "nal", "fit4D_x10", "samp_x10", "nal_x10"):
            ks = [(k, f) for k, f in EVK if var in L[(ds, m, k, f)]]
            d = np.array([L[(ds, m, k, f)][var] - L[(ds, m, k, f)]["without"] for k, f in ks])
            c = np.array([L[(ds, m, k, f)][var] - L[(ds, m, k, f)]["committed"] for k, f in ks])
            delta.append(dict(dataset=ds, member=m, tag=t, variant=var, n_evals=len(d), term_mean=float(d.mean()), term_sd=float(d.std(ddof=1)),
                              lnL_change_mean=float(c.mean()), lnL_change_sd=float(c.std(ddof=1))))
with open(f"{OUT}/term_at_winners.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, list(delta[0])); w.writeheader(); [w.writerow({k: f_(v) for k, v in r.items()}) for r in delta]
out = []
for ds, WIN in SETS.items():
    for a, b in itertools.combinations(WIN, 2):
        for var in ("committed", "fit4D", "samp", "nal", "without", "fit4D_x10", "samp_x10", "nal_x10"):
            g = np.array([L[(ds, a, k, f)][var] - L[(ds, b, k, f)][var] for k, f in EVK if (ds, a, k, f) in L and (ds, b, k, f) in L and var in L[(ds, a, k, f)] and var in L[(ds, b, k, f)]])
            if len(g) < 2: continue
            out.append(dict(dataset=ds, pair=f"{a} - {b}", variant=var, n_evals=len(g), gap_mean=float(g.mean()), gap_sd=float(g.std(ddof=1))))
with open(f"{OUT}/ranking_sensitivity.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, list(out[0])); w.writeheader(); [w.writerow({k: f_(v) for k, v in r.items()}) for r in out]
for r in depth + delta + out: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
