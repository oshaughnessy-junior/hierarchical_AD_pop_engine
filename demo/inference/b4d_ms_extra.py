"""Extra checks: (1) are converged basins distinct (max |d theta|/width over shape params, same 0.20 rule);
(2) static-draw tuning: lnL at the committed draws minus the mean over field seeds 1-3 (key 50005, n_seed 24000);
(3) member ranking on the field-seed mean (seeds 1-3, key 50005) instead of the committed draws.
Usage: b4d_ms_extra.py ROOT OUTDIR"""
import csv, json, math, os, sys
import numpy as np
ROOT, OUT = sys.argv[1], sys.argv[2]
ps = json.load(open(f"{ROOT}/plan_state.json")); SH = ["kappa", "m_max_A", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "spin_chi", "alpha_B", "f_peakB", "mu_C", "sig_C"]
res = lambda t: json.load(open(f"{ROOT}/fits/{t}/result.json"))
rows, rank = [], {}
for k, m in ps["members"].items():
    M, DS = k.split("|"); b = json.load(open(f"{ROOT}/bounds_{M}.json")); best = m.get("best", [])
    pts = []
    for e in best:
        r = res(e["tag"]); z = np.load(f"{ROOT}/eval/{e['tag']}.npz", allow_pickle=True)
        dflt = float(z["24000|50005|None|lnL"]); fs = np.array([float(z[f"24000|50005|{s}|lnL"]) for s in (1, 2, 3)])
        u = np.array([(r["theta_map"][q] - b[q][0]) / (b[q][1] - b[q][0]) for q in SH if q in b])
        pts.append((e["basin"], e["tag"], u, dflt, fs))
        rows.append(dict(dataset=DS, member=M, basin=e["basin"], tag=e["tag"], lnL_default_k50005=dflt, field_mean_123=float(fs.mean()), field_sd_123=float(fs.std(ddof=1)), tuning_bias=dflt - float(fs.mean())))
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.max(np.abs(pts[i][2] - pts[j][2]))); rows.append(dict(dataset=DS, member=M, basin=f"{pts[i][0]}-{pts[j][0]}", tag="pair", converged_distance=d, same_basin=bool(d <= 0.20)))
    bb = max(pts, key=lambda p: p[4].mean()); rank.setdefault(DS, []).append((float(bb[4].mean()), float(bb[4].std(ddof=1)), M, bb[1], len(res(bb[1])["names"])))
cols = ["dataset", "member", "basin", "tag", "lnL_default_k50005", "field_mean_123", "field_sd_123", "tuning_bias", "converged_distance", "same_basin"]
with open(f"{OUT}/basin_checks.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols, restval=""); w.writeheader(); [w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}) for r in rows]
tb = np.array([r["tuning_bias"] for r in rows if "tuning_bias" in r]); print("tuning bias over %d points: mean %.2f, median %.2f, min %.2f, max %.2f, positive %d" % (len(tb), tb.mean(), np.median(tb), tb.min(), tb.max(), (tb > 0).sum()))
merged = [(r["dataset"], r["member"], r["basin"], round(r["converged_distance"], 3)) for r in rows if r.get("tag") == "pair" and r["same_basin"]]; print("converged pairs within 0.20:", merged)
out = []
for DS, L in rank.items():
    L.sort(reverse=True); print(DS, "ranking on field-seed mean (seeds 1-3, key 50005, n24000):")
    for i, (a, s, M, t, n) in enumerate(L):
        nxt = L[i + 1] if i + 1 < len(L) else None
        g = a - nxt[0] if nxt else None; sc = math.hypot(s, nxt[1]) if nxt else None
        print(f"   {M:6s} {a:9.2f} (sd {s:.2f}) d={n}" + (f"  gap {g:5.2f} vs field sd {sc:4.2f}" if nxt else ""))
        out.append(dict(dataset=DS, rank=i + 1, member=M, tag=t, field_mean_123=a, field_sd_123=s, n_params=n, gap_to_next=g if g is not None else "", gap_scatter=sc if sc is not None else ""))
with open(f"{OUT}/ranking_field_mean.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(out[0])); w.writeheader(); [w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}) for r in out]
