"""Tables for the O4-only multi-start campaign. Usage: b4d_ms_report.py ROOT OUTDIR   (ROOT holds fits/ eval/ screen/ plan_state.json bounds_*.json)
Primary statistic: lnL at n_seed 24000 / n_b 18000, mean over held-out cascade keys 50005/60006/70007 with the committed static
draws. Cascade-key sd: over those 3 keys. Field-draw sd: over 4 static-draw sets (committed plus field seeds 1/2/3) at key 50005.
n_seed change: lnL(24000) - lnL(8000) at the 3 default keys. Scatter of a point: sqrt(sd_key^2 + sd_field^2 + dn^2) where dn is
half the n_seed change (the two n_seed values bracket it). Scatter of a gap: quadrature of the two points'. lnL is never differenced across data sets.
"""
import csv, json, math, os, sys
import numpy as np

ROOT, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
LAB = {"NULL": "null", "BB": "B_b", "BC": "B_c", "BD": "B_d", "BMIX": "Bmix", "NULLG": "null + g(eta)", "REFRC": "combined (REFRC)"}
ps = json.load(open(f"{ROOT}/plan_state.json")); KEYS = (50005, 60006, 70007); FS = (1, 2, 3)
res = lambda t: json.load(open(f"{ROOT}/fits/{t}/result.json"))
nparam = lambda t: len(res(t)["names"])
# the per-eval numbers of one tag
cache = {}
def ev(tag):
    if tag in cache: return cache[tag]
    p = f"{ROOT}/eval/{tag}.npz"
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=True); nm = [str(x) for x in z["names"]]; j = nm.index("S241011k") if "S241011k" in nm else None
    o = {"names": nm, "j": j}
    for ns in (8000, 24000):
        for k, fs in [(k, None) for k in KEYS] + [(50005, f) for f in FS]:
            t = f"{ns}|{k}|{fs}"; o[t] = dict(lnL=float(z[f"{t}|lnL"]), mu=float(z[f"{t}|mu"]), neff=float(z[f"{t}|neff"]), lnI=np.asarray(z[f"{t}|lnI"]), ess=np.asarray(z[f"{t}|ess"]))
    cache[tag] = o; return o
def stat(tag):
    e = ev(tag)
    if e is None: return None
    d = {}
    for ns in (8000, 24000):
        k3 = np.array([e[f"{ns}|{k}|None"]["lnL"] for k in KEYS]); fs4 = np.array([e[f"{ns}|50005|None"]["lnL"]] + [e[f"{ns}|50005|{f}"]["lnL"] for f in FS])
        d[ns] = dict(mean=float(k3.mean()), sd_key=float(k3.std(ddof=1)), mean_field=float(fs4.mean()), sd_field=float(fs4.std(ddof=1)), field_default=float(fs4[0]),
                     neff=float(min(e[f"{ns}|{k}|None"]["neff"] for k in KEYS)))
        if e["j"] is not None:
            Ik = np.array([e[f"{ns}|{k}|None"]["lnI"][e["j"]] for k in KEYS]); If = np.array([e[f"{ns}|50005|None"]["lnI"][e["j"]]] + [e[f"{ns}|50005|{f}"]["lnI"][e["j"]] for f in FS])
            Ek = [float(e[f"{ns}|{k}|None"]["ess"][e["j"]]) for k in KEYS] + [float(e[f"{ns}|50005|{f}"]["ess"][e["j"]]) for f in FS]
            d[ns].update(I_mean=float(If.mean()), I_sd_key=float(Ik.std(ddof=1)), I_sd_field=float(If.std(ddof=1)), I_min_ess=float(min(Ek)), I_max_ess=float(max(Ek)))
    dn = d[24000]["mean"] - d[8000]["mean"]
    sc = math.sqrt(d[24000]["sd_key"] ** 2 + d[24000]["sd_field"] ** 2 + (dn / 2) ** 2)
    return dict(d=d, dn=dn, scatter=sc, npar=nparam(tag))

rows, win = [], {}
for k, m in ps["members"].items():
    M, DS = k.split("|"); fits = m.get("fits", {})
    for i, o in enumerate(m["basins"], 1):
        tags = [(f, t) for kk, t in fits.items() for (j, f) in [(int(kk.split("|")[0]), float(kk.split("|")[1]))] if j == i and t]
        bt = next((b["tag"] for b in m.get("best", []) if b["basin"] == i), None)
        s = stat(bt) if bt else None
        r = res(bt) if bt else None
        row = dict(dataset=DS, member=M, label=LAB[M], basin=i, first_start=o["first"], screen_score=o["score"], n_starts_reached=len(o["starts"]), reached_by=";".join(o["starts"]),
                   n_unconverged=len(o.get("unconverged", [])), n_screen_starts=m["n_starts"], tag=bt or "", f_dep=(r["f_dep"] if r else ""), n_params=(nparam(bt) if bt else ""),
                   lnL_n3000_heldout=(r["lnL_heldout_mean"] if r else ""), lnL_n3000_sd=(r["lnL_heldout_sd"] if r else ""), negH_pd=(r["negH_pd"] if r else ""), rails=("/".join(r["railing"]) if r else ""))
        if r:
            for q in ("kappa", "m_max_A", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "spin_chi", "alpha_B", "f_peakB", "mu_C", "sig_C"):
                row[q] = r["theta_map"].get(q, "")
        if s:
            e24, e8 = s["d"][24000], s["d"][8000]
            row.update(lnL_24000=e24["mean"], sd_key=e24["sd_key"], field_mean_24000=e24["mean_field"], sd_field=e24["sd_field"], lnL_8000=e8["mean"], sd_key_8000=e8["sd_key"], n_seed_change=s["dn"], scatter=s["scatter"],
                       neff_min=e24["neff"])
            if "I_mean" in e24: row.update(S241_lnI_24000=e24["I_mean"], S241_sd_key=e24["I_sd_key"], S241_sd_field=e24["I_sd_field"], S241_ess_range=f"{e24['I_min_ess']:.1f}-{e24['I_max_ess']:.1f}", S241_lnI_8000=e8["I_mean"])
        rows.append(row)
        if s: win.setdefault((DS, M), []).append((s["d"][24000]["mean"], s["scatter"], bt, i, s))
cols = sorted({k for r in rows for k in r}, key=lambda c: (["dataset", "member", "label", "basin", "first_start", "screen_score", "n_starts_reached", "reached_by", "n_unconverged", "n_screen_starts", "tag", "f_dep", "n_params", "lnL_24000", "sd_key", "sd_field", "n_seed_change", "scatter", "lnL_8000", "lnL_n3000_heldout", "negH_pd", "rails"] + [c]).index(c) if c in ["dataset", "member", "label", "basin", "first_start", "screen_score", "n_starts_reached", "reached_by", "n_unconverged", "n_screen_starts", "tag", "f_dep", "n_params", "lnL_24000", "sd_key", "sd_field", "n_seed_change", "scatter", "lnL_8000", "lnL_n3000_heldout", "negH_pd", "rails"] else 99)
f_ = lambda v: f"{v:.3f}" if isinstance(v, float) else v
with open(f"{OUT}/basins.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols, restval=""); w.writeheader(); [w.writerow({k: f_(v) for k, v in r.items()}) for r in rows]
# winners, gap between the two best basins, ranking per data set
ranks = {}
wrows = []
for DS in ("ALL168", "NO241011"):
    tops = []
    for (d, M), L in win.items():
        if d != DS: continue
        L.sort(key=lambda x: -x[0]); (a, sa, tag, bi, s) = L[0]; second = L[1] if len(L) > 1 else None
        wrows.append(dict(dataset=DS, member=M, label=LAB[M], winner_basin=bi, winner_tag=tag, lnL_24000=a, scatter=sa, n_params=s["npar"], n_basins_evaluated=len(L),
                          gap_to_second=(a - second[0] if second else ""), gap_scatter=(math.hypot(sa, second[1]) if second else ""),
                          gap_exceeds_scatter=(bool(a - second[0] > math.hypot(sa, second[1])) if second else ""), kappa=res(tag)["theta_map"]["kappa"], f_dep=res(tag)["f_dep"]))
        tops.append((a, sa, M, tag, s["npar"]))
    tops.sort(reverse=True)
    for i in range(len(tops) - 1):
        a, sa, M, t, n = tops[i]; b, sb, M2, t2, n2 = tops[i + 1]
        ranks[(DS, i)] = dict(dataset=DS, rank_hi=LAB[M], lnL_hi=a, rank_lo=LAB[M2], lnL_lo=b, gap=a - b, combined_scatter=math.hypot(sa, sb), exceeds=bool(a - b > math.hypot(sa, sb)), d_params=n - n2)
with open(f"{OUT}/winners.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(wrows[0])); w.writeheader(); [w.writerow({k: f_(v) for k, v in r.items()}) for r in wrows]
with open(f"{OUT}/member_ranking.csv", "w", newline="") as f:
    rr = list(ranks.values()); w = csv.DictWriter(f, list(rr[0])); w.writeheader(); [w.writerow({k: f_(v) for k, v in r.items()}) for r in rr]
print(len(rows), "basin rows;", sum(1 for r in rows if r.get("lnL_24000") not in ("", None)), "evaluated")
for r in wrows: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
for r in ranks.values(): print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
