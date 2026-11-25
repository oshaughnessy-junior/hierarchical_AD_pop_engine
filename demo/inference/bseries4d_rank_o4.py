"""Ranking for the O4-only (168-event) refit with injection selection and free kappa.
Reads runs/BS4D_O4/scan/ and its plan_state.json; writes runs/BS4D_O4/{table,profile,saddles}.csv.
Rates are physical O4 rates under random thinning of the detected population for fit availability
(mu thinned 71/84 in O4a and 97/104 in O4b); they apply to this event subset and selection.
Usage: bseries4d_rank_o4.py [RUN_DIR]"""
import csv, json, os, sys
import numpy as np

RD = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/20261120-hier-reassess/runs/BS4D_O4")
SC = f"{RD}/scan"; ps = json.load(open(f"{SC}/plan_state.json"))
LABEL = {"NULL": "null", "BB": "B_b", "BC": "B_c", "BD": "B_d", "BMIX": "Bmix", "NULLG": "null + g(eta) exch 0", "REFRC": "combined (REFRC)"}
res = lambda t: json.load(open(f"{SC}/{t}/result.json"))
f2 = lambda v: f"{v:.2f}" if isinstance(v, float) else v

prof = []
for c, i in ps["cells"].items():
    if not i.get("best"): continue
    r = res(i["best"])
    prof.append(dict(member=i["member"], f_dep=i["f_dep"], tag=i["best"], variants="/".join(i["variants"]), kappa=r["theta_map"]["kappa"],
                     lnL_heldout=r["lnL_heldout_mean"], lnL_heldout_sd=r["lnL_heldout_sd"], lnZ=r["lnZ_laplace_bdy"],
                     pd=r["negH_pd"], eig0=min(r["negH_eigs"]), rails="/".join(r["railing"]),
                     neff_map=r.get("selection", {}).get("neff"), neff_min_fit=r.get("neff_trace", {}).get("min"),
                     n_eval_below_672=r.get("neff_trace", {}).get("n_below_4Nobs")))
prof.sort(key=lambda r: (r["member"], r["f_dep"]))
with open(f"{RD}/profile.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(prof[0])); w.writeheader(); [w.writerow({k: f2(v) for k, v in r.items()}) for r in prof]

rows = []
for M, s in ps["members"].items():
    t = s["best"]; r = res(t); e = json.load(open(f"{SC}/{t}/reeval_n8000.json")) if os.path.exists(f"{SC}/{t}/reeval_n8000.json") else None
    neff_re = min(ev["selection"]["neff"] for ev in e["evals"]) if e and all("selection" in ev for ev in e["evals"]) else None
    fl = []
    if not r["negH_pd"]: fl.append(f"non-PD (eig {min(r['negH_eigs']):.2g})")
    if r["railing"]: fl.append("rail " + "/".join(r["railing"]))
    if s["edge"]: fl.append(f"f_dep edge {s['best_fdep']}")
    nf = min(x for x in (r.get("selection", {}).get("neff"), r.get("neff_trace", {}).get("min"), neff_re) if x is not None)
    if nf < 4 * r["n_obs"]: fl.append(f"N_eff {nf:.0f} < {4 * r['n_obs']}")
    mp = [p for p in prof if p["member"] == M]; fz = max(mp, key=lambda p: p["lnZ"])
    rows.append(dict(member=M, label=LABEL[M], tag=t, f_dep=s["best_fdep"], fdeps="/".join(f"{x:.1f}" for x in s["fdeps"]),
                     kappa=r["theta_map"]["kappa"], kappa_sd=(r.get("laplace_sigma") or {}).get("kappa"),
                     lnL=e["lnL_mean"] if e else None, lnL_sd=e["lnL_sd"] if e else None,
                     lnL_heldout=r["lnL_heldout_mean"], lnZ=r["lnZ_laplace_bdy"], f_dep_lnZmax=fz["f_dep"],
                     neff_min=nf, mc_var_lnL=r.get("selection", {}).get("mc_var_lnL"), pd=r["negH_pd"], eig0=min(r["negH_eigs"]),
                     flags="; ".join(fl)))
ref = next(r for r in rows if r["member"] == "REFRC")
for r in rows:
    r["dlnZ"] = r["lnZ"] - ref["lnZ"]
    r["dlnL"] = r["lnL"] - ref["lnL"] if r["lnL"] is not None and ref["lnL"] is not None else None
rows.sort(key=lambda r: -r["lnZ"])
for i, r in enumerate(rows): r["rank"] = i + 1
cols = ["rank", "member", "label", "tag", "f_dep", "fdeps", "kappa", "kappa_sd", "lnL", "lnL_sd", "lnL_heldout", "lnZ", "dlnL", "dlnZ",
        "f_dep_lnZmax", "neff_min", "mc_var_lnL", "pd", "eig0", "flags"]
with open(f"{RD}/table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader(); [w.writerow({k: f2(v) for k, v in r.items()}) for r in rows]
sad = []
for c, i in ps["cells"].items():
    for e in i.get("escapes", []):
        sad.append(dict(cell=c, round=e["round"], parent=e["parent"], parent_eig0=e["parent_eig0"], predicted_gain=e["predicted_gain"],
                        dominant="/".join(e["dominant"]), outcome=e.get("outcome", "pending")))
if sad:
    with open(f"{RD}/saddles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(sad[0])); w.writeheader(); [w.writerow({k: f2(v) for k, v in r.items()}) for r in sad]
for r in rows:
    print(f"{r['rank']} {r['label']:22s} f_dep {r['f_dep']:.1f} kappa {r['kappa']:.2f} ({(r['kappa_sd'] or float('nan')):.2f}) "
          f"lnL {(r['lnL'] or float('nan')):.1f} ({(r['lnL_sd'] or float('nan')):.1f}) dlnZ {r['dlnZ']:+.1f} N_eff {r['neff_min']:.0f} | {r['flags'] or '-'}")
for p in prof: print("  ", p["member"], p["f_dep"], round(p["lnL_heldout"], 1), "pd" if p["pd"] else "nPD", p["rails"])
