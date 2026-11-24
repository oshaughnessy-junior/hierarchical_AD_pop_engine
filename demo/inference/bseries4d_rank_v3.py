"""4D B-series ranking, round 3 (v3). Reads runs/BS4D_scan/ and plan_state_v3.json; writes
runs/BS4D_rank/{table_v3_k27,table_v3_kfree,profile_v3,laplace_check_v3}.csv and README_v3.md.
v1 and v2 files are left in place.

kappa 2.7: rate priors log_R_* ~ U[logr0-8, logr0+8] as before. Free kappa: kappa ~ U[-1, 7] and every
log_R_* ~ U[logr0-8, logr0+H] (H from plan_state_v3.json), so free-kappa lnZ compares only within the
free-kappa table. f_dep per member by held-out lnL at kappa 2.7; lnL is the n_seed 8000 mean over keys
11/202/3003; lnZ is the boundary-corrected Laplace estimate; dlnZ is relative to REFRC.
"""
import csv, json, math, os

B = os.path.expanduser("~/20261120-hier-reassess/runs")
SC, OUT = f"{B}/BS4D_scan", f"{B}/BS4D_rank"
LABEL = {"NULL": "null", "BB": "B_b", "BC": "B_c", "BD": "B_d", "BMIX": "Bmix",
         "NULLG": "null + g(eta) exch 0", "REFRC": "combined (REFRC)"}
ps = json.load(open(f"{SC}/plan_state_v3.json"))
cells, members, H = ps["cells"], ps["members"], ps["kfree_level"]
u3 = {r["member"]: r for r in csv.DictReader(open(f"{B}/BS_rank/table_union.csv"))}
res = lambda t: json.load(open(f"{SC}/{t}/result.json"))


def ree(t):
    p = f"{SC}/{t}/reeval_n8000.json"
    if not os.path.exists(p): return None, None
    e = json.load(open(p)); return e["lnL_mean"], e["lnL_sd"]


def fmt(x, p=1): return "-" if x is None else f"{x:.{p}f}"


prof = []
for c, i in cells.items():
    if i["mode"] != "k27" or not i.get("best"): continue
    r = res(i["best"])
    prof.append(dict(member=i["member"], f_dep=i["f_dep"], tag=i["best"], variants="/".join(i["variants"]),
                     lnL_heldout=r["lnL_heldout_mean"], lnL_heldout_sd=r["lnL_heldout_sd"], lnL_map=r["lnL"],
                     lnZ=r["lnZ_laplace_bdy"], pd=r["negH_pd"], eig0=min(r["negH_eigs"]), rails="/".join(r["railing"])))
prof.sort(key=lambda r: (r["member"], r["f_dep"]))
with open(f"{OUT}/profile_v3.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(prof[0])); w.writeheader()
    for r in prof: w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})


def table(kind):
    rows = []
    for M, s in members.items():
        t = s.get(kind)
        if not t: continue
        r = res(t); L, Lsd = ree(t); mp = [p for p in prof if p["member"] == M]
        fz = max(mp, key=lambda p: p["lnZ"])
        nrate = sum(1 for n in r["names"] if n.startswith("log_R_"))
        fl = []
        if not r["negH_pd"]: fl.append(f"non-PD (eig {min(r['negH_eigs']):.0f})")
        if r["railing"]: fl.append("rail " + "/".join(r["railing"]))
        if s["best_fdep"] in (min(s["fdeps"]), max(s["fdeps"])): fl.append(f"f_dep edge {s['best_fdep']}")
        rows.append(dict(member=M, label=LABEL[M], tag=t, f_dep=s["best_fdep"], fdeps="/".join(f"{x:.1f}" for x in s["fdeps"]),
                         kappa=r["theta_map"].get("kappa", 2.7), kappa_sd=(r.get("laplace_sigma") or {}).get("kappa"),
                         lnL=L, lnL_sd=Lsd, lnL_heldout=r["lnL_heldout_mean"], lnL_heldout_sd=r["lnL_heldout_sd"],
                         lnZ=r["lnZ_laplace_bdy"], f_dep_lnZmax=fz["f_dep"],
                         dlnV_rates=(nrate * math.log((8 + H) / 16.0) if kind == "best_kfree" else 0.0),
                         pd=r["negH_pd"], eig0=min(r["negH_eigs"]), flags="; ".join(fl),
                         dlnZ_3D_union=float(u3[M]["dlnZ"]) if M in u3 else None))
    ref = next(r for r in rows if r["member"] == "REFRC")
    for r in rows:
        r["dlnZ"] = r["lnZ"] - ref["lnZ"]
        r["dlnL"] = r["lnL"] - ref["lnL"] if r["lnL"] is not None and ref["lnL"] is not None else None
    rows.sort(key=lambda r: -r["lnZ"])
    for i, r in enumerate(rows): r["rank"] = i + 1
    name = "k27" if kind == "best_k27" else "kfree"
    cols = ["rank", "member", "label", "tag", "f_dep", "fdeps", "kappa", "kappa_sd", "lnL", "lnL_sd", "lnL_heldout",
            "lnL_heldout_sd", "lnZ", "dlnL", "dlnZ", "dlnZ_3D_union", "f_dep_lnZmax", "dlnV_rates", "pd", "eig0", "flags"]
    with open(f"{OUT}/table_v3_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
    return rows


t27, tkf = table("best_k27"), table("best_kfree")

lap = []
for t in ps["laplace_check"]:
    p = f"{SC}/{t}/laplace_n8000.json"
    if not os.path.exists(p): continue
    e = json.load(open(p))
    lap.append(dict(tag=t, lnZ_n3000=e["ref_lnZ_laplace_bdy"], lnZ_n8000=e["lnZ_laplace_bdy"],
                    dlnZ=e["lnZ_laplace_bdy"] - e["ref_lnZ_laplace_bdy"], dlnL=e["lnL"] - e["ref_lnL"],
                    pd_n8000=e["negH_pd"], min_eig_n3000=min(e["ref_negH_eigs"]), min_eig_n8000=min(e["negH_eigs"]),
                    min_eig_dominant=e["min_eig_dominant"],
                    spin_chi_weight=abs(e["min_eig_vector"].get("spin_chi", 0.0))))
if lap:
    with open(f"{OUT}/laplace_check_v3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(lap[0])); w.writeheader()
        for r in lap: w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})

L = ["# B-series on the revised 4D likelihoods, round 3 (v3)", "",
     f"kappa 2.7 with rate priors U[logr0-8, logr0+8]; free kappa ~ U[-1, 7] with rate priors U[logr0-8, logr0+{H:g}].",
     "f_dep per member by held-out lnL at kappa 2.7. lnL: n_seed 8000 mean (keys 11/202/3003). lnZ: Laplace,",
     "boundary-corrected. dlnZ relative to REFRC; free-kappa lnZ compares only within its table.", ""]
for title, rows, kf in (("kappa 2.7", t27, False), (f"free kappa (rates to logr0+{H:g})", tkf, True)):
    L += [f"## {title}", "", "| rank | member | f_dep | " + ("kappa | " if kf else "") +
          "lnL (sd) | dlnZ | dlnZ 3D union | f_dep max lnZ | flags |", "|---" * (8 + kf) + "|"]
    for r in rows:
        L.append(f"| {r['rank']} | {r['label']} | {r['f_dep']:.1f} | " +
                 (f"{fmt(r['kappa'], 2)} ({fmt(r['kappa_sd'], 2)}) | " if kf else "") +
                 f"{fmt(r['lnL'])} ({fmt(r['lnL_sd'])}) | {r['dlnZ']:+.1f} | {fmt(r['dlnZ_3D_union'])} | "
                 f"{r['f_dep_lnZmax']:.1f} | {r['flags'] or '-'} |")
    L.append("")
L += [f"Wider rate prior: each log_R_* width 16 -> {8 + H:g}, ln prior volume +{math.log((8 + H) / 16):.3f} per rate "
      "parameter (2 for null/Bmix/NULLG/REFRC, 3 for B_b/B_c/B_d).", ""]
fds = sorted({p["f_dep"] for p in prof})
L += ["## f_dep profiles (kappa 2.7, held-out lnL; * = best)", "", "| member | " + " | ".join(f"{x:.1f}" for x in fds) + " |",
      "|---" * (len(fds) + 1) + "|"]
for M in members:
    mp = {p["f_dep"]: p for p in prof if p["member"] == M}; bf = members[M]["best_fdep"]
    L.append(f"| {LABEL[M]} | " + " | ".join((f"{mp[x]['lnL_heldout']:.1f}{'*' if x == bf else ''}{'' if mp[x]['pd'] else ' (nPD)'}"
                                              if x in mp else "-") for x in fds) + " |")
if lap:
    L += ["", "## Laplace at n_seed 8000 (same MAP, key 11)", "", "| point | lnZ n3000 | lnZ n8000 | dlnZ | min eig along |",
          "|---|---|---|---|---|"]
    L += [f"| {r['tag']} | {r['lnZ_n3000']:.2f} | {r['lnZ_n8000']:.2f} | {r['dlnZ']:+.2f} | {r['min_eig_dominant']} |" for r in lap]
open(f"{OUT}/README_v3.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
