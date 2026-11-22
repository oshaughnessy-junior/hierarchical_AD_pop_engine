"""4D B-series ranking, closing pass (v2). Reads runs/BS4D_scan/<tag>/ and plan_state.json from
b4d_v2_planner.py; writes runs/BS4D_rank/{table_v2_k27,table_v2_kfree,profile_v2,saddles_v2}.csv
and README_v2.md. The v1 files are left in place.

Per member: f_dep chosen by held-out lnL (n_seed 3000, keys 202/3003/40004) at kappa 2.7; the free-kappa
fit sits at that f_dep. lnL is the n_seed 8000 mean over keys 11/202/3003; lnZ is the boundary-corrected
Laplace estimate. dlnZ is relative to REFRC; the 3D union column comes from runs/BS_rank/table_union.csv.
"""
import csv, json, os
import numpy as np

B = os.path.expanduser("~/20261120-hier-reassess/runs")
SC, OUT = f"{B}/BS4D_scan", f"{B}/BS4D_rank"
os.makedirs(OUT, exist_ok=True)
LABEL = {"NULL": "null", "BB": "B_b", "BC": "B_c", "BD": "B_d", "BMIX": "Bmix",
         "NULLG": "null + g(eta) exch 0", "REFRC": "combined (REFRC)"}
ps = json.load(open(f"{SC}/plan_state.json"))
cells, members = ps["cells"], ps["members"]
u3 = {r["member"]: r for r in csv.DictReader(open(f"{B}/BS_rank/table_union.csv"))}


def res(tag): return json.load(open(f"{SC}/{tag}/result.json"))


def ree(tag):
    p = f"{SC}/{tag}/reeval_n8000.json"
    if not os.path.exists(p): return None, None
    e = json.load(open(p)); return e["lnL_mean"], e["lnL_sd"]


def flags(r, fdeps=None, fd=None):
    f = []
    if not r["negH_pd"]: f.append(f"non-PD (eig {min(r['negH_eigs']):.0f})")
    if r["railing"]: f.append("rail " + "/".join(r["railing"]))
    if fdeps and fd in (min(fdeps), max(fdeps)): f.append(f"f_dep edge {fd}")
    return "; ".join(f)


# profile (kappa 2.7) per member and f_dep: the cell best
prof = []
for c, i in cells.items():
    if i["kappa"] != 2.7 or not i.get("best"): continue
    r = res(i["best"])
    prof.append(dict(member=i["member"], f_dep=i["f_dep"], tag=i["best"], n_variants=len(i["variants"]),
                     lnL_heldout=r["lnL_heldout_mean"], lnL_heldout_sd=r["lnL_heldout_sd"], lnL_map=r["lnL"],
                     lnZ=r["lnZ_laplace_bdy"], m_max_A=r["theta_map"].get("m_max_A"),
                     pd=r["negH_pd"], eig0=min(r["negH_eigs"]), rails="/".join(r["railing"])))
prof.sort(key=lambda r: (r["member"], r["f_dep"]))
with open(f"{OUT}/profile_v2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(prof[0])); w.writeheader()
    for r in prof: w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})


def table(kind):
    rows = []
    for M, s in members.items():
        tag = s["best_k27"] if kind == "k27" else s["best_kfree"]
        if not tag: continue
        r = res(tag); L, Lsd = ree(tag)
        mp = [p for p in prof if p["member"] == M]
        fz = max(mp, key=lambda p: p["lnZ"])
        row = dict(member=M, label=LABEL[M], tag=tag, f_dep=s["best_fdep"], fdeps="/".join(f"{x:.1f}" for x in s["fdeps"]),
                   lnL=L, lnL_sd=Lsd, lnL_heldout=r["lnL_heldout_mean"], lnL_heldout_sd=r["lnL_heldout_sd"],
                   lnZ=r["lnZ_laplace_bdy"], f_dep_lnZmax=fz["f_dep"], lnZ_at_lnZmax_k27=fz["lnZ"],
                   kappa=r["theta_map"].get("kappa", 2.7),
                   kappa_sd=(r.get("laplace_sigma") or {}).get("kappa") if kind == "kfree" else None,
                   m_max_A=r["theta_map"].get("m_max_A"), pd=r["negH_pd"], eig0=min(r["negH_eigs"]),
                   flags=flags(r, s["fdeps"], s["best_fdep"]),
                   dlnZ_3D_union=float(u3[M]["dlnZ"]) if M in u3 else None)
        rows.append(row)
    ref = next((r for r in rows if r["member"] == "REFRC"), None)
    for r in rows:
        r["dlnZ"] = r["lnZ"] - ref["lnZ"] if ref else None
        r["dlnL"] = (r["lnL"] - ref["lnL"]) if ref and r["lnL"] is not None and ref["lnL"] is not None else None
    rows.sort(key=lambda r: -r["lnZ"])
    for i, r in enumerate(rows): r["rank"] = i + 1
    with open(f"{OUT}/table_v2_{kind}.csv", "w", newline="") as f:
        cols = ["rank", "member", "label", "tag", "f_dep", "fdeps", "kappa", "kappa_sd", "lnL", "lnL_sd", "lnL_heldout",
                "lnL_heldout_sd", "lnZ", "dlnL", "dlnZ", "dlnZ_3D_union", "f_dep_lnZmax", "lnZ_at_lnZmax_k27",
                "m_max_A", "pd", "eig0", "flags"]
        w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
    return rows


t27, tkf = table("k27"), table("kfree")

sad = []
for c, i in cells.items():
    for e in i.get("escapes", []):
        sad.append(dict(cell=c, round=e["round"], parent=e["parent"], parent_lnL_heldout=e["parent_lnL_heldout"],
                        parent_eig0=e["parent_eig0"], step=e["step"], step_unclipped=e["step_unclipped"],
                        predicted_gain=e["predicted_gain"], dominant="/".join(e["dominant"]),
                        outcome=e.get("outcome", "pending"),
                        results=json.dumps(e.get("results", {}), default=str)))
with open(f"{OUT}/saddles_v2.csv", "w", newline="") as f:
    cols = ["cell", "round", "parent", "parent_lnL_heldout", "parent_eig0", "step", "step_unclipped",
            "predicted_gain", "dominant", "outcome", "results"]
    w = csv.DictWriter(f, cols); w.writeheader()
    for r in sad: w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


def fmt(x, p=1): return "-" if x is None else f"{x:.{p}f}"


L = ["# B-series on the revised 4D likelihoods, closing pass (v2)", "",
     "native4d (189 events on certified 4D fits, 47 on the 3D NAL), m_max_A floor 20. f_dep chosen per member by",
     "held-out lnL at kappa 2.7; the free-kappa fit (kappa ~ U[-1, 7]) uses that f_dep. lnL: n_seed 8000 mean over",
     "keys 11/202/3003. lnZ: Laplace, boundary-corrected. dlnZ relative to combined (REFRC). Saddles were escaped",
     "along the most negative eigenvector of -H (saddles_v2.csv). v1 files are unchanged.", ""]
for title, rows, kf in (("kappa 2.7", t27, False), ("free kappa", tkf, True)):
    L += [f"## {title}", "", "| rank | member | f_dep | " + ("kappa | " if kf else "") +
          "lnL (sd) | dlnZ | dlnZ 3D union | f_dep max lnZ | flags |", "|---" * (8 + kf) + "|"]
    for r in rows:
        L.append(f"| {r['rank']} | {r['label']} | {r['f_dep']:.1f} | " +
                 (f"{fmt(r['kappa'], 2)} ({fmt(r['kappa_sd'], 2)}) | " if kf else "") +
                 f"{fmt(r['lnL'])} ({fmt(r['lnL_sd'])}) | {fmt(r['dlnZ'])} | {fmt(r['dlnZ_3D_union'])} | "
                 f"{r['f_dep_lnZmax']:.1f} | {r['flags'] or '-'} |")
    L.append("")
L += ["## f_dep profiles (kappa 2.7, held-out lnL; * = best)", "", "| member | " + " | ".join(
      f"{x:.1f}" for x in (0.3, 0.4, 0.5, 0.7, 0.9)) + " |", "|---" * 6 + "|"]
for M in members:
    mp = {p["f_dep"]: p for p in prof if p["member"] == M}
    bf = members[M]["best_fdep"]
    L.append(f"| {LABEL[M]} | " + " | ".join(
        (f"{mp[x]['lnL_heldout']:.1f}{'*' if x == bf else ''}{'' if mp[x]['pd'] else ' (nPD)'}" if x in mp else "-")
        for x in (0.3, 0.4, 0.5, 0.7, 0.9)) + " |")
L += ["", "nPD: -H not positive definite at the cell's best point after escapes."]
open(f"{OUT}/README_v2.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
