"""Rank the B-series members on the revised 4D likelihoods (native4d, kappa 2.7).

Reads runs/BS4D_scan/<MODEL>_f<fdep>/result.json (held-out lnL at n_seed 3000, Laplace lnZ)
and reeval_n8000.json (keys 11/202/3003) for each model's best f_dep. Writes
runs/BS4D_rank/{table.csv,profile.csv,kappa_profile.csv,README.md}, beside the 3D union ranking
(runs/BS_rank/table_union.csv). 4D lnL has a different absolute scale from 3D: compare
only differences within each likelihood.
"""
import csv, glob, json, os, re
import numpy as np

B = os.path.expanduser("~/20261120-hier-reassess/runs")
SC, OUT = f"{B}/BS4D_scan", f"{B}/BS4D_rank"
os.makedirs(OUT, exist_ok=True)
LABEL = {"NULL": "null", "BB": "B_b", "BC": "B_c", "BD": "B_d", "BMIX": "Bmix",
         "NULLG": "null + g(eta) exch 0", "REFRC": "combined (REFRC)"}

prof, kap = [], []
for p in sorted(glob.glob(f"{SC}/*/result.json")):
    tag = os.path.basename(os.path.dirname(p)); r = json.load(open(p))
    row = dict(tag=tag, member=r["model"], f_dep=r["f_dep"],
               m_max_A=dict(r["theta_map"]).get("m_max_A", np.nan),
               lnL_heldout=r["lnL_heldout_mean"], lnL_heldout_sd=r["lnL_heldout_sd"],
               lnZ=r.get("lnZ_laplace_bdy", r.get("lnZ_laplace")), negH_pd=r.get("negH_pd"),
               railing=";".join(r.get("railing") or []))
    m = re.search(r"_k([0-9.]+)$", tag)
    (kap if m else prof).append(dict(row, kappa=float(m.group(1)) if m else 2.7))

best = {}
for row in prof:
    b = best.get(row["member"])
    if b is None or row["lnL_heldout"] > b["lnL_heldout"]: best[row["member"]] = row
for m, row in best.items():
    rp = f"{SC}/{row['tag']}/reeval_n8000.json"
    if os.path.exists(rp):
        e = json.load(open(rp)); row.update(lnL=e["lnL_mean"], lnL_sd=e["lnL_sd"], lnL_src="n8000")
    else:
        row.update(lnL=row["lnL_heldout"], lnL_sd=row["lnL_heldout_sd"], lnL_src="heldout_n3000")
    row["bracketed"] = (row["f_dep"] not in (min(x["f_dep"] for x in prof if x["member"] == m),
                                             max(x["f_dep"] for x in prof if x["member"] == m)))

u3 = {r["member"]: r for r in csv.DictReader(open(f"{B}/BS_rank/table_union.csv"))}
ref = best.get("REFRC")
rows = sorted(best.values(), key=lambda r: -r["lnZ"])
for r in rows:
    r["dlnL"] = r["lnL"] - ref["lnL"] if ref else np.nan
    r["dlnZ"] = r["lnZ"] - ref["lnZ"] if ref else np.nan
    r["dlnZ_3D_union"] = float(u3[r["member"]]["dlnZ"]) if r["member"] in u3 else np.nan
    r["f_dep_3D_union"] = float(u3[r["member"]]["f_dep"]) if r["member"] in u3 else np.nan
r3 = sorted(u3, key=lambda k: -float(u3[k]["dlnZ"]))
for i, r in enumerate(rows):
    r["rank_4D"], r["rank_3D"] = i + 1, (r3.index(r["member"]) + 1 if r["member"] in r3 else None)

cols = ["member", "label", "tag", "f_dep", "bracketed", "m_max_A", "lnL", "lnL_sd", "lnL_src", "lnL_heldout",
        "lnZ", "dlnL", "dlnZ", "dlnZ_3D_union", "f_dep_3D_union", "rank_4D", "rank_3D", "negH_pd", "railing"]
with open(f"{OUT}/table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for r in rows:
        r["label"] = LABEL.get(r["member"], r["member"])
        w.writerow([f"{r[c]:.2f}" if isinstance(r.get(c), float) else r.get(c) for c in cols])
with open(f"{OUT}/profile.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["member", "f_dep", "tag", "m_max_A", "lnL_heldout", "lnL_heldout_sd", "lnZ", "railing"])
    for r in sorted(prof, key=lambda r: (r["member"], r["f_dep"])):
        w.writerow([r["member"], f"{r['f_dep']:.2f}", r["tag"], f"{r['m_max_A']:.2f}", f"{r['lnL_heldout']:.2f}",
                    f"{r['lnL_heldout_sd']:.2f}", f"{r['lnZ']:.2f}", r["railing"]])
ktag = re.sub(r"_k[0-9.]+$", "", kap[0]["tag"]) if kap else None
kbase = [dict(r, kappa=2.7) for r in prof if r["tag"] == ktag]
krows = sorted(kap + kbase, key=lambda r: r["kappa"])
if kap:
    with open(f"{OUT}/kappa_profile.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["kappa", "tag", "lnL_heldout", "lnL_heldout_sd", "lnZ"])
        for r in krows: w.writerow([r["kappa"], r["tag"], f"{r['lnL_heldout']:.2f}", f"{r['lnL_heldout_sd']:.2f}", f"{r['lnZ']:.2f}"])

L = ["# B-series on the revised 4D likelihoods", "",
     "native4d: 189 events on certified 4D fits, 47 on the 3D NAL; kappa 2.7; z-resolved selection;",
     "exact cascade; m_max_A floor 20; f_dep in {0.3, 0.4, 0.5}. lnL is the n_seed 8000 mean over",
     "keys 11/202/3003 where available; lnZ is Laplace (boundary-corrected) at n_seed 3000.",
     "4D lnL has a different absolute scale from 3D, so only differences are comparable.", "",
     "| member | f_dep | m_max_A | lnL (sd) | dlnL | dlnZ 4D | dlnZ 3D union | rank 4D / 3D |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['label']} | {r['f_dep']:.1f}{'' if r['bracketed'] else ' (edge)'} | {r['m_max_A']:.1f} | "
             f"{r['lnL']:.1f} ({r['lnL_sd']:.1f}) | {r['dlnL']:+.1f} | {r['dlnZ']:+.1f} | {r['dlnZ_3D_union']:+.1f} | "
             f"{r['rank_4D']} / {r['rank_3D']} |")
if kap:
    L += ["", f"kappa profile on {ktag} (held-out lnL, n_seed 3000):", "",
          "| kappa | lnL_heldout | lnZ |", "|---|---|---|"]
    L += [f"| {r['kappa']} | {r['lnL_heldout']:.1f} ({r['lnL_heldout_sd']:.1f}) | {r['lnZ']:.1f} |" for r in krows]
L += ["", "Edge = best f_dep sits on the end of the scanned grid (profile not bracketed).",
      "Rails and non-positive-definite Hessians are listed in table.csv."]
open(f"{OUT}/README.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
