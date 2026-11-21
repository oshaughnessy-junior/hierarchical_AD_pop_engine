"""Union of the floor-40 and floor-20 B-series scans (reassessment 2026-11-21).

The floor-40 prior U[40,70] is a subset of U[20,70], so every floor-40 optimum is an
admissible floor-20 point. This takes, per model, the best held-out lnL over both
scans (any f_dep, any start) and reports its n_seed 8000 lnL and Laplace lnZ (the
n_seed 3000 Occam factor shifted to the 8000 lnL, as bseries_rank.py does). The lnZ
of a floor-40 fit uses that fit's prior width; with the wider floor-20 box it would
be lower by ln(50/30) = 0.51 per such model. Writes runs/BS_rank/table_union.csv.
"""
import glob, json, math, os
WS = os.path.expanduser("~/20261120-hier-reassess")
SC = {"floor40": "runs/BS_scan", "floor20": "runs/BS_scan_mmax20"}
rows = {}
for lab, d in SC.items():
    for f in glob.glob(os.path.join(WS, d, "*/result.json")):
        r = json.load(open(f)); rf = os.path.join(os.path.dirname(f), "reeval_n8000.json")
        r["reeval"] = json.load(open(rf)) if os.path.exists(rf) else None
        r["scan"] = lab; r["tag"] = os.path.basename(os.path.dirname(f))
        if r["model"] not in rows or r["lnL_heldout_mean"] > rows[r["model"]]["lnL_heldout_mean"]:
            rows[r["model"]] = r
out = []
for m, r in rows.items():
    L = r["reeval"]["lnL_mean"] if r["reeval"] else r["lnL_heldout_mean"]
    sd = r["reeval"]["lnL_sd"] if r["reeval"] else r["lnL_heldout_sd"]
    lnZ = r["lnZ_laplace_bdy"] - r["lnL"] + L
    if r["bounds"]["m_max_A"][0] >= 40:
        lnZ -= math.log(50.0 / 30.0)          # same U[20,70] prior width for every row
    out.append(dict(member=m, scan=r["scan"], tag=r["tag"], f_dep=r["f_dep"], m_max_A=r["theta_map"]["m_max_A"],
                    lnL_heldout=r["lnL_heldout_mean"], lnL=L, lnL_sd=sd, lnL_src="n8000" if r["reeval"] else "heldout",
                    lnZ=lnZ))
ref = next(o for o in out if o["member"] == "REFRC")
for o in out:
    o["dlnL"] = o["lnL"] - ref["lnL"]; o["dlnZ"] = o["lnZ"] - ref["lnZ"]
out.sort(key=lambda o: -o["dlnZ"])
cols = list(out[0])
with open(os.path.join(WS, "runs/BS_rank/table_union.csv"), "w") as f:
    f.write(",".join(cols) + "\n")
    for o in out:
        f.write(",".join(f"{o[c]:.2f}" if isinstance(o[c], float) else str(o[c]) for c in cols) + "\n")
for o in out:
    print(f"{o['member']:6s} {o['scan']} {o['tag']:15s} f_dep={o['f_dep']} m_max={o['m_max_A']:.1f} "
          f"lnL={o['lnL']:.1f}({o['lnL_sd'] or 0:.1f},{o['lnL_src']}) dlnL={o['dlnL']:+.1f} dlnZ={o['dlnZ']:+.1f}")
