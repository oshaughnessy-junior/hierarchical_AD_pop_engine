"""Rank the B-series members refit with the exact cascade (reassessment 2026-11-20).

  bseries_rank.py --emit-jobs F1 F2 ...   fit-queue lines at those f_dep, each started
                                          from the model's f_dep 0.5 optimum
  bseries_rank.py --emit-extend           0.1 jobs for models whose best f_dep is 0.2
  bseries_rank.py --emit-reeval           each model's best-f_dep tag (n_seed 8000 pass)
  bseries_rank.py                         runs/BS_rank/{table.csv, profile.csv, README.md,
                                          profile_fdep.png}

Best f_dep per member = highest held-out lnL (n_seed 3000, keys 202/3003/40004).
The ranking uses the n_seed 8000 re-evaluation (keys 11/202/3003) at that f_dep.
Laplace lnZ is the n_seed 3000 Occam factor shifted by (lnL_8000 - lnL_fit).
Every row, REFRC included, comes from this driver (hier_map_laplace.py in
engine-bseries). Differences are relative to REFRC.
"""
import argparse, glob, json, os
import numpy as np

WS = os.path.expanduser("~/20261120-hier-reassess")
SCAN = os.path.join(WS, "runs/BS_scan")
OUT = os.path.join(WS, "runs/BS_rank")
MEMBERS = ["NULL", "BB", "BC", "BD", "BMIX", "NULLG", "REFRC"]
LABEL = {"NULL": "null", "BB": "B_b", "BC": "B_c", "BD": "B_d", "BMIX": "Bmix",
         "NULLG": "null + g(eta) exch 0", "REFRC": "combined (REFRC)"}
# paper tab:bseries, committed proposal (macros_results.tex, model_ranking etafix_k10_vesc1000)
PAPER = {"NULL": (-36.5, -34.4), "BB": (-12.5, -12.8), "BC": (-40.3, -38.5),
         "BD": (-19.3, -21.3), "BMIX": (-37.7, -34.3), "REFRC": (0.0, 0.0), "NULLG": (None, None)}
PAPER_FDEP = {"NULL": 0.9, "BB": 0.9, "BC": 0.9, "BD": 0.9, "BMIX": 0.9, "REFRC": 0.7, "NULLG": None}
# model_spec bounds are needed to keep a start point off the box edge
B_SRC = os.path.join(WS, "engine-bseries/demo/inference")


def load():
    res = {}
    for f in glob.glob(os.path.join(SCAN, "*/result.json")):
        r = json.load(open(f)); tag = os.path.basename(os.path.dirname(f))
        r["tag"] = tag
        rf = os.path.join(os.path.dirname(f), "reeval_n8000.json")
        r["reeval"] = json.load(open(rf)) if os.path.exists(rf) else None
        res.setdefault(r["model"], []).append(r)
    return res


def best(rs):
    return max(rs, key=lambda r: r["lnL_heldout_mean"])


def start_from(r, pad=0.002):
    th = {}
    for k, v in r["theta_map"].items():
        lo, hi = r["bounds"][k]; w = hi - lo
        th[k] = float(min(max(v, lo + pad * w), hi - pad * w))
    return json.dumps(th, separators=(",", ":"))


def job(model, f, src):
    return f"{model}_f{f} --model {model} --f-dep {f} --init-json {start_from(src)}"


def profile_fig(res, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in MEMBERS:
        if m not in res:
            continue
        rs = sorted(res[m], key=lambda r: r["f_dep"])
        f = np.array([r["f_dep"] for r in rs]); L = np.array([r["lnL_heldout_mean"] for r in rs])
        sd = np.array([r["lnL_heldout_sd"] or 0.0 for r in rs])
        ax.errorbar(f, L, yerr=sd, marker="o", ms=4, capsize=2, lw=1.2, label=LABEL[m])
    ax.set_xlabel(r"$f_{\rm dep}$"); ax.set_ylabel(r"held-out $\ln\mathcal{L}$ ($n_{\rm seed}=3000$)")
    ax.set_title("B-series, exact cascade: profile in $f_{\\rm dep}$")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-jobs", type=float, nargs="+")
    ap.add_argument("--emit-extend", action="store_true")
    ap.add_argument("--emit-reeval", action="store_true")
    a = ap.parse_args()
    res = load()
    if a.emit_jobs:
        for f in a.emit_jobs:
            for m in MEMBERS:
                src = next(r for r in res[m] if r["f_dep"] == 0.5)
                print(job(m, f, src))
        return
    if a.emit_extend:
        for m in MEMBERS:
            b = best(res[m])
            have = {r["f_dep"] for r in res[m]}
            if b["f_dep"] == min(have) and b["f_dep"] <= 0.2 + 1e-9 and 0.1 not in have:
                print(job(m, 0.1, b))
        return
    if a.emit_reeval:
        for m in MEMBERS:
            if m in res:
                print(best(res[m])["tag"])
        return
    rows, prof = [], []
    for m in MEMBERS:
        if m not in res:
            continue
        rs = sorted(res[m], key=lambda r: r["f_dep"])
        for r in rs:
            prof.append(dict(member=m, f_dep=r["f_dep"], lnL_heldout=r["lnL_heldout_mean"],
                             lnL_heldout_sd=r["lnL_heldout_sd"], lnZ_bdy=r["lnZ_laplace_bdy"],
                             railing=";".join(r["railing"]) if r.get("railing") else ""))
        b = best(rs)
        fs = [r["f_dep"] for r in rs]
        bracketed = min(fs) < b["f_dep"] < max(fs)
        if b["reeval"]:
            L, sd, src = b["reeval"]["lnL_mean"], b["reeval"]["lnL_sd"], "n8000"
        else:
            L, sd, src = b["lnL_heldout_mean"], b["lnL_heldout_sd"], "heldout_n3000"
        lnZ = b["lnZ_laplace_bdy"] - b["lnL"] + L
        rows.append(dict(member=m, label=LABEL[m], best_f_dep=b["f_dep"], bracketed=bracketed,
                         lnL=L, lnL_sd=sd, lnL_src=src, lnZ=lnZ,
                         railing=";".join(b["railing"]) if b.get("railing") else "",
                         negH_pd=b["negH_pd"], n_par=len(b["names"]),
                         scan=" ".join(f"{r['f_dep']}:{r['lnL_heldout_mean']:.1f}" for r in rs),
                         theta=b["theta_map"], paper_dlnL=PAPER[m][0], paper_dlnZ=PAPER[m][1],
                         paper_f_dep=PAPER_FDEP[m]))
    ref = next((r for r in rows if r["member"] == "REFRC"), None)
    for r in rows:
        r["dlnL"] = r["lnL"] - ref["lnL"] if ref else None
        r["dlnZ"] = r["lnZ"] - ref["lnZ"] if ref else None
    os.makedirs(OUT, exist_ok=True)
    cols = ["member", "label", "best_f_dep", "bracketed", "lnL", "lnL_sd", "lnL_src", "lnZ", "dlnL",
            "dlnZ", "paper_dlnL", "paper_dlnZ", "paper_f_dep", "n_par", "negH_pd", "railing", "scan"]
    fmt = lambda v: "" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    with open(os.path.join(OUT, "table.csv"), "w") as f:
        f.write(",".join(cols) + ",theta_map\n")
        for r in rows:
            f.write(",".join(fmt(r[c]) for c in cols) + ',"'
                    + json.dumps({k: round(v, 4) for k, v in r["theta"].items()}).replace('"', "'") + '"\n')
    pc = ["member", "f_dep", "lnL_heldout", "lnL_heldout_sd", "lnZ_bdy", "railing"]
    with open(os.path.join(OUT, "profile.csv"), "w") as f:
        f.write(",".join(pc) + "\n")
        for p in prof:
            f.write(",".join(fmt(p[c]) for c in pc) + "\n")
    profile_fig(res, os.path.join(OUT, "profile_fdep.png"))
    rk_new = sorted(rows, key=lambda r: -r["dlnZ"])
    rk_old = sorted([r for r in rows if r["paper_dlnZ"] is not None], key=lambda r: -r["paper_dlnZ"])
    fd = sorted({p["f_dep"] for p in prof})
    lines = ["# B-series refit with the exact cascade", "",
             "Generated by `engine-bseries/demo/inference/bseries_rank.py`. Each member is",
             "refit by MAP + Laplace with every cascade proposal equal to its target",
             "(`hier_map_laplace.py`). Every row, REFRC included, comes from this driver.",
             "Best f_dep = highest held-out lnL at n_seed 3000; lnL there is the n_seed 8000 /",
             "n_b 6000 mean over keys 11, 202, 3003. Laplace lnZ keeps the n_seed 3000 Occam",
             "factor. Differences are relative to REFRC. Paper columns are tab:bseries",
             "(committed proposal, posterior mean, 10 keys).", "",
             "| member | best f_dep | bracketed | lnL (sd) | dlnL | dlnZ | paper dlnL | paper dlnZ | rails |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rk_new:
        lines.append(f"| {r['label']} | {r['best_f_dep']} | {'yes' if r['bracketed'] else 'NO'} | "
                     f"{r['lnL']:.1f} ({(r['lnL_sd'] or float('nan')):.1f}, {r['lnL_src']}) | "
                     f"{r['dlnL']:+.1f} | {r['dlnZ']:+.1f} | {fmt(r['paper_dlnL'])} | {fmt(r['paper_dlnZ'])} | {r['railing']} |")
    lines += ["", "Rank by dlnZ, exact: " + " > ".join(r["label"] for r in rk_new),
              "Rank by dlnZ, paper: " + " > ".join(r["label"] for r in rk_old), "",
              "## Profile: held-out lnL (n_seed 3000) against f_dep", "",
              "| member | " + " | ".join(str(f) for f in fd) + " |",
              "|---|" + "---|" * len(fd)]
    for m in MEMBERS:
        if m not in res:
            continue
        d = {r["f_dep"]: r["lnL_heldout_mean"] for r in res[m]}
        lines.append(f"| {LABEL[m]} | " + " | ".join(f"{d[f]:.1f}" if f in d else "" for f in fd) + " |")
    lines += ["", "![profile](profile_fdep.png)"]
    open(os.path.join(OUT, "README.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
