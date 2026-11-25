"""Multi-start O4-only campaign (all seven members, data sets ALL168 and NO241011): decide next units from finished results.

Rules fixed before screening:
 - Screening runs at f_dep 0.3 only (basins are locations in parameter space; f_dep is not a fitted parameter).
   Each start: 20 Adam steps at n_seed 3000; score = mean held-out lnL at keys 202 and 3003 of its final point.
 - Two end points are the same basin if every shape parameter (kappa, m_max_A, mu_B, sig_B, mu_chiB, sig_chiB, spin_chi,
   and alpha_B, f_peakB, mu_C, sig_C where present) agrees to 0.20 of its prior width; a basin is the best-scoring member.
   A start counts as having reached a basin only if its own score is within 25 nats of that basin's best (else "unconverged").
 - Per member and data set: the 3 best distinct basins (by score, whatever the gap) are converged by the driver's fit
   at f_dep 0.3 (all kept basins) and at 0.4 (the 2 best at 0.3); one saddle escape round where -H is not PD (v2 rule: +/- step
   along the most negative eigenvector, 3-nat gain, 5% prior-width box). Basin result = PD first, then lnL_heldout.
 - Reduced f_dep grid (cost): 0.2 and 0.5 are run only as the one-step extension of the best basin at the best f_dep, when that is
   0.3 or 0.4 (edge of the grid). Full grid 0.2-0.5 per basin is NOT run.
 - Evaluation: bs4d_ms_eval.py at the best f_dep of every basin of every member/data set.
Files: screen/<M>_<DS>_f0.3.jsonl, fits/<M>_<DS>_b<k>_f<fd>, eval/<tag>.npz.
"""
import json, math, os, sys
import numpy as np

ROOT = os.path.expanduser("~/hier-reassess-20261121/runs/BS4D_O4_ms"); SCR, FIT, EVL = f"{ROOT}/screen", f"{ROOT}/fits", f"{ROOT}/eval"
MEMBERS = ["NULL", "BB", "BC", "BD", "BMIX", "NULLG", "REFRC"]; DSS = ["ALL168", "NO241011"]
SHAPE = ["kappa", "m_max_A", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "spin_chi", "alpha_B", "f_peakB", "mu_C", "sig_C"]
TOL, NKEEP, WITHIN = 0.20, 3, 25.0
FL = "--m-max-lo 20 --likelihood native4d --native4d-nodes 32 --selection injections --event-subset o4 --kappa 2.7 --free-kappa -1 7 --log-r-range -8 12"
G_NATS, BOX, INSET = 3.0, 0.05, 0.005
units, order, state = {}, [], {}
EC = f"{ROOT}/escape_children.json"; children = json.load(open(EC)) if os.path.exists(EC) else {}


def want(u, a):
    if u not in units: units[u] = a; order.append(u)


def status(t): return "done" if os.path.exists(f"{FIT}/{t}/result.json") else ("failed" if os.path.exists(f"{FIT}/{t}/FAILED") else "pending")
def res(t): return json.load(open(f"{FIT}/{t}/result.json"))


def screen_rows(M, DS):
    f = f"{SCR}/{M}_{DS}_f0.3"
    if not os.path.exists(f + ".done"): return None
    return [json.loads(l) for l in open(f + ".jsonl")]


def load_bounds(M):
    return json.load(open(f"{ROOT}/bounds_{M}.json"))


def basins(M, DS):
    """Distinct basins (best member first), per the rule above. Returns list of dicts with label, theta, score, starts."""
    rows = [r for r in (screen_rows(M, DS) or []) if r["sup"] and np.isfinite(r["lnL_ho_mean"])]
    b = load_bounds(M); keys = [k for k in SHAPE if k in b]; rows.sort(key=lambda r: -r["lnL_ho_mean"])
    out = []
    for r in rows:
        u = np.array([(r["theta"][k] - b[k][0]) / (b[k][1] - b[k][0]) for k in keys])
        for o in out:
            if np.max(np.abs(u - o["u"])) <= TOL:
                (o["starts"] if r["lnL_ho_mean"] >= o["score"] - WITHIN else o["unconverged"]).append(r["start"]); break
        else: out.append(dict(u=u, theta=r["theta"], score=r["lnL_ho_mean"], starts=[r["start"]], unconverged=[], first=r["start"]))
    return out[:NKEEP], out


def fit_args(M, DS, fd, th):
    x = " --exclude-events S241011k" if DS == "NO241011" else ""
    return f"--model {M} --f-dep {fd:.1f} --init-json {json.dumps({k: float(v) for k, v in th.items()}, separators=(',', ':'))} {FL}{x}"


def escape(r):
    n = r["names"]; th = np.array([r["theta_map"][k] for k in n]); lo = np.array([r["bounds"][k][0] for k in n]); hi = np.array([r["bounds"][k][1] for k in n]); w = hi - lo
    ev, V = np.linalg.eigh(-np.array(r["hessian"])); lam, v = ev[0], V[:, 0]
    s = min(np.sqrt(2 * G_NATS / abs(lam)), float(np.min(BOX * w / np.maximum(np.abs(v), 1e-300))))
    return {lab: dict(zip(n, np.clip(th + sg * s * v, lo + INSET * w, hi - INSET * w).tolist())) for sg, lab in ((1, "p"), (-1, "m"))}


def key(t): r = res(t); return (bool(r["negH_pd"]), r["lnL_heldout_mean"])


def run_fit_cell(M, DS, fd, b, theta):
    """one basin at one f_dep with at most one escape round. Returns (settled, best_tag)."""
    t = f"{M}_{DS}_b{b}_f{fd:.1f}"
    if status(t) == "pending": want(t, fit_args(M, DS, fd, theta)); return False, None
    if status(t) == "failed": return True, None
    best = t
    if not res(t)["negH_pd"]:
        if t not in children: children[t] = [f"{t}_xp", f"{t}_xm"]
        st = escape(res(t)); pend = False
        for c, lab in zip(children[t], ("p", "m")):
            if status(c) == "pending": want(c, fit_args(M, DS, fd, st[lab])); pend = True
        if pend: return False, None
        cand = [t] + [c for c in children[t] if status(c) == "done"]; best = max(cand, key=key)
    return True, best


done_all = True; members = {}
for M in MEMBERS:
    if not os.path.exists(f"{ROOT}/bounds_{M}.json"): done_all = False
    for DS in DSS:
        if not os.path.exists(f"{SCR}/{M}_{DS}_f0.3.done"):
            want(f"screen__{M}__{DS}", ""); done_all = False; continue
        if not os.path.exists(f"{ROOT}/bounds_{M}.json"): continue
        kept, allb = basins(M, DS)
        rec = dict(n_starts=len(screen_rows(M, DS)), basins=[dict(first=o["first"], score=o["score"], starts=o["starts"], unconverged=o["unconverged"]) for o in allb], kept=len(kept)); members[f"{M}|{DS}"] = rec
        fitted = {}; settled = True
        for i, o in enumerate(kept, 1):
            ok, b = run_fit_cell(M, DS, 0.3, i, o["theta"]); settled &= ok
            if ok: fitted[(i, 0.3)] = b
        if not settled: done_all = False; continue
        rank = sorted([i for i in range(1, len(kept) + 1) if fitted.get((i, 0.3))], key=lambda i: -res(fitted[(i, 0.3)])["lnL_heldout_mean"])
        for i in rank[:2]:
            ok, b = run_fit_cell(M, DS, 0.4, i, res(fitted[(i, 0.3)])["theta_map"]); settled &= ok
            if ok: fitted[(i, 0.4)] = b
        if not settled: done_all = False; continue
        lh = {(i, fd): res(b)["lnL_heldout_mean"] for (i, fd), b in fitted.items() if b}
        (bi, bfd) = max(lh, key=lh.get); ext = {0.4: 0.5, 0.3: 0.2}[bfd]
        ok, b = run_fit_cell(M, DS, ext, bi, res(fitted[(bi, bfd)])["theta_map"]); settled &= ok
        if ok and b: fitted[(bi, ext)] = b
        if not settled: done_all = False; continue
        tags = []
        for i, o in enumerate(kept, 1):
            cand = {fd: b for (j, fd), b in fitted.items() if j == i and b}
            if not cand: continue
            fd = max(cand, key=lambda f: res(cand[f])["lnL_heldout_mean"]); tags.append((i, fd, cand[fd]))
        rec.update(fits={f"{i}|{fd}": b for (i, fd), b in fitted.items()}, best=[dict(basin=i, f_dep=fd, tag=t) for i, fd, t in tags])
        for i, fd, t in tags:
            if not os.path.exists(f"{EVL}/{t}.done"): want(f"{t}__eval", f"{FIT}/{t}/result.json {DS}"); done_all = False

json.dump(children, open(EC + ".tmp", "w"), indent=1); os.replace(EC + ".tmp", EC)
json.dump(dict(members=members, complete=bool(done_all and all(f"{M}|{DS}" in members for M in MEMBERS for DS in DSS))), open(f"{ROOT}/plan_state.json.tmp", "w"), indent=1, default=str); os.replace(f"{ROOT}/plan_state.json.tmp", f"{ROOT}/plan_state.json")
for u in order: print(u, units[u])
if done_all and all(f"{M}|{DS}" in members for M in MEMBERS for DS in DSS): print("#COMPLETE")
