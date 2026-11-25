"""O4-only follow-up: f_dep extension (B_d 0.1, REFRC 0.5, B_b 0.2), B_b multi-start (0.2-0.5), n_seed 8000
Laplace (keys 11/202/3003). Reads ../scan (never writes it); writes ../scan_n8000.
Prints wanted units "UNIT args" and "#COMPLETE". Units ending __lap run the Laplace script.
"""
import json, math, os, sys
import numpy as np

ROOT = os.path.expanduser("~/hier-reassess-20261121/runs/BS4D_O4")
OLD, R, LAPDIR = f"{ROOT}/scan", f"{ROOT}/scan_n8000", f"{ROOT}/laplace_n8000"
SHIFT = math.log(168 / 236.0) + math.log(3.0456693791669833 / (52962816.0 / (365.25 * 86400.0)))
FL = "--m-max-lo 20 --likelihood native4d --native4d-nodes 32 --selection injections --event-subset o4 --kappa 2.7 --free-kappa -1 7 --log-r-range -8 12"
G_NATS, BOX, INSET, MAX_ROUNDS = 3.0, 0.05, 0.005, 3
LAPKEYS = (11, 202, 3003)
units, order, state = {}, [], {}
EC = f"{R}/escape_children.json"; children = json.load(open(EC)) if os.path.exists(EC) else {}


def want(u, a):
    if u not in units: units[u] = a; order.append(u)


def path(t):
    for d in (R, OLD):
        if os.path.exists(f"{d}/{t}/result.json"): return f"{d}/{t}/result.json"
    return None


def res(t): return json.load(open(path(t)))


def status(t):
    if path(t): return "done"
    return "failed" if os.path.exists(f"{R}/{t}/FAILED") else "pending"


def fit_args(M, fd, th): return f"--model {M} --f-dep {fd:.1f} --init-json {json.dumps({k: float(v) for k, v in th.items()}, separators=(',', ':'))} {FL}"


def key(t): r = res(t); return (bool(r["negH_pd"]), r["lnL_heldout_mean"])


def choose(ts):
    ok = [t for t in ts if status(t) == "done"]; return max(ok, key=key) if ok else None


def escape(r):
    n = r["names"]; th = np.array([r["theta_map"][k] for k in n]); lo = np.array([r["bounds"][k][0] for k in n])
    hi = np.array([r["bounds"][k][1] for k in n]); w = hi - lo; ev, V = np.linalg.eigh(-np.array(r["hessian"])); lam, v = ev[0], V[:, 0]
    s = min(np.sqrt(2 * G_NATS / abs(lam)), float(np.min(BOX * w / np.maximum(np.abs(v), 1e-300))))
    return {lab: dict(zip(n, np.clip(th + sg * s * v, lo + INSET * w, hi - INSET * w).tolist())) for sg, lab in ((1, "p"), (-1, "m"))}


def run_cell(M, fd, starts, cell):
    """starts: list of (tag, theta or None). Returns (settled, best)."""
    tags = []
    for t, th in starts:
        if status(t) == "pending":
            if th is None: continue
            want(t, fit_args(M, fd, th))
        tags.append(t)
    info = dict(member=M, f_dep=fd, variants=list(tags), escapes=[])
    if any(status(t) == "pending" for t in tags): state[cell] = info; return False, None
    best = choose(tags); rnd = 1
    while best and not res(best)["negH_pd"] and rnd <= MAX_ROUNDS:
        st = escape(res(best))
        if best not in children: children[best] = [f"{cell}_x{rnd}p", f"{cell}_x{rnd}m"]
        pair = children[best]
        for t, lab in zip(pair, ("p", "m")):
            if status(t) == "pending": want(t, fit_args(M, fd, st[lab]))
        tags += pair; info["variants"] = list(tags)
        if any(status(t) == "pending" for t in pair): state[cell] = info; return False, None
        new = choose(tags); info["escapes"].append(dict(parent=best, children=pair, new=new))
        if new not in pair: break
        best = new; rnd += 1
    info["best"] = best; state[cell] = info
    return True, best


def lap(t):
    if all(os.path.exists(f"{LAPDIR}/{t}_n8000_k{k}.json") for k in LAPKEYS) or os.path.exists(f"{R}/{t}/FAILED_lap"): return True
    want(f"{t}__lap", path(t)); return False


done = True
# 1. Laplace at the table points and the best-lnZ points of every member (existing fits)
old = json.load(open(f"{OLD}/plan_state.json"))["members"]; pts = []
for M, m in old.items():
    prof = {float(f): t for f, t in m["profile"].items()}
    bz = max(prof.values(), key=lambda t: res(t)["lnZ_laplace_bdy"]); pts += [m["best"]] + ([bz] if bz != m["best"] else [])
for t in pts: done &= lap(t)
# 2. f_dep extensions (one step), started from the edge cell's fit; 3. B_b starts at 0.2-0.5
thBB = res("BB_f0.4")["theta_map"]
ext = [("BD", 0.1, "BD_f0.2_e1p"), ("REFRC", 0.5, "REFRC_f0.4")]
for M, fd, src in ext:
    ok, b = run_cell(M, fd, [(f"{M}_f{fd:.1f}", res(src)["theta_map"])], f"{M}_f{fd:.1f}"); done &= ok
    if ok and b: done &= lap(b)


def bb_starts(fd):
    names = list(thBB); bd = res("BB_f0.4")["bounds"]
    c, d = res("BC_f0.3_e2m")["theta_map"], res("BD_f0.3")["theta_map"]
    b3 = json.load(open(os.path.expanduser("~/hier-reassess-20261121/runs/diag_src/BS_scan_mmax20/BB_f0.3/result.json")))["theta_map"]
    th3 = {k: (b3[k] - 0.6365 + SHIFT if k.startswith("log_R_") else b3.get(k, 2.2)) for k in names}
    d1 = dict(thBB, kappa=1.0, m_max_A=50.0, mu_B=12.0, sig_B=1.0, mu_chiB=0.2, sig_chiB=0.1, spin_chi=0.15)
    rng = np.random.default_rng(20261125)
    d2 = {k: (thBB[k] + rng.uniform(-1, 1) if k.startswith("log_R_") else float(bd[k][0] + (0.2 + 0.6 * rng.random()) * (bd[k][1] - bd[k][0]))) for k in names}
    clip = lambda th: {k: float(np.clip(th[k], bd[k][0] + 1e-3 * (bd[k][1] - bd[k][0]), bd[k][1] - 1e-3 * (bd[k][1] - bd[k][0]))) for k in names}
    return [(f"BB_f{fd:.1f}_s{n}", clip(t)) for n, t in (("BC", {k: c[k] for k in names}), ("BD", {k: d[k] for k in names}), ("3D", th3), ("D1", d1), ("D2", d2))]


bb = {}
for fd in (0.2, 0.3, 0.4, 0.5):
    base = f"BB_f{fd:.1f}"
    st = [(base, None if path(base) else res("BB_f0.3")["theta_map"])] + bb_starts(fd)
    ok, b = run_cell("BB", fd, st, f"BBms_f{fd:.1f}"); done &= ok
    if ok: bb[fd] = b
if len(bb) == 4:
    L = {f: res(b)["lnL_heldout_mean"] for f, b in bb.items()}; Z = {f: res(b)["lnZ_laplace_bdy"] for f, b in bb.items()}
    for f in {max(L, key=L.get), max(Z, key=Z.get)}: done &= lap(bb[f])
    state["BB_best"] = dict(best_heldout=bb[max(L, key=L.get)], best_lnZ=bb[max(Z, key=Z.get)], per_fdep=bb)

# 4. saddle escape of the mapped-B_c B_b basin where it is not positive definite (0.2, 0.4), and n8000 Laplace of that basin
for fd in (0.2, 0.3, 0.4, 0.5):
    t = f"BB_f{fd:.1f}_sBC"
    if path(t):
        ok, b = run_cell("BB", fd, [(t, None)], f"BBsBC_f{fd:.1f}"); done &= ok
        if ok and b: done &= lap(b) & lap(t)
json.dump(children, open(EC + ".tmp", "w"), indent=1); os.replace(EC + ".tmp", EC)
json.dump(dict(cells=state, laplace_points=pts, complete=bool(done)), open(f"{R}/plan_state.json.tmp", "w"), indent=1, default=str)
os.replace(f"{R}/plan_state.json.tmp", f"{R}/plan_state.json")
for u in order: print(u, units[u])
if done: print("#COMPLETE")
