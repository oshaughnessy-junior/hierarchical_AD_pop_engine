"""Round 3 of the 4D B-series: decide the next units from finished results only.

Prints wanted units as "UNIT args" lines, writes plan_state_v3.json and escape_children.json,
and prints "#COMPLETE" when nothing is left. Per member:
  1. kappa 2.7 f_dep profile. A cell is restarted from a neighbouring cell's optimum when that
     neighbour's held-out lnL is more than 5 nats higher (once per neighbour); REFRC 0.4 and 0.5
     are also restarted from the 0.3 optimum. Non-positive-definite -H gets the v2 saddle escape.
     f_dep extension: up as in v2 (0.7 for BD/BMIX/REFRC or best at 0.5; 0.9 if best at 0.7);
     down to 0.2 for REFRC and NULL or any member whose best is at 0.3, and 0.1 if best at 0.2.
  2. free kappa ~ U[-1, 7] with every log_R_* on [logr0-8, logr0+H] (H = 12; all members move to
     16 if any best free-kappa fit still rails a rate at the upper bound), at the member's best
     kappa 2.7 f_dep, from the v2 free-kappa optimum (or the kappa 2.7 optimum if f_dep moved).
  3. n_seed 8000 re-evaluation of both best points.
  4. once every member is done: Laplace at n_seed 8000 (key 11) for the top three of each table.
Cell best = positive-definite variants first, then highest held-out lnL.
"""
import json, os
import numpy as np

R = os.path.expanduser("~/hier-reassess-20261121/runs/BS4D_scan")
MODELS = ["NULL", "BB", "BC", "BD", "BMIX", "NULLG", "REFRC"]
GRID = [0.3, 0.4, 0.5]
UP = {0.5: 0.7, 0.7: 0.9}; DOWN = {0.3: 0.2, 0.2: 0.1}
FORCED_UP = {"BD", "BMIX", "REFRC"}; FORCED_DOWN = {"REFRC", "NULL"}
EXTRA_RESTART = {("REFRC", 0.4): 0.3, ("REFRC", 0.5): 0.3}
DIP = 5.0
HI_LEVELS = [12.0, 16.0]
FL = "--m-max-lo 20 --likelihood native4d --native4d-window --native4d-nodes 32"
G_NATS, BOX, INSET, MAX_ROUNDS = 3.0, 0.05, 0.005, 3
SEEDS = "--key 11 --n-seed 8000 --n-b 6000"

units, order, cells_state = {}, [], {}
EC_PATH = f"{R}/escape_children.json"
if os.path.exists(EC_PATH):
    children = json.load(open(EC_PATH))
else:                                         # seed from the v2 closing pass
    children = {}
    v2 = json.load(open(f"{R}/plan_state.json"))
    for c, i in v2["cells"].items():
        for e in i.get("escapes", []):
            children.setdefault(e["parent"], [f"{c}_e{e['round']}p", f"{c}_e{e['round']}m"])


def want(unit, args):
    if unit not in units:
        units[unit] = args; order.append(unit)


def res(tag):
    p = f"{R}/{tag}/result.json"
    return json.load(open(p)) if os.path.exists(p) else None


def status(tag):
    if os.path.exists(f"{R}/{tag}/result.json"): return "done"
    if os.path.exists(f"{R}/{tag}/FAILED"): return "failed"
    return "pending"


def ftag(fd): return f"f{fd:.1f}"
def fcode(fd): return f"{int(round(fd * 10)):02d}"


def mode_flags(mode):
    if mode == "k27": return "--kappa 2.7"
    return f"--kappa 2.7 --free-kappa -1 7 --log-r-range -8 {mode[1]:g}"


def fit_args(M, fd, mode, theta):
    th = {k: float(v) for k, v in theta.items()}
    if mode == "k27": th.pop("kappa", None)
    else: th.setdefault("kappa", 2.7)
    return (f"--model {M} --f-dep {fd:.1f} --init-json {json.dumps(th, separators=(',', ':'))} "
            f"{FL} {mode_flags(mode)}")


def key(t): r = res(t); return (bool(r["negH_pd"]), r["lnL_heldout_mean"])


def choose(tags):
    ok = [t for t in tags if status(t) == "done"]
    return max(ok, key=key) if ok else None


def escape(r):
    names = r["names"]; th = np.array([r["theta_map"][n] for n in names])
    lo = np.array([r["bounds"][n][0] for n in names]); hi = np.array([r["bounds"][n][1] for n in names]); w = hi - lo
    ev, V = np.linalg.eigh(-np.array(r["hessian"])); lam, v = ev[0], V[:, 0]
    s0 = np.sqrt(2 * G_NATS / abs(lam)); s = min(s0, float(np.min(BOX * w / np.maximum(np.abs(v), 1e-300))))
    starts, eff = {}, {}
    for sg, lab in ((1, "p"), (-1, "m")):
        t = np.clip(th + sg * s * v, lo + INSET * w, hi - INSET * w)
        starts[lab] = dict(zip(names, t.tolist())); eff[lab] = float(np.dot(t - th, sg * v) / s)
    dom = [names[i] for i in np.argsort(-np.abs(v))[:3]]
    info = dict(lambda_min=float(lam), step=float(s), step_unclipped=float(s0),
                predicted_gain=float(0.5 * abs(lam) * s * s), dominant=dom, achieved=eff)
    blocked = max(eff.values()) < 0.1
    if blocked:
        rails = set(r.get("railing") or []); info["blocked_by_rail"] = ([n for n in dom if n in rails] or dom)[0]
    return starts, info, blocked


def child_tags(cell, rnd, parent):
    if parent in children: return children[parent]
    used = {t for v in children.values() for t in v}
    base = f"{cell}_e{rnd}"; sfx = ""
    k = 0
    while f"{base}p{sfx}" in used or status(f"{base}p{sfx}") != "pending":
        k += 1; sfx = f"_{k}"
    children[parent] = [f"{base}p{sfx}", f"{base}m{sfx}"]
    return children[parent]


def run_cell(M, fd, mode, starts):
    cell = f"{M}_{ftag(fd)}" + ("" if mode == "k27" else f"_kw{mode[1]:g}")
    info = dict(cell=cell, member=M, f_dep=fd, mode=str(mode), variants=[], escapes=[])
    tags = []
    for tag, th in starts:
        if status(tag) == "pending" and th is not None: want(tag, fit_args(M, fd, mode, th))
        if status(tag) == "pending" and th is None: continue        # restart not yet possible
        tags.append(tag)
    info["variants"] = list(tags)
    if not tags or any(status(t) == "pending" for t in tags):
        cells_state[cell] = info; return False, None
    best = choose(tags); rnd = 1
    while best is not None and not res(best)["negH_pd"] and rnd <= MAX_ROUNDS:
        rb = res(best); st, e, blocked = escape(rb)
        e.update(round=rnd, parent=best, parent_lnL_heldout=rb["lnL_heldout_mean"], parent_eig0=min(rb["negH_eigs"]))
        info["escapes"].append(e)
        if blocked: e["outcome"] = f"blocked by rail {e['blocked_by_rail']}"; break
        pair = child_tags(cell, rnd, best)
        for t, lab in zip(pair, ("p", "m")):
            if status(t) == "pending": want(t, fit_args(M, fd, mode, st[lab]))
        tags += pair; info["variants"] = list(tags)
        if any(status(t) == "pending" for t in pair):
            cells_state[cell] = info; return False, None
        new = choose(tags)
        e["results"] = {t: (dict(lnL_heldout=res(t)["lnL_heldout_mean"], lnL_map=res(t)["lnL"], pd=res(t)["negH_pd"],
                                 eig0=min(res(t)["negH_eigs"]), rails=res(t)["railing"]) if status(t) == "done" else "failed")
                        for t in pair}
        if new not in pair: e["outcome"] = "no improvement on parent"; break
        newr = set(res(new)["railing"]) - set(rb["railing"])
        e["outcome"] = f"moved to {new}: dlnL_heldout {res(new)['lnL_heldout_mean'] - rb['lnL_heldout_mean']:+.2f}"
        best = new; rnd += 1
        if newr: e["outcome"] += f"; new rail {sorted(newr)}"; break
    info["best"] = best
    if best:
        r = res(best); info.update(pd=r["negH_pd"], eig0=min(r["negH_eigs"]), rails=r["railing"],
                                   lnL_heldout=r["lnL_heldout_mean"], lnZ=r["lnZ_laplace_bdy"])
    cells_state[cell] = info
    return True, best


def profile(M):
    """kappa 2.7 profile with restarts and extension. Returns (settled, {fd: best_tag})."""
    fds = list(GRID); starts = {fd: [(f"{M}_{ftag(fd)}", None)] for fd in fds}
    for _ in range(40):
        bests, settled = {}, True
        for fd in sorted(fds):
            ok, b = run_cell(M, fd, "k27", starts[fd]); bests[fd] = b; settled &= ok
        if not settled: return False, bests
        L = {fd: res(b)["lnL_heldout_mean"] for fd, b in bests.items() if b}
        changed = False
        sf = sorted(fds)
        for i, fd in enumerate(sf):
            nbrs = [sf[j] for j in (i - 1, i + 1) if 0 <= j < len(sf)]
            src = [n for n in nbrs if n in L and fd in L and L[n] - L[fd] > DIP]
            if (M, fd) in EXTRA_RESTART: src.append(EXTRA_RESTART[(M, fd)])
            for n in src:
                tag = f"{M}_{ftag(fd)}_r{fcode(n)}"
                if all(t != tag for t, _ in starts[fd]):
                    starts[fd].append((tag, res(bests[n])["theta_map"])); changed = True
        if changed: continue
        top, bot = max(fds), min(fds); bfd = max(L, key=L.get)
        nx = UP.get(top)
        if nx and (bfd == top or (top == 0.5 and M in FORCED_UP)):
            fds.append(nx); starts[nx] = [(f"{M}_{ftag(nx)}", res(bests[top])["theta_map"])]; continue
        nx = DOWN.get(bot)
        if nx and (bfd == bot or (bot == 0.3 and M in FORCED_DOWN)):
            fds.append(nx); starts[nx] = [(f"{M}_{ftag(nx)}", res(bests[bot])["theta_map"])]; continue
        return True, bests
    raise SystemExit(f"profile loop did not converge for {M}")


def reeval(tag, mode):
    if os.path.exists(f"{R}/{tag}/reeval_n8000.json") or os.path.exists(f"{R}/{tag}/FAILED_reeval"): return True
    want(f"{tag}__reeval", f"--mode reeval --n-seed 8000 --n-b 6000 --keys 11 202 3003 {FL} {mode_flags(mode)}")
    return False


v2 = json.load(open(f"{R}/plan_state.json"))["members"]
members, ready = {}, {}
for M in MODELS:
    ok, bests = profile(M)
    if not ok: continue
    L = {fd: res(b)["lnL_heldout_mean"] for fd, b in bests.items()}
    bfd = max(L, key=L.get); b27 = bests[bfd]
    members[M] = dict(best_fdep=bfd, best_k27=b27, fdeps=sorted(bests), profile={str(k): v for k, v in bests.items()},
                      edge=bool(bfd in (min(bests), max(bests))), k27_reeval=reeval(b27, "k27"))
    ready[M] = (bfd, b27)


def kw_start(M, bfd, b27):
    kf = v2.get(M, {}).get("best_kfree")
    if kf and abs(v2[M]["best_fdep"] - bfd) < 1e-9:
        return res(kf)["theta_map"]
    th = dict(res(b27)["theta_map"]); th["kappa"] = res(kf)["theta_map"]["kappa"] if kf else 2.7
    return th


level_used = None
for H in HI_LEVELS:
    mode = ("kw", H); done_all = True; railed = []
    for M, (bfd, b27) in ready.items():
        ok, b = run_cell(M, bfd, mode, [(f"{M}_{ftag(bfd)}_kw{H:g}", kw_start(M, bfd, b27))])
        if not ok: done_all = False; continue
        r = res(b); members[M][f"best_kw{H:g}"] = b
        for n in r["names"]:
            if n.startswith("log_R_") and r["bounds"][n][1] - r["theta_map"][n] < 0.01 * (r["bounds"][n][1] - r["bounds"][n][0]):
                railed.append(f"{M}:{n}")
    level_used = H
    if not (done_all and len(ready) == len(MODELS)): break
    if not railed: break
    if H == HI_LEVELS[-1]: break
complete = len(ready) == len(MODELS)
for M in ready:
    b = members[M].get(f"best_kw{level_used:g}")
    members[M]["best_kfree"] = b
    members[M]["kfree_level"] = level_used
    if b is None: complete = False; continue
    complete &= reeval(b, ("kw", level_used)) & members[M]["k27_reeval"]

lap = []
if complete:
    for kind, mode in (("best_k27", "k27"), ("best_kfree", ("kw", level_used))):
        top = sorted(members, key=lambda M: -res(members[M][kind])["lnZ_laplace_bdy"])[:3]
        for M in top:
            t = members[M][kind]; lap.append(t)
            if not (os.path.exists(f"{R}/{t}/laplace_n8000.json") or os.path.exists(f"{R}/{t}/FAILED_lap")):
                want(f"{t}__lap", f"--mode laplace {SEEDS} {FL} {mode_flags(mode)}"); complete = False

json.dump(children, open(EC_PATH + ".tmp", "w"), indent=1); os.replace(EC_PATH + ".tmp", EC_PATH)
st = dict(cells=cells_state, members=members, kfree_level=level_used, laplace_check=lap, complete=complete)
json.dump(st, open(f"{R}/plan_state_v3.json.tmp", "w"), indent=1, default=str)
os.replace(f"{R}/plan_state_v3.json.tmp", f"{R}/plan_state_v3.json")
for u in order: print(f"{u} {units[u]}")
if complete: print("#COMPLETE")
