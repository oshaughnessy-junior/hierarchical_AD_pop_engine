"""O4-only (168 events) B-series refit with injection selection and free kappa: next units from finished results.

Prints wanted units as "UNIT args", writes plan_state.json and escape_children.json, "#COMPLETE" at the end.
Every fit: native4d (no Mc windows), --event-subset o4, --selection injections, kappa ~ U[-1, 7], every log_R_*
on [logr0-8, logr0+12]. Per member: f_dep at its v3 best and the two ladder neighbours (ladder 0.1 0.2 0.3 0.4
0.5 0.7 0.9), each started from the v3 free-kappa optimum with log_R shifted by ln(168/236) + ln(T_v3/T_O4);
extension one ladder step past an edge that holds the best held-out lnL; restart from a neighbour more than 5
nats better; saddle escape (v2 rule) where -H is not positive definite. Then n_seed 8000 rescoring of each
member's best point. Cell best = positive-definite first, then highest held-out lnL.
"""
import json, math, os
import numpy as np

R = os.path.expanduser("~/hier-reassess-20261121/runs/BS4D_O4/scan")
V3 = os.path.expanduser("~/hier-reassess-20261121/runs/BS4D_scan")
LADDER = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
T_V3, T_O4 = 3.0456693791669833, 52962816.0 / (365.25 * 86400.0)
SHIFT = math.log(168 / 236.0) + math.log(T_V3 / T_O4)
MODELS = ["NULL", "BB", "BC", "BD", "BMIX", "NULLG", "REFRC"]
DIP = 5.0
FL = "--m-max-lo 20 --likelihood native4d --native4d-nodes 32 --selection injections --event-subset o4"
G_NATS, BOX, INSET, MAX_ROUNDS = 3.0, 0.05, 0.005, 3
SEEDS = "--key 11 --n-seed 8000 --n-b 6000"

units, order, cells_state = {}, [], {}
EC_PATH = f"{R}/escape_children.json"
children = json.load(open(EC_PATH)) if os.path.exists(EC_PATH) else {}


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
    return "--kappa 2.7 --free-kappa -1 7 --log-r-range -8 12"


def fit_args(M, fd, mode, theta):
    th = {k: float(v) for k, v in theta.items()}
    th.setdefault("kappa", 2.7)
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
    cell = f"{M}_{ftag(fd)}"
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


def v3_start(M):
    v3 = json.load(open(f"{V3}/plan_state_v3.json"))["members"][M]
    th = dict(json.load(open(f"{V3}/{v3['best_kfree']}/result.json"))["theta_map"])
    for k in th:
        if k.startswith("log_R_"): th[k] += SHIFT
    return v3["best_fdep"], th


def profile(M):
    f0, th0 = v3_start(M); i0 = LADDER.index(f0)
    fds = [LADDER[j] for j in (i0 - 1, i0, i0 + 1) if 0 <= j < len(LADDER)]
    starts = {fd: [(f"{M}_{ftag(fd)}", th0)] for fd in fds}
    for _ in range(40):
        bests, settled = {}, True
        for fd in sorted(fds):
            ok, b = run_cell(M, fd, "o4", starts[fd]); bests[fd] = b; settled &= ok
        if not settled: return False, bests
        L = {fd: res(b)["lnL_heldout_mean"] for fd, b in bests.items() if b}
        changed = False; sf = sorted(fds)
        for i, fd in enumerate(sf):
            for n in [sf[j] for j in (i - 1, i + 1) if 0 <= j < len(sf)]:
                if n in L and fd in L and L[n] - L[fd] > DIP:
                    tag = f"{M}_{ftag(fd)}_r{fcode(n)}"
                    if all(t != tag for t, _ in starts[fd]):
                        starts[fd].append((tag, res(bests[n])["theta_map"])); changed = True
        if changed: continue
        bfd = max(L, key=L.get); top, bot = max(fds), min(fds)
        for edge, step in ((top, 1), (bot, -1)):
            j = LADDER.index(edge) + step
            if bfd == edge and 0 <= j < len(LADDER) and LADDER[j] not in fds:
                nx = LADDER[j]; fds.append(nx); starts[nx] = [(f"{M}_{ftag(nx)}", res(bests[edge])["theta_map"])]; changed = True; break
        if changed: continue
        return True, bests
    raise SystemExit(f"profile loop did not converge for {M}")


def reeval(tag):
    if os.path.exists(f"{R}/{tag}/reeval_n8000.json") or os.path.exists(f"{R}/{tag}/FAILED_reeval"): return True
    want(f"{tag}__reeval", f"--mode reeval --n-seed 8000 --n-b 6000 --keys 11 202 3003 {FL} {mode_flags('o4')}")
    return False


members, complete = {}, True
for M in MODELS:
    ok, bests = profile(M)
    if not ok: complete = False; continue
    L = {fd: res(b)["lnL_heldout_mean"] for fd, b in bests.items()}
    bfd = max(L, key=L.get); b = bests[bfd]
    members[M] = dict(best_fdep=bfd, best=b, fdeps=sorted(bests), profile={str(k): v for k, v in bests.items()},
                      edge=bool(bfd in (min(bests), max(bests))))
    complete &= reeval(b)

json.dump(children, open(EC_PATH + ".tmp", "w"), indent=1); os.replace(EC_PATH + ".tmp", EC_PATH)
st = dict(cells=cells_state, members=members, complete=complete, log_R_shift=SHIFT)
json.dump(st, open(f"{R}/plan_state.json.tmp", "w"), indent=1, default=str)
os.replace(f"{R}/plan_state.json.tmp", f"{R}/plan_state.json")
for u in order: print(f"{u} {units[u]}")
if complete: print("#COMPLETE")
