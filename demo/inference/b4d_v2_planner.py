"""Closing pass on the 4D B-series ranking: decide the next units from finished results only.

Prints the wanted units as "UNIT args" lines (deterministic for a given set of finished
results), writes plan_state.json, and prints "#COMPLETE" when nothing is left.

Per member, in order (each step waits for the previous one to finish):
  1. every fit of the f_dep profile at kappa 2.7; a cell whose best fit has a non-positive-definite
     -H gets a saddle escape: restart at theta +/- s v (v = eigenvector of the most negative
     eigenvalue lambda of -H, s = sqrt(2 G / |lambda|) with G = 3 nats predicted gain, clipped to
     5% of each prior width and kept 0.5% inside the bounds); up to 3 rounds, continuing only while
     the escape improves on its parent and nothing new rails;
  2. BD f_dep 0.4 restarts from the BD 0.3 and 0.5 optima;
  3. f_dep 0.7 for BD, BMIX, REFRC and any member whose best (held-out lnL) is at 0.5; 0.9 for any
     member whose best is at 0.7;
  4. free kappa ~ U[-1, 7] at the member's best f_dep, from its kappa 2.7 optimum (with escapes);
  5. n_seed 8000 re-evaluation (keys 11 202 3003) of the kappa 2.7 and free-kappa best points.
Cell best = positive-definite variants first, then highest held-out lnL.
"""
import json, os, sys
import numpy as np

R = os.path.expanduser("~/hier-reassess-20261121/runs/BS4D_scan")
MODELS = ["NULL", "BB", "BC", "BD", "BMIX", "NULLG", "REFRC"]
GRID = [0.3, 0.4, 0.5]
NEXT = {0.5: 0.7, 0.7: 0.9}
FORCED07 = {"BD", "BMIX", "REFRC"}
FL = "--m-max-lo 20 --likelihood native4d --native4d-window --native4d-nodes 32"
KFREE = "--free-kappa -1 7"
G_NATS, BOX, INSET, MAX_ROUNDS = 3.0, 0.05, 0.005, 3

units, order, state = {}, [], {}


def want(unit, args):
    if unit not in units:
        units[unit] = args; order.append(unit)


def res(tag):
    p = f"{R}/{tag}/result.json"
    return json.load(open(p)) if os.path.exists(p) else None


def fit_status(tag):
    if os.path.exists(f"{R}/{tag}/result.json"): return "done"
    if os.path.exists(f"{R}/{tag}/FAILED"): return "failed"
    return "pending"


def ftag(fd): return f"f{fd:.1f}"


def fit_args(M, fd, kf, theta):
    th = {k: float(v) for k, v in theta.items()}
    if not kf: th.pop("kappa", None)
    elif "kappa" not in th: th["kappa"] = 2.7
    return (f"--model {M} --f-dep {fd:.1f} --init-json {json.dumps(th, separators=(',', ':'))} "
            f"--kappa 2.7 {FL}" + (f" {KFREE}" if kf else ""))


def key(tag):
    r = res(tag); return (bool(r["negH_pd"]), r["lnL_heldout_mean"])


def choose(tags):
    ok = [t for t in tags if fit_status(t) == "done"]
    return max(ok, key=key) if ok else None


def escape(r):
    names = r["names"]; th = np.array([r["theta_map"][n] for n in names])
    lo = np.array([r["bounds"][n][0] for n in names]); hi = np.array([r["bounds"][n][1] for n in names])
    w = hi - lo
    ev, V = np.linalg.eigh(-np.array(r["hessian"]))
    lam, v = ev[0], V[:, 0]
    s0 = np.sqrt(2 * G_NATS / abs(lam))
    s = min(s0, float(np.min(BOX * w / np.maximum(np.abs(v), 1e-300))))
    starts, eff = {}, {}
    for sg, lab in ((1, "p"), (-1, "m")):
        t = np.clip(th + sg * s * v, lo + INSET * w, hi - INSET * w)
        starts[lab] = dict(zip(names, t.tolist())); eff[lab] = float(np.dot(t - th, sg * v) / s)
    rails = set(r.get("railing") or [])
    dom = [names[i] for i in np.argsort(-np.abs(v))[:3]]
    info = dict(lambda_min=float(lam), step=float(s), step_unclipped=float(s0),
                predicted_gain=float(0.5 * abs(lam) * s * s), dominant=dom,
                v={n: float(x) for n, x in zip(names, v)}, achieved=eff)
    blocked = max(eff.values()) < 0.1
    if blocked:
        rr = [n for n in dom if n in rails] or [dom[0]]
        info["blocked_by_rail"] = rr[0]
    return starts, info, blocked


def run_cell(M, fd, kf, starts):
    """starts: list of (tag, theta). Returns (settled, best_tag, info)."""
    cell = f"{M}_{ftag(fd)}" + ("_kf" if kf else "")
    info = dict(cell=cell, member=M, f_dep=fd, kappa="free" if kf else 2.7, variants=[], escapes=[])
    tags = []
    for tag, th in starts:
        if fit_status(tag) == "pending" and th is not None:
            want(tag, fit_args(M, fd, kf, th))
        tags.append(tag)
    info["variants"] = list(tags)
    if any(fit_status(t) == "pending" for t in tags):
        state[cell] = info; return False, None, info
    best = choose(tags); info["start_best"] = best
    rnd = 1
    while best is not None and not res(best)["negH_pd"] and rnd <= MAX_ROUNDS:
        rb = res(best)
        st, einfo, blocked = escape(rb)
        einfo.update(round=rnd, parent=best, parent_lnL_heldout=rb["lnL_heldout_mean"],
                     parent_eig0=min(rb["negH_eigs"]))
        info["escapes"].append(einfo)
        if blocked:
            einfo["outcome"] = f"blocked: step points into rail {einfo['blocked_by_rail']}"; break
        pair = [f"{cell}_e{rnd}{lab}" for lab in ("p", "m")]
        for t, lab in zip(pair, ("p", "m")):
            if fit_status(t) == "pending": want(t, fit_args(M, fd, kf, st[lab]))
        tags += pair; info["variants"] = list(tags)
        if any(fit_status(t) == "pending" for t in pair):
            state[cell] = info; return False, None, info
        new = choose(tags)
        einfo["results"] = {t: (dict(lnL_heldout=res(t)["lnL_heldout_mean"], pd=res(t)["negH_pd"],
                                     eig0=min(res(t)["negH_eigs"]), rails=res(t)["railing"])
                                if fit_status(t) == "done" else "failed") for t in pair}
        if new not in pair:
            einfo["outcome"] = "no improvement on parent"; break
        newr = set(res(new)["railing"]) - set(rb["railing"])
        einfo["outcome"] = f"moved to {new}: dlnL_heldout {res(new)['lnL_heldout_mean'] - rb['lnL_heldout_mean']:+.2f}"
        best = new; rnd += 1
        if newr:
            einfo["outcome"] += f"; new rail {sorted(newr)}"; break
    info["best"] = best
    if best is not None:
        rb = res(best)
        info.update(pd=rb["negH_pd"], eig0=min(rb["negH_eigs"]), rails=rb["railing"],
                    lnL_heldout=rb["lnL_heldout_mean"], lnZ=rb["lnZ_laplace_bdy"])
    state[cell] = info
    return True, best, info


def reeval(tag, kf):
    if os.path.exists(f"{R}/{tag}/reeval_n8000.json") or os.path.exists(f"{R}/{tag}/FAILED_reeval"):
        return True
    want(f"{tag}__reeval", f"--mode reeval --n-seed 8000 --n-b 6000 --keys 11 202 3003 --kappa 2.7 {FL}"
         + (f" {KFREE}" if kf else ""))
    return False


complete = True
summary = {}
for M in MODELS:
    # explicit sequencing (clearer than a generic loop)
    best_src = {}
    cells = {}
    def do(fd, starts):
        ok, b, _ = run_cell(M, fd, False, starts); cells[fd] = (ok, b); return ok
    for fd in GRID:
        if fd == 0.4 and M == "BD":
            continue
        do(fd, [(f"{M}_{ftag(fd)}", None)])
    if M == "BD":
        s = [("BD_f0.4", None)]
        if cells[0.3][0] and cells[0.5][0]:
            s += [("BD_f0.4_r03", res(cells[0.3][1])["theta_map"]), ("BD_f0.4_r05", res(cells[0.5][1])["theta_map"])]
            do(0.4, s)
        else:
            cells[0.4] = (False, None)
    while all(ok for ok, _ in cells.values()):
        top = max(cells); bfd = max(cells, key=lambda f: res(cells[f][1])["lnL_heldout_mean"] if cells[f][1] else -1e99)
        nxt = NEXT.get(top)
        if nxt is None or not (bfd == top or (top == 0.5 and M in FORCED07)):
            break
        do(nxt, [(f"{M}_{ftag(nxt)}", res(cells[top][1])["theta_map"])])
    m_done = False
    if all(ok for ok, _ in cells.values()):
        bfd = max(cells, key=lambda f: res(cells[f][1])["lnL_heldout_mean"] if cells[f][1] else -1e99)
        b27 = cells[bfd][1]
        r1 = reeval(b27, False)
        okf, bkf, _ = run_cell(M, bfd, True, [(f"{M}_{ftag(bfd)}_kf", res(b27)["theta_map"])])
        r2 = reeval(bkf, True) if okf and bkf else False
        m_done = r1 and okf and r2
        summary[M] = dict(best_fdep=bfd, best_k27=b27, best_kfree=bkf if okf else None,
                          fdeps=sorted(cells), edge=bool(bfd in (min(cells), max(cells))))
    complete &= m_done

tmp = f"{R}/plan_state.json.tmp"
json.dump(dict(cells=state, members=summary, complete=complete), open(tmp, "w"), indent=1, default=str)
os.replace(tmp, f"{R}/plan_state.json")
for u in order:
    print(f"{u} {units[u]}")
if complete:
    print("#COMPLETE")
