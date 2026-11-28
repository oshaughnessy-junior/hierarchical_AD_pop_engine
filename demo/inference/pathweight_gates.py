"""Gates for the path-weighted cascade (runs/BS_weights, 2026-11-28).

One process = one MAP JSON and one cascade key.

  g0     at the given points: the committed exact cascade (proposal recentred at the
         point) against the path-weighted cascade with proposal = target ("exact").
  g1     at the bias_laplace points: path-weighted lnL on ONE fixed proposal
         (--path-proposal, built from the MAP's final_proposal), plus merger-weight
         ESS. The exact (recentred) lnL comes from --exact-summary.
  spinfreeze  mass proposal = target at each point, shadow spin frozen at the MAP's
         (isolates the spin approximation); compare with the exact summary.
  line   lnL and its gradient along a line in one parameter (others at the MAP),
         --n-line points across +-2 sigma (Laplace), plus central finite differences.
  time   value_and_grad timing at the MAP for the committed pinned cascade and the
         path-weighted one.
"""
import argparse, json, os, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import hier_map_laplace as hml
import hier_nuts as hn
import bias_laplace as bl


def kish(w):
    w = np.asarray(w, float)
    w = w[np.isfinite(w) & (w > 0)]
    return float(w.sum() ** 2 / (w ** 2).sum()) if w.size else 0.0


def channelA_weights(B, th):
    """channel-A merger weights (linear) at theta."""
    if "_wA" not in B:
        names, consts, nA, comb = B["names"], B["consts"], B["nA"], B["comb"]
        B["_wA"] = jax.jit(lambda t: comb(**{k: t[i] for i, k in enumerate(names)}, **consts)[1][:nA])
    return np.exp(np.asarray(B["_wA"](jnp.asarray(th))))


def setup_exact(a, r, th, names):
    prop = hml.prop_ref_from(r["model"], dict(zip(names, th)))
    a2 = argparse.Namespace(**vars(a)); a2.path_proposal = "none"
    return hn.setup(a2, r, key=a.key, prop=prop), prop


def setup_path(a, r, kind, ref=None):
    a2 = argparse.Namespace(**vars(a)); a2.path_proposal = kind
    if kind == "exact" and ref is not None:     # G0 / spinfreeze: spins reparameterised
        a2.path_spin = "reparam"
    if ref is not None:          # proposal reference other than the MAP's final_proposal
        r = dict(r); r["final_proposal"] = ref
    return hn.setup(a2, r, key=a.key)


def run_points(a, r, pts, names):
    out = []
    if a.mode == "g0":
        for pt in pts:
            th = pt["theta"]
            Be, prop = setup_exact(a, r, th, names)
            Le, (mue, _) = jax.jit(Be["parts"])(jnp.asarray(th))
            Bw = setup_path(a, r, "exact", ref=prop)
            Lw, (muw, _) = jax.jit(Bw["parts"])(jnp.asarray(th))
            we = channelA_weights(Be, th); ww = channelA_weights(Bw, th)
            row = dict(label=pt["label"], mahal=pt["mahal"], theta=dict(zip(names, th.tolist())),
                       lnL_exact=float(Le), lnL_path=float(Lw), diff=float(Lw - Le),
                       mu_exact=float(mue), mu_path=float(muw),
                       w_maxabs_reldiff=float(np.max(np.abs(ww - we)) / np.max(we)),
                       ess_exact=kish(we), ess_path=kish(ww))
            print(f"g0 {pt['label']:7s} exact {float(Le):.6f} path {float(Lw):.6f} diff {float(Lw-Le):+.3e} "
                  f"w reldiff {row['w_maxabs_reldiff']:.1e}", flush=True)
            out.append(row)
    elif a.mode == "spinfreeze":
        # mass proposal = target at the point; shadow spin scale frozen at the MAP's.
        # Isolates the reparameterised-spin approximation (spin enters the pair
        # sequence only through remnant masses and kicks).
        s0 = float(r["final_proposal"]["spin"])
        for pt in pts:
            th = pt["theta"]
            prop = hml.prop_ref_from(r["model"], dict(zip(names, th)))
            ref = dict(prop); ref["spin"] = s0
            Bw = setup_path(a, r, "exact", ref=ref)
            Lw, _ = jax.jit(Bw["parts"])(jnp.asarray(th))
            row = dict(label=pt["label"], mahal=pt["mahal"], theta=dict(zip(names, th.tolist())),
                       spin0=s0, lnL_path=float(Lw))
            print(f"spinfreeze {pt['label']:7s} spin {th[names.index('spin_chi')]:.4f} (s0 {s0:.4f}) "
                  f"path {float(Lw):.3f}", flush=True)
            out.append(row)
    elif a.mode == "g1":
        t0 = time.time()
        Bw = setup_path(a, r, a.path_proposal)
        f = jax.jit(Bw["parts"])
        sh = Bw["pl"].PL_PATHWEIGHT
        print(f"path proposal {sh} built in {time.time()-t0:.0f}s", flush=True)
        for pt in pts:
            th = pt["theta"]
            Lw, (muw, _) = f(jnp.asarray(th))
            ww = channelA_weights(Bw, th)
            row = dict(label=pt["label"], mahal=pt["mahal"], theta=dict(zip(names, th.tolist())),
                       lnL_path=float(Lw), mu_path=float(muw), ess_path=kish(ww), n_rec=int(ww.size))
            if a.with_exact:
                Be, _ = setup_exact(a, r, th, names)
                Le, _ = jax.jit(Be["parts"])(jnp.asarray(th))
                row["lnL_exact"] = float(Le); row["ess_exact"] = kish(channelA_weights(Be, th))
            print(f"g1 {pt['label']:7s} d={pt['mahal']:.2f} path {float(Lw):.3f} ess {row['ess_path']:.0f}"
                  + (f" exact {row['lnL_exact']:.3f} ess {row['ess_exact']:.0f}" if a.with_exact else "")
                  + f" ({time.time()-t0:.0f}s)", flush=True)
            out.append(row)
    return out


def run_line(a, r, names):
    th0 = np.array([r["theta_map"][k] for k in names])
    negH = -np.array(r["hessian"])
    S = np.linalg.inv(negH)
    s = names.index(a.param)
    lo = np.array([r["bounds"][k][0] for k in names]); hi = np.array([r["bounds"][k][1] for k in names])
    w = hi - lo
    sd = np.sqrt(S[s, s])
    t_lo = -2.0 if a.t_lo is None else a.t_lo
    t_hi = 2.0 if a.t_hi is None else a.t_hi
    if a.t_lo is None and th0[s] - 2 * sd < lo[s]:   # MAP on the lower rail: one-sided line
        t_lo, t_hi = 0.0, 4.0
    ts = np.linspace(t_lo, t_hi, a.n_line)
    B = setup_path(a, r, a.path_proposal) if a.path_proposal != "none" else hn.setup(a, r, key=a.key)
    vg = jax.jit(jax.value_and_grad(lambda t: B["parts"](t)[0]))
    f = jax.jit(lambda t: B["parts"](t)[0])
    rows = []
    for t in ts:
        th = th0.copy(); th[s] = np.clip(th0[s] + t * sd, lo[s] + 1e-6 * w[s], hi[s] - 1e-6 * w[s])
        L, g = vg(jnp.asarray(th))
        h = a.fd_rel * sd
        tp = th.copy(); tp[s] += h; tm = th.copy(); tm[s] -= h
        fd = (float(f(jnp.asarray(tp))) - float(f(jnp.asarray(tm)))) / (2 * h)
        rows.append(dict(t=float(t), x=float(th[s]), lnL=float(L), grad=float(np.asarray(g)[s]), fd=fd))
        print(f"line {a.param} t={t:+.2f} x={th[s]:.5f} lnL {float(L):.4f} grad {float(np.asarray(g)[s]):+.4e} "
              f"fd {fd:+.4e}", flush=True)
    return dict(param=a.param, sd=float(sd), fd_h=float(a.fd_rel * sd), rows=rows)


def run_time(a, r, names):
    th = jnp.asarray(np.array([r["theta_map"][k] for k in names]))
    res = {}
    for kind in ("none", a.path_proposal):
        t0 = time.time()
        a2 = argparse.Namespace(**vars(a)); a2.path_proposal = kind
        B = hn.setup(a2, r, key=a.key)
        tb = time.time() - t0
        vg = jax.jit(jax.value_and_grad(lambda t: B["parts"](t)[0]))
        t1 = time.time(); o = vg(th); jax.block_until_ready(o); tc = time.time() - t1
        tt = []
        for _ in range(a.n_time):
            t1 = time.time(); o = vg(th); jax.block_until_ready(o); tt.append(time.time() - t1)
        res[kind] = dict(build_s=tb, first_s=tc, vg_s=float(np.median(tt)), vg_all=tt, lnL=float(o[0]))
        print(f"time {kind}: build {tb:.1f}s first {tc:.1f}s vg median {np.median(tt):.2f}s lnL {float(o[0]):.4f}",
              flush=True)
    res["threads"] = os.environ.get("OMP_NUM_THREADS"); res["host"] = os.uname().nodename
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["g0", "g1", "spinfreeze", "line", "time"])
    p.add_argument("--map-json", required=True)
    p.add_argument("--key", type=int, required=True)
    p.add_argument("--n-seed", type=int, default=8000)
    p.add_argument("--n-b", type=int, default=6000)
    p.add_argument("--labels", nargs="*", default=None, help="subset of bias_laplace point labels")
    p.add_argument("--n-rand", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--path-proposal", default="defensive")
    p.add_argument("--path-budget-scale", type=float, default=1.5)
    p.add_argument("--path-steps-scale", type=float, default=1.0)
    p.add_argument("--path-spin", default="reparam")
    p.add_argument("--with-exact", type=int, default=0)
    p.add_argument("--param", default="m_max_A")
    p.add_argument("--n-line", type=int, default=41)
    p.add_argument("--fd-rel", type=float, default=1e-3)
    p.add_argument("--t-lo", type=float, default=None)
    p.add_argument("--t-hi", type=float, default=None)
    p.add_argument("--n-time", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument("--catalog", default=os.path.join(hml.WS, "inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz"))
    p.add_argument("--pdet-grid", default=os.path.join(hml.WS, "inputs/mass_pdet_grid_kappa2.7.npz"))
    p.add_argument("--likelihood", default="nal3d")
    a = p.parse_args()
    a.map_json = os.path.abspath(a.map_json)
    r = hn.load_map(a.map_json)
    tag = os.path.basename(os.path.dirname(a.map_json))
    names = sorted(r["names"])
    assert names == r["names"]
    res = dict(mode=a.mode, map_json=a.map_json, tag=tag, key=a.key, n_seed=a.n_seed, n_b=a.n_b,
               path_proposal=a.path_proposal, path_budget_scale=a.path_budget_scale,
               path_steps_scale=a.path_steps_scale, path_spin=a.path_spin,
               host=os.uname().nodename, threads=os.environ.get("OMP_NUM_THREADS"))
    t0 = time.time()
    if a.mode in ("g0", "g1", "spinfreeze"):
        _, pts = bl.points(r, a.n_rand, a.seed)
        if a.labels:
            pts = [x for x in pts if x["label"] in a.labels]
        res["rows"] = run_points(a, r, pts, names)
    elif a.mode == "line":
        res.update(run_line(a, r, names))
    else:
        res.update(run_time(a, r, names))
    res["wall_s"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    raise SystemExit(main())
