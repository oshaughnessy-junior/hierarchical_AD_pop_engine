"""NUTS posteriors with the exact cascade, pinned at a MAP proposal (2026-11-27).

Sibling of hier_map_laplace.py; reuses its build(). One process = one chain.

  * proposal: pinned at the MAP JSON's final_proposal (every cascade proposal
    recentred to equal its target there), sampler key --key, n_seed/n_b as given.
  * lnL: build()["parts"], the full Poisson lnL  -mu + sum_i log sum_j w_j L_i,
    the same function the fit's L-BFGS-B polish, final lnL and Laplace use.
  * rates: log_R_* are SAMPLED (not profiled), uniform on the fit's bounds,
    like every other parameter. Priors: Uniform on result.json["bounds"].
  * init: the MAP theta, moved at most --edge-frac x width inside a prior edge
    (several MAPs sit on m_max_A / sig_B rails, where the logit map is singular).
  * sampler: gwkokab.inference.run_nuts_with_checkpoints (warmup once, then
    chunks; resumes from <outdir>/nuts_ckpt.pkl). Its MCMC class is wrapped so
    that each chunk also stores diverging / num_steps / accept_prob /
    potential_energy (keys prefixed "_").

Modes
  parity  lnL + value_and_grad timing at the exact MAP theta; compares with the
          MAP JSON's lnL (same key, n_seed, n_b) or with reeval_n<n_seed>.json.
  nuts    one chain -> <outdir>/nuts_ckpt.pkl, <outdir>/samples.npz at the end.
  status  one-line health summary of every chain dir given (--ckpt-dirs).
  bias    lnL at --n-draws posterior draws with the sampled cascade (pinned at the MAP, or
          path-weighted with --path-proposal; "lnL_pinned" in the output) versus recentred
          at the draw (exact there) -> <outdir>/bias_check.json.

--path-proposal P (2026-11-28): path-weighted cascade on a fixed proposal built from the
MAP (runs/BS_weights). --m-max-cap X: m_max_A prior U[lo, X], recorded in meta.json.
"""
import argparse, glob, json, os, pickle, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import hier_map_laplace as hml

EXTRA = ("diverging", "num_steps", "accept_prob", "potential_energy")


def load_map(path):
    r = json.load(open(path))
    if not r.get("recentred") or r.get("final_proposal") is None:
        raise SystemExit(f"{path}: no recentred final_proposal; cannot pin the exact cascade")
    return r


def path_spec(args, r):
    """--path-proposal: None (pinned exact cascade) or the path-weighted spec."""
    kind = getattr(args, "path_proposal", "none")
    if kind == "none":
        return None
    return hml.pathw_spec(r["model"], r["final_proposal"], kind,
                          getattr(args, "path_budget_scale", 1.5), getattr(args, "path_steps_scale", 1.0),
                          getattr(args, "path_spin", "reparam"))


def setup(args, r, key=None, prop=None):
    """build() for the MAP's model / f_dep / m_max_A floor; check bounds agree.

    prop given: recentred exact cascade at prop. Otherwise --path-proposal none pins
    the proposal at the MAP; any other value uses the path-weighted cascade with a
    fixed proposal built from the MAP's final_proposal (runs/BS_weights)."""
    hml.M_MAX_LO = float(r["bounds"]["m_max_A"][0])
    cap = getattr(args, "m_max_cap", None)
    if cap is not None:              # narrower m_max_A prior U[lo, cap] (path "exact" coverage)
        if not hml.M_MAX_LO < cap <= 70.0:
            raise SystemExit(f"--m-max-cap {cap} outside ({hml.M_MAX_LO}, 70]")
        hml.M_MAX_HI = float(cap)
        r["bounds"]["m_max_A"] = [hml.M_MAX_LO, float(cap)]
    args.f_dep = float(r["f_dep"])
    pw = path_spec(args, r) if prop is None else None
    B = hml.build(args, r["model"], args.key if key is None else key, args.n_seed, args.n_b,
                  None if pw is not None else (r["final_proposal"] if prop is None else prop),
                  pathw=pw)
    B["pathw"] = pw
    if B["names"] != r["names"]:
        raise SystemExit(f"parameter names differ from the MAP JSON: {B['names']} vs {r['names']}")
    for k in B["names"]:
        if not np.allclose(B["bounds"][k], r["bounds"][k]):
            raise SystemExit(f"bounds differ for {k}: {B['bounds'][k]} vs {r['bounds'][k]}")
    return B


def map_theta(r, names):
    return np.array([r["theta_map"][k] for k in names], float)


def ref_lnL(args, r):
    """the MAP JSON's own lnL at this (key, n_seed, n_b), if one exists."""
    if (r["key"], r["n_seed"], r["n_b"]) == (args.key, args.n_seed, args.n_b):
        return float(r["lnL"]), "result.json lnL"
    rp = os.path.join(os.path.dirname(os.path.abspath(args.map_json)), f"reeval_n{args.n_seed}.json")
    if os.path.exists(rp):
        e = json.load(open(rp))
        if e["n_b"] == args.n_b:
            for ev in e["evals"]:
                if ev["key"] == args.key:
                    return float(ev["lnL"]), os.path.basename(rp)
    return None, None


def run_parity(args):
    r = load_map(args.map_json)
    t0 = time.time()
    B = setup(args, r)
    tb = time.time() - t0
    th = jnp.asarray(map_theta(r, B["names"]))
    vg = jax.jit(jax.value_and_grad(B["parts"], has_aux=True))
    t1 = time.time(); out = vg(th); jax.block_until_ready(out); t2 = time.time()
    ts = []
    for _ in range(args.n_time):
        a = time.time(); out = vg(th); jax.block_until_ready(out); ts.append(time.time() - a)
    (L, (mu, muA)), g = out
    L0, src = ref_lnL(args, r)
    res = dict(map_json=os.path.abspath(args.map_json), model=r["model"], f_dep=r["f_dep"],
               m_max_lo=hml.M_MAX_LO, key=args.key, n_seed=args.n_seed, n_b=args.n_b,
               final_proposal=r["final_proposal"], lnL=float(L), mu=float(mu),
               ref_lnL=L0, ref_src=src, diff=None if L0 is None else float(L) - L0,
               build_s=tb, first_vg_s=t2 - t1, vg_s=float(np.median(ts)), vg_s_all=ts,
               finite_grad=bool(np.all(np.isfinite(np.asarray(g)))),
               threads=os.environ.get("OMP_NUM_THREADS"), host=os.uname().nodename)
    print(json.dumps(res, indent=1), flush=True)
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, f"parity_k{args.key}_n{args.n_seed}.json"), "w") as f:
            json.dump(res, f, indent=1)


def numpyro_model(B):
    import numpyro, numpyro.distributions as dist
    names, bounds, parts = B["names"], B["bounds"], B["parts"]

    def model():
        th = jnp.stack([numpyro.sample(k, dist.Uniform(*bounds[k])) for k in names])
        L, (mu, muA) = parts(th)
        numpyro.deterministic("lnL", L)
        numpyro.deterministic("mu", mu)
        numpyro.factor("loglike", L)
    return model


def patch_extra_fields():
    """make run_nuts_with_checkpoints keep per-sample diagnostics in its checkpoint."""
    import gwkokab.inference.nuts_checkpoint as nc
    base = nc.MCMC

    class MCMCx(base):
        def run(self, *a, **kw):
            kw.setdefault("extra_fields", EXTRA)
            return super().run(*a, **kw)

        def get_samples(self, group_by_chain=False):
            s = dict(super().get_samples(group_by_chain))
            for k, v in self.get_extra_fields(group_by_chain).items():
                s["_" + k] = v
            return s
    nc.MCMC = MCMCx
    return nc


def run_nuts(args):
    r = load_map(args.map_json)
    os.makedirs(args.outdir, exist_ok=True)
    meta = os.path.join(args.outdir, "meta.json")
    B = setup(args, r)
    names, bounds = B["names"], B["bounds"]
    th = map_theta(r, names)
    lo = np.array([bounds[k][0] for k in names]); hi = np.array([bounds[k][1] for k in names])
    w = hi - lo
    th0 = np.clip(th, lo + args.edge_frac * w, hi - args.edge_frac * w)
    L_map = float(jax.jit(B["parts"])(jnp.asarray(th))[0])
    L_init = float(jax.jit(B["parts"])(jnp.asarray(th0))[0])
    L0, src = ref_lnL(args, r)
    info = dict(map_json=os.path.abspath(args.map_json), model=r["model"], f_dep=r["f_dep"],
                m_max_lo=hml.M_MAX_LO, key=args.key, n_seed=args.n_seed, n_b=args.n_b,
                chain_seed=args.chain_seed, final_proposal=r["final_proposal"], names=names,
                cascade_pathw=B.get("pathw"), m_max_cap=getattr(args, "m_max_cap", None),
                bounds={k: list(bounds[k]) for k in names},
                theta_map=dict(zip(names, th.tolist())), theta_init=dict(zip(names, th0.tolist())),
                lnL_map=L_map, lnL_init=L_init, ref_lnL=L0, ref_src=src,
                num_warmup=args.num_warmup, num_samples=args.num_samples, chunk=args.chunk,
                max_tree_depth=args.max_tree_depth, target_accept=args.target_accept,
                dense_mass=args.dense_mass, host=os.uname().nodename,
                slurm_job=os.environ.get("SLURM_JOB_ID"), pid=os.getpid(),
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    if not os.path.exists(meta):
        with open(meta, "w") as f:
            json.dump(info, f, indent=1)
    print(f"[{r['model']} f_dep={r['f_dep']} floor={hml.M_MAX_LO} key={args.key} n_seed={args.n_seed} "
          f"n_b={args.n_b} chain_seed={args.chain_seed}] proposal={r['final_proposal']}", flush=True)
    print(f"lnL at MAP {L_map:.3f} (reference {L0} from {src}); at init {L_init:.3f}", flush=True)
    from numpyro.infer import init_to_value
    nc = patch_extra_fields()
    kw = dict(init_strategy=init_to_value(values=dict(zip(names, th0.tolist()))),
              max_tree_depth=args.max_tree_depth, target_accept_prob=args.target_accept,
              dense_mass=args.dense_mass)
    t0 = time.time()
    post = nc.run_nuts_with_checkpoints(numpyro_model(B), (), num_warmup=args.num_warmup,
                                        num_samples=args.num_samples,
                                        key=jax.random.PRNGKey(args.chain_seed),
                                        ckpt_dir=args.outdir, chunk=args.chunk, nuts_kwargs=kw,
                                        progress=args.progress)
    np.savez(os.path.join(args.outdir, "samples.npz"), **post)
    print(f"DONE {len(post[names[0]])} samples in {(time.time()-t0)/3600:.2f} h", flush=True)


def chain_status(d):
    p = os.path.join(d, "nuts_ckpt.pkl")
    if not os.path.exists(p):
        return dict(dir=d, phase="warmup (no checkpoint yet)")
    with open(p, "rb") as f:
        c = pickle.load(f)
    st, s = c["state"], c["samples"]
    out = dict(dir=d, phase=c["phase"], n_done=int(c["n_done"]),
               step_size=float(st.adapt_state.step_size),
               last_num_steps=int(st.num_steps), mtime=time.ctime(os.path.getmtime(p)))
    if s:
        L = np.asarray(s["lnL"])
        out.update(lnL_mean=float(np.mean(L)), lnL_sd=float(np.std(L)),
                   n_nonfinite=int(sum(np.sum(~np.isfinite(np.asarray(v, float))) for k, v in s.items())),
                   mean_accept=float(np.mean(s["_accept_prob"])) if "_accept_prob" in s else None,
                   n_div=int(np.sum(s["_diverging"])) if "_diverging" in s else None,
                   mean_steps=float(np.mean(s["_num_steps"])) if "_num_steps" in s else None)
    return out


def run_status(args):
    for pat in args.ckpt_dirs:
        for d in sorted(glob.glob(pat)):
            print(json.dumps(chain_status(d)), flush=True)


def run_bias(args):
    """pinned-proposal lnL vs exact (recentred-at-draw) lnL across the posterior."""
    r = load_map(args.map_json)
    acc = {}
    for pat in args.ckpt_dirs:
        for d in sorted(glob.glob(pat)):
            p = os.path.join(d, "nuts_ckpt.pkl")
            if not os.path.exists(p):
                continue
            s = pickle.load(open(p, "rb"))["samples"]
            if s:
                for k, v in s.items():
                    acc.setdefault(k, []).append(np.asarray(v))
    if not acc:
        raise SystemExit("no posterior samples found")
    post = {k: np.concatenate(v) for k, v in acc.items()}
    Bp = setup(args, r)
    names = Bp["names"]
    fp = jax.jit(Bp["parts"])
    n = len(post[names[0]])
    rng = np.random.default_rng(args.bias_seed)
    idx = np.sort(rng.choice(n, size=min(args.n_draws, n), replace=False))
    rows = []
    for i in idx:
        th = np.array([post[k][i] for k in names], float)
        Lp = float(fp(jnp.asarray(th))[0])
        prop = hml.prop_ref_from(r["model"], dict(zip(names, th)))
        Be = setup(args, r, prop=prop)
        Le = float(jax.jit(Be["parts"])(jnp.asarray(th))[0])
        rows.append(dict(index=int(i), theta=dict(zip(names, th.tolist())), proposal=prop,
                         lnL_pinned=Lp, lnL_exact=Le, diff=Lp - Le))
        print(f"draw {i}: pinned {Lp:.3f} exact {Le:.3f} diff {Lp-Le:+.3f}", flush=True)
    d = np.array([x["diff"] for x in rows])
    out = dict(map_json=os.path.abspath(args.map_json), model=r["model"], f_dep=r["f_dep"],
               key=args.key, n_seed=args.n_seed, n_b=args.n_b, n_posterior=n,
               pinned_proposal=r["final_proposal"], draws=rows,
               sampled_cascade="path" if Bp.get("pathw") else "pinned", cascade_pathw=Bp.get("pathw"),
               m_max_cap=getattr(args, "m_max_cap", None),
               diff_mean=float(d.mean()), diff_sd=float(d.std(ddof=1)) if len(d) > 1 else None,
               diff_maxabs=float(np.abs(d).max()))
    print(f"pinned - exact: mean {out['diff_mean']:+.3f} sd {out['diff_sd']} max|.| {out['diff_maxabs']:.3f}")
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "bias_check.json"), "w") as f:
        json.dump(out, f, indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["parity", "nuts", "status", "bias"])
    p.add_argument("--map-json", help="MAP result.json with final_proposal")
    p.add_argument("--outdir")
    p.add_argument("--key", type=int, default=11, help="cascade sampler key (fixed across chains)")
    p.add_argument("--n-seed", type=int, default=8000)
    p.add_argument("--n-b", type=int, default=6000)
    p.add_argument("--chain-seed", type=int, default=0, help="NUTS PRNG key")
    p.add_argument("--num-warmup", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--max-tree-depth", type=int, default=6)
    p.add_argument("--target-accept", type=float, default=0.8)
    p.add_argument("--dense-mass", type=int, default=1)
    p.add_argument("--edge-frac", type=float, default=1e-4)
    p.add_argument("--progress", type=int, default=1)
    p.add_argument("--n-time", type=int, default=3)
    p.add_argument("--ckpt-dirs", nargs="+", default=[])
    p.add_argument("--n-draws", type=int, default=20)
    p.add_argument("--bias-seed", type=int, default=0)
    p.add_argument("--catalog", default=os.path.join(hml.WS, "inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz"))
    p.add_argument("--pdet-grid", default=os.path.join(hml.WS, "inputs/mass_pdet_grid_kappa2.7.npz"))
    p.add_argument("--likelihood", default="nal3d", choices=["nal3d"])
    p.add_argument("--path-proposal", default="none", choices=["none", "tail", "mild", "defensive", "exact", "broad"],
                   help="none = exact cascade pinned at the MAP proposal (committed); else the "
                        "path-weighted cascade on this fixed proposal (runs/BS_weights/DESIGN.md)")
    p.add_argument("--path-budget-scale", type=float, default=1.5)
    p.add_argument("--path-steps-scale", type=float, default=1.0)
    p.add_argument("--path-spin", default="reparam", choices=["reparam", "is"])
    p.add_argument("--m-max-cap", type=float, default=None,
                   help="upper edge of the m_max_A prior (default: the MAP JSON's, 70)")
    args = p.parse_args()
    args.dense_mass = bool(args.dense_mass)
    return dict(parity=run_parity, nuts=run_nuts, status=run_status, bias=run_bias)[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
