"""S4(a): coag-engine population inference on the REAL O3+O4 catalogue.

Loads the deconvolved per-event (Mc, eta, chi_eff) centres+widths (built by
planning_o4_mixed/extract_full_catalog.py with proper marginal-prior
deconvolution + FAR<1/yr), builds Genie-style per-event likelihoods, and runs
the plpeak sampled-mode NUTS (same machinery as the synthetic JOB A) to infer
the natal-mass PL+Peak + spin + rate on real data.

CAVEATS (documented, to refine): selection uses the TOY pdet ~ Mc^15/6
(placeholder; real injection-set VT is the next step) -> the SHAPE params
(mu_peak, sigma_peak, alpha) are meaningful; the absolute rate is VT-limited.

    python real_plpeak_recovery.py --catalog full_real_catalog.npz --mode nuts
"""
import argparse, importlib.util, os, math
import numpy as np, jax, jax.numpy as jnp

_HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location(
    "plpeak", os.path.join(_HERE, "production_plpeak_sampled_recovery.py"))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--mode", choices=("check", "nuts"), default="nuts")
    p.add_argument("--n-seed", type=int, default=50000)
    p.add_argument("--num-warmup", type=int, default=300)
    p.add_argument("--num-samples", type=int, default=600)
    p.add_argument("--outdir", default=os.path.join(_HERE, "figures_real"))
    p.add_argument("--init-median", action="store_true")
    p.add_argument("--max-tree-depth", type=int, default=8)
    p.add_argument("--target-accept", type=float, default=0.8)
    p.add_argument("--real-pdet", action="store_true",
                   help="use the real GWTC-5.0 z-averaged detection probability "
                        "xi(m1,m2) in [0,1] instead of the toy Mc^15/6 "
                        "(mass_pdet_interp.make_real_pdet_fn)")
    p.add_argument("--pdet-grid",
                   default="/home/richard.oshaughnessy/coag_container/vt_data/mass_pdet_grid.npz")
    p.add_argument("--checkpoint-dir", default=None,
                   help="enable checkpoint-safe NUTS (warmup once + chunked sampling, "
                        "resumes from newest checkpoint on restart)")
    p.add_argument("--chunk", type=int, default=50,
                   help="samples per checkpoint chunk (smaller = more eviction-resilient)")
    p.add_argument("--isotropic-spin", action="store_true",
                   help="oriented spins: isotropic tilts -> signed chi_eff "
                        "(production_plpeak.ISOTROPIC_SPIN)")
    p.add_argument("--f-dep", type=float, default=None,
                   help="fix the cascade depletion fraction (profile point); "
                        "default keeps the module value")
    args = p.parse_args()

    # set the cascade knobs on the imported module BEFORE building the sampler
    if args.isotropic_spin:
        M.ISOTROPIC_SPIN = True
        print("SPIN: isotropic orientation (signed chi_eff)")
    if args.f_dep is not None:
        M.F_DEP = float(args.f_dep)
        print(f"CASCADE: F_DEP = {M.F_DEP}")

    if args.real_pdet:
        from mass_pdet_interp import make_real_pdet_fn
        pdet_fn = make_real_pdet_fn(args.pdet_grid)
        print(f"SELECTION: REAL injection pdet xi(m1,m2) from {args.pdet_grid}")
    else:
        pdet_fn = M.pdet_fn
        print("SELECTION: toy Mc^15/6 (placeholder)")

    z = np.load(args.catalog, allow_pickle=True)
    centers, sigs = z["centers"], z["sigs"]          # (N,3) Mc,eta,chi_eff
    n_obs = len(centers)
    Ls = [M.GenieMcEtaChieffLikelihood(jnp.asarray(centers[i]), jnp.asarray(sigs[i]))
          for i in range(n_obs)]
    sampler_fn = M.make_sampler_fn(args.n_seed)
    # rough rate scale: mu ~ n_obs at the MLE (placeholder VT)
    logr0 = math.log(n_obs / M.T_OBS)
    print(f"REAL catalog: {n_obs} events (O3+O4); Mc [{centers[:,0].min():.1f},"
          f"{centers[:,0].max():.1f}]  logr0={logr0:.2f}")

    if args.mode == "check":
        ll, diag = M.loglik(sampler_fn, Ls, **M.TRUTH, log_rate=logr0)
        ess = np.asarray(diag["ess_per_event"])
        print(f"logL = {float(ll):.1f}; per-event ESS min {ess.min():.0f} med "
              f"{np.median(ess):.0f}; events ESS<10: {int((ess<10).sum())}/{n_obs}")
        return 0

    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_median
    from gwkokab.inference.numpyro_sampled_poisson_likelihood import (
        numpyro_sampled_poisson_likelihood)
    from gwkokab.models.utils import JointDistribution
    variables = {
        "alpha": dist.Uniform(-2.0, 8.0), "lambda_peak": dist.Uniform(0.0, 0.6),
        "mu_peak": dist.Uniform(20.0, 45.0), "sigma_peak": dist.Uniform(1.0, 10.0),
        "spin_scale": dist.Uniform(0.01, 0.5),
        "log_rate": dist.Uniform(logr0 - 4, logr0 + 4)}
    names = sorted(variables)
    priors = JointDistribution(*[variables[k] for k in names])
    model = numpyro_sampled_poisson_likelihood(
        sampler_fn, priors, variables, {}, {k: i for i, k in enumerate(names)},
        Ls, pdet_fn, None)
    kw = dict(max_tree_depth=args.max_tree_depth, target_accept_prob=args.target_accept)
    if args.init_median: kw["init_strategy"] = init_to_median
    if args.checkpoint_dir:
        from gwkokab.inference import run_nuts_with_checkpoints
        print(f"CHECKPOINT-SAFE NUTS: dir={args.checkpoint_dir} chunk={args.chunk}")
        post = run_nuts_with_checkpoints(
            model, (jnp.asarray(M.T_OBS),), num_warmup=args.num_warmup,
            num_samples=args.num_samples, key=jax.random.PRNGKey(5),
            ckpt_dir=args.checkpoint_dir, chunk=args.chunk, nuts_kwargs=kw)
    else:
        mcmc = MCMC(NUTS(model, **kw), num_warmup=args.num_warmup,
                    num_samples=args.num_samples, progress_bar=True)
        mcmc.run(jax.random.PRNGKey(5), jnp.asarray(M.T_OBS))
        post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    print("\nREAL-DATA mass-function inference (no truth; posterior mean +/- sd):")
    for k in names:
        print(f"  {k:12s}: {post[k].mean():7.3f} +/- {post[k].std():.3f}")
    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, "real_post.npz"), **post, n_obs=n_obs)
    print(f"wrote {args.outdir}/real_post.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
