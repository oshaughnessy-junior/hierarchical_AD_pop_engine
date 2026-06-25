"""Two-channel mixture, POWER-LAW natal variant (ROS 2026-06-25 model).

  (a) Channel B: a low-mass ALIGNED Gaussian (narrow) -- the dominant low-mass
      population, aligned spins.
  (b) Channel A: a POWER-LAW natal (~10-30 Msun, natal cutoff M_MAX=35) feeding a
      FAINT cascade (low F_DEP), FULLY ISOTROPIC, inheriting the natal spin
      distribution. The cascade builds only the high-mass tail (remnant masses
      exceed the natal cutoff).

The mixture of the two natal spin distributions (aligned B + isotropic-small A)
gives the narrow overall chi_eff; the faint cascade avoids the over-wide
high-mass chi_eff of the deep-cascade Gaussian-seed model. Reuses the aligned
Gaussian sampler and the real-catalog / checkpoint machinery of two_channel_demo.

  mode=check : forward structure
  mode=nuts  : recovery (synthetic or --catalog real, --real-pdet, --checkpoint-dir)
"""
import argparse, importlib.util, os, math, numpy as np, jax, jax.numpy as jnp
from jax.scipy.special import logsumexp

_HERE = os.path.dirname(__file__)
TC = importlib.util.spec_from_file_location("tc", os.path.join(_HERE, "two_channel_demo.py"))
tc = importlib.util.module_from_spec(TC); TC.loader.exec_module(tc)
M = tc.M
M.ISOTROPIC_SPIN = True
M.M_MAX = 35.0                      # natal cutoff; cascade remnants exceed it


def make_combined_pl(n_seed_A, n_b):
    samp_A = M.make_sampler_fn(n_seed_A)        # power-law natal (lambda_peak=0) cascade
    samp_B = tc.make_channelB_sampler(n_b)      # aligned Gaussian, no cascade

    def combined(*, alpha_A, spin_chi, mu_B, sig_B, mu_chiB, sig_chiB, log_R_A, log_R_B):
        sA, lwA = samp_A(alpha_A, 0.0, 30.0, 3.0, spin_chi, log_R_A)   # lambda_peak=0
        sB, lwB = samp_B(mu_B, sig_B, mu_chiB, sig_chiB, log_R_B)
        return jnp.concatenate([sA, sB], 0), jnp.concatenate([lwA, lwB], 0)
    sA0, _ = samp_A(3.0, 0.0, 30.0, 3.0, 0.1, 0.0)
    return combined, int(sA0.shape[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="check")
    p.add_argument("--f-dep", type=float, default=0.15)    # faint cascade
    p.add_argument("--n-seed", type=int, default=8000)
    p.add_argument("--n-b", type=int, default=6000)
    p.add_argument("--n-obs", type=int, default=200)
    p.add_argument("--num-warmup", type=int, default=150)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--max-tree-depth", type=int, default=5)
    p.add_argument("--target-accept", type=float, default=0.8)
    p.add_argument("--catalog", default=None)
    p.add_argument("--real-pdet", action="store_true")
    p.add_argument("--pdet-grid", default="/home/richard.oshaughnessy/coag_container/vt_data/mass_pdet_grid.npz")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--outdir", default=os.path.join(_HERE, "figures_two_channel_pl"))
    args = p.parse_args()
    M.F_DEP = args.f_dep
    comb, nA = make_combined_pl(args.n_seed, args.n_b)

    TRUTH = dict(alpha_A=3.0, spin_chi=0.05, mu_B=9.0, sig_B=1.2, mu_chiB=0.15,
                 sig_chiB=0.05, log_R_A=math.log(0.4), log_R_B=math.log(0.6))

    if args.mode == "check":
        s, lw = comb(**TRUTH); s = np.asarray(s); w = np.exp(np.asarray(lw))
        m1 = np.maximum(s[:, 0], s[:, 1]); c1, c2 = s[:, 2], s[:, 3]
        mc = (s[:, 0] * s[:, 1]) ** 0.6 / (s[:, 0] + s[:, 1]) ** 0.2
        chie = (s[:, 0] * c1 + s[:, 1] * c2) / (s[:, 0] + s[:, 1])
        isA = np.zeros(len(s), bool); isA[:nA] = True
        print(f"R_A={w[isA].sum():.3f} R_B={w[~isA].sum():.3f}  (max m1={m1.max():.0f})")
        for lo, hi in [(4, 9), (9, 14), (14, 22), (22, 35), (35, 70)]:
            m = (mc >= lo) & (mc < hi); ww = w[m]
            if ww.sum() > 1e-6:
                wn = ww / ww.sum(); mu = np.sum(wn * chie[m]); sd = np.sqrt(np.sum(wn * (chie[m] - mu) ** 2))
                print(f"  Mc[{lo},{hi}): <chi_eff>={mu:+.3f} sd={sd:.3f} Bfrac={w[m & ~isA].sum()/ww.sum():.2f}")
        return 0

    # ---- recovery ----
    if args.real_pdet:
        from mass_pdet_interp import make_real_pdet_fn
        pdet_fn = make_real_pdet_fn(args.pdet_grid); print("SELECTION: real xi")
    else:
        pdet_fn = M.pdet_fn
    REAL = args.catalog is not None
    if REAL:
        zc = np.load(args.catalog, allow_pickle=True); centers, sigs = zc["centers"], zc["sigs"]; n_obs = len(centers)
        Ls = [M.GenieMcEtaChieffLikelihood(jnp.asarray(centers[i]), jnp.asarray(sigs[i])) for i in range(n_obs)]
        logr0 = math.log(n_obs / M.T_OBS / 2)
        print(f"REAL catalog: {n_obs} events; logr0={logr0:.2f}")
    else:
        s, lw = comb(**TRUTH); s = np.asarray(s)
        dw = np.exp(np.asarray(lw)) * np.asarray(pdet_fn(jnp.asarray(s))); dw = np.clip(dw, 0, None); dw /= dw.sum()
        rng = np.random.default_rng(7); pick = rng.choice(len(s), size=args.n_obs, p=dw)
        x = np.asarray(M.mc_eta_chieff(jnp.asarray(s[pick]))); sig = M.genie_widths(x[:, 0], x[:, 1], x[:, 2])
        obs = x + sig * rng.standard_normal(x.shape)
        Ls = [M.GenieMcEtaChieffLikelihood(jnp.asarray(obs[i]), jnp.asarray(sig[i])) for i in range(args.n_obs)]
        logr0 = math.log(args.n_obs / M.T_OBS / 2)

    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_median
    from gwkokab.inference.numpyro_sampled_poisson_likelihood import numpyro_sampled_poisson_likelihood
    from gwkokab.models.utils import JointDistribution
    variables = {
        "alpha_A": dist.Uniform(1.5, 6.0), "spin_chi": dist.Uniform(0.01, 0.3),
        "mu_B": dist.Uniform(5., 15.), "sig_B": dist.Uniform(0.5, 3.0),   # NARROW low-mass peak
        "mu_chiB": dist.Uniform(-0.1, 0.5), "sig_chiB": dist.Uniform(0.01, 0.3),
        "log_R_A": dist.Uniform(logr0 - 3, logr0 + 3), "log_R_B": dist.Uniform(logr0 - 3, logr0 + 3)}
    names = sorted(variables)
    priors = JointDistribution(*[variables[k] for k in names])
    kw = dict(init_strategy=init_to_median, max_tree_depth=args.max_tree_depth, target_accept_prob=args.target_accept)
    model = numpyro_sampled_poisson_likelihood(comb, priors, variables, {}, {k: i for i, k in enumerate(names)}, Ls, pdet_fn, None)
    if args.checkpoint_dir:
        from gwkokab.inference import run_nuts_with_checkpoints
        print(f"CHECKPOINT-SAFE: {args.checkpoint_dir} chunk={args.chunk}")
        post = run_nuts_with_checkpoints(model, (jnp.asarray(M.T_OBS),), num_warmup=args.num_warmup,
                                         num_samples=args.num_samples, key=jax.random.PRNGKey(0),
                                         ckpt_dir=args.checkpoint_dir, chunk=args.chunk, nuts_kwargs=kw)
    else:
        mcmc = MCMC(NUTS(model, **kw), num_warmup=args.num_warmup, num_samples=args.num_samples, progress_bar=True)
        mcmc.run(jax.random.PRNGKey(0), jnp.asarray(M.T_OBS)); post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    print("\nPOWER-LAW two-channel recovery:")
    for k in names:
        tag = "" if REAL else f"  [{TRUTH[k]:.3f}]"
        print(f"  {k:9s}: {post[k].mean():7.3f} +/- {post[k].std():.3f}{tag}")
    print(f"  R_A/R_B = {np.exp(post['log_R_A']-post['log_R_B']).mean():.2f}")
    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, "two_channel_pl_post.npz"), **post, f_dep=args.f_dep,
             **({} if REAL else {f"truth_{k}": TRUTH[k] for k in TRUTH}))
    print(f"wrote {args.outdir}/two_channel_pl_post.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
