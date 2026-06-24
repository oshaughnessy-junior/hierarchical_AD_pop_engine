"""Two-channel mixture prototype (route a: combined sampler) + controlled recovery.

Channel A = Gaussian-seeded (lambda_peak=1) ISOTROPIC-spin coagulation cascade
[engine sampler]. Channel B = aligned Gaussian, NO cascade [direct draws]. The
combined sampler concatenates {x^A, ln w^A} and {x^B, ln(R_B/N_B)} so the existing
sampled_poisson_likelihood machinery sees one mixture population with two
independent rates (R_A, R_B in the weight sums).

  mode=check : forward de-risk (mass-dependent spin structure)
  mode=nuts  : controlled recovery -- inject a known 2-channel population (mu_A !=
               mu_B to test the test's POWER), recover both channels + rates,
               report mu_A vs mu_B.
"""
import argparse, importlib.util, os, math, numpy as np, jax, jax.numpy as jnp
from jax.scipy.special import logsumexp

_HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location(
    "plpeak", os.path.join(_HERE, "production_plpeak_sampled_recovery.py"))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
M.ISOTROPIC_SPIN = True

B_KEY = jax.random.PRNGKey(101)


def make_channelB_sampler(n_b):
    z_m = jax.random.normal(B_KEY, (n_b, 2))
    z_chi = jax.random.normal(jax.random.fold_in(B_KEY, 1), (n_b,))

    def sampler_B(mu_B, sig_B, mu_chiB, sig_chiB, log_R_B):
        m = mu_B + sig_B * z_m
        m1 = jnp.clip(jnp.maximum(m[:, 0], m[:, 1]), M.M_MIN * 1.01, M.M_MAX * 0.99)
        m2 = jnp.clip(jnp.minimum(m[:, 0], m[:, 1]), M.M_MIN * 1.01, M.M_MAX * 0.99)
        chi = jnp.clip(mu_chiB + sig_chiB * z_chi, -0.99, 0.99)
        s = jnp.stack([m1, m2, chi, chi], axis=1)
        return s, jnp.full((n_b,), log_R_B - math.log(n_b))
    return sampler_B


def make_combined_sampler(n_seed_A, n_b):
    samp_A = M.make_sampler_fn(n_seed_A)
    samp_B = make_channelB_sampler(n_b)

    def combined(*, mu_A, sig_A, spin_chi, mu_B, sig_B, mu_chiB, sig_chiB, log_R_A, log_R_B):
        sA, lwA = samp_A(3.0, 1.0, mu_A, sig_A, spin_chi, log_R_A)   # lambda_peak=1
        sB, lwB = samp_B(mu_B, sig_B, mu_chiB, sig_chiB, log_R_B)
        return jnp.concatenate([sA, sB], 0), jnp.concatenate([lwA, lwB], 0)
    # probe cascade output size for the check
    sA0, _ = samp_A(3.0, 1.0, 10.0, 1.5, 0.05, 0.0)
    return combined, int(sA0.shape[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="check")
    p.add_argument("--f-dep", type=float, default=0.20)
    p.add_argument("--n-seed", type=int, default=12000)
    p.add_argument("--n-b", type=int, default=6000)
    p.add_argument("--n-obs", type=int, default=200)
    p.add_argument("--num-warmup", type=int, default=150)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--max-tree-depth", type=int, default=5)
    p.add_argument("--target-accept", type=float, default=0.8)
    p.add_argument("--catalog", default=None, help="real catalog npz (else synthetic)")
    p.add_argument("--real-pdet", action="store_true")
    p.add_argument("--pdet-grid",
                   default="/home/richard.oshaughnessy/coag_container/vt_data/mass_pdet_grid.npz")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--outdir", default=os.path.join(_HERE, "figures_two_channel"))
    args = p.parse_args()
    M.F_DEP = args.f_dep
    comb, nA = make_combined_sampler(args.n_seed, args.n_b)

    # injected truth: DIFFERENT mu_A, mu_B -> tests the mu_A==mu_B test's POWER
    TRUTH = dict(mu_A=12.0, sig_A=1.5, spin_chi=0.05, mu_B=8.0, sig_B=1.2,
                 mu_chiB=0.18, sig_chiB=0.05,
                 log_R_A=math.log(0.5), log_R_B=math.log(0.5))

    if args.mode == "check":
        s, lw = comb(**TRUTH); s = np.asarray(s); w = np.exp(np.asarray(lw))
        m1, m2, c1, c2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2; chie = (m1 * c1 + m2 * c2) / (m1 + m2)
        isA = np.zeros(len(s), bool); isA[:nA] = True
        print(f"R_A={w[isA].sum():.3f} R_B={w[~isA].sum():.3f} (inj 0.5/0.5)")
        for lo, hi in [(4, 9), (9, 14), (14, 22), (22, 35), (35, 60)]:
            m = (mc >= lo) & (mc < hi); ww = w[m]
            if ww.sum() > 1e-6:
                wn = ww / ww.sum(); mu = np.sum(wn * chie[m]); sd = np.sqrt(np.sum(wn * (chie[m] - mu) ** 2))
                print(f"  Mc[{lo},{hi}): <chi_eff>={mu:+.3f} sd={sd:.3f} Bfrac={w[m & ~isA].sum()/ww.sum():.2f}")
        return 0

    # ---- mode=nuts: real catalog OR synthetic, then recover ----
    if args.real_pdet:
        from mass_pdet_interp import make_real_pdet_fn
        pdet_fn = make_real_pdet_fn(args.pdet_grid); print("SELECTION: real xi(m1,m2)")
    else:
        pdet_fn = M.pdet_fn
    REAL = args.catalog is not None
    if REAL:
        zc = np.load(args.catalog, allow_pickle=True)
        centers, sigs = zc["centers"], zc["sigs"]; n_obs = len(centers)
        Ls = [M.GenieMcEtaChieffLikelihood(jnp.asarray(centers[i]), jnp.asarray(sigs[i])) for i in range(n_obs)]
        logr0 = math.log(n_obs / M.T_OBS / 2)
        print(f"REAL catalog: {n_obs} events; Mc [{centers[:,0].min():.1f},{centers[:,0].max():.1f}]; logr0={logr0:.2f}")
    else:
        s, lw = comb(**TRUTH); s = np.asarray(s)
        pd = np.asarray(pdet_fn(jnp.asarray(s)))
        dw = np.exp(np.asarray(lw)) * pd; dw = np.clip(dw, 0, None); dw /= dw.sum()
        rng = np.random.default_rng(7)
        pick = rng.choice(len(s), size=args.n_obs, p=dw)
        x = np.asarray(M.mc_eta_chieff(jnp.asarray(s[pick])))
        sig = M.genie_widths(x[:, 0], x[:, 1], x[:, 2])
        obs = x + sig * rng.standard_normal(x.shape)
        Ls = [M.GenieMcEtaChieffLikelihood(jnp.asarray(obs[i]), jnp.asarray(sig[i])) for i in range(args.n_obs)]
        logr0 = math.log(args.n_obs / M.T_OBS / 2)
        print(f"synthetic catalog: {args.n_obs} events; Mc [{x[:,0].min():.1f},{x[:,0].max():.1f}]; logr0={logr0:.2f}")

    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_median
    from gwkokab.inference.numpyro_sampled_poisson_likelihood import numpyro_sampled_poisson_likelihood
    from gwkokab.models.utils import JointDistribution
    mass_hi = 35. if REAL else 20.
    variables = {
        "mu_A": dist.Uniform(5., mass_hi), "sig_A": dist.Uniform(0.5, 6.),
        "spin_chi": dist.Uniform(0.01, 0.3),
        "mu_B": dist.Uniform(5., mass_hi), "sig_B": dist.Uniform(0.5, 6.),
        "mu_chiB": dist.Uniform(-0.1, 0.5), "sig_chiB": dist.Uniform(0.01, 0.3),
        "log_R_A": dist.Uniform(logr0 - 3, logr0 + 3),
        "log_R_B": dist.Uniform(logr0 - 3, logr0 + 3)}
    names = sorted(variables)
    priors = JointDistribution(*[variables[k] for k in names])
    kw = dict(init_strategy=init_to_median, max_tree_depth=args.max_tree_depth,
              target_accept_prob=args.target_accept)
    model = numpyro_sampled_poisson_likelihood(
        comb, priors, variables, {}, {k: i for i, k in enumerate(names)}, Ls, pdet_fn, None)
    if args.checkpoint_dir:
        from gwkokab.inference import run_nuts_with_checkpoints
        print(f"CHECKPOINT-SAFE NUTS: {args.checkpoint_dir} chunk={args.chunk}")
        post = run_nuts_with_checkpoints(model, (jnp.asarray(M.T_OBS),), num_warmup=args.num_warmup,
                                         num_samples=args.num_samples, key=jax.random.PRNGKey(0),
                                         ckpt_dir=args.checkpoint_dir, chunk=args.chunk, nuts_kwargs=kw)
    else:
        mcmc = MCMC(NUTS(model, **kw), num_warmup=args.num_warmup, num_samples=args.num_samples,
                    progress_bar=True)
        mcmc.run(jax.random.PRNGKey(0), jnp.asarray(M.T_OBS))
        post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    print("\nTWO-CHANNEL recovery:")
    for k in names:
        tag = "" if REAL else f"  [{TRUTH[k]:.3f}]  z={(post[k].mean()-TRUTH[k])/post[k].std():+.2f}"
        print(f"  {k:9s}: {post[k].mean():7.3f} +/- {post[k].std():.3f}{tag}")
    dmu = post["mu_A"] - post["mu_B"]
    print(f"\n  *** mu_A - mu_B = {dmu.mean():+.2f} +/- {dmu.std():.2f}  ***  "
          f"({'consistent with 0 (SAME natal mass!)' if abs(dmu.mean())<2*dmu.std() else 'DISTINCT masses'})")
    print(f"  R_A/R_B = {np.exp(post['log_R_A']-post['log_R_B']).mean():.2f}")
    os.makedirs(args.outdir, exist_ok=True)
    extra = {} if REAL else {f"truth_{k}": TRUTH[k] for k in TRUTH}
    np.savez(os.path.join(args.outdir, "two_channel_post.npz"), **post, **extra, f_dep=args.f_dep)
    print(f"wrote {args.outdir}/two_channel_post.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
