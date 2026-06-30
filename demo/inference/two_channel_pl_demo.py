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
M.M_MAX = 70.0                      # WIDE proposal cap = top of the m_max prior;
                                    # the natal cutoff itself is a FREE parameter
                                    # (smooth target cutoff below), so the abundant
                                    # high-mass population can be NATAL, not 2g.
M_MAX_PROP = 70.0                   # proposal upper mass bound (must span the prior)
_CUT_W = 1.5                        # smoothing width of the high-mass natal cutoff
_NORM_GRID = jnp.linspace(M.M_MIN, M_MAX_PROP, 4001)


def _log_p_mass_pl_cutoff(m, alpha, m_max):
    """Pure power law (GWTC-3 low-mass Planck taper) with a SMOOTH, differentiable
    high-mass cutoff at the free natal edge m_max; normalized over the fixed grid."""
    def unnorm(mm):
        pl = M._smoothing(mm) * mm ** (-alpha)
        hi = jax.nn.sigmoid(-(mm - m_max) / _CUT_W)      # ~1 below m_max, ->0 above
        return pl * hi
    dm = _NORM_GRID[1] - _NORM_GRID[0]
    norm = jnp.sum(unnorm(_NORM_GRID)) * dm
    return jnp.log(unnorm(m) + 1e-300) - jnp.log(norm)


def make_sampler_A_mmax(n_seed):
    """Channel-A sampler: isotropic power-law natal with a FREE high-mass cutoff
    m_max, feeding the (faint) cascade. Replicates M.make_sampler_fn's IS machinery
    but threads a differentiable m_max through the target density."""
    m_prop, chi_prop, log_q = M.make_proposal(n_seed)

    def proposal(_key):
        return m_prop, chi_prop, log_q

    def _cfg(weight_array, kernel_a=M.KERNEL_A, kernel_b=M.KERNEL_B):
        return M.MCConfig(
            m_min=jnp.asarray(M.M_MIN), m_max=jnp.asarray(M_MAX_PROP),
            natal_spin_loc=jnp.asarray(M.CFG_SPIN_LOC0),
            natal_spin_scale=jnp.asarray(M.CFG_SPIN_SCALE0),
            kernel_a=jnp.asarray(kernel_a), kernel_b=jnp.asarray(kernel_b),
            v_esc=jnp.asarray(M.V_ESC), n_seed=int(n_seed),
            depletion_fraction=float(M.F_DEP),
            mu_natal=jnp.asarray(M.CFG_MU0), sigma_natal=jnp.asarray(M.CFG_SIGMA0),
            proposal=proposal, weight_array=weight_array)

    # natal IS denominator is kernel-independent -> fixed cfg0; n_steps (scan
    # length) is computed once from depletion_fraction * n_init * k_ratio(=1.5),
    # so the traced pairing kernel does NOT change the static scan length.
    log_p_cfg = M._natal_log_pdf_jax(m_prop, chi_prop, _cfg(None))

    def sampler_A(alpha, m_max, spin_scale, kernel_a, kernel_b, log_rate):
        log_p_target = (_log_p_mass_pl_cutoff(m_prop, alpha, m_max)
                        + M.log_p_spin(chi_prop, spin_scale))
        weight_array = jnp.exp(log_p_target - log_p_cfg)
        out = M.particle_mc_jax_soft(_cfg(weight_array, kernel_a, kernel_b), M.SAMPLER_KEY)
        samples = out["mergers"][:, :4]
        if M.ISOTROPIC_SPIN:
            n = samples.shape[0]
            cost = jax.random.uniform(M.SPIN_ORIENT_KEY, (n, 2), minval=-1.0, maxval=1.0)
            samples = samples.at[:, 2].multiply(cost[:, 0])
            samples = samples.at[:, 3].multiply(cost[:, 1])
        w_eff = out["w_mergers"] * out["step_active"]
        benign = jnp.asarray([10.0, 8.0, 0.0, 0.0])
        samples = jnp.where(jnp.isfinite(samples), samples, benign[None, :])
        log_w = jnp.log(jnp.maximum(w_eff, 1e-300))
        log_w = log_w - logsumexp(log_w) + log_rate
        return samples, log_w

    return sampler_A


def make_combined_pl(n_seed_A, n_b):
    samp_A = make_sampler_A_mmax(n_seed_A)       # power-law natal, FREE m_max, cascade
    samp_B = tc.make_channelB_sampler(n_b)       # aligned Gaussian, no cascade

    # NOTE: kernel_a/kernel_b are kept fixed (M.KERNEL_A/B). Freeing them was a
    # confirmed dead end -- a resolved (16k-seed) sweep shows natal pairing cannot
    # make the equal-mass high-mass (Mc>40) tail at ANY (a,b) (the tail is too rare
    # to resolve; gradient ~0). The high-mass tail is instead the genuine
    # low-spin population that disfavors 2g hierarchical formation (twog_exclusion.py).
    def combined(*, alpha_A, m_max_A, spin_chi, mu_B, sig_B, mu_chiB, sig_chiB, log_R_A, log_R_B):
        sA, lwA = samp_A(alpha_A, m_max_A, spin_chi, M.KERNEL_A, M.KERNEL_B, log_R_A)
        sB, lwB = samp_B(mu_B, sig_B, mu_chiB, sig_chiB, log_R_B)
        return jnp.concatenate([sA, sB], 0), jnp.concatenate([lwA, lwB], 0)
    sA0, _ = samp_A(2.0, 50.0, 0.1, M.KERNEL_A, M.KERNEL_B, 0.0)
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
    p.add_argument("--use-cov", action="store_true", help="full 3x3 NAL covariance per event (else diagonal sigs)")
    p.add_argument("--real-pdet", action="store_true")
    p.add_argument("--pdet-grid", default="/home/richard.oshaughnessy/coag_container/vt_data/mass_pdet_grid.npz")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--outdir", default=os.path.join(_HERE, "figures_two_channel_pl"))
    args = p.parse_args()
    M.F_DEP = args.f_dep
    comb, nA = make_combined_pl(args.n_seed, args.n_b)

    TRUTH = dict(alpha_A=2.0, m_max_A=50.0, spin_chi=0.05, mu_B=9.0, sig_B=1.2,
                 mu_chiB=0.15, sig_chiB=0.05, log_R_A=math.log(0.4), log_R_B=math.log(0.6))

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
        covs = zc["covs"] if (args.use_cov and "covs" in zc) else None
        if covs is not None:
            Ls = [M.GenieMcEtaChieffLikelihood(jnp.asarray(centers[i]), cov=jnp.asarray(covs[i])) for i in range(n_obs)]
            print(f"REAL catalog: {n_obs} events; FULL-COV NAL likelihood")
        else:
            Ls = [M.GenieMcEtaChieffLikelihood(jnp.asarray(centers[i]), jnp.asarray(sigs[i])) for i in range(n_obs)]
            print(f"REAL catalog: {n_obs} events; diagonal-sigma likelihood")
        logr0 = math.log(n_obs / M.T_OBS / 2)
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
        "alpha_A": dist.Uniform(1.5, 6.0), "m_max_A": dist.Uniform(40.0, 70.0),
        "spin_chi": dist.Uniform(0.01, 0.3),
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
