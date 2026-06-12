"""S3b — ENGINE-backed mixed-likelihood injection-recovery.

Like mixed_recovery_demo.py, but the population's two representations are the
REAL coagulation engine outputs, not a controlled mixture:

  * sampled term  <- the engine particle-MC sampler  (toy make_sampler_fn),
  * discrete term <- the engine implicit-grid density (make_coagulation_dist_fn,
                     solver="implicit"), the same density mode the two-method
                     demo uses.

Single-pulse population (mu_natal, sigma_natal, log_rate). The natal pulse
gives low-mass events (narrow PE -> discrete); the hierarchical 2g+ descendants
give higher-mass events (broad PE -> sampled). Routed by sampled-estimator ESS.

This is the stretch goal: it exercises the engine implicit density inside the
mixed likelihood. The mixed DISCRETE term masks out-of-support PE points
(``safe_data``) before ``log_prob`` and uses a SAMPLE-based mu, so it avoids the
two nan sources that broke the analytical (density-arm) path in the two-method
demo (raw log_prob at PE samples + the total_rate estimator).

    python demo/inference/mixed_recovery_engine_demo.py --mode check
    python demo/inference/mixed_recovery_engine_demo.py --mode nuts
"""
import argparse
import importlib.util
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import numpyro.distributions as dist

from gwkokab.inference.event_likelihoods import gaussian_event_log_likelihoods
from gwkokab.inference.poissonlikelihood_utils import sampled_poisson_likelihood_fn
from gwkokab.inference.numpyro_mixed_poisson_likelihood import (
    numpyro_mixed_poisson_likelihood,
)
from gwkokab.models.utils import JointDistribution

from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (
    make_coagulation_dist_fn,
)

_HERE = os.path.dirname(__file__)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TOY = _load("toy", os.path.join(_HERE, "toy_single_pulse_sampled_likelihood.py"))
CMP = _load("cmp", os.path.join(_HERE, "compare_recovery_two_methods.py"))

N_SEED = int(os.environ.get("MR_N_SEED", "20000"))
N_OBS = int(os.environ.get("MR_N_OBS", "80"))
N_M = int(os.environ.get("MR_N_M", "16"))
N_CHI = int(os.environ.get("MR_N_CHI", "12"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("check", "nuts"), default="check")
    p.add_argument("--num-warmup", type=int, default=300)
    p.add_argument("--num-samples", type=int, default=600)
    args = p.parse_args()

    # ---- engine sampler_fn (MC) and dist_fn (implicit density) -------------
    toy_sampler = TOY.make_sampler_fn(N_SEED)

    def sampler_wrap(mu_natal, sigma_natal, log_rate):
        return toy_sampler(mu_natal, sigma_natal, log_rate)

    grid = BinaryGrid.clustered_chi(m_min=TOY.M_MIN, m_max=TOY.M_MAX, n_m=N_M, n_chi=N_CHI)
    eps_star = CMP._calibrate_epsilon(grid, TOY.F_DEP)
    _engine_dist = make_coagulation_dist_fn(
        v_esc=TOY.V_ESC, m_min=TOY.M_MIN, m_max=TOY.M_MAX, kernel_a=TOY.KERNEL_A,
        grid=grid, solver="implicit", epsilon=eps_star,
    )

    def dist_wrap(mu_natal, sigma_natal, log_rate):
        return _engine_dist(
            mu_natal=mu_natal, sigma_natal=sigma_natal,
            natal_spin_loc=TOY.SPIN_LOC, natal_spin_scale=TOY.SPIN_SCALE,
            kernel_b=TOY.KERNEL_B, log_rate=log_rate,
        )

    # ---- catalogue from the engine population, with differential PE --------
    Ls0, truths, observed = TOY.draw_catalog(toy_sampler, N_OBS, narrow=False)
    truths = np.asarray(truths)                         # (N_OBS, 4) m1,m2,chi1,chi2
    # mass-dependent measurement width: low-mass tight (~3%), high-mass broad.
    mc = (truths[:, 0] * truths[:, 1]) ** 0.6 / (truths[:, 0] + truths[:, 1]) ** 0.2
    frac_m = np.clip(0.03 + 0.004 * (mc - mc.min()), 0.03, 0.20)
    sig_m = (frac_m * truths[:, [0, 1]].T).T            # (N_OBS,2) on m1,m2
    sig_chi = np.full((N_OBS, 2), 0.15)
    sig = np.concatenate([sig_m, sig_chi], axis=1)      # (N_OBS,4)
    rng = np.random.default_rng(TOY.CATALOG_KEY + 5)
    obs = truths + sig * rng.standard_normal(truths.shape)
    obs[:, :2] = np.clip(obs[:, :2], TOY.M_MIN * 1.01, TOY.M_MAX * 0.99)
    obs[:, 2:] = np.clip(obs[:, 2:], 1e-3, 0.999)
    covs = np.zeros((N_OBS, 4, 4))
    covs[:, np.arange(4), np.arange(4)] = sig ** 2

    # ---- partition by sampled ESS at the truth -----------------------------
    s_samp, log_w = sampler_wrap(TOY.MU_TRUE, TOY.SIGMA_TRUE, TOY.LOGR_TRUE)
    Ls_all = gaussian_event_log_likelihoods(jnp.asarray(obs), jnp.asarray(covs))
    _, diag = sampled_poisson_likelihood_fn(
        Ls_all, s_samp, log_w, None, jnp.asarray(TOY.T_OBS), None
    )
    ess = np.asarray(diag["ess_per_event"])
    broad = ess >= 4 * N_OBS
    bi, ni = np.where(broad)[0], np.where(~broad)[0]
    print(f"eps* = {eps_star:.4f}")
    print(f"catalog: {N_OBS} events; Mc range [{mc.min():.1f}, {mc.max():.1f}]")
    print(f"partition: {bi.size} SAMPLED (Mc med {np.median(mc[bi]) if bi.size else float('nan'):.1f}, "
          f"ESS med {np.median(ess[bi]) if bi.size else float('nan'):.0f}) | "
          f"{ni.size} DISCRETE (Mc med {np.median(mc[ni]) if ni.size else float('nan'):.1f}, "
          f"ESS med {np.median(ess[ni]) if ni.size else float('nan'):.0f})")

    Ls_broad = gaussian_event_log_likelihoods(
        jnp.asarray(obs[bi]), jnp.asarray(covs[bi])
    )
    # discrete group: PE samples ~ N(obs_i, cov_i), masked to support, flat ref prior
    M = 2000
    pe = obs[ni][:, None, :] + sig[ni][:, None, :] * rng.standard_normal((ni.size, M, 4))
    pe[..., :2] = np.clip(pe[..., :2], TOY.M_MIN * 1.01, TOY.M_MAX * 0.99)
    pe[..., 2:] = np.clip(pe[..., 2:], 1e-3, 0.999)
    pe = jnp.asarray(pe)
    # in-support mask via the density's own support
    model0 = dist_wrap(TOY.MU_TRUE, TOY.SIGMA_TRUE, TOY.LOGR_TRUE)
    mask = jnp.asarray(model0.support.check(pe))
    data_group = (pe,)
    log_ref_priors_group = (jnp.zeros((ni.size, M)),)
    masks_group = (mask,)
    N_pes = (jnp.full((ni.size,), float(M)),)

    variables = {"mu_natal": dist.Uniform(5.0, 12.0),
                 "sigma_natal": dist.Uniform(0.3, 3.0),
                 "log_rate": dist.Uniform(TOY.LOGR_TRUE - 3, TOY.LOGR_TRUE + 3)}
    names = sorted(variables)                            # log_rate, mu_natal, sigma_natal
    vindex = {k: names.index(k) for k in variables}
    priors = JointDistribution(*[variables[k] for k in names])

    model = numpyro_mixed_poisson_likelihood(
        sampler_wrap, dist_wrap, priors, variables, {}, vindex,
        event_log_likelihood_fns=Ls_broad,
        data_group=data_group, log_ref_priors_group=log_ref_priors_group,
        masks_group=masks_group, N_pes=N_pes,
        pdet_fn=None, variance_cut_threshold=None,
    )

    truth = {"mu_natal": TOY.MU_TRUE, "sigma_natal": TOY.SIGMA_TRUE,
             "log_rate": TOY.LOGR_TRUE}

    if args.mode == "check":
        from numpyro.infer.util import log_density
        lp, _ = log_density(model, (jnp.asarray(TOY.T_OBS),), {}, truth)
        g = jax.grad(lambda mu: log_density(
            model, (jnp.asarray(TOY.T_OBS),), {},
            {**truth, "mu_natal": mu})[0])(jnp.asarray(TOY.MU_TRUE))
        # also probe near the prior edges (the init_to_uniform danger zone)
        edge = {"mu_natal": 11.8, "sigma_natal": 0.32, "log_rate": TOY.LOGR_TRUE}
        lp_e, _ = log_density(model, (jnp.asarray(TOY.T_OBS),), {}, edge)
        print(f"logL(truth) = {float(lp):.2f} (finite {np.isfinite(float(lp))}); "
              f"d/dmu_natal = {float(g):.3f} (finite {np.isfinite(float(g))})")
        print(f"logL(prior edge mu=11.8,sig=0.32) = {float(lp_e):.2f} "
              f"(finite {np.isfinite(float(lp_e))})  <- init robustness")
        return 0

    from numpyro.infer import MCMC, NUTS, init_to_median
    mcmc = MCMC(NUTS(model, init_strategy=init_to_median, target_accept_prob=0.8,
                     max_tree_depth=8),
                num_warmup=args.num_warmup, num_samples=args.num_samples,
                progress_bar=True)
    mcmc.run(jax.random.PRNGKey(3), jnp.asarray(TOY.T_OBS))
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    print("\nENGINE-backed MIXED recovery (truth in brackets):")
    for k in ("mu_natal", "sigma_natal", "log_rate"):
        v = post[k]
        z = (v.mean() - truth[k]) / v.std()
        print(f"  {k:12s}: {v.mean():7.3f} +/- {v.std():.3f}   [{truth[k]:.3f}]  z={z:+.2f}")
    outdir = os.path.join(_HERE, "figures_mixed_engine")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "mixed_engine_post.npz"), **post,
             **{f"truth_{k}": truth[k] for k in truth},
             n_broad=bi.size, n_narrow=ni.size, eps_star=eps_star)
    print(f"wrote {outdir}/mixed_engine_post.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
