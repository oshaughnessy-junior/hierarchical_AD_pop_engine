"""S3 — injection-recovery with the MIXED Poisson likelihood.

Validates gwkokab's mixed_poisson_likelihood_fn / numpyro_mixed_poisson_
likelihood end-to-end in NUTS: a two-component population (a narrow low-mass
NATAL pulse + a broad high-mass HIERARCHICAL bump) is injected; a catalogue is
drawn with mass-dependent measurement widths (low-mass -> narrow PE, high-mass
-> broad PE); each event is routed by its SAMPLED-estimator ESS to either the
sampled term (continuous Gaussian-approx L_i at the model draws, broad events)
or the discrete term (posterior samples vs the model density, narrow events);
and the population (mu_bump, frac_hier, log_rate) is recovered.

This uses a controlled population with BOTH a tractable AD-safe sampler and a
tractable density (a truncated-Gaussian mixture), so it isolates the mixed
LIKELIHOOD mechanism. Engine-backed recovery (the coagulation sampler_fn +
implicit dist_fn) is the S3b follow-up; see the plan.

1-D mass axis (Msun). Run:
    python demo/inference/mixed_recovery_demo.py --mode check
    python demo/inference/mixed_recovery_demo.py --mode nuts --num-warmup 300 --num-samples 600
"""
import argparse
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

from gwkokab.inference.event_likelihoods import gaussian_event_log_likelihoods
from gwkokab.inference.poissonlikelihood_utils import (
    sampled_poisson_likelihood_fn,
    low_ess_events,
)
from gwkokab.inference.numpyro_mixed_poisson_likelihood import (
    numpyro_mixed_poisson_likelihood,
)
from gwkokab.models.utils import JointDistribution

# ---- truth + fixed model constants ---------------------------------------
M_MIN, M_MAX = 4.0, 80.0
MU1_TRUE, S1_TRUE = 10.0, 1.6        # narrow low-mass natal pulse
MU2_TRUE, S2_TRUE = 32.0, 6.0        # broad high-mass hierarchical bump
FRAC_TRUE = 0.30                      # hierarchical fraction
LOGR_TRUE = math.log(400.0)
T_OBS = 1.0
MC_REF, PDET_P = 22.0, 2.0            # pdet = min((m/MC_REF)^p, 1)
CATALOG_SEED, SAMPLER_SEED = 7, 11
N_SEED = 30000
N_OBS = 150

# proposal for the AD-safe sampler (broad, fixed)
_QM, _QS = 28.0, 22.0


def _tn_logpdf(m, mu, s):
    a, b = (M_MIN - mu) / s, (M_MAX - mu) / s
    return jax.scipy.stats.truncnorm.logpdf(m, a, b, loc=mu, scale=s)


def shape_logpdf(m, mu1, s1, mu2, s2, frac):
    """log of the (unit-norm) 2-component truncated-Gaussian natal+hier shape."""
    lo = jnp.log1p(-frac) + _tn_logpdf(m, mu1, s1)
    hi = jnp.log(frac) + _tn_logpdf(m, mu2, s2)
    return jnp.logaddexp(lo, hi)


# ---- sampler_fn: fixed-proposal IS draws (AD-safe in the shape params) ----
_KEY = jax.random.PRNGKey(SAMPLER_SEED)
_a, _b = (M_MIN - _QM) / _QS, (M_MAX - _QM) / _QS
_m_prop = (_QM + _QS * jax.random.truncated_normal(_KEY, _a, _b, (N_SEED,)))[:, None]
_logq = _tn_logpdf(_m_prop[:, 0], _QM, _QS)


def sampler_fn(mu2, frac, log_rate,
               mu1=MU1_TRUE, s1=S1_TRUE, s2=S2_TRUE):
    log_shape = shape_logpdf(_m_prop[:, 0], mu1, s1, mu2, s2, frac)
    log_w = log_rate + log_shape - _logq          # log[ R n_shape / q ]
    log_w = log_w - jax.nn.logsumexp(log_w) + log_rate   # normalise Sum w = R
    return _m_prop, log_w


# ---- dist_fn: the population density as a numpyro Distribution -------------
class NatalHierMixture(dist.Distribution):
    support = constraints.interval(M_MIN, M_MAX)

    def __init__(self, log_rate, mu1, s1, mu2, s2, frac):
        self._lr, self._mu1, self._s1 = log_rate, mu1, s1
        self._mu2, self._s2, self._frac = mu2, s2, frac
        super().__init__(event_shape=(1,))

    def log_prob(self, value):
        m = value[..., 0]
        return self._lr + shape_logpdf(m, self._mu1, self._s1,
                                       self._mu2, self._s2, self._frac)


def dist_fn(mu2, frac, log_rate, mu1=MU1_TRUE, s1=S1_TRUE, s2=S2_TRUE):
    return NatalHierMixture(log_rate, mu1, s1, mu2, s2, frac)


def pdet_fn(samples):
    m = samples[:, 0]
    return jnp.minimum((m / MC_REF) ** PDET_P, 1.0)


# ---- catalogue + per-event representations --------------------------------
def draw_catalog():
    m_prop, log_w = sampler_fn(MU2_TRUE, FRAC_TRUE, LOGR_TRUE)
    x = np.asarray(m_prop[:, 0])
    w = np.exp(np.asarray(log_w - jax.nn.logsumexp(log_w)))
    pd = np.asarray(pdet_fn(m_prop))
    p = w * pd
    p = p / p.sum()
    rng = np.random.default_rng(CATALOG_SEED)
    idx = rng.choice(x.shape[0], size=N_OBS, replace=True, p=p)
    truths = x[idx]
    # DIFFERENTIAL measurement width (the real split): low-mass events are
    # tightly measured in chirp mass (~2% -> sampled ESS collapses -> discrete),
    # high-mass broad (~up to ~22% -> sampled ESS healthy -> sampled). Matches
    # the S1 real-data smoke (low-mass Mc 5.7+/-0.1; high-mass Mc 43+/-5).
    sig_frac = np.clip(0.02 + 0.005 * (truths - 8.0), 0.02, 0.22)
    sig = sig_frac * truths
    obs = truths + sig * rng.standard_normal(N_OBS)
    obs = np.clip(obs, M_MIN * 1.01, M_MAX * 0.99)
    return obs, sig, rng


def build_partition(obs, sig):
    """Route each event by its SAMPLED-estimator ESS at the fiducial truth."""
    m_prop, log_w = sampler_fn(MU2_TRUE, FRAC_TRUE, LOGR_TRUE)
    Ls_all = gaussian_event_log_likelihoods(
        jnp.asarray(obs)[:, None], (jnp.asarray(sig) ** 2)[:, None, None]
    )
    _, diag = sampled_poisson_likelihood_fn(
        Ls_all, m_prop, log_w, pdet_fn, jnp.asarray(T_OBS), None
    )
    ess = np.asarray(diag["ess_per_event"])
    thr = 4 * N_OBS
    broad = ess >= thr                          # sampled term
    narrow = ~broad                             # discrete term
    return broad, narrow, ess


def make_discrete_group(obs, sig, narrow_idx, rng, M=4000):
    """Posterior samples y ~ N(obs_i, sig_i), flat ref prior (log pi = 0)."""
    d = obs[narrow_idx]
    s = sig[narrow_idx]
    n = len(narrow_idx)
    pe = np.clip(
        d[:, None] + s[:, None] * rng.standard_normal((n, M)),
        M_MIN * 1.01, M_MAX * 0.99,
    )[..., None]                                # (n, M, 1)
    data_group = (jnp.asarray(pe),)
    log_ref_priors_group = (jnp.zeros((n, M)),)
    masks_group = (jnp.ones((n, M), dtype=bool),)
    N_pes = (jnp.full((n,), float(M)),)
    return data_group, log_ref_priors_group, masks_group, N_pes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("check", "nuts"), default="check")
    p.add_argument("--num-warmup", type=int, default=300)
    p.add_argument("--num-samples", type=int, default=600)
    args = p.parse_args()

    obs, sig, rng = draw_catalog()
    broad, narrow, ess = build_partition(obs, sig)
    bi = np.where(broad)[0]
    ni = np.where(narrow)[0]
    print(f"catalog: {N_OBS} events; mass range [{obs.min():.1f}, {obs.max():.1f}]")
    print(f"partition: {bi.size} SAMPLED (broad/high-mass), {ni.size} DISCRETE "
          f"(narrow/low-mass); ESS thresh 4*N_obs={4*N_OBS}")
    print(f"  broad  mass median {np.median(obs[bi]):.1f}, sampled-ESS median "
          f"{np.median(ess[bi]):.0f}")
    print(f"  narrow mass median {np.median(obs[ni]):.1f}, sampled-ESS median "
          f"{np.median(ess[ni]):.0f}")

    Ls_broad = gaussian_event_log_likelihoods(
        jnp.asarray(obs[bi])[:, None], (jnp.asarray(sig[bi]) ** 2)[:, None, None]
    )
    dg, lpg, mg, npes = make_discrete_group(obs, sig, ni, rng)

    if args.mode == "check":
        # finite + differentiable potential at truth
        from numpyro.infer.util import log_density
        variables = {"frac": dist.Uniform(0.01, 0.8),
                     "log_rate": dist.Uniform(LOGR_TRUE - 3, LOGR_TRUE + 3),
                     "mu2": dist.Uniform(20.0, 45.0)}
        vindex = {"mu2": 2, "frac": 0, "log_rate": 1}  # sorted: frac,log_rate,mu2
        priors = JointDistribution(*[variables[k] for k in sorted(variables)])
        model = numpyro_mixed_poisson_likelihood(
            sampler_fn, dist_fn, priors, variables, {}, vindex,
            event_log_likelihood_fns=Ls_broad,
            data_group=dg, log_ref_priors_group=lpg, masks_group=mg, N_pes=npes,
            pdet_fn=pdet_fn, variance_cut_threshold=None,
        )
        truth = {"frac": FRAC_TRUE, "log_rate": LOGR_TRUE, "mu2": MU2_TRUE}
        lp, _ = log_density(model, (jnp.asarray(T_OBS),), {}, truth)
        g = jax.grad(lambda mu: log_density(
            model, (jnp.asarray(T_OBS),), {},
            {**truth, "mu2": mu})[0])(jnp.asarray(MU2_TRUE))
        print(f"logL(truth) = {float(lp):.2f} (finite {np.isfinite(float(lp))}); "
              f"d/dmu2 = {float(g):.3f} (finite {np.isfinite(float(g))})")
        return 0

    # NUTS recovery
    from numpyro.infer import MCMC, NUTS, init_to_median
    variables = {"frac": dist.Uniform(0.01, 0.8),
                 "log_rate": dist.Uniform(LOGR_TRUE - 3, LOGR_TRUE + 3),
                 "mu2": dist.Uniform(20.0, 45.0)}
    vindex = {"mu2": 2, "frac": 0, "log_rate": 1}
    priors = JointDistribution(*[variables[k] for k in sorted(variables)])
    model = numpyro_mixed_poisson_likelihood(
        sampler_fn, dist_fn, priors, variables, {}, vindex,
        event_log_likelihood_fns=Ls_broad,
        data_group=dg, log_ref_priors_group=lpg, masks_group=mg, N_pes=npes,
        pdet_fn=pdet_fn, variance_cut_threshold=None,
    )
    mcmc = MCMC(NUTS(model, init_strategy=init_to_median, target_accept_prob=0.8),
                num_warmup=args.num_warmup, num_samples=args.num_samples,
                progress_bar=True)
    mcmc.run(jax.random.PRNGKey(3), jnp.asarray(T_OBS))
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    truth = {"frac": FRAC_TRUE, "log_rate": LOGR_TRUE, "mu2": MU2_TRUE}
    print("\nMIXED-likelihood recovery (truth in brackets):")
    for k in ("mu2", "frac", "log_rate"):
        v = post[k]
        z = (v.mean() - truth[k]) / v.std()
        print(f"  {k:9s}: {v.mean():7.3f} +/- {v.std():.3f}   [{truth[k]:.3f}]  z={z:+.2f}")
    outdir = os.path.join(os.path.dirname(__file__), "figures_mixed")
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "mixed_post.npz"), **post,
             **{f"truth_{k}": truth[k] for k in truth},
             n_broad=bi.size, n_narrow=ni.size)
    print(f"wrote {outdir}/mixed_post.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
