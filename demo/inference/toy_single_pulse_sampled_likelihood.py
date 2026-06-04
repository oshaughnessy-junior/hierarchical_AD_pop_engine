"""Toy: single pulse + genie likelihoods + MC traces + full gwkokab path.

The first end-to-end exercise of the **sample-mode** Poisson likelihood
(gwkokab fork branch ``feature/sampled-poisson-likelihood``): the
population model enters ONLY as weighted merger draws from the engine's
JAX particle MC — no grid, no density, no KDE — and each observed event
enters ONLY as a per-event log-likelihood evaluator (genie-style Gaussian
measurement model around a scattered observed center; see
``gwkokab.errors.truncated_normal_error`` for the realistic version).

Pipeline (``--mode all``):

1. ``sampler_fn(mu_natal, sigma_natal, log_rate)`` →
   ``(samples (n,4), log_w)`` via ``particle_mc_jax_soft`` (AD-smooth
   weights; fixed key; mergers columns 0–3 = (m1, m2, chi1, chi2)).
2. Synthetic catalog: weighted-resample ``--n-obs`` mergers from the
   TRUTH run (independent key), scatter each by the genie-style widths
   (fractional in mass, absolute in spin), build
   ``gaussian_event_log_likelihoods`` evaluators.
3. ``check``: log-likelihood at truth is finite; print the per-event ESS
   table (``ess_per_event`` diagnostic) — THE quantity that decides
   whether this approach survives narrow posteriors (ROS: broad BBH
   posteriors are forgiving; low-mass will be hard — see ``--narrow``).
4. ``scan``: 1-D profile of the sampled log-likelihood over ``mu_natal``
   (sigma, log_rate fixed at truth) — peak should bracket truth within
   MC noise. Figure → ``figures_toy/``.
5. ``nuts``: full ``numpyro_sampled_poisson_likelihood`` 3-parameter
   recovery (mu_natal, sigma_natal, log_rate). Workstation-scale by
   default knobs; reduce for smoke tests.

Requires on PYTHONPATH: engine ``src`` + repo root (red_team), and the
gwkokab FORK ``src`` (branch ``feature/sampled-poisson-likelihood``).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from red_team.backup_particle_mc import (  # noqa: E402
    MCConfig,
    particle_mc_jax_soft,
)
from gwkokab.inference.event_likelihoods import (  # noqa: E402
    gaussian_event_log_likelihoods,
)
from gwkokab.inference.poissonlikelihood_utils import (  # noqa: E402
    sampled_poisson_likelihood_fn,
)

# ── Truth + fixed physics (Stage-1-like single pulse) ─────────────────
# NOTE on the rate truth: the catalog is drawn at FIXED size --n-obs (we
# do not Poisson-draw the event count), so the only self-consistent rate
# truth is the one implied by the catalog: e^{log_rate}·T_obs = n_obs.
# main() therefore sets logr_true = log(n_obs / T_obs) at runtime; the
# module constant below is only the import-time fallback. Getting this
# wrong shows up as a cleanly-converged-but-biased log_rate (the Poisson
# term is doing its job — first sandbox run recovered log(12) against a
# claimed truth of log(60)).
MU_TRUE, SIGMA_TRUE, LOGR_TRUE = 8.0, 1.0, np.log(60.0)
M_MIN, M_MAX = 4.0, 80.0
SPIN_LOC, SPIN_SCALE = 0.0, 0.05
KERNEL_A, KERNEL_B = 0.0, 2.0
V_ESC = 300.0
F_DEP = 0.10
T_OBS = 1.0
SAMPLER_KEY = jax.random.PRNGKey(7)   # fixed → deterministic objective
CATALOG_KEY = 20260604                # independent of the sampler key


def make_sampler_fn(n_seed: int):
    """θ → (merger samples (n, 4), log-weights) with Σw = e^{log_rate}."""

    def sampler_fn(mu_natal, sigma_natal, log_rate):
        cfg = MCConfig(
            m_min=jnp.asarray(M_MIN), m_max=jnp.asarray(M_MAX),
            natal_spin_loc=jnp.asarray(SPIN_LOC),
            natal_spin_scale=jnp.asarray(SPIN_SCALE),
            kernel_a=jnp.asarray(KERNEL_A), kernel_b=jnp.asarray(KERNEL_B),
            v_esc=jnp.asarray(V_ESC),
            n_seed=int(n_seed),
            depletion_fraction=float(F_DEP),
            mu_natal=jnp.asarray(mu_natal),
            sigma_natal=jnp.asarray(sigma_natal),
        )
        out = particle_mc_jax_soft(cfg, SAMPLER_KEY)
        samples = out["mergers"][:, :4]            # (n, 4): m1 m2 chi1 chi2
        w_eff = out["w_mergers"] * out["step_active"]
        # Dead merger slots carry NaN coordinates (fixed-shape scan
        # padding). Their weights are ~0, but NaN × anything poisons the
        # logsumexp, so replace dead coords with a benign in-domain point.
        # jnp.where selects the constant branch for dead slots, which also
        # kills the NaN in the backward pass (no gradient flows through
        # the unselected branch).
        benign = jnp.asarray([10.0, 8.0, 0.0, 0.0])
        samples = jnp.where(jnp.isfinite(samples), samples, benign[None, :])
        # Floor (not where-guard) keeps the backward pass finite: the
        # logsumexp cotangent on a -690 log-weight underflows to exact 0
        # before it can meet the 1/w factor — see
        # feedback_where_guards_forward_not_backward.
        log_w = jnp.log(jnp.maximum(w_eff, 1e-300))
        log_w = log_w - jax.scipy.special.logsumexp(log_w) + log_rate
        return samples, log_w

    return sampler_fn


def draw_catalog(sampler_fn, n_obs: int, narrow: bool):
    """Weighted-resample mergers at truth + genie-style Gaussian scatter.

    Widths (broad BBH-like; ``--narrow`` shrinks 10× to demonstrate the
    ESS collapse expected for low-mass events): 10% fractional in m1/m2,
    0.2 absolute in chi1/chi2. The per-event likelihood evaluator uses
    the SAME widths (a self-consistent Gaussian measurement model — the
    toy stand-in for genie's truncated_normal_error / banana_error).
    """
    samples, log_w = sampler_fn(MU_TRUE, SIGMA_TRUE, LOGR_TRUE)
    x = np.asarray(samples)
    p = np.exp(np.asarray(log_w - jax.scipy.special.logsumexp(log_w)))
    rng = np.random.default_rng(CATALOG_KEY)
    idx = rng.choice(x.shape[0], size=n_obs, replace=True, p=p)
    truths = x[idx]                                  # (n_obs, 4)

    shrink = 0.1 if narrow else 1.0
    sig = np.stack(
        [0.10 * truths[:, 0], 0.10 * truths[:, 1],
         np.full(n_obs, 0.2), np.full(n_obs, 0.2)], axis=1,
    ) * shrink                                       # (n_obs, 4)
    observed = truths + sig * rng.standard_normal(truths.shape)
    # keep spins physical at the toy level
    observed[:, 2:] = np.clip(observed[:, 2:], -0.998, 0.998)

    covs = np.zeros((n_obs, 4, 4))
    covs[:, np.arange(4), np.arange(4)] = sig ** 2
    Ls = gaussian_event_log_likelihoods(
        jnp.asarray(observed), jnp.asarray(covs)
    )
    return Ls, truths, observed


def loglik(sampler_fn, Ls, mu, sigma=SIGMA_TRUE, logr=None):
    if logr is None:
        logr = LOGR_TRUE
    samples, log_w = sampler_fn(mu, sigma, logr)
    ll, diag = sampled_poisson_likelihood_fn(
        Ls, samples, log_w, None, jnp.asarray(T_OBS), None,
    )
    return ll, diag


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("check", "scan", "nuts", "all"),
                   default="check")
    p.add_argument("--n-seed", type=int, default=4000)
    p.add_argument("--n-obs", type=int, default=30)
    p.add_argument("--narrow", action="store_true",
                   help="shrink event widths 10x — the low-mass ESS-collapse "
                        "demonstrator")
    p.add_argument("--scan-points", type=int, default=11)
    p.add_argument("--num-warmup", type=int, default=300)
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--outdir", type=str,
                   default=os.path.join(os.path.dirname(__file__),
                                        "figures_toy"))
    args = p.parse_args()

    sampler_fn = make_sampler_fn(args.n_seed)
    # Rate truth implied by the fixed-size catalog (see module note).
    logr_true = float(np.log(args.n_obs / T_OBS))
    Ls, truths, observed = draw_catalog(sampler_fn, args.n_obs, args.narrow)
    print(f"catalog: {args.n_obs} events"
          f"{' (NARROW widths)' if args.narrow else ''}; "
          f"m1 range [{truths[:, 0].min():.1f}, {truths[:, 0].max():.1f}]; "
          f"implied log_rate truth {logr_true:.3f}")

    if args.mode in ("check", "all"):
        ll, diag = loglik(sampler_fn, Ls, MU_TRUE, logr=logr_true)
        ess = np.asarray(diag["ess_per_event"])
        print(f"\nlogL(truth) = {float(ll):.3f}  (finite: {np.isfinite(float(ll))})")
        print(f"ESS per event: min {ess.min():.1f} | median "
              f"{np.median(ess):.1f} | max {ess.max():.1f}  "
              f"(n_model_samples with nonzero weight ~ "
              f"{int(np.sum(np.asarray(sampler_fn(MU_TRUE, SIGMA_TRUE, LOGR_TRUE)[1]) > -600))})")
        n_low = int(np.sum(ess < 10.0))
        print(f"events with ESS < 10: {n_low}/{args.n_obs}"
              + ("  <-- ESS COLLAPSE (expected under --narrow): grow n_seed"
                 " or stratify natal proposals (M3a/M3b)" if n_low else ""))

    if args.mode in ("scan", "all"):
        mus = np.linspace(MU_TRUE - 1.5, MU_TRUE + 1.5, args.scan_points)
        lls = np.asarray(
            [float(loglik(sampler_fn, Ls, m, logr=logr_true)[0]) for m in mus]
        )
        best = mus[int(np.argmax(lls))]
        print(f"\nprofile scan over mu_natal: argmax {best:.3f} "
              f"(truth {MU_TRUE}); range Delta logL = "
              f"{lls.max() - lls.min():.1f}")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(mus, lls - lls.max(), "o-")
        ax.axvline(MU_TRUE, ls=":", color="k", label="truth")
        ax.axvline(best, ls="--", color="C3", label="argmax")
        ax.set_xlabel(r"$\mu_{\rm natal}$")
        ax.set_ylabel(r"$\Delta \log\mathcal{L}$ (sampled mode)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        os.makedirs(args.outdir, exist_ok=True)
        out = os.path.join(args.outdir, "toy_mu_profile.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")

    if args.mode in ("nuts", "all"):
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        from gwkokab.inference.numpyro_sampled_poisson_likelihood import (
            numpyro_sampled_poisson_likelihood,
        )
        from gwkokab.models.utils import JointDistribution

        variables = {
            "mu_natal": dist.Uniform(5.0, 12.0),
            "sigma_natal": dist.Uniform(0.3, 3.0),
            "log_rate": dist.Uniform(np.log(5.0), np.log(500.0)),
        }
        names = sorted(variables)  # numpyro/gwkokab sorted-order contract
        priors = JointDistribution(*[variables[k] for k in names])
        model = numpyro_sampled_poisson_likelihood(
            sampler_fn,
            priors,
            variables,
            {},
            {k: i for i, k in enumerate(names)},
            Ls,
            None,
            None,
        )
        mcmc = MCMC(NUTS(model), num_warmup=args.num_warmup,
                    num_samples=args.num_samples, progress_bar=True)
        mcmc.run(jax.random.PRNGKey(3), jnp.asarray(T_OBS))
        post = mcmc.get_samples()
        print("\nNUTS recovery (truth in brackets):")
        for k, t in (("mu_natal", MU_TRUE), ("sigma_natal", SIGMA_TRUE),
                     ("log_rate", logr_true)):
            v = np.asarray(post[k])
            print(f"  {k:12s}: {v.mean():.3f} +/- {v.std():.3f}   [{t:.3f}]")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        for ax, (k, t) in zip(axes, (("mu_natal", MU_TRUE),
                                     ("sigma_natal", SIGMA_TRUE),
                                     ("log_rate", logr_true))):
            ax.hist(np.asarray(post[k]), bins=30, density=True, alpha=0.7)
            ax.axvline(t, ls=":", color="k")
            ax.set_xlabel(k)
        fig.tight_layout()
        os.makedirs(args.outdir, exist_ok=True)
        out = os.path.join(args.outdir, "toy_nuts_posteriors.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
