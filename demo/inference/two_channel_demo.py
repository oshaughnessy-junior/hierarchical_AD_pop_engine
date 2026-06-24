"""Two-channel mixture prototype (route a: combined sampler).

Channel A = Gaussian-seeded (lambda_peak=1) ISOTROPIC-spin coagulation cascade
[engine sampler]. Channel B = aligned Gaussian, NO cascade [drawn directly].
The combined sampler concatenates {x^A, ln(w^A)} and {x^B, ln(R_B/N_B)} so the
existing sampled_poisson_likelihood machinery sees one mixture population with
two independent rates.

This file (mode=check) is the FORWARD de-risk: verify the combined population has
the intended structure -- low mass = narrow, partially aligned (Channel B + the
cascade's 1g survivors); high mass = wide, isotropic chi_eff (the cascade's 2g+
remnants). That mass-dependent spin width, with no per-object orientation law, is
the whole point.

  python two_channel_demo.py --mode check
"""
import argparse, importlib.util, os, numpy as np, jax, jax.numpy as jnp
from jax.scipy.special import logsumexp

_HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location(
    "plpeak", os.path.join(_HERE, "production_plpeak_sampled_recovery.py"))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
M.ISOTROPIC_SPIN = True                       # Channel A: isotropic cascade spins

B_KEY = jax.random.PRNGKey(101)


def make_channelB_sampler(n_b):
    """Aligned Gaussian channel (no cascade): m1,m2 ~ N(mu_B,sig_B) (m1>=m2),
    aligned spins c1=c2=chi_eff_B ~ N(mu_chiB,sig_chiB). Exact draws -> each
    carries weight R_B/N_B (log_w = log_R_B - log n_b)."""
    z_m = jax.random.normal(B_KEY, (n_b, 2))
    z_chi = jax.random.normal(jax.random.fold_in(B_KEY, 1), (n_b,))

    def sampler_B(mu_B, sig_B, mu_chiB, sig_chiB, log_R_B):
        m = mu_B + sig_B * z_m
        m1 = jnp.maximum(m[:, 0], m[:, 1]); m2 = jnp.minimum(m[:, 0], m[:, 1])
        m1 = jnp.clip(m1, M.M_MIN * 1.01, M.M_MAX * 0.99)
        m2 = jnp.clip(m2, M.M_MIN * 1.01, M.M_MAX * 0.99)
        chi = jnp.clip(mu_chiB + sig_chiB * z_chi, -0.99, 0.99)   # aligned chi_eff
        s = jnp.stack([m1, m2, chi, chi], axis=1)                 # c1=c2=chi_eff
        log_w = jnp.full((n_b,), log_R_B - jnp.log(float(n_b)))
        return s, log_w
    return sampler_B


def make_combined_sampler(n_seed_A, n_b):
    samp_A = M.make_sampler_fn(n_seed_A)        # cascade (lambda_peak=1 -> Gaussian seed)
    samp_B = make_channelB_sampler(n_b)

    def combined(mu_A, sig_A, spin_chi, fdep_unused,
                 mu_B, sig_B, mu_chiB, sig_chiB, log_R_A, log_R_B):
        sA, lwA = samp_A(3.0, 1.0, mu_A, sig_A, spin_chi, log_R_A)   # lambda_peak=1
        sB, lwB = samp_B(mu_B, sig_B, mu_chiB, sig_chiB, log_R_B)
        return jnp.concatenate([sA, sB], 0), jnp.concatenate([lwA, lwB], 0), sA.shape[0]
    return combined


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="check")
    p.add_argument("--f-dep", type=float, default=0.30)
    p.add_argument("--n-seed", type=int, default=20000)
    p.add_argument("--n-b", type=int, default=8000)
    args = p.parse_args()
    M.F_DEP = args.f_dep

    comb = make_combined_sampler(args.n_seed, args.n_b)
    # fiducial truth: Channel A natal Gaussian seed at mu_A=10, deep cascade;
    # Channel B aligned Gaussian at mu_B=10 (the hopeful coincidence), equal rates.
    th = dict(mu_A=10.0, sig_A=1.5, spin_chi=0.05, fdep_unused=0.0,
              mu_B=10.0, sig_B=1.5, mu_chiB=0.15, sig_chiB=0.05,
              log_R_A=np.log(0.5), log_R_B=np.log(0.5))
    s, lw, nA = comb(**th)
    s = np.asarray(s); w = np.exp(np.asarray(lw))           # rate weights (sum ~ R_A+R_B)
    m1, m2, c1, c2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    chie = (m1 * c1 + m2 * c2) / (m1 + m2)
    isA = np.zeros(len(s), bool); isA[:nA] = True
    print(f"combined draws: {nA} cascade(A) + {len(s)-nA} gaussian(B)")
    print(f"weight sums: R_A={w[isA].sum():.3f} (inj 0.5)  R_B={w[~isA].sum():.3f} (inj 0.5)")
    print("\n  Mc bin   <chi_eff>   sd(chi_eff)   B-fraction   (intrinsic, rate-weighted)")
    for lo, hi in [(4, 9), (9, 14), (14, 22), (22, 35), (35, 60)]:
        m = (mc >= lo) & (mc < hi); ww = w[m]
        if ww.sum() > 1e-6:
            wn = ww / ww.sum(); mu = np.sum(wn * chie[m])
            sd = np.sqrt(np.sum(wn * (chie[m] - mu) ** 2))
            bfrac = w[m & ~isA].sum() / ww.sum()
            print(f"  [{lo:2d},{hi:2d})   {mu:+.3f}      {sd:.3f}        {bfrac:.2f}")
    print("\nEXPECT: low-Mc -> aligned (<chi_eff> > 0, narrow, B-frac high); "
          "high-Mc -> isotropic (<chi_eff> ~ 0, wide, B-frac low).")


if __name__ == "__main__":
    raise SystemExit(main())
