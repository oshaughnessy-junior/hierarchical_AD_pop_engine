"""Production-scale sampled-likelihood test: GWTC-3-like PL+Peak injection.

Scales the toy (``toy_single_pulse_sampled_likelihood.py``) to a
production-shaped problem (ROS spec 2026-06-04):

* **Injection**: natal mass = smoothed power law + a single Gaussian pulse
  at 30 M⊙ (GWTC-3 PL+Peak-like: α=3.4, m_min=5, smoothing δ=5, λ_peak
  small), small natal spin (truncnorm scale 0.05). Mild cascade
  (f_dep=0.10) on top — the hierarchical tail.
* **Selection**: real-ish ``pdet(x) ∝ Mc^{15/6}`` (SNR-volume scaling),
  capped at 1 at a reference chirp mass. (gwkokab's neural/injection VT
  machinery needs trained-network / injection files not present here;
  swap ``pdet_fn`` when they are.)
* **Measurements**: genie-style Gaussian event likelihoods in
  ``(Mc, η, χ_eff)`` ONLY. The evaluator maps model samples
  ``(m1, m2, χ1, χ2) → (Mc, η, χ_eff)`` *inside* the likelihood — in
  sample mode this needs **no coordinate Jacobian** (the integration
  measure is the model's own samples), neatly sidestepping the
  PLAN_measurement_model "THE CATCH" that the density mode must solve.
* **Target**: ~100 detected events. **Recover**: rate, natal-mass shape
  (α, λ_peak, μ_peak, σ_peak), natal spin scale — 6 free parameters.

Natal PL+Peak WITHOUT engine surgery
------------------------------------
``sample_natals_jax`` accepts a caller proposal + ``cfg.weight_array``
multiplied into the base weights (both jnp-traceable). We draw FIXED
(θ-free) proposal samples once — masses log-uniform, spins uniform —
then per likelihood call set::

    weight_array(θ) = p_plpeak(m; θ) · p_spin(χ; θ)
                      / p_cfg(m, χ; fixed constants)

so total natal weight = target/proposal exactly, smooth in ALL free
parameters (the same fixed-proposal IS pattern as the engine's two-pulse
λ path). The cascade then runs on the weighted particles unchanged.
Promoting a first-class ``natal_family="pl_peak"`` into
``sampled_wiring`` is the follow-up once this pattern is validated.

Run::

    # sandbox smoke (~30 s)
    python demo/inference/production_plpeak_sampled_recovery.py --mode check \
        --n-seed 4000 --n-obs 40
    # workstation production
    python demo/inference/production_plpeak_sampled_recovery.py --mode nuts \
        --n-seed 50000 --n-obs 100 --num-warmup 500 --num-samples 1000
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm as jnorm

jax.config.update("jax_enable_x64", True)

from red_team.backup_particle_mc import (  # noqa: E402
    MCConfig,
    particle_mc_jax_soft,
    _natal_log_pdf_jax,
)
from gwkokab.inference.poissonlikelihood_utils import (  # noqa: E402
    sampled_poisson_likelihood_fn,
)

# ── Fixed physics + truth ─────────────────────────────────────────────
M_MIN, M_MAX = 5.0, 80.0          # natal mass support
DELTA_M = 5.0                     # GWTC-3-style low-mass smoothing length
ALPHA_TRUE = 3.4                  # PL slope (p ∝ m^-α)
LAMBDA_TRUE = 0.05                # fraction of natal mass density in the pulse
MU_PEAK_TRUE, SIGMA_PEAK_TRUE = 30.0, 3.0
SPIN_SCALE_TRUE = 0.05            # small natal spin (truncnorm [0,1], loc 0)
KERNEL_A, KERNEL_B = 0.0, 2.0
V_ESC = 300.0
F_DEP = 0.10
T_OBS = 1.0
MC_REF = 35.0                     # pdet = min((Mc/MC_REF)^{15/6}, 1)
SAMPLER_KEY = jax.random.PRNGKey(11)   # cascade key (fixed; deterministic)
PROPOSAL_KEY = jax.random.PRNGKey(17)  # natal proposal draws (fixed)
SPIN_ORIENT_KEY = jax.random.PRNGKey(23)  # isotropic spin tilts (fixed)
CATALOG_SEED = 20260605

# Spin orientation for the OBSERVABLE chi_eff. The cascade tracks spin
# MAGNITUDES (1g small ~sigma_chi, 2g+ ~0.7); the detector-frame chi_eff needs
# orientations. ISOTROPIC_SPIN=True draws isotropic tilts (cos t ~ U[-1,1]) per
# component and forms the SIGNED aligned spin a_i*cos t_i, so chi_eff can be
# negative and its DISPERSION widens where the high-spin 2g+ population lives
# (the dynamical-channel hierarchical signature). Default False preserves the
# magnitude-only behaviour of the synthetic JOB A/E runs.
ISOTROPIC_SPIN = False

# Fixed cfg constants for the IS denominator (any broad valid values).
CFG_MU0, CFG_SIGMA0 = 20.0, 25.0
CFG_SPIN_LOC0, CFG_SPIN_SCALE0 = 0.0, 0.3

FREE_PARAMS = ("alpha", "lambda_peak", "mu_peak", "sigma_peak",
               "spin_scale", "log_rate")
TRUTH = dict(alpha=ALPHA_TRUE, lambda_peak=LAMBDA_TRUE,
             mu_peak=MU_PEAK_TRUE, sigma_peak=SIGMA_PEAK_TRUE,
             spin_scale=SPIN_SCALE_TRUE)  # log_rate implied by catalog


# ── Natal PL+Peak target density (differentiable in θ) ────────────────
_M_NORM_GRID = jnp.linspace(M_MIN, M_MAX, 4001)


def _smoothing(m):
    """GWTC-3-style Planck-taper turn-on over [M_MIN, M_MIN+DELTA_M].

    θ-free (M_MIN, DELTA_M fixed) — exponent clipped for float safety.
    """
    x = jnp.clip(m - M_MIN, 1e-9, DELTA_M - 1e-9)
    t = jnp.clip(DELTA_M / x + DELTA_M / (x - DELTA_M), -50.0, 50.0)
    s_mid = 1.0 / (1.0 + jnp.exp(t))
    return jnp.where(m < M_MIN, 0.0,
                     jnp.where(m >= M_MIN + DELTA_M, 1.0, s_mid))


def log_p_mass_plpeak(m, alpha, lambda_peak, mu_peak, sigma_peak):
    """log of the normalized smoothed-PL + Gaussian-peak natal mass pdf."""
    def _unnorm(mm):
        pl = _smoothing(mm) * mm ** (-alpha)
        peak = jnorm.pdf(mm, loc=mu_peak, scale=sigma_peak)
        return (1.0 - lambda_peak) * pl + lambda_peak * peak

    # separate normalisation of the PL piece (the peak is ~normalized on
    # the support already; normalise the MIXTURE numerically for honesty)
    vals = _unnorm(_M_NORM_GRID)
    dm = _M_NORM_GRID[1] - _M_NORM_GRID[0]
    Z = jnp.sum(0.5 * (vals[1:] + vals[:-1])) * dm
    return jnp.log(jnp.maximum(_unnorm(m), 1e-300)) - jnp.log(Z)


def log_p_spin(chi, spin_scale):
    """truncnorm([0, 1], loc=0, scale=s) log-pdf (differentiable in s)."""
    z = chi / spin_scale
    log_norm = jnp.log(
        jnorm.cdf(1.0 / spin_scale) - jnorm.cdf(0.0)
    )
    inside = (chi >= 0.0) & (chi <= 1.0)
    lp = jnorm.logpdf(z) - jnp.log(spin_scale) - log_norm
    return jnp.where(inside, lp, -jnp.inf)


# ── Fixed natal proposal (θ-free; drawn once) ─────────────────────────
# Proposal quality is what keeps the natal IS effective sample size (and
# hence the per-event ESS downstream) alive: a lazy log-uniform proposal
# against an α≈3.4 powerlaw loses ~80% of the seeds to weight variance
# before the cascade starts. Use a fixed MIXTURE matched to the problem
# family (the engine's two-pulse fixed-mixture-proposal pattern):
#   masses: 0.6·PL(α0=3.4) + 0.4·TruncNorm(30, 5)   (peak well populated)
#   spins:  0.7·TruncNorm(0, 0.08) + 0.3·Uniform[0,1] (small-spin bulk +
#           a floor so large recovered scales stay integrable)
# All θ-free; exact analytic q; drawn once with a fixed key.
_ALPHA0, _PL_FRAC = 3.4, 0.6
_MPK0, _SPK0 = 30.0, 5.0
_SPIN_S0, _SPIN_TN_FRAC = 0.08, 0.7


def _pl_logpdf(m, alpha0=_ALPHA0):
    z = (M_MIN ** (1.0 - alpha0) - M_MAX ** (1.0 - alpha0)) / (alpha0 - 1.0)
    return -alpha0 * jnp.log(m) - jnp.log(z)


def _tn_logpdf(x, loc, scale, lo, hi):
    norm = jnorm.cdf((hi - loc) / scale) - jnorm.cdf((lo - loc) / scale)
    return jnorm.logpdf((x - loc) / scale) - jnp.log(scale) - jnp.log(norm)


def make_proposal(n_seed: int):
    km1, km2, km3, kc1, kc2, kc3 = jax.random.split(PROPOSAL_KEY, 6)
    # masses: PL inverse-CDF or truncnorm, fixed component assignment
    pick_pl = jax.random.uniform(km1, (n_seed,)) < _PL_FRAC
    u = jax.random.uniform(km2, (n_seed,))
    one_m_a = 1.0 - _ALPHA0
    m_pl = (M_MIN ** one_m_a + u * (M_MAX ** one_m_a - M_MIN ** one_m_a)
            ) ** (1.0 / one_m_a)
    a = (M_MIN - _MPK0) / _SPK0
    b = (M_MAX - _MPK0) / _SPK0
    m_tn = _MPK0 + _SPK0 * jax.random.truncated_normal(km3, a, b, (n_seed,))
    m_prop = jnp.where(pick_pl, m_pl, m_tn)
    log_q_m = jnp.logaddexp(
        jnp.log(_PL_FRAC) + _pl_logpdf(m_prop),
        jnp.log(1.0 - _PL_FRAC) + _tn_logpdf(m_prop, _MPK0, _SPK0,
                                             M_MIN, M_MAX),
    )
    # spins: truncnorm bulk + uniform floor
    pick_tn = jax.random.uniform(kc1, (n_seed,)) < _SPIN_TN_FRAC
    c_tn = _SPIN_S0 * jax.random.truncated_normal(
        kc2, 0.0, 1.0 / _SPIN_S0, (n_seed,))
    c_u = jax.random.uniform(kc3, (n_seed,))
    chi_prop = jnp.where(pick_tn, c_tn, c_u)
    log_q_c = jnp.logaddexp(
        jnp.log(_SPIN_TN_FRAC) + _tn_logpdf(chi_prop, 0.0, _SPIN_S0,
                                            0.0, 1.0),
        jnp.log(1.0 - _SPIN_TN_FRAC),  # uniform on [0, 1]
    )
    return m_prop, chi_prop, log_q_m + log_q_c


# ── Coordinate map + genie-(Mc, η, χeff) event likelihoods ────────────
def mc_eta_chieff(samples):
    """(n, 4) (m1, m2, chi1, chi2) → (n, 3) (Mc, η, χ_eff). No Jacobian
    needed in sample mode — L_i is just a function of x."""
    m1, m2, c1, c2 = (samples[:, 0], samples[:, 1],
                      samples[:, 2], samples[:, 3])
    mt = m1 + m2
    eta = m1 * m2 / mt ** 2
    mc = mt * eta ** 0.6
    chieff = (m1 * c1 + m2 * c2) / mt
    return jnp.stack([mc, eta, chieff], axis=1)


class GenieMcEtaChieffLikelihood:
    """Per-event Gaussian log-likelihood in (Mc, η, χ_eff).

    Diagonal (``sigma``) by default; pass ``cov`` (3×3) for a full multivariate
    normal that uses the off-diagonal Mc–η–χ_eff correlations of the NAL fit.
    The per-event normalisation (incl. log-det) is θ-independent, hence
    irrelevant to the population posterior, but kept for cleanliness.
    """

    def __init__(self, center, sigma=None, cov=None):
        self.center = jnp.asarray(center)        # (3,)
        if cov is not None:
            cov = jnp.asarray(cov)               # (3,3)
            cov = 0.5 * (cov + cov.T) + 1e-12 * jnp.eye(cov.shape[0])  # PSD-safe
            self.prec = jnp.linalg.inv(cov)
            _, self.logdet = jnp.linalg.slogdet(cov)
            self.sigma = None
        else:
            self.sigma = jnp.asarray(sigma)      # (3,)
            self.prec = None

    def __call__(self, samples):
        z = mc_eta_chieff(samples) - self.center[None, :]    # (n,3)
        if self.prec is not None:
            q = jnp.einsum("ni,ij,nj->n", z, self.prec, z)   # Mahalanobis²
            return -0.5 * q - 0.5 * self.logdet - 1.5 * jnp.log(2.0 * jnp.pi)
        zz = z / self.sigma[None, :]
        return -0.5 * jnp.sum(zz ** 2, axis=1) - jnp.sum(
            jnp.log(self.sigma)) - 1.5 * jnp.log(2.0 * jnp.pi)


def genie_widths(mc, eta, chieff, narrow=False):
    """Genie-ish broad-BBH widths: 3% in Mc, 0.02 in η, 0.1 in χ_eff."""
    f = 0.1 if narrow else 1.0
    return np.stack([0.03 * np.asarray(mc),
                     np.full_like(np.asarray(mc), 0.02),
                     np.full_like(np.asarray(mc), 0.10)], axis=1) * f


# ── Selection ─────────────────────────────────────────────────────────
def pdet_fn(samples):
    """Real-ish SNR-volume selection: pdet = min((Mc/MC_REF)^{15/6}, 1)."""
    mc = mc_eta_chieff(samples)[:, 0]
    return jnp.minimum((mc / MC_REF) ** (15.0 / 6.0), 1.0)


# ── Sampler: PL+Peak natal via fixed proposal + IS weight_array ───────
def make_sampler_fn(n_seed: int):
    m_prop, chi_prop, log_q = make_proposal(n_seed)

    def proposal(_key):
        return m_prop, chi_prop, log_q

    # fixed-constant cfg pieces for the IS denominator
    def _cfg(weight_array):
        return MCConfig(
            m_min=jnp.asarray(M_MIN), m_max=jnp.asarray(M_MAX),
            natal_spin_loc=jnp.asarray(CFG_SPIN_LOC0),
            natal_spin_scale=jnp.asarray(CFG_SPIN_SCALE0),
            kernel_a=jnp.asarray(KERNEL_A), kernel_b=jnp.asarray(KERNEL_B),
            v_esc=jnp.asarray(V_ESC),
            n_seed=int(n_seed),
            depletion_fraction=float(F_DEP),
            mu_natal=jnp.asarray(CFG_MU0),
            sigma_natal=jnp.asarray(CFG_SIGMA0),
            proposal=proposal,
            weight_array=weight_array,
        )

    # denominator: cfg-family joint natal pdf at the proposal points —
    # computed ONCE (θ-free constants).
    cfg0 = _cfg(None)
    log_p_cfg = _natal_log_pdf_jax(m_prop, chi_prop, cfg0)

    def sampler_fn(alpha, lambda_peak, mu_peak, sigma_peak, spin_scale,
                   log_rate):
        log_p_target = (
            log_p_mass_plpeak(m_prop, alpha, lambda_peak, mu_peak, sigma_peak)
            + log_p_spin(chi_prop, spin_scale)
        )
        weight_array = jnp.exp(log_p_target - log_p_cfg)
        out = particle_mc_jax_soft(_cfg(weight_array), SAMPLER_KEY)
        samples = out["mergers"][:, :4]
        if ISOTROPIC_SPIN:
            # signed aligned spin a_i*cos(tilt_i), tilt isotropic (cos~U[-1,1]);
            # cols 2,3 hold spin MAGNITUDES out of the cascade -> make them signed.
            n = samples.shape[0]
            cost = jax.random.uniform(SPIN_ORIENT_KEY, (n, 2),
                                      minval=-1.0, maxval=1.0)
            samples = samples.at[:, 2].multiply(cost[:, 0])
            samples = samples.at[:, 3].multiply(cost[:, 1])
        w_eff = out["w_mergers"] * out["step_active"]
        benign = jnp.asarray([10.0, 8.0, 0.0, 0.0])
        samples = jnp.where(jnp.isfinite(samples), samples, benign[None, :])
        log_w = jnp.log(jnp.maximum(w_eff, 1e-300))
        log_w = log_w - logsumexp(log_w) + log_rate
        return samples, log_w

    return sampler_fn


# ── Catalog: detected events at truth ─────────────────────────────────
def draw_catalog(sampler_fn, n_obs, narrow):
    samples, log_w = sampler_fn(**TRUTH, log_rate=0.0)  # scale irrelevant
    x = np.asarray(samples)
    pd = np.asarray(pdet_fn(samples))
    w = np.exp(np.asarray(log_w - logsumexp(log_w)))
    p_det_w = w * pd
    mean_pdet = float(p_det_w.sum())            # ⟨pdet⟩ under the model
    p = p_det_w / p_det_w.sum()
    rng = np.random.default_rng(CATALOG_SEED)
    idx = rng.choice(x.shape[0], size=n_obs, replace=True, p=p)
    truths = x[idx]
    obs3 = np.asarray(mc_eta_chieff(jnp.asarray(truths)))
    sig = genie_widths(obs3[:, 0], obs3[:, 1], obs3[:, 2], narrow)
    observed = obs3 + sig * rng.standard_normal(obs3.shape)
    Ls = [GenieMcEtaChieffLikelihood(observed[i], sig[i])
          for i in range(n_obs)]
    # implied rate truth: mu(θ*) = e^{logR}·⟨pdet⟩·T_obs = n_obs
    logr_true = float(np.log(n_obs / T_OBS) - np.log(mean_pdet))
    return Ls, truths, observed, logr_true, mean_pdet


def loglik(sampler_fn, Ls, **theta):
    samples, log_w = sampler_fn(**theta)
    return sampled_poisson_likelihood_fn(
        Ls, samples, log_w, pdet_fn, jnp.asarray(T_OBS), None,
    )


def main() -> int:
    global TRUTH, ALPHA_TRUE, LAMBDA_TRUE, MU_PEAK_TRUE, SIGMA_PEAK_TRUE
    global SPIN_SCALE_TRUE, F_DEP, MC_REF, V_ESC
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("check", "scan", "nuts", "all",
                                      "plot-only"),
                   default="check")
    p.add_argument("--n-seed", type=int, default=4000)
    p.add_argument("--n-obs", type=int, default=40)
    p.add_argument("--narrow", action="store_true")
    p.add_argument("--scan-points", type=int, default=9)
    p.add_argument("--num-warmup", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--outdir", type=str,
                   default=os.path.join(os.path.dirname(__file__),
                                        "figures_plpeak"))
    # Injection knobs (defaults reproduce the committed GWTC-3-like run).
    # Override to inject a different population, e.g. a deeper hierarchical
    # cascade (--f-dep 0.5) with a flatter, detectable power law
    # (--alpha-true 2.5) and a more sensitive selection (--mc-ref 20).
    p.add_argument("--alpha-true", type=float, default=ALPHA_TRUE)
    p.add_argument("--lambda-true", type=float, default=LAMBDA_TRUE)
    p.add_argument("--mu-peak-true", type=float, default=MU_PEAK_TRUE)
    p.add_argument("--sigma-peak-true", type=float, default=SIGMA_PEAK_TRUE)
    p.add_argument("--spin-scale-true", type=float, default=SPIN_SCALE_TRUE)
    p.add_argument("--f-dep", type=float, default=F_DEP,
                   help="depletion fraction = cascade depth (0.1 shallow .. 0.85 deep)")
    p.add_argument("--mc-ref", type=float, default=MC_REF,
                   help="selection scale: pdet=min((Mc/mc_ref)^2.5,1); lower=more sensitive")
    p.add_argument("--v-esc", type=float, default=V_ESC)
    p.add_argument("--init-median", action="store_true",
                   help="init NUTS with init_to_median (robust warmup for "
                        "deeper-cascade populations)")
    p.add_argument("--max-tree-depth", type=int, default=0,
                   help="cap NUTS tree depth (e.g. 7) to bound per-iteration "
                        "cost on rough/deep-cascade likelihoods; 0 = default(10)")
    p.add_argument("--target-accept", type=float, default=0.0,
                   help="NUTS target acceptance (e.g. 0.7); 0 = default(0.8). "
                        "Lower lets a larger step skim a rough potential.")
    args = p.parse_args()

    # Apply injection overrides BEFORE building the sampler/selection (these
    # module globals are read at call time by make_sampler_fn / pdet_fn).
    ALPHA_TRUE, LAMBDA_TRUE = args.alpha_true, args.lambda_true
    MU_PEAK_TRUE, SIGMA_PEAK_TRUE = args.mu_peak_true, args.sigma_peak_true
    SPIN_SCALE_TRUE = args.spin_scale_true
    F_DEP, MC_REF, V_ESC = args.f_dep, args.mc_ref, args.v_esc
    TRUTH = dict(alpha=ALPHA_TRUE, lambda_peak=LAMBDA_TRUE,
                 mu_peak=MU_PEAK_TRUE, sigma_peak=SIGMA_PEAK_TRUE,
                 spin_scale=SPIN_SCALE_TRUE)
    print(f"injection: alpha={ALPHA_TRUE} lambda_peak={LAMBDA_TRUE} "
          f"mu_peak={MU_PEAK_TRUE} sigma_peak={SIGMA_PEAK_TRUE} "
          f"spin_scale={SPIN_SCALE_TRUE} | F_DEP={F_DEP} MC_REF={MC_REF} "
          f"V_ESC={V_ESC}")

    sampler_fn = make_sampler_fn(args.n_seed)
    Ls, truths, observed, logr_true, mean_pdet = draw_catalog(
        sampler_fn, args.n_obs, args.narrow)
    theta_true = dict(TRUTH, log_rate=logr_true)
    print(f"catalog: {args.n_obs} detected events"
          f"{' (NARROW)' if args.narrow else ''}; "
          f"Mc range [{observed[:, 0].min():.1f}, {observed[:, 0].max():.1f}]"
          f"; <pdet> = {mean_pdet:.3f}; implied log_rate truth "
          f"{logr_true:.3f}")

    if args.mode in ("check", "all"):
        ll, diag = loglik(sampler_fn, Ls, **theta_true)
        ess = np.asarray(diag["ess_per_event"])
        print(f"\nlogL(truth) = {float(ll):.3f} "
              f"(finite: {np.isfinite(float(ll))})")
        print(f"ESS per event: min {ess.min():.1f} | median "
              f"{np.median(ess):.1f} | max {ess.max():.1f}")
        n_low = int(np.sum(ess < 10))
        print(f"events with ESS < 10: {n_low}/{args.n_obs}")
        # gradient sanity on two physics knobs
        g = jax.grad(lambda a, lam: loglik(
            sampler_fn, Ls, alpha=a, lambda_peak=lam,
            mu_peak=MU_PEAK_TRUE, sigma_peak=SIGMA_PEAK_TRUE,
            spin_scale=SPIN_SCALE_TRUE, log_rate=logr_true)[0],
            argnums=(0, 1))(jnp.asarray(ALPHA_TRUE), jnp.asarray(LAMBDA_TRUE))
        print(f"grad logL: d/dalpha = {float(g[0]):.3f}, "
              f"d/dlambda_peak = {float(g[1]):.3f} "
              f"(finite: {all(np.isfinite(float(x)) for x in g)})")

    if args.mode in ("scan", "all"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        scans = {
            "alpha": np.linspace(2.2, 4.6, args.scan_points),
            "mu_peak": np.linspace(24.0, 36.0, args.scan_points),
        }
        fig, axes = plt.subplots(1, len(scans), figsize=(9.5, 3.6))
        for ax, (name, grid) in zip(np.atleast_1d(axes), scans.items()):
            lls = []
            for v in grid:
                th = dict(theta_true)
                th[name] = float(v)
                lls.append(float(loglik(sampler_fn, Ls, **th)[0]))
            lls = np.asarray(lls)
            best = grid[int(np.argmax(lls))]
            print(f"profile {name}: argmax {best:.3f} "
                  f"(truth {theta_true[name]:.3f})")
            ax.plot(grid, lls - lls.max(), "o-")
            ax.axvline(theta_true[name], ls=":", color="k")
            ax.axvline(best, ls="--", color="C3")
            ax.set_xlabel(name)
            ax.set_ylabel(r"$\Delta\log\mathcal{L}$")
        fig.tight_layout()
        os.makedirs(args.outdir, exist_ok=True)
        out = os.path.join(args.outdir, "plpeak_profiles.png")
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
            "alpha": dist.Uniform(1.5, 6.0),
            "lambda_peak": dist.Uniform(0.0, 0.3),
            "mu_peak": dist.Uniform(20.0, 45.0),
            "sigma_peak": dist.Uniform(1.0, 8.0),
            "spin_scale": dist.Uniform(0.01, 0.3),
            "log_rate": dist.Uniform(logr_true - 3.0, logr_true + 3.0),
        }
        names = sorted(variables)
        priors = JointDistribution(*[variables[k] for k in names])
        model = numpyro_sampled_poisson_likelihood(
            sampler_fn, priors, variables, {},
            {k: i for i, k in enumerate(names)},
            Ls, pdet_fn, None,
        )
        nuts_kw = {}
        if args.init_median:
            from numpyro.infer import init_to_median
            nuts_kw["init_strategy"] = init_to_median
        if args.max_tree_depth:
            nuts_kw["max_tree_depth"] = args.max_tree_depth
        if args.target_accept:
            nuts_kw["target_accept_prob"] = args.target_accept
        mcmc = MCMC(NUTS(model, **nuts_kw), num_warmup=args.num_warmup,
                    num_samples=args.num_samples, progress_bar=True)
        mcmc.run(jax.random.PRNGKey(5), jnp.asarray(T_OBS))
        post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        print("\nNUTS recovery (truth in brackets):")
        for k in names:
            v = post[k]
            print(f"  {k:12s}: {v.mean():.3f} +/- {v.std():.3f}   "
                  f"[{theta_true[k]:.3f}]")
        # ALWAYS persist the posterior — figures are then re-stylable
        # forever via --mode plot-only (lesson of the 2026-06-06 run,
        # which kept only the PNG).
        os.makedirs(args.outdir, exist_ok=True)
        np.savez(os.path.join(args.outdir, "plpeak_post.npz"),
                 **post, **{f"truth_{k}": theta_true[k] for k in names},
                 n_obs=args.n_obs, n_seed=args.n_seed,
                 f_dep=F_DEP, mc_ref=MC_REF, v_esc=V_ESC)
        _make_recovery_figures(post, theta_true, args.outdir)

    if args.mode == "plot-only":
        path = os.path.join(args.outdir, "plpeak_post.npz")
        if not os.path.exists(path):
            print(f"no cached posterior at {path} — run --mode nuts first")
            return 1
        with np.load(path) as z:
            names = sorted(k for k in z.files if not k.startswith("truth_")
                           and k not in ("n_obs", "n_seed"))
            post = {k: z[k] for k in names}
            theta_cached = {k: float(z[f"truth_{k}"]) for k in names}
        _make_recovery_figures(post, theta_cached, args.outdir)

    return 0


_CORNER_LABELS = {
    "alpha": r"$\alpha$",
    "lambda_peak": r"$\lambda_{\rm peak}$",
    "mu_peak": r"$\mu_{\rm peak}\ [M_\odot]$",
    "sigma_peak": r"$\sigma_{\rm peak}\ [M_\odot]$",
    "spin_scale": r"$\sigma_\chi$",
    "log_rate": r"$\log R$",
}
_CORNER_ORDER = ("alpha", "lambda_peak", "mu_peak", "sigma_peak",
                 "spin_scale", "log_rate")


def _make_recovery_figures(post, theta_true, outdir):
    """Marginal strip + pretty corner from a posterior dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    names = [k for k in _CORNER_ORDER if k in post]
    # marginal strip (the quick-look figure)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, k in zip(axes.ravel(), names):
        ax.hist(post[k], bins=30, density=True, alpha=0.7)
        ax.axvline(theta_true[k], ls=":", color="k")
        ax.set_xlabel(_CORNER_LABELS.get(k, k))
    fig.tight_layout()
    out = os.path.join(outdir, "plpeak_nuts_posteriors.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")

    # pretty corner (the paper figure)
    try:
        import corner
    except ImportError:
        print("corner not installed — skipping plpeak_corner.png")
        return
    data = np.stack([post[k] for k in names], axis=1)
    fig = corner.corner(
        data,
        labels=[_CORNER_LABELS.get(k, k) for k in names],
        truths=[theta_true[k] for k in names],
        truth_color="C3",
        quantiles=(0.16, 0.5, 0.84),
        show_titles=True,
        title_fmt=".2f",
        title_kwargs=dict(fontsize=9),
        label_kwargs=dict(fontsize=11),
        smooth=0.9,
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.39, 0.86),  # 1- and 2-sigma in 2D
        color="C0",
    )
    out = os.path.join(outdir, "plpeak_corner.png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
