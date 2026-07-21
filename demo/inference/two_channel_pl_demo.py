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
# The opt-in EXCHANGE / STE / soft-retention knobs live as module globals on
# red_team.backup_particle_mc (read at trace time inside particle_mc_jax_soft).
# M re-exports the *function* from that module, so the globals must be set on
# the module object itself.  Defaults leave everything OFF (bit-for-bit the
# committed engine); --exchange-mode / --free-v-esc below flip them on.
import red_team.backup_particle_mc as bpm
M.ISOTROPIC_SPIN = True
M.M_MAX = 70.0                      # WIDE proposal cap = top of the m_max prior;
                                    # the natal cutoff itself is a FREE parameter
                                    # (smooth target cutoff below), so the abundant
                                    # high-mass population can be NATAL, not 2g.
M_MAX_PROP = 70.0                   # proposal upper mass bound (must span the prior)
_CUT_W = 1.5                        # smoothing width of the high-mass natal cutoff
_NORM_GRID = jnp.linspace(M.M_MIN, M_MAX_PROP, 4001)

# ---- opt-in ALIGNED-SEED cascade channel (notes 58 / 59 Part B) ------------
# Replaces the non-cascading channel-B aligned Gaussian with the REAL thing: the
# aligned low-mass field IS the seed pool of an exchange cascade, so its 2g is
# MC-generated (masses sum, Barausse-Rezzolla remnant chi~0.7, tilts dynamical).
# chi_eff is computed through the DIFFERENTIABLE generation-aware tilt-moment path
# (bpm.generation_tilt_chieff_moments, Prototype 6) with a FROZEN standard-normal
# reparametrisation draw, NOT the hard random cos-tilt map -- so the fit gets spin
# gradients (note 58 §4 blocker, solved).  All OFF unless --aligned-seed is set.
ALIGNED_SEED = False                 # master switch (set by --aligned-seed)
AS_SEED_LO, AS_SEED_HI = 5.0, 45.0   # aligned-field natal-mass truncation [Msun]
AS_TWOG_MEMORY = 0.0                 # 2g tilt retained-alignment fraction (0 = companion isotropic)
AS_F_DEP = 0.5                       # aligned-seed cascade DEPTH (note 58 value; separate from
                                     # channel A's --f-dep so a deep A cascade does not over-
                                     # cascade the low-mass seeds and wash out the 1g aligned feature)
AS_Z_KEY = jax.random.PRNGKey(858)   # FROZEN tilt-reparam normal draws (deterministic in theta)


def _tn_logpdf_trunc(x, loc, sc, lo, hi):
    """truncnorm([lo,hi], loc, sc) log-pdf, -inf outside [lo,hi] (masked)."""
    from jax.scipy.stats import norm as _jn
    z = (x - loc) / sc
    norm = _jn.cdf((hi - loc) / sc) - _jn.cdf((lo - loc) / sc)
    inside = (x >= lo) & (x <= hi)
    return jnp.where(inside, _jn.logpdf(z) - jnp.log(sc) - jnp.log(norm), -jnp.inf)


# ---- opt-in POWER-LAW + LOW-MASS-PEAK natal for channel A (driven wide-natal demo).
# The steadily-seeded (driven) mode re-injects the natal reservoir each step in
# proportion to the natal weights, so the natal TARGET density IS the shape of the
# source r(x).  Adding a low-mass Gaussian peak to the channel-A natal therefore feeds
# a "power-law + peak" source with the peak at LOW mass -- letting us test how injecting
# low-mass STRUCTURE (vs the flat power law) affects the pairing.  NATAL_PEAK_FRAC = 0
# is the identity (pure power law, bit-for-bit the committed target); the Python-level
# guard keeps the traced graph unchanged.  Set BEFORE the first trace.
NATAL_PEAK_FRAC = 0.0                 # weight of the low-mass peak (0 = pure PL, committed)
NATAL_PEAK_MU = 10.0                  # peak centre [Msun] (low-mass, cf. the ~30 obs peak)
NATAL_PEAK_SIG = 2.0                  # peak width [Msun]


def _log_p_mass_pl_cutoff(m, alpha, m_max):
    """Pure power law (GWTC-3 low-mass Planck taper) with a SMOOTH, differentiable
    high-mass cutoff at the free natal edge m_max; normalized over the fixed grid.
    With NATAL_PEAK_FRAC > 0 a normalized low-mass Gaussian peak is mixed in (opt-in
    wide-natal source for the driven demo); FRAC = 0 is bit-for-bit the pure PL."""
    def unnorm(mm):
        pl = M._smoothing(mm) * mm ** (-alpha)
        hi = jax.nn.sigmoid(-(mm - m_max) / _CUT_W)      # ~1 below m_max, ->0 above
        return pl * hi
    dm = _NORM_GRID[1] - _NORM_GRID[0]
    norm = jnp.sum(unnorm(_NORM_GRID)) * dm
    log_pl = jnp.log(unnorm(m) + 1e-300) - jnp.log(norm)
    if NATAL_PEAK_FRAC <= 0.0:
        return log_pl
    # normalized low-mass Gaussian peak (truncated to the same grid support), mixed
    # with weight NATAL_PEAK_FRAC.  logaddexp keeps it in log space + differentiable.
    def gauss(mm):
        return jnp.exp(-0.5 * ((mm - NATAL_PEAK_MU) / NATAL_PEAK_SIG) ** 2)
    norm_pk = jnp.sum(gauss(_NORM_GRID)) * dm
    log_pk = jnp.log(gauss(m) + 1e-300) - jnp.log(norm_pk)
    w = NATAL_PEAK_FRAC
    return jnp.logaddexp(jnp.log(1.0 - w) + log_pl, jnp.log(w) + log_pk)


def make_sampler_A_mmax(n_seed):
    """Channel-A sampler: isotropic power-law natal with a FREE high-mass cutoff
    m_max, feeding the (faint) cascade. Replicates M.make_sampler_fn's IS machinery
    but threads a differentiable m_max through the target density."""
    m_prop, chi_prop, log_q = M.make_proposal(n_seed)

    def proposal(_key):
        return m_prop, chi_prop, log_q

    def _cfg(weight_array, kernel_a=M.KERNEL_A, kernel_b=M.KERNEL_B, v_esc=M.V_ESC):
        return M.MCConfig(
            m_min=jnp.asarray(M.M_MIN), m_max=jnp.asarray(M_MAX_PROP),
            natal_spin_loc=jnp.asarray(M.CFG_SPIN_LOC0),
            natal_spin_scale=jnp.asarray(M.CFG_SPIN_SCALE0),
            kernel_a=jnp.asarray(kernel_a), kernel_b=jnp.asarray(kernel_b),
            v_esc=jnp.asarray(v_esc), n_seed=int(n_seed),
            depletion_fraction=float(M.F_DEP),
            mu_natal=jnp.asarray(M.CFG_MU0), sigma_natal=jnp.asarray(M.CFG_SIGMA0),
            proposal=proposal, weight_array=weight_array)

    # natal IS denominator is kernel-independent -> fixed cfg0; n_steps (scan
    # length) is computed once from depletion_fraction * n_init * k_ratio(=1.5),
    # so the traced pairing kernel does NOT change the static scan length.
    # v_esc likewise NEVER changes the scan length (it only gates retention).
    log_p_cfg = M._natal_log_pdf_jax(m_prop, chi_prop, _cfg(None))

    def sampler_A(alpha, m_max, spin_scale, kernel_a, kernel_b, log_rate, v_esc):
        log_p_target = (_log_p_mass_pl_cutoff(m_prop, alpha, m_max)
                        + M.log_p_spin(chi_prop, spin_scale))
        weight_array = jnp.exp(log_p_target - log_p_cfg)
        out = M.particle_mc_jax_soft(
            _cfg(weight_array, kernel_a, kernel_b, v_esc), M.SAMPLER_KEY)
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


def make_sampler_alignedseed(n_seed):
    """Channel-B REPLACEMENT (--aligned-seed): the aligned low-mass Gaussian is the
    SEED pool of a real exchange cascade; chi_eff via the smooth, differentiable
    generation-aware tilt moments (Prototype 6) with a frozen-z reparametrisation.

    Closure signature mirrors tc.make_channelB_sampler's
    (mu_B, sig_B, mu_chiB, sig_chiB, log_R_B) so the numpyro **params wiring and the
    variables dict are UNCHANGED -- but the four are reinterpreted as the aligned
    SEED: mass ~ truncnorm(mu_B, sig_B)[lo,hi], natal spin MAGNITUDE ~ truncnorm
    (mu_chiB, sig_chiB)[0,1] (aligned -> +1 tilt at 1g).  Takes v_esc as a 6th arg
    (fixed; hard/soft retention set by the caller like channel A)."""
    m_prop, chi_prop, log_q = M.make_proposal(n_seed)

    def _cfg(weight_array, v_esc=M.V_ESC):
        return M.MCConfig(
            m_min=jnp.asarray(M.M_MIN), m_max=jnp.asarray(M_MAX_PROP),
            natal_spin_loc=jnp.asarray(M.CFG_SPIN_LOC0),
            natal_spin_scale=jnp.asarray(M.CFG_SPIN_SCALE0),
            kernel_a=jnp.asarray(M.KERNEL_A), kernel_b=jnp.asarray(M.KERNEL_B),
            v_esc=jnp.asarray(v_esc), n_seed=int(n_seed),
            depletion_fraction=float(AS_F_DEP),
            mu_natal=jnp.asarray(M.CFG_MU0), sigma_natal=jnp.asarray(M.CFG_SIGMA0),
            proposal=lambda _k: (m_prop, chi_prop, log_q),
            weight_array=weight_array)

    # natal IS denominator is theta-free (kernel/v_esc-independent) -> fixed cfg0.
    log_p_cfg = M._natal_log_pdf_jax(m_prop, chi_prop, _cfg(None))

    def sampler_alignedseed(seed_mu, seed_sig, chi_mu, chi_sig, log_R, v_esc=M.V_ESC):
        # aligned-field target: truncnorm mass x truncnorm aligned spin MAGNITUDE.
        log_p_target = (_tn_logpdf_trunc(m_prop, seed_mu, seed_sig, AS_SEED_LO, AS_SEED_HI)
                        + _tn_logpdf_trunc(chi_prop, chi_mu, chi_sig, 0.0, 1.0))
        weight_array = jnp.exp(log_p_target - log_p_cfg)
        out = M.particle_mc_jax_soft(_cfg(weight_array, v_esc), M.SAMPLER_KEY)
        mg = out["mergers"]                       # (N,8): m1,m2,chi1,chi2,g1,g2,ret,mrem
        m1, m2 = mg[:, 0], mg[:, 1]
        a1, a2 = mg[:, 2], mg[:, 3]               # PRE-tilt spin MAGNITUDES out of cascade
        g1, g2 = mg[:, 4], mg[:, 5]               # generations
        # DIFFERENTIABLE generation-aware tilt: analytic (E[chieff], Var[chieff]) then a
        # FROZEN normal draw -> a smooth chi_eff observable (reproduces the hard random
        # cos-tilt map to MC noise, note 59 §5) that carries the spin/mass gradients.
        mu_ce, var_ce = bpm.generation_tilt_chieff_moments(
            m1, m2, a1, a2, g1, g2, AS_TWOG_MEMORY)
        z = jax.random.normal(AS_Z_KEY, (mg.shape[0],))     # mg.shape[0] static at trace
        chieff = jnp.clip(mu_ce + jnp.sqrt(jnp.maximum(var_ce, 0.0)) * z, -0.999, 0.999)
        # store the SAME chieff in both spin cols: downstream
        # chi_eff = (m1 c1 + m2 c2)/(m1+m2) == chieff regardless of masses.
        samples = jnp.stack([m1, m2, chieff, chieff], axis=1)
        benign = jnp.asarray([10.0, 8.0, 0.0, 0.0])
        samples = jnp.where(jnp.isfinite(samples), samples, benign[None, :])
        w_eff = out["w_mergers"] * out["step_active"]
        log_w = jnp.log(jnp.maximum(w_eff, 1e-300))
        log_w = log_w - logsumexp(log_w) + log_R
        return samples, log_w

    return sampler_alignedseed


def make_combined_pl(n_seed_A, n_b):
    samp_A = make_sampler_A_mmax(n_seed_A)       # power-law natal, FREE m_max, cascade
    # channel B: non-cascading aligned Gaussian (default) OR the REAL aligned-seed
    # exchange cascade with differentiable tilt-moment chi_eff (--aligned-seed).
    samp_B = (make_sampler_alignedseed(n_b) if ALIGNED_SEED
              else tc.make_channelB_sampler(n_b))  # aligned Gaussian, no cascade

    # NOTE: kernel_a/kernel_b are kept fixed (M.KERNEL_A/B). Freeing them was a
    # confirmed dead end -- a resolved (16k-seed) sweep shows natal pairing cannot
    # make the equal-mass high-mass (Mc>40) tail at ANY (a,b) (the tail is too rare
    # to resolve; gradient ~0). The high-mass tail is instead the genuine
    # low-spin population that disfavors 2g hierarchical formation (twog_exclusion.py).
    # ``v_esc`` is threaded through channel A only (channel B has no cascade).
    # It arrives either as a fixed constant (default / v_esc scan) or as a
    # sampled tracer (--free-v-esc, which also flips RETAIN_SOFT on so
    # d(logL)/d(v_esc) flows).  Keyword-only so numpyro's **params call maps it.
    # sigq_2g / gamma_2g: the generation-gated degenerate 2g kernel params
    # (Prototype 5).  Passed as kwargs so the fit can SAMPLE them (free run) or
    # pin them (anchor / constants) — both routed here by numpyro as **params.
    # Threaded onto the engine module globals (which particle_mc_jax_soft reads
    # at trace time); a JAX TRACER flows through and gradients back-propagate
    # (the boost is smooth, no STE needed).  None = leave the pre-set globals
    # (bit-for-bit unchanged, e.g. the check/data-gen probes that omit them).
    # kernel_b1: the 1g pairing mass-ratio exponent (note 60 family C freed it
    # NEGATIVE = unequal-preferring at low mass; the old fits pinned it +2).  It
    # is the BASE kernel q^b applied to every pairing; the 2g boost multiplies on
    # top, so the 2g effective q-power is (kernel_b1 + KERNEL_2G_B).  For family C
    # the 2g sector is degenerate-EQUAL (a Gaussian in q, net q-power ~0), so when
    # both the 2g kernel and a free/negative b1 are on we set KERNEL_2G_B = -b1 so
    # the 2g net q-power is 0 (the boost cancels the base q^b1 for >=2g pairs and
    # leaves exp[-(1-q)^2/2sigq^2]*(M/M*)^gamma).  None = keep the pinned M.KERNEL_B.
    def combined(*, alpha_A, m_max_A, spin_chi, mu_B, sig_B, mu_chiB, sig_chiB,
                 log_R_A, log_R_B, v_esc, sigq_2g=None, gamma_2g=None, kernel_b1=None):
        if sigq_2g is not None:
            bpm.KERNEL_2G_SIGQ = sigq_2g
        if gamma_2g is not None:
            bpm.KERNEL_2G_GAMMA = gamma_2g
        kb = M.KERNEL_B if kernel_b1 is None else kernel_b1
        if kernel_b1 is not None and bpm.KERNEL_2G_ON:
            bpm.KERNEL_2G_B = -kernel_b1          # 2g net q-power 0 (family C)
        sA, lwA = samp_A(alpha_A, m_max_A, spin_chi, M.KERNEL_A, kb, log_R_A, v_esc)
        if ALIGNED_SEED:
            sB, lwB = samp_B(mu_B, sig_B, mu_chiB, sig_chiB, log_R_B, v_esc)
        else:
            sB, lwB = samp_B(mu_B, sig_B, mu_chiB, sig_chiB, log_R_B)
        return jnp.concatenate([sA, sB], 0), jnp.concatenate([lwA, lwB], 0)
    sA0, _ = samp_A(2.0, 50.0, 0.1, M.KERNEL_A, M.KERNEL_B, 0.0, M.V_ESC)
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
    p.add_argument("--fix-alpha-a", type=float, default=None,
                   help="pin the natal interior slope alpha_A to this value (not sampled); "
                        "with the corrected selection grid the free alpha rails at ~flat, so "
                        "the headline run fixes alpha_A=0 (flat natal, cascade-set high mass)")
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--log-r-width", type=float, default=3.0,
                   help="half-width of the log_R_A/log_R_B Uniform priors about logr0=log(n_obs/T_obs/2). "
                        "Default 3 (=exp(3)~=20x). Widen (e.g. 6) for the flat-natal + corrected-grid fits: "
                        "the under-detectable heavy natal population drives the rate MLE (mu=n_obs) above the "
                        "logr0+3 ceiling, so log_R rails and the count is under-predicted. A wider ceiling lets "
                        "log_R reach mu=n_obs (rate absorbs the low detectability; absolute R0 still from vt_calibration).")
    # ---- opt-in binary-single EXCHANGE (defaults OFF = committed behaviour) ----
    p.add_argument("--exchange-mode", type=int, default=0,
                   help="0=off (committed), 1=HHM96 smooth (preferred for inference), "
                        "2=RAPSTER prompt rule. Sets red_team.backup_particle_mc.EXCHANGE_MODE.")
    p.add_argument("--exchange-c", type=float, default=1.0,
                   help="exchange prefactor (mode 1) / prompt-exchange probability (mode 2). "
                        "FIXED, not sampled: the STE gradient in C is biased (can flip sign vs FD), "
                        "so scan C on a grid rather than gradient-fit it.")
    p.add_argument("--exchange-intruder-n", type=float, default=1.0,
                   help="intruder draw exponent p ~ occ*m^n (RAPSTER draws exchange singles ~ m).")
    p.add_argument("--segregation-n", type=float, default=0.0,
                   help="pair-draw mass-segregation exponent (keep near 0; n>=1 overshoots high mass).")
    # ---- opt-in GENERATION-GATED DEGENERATE 2g KERNEL (Prototype 5) ----------
    # Boost q^b2*exp[-(1-q)^2/2sigq^2]*(M/M*)^gamma for pairs with a >=2g body
    # (notes/forward_model_builds/57). sigma_q + gamma are SMOOTH (gradients
    # flow, no STE) so they can be FREE sampled params or FIXED (anchor).
    p.add_argument("--kernel-2g", action="store_true",
                   help="turn on the generation-gated degenerate 2g kernel "
                        "(sets bpm.KERNEL_2G_ON). Requires --exchange-mode 1 (seeder).")
    p.add_argument("--kernel-2g-sigq", type=float, default=0.17,
                   help="sigma_q (equal-mass width); ANCHOR value / free-run init. Shooter adam 0.17.")
    p.add_argument("--kernel-2g-gamma", type=float, default=1.07,
                   help="gamma (IMF-undoing total-mass exponent); ANCHOR value / free-run init. Shooter adam 1.07.")
    p.add_argument("--kernel-2g-b", type=float, default=0.0,
                   help="b2 (extra q-tilt prefactor exponent); FIXED (default 0 = shooter net q-law).")
    p.add_argument("--kernel-2g-mstar", type=float, default=40.0,
                   help="M* total-mass pivot; FIXED (shooter M*=40).")
    p.add_argument("--free-2g-kernel", action="store_true",
                   help="SAMPLE sigma_q ~ LogUniform[sigq-min,sigq-max] and gamma ~ "
                        "Uniform[gamma-min,gamma-max] as free parameters (both smooth).")
    p.add_argument("--sigq-min", type=float, default=0.05)
    p.add_argument("--sigq-max", type=float, default=1.0)
    p.add_argument("--gamma-min", type=float, default=0.0)
    p.add_argument("--gamma-max", type=float, default=3.0)
    # ---- 1g pairing exponent b1 (note 60 family C: free it NEGATIVE) ----------
    p.add_argument("--base-b", type=float, default=None,
                   help="FIXED 1g pairing mass-ratio exponent b1 (sets M.KERNEL_B). "
                        "Note 60 family C anchor uses -0.5 (unequal-preferring); "
                        "committed fits used +2. None = keep M.KERNEL_B default.")
    p.add_argument("--free-base-b", action="store_true",
                   help="SAMPLE the 1g exponent b1 ~ Uniform[base-b-min,base-b-max] "
                        "(prior EXTENDS NEGATIVE per note 60). Watch mu / step size: "
                        "b1<0 reweights the abundant low-mass pairings.")
    p.add_argument("--base-b-min", type=float, default=-2.0)
    p.add_argument("--base-b-max", type=float, default=3.0)
    # ---- v_esc: FREE sampled retention parameter, or FIXED for a scan ---------
    p.add_argument("--free-v-esc", action="store_true",
                   help="sample v_esc as a free (log-uniform) parameter; also enables "
                        "RETAIN_SOFT (sigmoid retention) so d(logL)/d(v_esc) flows.")
    p.add_argument("--v-esc", type=float, default=float(M.V_ESC),
                   help="fixed v_esc [km/s] when --free-v-esc is NOT set (enables a v_esc scan: "
                        "run 300/600/1000 as separate jobs). Default = committed M.V_ESC.")
    p.add_argument("--v-esc-min", type=float, default=100.0, help="free-v_esc log-uniform prior lower [km/s].")
    p.add_argument("--v-esc-max", type=float, default=2000.0, help="free-v_esc log-uniform prior upper [km/s].")
    p.add_argument("--retain-tau", type=float, default=50.0,
                   help="sigmoid retention softness [km/s] (RETAIN_SOFT / non-aligned paths only; "
                        "default 50 = committed value). Widen to ~100-150 if NUTS step-size "
                        "collapses in the v_esc direction (rough IS-weighted likelihood at "
                        "modest n_seed; observed 2026-07-15, adapted step 1.9e-07).")
    # ---- opt-in ALIGNED-SEED channel B (notes 58 / 59 Part B) -----------------
    # Channel B becomes a REAL aligned-low-mass-seeded exchange cascade; its
    # chi_eff is the differentiable generation-aware tilt moment (Prototype 6).
    # The four channel-B params (mu_B, sig_B, mu_chiB, sig_chiB) are REINTERPRETED
    # as the aligned SEED (mass mu/sig, natal spin-magnitude mu/sig) -- the numpyro
    # variables dict / priors are UNCHANGED (the borrowed values sit inside the
    # existing priors: mu_B~9.5 in U[5,15], sig_B~1 in U[.5,3], mu_chiB~0.10 in
    # U[-.1,.5], sig_chiB~0.05 in U[.01,.3]).  DOF is unchanged vs the exchange fit.
    p.add_argument("--aligned-seed", action="store_true",
                   help="channel B = REAL aligned-seed exchange cascade with the smooth "
                        "differentiable tilt-moment chi_eff (else the non-cascading Gaussian).")
    p.add_argument("--as-f-dep", type=float, default=0.5,
                   help="aligned-seed cascade DEPTH (note 58 = 0.5; kept SEPARATE from "
                        "channel A's --f-dep so a deep A cascade does not over-cascade the "
                        "low-mass seeds and wash out the aligned 1g feature). FIXED, not fit.")
    p.add_argument("--as-seed-lo", type=float, default=5.0)
    p.add_argument("--as-seed-hi", type=float, default=45.0)
    p.add_argument("--as-twog-memory", type=float, default=0.0,
                   help="fraction of 2g tilts that RETAIN alignment (0 = companion isotropic). "
                        "FIXED (freeing it is a razor-rough direction; scan if needed).")
    p.add_argument("--fix-aligned-seed", action="store_true",
                   help="ANCHOR: PIN the four aligned-seed shape params (mu_B,sig_B,mu_chiB,"
                        "sig_chiB) at the borrowed companion values (--as-* below); m_max_A, "
                        "spin_chi, log_R_A, log_R_B stay free. (Mirror of the 2gker anchor.)")
    p.add_argument("--as-mu", type=float, default=9.5, help="borrowed aligned-field mass mean")
    p.add_argument("--as-sig", type=float, default=1.0, help="borrowed aligned-field mass width")
    p.add_argument("--as-chimu", type=float, default=0.10, help="borrowed natal aligned chi mean")
    p.add_argument("--as-chisig", type=float, default=0.05, help="borrowed natal aligned chi width")
    p.add_argument("--outdir", default=os.path.join(_HERE, "figures_two_channel_pl"))
    p.add_argument("--max-wall-s", type=float, default=None,
                   help="exit-driven checkpointing: wall-clock budget (s) per process. After a "
                        "checkpoint that leaves work remaining, if elapsed > this, exit with "
                        "--checkpoint-exit-code so HTCondor spools the checkpoint and restarts "
                        "(resumes from nuts_ckpt.pkl). Robust to hard evictions, unlike relying on "
                        "ON_EXIT_OR_EVICT. Default None = run to completion in one process. "
                        "Requires --checkpoint-dir.")
    p.add_argument("--checkpoint-exit-code", type=int, default=85,
                   help="exit code emitted on an exit-driven checkpoint (must match the sub's "
                        "checkpoint_exit_code); only used when --max-wall-s is set.")
    args = p.parse_args()
    M.F_DEP = args.f_dep
    # Configure the opt-in engine knobs BEFORE building the sampler (globals are
    # read at trace time; make_combined_pl traces once during construction).
    bpm.EXCHANGE_MODE = int(args.exchange_mode)
    bpm.EXCHANGE_C = float(args.exchange_c)
    bpm.EXCHANGE_INTRUDER_N = float(args.exchange_intruder_n)
    bpm.SEGREGATION_N = float(args.segregation_n)
    # RETAIN_SOFT (sigmoid v_esc retention) is required for a free-v_esc grad;
    # for a FIXED-v_esc scan we keep the committed hard Heaviside retention.
    bpm.RETAIN_SOFT = bool(args.free_v_esc)
    bpm.RETAIN_TAU = float(args.retain_tau)
    if not args.free_v_esc:
        M.V_ESC = float(args.v_esc)
    # 1g pairing exponent b1 (base kernel q^b): fixed value / free-run init.
    if args.base_b is not None:
        M.KERNEL_B = float(args.base_b)
        print(f"1g pairing exponent b1 = {M.KERNEL_B} "
              f"{'FREE (sampled)' if args.free_base_b else 'FIXED'}")
    # 2g kernel: concrete guard on BEFORE trace; init/anchor values on the value
    # globals (free run overrides sigq/gamma with sampled tracers inside combined).
    if args.kernel_2g:
        bpm.KERNEL_2G_ON = True
        bpm.KERNEL_2G_B = float(args.kernel_2g_b)
        bpm.KERNEL_2G_MSTAR = float(args.kernel_2g_mstar)
        bpm.KERNEL_2G_SIGQ = float(args.kernel_2g_sigq)
        bpm.KERNEL_2G_GAMMA = float(args.kernel_2g_gamma)
        print(f"KERNEL_2G ON: b2={bpm.KERNEL_2G_B} mstar={bpm.KERNEL_2G_MSTAR} "
              f"sigq={bpm.KERNEL_2G_SIGQ} gamma={bpm.KERNEL_2G_GAMMA} "
              f"{'FREE (sampled)' if args.free_2g_kernel else 'FIXED (anchor)'}")
    print(f"ENGINE KNOBS: exchange_mode={bpm.EXCHANGE_MODE} C={bpm.EXCHANGE_C} "
          f"intruder_n={bpm.EXCHANGE_INTRUDER_N} seg_n={bpm.SEGREGATION_N} "
          f"STE={bpm.EXCHANGE_STE} RETAIN_SOFT={bpm.RETAIN_SOFT} tau={bpm.RETAIN_TAU} | "
          f"v_esc={'FREE[%g,%g]' % (args.v_esc_min, args.v_esc_max) if args.free_v_esc else 'FIXED %g' % args.v_esc}")
    # aligned-seed channel B (opt-in): set the module globals BEFORE the sampler is
    # built (make_sampler_alignedseed reads them at construction/trace time).
    global ALIGNED_SEED, AS_F_DEP, AS_SEED_LO, AS_SEED_HI, AS_TWOG_MEMORY
    if args.aligned_seed:
        ALIGNED_SEED = True
        AS_F_DEP = float(args.as_f_dep)
        AS_SEED_LO = float(args.as_seed_lo); AS_SEED_HI = float(args.as_seed_hi)
        AS_TWOG_MEMORY = float(args.as_twog_memory)
        print(f"ALIGNED-SEED channel B: cascade seed ~ N(mu_B,sig_B)[{AS_SEED_LO},{AS_SEED_HI}] "
              f"x aligned natal chi ~ N(mu_chiB,sig_chiB)[0,1]; as_f_dep={AS_F_DEP} "
              f"twog_memory={AS_TWOG_MEMORY}; chi_eff via SMOOTH generation_tilt_chieff_moments")
    comb, nA = make_combined_pl(args.n_seed, args.n_b)

    TRUTH = dict(alpha_A=2.0, m_max_A=50.0, spin_chi=0.05, mu_B=9.0, sig_B=1.2,
                 mu_chiB=0.15, sig_chiB=0.05, log_R_A=math.log(0.4), log_R_B=math.log(0.6),
                 v_esc=float(args.v_esc))

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
        # alpha_A widened DOWN to 0 (flat) to un-rail the old 1.5 floor: the corrected
        # selection grid wants a shallower natal INTERIOR slope, and physically it can be
        # ~flat since the observed high-mass population is reprocessed by the cascade, not
        # set by the natal power law. Lower bound kept at 0 (not negative) -- a RISING
        # natal (alpha<0) is unphysical AND makes the cascade MC top-heavy/slow (a -1,6
        # run ground for 20h). (ROS 2026-07-04)
        "alpha_A": dist.Uniform(0.0, 3.0), "m_max_A": dist.Uniform(40.0, 70.0),
        "spin_chi": dist.Uniform(0.01, 0.3),
        "mu_B": dist.Uniform(5., 15.), "sig_B": dist.Uniform(0.5, 3.0),   # NARROW low-mass peak
        "mu_chiB": dist.Uniform(-0.1, 0.5), "sig_chiB": dist.Uniform(0.01, 0.3),
        "log_R_A": dist.Uniform(logr0 - args.log_r_width, logr0 + args.log_r_width),
        "log_R_B": dist.Uniform(logr0 - args.log_r_width, logr0 + args.log_r_width)}
    # optionally PIN alpha_A (move it from sampled `variables` to `constant_params`)
    constants = {}
    if args.fix_alpha_a is not None:
        constants["alpha_A"] = float(args.fix_alpha_a); del variables["alpha_A"]
        print(f"FIXED alpha_A = {args.fix_alpha_a} (natal interior slope pinned, not sampled)")
    # ANCHOR: pin the four aligned-seed shape params at the borrowed companion values
    # (move them variables -> constants); m_max_A / spin_chi / rates stay free.
    if args.fix_aligned_seed:
        for k, v in (("mu_B", args.as_mu), ("sig_B", args.as_sig),
                     ("mu_chiB", args.as_chimu), ("sig_chiB", args.as_chisig)):
            constants[k] = float(v)
            if k in variables:
                del variables[k]
        print(f"FIXED aligned-seed shape: mu_B={args.as_mu} sig_B={args.as_sig} "
              f"mu_chiB={args.as_chimu} sig_chiB={args.as_chisig} (ANCHOR; not sampled)")
    # v_esc enters channel A as either a FREE (log-uniform) retention parameter
    # or a FIXED constant (for a v_esc scan / committed behaviour).
    if args.free_v_esc:
        variables["v_esc"] = dist.LogUniform(args.v_esc_min, args.v_esc_max)
        print(f"FREE v_esc ~ LogUniform({args.v_esc_min}, {args.v_esc_max}) km/s (RETAIN_SOFT on)")
    else:
        constants["v_esc"] = float(args.v_esc)
    # 2g kernel params: sampled (free) or pinned (anchor) -> combined() kwargs.
    if args.kernel_2g:
        if args.free_2g_kernel:
            variables["sigq_2g"] = dist.LogUniform(args.sigq_min, args.sigq_max)
            variables["gamma_2g"] = dist.Uniform(args.gamma_min, args.gamma_max)
            print(f"FREE 2g kernel: sigq_2g ~ LogUniform({args.sigq_min},{args.sigq_max}), "
                  f"gamma_2g ~ Uniform({args.gamma_min},{args.gamma_max})")
        else:
            constants["sigq_2g"] = float(args.kernel_2g_sigq)
            constants["gamma_2g"] = float(args.kernel_2g_gamma)
    # 1g exponent b1: sampled (free, negative-extending) or pinned via M.KERNEL_B.
    if args.free_base_b:
        variables["kernel_b1"] = dist.Uniform(args.base_b_min, args.base_b_max)
        print(f"FREE 1g exponent: kernel_b1 ~ Uniform({args.base_b_min},{args.base_b_max}) "
              "(note 60 family C: extends NEGATIVE)")
    elif args.base_b is not None:
        constants["kernel_b1"] = float(args.base_b)
    names = sorted(variables)
    priors = JointDistribution(*[variables[k] for k in names])
    kw = dict(init_strategy=init_to_median, max_tree_depth=args.max_tree_depth, target_accept_prob=args.target_accept)
    model = numpyro_sampled_poisson_likelihood(comb, priors, variables, constants, {k: i for i, k in enumerate(names)}, Ls, pdet_fn, None)
    if args.checkpoint_dir:
        import sys
        from gwkokab.inference import run_nuts_with_checkpoints, CheckpointExit
        print(f"CHECKPOINT-SAFE: {args.checkpoint_dir} chunk={args.chunk}"
              + (f" max_wall_s={args.max_wall_s} exit_code={args.checkpoint_exit_code}" if args.max_wall_s else ""))
        try:
            post = run_nuts_with_checkpoints(model, (jnp.asarray(M.T_OBS),), num_warmup=args.num_warmup,
                                             num_samples=args.num_samples, key=jax.random.PRNGKey(0),
                                             ckpt_dir=args.checkpoint_dir, chunk=args.chunk, nuts_kwargs=kw,
                                             max_wall_s=args.max_wall_s)
        except CheckpointExit:
            # Checkpoint is on disk; exit with the sentinel so HTCondor spools it
            # and restarts this job, which resumes from nuts_ckpt.pkl.
            sys.exit(args.checkpoint_exit_code)
    else:
        mcmc = MCMC(NUTS(model, **kw), num_warmup=args.num_warmup, num_samples=args.num_samples, progress_bar=True)
        mcmc.run(jax.random.PRNGKey(0), jnp.asarray(M.T_OBS)); post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    # re-insert any pinned constant as a constant "posterior" column so downstream
    # readers (figures, vt_calibration) that expect all 9 params still work.
    for k, v in constants.items():
        post[k] = np.full_like(post["m_max_A"], float(v))
    print("\nPOWER-LAW two-channel recovery:")
    for k in names:
        tag = "" if (REAL or k not in TRUTH) else f"  [{TRUTH[k]:.3f}]"
        print(f"  {k:9s}: {post[k].mean():7.3f} +/- {post[k].std():.3f}{tag}")
    print(f"  R_A/R_B = {np.exp(post['log_R_A']-post['log_R_B']).mean():.2f}")
    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, "two_channel_pl_post.npz"), **post, f_dep=args.f_dep,
             **({} if REAL else {f"truth_{k}": TRUTH[k] for k in TRUTH}))
    print(f"wrote {args.outdir}/two_channel_pl_post.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
