"""MAP + Laplace evidence for the fully hierarchical reassessment (2026-11-20).

Models (all on the 3D NAL catalog, kappa2.7 selection grid, v_esc=1000, rational
g(eta) kernel lor eta0=0.23 w=0.03, exchange mode 0 -- the headline settings):

  REF  current combined headline: flat-PL natal (alpha_A=0) with free m_max_A
       feeding the cascade + channel-B mixture field (PL + narrow peak on [5,20]).
  H0   Gaussian-seeded cascade (--seed-gauss), NO field.
  H1   H0 + broad aligned power-law field on [5,80] (--field broadpl).
  H2   H0 + narrow aligned field sharing the seed's (mu_A, sig_A) (--field shared).

B-series members of the paper's tab:bseries (flags from notes/refit_exchange/
launch_etafix_fleet_pcdev1{1,3}.sh and launch_headline_etafix.sh): flat-PL channel A
with free m_max_A, exchange mode 1 (HHM96), no g(eta) kernel, v_esc 1000:
  NULL  channel B = single aligned Gaussian.     BMIX  B = PL + narrow peak.
  BB    NULL + channel C aligned-seed cascade (shape pinned, log_R_C free).
  BC    NULL + channel C PL m^-2 on [20,45] (shape fixed, log_R_C free).
  BD    NULL + channel C Gaussian (mu_C, sig_C, log_R_C free).
  NULLG NULL with the combined kernel (exchange 0 + g(eta) lor) instead.
Every one of them is recentred (exact cascade) unless built with prop=None.

The objective is the fixed-key sampled Poisson log likelihood, written as in
notes/refit_exchange/model_ranking.py (eval_logL): logL = -mu + sum_i log sum_j
w_j L_i(x_j), full-covariance NAL Gaussians. Priors are flat boxes, so the MAP
is the box-constrained maximum of logL. The cascade pair sequence does not
depend on theta (pairs are drawn by occupancy; theta enters through the IS
weights only), so at a fixed key the objective is smooth in theta.

Optimizer: Adam in logit coordinates on the RATE-PROFILED lnL (shifting every
log_R by s gives -e^s mu + N s + sum log I, maximized at s = ln(N/mu), so
lnL_prof = sum log I - N ln mu + N ln N - N exactly), then the rates are put on
that profile and a bounded L-BFGS-B polish runs on the full lnL in theta.
Both phases checkpoint to <outdir>/state.npz and resume.

Laplace: lnZ = lnL(MAP) + d/2 ln 2pi - 1/2 ln det(-H) - sum ln(prior width),
H the exact Hessian of lnL in theta (jvp of grad, one direction at a time) --
the same formula as model_ranking.py with the posterior covariance replaced by
(-H)^-1. Parameters within 1% of a prior edge are flagged; lnZ_bdy subtracts
ln 2 for each (half-Gaussian), and is the less wrong of the two when any rail.

  --mode smoke   time one forward eval and one gradient at the init point
  --mode fit     Adam + L-BFGS-B + Laplace -> <outdir>/result.json
  --mode reeval  lnL at the MAP of <outdir>/result.json for several sampler
                 keys and a different n_seed/n_b -> <outdir>/reeval_<tag>.json
"""
import argparse, json, math, os, sys, time, importlib.util
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.scipy.special import logsumexp

HERE = os.path.dirname(os.path.realpath(__file__))
WS = os.environ.get("HIER_WS", os.path.expanduser("~/20261120-hier-reassess"))


def load_pl():
    spec = importlib.util.spec_from_file_location(
        "tcpl", os.path.join(HERE, "two_channel_pl_demo.py"))
    pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)
    return pl


# lower edge of the m_max_A prior (paper/committed: 40). Set by --m-max-lo.
M_MAX_LO = 40.0
FREE_KAPPA = None      # (lo, hi) when kappa is fitted (--free-kappa)
KAPPA_INIT = 2.7
LOG_R_RANGE = None     # (lo, hi) offsets from logr0 for every log_R_* (--log-r-range)

BSERIES = {
    "NULL": dict(exch=1, geta=False, bmix=False, chc="off"),
    "BB": dict(exch=1, geta=False, bmix=False, chc="alignedseed"),
    "BC": dict(exch=1, geta=False, bmix=False, chc="pl"),
    "BD": dict(exch=1, geta=False, bmix=False, chc="gauss"),
    "BMIX": dict(exch=1, geta=False, bmix=True, chc="off"),
    "NULLG": dict(exch=0, geta=True, bmix=False, chc="off"),
}
# launch-script --init-json points (committed fits' starting points)
BSERIES_INIT = {
    "NULL": dict(m_max_A=40.33, spin_chi=0.214, mu_B=9.15, sig_B=2.31, mu_chiB=0.073,
                 sig_chiB=0.098, log_R_A=10.12, log_R_B=11.79),
    "BB": dict(m_max_A=40.43, spin_chi=0.192, mu_B=8.95, sig_B=1.80, mu_chiB=0.050,
               sig_chiB=0.047, log_R_A=10.01, log_R_B=11.76, log_R_C=9.28),
    "BC": dict(m_max_A=40.9, spin_chi=0.22, mu_B=8.98, sig_B=1.81, mu_chiB=0.06,
               sig_chiB=0.10, log_R_A=9.97, log_R_B=11.82, log_R_C=6.2),
    "BD": dict(m_max_A=42.4, spin_chi=0.19, mu_B=8.15, sig_B=2.51, mu_chiB=0.06,
               sig_chiB=0.10, log_R_A=9.68, log_R_B=11.83, log_R_C=9.0, mu_C=21.6, sig_C=10.2),
    "BMIX": dict(m_max_A=41.0, spin_chi=0.20, mu_B=9.35, sig_B=1.20, mu_chiB=0.06,
                 sig_chiB=0.10, log_R_A=10.05, log_R_B=11.85, f_peakB=0.82, alpha_B=2.7),
}
BSERIES_INIT["NULLG"] = dict(BSERIES_INIT["NULL"])


def model_spec(model, logr0, rw=8.0):
    """(bounds, constants) for each model. Bounds mirror two_channel_pl_demo's
    priors under the headline flags (--sig-b-min 0.2 --log-r-width 8).
    With --free-kappa LO HI (native4d only), kappa ~ U[LO, HI] is added to every model."""
    b, c = _model_spec(model, logr0, rw)
    if LOG_R_RANGE is not None:        # --log-r-range LO HI: every log_R_* on [logr0+LO, logr0+HI]
        for k in b:
            if k.startswith("log_R_"):
                b[k] = (logr0 + LOG_R_RANGE[0], logr0 + LOG_R_RANGE[1])
    if FREE_KAPPA is not None:
        b["kappa"] = tuple(FREE_KAPPA)
    return b, c


def _model_spec(model, logr0, rw=8.0):
    b = {"spin_chi": (0.01, 0.3), "log_R_A": (logr0 - rw, logr0 + rw)}
    c = {"v_esc": 1000.0}
    rB = {"log_R_B": (logr0 - rw, logr0 + rw)}
    spinB = {"mu_chiB": (-0.1, 0.5), "sig_chiB": (0.01, 0.3)}
    if model in BSERIES:
        sp = BSERIES[model]
        b.update(m_max_A=(M_MAX_LO, 70.0), mu_B=(5.0, 15.0),
                 sig_B=(0.2 if sp["bmix"] else 0.5, 3.0), **spinB, **rB)
        c["alpha_A"] = 0.0
        if sp["bmix"]:
            b.update(f_peakB=(0.01, 0.99), alpha_B=(-4.0, 8.0))
        if sp["chc"] != "off":
            b["log_R_C"] = (logr0 - rw, logr0 + rw)
        if sp["chc"] == "gauss":
            b.update(mu_C=(12.0, 45.0), sig_C=(2.0, 15.0))
        return b, c
    if model in ("REF", "REFRC"):
        b.update(m_max_A=(M_MAX_LO, 70.0), mu_B=(5.0, 15.0), sig_B=(0.2, 3.0),
                 f_peakB=(0.01, 0.99), alpha_B=(-4.0, 8.0), **spinB, **rB)
        c["alpha_A"] = 0.0
    else:
        b.update(mu_A=(5.0, 20.0), sig_A=(0.3, 5.0))
        if model in ("H1", "H2"):
            b.update(**spinB, **rB)
        if model == "H1":
            b["alpha_B"] = (-4.0, 8.0)
    return b, c


def default_init(model, logr0):
    th = _default_init(model, logr0)
    if FREE_KAPPA is not None:
        th["kappa"] = KAPPA_INIT
    return th


def _default_init(model, logr0):
    if model in BSERIES:
        return dict(BSERIES_INIT[model])
    if model in ("REF", "REFRC"):   # headline posterior means (inputs/headline_combined_vesc1000_post.npz)
        return dict(m_max_A=40.82, spin_chi=0.038, mu_B=9.30, sig_B=1.21,
                    mu_chiB=0.054, sig_chiB=0.043, log_R_A=10.14, log_R_B=11.77,
                    f_peakB=0.80, alpha_B=3.70)
    th = dict(mu_A=10.0, sig_A=1.5, spin_chi=0.05, log_R_A=logr0 + 4.0)
    if model in ("H1", "H2"):
        th.update(mu_chiB=0.05, sig_chiB=0.05, log_R_B=logr0 + 5.0)
    if model == "H1":
        th["alpha_B"] = 2.5
    return th


def configure(pl, model, f_dep, key, prop=None):
    import red_team.backup_particle_mc as bpm
    M = pl.M
    M.F_DEP = float(f_dep)
    M.V_ESC = 1000.0
    M.SAMPLER_KEY = jax.random.PRNGKey(int(key))
    bpm.EXCHANGE_MODE = 0; bpm.EXCHANGE_C = 1.0; bpm.EXCHANGE_INTRUDER_N = 1.0
    bpm.SEGREGATION_N = 0.0; bpm.RETAIN_SOFT = False; bpm.RETAIN_TAU = 50.0
    bpm.KERNEL_GETA_ON = True; bpm.KERNEL_GETA_FORM = "lor"
    bpm.KERNEL_GETA_ETA0 = 0.23; bpm.KERNEL_GETA_P = 4.0
    bpm.KERNEL_GETA_W = 0.03; bpm.KERNEL_GETA_FLOOR = 0.0
    pl.CHANNEL_C = "off"; pl.ALIGNED_SEED = False
    if model in BSERIES:
        sp = BSERIES[model]
        bpm.EXCHANGE_MODE = sp["exch"]
        bpm.KERNEL_GETA_ON = sp["geta"]
        pl.CHANNEL_B_MIX = sp["bmix"]; pl.BMIX_MLO, pl.BMIX_MHI = 5.0, 20.0
        pl.SEED_GAUSS = False; pl.FIELD = "default"
        pl.CHANNEL_C = sp["chc"]; pl.N_C = 2500
        if sp["chc"] == "pl":
            pl.C_MLO, pl.C_MHI, pl.C_ALPHA = 20.0, 45.0, 2.0
            pl.C_QA, pl.C_QB, pl.C_CHI_SIG = 4.0, 1.0, 0.2
        elif sp["chc"] == "gauss":
            pl.C_QA, pl.C_QB, pl.C_CHI_SIG = 6.0, 1.0, 0.1
        elif sp["chc"] == "alignedseed":
            pl.AS_MU_C, pl.AS_SIG_C, pl.AS_CHIMU_C, pl.AS_CHISIG_C = 9.5, 1.0, 0.10, 0.05
            pl.AS_F_DEP = 0.5; pl.AS_SEED_LO, pl.AS_SEED_HI = 5.0, 45.0
            pl.AS_TWOG_MEMORY = 0.0
        if prop is not None:        # exact: every cascade proposal == its target
            pl.PL_PROP_REF = (prop["alpha"], prop["m_max"], prop["spin"])
            pl.AS_C_PROP_EXACT = True
        return
    if model in ("REF", "REFRC"):
        pl.CHANNEL_B_MIX = True; pl.BMIX_MLO, pl.BMIX_MHI = 5.0, 20.0
        pl.SEED_GAUSS = False; pl.FIELD = "default"
        if model == "REFRC" and prop is not None:
            pl.PL_PROP_REF = (prop["alpha"], prop["m_max"], prop["spin"])
    else:
        pl.CHANNEL_B_MIX = False; pl.SEED_GAUSS = True
        pl.FIELD = {"H0": "none", "H1": "broadpl", "H2": "shared"}[model]
        pl.FIELD_MLO, pl.FIELD_MHI = 5.0, 80.0
        if prop is not None:            # proposal == target at prop (recentred)
            pl.SEED_PROP_MU, pl.SEED_PROP_SIG = prop["mu"], prop["sig"]
            pl.SEED_PROP_FRAC = 1.0; pl.SEED_PROP_SPIN = prop["spin"]


def build(args, model, key, n_seed, n_b, prop=None, dry=False):
    cat = np.load(args.catalog, allow_pickle=True)
    if dry:
        T_dry = 1.0
        if getattr(args, "likelihood", "nal3d") == "native4d":   # same rate units as _build_native4d
            g = np.load(args.pdet_z_grid, allow_pickle=True)
            T_dry = float(json.loads(str(g["metadata_json"]))["T_obs_years"])
        n_obs = len(cat["centers"]); logr0 = float(np.log(n_obs / T_dry / 2.0))
        bounds, consts = model_spec(model, logr0)
        return dict(names=sorted(bounds), bounds=bounds, consts=consts, n_obs=n_obs, logr0=logr0)
    pl = load_pl()
    configure(pl, model, args.f_dep, key, prop)
    M = pl.M
    assert float(M.T_OBS) == 1.0
    comb, nA = pl.make_combined_pl(n_seed, n_b)
    if getattr(args, "likelihood", "nal3d") == "native4d":
        return _build_native4d(args, cat, pl, M, model, comb, nA)
    from mass_pdet_interp import make_real_pdet_fn
    pdet = make_real_pdet_fn(args.pdet_grid)
    ce = jnp.asarray(cat["centers"]); cov = jnp.asarray(cat["covs"])
    prec = jnp.linalg.inv(0.5 * (cov + jnp.transpose(cov, (0, 2, 1))) + 1e-12 * jnp.eye(3))
    _, logdet = jnp.linalg.slogdet(cov)
    n_obs = int(ce.shape[0]); T = float(M.T_OBS)
    logr0 = float(np.log(n_obs / T / 2.0))
    bounds, consts = model_spec(model, logr0)
    names = sorted(bounds)

    def parts(theta):
        th = {k: theta[i] for i, k in enumerate(names)}
        th.update(consts)
        s, lw = comb(**th)
        pd = pdet(s)
        wpd = T * jnp.exp(lw) * pd
        mu = jnp.sum(wpd); muA = jnp.sum(wpd[:nA])
        x = M.mc_eta_chieff(s)
        dz = x[None, :, :] - ce[:, None, :]
        qf = jnp.einsum("eni,eij,enj->en", dz, prec, dz)
        logLmat = -0.5 * qf - 0.5 * logdet[:, None] - 1.5 * jnp.log(2 * jnp.pi)
        logI = logsumexp(lw[None, :] + logLmat, axis=1)
        return -mu + jnp.sum(logI), (mu, muA)

    def prof(theta):
        L, (mu, muA) = parts(theta)
        sl = L + mu
        return sl - n_obs * jnp.log(mu) + n_obs * math.log(n_obs) - n_obs, (mu, muA)

    return dict(prof=prof, pl=pl, M=M, names=names, bounds=bounds, consts=consts, parts=parts,
                n_obs=n_obs, logr0=logr0, nA=nA, comb=comb, pdet=pdet)


def _build_native4d(args, cat, pl, M, model, comb, nA):
    """--likelihood native4d: revised 4D event fits (A's HybridPoissonLikelihood: z integrated
    per event by Gauss-Legendre quadrature against p(z|kappa) ~ dVc/dz (1+z)^(kappa-1) on
    (0, 2.5]; 3D NAL fallback for events without a fit; selection pdet(m1, m2, z) from the
    refinedz grid). kappa is fixed at --kappa. Rates are per year of detector time
    (T_obs from the grid). Draw positions depend on the proposal and key only, so one
    reference draw sizes the chirp-mass windows for this build."""
    import native4d_population as NP
    NP.install_rift_stub()
    kappa = float(args.kappa)
    n_obs = int(len(cat["centers"]))
    T_ref = float(NP.ZSelection(args.pdet_z_grid).T_obs_years)
    logr0 = float(np.log(n_obs / T_ref / 2.0))
    bounds, consts = model_spec(model, logr0)
    names = sorted(bounds)
    ref = default_init(model, logr0)
    ref.update(consts)
    ref.pop("kappa", None)
    mc_ref = None
    if args.native4d_window:
        s0, _ = comb(**ref)
        mc_ref = np.asarray(NP.mc_eta_chieff(jnp.asarray(s0))[:, 0])
    like = NP.HybridPoissonLikelihood(cat, args.native4d_tables, args.pdet_z_grid,
                                      tiers=tuple(args.native4d_tiers.split(",")),
                                      nodes=args.native4d_nodes, mc_ref=mc_ref)
    T = like.T_obs
    if like.window is not None:      # positions are theta-independent: check overflow once
        s0, lw0 = comb(**ref)
        _, d0 = like(s0, lw0, kappa)
        if int(d0["window_overflow"]) > 0:
            raise SystemExit("native4d window overflow for this proposal/key; rerun without --native4d-window")

    def parts(theta):
        th = {k: theta[i] for i, k in enumerate(names)}
        th.update(consts)
        kap = th.pop("kappa", kappa)
        s, lw = comb(**th)
        L, d = like(s, lw, kap)
        mu = d["mu"]
        muA = T * jnp.sum(jnp.exp(lw[:nA]) * like.sel.xi(s[:nA], kap))
        return L, (mu, muA)

    def prof(theta):
        L, (mu, muA) = parts(theta)
        sl = L + mu
        return sl - n_obs * jnp.log(mu) + n_obs * math.log(n_obs) - n_obs, (mu, muA)

    win = "on" if like.window is not None else "off"
    kdesc = f"free U{tuple(FREE_KAPPA)}" if FREE_KAPPA is not None else kappa
    print(f"NATIVE4D: {like.n4} events on 4D fits, {like.n3} on 3D NAL; kappa={kdesc}; "
          f"T_obs={T:.4f} yr; window={win}", flush=True)
    return dict(prof=prof, pl=pl, M=M, names=names, bounds=bounds, consts=consts, parts=parts,
                n_obs=n_obs, logr0=logr0, nA=nA, comb=comb, pdet=None, like=like)


# ---- coordinate maps -------------------------------------------------------
def to_z(theta, lo, hi):
    u = np.clip((theta - lo) / (hi - lo), 1e-6, 1 - 1e-6)
    return np.log(u) - np.log1p(-u)


def to_theta(z, lo, hi):
    return lo + (hi - lo) * jax.nn.sigmoid(z)


def rate_rescale(theta, names, lo, hi, mu, n_obs):
    """exact rate profile: shifting every log_R by s multiplies mu by e^s and adds
    s to each log I_i, so the optimum over s is s = ln(N_obs/mu)."""
    s = math.log(n_obs / max(float(mu), 1e-300))
    th = np.array(theta, float)
    for i, k in enumerate(names):
        if k.startswith("log_R_"):
            th[i] = np.clip(th[i] + s, lo[i] + 1e-3, hi[i] - 1e-3)
    return th


def save_state(path, **kw):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **kw)
    os.replace(tmp, path)


def laplace(B, theta, logL):
    names, bounds = B["names"], B["bounds"]
    d = len(names)
    g = jax.jit(jax.grad(lambda t: B["parts"](t)[0]))
    hv = jax.jit(lambda t, v: jax.jvp(g, (t,), (v,))[1])
    H = np.zeros((d, d))
    th = jnp.asarray(theta)
    for i in range(d):
        e = np.zeros(d); e[i] = 1.0
        H[:, i] = np.asarray(hv(th, jnp.asarray(e)))
    H = 0.5 * (H + H.T)
    negH = -H
    ev = np.linalg.eigvalsh(negH)
    pd = bool(np.all(ev > 0))
    widths = np.array([bounds[k][1] - bounds[k][0] for k in names])
    ln_prior = -float(np.sum(np.log(widths)))
    lndet = float(np.sum(np.log(np.abs(ev))))     # |eig| if not PD (flagged)
    lnZ = float(logL) + 0.5 * d * math.log(2 * math.pi) - 0.5 * lndet + ln_prior
    lo = np.array([bounds[k][0] for k in names]); hi = np.array([bounds[k][1] for k in names])
    rail = [k for i, k in enumerate(names)
            if min(theta[i] - lo[i], hi[i] - theta[i]) < 0.01 * widths[i]]
    lnZ_bdy = lnZ - math.log(2.0) * len(rail)
    sig = None
    if pd:
        Sig = np.linalg.inv(negH)
        sig = {k: float(np.sqrt(Sig[i, i])) for i, k in enumerate(names)}
    return dict(hessian=H.tolist(), negH_eigs=ev.tolist(), negH_pd=pd,
                lnZ_laplace=lnZ, lnZ_laplace_bdy=lnZ_bdy, railing=rail,
                laplace_sigma=sig, ln_prior=ln_prior)


def prop_ref_from(model, th):
    """proposal reference point for the recentred models (None = fixed proposal)."""
    if model in ("H0", "H1", "H2"):
        return dict(mu=float(th["mu_A"]), sig=float(th["sig_A"]), spin=float(th["spin_chi"]))
    if model == "REFRC" or model in BSERIES:
        return dict(alpha=0.0, m_max=float(th["m_max_A"]), spin=float(th["spin_chi"]))
    return None


def adam(vgp, z, lo, hi, nsteps, lr0, lr1, log, ckpt, step0=0, mstate=None):
    """Adam on -lnL_prof in logit coordinates; returns (best_theta, best_L, z)."""
    b1, b2, eps = 0.9, 0.999, 1e-8
    lo_j, hi_j = jnp.asarray(lo), jnp.asarray(hi)
    m, v, best_theta, best_L = mstate if mstate else (np.zeros_like(z), np.zeros_like(z), None, -np.inf)
    for step in range(step0, nsteps):
        ts = time.time()
        theta = np.asarray(to_theta(jnp.asarray(z), lo_j, hi_j))
        (L, (mu, muA)), g = vgp(jnp.asarray(theta))
        L = float(L); g = np.asarray(g)
        if not np.isfinite(L) or not np.all(np.isfinite(g)):
            log(f"  step {step}: non-finite lnL/grad; moving halfway back to best")
            if best_theta is not None:
                z = 0.5 * (z + to_z(best_theta, lo, hi))
            m[:] = 0; v[:] = 0
            continue
        if L > best_L:
            best_L, best_theta = L, theta.copy()
        sg = 1 / (1 + np.exp(-z))
        gz = -g * (hi - lo) * sg * (1 - sg)
        lr = lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * step / nsteps))
        m = b1 * m + (1 - b1) * gz; v = b2 * v + (1 - b2) * gz ** 2
        z = z - lr * (m / (1 - b1 ** (step + 1))) / (np.sqrt(v / (1 - b2 ** (step + 1))) + eps)
        if (step + 1) % 5 == 0 or step + 1 == nsteps:
            log(f"  adam {step+1:4d} lnLprof={L:10.3f} best={best_L:10.3f} mu={float(mu):7.1f} "
                f"fA_det={float(muA)/float(mu):.3f} |gz|={np.linalg.norm(gz):.2e} {time.time()-ts:.1f}s/it "
                + " ".join(f"{x:.4g}" for x in theta))
            ckpt(step + 1, z, (m, v, best_theta, best_L))
    return best_theta, best_L, z


def run_fit(args):
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    st_path = os.path.join(args.outdir, "state.npz")
    logf = open(os.path.join(args.outdir, "fit.log"), "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    rc = args.model != "REF"            # recentred proposal for every model but REF
    # stage plan: (adam steps, lr0, lr1); after each stage the proposal moves to
    # the stage optimum (rc models only), then L-BFGS-B, then a final recentre.
    plan = [(args.adam_steps, args.lr, args.lr_min)]
    if rc:
        plan += [(args.adam_steps_rc, args.lr_rc, args.lr_min)] * args.n_recentre
    st = dict(np.load(st_path, allow_pickle=True)) if os.path.exists(st_path) else None
    if st is not None:
        st = {k: (st[k].item() if st[k].shape == () else st[k]) for k in st}
        stage, step = int(st["stage"]), int(st["step"])
        prop = json.loads(str(st["prop"])) if str(st["prop"]) != "null" else None
        z = st["z"]; best_theta = st["best_theta"]; best_L = float(st["best_L"])
        mstate = (st["m"], st["v"], best_theta, best_L) if step > 0 else None
        log(f"RESUME stage={stage} step={step} prop={prop} best_L={best_L:.3f}")
    else:
        stage, step, prop, mstate, best_theta, best_L, z = 0, 0, None, None, None, -np.inf, None

    def mk(prop):
        B = build(args, args.model, args.key, args.n_seed, args.n_b, prop)
        return (B, jax.jit(jax.value_and_grad(B["parts"], has_aux=True)),
                jax.jit(jax.value_and_grad(B["prof"], has_aux=True)))

    # names / bounds / init (proposal-independent)
    B0 = build(args, args.model, args.key, args.n_seed, args.n_b, None, dry=True)
    names, bounds, n_obs = B0["names"], B0["bounds"], B0["n_obs"]
    lo = np.array([bounds[k][0] for k in names]); hi = np.array([bounds[k][1] for k in names])
    if z is None:
        init = default_init(args.model, B0["logr0"])
        if args.init_json:
            init.update({k: float(v) for k, v in json.loads(args.init_json).items()})
        theta0 = np.array([init[k] for k in names], float)
        z = to_z(theta0, lo, hi)
        prop = prop_ref_from(args.model, dict(zip(names, theta0))) if rc else None
    log(f"[{args.model} f_dep={args.f_dep} n_seed={args.n_seed} n_b={args.n_b} key={args.key}] "
        f"params={names} recentre={rc}")

    def save(stage, step, z, ms, prop, phase="adam"):
        m_, v_, bt, bl = ms if ms else (np.zeros_like(z), np.zeros_like(z), best_theta, best_L)
        save_state(st_path, stage=stage, step=step, z=z, m=m_, v=v_,
                   best_theta=bt if bt is not None else np.asarray(to_theta(jnp.asarray(z), jnp.asarray(lo), jnp.asarray(hi))),
                   best_L=bl, prop=json.dumps(prop), phase=phase, names=np.array(names))

    trail = []
    while stage < len(plan):
        nst, lr0, lr1 = plan[stage]
        B, vg, vgp = mk(prop)
        log(f"stage {stage}: proposal {prop}; Adam {nst} steps lr {lr0}->{lr1} (nA={B['nA']}, "
            f"build+{time.time()-t0:.0f}s)")
        bt, bl, z = adam(vgp, z, lo, hi, nst, lr0, lr1, log,
                         lambda s_, z_, ms_: save(stage, s_, z_, ms_, prop), step, mstate)
        # rate profile at the stage optimum (exact)
        (L, (mu, _)), _ = vg(jnp.asarray(bt))
        bt = rate_rescale(bt, names, lo, hi, mu, n_obs)
        trail.append(dict(stage=stage, prop=prop, lnL_prof=bl, theta={k: float(bt[i]) for i, k in enumerate(names)}))
        log(f"stage {stage} done: best lnLprof {bl:.3f} at " + " ".join(f"{k}={bt[i]:.4g}" for i, k in enumerate(names)))
        best_theta, best_L = bt, bl
        z = to_z(bt, lo, hi); stage += 1; step = 0; mstate = None
        if rc:
            prop = prop_ref_from(args.model, dict(zip(names, bt)))
        save(stage, 0, z, None, prop)

    # ---------------- L-BFGS-B polish (full lnL, theta, current proposal) -------
    B, vg, vgp = mk(prop)
    from scipy.optimize import minimize
    cache = {}

    def fg(th):
        (L, _), g = vg(jnp.asarray(th))
        L = float(L)
        if not np.isfinite(L) or not np.all(np.isfinite(np.asarray(g))):
            return 1e30, np.zeros_like(th)
        cache["last"] = (np.array(th), L)
        return -L, -np.asarray(g)

    x0 = np.clip(best_theta, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    (L0, _), _ = vg(jnp.asarray(x0))
    state = {"th": x0, "L": float(L0)}
    log(f"L-BFGS-B start lnL={float(L0):.3f} (proposal {prop})")

    def cb(xk):
        th, L = cache.get("last", (None, -np.inf))
        if th is not None and L > state["L"]:
            state["L"], state["th"] = L, th
        log(f"  lbfgs lnL={state['L']:.3f} " + " ".join(f"{x:.4g}" for x in state["th"]))

    # trust box: the IS objective is only reliable near the proposal reference,
    # so the polish may move each parameter by at most lbfgs_trust x prior width
    tw = args.lbfgs_trust * (hi - lo)
    blo = np.maximum(lo + 1e-9 * (hi - lo), x0 - tw); bhi = np.minimum(hi - 1e-9 * (hi - lo), x0 + tw)
    res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=list(zip(blo, bhi)),
                   callback=cb, options=dict(maxiter=args.lbfgs_iters, maxfun=2 * args.lbfgs_iters))
    (Lr, _), _ = vg(jnp.asarray(res.x))
    if float(Lr) > state["L"]:
        state["L"], state["th"] = float(Lr), np.asarray(res.x)
    log(f"L-BFGS-B: {res.message} nit={res.nit} nfev={res.nfev} lnL={state['L']:.3f}")
    theta_map, lnL_preprop = state["th"], state["L"]

    # ---------------- final recentre + exact lnL + Laplace ----------------
    # rc models: score the polished point AND the pre-polish point, each on its
    # own recentred proposal (same key), and keep the better one.
    if rc:
        cands = []
        for lab, th in (("lbfgs", theta_map), ("pre-lbfgs", x0)):
            pr = prop_ref_from(args.model, dict(zip(names, th)))
            Bc, vgc, vgpc = mk(pr)
            (Lc, aux), gc = vgc(jnp.asarray(th))
            log(f"  candidate {lab}: recentred lnL {float(Lc):.3f}")
            cands.append((float(Lc), lab, th, pr, Bc, vgc))
        Lc, lab, theta_map, prop, B, vg = max(cands, key=lambda c: c[0])
        log(f"  keep {lab}")
    (L, (mu, muA)), g = vg(jnp.asarray(theta_map))
    log(f"final proposal {prop}: lnL {float(L):.3f} (L-BFGS on the previous proposal {lnL_preprop:.3f})")
    lap = laplace(B, np.asarray(theta_map), float(L))
    # held-out cascade keys: the optimum was found on key args.key, so lnL there
    # is optimistic (fixed-realization winner's curse); fresh keys are not.
    ho = []
    for k in args.heldout_keys:
        Bk = build(args, args.model, k, args.n_seed, args.n_b, prop)
        Lk, (muk, muAk) = jax.jit(Bk["parts"])(jnp.asarray(theta_map))
        ho.append(dict(key=k, lnL=float(Lk), mu=float(muk), mu_A=float(muAk)))
        log(f"  held-out key {k}: lnL {float(Lk):.3f} mu {float(muk):.1f}")
    Lho = np.array([e["lnL"] for e in ho])
    lnL_ho = float(Lho.mean()) if len(Lho) else float("nan")
    lap["lnZ_heldout"] = lap["lnZ_laplace_bdy"] - float(L) + lnL_ho
    out = dict(heldout=ho, lnL_heldout_mean=lnL_ho,
               lnL_heldout_sd=float(Lho.std(ddof=1)) if len(Lho) > 1 else None,model=args.model, f_dep=args.f_dep, start=args.tag, n_seed=args.n_seed,
               n_b=args.n_b, key=args.key, names=names, recentred=rc, final_proposal=prop,
               theta_map={k: float(theta_map[i]) for i, k in enumerate(names)},
               bounds={k: list(bounds[k]) for k in names}, constants=B["consts"],
               lnL=float(L), lnL_before_final_recentre=lnL_preprop, mu=float(mu), mu_A=float(muA),
               field_frac_det=float(1.0 - float(muA) / float(mu)),
               grad_at_map={k: float(np.asarray(g)[i]) for i, k in enumerate(names)},
               stages=trail, lbfgs_message=str(res.message), n_obs=n_obs,
               wall_s=time.time() - t0, **lap)
    with open(os.path.join(args.outdir, "result.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"DONE lnL={out['lnL']:.3f} lnZ={out['lnZ_laplace']:.3f} (bdy {out['lnZ_laplace_bdy']:.3f}) "
        f"lnL_heldout={lnL_ho:.3f} negH_pd={out['negH_pd']} rail={out['railing']} field_frac_det={out['field_frac_det']:.3f} "
        f"wall={out['wall_s']/3600:.2f}h")


def run_reeval(args):
    r = json.load(open(os.path.join(args.outdir, "result.json")))
    args.f_dep = r["f_dep"]
    out = dict(src=args.outdir, n_seed=args.n_seed, n_b=args.n_b, evals=[])
    for k in args.keys:
        t0 = time.time()
        B = build(args, r["model"], k, args.n_seed, args.n_b, r.get("final_proposal"))
        th = jnp.asarray([r["theta_map"][n] for n in B["names"]])
        L, (mu, muA) = jax.jit(B["parts"])(th)
        out["evals"].append(dict(key=k, lnL=float(L), mu=float(mu), mu_A=float(muA)))
        print(f"reeval key={k} lnL={float(L):.3f} mu={float(mu):.1f} ({time.time()-t0:.0f}s)", flush=True)
    Ls = np.array([e["lnL"] for e in out["evals"]])
    out["lnL_mean"] = float(Ls.mean()); out["lnL_sd"] = float(Ls.std(ddof=1)) if len(Ls) > 1 else None
    with open(os.path.join(args.outdir, f"reeval_n{args.n_seed}.json"), "w") as f:
        json.dump(out, f, indent=1)


def run_laplace(args):
    """Laplace at the MAP of <outdir>/result.json, re-evaluated at --n-seed/--n-b with --key and the
    fit's final proposal (no re-optimisation). Writes laplace_n<n_seed>.json."""
    r = json.load(open(os.path.join(args.outdir, "result.json")))
    args.f_dep = r["f_dep"]
    t0 = time.time()
    B = build(args, r["model"], args.key, args.n_seed, args.n_b, r.get("final_proposal"))
    th = np.array([r["theta_map"][n] for n in B["names"]])
    L, (mu, muA) = jax.jit(B["parts"])(jnp.asarray(th))
    g = np.asarray(jax.jit(jax.grad(lambda t: B["parts"](t)[0]))(jnp.asarray(th)))
    out = laplace(B, th, float(L))
    ev, V = np.linalg.eigh(-np.array(out["hessian"]))
    v = V[:, 0]; order = np.argsort(-np.abs(v))
    out.update(src=args.outdir, n_seed=args.n_seed, n_b=args.n_b, key=args.key, lnL=float(L), mu=float(mu),
               names=B["names"], grad=dict(zip(B["names"], g.tolist())),
               min_eig_vector={B["names"][i]: float(v[i]) for i in order},
               min_eig_dominant=B["names"][order[0]], wall_s=time.time() - t0,
               ref_n_seed=r["n_seed"], ref_lnL=r["lnL"], ref_lnZ_laplace_bdy=r["lnZ_laplace_bdy"],
               ref_negH_eigs=r["negH_eigs"])
    with open(os.path.join(args.outdir, f"laplace_n{args.n_seed}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"laplace n_seed={args.n_seed}: lnL={float(L):.3f} lnZ_bdy={out['lnZ_laplace_bdy']:.3f} "
          f"(ref {r['lnZ_laplace_bdy']:.3f}) pd={out['negH_pd']} min eig {ev[0]:.3g} along {B['names'][order[0]]}", flush=True)


def run_smoke(args):
    t0 = time.time()
    B0 = build(args, args.model, args.key, args.n_seed, args.n_b, dry=True)
    init = default_init(args.model, B0["logr0"])
    if args.init_json:
        init.update({k: float(v) for k, v in json.loads(args.init_json).items()})
    prop = prop_ref_from(args.model, init)
    B = build(args, args.model, args.key, args.n_seed, args.n_b, prop)
    print(f"build {time.time()-t0:.1f}s nA={B['nA']} names={B['names']} proposal={prop}", flush=True)
    th = jnp.asarray([init[k] for k in B["names"]])
    f = jax.jit(B["parts"]); vg = jax.jit(jax.value_and_grad(B["parts"], has_aux=True))
    for lab, fn in (("forward", f), ("value+grad", vg)):
        t1 = time.time(); r = fn(th); jax.block_until_ready(r)
        t2 = time.time(); r = fn(th); jax.block_until_ready(r)
        t3 = time.time()
        print(f"{lab}: first {t2-t1:.1f}s (incl. compile), second {t3-t2:.1f}s", flush=True)
    (L, (mu, muA)), g = r
    print(f"lnL={float(L):.9f} mu={float(mu):.6f} muA={float(muA):.2f} grad=" +
          " ".join(f"{k}:{float(gi):+.3g}" for k, gi in zip(B["names"], np.asarray(g))), flush=True)
    print("finite grad:", bool(np.all(np.isfinite(np.asarray(g)))), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="fit", choices=["smoke", "fit", "reeval", "laplace"])
    p.add_argument("--model", default="H0", choices=["REF", "REFRC", "H0", "H1", "H2"] + list(BSERIES))
    p.add_argument("--f-dep", type=float, default=0.7)
    p.add_argument("--n-seed", type=int, default=3000)
    p.add_argument("--n-b", type=int, default=3000)
    p.add_argument("--key", type=int, default=11, help="cascade sampler key (M.SAMPLER_KEY)")
    p.add_argument("--keys", type=int, nargs="+", default=[11, 202, 3003],
                   help="reeval keys (n_seed 8000 realizations are all fresh)")
    p.add_argument("--heldout-keys", type=int, nargs="+", default=[202, 3003, 40004],
                   help="fresh cascade keys evaluated at the MAP at the fit n_seed")
    p.add_argument("--catalog", default=os.path.join(WS, "inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz"))
    p.add_argument("--pdet-grid", default=os.path.join(WS, "inputs/mass_pdet_grid_kappa2.7.npz"))
    p.add_argument("--init-json", default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--adam-steps", type=int, default=60, help="first-stage Adam steps")
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--lr-min", type=float, default=0.005)
    p.add_argument("--n-recentre", type=int, default=3,
                   help="recentred stages after the first (all models except REF)")
    p.add_argument("--adam-steps-rc", type=int, default=50)
    p.add_argument("--lr-rc", type=float, default=0.04)
    p.add_argument("--lbfgs-iters", type=int, default=40)
    p.add_argument("--lbfgs-trust", type=float, default=0.05,
                   help="L-BFGS-B may move each parameter by at most this x prior width")
    p.add_argument("--outdir", default=None)
    # ---- opt-in revised 4D event likelihoods (A's native4d path) ----
    p.add_argument("--likelihood", default="nal3d", choices=["nal3d", "native4d"])
    p.add_argument("--kappa", type=float, default=2.7, help="native4d: fixed redshift-evolution index "
                   "(with --free-kappa: the starting value)")
    p.add_argument("--log-r-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="every log_R_* prior on [logr0+LO, logr0+HI] (default -8 8)")
    p.add_argument("--free-kappa", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="native4d: fit kappa with a uniform prior on [LO, HI]")
    p.add_argument("--native4d-tables", default=os.path.join(WS, "runs/A_tables/native4d_events.npz"))
    p.add_argument("--native4d-tiers", default="primary")
    p.add_argument("--native4d-nodes", type=int, default=32)
    p.add_argument("--native4d-window", action="store_true")
    p.add_argument("--pdet-z-grid", default=os.path.join(WS, "inputs/combined_pdet_grid_mmin2_refinedz.npz"))
    p.add_argument("--m-max-lo", type=float, default=40.0,
                   help="lower edge of the m_max_A prior U[lo, 70] (committed 40)")
    args = p.parse_args()
    global M_MAX_LO, FREE_KAPPA, KAPPA_INIT
    M_MAX_LO = float(args.m_max_lo)
    if args.free_kappa is not None and args.likelihood != "native4d":
        raise SystemExit("--free-kappa needs --likelihood native4d")
    FREE_KAPPA = args.free_kappa; KAPPA_INIT = float(args.kappa)
    global LOG_R_RANGE
    LOG_R_RANGE = args.log_r_range
    if args.mode == "smoke":
        return run_smoke(args)
    if args.outdir is None:
        raise SystemExit("--outdir required")
    if args.mode == "laplace":
        return run_laplace(args)
    return run_fit(args) if args.mode == "fit" else run_reeval(args)


if __name__ == "__main__":
    raise SystemExit(main())
