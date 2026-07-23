"""Forward-model gallery driver — ONE configuration per process.

Forward-samples the fitted two-channel posterior through the real selection at a
chosen interaction-model configuration (pairing-kernel exponents a/b, the
mass-dependent b(M) hook, v_esc, F_DEP, and the opt-in binary-single EXCHANGE /
mass-segregation knobs in red_team.backup_particle_mc) and saves the detected +
intrinsic observables to a compact npz for the gallery collector
(gallery_figures.py).  See notes/forward_model_builds/ for the writeup.

Module globals (KERNEL_B1, EXCHANGE_*, SEGREGATION_N) are set BEFORE the first
trace; run one configuration per process (same rule as proto_kernel_bM.py).

  python forward_gallery.py --tag base_f090_v300                      # baseline
  python forward_gallery.py --tag exch_prompt_v300 --exchange-mode 2  # exchange
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np, jax, jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
_NOTES = os.path.normpath(os.path.join(
    _HERE, "../../..", "notes", "forward_model_builds"))
POST_ROOT = "/home/richard.oshaughnessy/coag_container"
CAT = os.path.join(_HERE, "../../../planning_o4_mixed/full_o3_o4a_o4b_catalog_NAL.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="output name (npz stem)")
    # mean-field kernel family
    p.add_argument("--kernel-a", type=float, default=0.0)
    p.add_argument("--kernel-b", type=float, default=2.0)
    p.add_argument("--eta-kernel", action="store_true",
                   help="use the EXCHANGE-SYMMETRIC kernel M^a (2 eta)^b instead of M^a q^b.")
    p.add_argument("--b1", type=float, default=0.0, help="mass-dependent b(M) slope")
    p.add_argument("--mstar", type=float, default=40.0)
    # RATIONAL g(eta) kernel (note 61; opt-in, replaces the q/eta power factor)
    p.add_argument("--geta-form", default=None, choices=["sat", "lor"],
                   help="rational g(eta) kernel: K = M^a [floor + g(eta)]. "
                        "'sat' = eta^p/(eta0^p+eta^p); 'lor' = Pade bump "
                        "1/(1+((eta-eta0)/w)^2). None = off (committed kernel).")
    p.add_argument("--geta-eta0", type=float, default=0.1875,
                   help="eta scale: sat half-max / lor centre (eta(q=1/3)=0.1875)")
    p.add_argument("--geta-p", type=float, default=4.0, help="sat sharpness p")
    p.add_argument("--geta-w", type=float, default=0.03, help="lor half-width in eta")
    p.add_argument("--geta-floor", type=float, default=0.0,
                   help="additive floor on g (keeps unequal pairings alive)")
    # environment
    p.add_argument("--v-esc", type=float, default=300.0)
    p.add_argument("--f-dep", type=float, default=0.9)
    p.add_argument("--alpha-a", type=float, default=None,
                   help="OVERRIDE the natal power-law slope alpha_A drawn from --post "
                        "(the posterior pins alpha_A=0). For the alpha x f_dep scan.")
    # exchange / segregation (opt-in; defaults reproduce committed engine)
    p.add_argument("--exchange-mode", type=int, default=0,
                   help="0=off, 1=HHM96 smooth prob, 2=prompt (m3>min(m1,m2))")
    p.add_argument("--exchange-c", type=float, default=1.0)
    p.add_argument("--seg-n", type=float, default=0.0,
                   help="pair draw p ~ occ*m^n (RAPSTER 3bb: n=5)")
    p.add_argument("--seg-k-n", type=float, default=0.0,
                   help="MULTIPLICATIVE K segregation (Thrust 2.5): "
                        "K *= ((m1*m2)/mref^2)^(n/2); 0 = off (identity)")
    p.add_argument("--seg-k-mref", type=float, default=50.0,
                   help="mass scale for the multiplicative K factor")
    p.add_argument("--intruder-n", type=float, default=1.0,
                   help="intruder draw p ~ occ*m^n (RAPSTER exchanges: n=1)")
    # TRUE in-cluster SUM OF KERNELS (Prototype 4; opt-in, defaults = committed).
    # Channel 1 = the exchange process (set --exchange-mode); channel 2 = a
    # plain-merger kernel (m1+m2)^a2 q^b2.  K_tot = C1 K1 + C2 K2 in ONE cascade.
    p.add_argument("--sumk-on", type=int, default=0,
                   help="1 = in-cluster sum of two kernels in the SAME cascade")
    p.add_argument("--sumk-a2", type=float, default=0.0,
                   help="channel-2 total-mass exponent a2")
    p.add_argument("--sumk-b2", type=float, default=0.0,
                   help="channel-2 mass-ratio exponent b2 (0 = no eq-mass pref)")
    p.add_argument("--sumk-c1", type=float, default=1.0,
                   help="channel-1 (exchange) kernel prefactor")
    # GENERATION-GATED DEGENERATE 2g KERNEL (Prototype 5; opt-in, defaults=off).
    p.add_argument("--kernel-2g", type=int, default=0,
                   help="1 = turn on the generation-gated degenerate 2g kernel "
                        "(boost q^b2*exp[-(1-q)^2/2sigq^2]*(M/M*)^gamma for pairs "
                        "with max(gen_i,gen_j)>=2). Sets bpm.KERNEL_2G_ON.")
    p.add_argument("--kernel-2g-sigq", type=float, default=0.17,
                   help="sigma_q (equal-mass width); shooter adam optimum 0.17.")
    p.add_argument("--kernel-2g-gamma", type=float, default=1.07,
                   help="gamma (IMF-undoing total-mass exponent); shooter 1.07.")
    p.add_argument("--kernel-2g-b", type=float, default=0.0,
                   help="b2 (extra q-tilt prefactor); default 0 (shooter net q-law).")
    p.add_argument("--kernel-2g-mstar", type=float, default=40.0,
                   help="M* total-mass pivot; shooter M*=40.")
    p.add_argument("--sumk-c2", type=float, default=1.0,
                   help="channel-2 (plain merger) kernel prefactor (sweep C2/C1)")
    # STEADILY-SEEDED / DRIVEN cascade (Prototype 6; opt-in, defaults = committed
    # source-free r=0). --driven 1 turns on a continuous natal source r(x;t)!=0
    # that replenishes the low-mass reservoir each step (--driven-seed-rate: 1.0 =
    # reservoir held full = fully driven; 0 = source-free). In driven mode the
    # depletion-budget halt is lifted, so --f-dep is REINTERPRETED as the driven
    # DEPTH knob (it sets n_steps = f_dep*n_init*3 = encounters per body); scan it
    # to LARGER values (2-6) to let the driven scale-free tail build.
    p.add_argument("--driven", type=int, default=0,
                   help="1 = steadily-seeded (driven) cascade, source term r!=0")
    p.add_argument("--driven-seed-rate", type=float, default=1.0,
                   help="per-step natal replenishment fraction (1.0=full reservoir)")
    # WIDE natal for the driven source: power-law + LOW-mass peak (the peak shapes
    # r(x) because the driven refill is proportional to the natal weights). 0 = pure PL.
    p.add_argument("--natal-peak-frac", type=float, default=0.0,
                   help="weight of a low-mass Gaussian peak added to channel-A natal (0=off)")
    p.add_argument("--natal-peak-mu", type=float, default=10.0,
                   help="low-mass natal peak centre [Msun]")
    p.add_argument("--natal-peak-sig", type=float, default=2.0,
                   help="low-mass natal peak width [Msun]")
    # posterior + sampling budget
    p.add_argument("--post", default="figures_nal_f090_fa0w",
                   help="posterior dir under coag_container (matched f_dep fit)")
    p.add_argument("--n-seed", type=int, default=8000)
    p.add_argument("--n-b", type=int, default=6000)
    p.add_argument("--n-draw", type=int, default=20)
    p.add_argument("--outdir", default=os.path.join(_NOTES, "data"))
    args = p.parse_args()

    t0 = time.time()
    # ---- set engine module globals BEFORE any trace ----
    import red_team.backup_particle_mc as bpm
    bpm.KERNEL_B1 = float(args.b1)
    bpm.KERNEL_MSTAR = float(args.mstar)
    if args.eta_kernel:
        bpm.ETA_KERNEL = True
        print("ETA_KERNEL ON: K = M^a (2 eta)^b (exchange-symmetric)", flush=True)
    if args.geta_form is not None:
        bpm.KERNEL_GETA_ON = True
        bpm.KERNEL_GETA_FORM = str(args.geta_form)
        bpm.KERNEL_GETA_ETA0 = float(args.geta_eta0)
        bpm.KERNEL_GETA_P = float(args.geta_p)
        bpm.KERNEL_GETA_W = float(args.geta_w)
        bpm.KERNEL_GETA_FLOOR = float(args.geta_floor)
        print(f"KERNEL_GETA ON: form={bpm.KERNEL_GETA_FORM} eta0={bpm.KERNEL_GETA_ETA0} "
              f"p={bpm.KERNEL_GETA_P} w={bpm.KERNEL_GETA_W} floor={bpm.KERNEL_GETA_FLOOR} "
              f"(kernel_b ignored)", flush=True)
    if args.exchange_mode:
        bpm.EXCHANGE_MODE = int(args.exchange_mode)
        bpm.EXCHANGE_C = float(args.exchange_c)
        bpm.EXCHANGE_INTRUDER_N = float(args.intruder_n)
    if args.seg_n:
        bpm.SEGREGATION_N = float(args.seg_n)
    if args.seg_k_n:
        bpm.KERNEL_SEG_N = float(args.seg_k_n)
        bpm.KERNEL_SEG_MREF = float(args.seg_k_mref)
    if args.sumk_on:
        bpm.SUMK_ON = int(args.sumk_on)
        bpm.SUMK_A2 = float(args.sumk_a2)
        bpm.SUMK_B2 = float(args.sumk_b2)
        bpm.SUMK_C1 = float(args.sumk_c1)
        bpm.SUMK_C2 = float(args.sumk_c2)
    if args.driven:
        bpm.DRIVEN_MODE = int(args.driven)
        bpm.DRIVEN_SEED_RATE = float(args.driven_seed_rate)
        print(f"DRIVEN ON: steadily-seeded cascade, seed_rate={bpm.DRIVEN_SEED_RATE} "
              f"(f_dep={args.f_dep} reinterpreted as depth); source term r(x;t)!=0",
              flush=True)
    if args.kernel_2g:
        bpm.KERNEL_2G_ON = True
        bpm.KERNEL_2G_SIGQ = float(args.kernel_2g_sigq)
        bpm.KERNEL_2G_GAMMA = float(args.kernel_2g_gamma)
        bpm.KERNEL_2G_B = float(args.kernel_2g_b)
        bpm.KERNEL_2G_MSTAR = float(args.kernel_2g_mstar)
        print(f"KERNEL_2G ON: sigq={bpm.KERNEL_2G_SIGQ} gamma={bpm.KERNEL_2G_GAMMA} "
              f"b2={bpm.KERNEL_2G_B} mstar={bpm.KERNEL_2G_MSTAR}", flush=True)

    PL = importlib.util.spec_from_file_location(
        "pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
    pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
    if args.natal_peak_frac > 0.0:
        pl.NATAL_PEAK_FRAC = float(args.natal_peak_frac)
        pl.NATAL_PEAK_MU = float(args.natal_peak_mu)
        pl.NATAL_PEAK_SIG = float(args.natal_peak_sig)
        print(f"WIDE NATAL: channel-A = power-law + low-mass peak "
              f"(frac={pl.NATAL_PEAK_FRAC}, mu={pl.NATAL_PEAK_MU}, sig={pl.NATAL_PEAK_SIG})",
              flush=True)
    M = pl.M
    M.F_DEP = float(args.f_dep)
    M.V_ESC = float(args.v_esc)
    M.KERNEL_A = float(args.kernel_a)
    M.KERNEL_B = float(args.kernel_b)
    from mass_pdet_interp import make_real_pdet_fn
    pdet = make_real_pdet_fn(); T = float(M.T_OBS)
    KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B",
          "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]

    post = np.load(os.path.join(POST_ROOT, args.post, "two_channel_pl_post.npz"))
    npost = len(post["m_max_A"])
    idx = np.linspace(0, npost - 1, args.n_draw).astype(int)
    bk = jax.random.PRNGKey(0)

    m1L, m2L, chieL, wL, dwL, isAL = [], [], [], [], [], []
    mu_j, n60_j = [], []
    for j in range(args.n_draw):
        M.PROPOSAL_KEY = jax.random.fold_in(bk, j)
        M.SAMPLER_KEY = jax.random.fold_in(bk, 1000 + j)
        M.SPIN_ORIENT_KEY = jax.random.fold_in(bk, 2000 + j)
        comb, nA = pl.make_combined_pl(args.n_seed, args.n_b)
        th = {k: float(post[k][idx[j]]) for k in KS}
        # two_channel_pl_demo.combined() now takes v_esc as a required kwarg
        # (free-v_esc refit).  For the forward gallery v_esc is FIXED at
        # M.V_ESC (== args.v_esc), matched to the fixed-v_esc posterior.
        if "v_esc" not in KS:
            th["v_esc"] = float(M.V_ESC)
        if args.alpha_a is not None:
            th["alpha_A"] = float(args.alpha_a)   # scan override (posterior pins 0)
        s, lw = comb(**th)
        s = np.asarray(s); w = np.exp(np.asarray(lw))
        pd = np.asarray(pdet(jnp.asarray(s)))
        m1 = np.maximum(s[:, 0], s[:, 1]); m2 = np.minimum(s[:, 0], s[:, 1])
        chie = (s[:, 0] * s[:, 2] + s[:, 1] * s[:, 3]) / (s[:, 0] + s[:, 1])
        dw = w * pd
        isA = np.zeros(len(s), bool); isA[:nA] = True
        ok = np.isfinite(m1) & (w > 0)
        m1L.append(m1[ok]); m2L.append(m2[ok]); chieL.append(chie[ok])
        wL.append(w[ok]); dwL.append(dw[ok]); isAL.append(isA[ok])
        mu_j.append(T * dw[ok].sum())
        n60_j.append(T * dw[ok][m1[ok] > 60].sum())
        print(f"  draw {j}: kept {ok.sum()}  mu_j={mu_j[-1]:.1f}  "
              f"Ndet60_j={n60_j[-1]:.2f}", flush=True)

    m1 = np.concatenate(m1L); m2 = np.concatenate(m2L)
    chie = np.concatenate(chieL); w = np.concatenate(wL)
    dw = np.concatenate(dwL); isA = np.concatenate(isAL)
    q = m2 / m1
    mu = T * dw.sum() / args.n_draw

    # observed comparison
    cat = np.load(CAT, allow_pickle=True); cen = cat["centers"]
    eta = np.clip(cen[:, 1], 1e-6, .25)
    s4 = np.sqrt(np.clip(1 - 4 * eta, 0, 1))
    Mt = cen[:, 0] / eta ** .6
    m1o = Mt / 2 * (1 + s4); m2o = Mt / 2 * (1 - s4); qo = m2o / m1o

    def wmed(x, ww):
        if len(x) == 0 or ww.sum() <= 0:
            return float("nan")
        o = np.argsort(x); c = np.cumsum(ww[o])
        return float(x[o][np.searchsorted(c, c[-1] / 2)])

    scal = dict(mu=mu, T=T, n_draw=args.n_draw)
    print(f"\n=== {args.tag}  mu={mu:.0f} (obs 238) ===")
    for t in (40, 60, 80):
        sel = m1 > t
        Ndet = mu * (dw[sel].sum() / dw.sum()) if dw.sum() > 0 else 0.0
        mq = wmed(q[sel], dw[sel])
        Nobs = int((m1o > t).sum()); mqo = float(np.median(qo[m1o > t]))
        scal[f"Ndet{t}"] = Ndet; scal[f"detq{t}"] = mq
        print(f"  N_det(>{t})={Ndet:6.1f} (obs {Nobs:3d})   "
              f"det_med_q={mq:.2f} (obs {mqo:.2f})")

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.tag}.npz")
    np.savez_compressed(
        out,
        m1=m1.astype(np.float32), q=q.astype(np.float32),
        chieff=chie.astype(np.float32),
        w=w.astype(np.float32), dw=dw.astype(np.float32), isA=isA,
        mu_j=np.asarray(mu_j), n60_j=np.asarray(n60_j),
        config=json.dumps(vars(args)),
        **{k: float(v) for k, v in scal.items()},
    )
    print(f"wrote {out}  [{time.time()-t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
