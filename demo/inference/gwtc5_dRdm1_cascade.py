"""ALTERNATIVE version of Fig. 6 panel (a) [real_gwtc5_overlay_nal, top-left
dR/dm1], built to *better illustrate the cascade*. Instead of the natal+cascade
(A) / aligned (B) channel split, this decomposes the hierarchical channel A by
generation and adds the implied natal consumption rate:

  * total        -- all channel-A mergers, every generation (the observable
                    dR/dm1 that the headline overlay shows).
  * 1G x 1G      -- primordial mergers only: both progenitors are natal,
                    g1 = g2 = 1.
  * natal (cons) -- mass distribution of ALL generation-1 progenitors entering
                    ANY merger (both binary slots with g == 1), weighted by the
                    same rate weight. This is the rate at which *natal* black
                    holes are consumed. It is LARGER than the 1Gx1G merger rate
                    exactly insofar as 2G+ systems form: every 2G merger spends
                    extra natal black holes further up the merger tree, so the
                    natal budget exceeds the primordial-merger count.

Channel B (aligned low-mass, no cascade) is purely 1G by construction and is
drawn thin for context. The GWTC-5.0 dR/dm1 is overlaid (grey), with the model
TOTAL normalized to the GWTC-5.0 integrated rate over [5,80] Msun exactly as in
gwtc5_overlay_nal.py; the 1Gx1G and natal curves share that single scale factor,
so the three curves sit on one common (arbitrary) absolute rate axis.

This is an alternative/pedagogical figure kept out of the main draft for now.
Regenerate on GPU (or with the OOM-safe wrapper on this host):

  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=4 \
    taskset -c 0-3 python gwtc5_dRdm1_cascade.py --n-seed 4000 --n-draw 6

Definition note: "natal (cons)" is one defensible reading of dR_natal/dm1 (natal
BHs consumed, per unit natal mass). A fuller unfolding would recurse each 2G+
progenitor back to its natal leaves; the direct-progenitor version here is the
simplest curve that is unambiguously >= the 1Gx1G rate. Adjust if a different
convention is wanted.
"""
import argparse, importlib.util, os, glob, numpy as np, h5py
import jax, jax.numpy as jnp
from scipy.stats import gaussian_kde

_HERE = os.path.dirname(__file__)
PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
M = pl.M
M.ISOTROPIC_SPIN = False        # keep magnitudes; we only need masses+gens here

KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]


def make_sampler_A_with_gens(n_seed):
    """Channel-A sampler that RETAINS the merger generation labels (cols 4,5 of
    out['mergers']) which the production sampler discards. Config is a verbatim
    copy of pl.make_sampler_A_mmax so the physics cannot drift."""
    m_prop, chi_prop, log_q = M.make_proposal(n_seed)

    def proposal(_key):
        return m_prop, chi_prop, log_q

    def _cfg(weight_array):
        return M.MCConfig(
            m_min=jnp.asarray(M.M_MIN), m_max=jnp.asarray(pl.M_MAX_PROP),
            natal_spin_loc=jnp.asarray(M.CFG_SPIN_LOC0),
            natal_spin_scale=jnp.asarray(M.CFG_SPIN_SCALE0),
            kernel_a=jnp.asarray(M.KERNEL_A), kernel_b=jnp.asarray(M.KERNEL_B),
            v_esc=jnp.asarray(M.V_ESC), n_seed=int(n_seed),
            depletion_fraction=float(M.F_DEP),
            mu_natal=jnp.asarray(M.CFG_MU0), sigma_natal=jnp.asarray(M.CFG_SIGMA0),
            proposal=proposal, weight_array=weight_array)

    log_p_cfg = M._natal_log_pdf_jax(m_prop, chi_prop, _cfg(None))

    def sampler(alpha, m_max, spin_scale, log_rate):
        log_p_target = (pl._log_p_mass_pl_cutoff(m_prop, alpha, m_max)
                        + M.log_p_spin(chi_prop, spin_scale))
        weight_array = jnp.exp(log_p_target - log_p_cfg)
        out = M.particle_mc_jax_soft(_cfg(weight_array), M.SAMPLER_KEY)
        mrg = np.asarray(out["mergers"])           # (n_steps, 8): m1 m2 c1 c2 g1 g2 ret m_rem
        w = np.asarray(out["w_mergers"]) * np.asarray(out["step_active"])
        w = w / max(w.sum(), 1e-300) * float(np.exp(log_rate))
        return mrg, w

    return sampler


def gw_dm1(path):
    with h5py.File(path, "r") as h:
        g = h["posterior/rates_on_grids/mass_1"]
        x = np.asarray(g["positions"]).squeeze(); r = np.asarray(g["rates"])
    return x, np.nanpercentile(r, [5, 50, 95], axis=0)


def wkde(x, w, grid, bw=0.10):
    ok = np.isfinite(x) & (w > 0)
    if w[ok].sum() < 1e-12:
        return np.zeros_like(grid)
    return gaussian_kde(x[ok], weights=w[ok], bw_method=bw)(grid) * w[ok].sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--post", default="/home/richard.oshaughnessy/coag_container/figures_nal_f060/two_channel_pl_post.npz")
    p.add_argument("--f-dep", type=float, default=0.60)
    p.add_argument("--n-seed", type=int, default=8000)
    p.add_argument("--n-b", type=int, default=6000)
    p.add_argument("--n-draw", type=int, default=25)
    p.add_argument("--out", default=os.path.join(_HERE, "figures_nal_f060", "real_gwtc5_dRdm1_cascade.png"))
    args = p.parse_args()
    M.F_DEP = args.f_dep

    sampA = make_sampler_A_with_gens(args.n_seed)
    sampB = pl.tc.make_channelB_sampler(args.n_b)
    post = np.load(args.post)
    idx = np.linspace(0, len(post["alpha_A"]) - 1, min(args.n_draw, len(post["alpha_A"]))).astype(int)

    m1g = np.linspace(3, 100, 120)
    tot, prim, natal, chB = [], [], [], []
    for i in idx:
        th = {k: float(post[k][i]) for k in KS}
        mrg, wA = sampA(th["alpha_A"], th["m_max_A"], th["spin_chi"], th["log_R_A"])
        m1, m2, g1, g2 = mrg[:, 0], mrg[:, 1], mrg[:, 4], mrg[:, 5]
        prim_mask = (g1 <= 1.5) & (g2 <= 1.5)                 # both natal (g==1)
        tot.append(wkde(m1, wA, m1g))                          # all channel-A mergers
        prim.append(wkde(m1, wA * prim_mask, m1g))             # 1G x 1G only
        # natal BHs consumed: every g==1 progenitor slot, at its natal mass
        nat_m = np.concatenate([m1, m2]); nat_w = np.concatenate([wA * (g1 <= 1.5), wA * (g2 <= 1.5)])
        natal.append(wkde(nat_m, nat_w, m1g))
        sB, lwB = sampB(th["mu_B"], th["sig_B"], th["mu_chiB"], th["sig_chiB"], th["log_R_B"])
        sB = np.asarray(sB); wB = np.exp(np.asarray(lwB)); mB = np.maximum(sB[:, 0], sB[:, 1])
        chB.append(wkde(mB, wB, m1g))
    tot, prim, natal, chB = map(np.array, (tot, prim, natal, chB))

    # normalize model TOTAL (A+B) to the GWTC-5 integrated rate over [5,80]; one
    # shared scale for all curves.
    base = os.path.expanduser("~/coag_container/vt_data/gwtc5_pop/popsummary_files")
    gw_def = glob.glob(f"{base}/**/*default_mmax*popsummary_result.h5", recursive=True)[0]
    x_m, (mlo, mmd, mhi) = gw_dm1(gw_def)
    sel = (x_m >= 5) & (x_m <= 80); gw_int = np.trapezoid(mmd[sel], x_m[sel])
    tot_med = np.median(tot + chB, 0); ms = (m1g >= 5) & (m1g <= 80)
    sc = gw_int / np.trapezoid(tot_med[ms], m1g[ms])
    for z in (tot, prim, natal, chB):
        z *= sc

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, a0 = plt.subplots(figsize=(6.4, 5.0))
    a0.fill_between(x_m, mlo, mhi, color="grey", alpha=0.3, label="GWTC-5.0 90%"); a0.plot(x_m, mmd, "k", lw=2)
    tt = np.median(tot + chB, 0)
    a0.plot(m1g, tt, "C0", lw=2.4, label="model total (all gen.)")
    a0.plot(m1g, np.median(natal, 0), "C4-.", lw=1.8, label=r"natal consumed $dR_{\rm natal}/dm_1$")
    a0.plot(m1g, np.median(prim, 0), "C3--", lw=1.8, label=r"$1\mathrm{G}\times1\mathrm{G}$ (primordial)")
    a0.plot(m1g, np.median(chB, 0), "C2:", lw=1.4, label="aligned B (1G)")
    a0.set_yscale("log"); a0.set_xlim(4, 90); a0.set_ylim(tt.max() * 3e-4, tt.max() * 3)
    a0.set_xlabel(r"$m_1\ [M_\odot]$"); a0.set_ylabel(r"$dR/dm_1$ (rate, arb. norm.)")
    a0.set_title(r"Cascade decomposition of $dR/dm_1$ ($F_{\rm dep}=%.2f$)" % args.f_dep)
    a0.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight"); print("wrote", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
