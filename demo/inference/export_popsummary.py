"""Export the coagulation two-channel result to a `popsummary`-format HDF5 file,
so hierarchical_AD_engine (and any GWTC-style tool) can consume it exactly like a
GWTC-5.0 popsummary_result.h5.

popsummary is a FILE-FORMAT CONVENTION (no python package); we write the same
layout our own gw() reader (marginal_ppd.py) uses on the GWTC-5.0 files:

  root attrs: hyperparameters, hyperparameter_latex_labels, model_names,
              log_bayes_factor, log_evidence
  /posterior/hyperparameter_samples                 (n_samp, n_hyper)  float64
  /posterior/rates_on_grids/<param>/positions       (1, n_grid)        float64
  /posterior/rates_on_grids/<param>/rates           (n_samp, n_grid)   float64
      attrs: parameters=[<param>], normalised=False
  /prior/   (empty)

Each rates[i] is dR/d<param> [Gpc^-3 yr^-1 / unit] for posterior draw i, in REAL
units: per draw we self-calibrate the total z=0 rate R0 from the O3+O4a+O4b
injection VT (vt_calibration; validated to ~20% on a GWTC-5 self-test) and set
dR/dx = R0 * (normalized marginal of x). This is a genuine merger-rate density,
NOT an arbitrary norm.

  python export_popsummary.py --n-draw 40 --f-dep 0.90
"""
import argparse, importlib.util, os, math, numpy as np, h5py
import jax
from scipy.stats import gaussian_kde
import vt_calibration as V

_HERE = os.path.dirname(os.path.abspath(__file__))
PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
M = pl.M; M.ISOTROPIC_SPIN = False
ROOT = "/home/richard.oshaughnessy/coag_container"
KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
LATEX = {"alpha_A": r"$\alpha_A$", "m_max_A": r"$m_{\max,A}$", "spin_chi": r"$\chi_A$",
         "mu_B": r"$\mu_B$", "sig_B": r"$\sigma_B$", "mu_chiB": r"$\mu_{\chi,B}$",
         "sig_chiB": r"$\sigma_{\chi,B}$", "log_R_A": r"$\ln R_A$", "log_R_B": r"$\ln R_B$"}

# grids matched to the GWTC-5.0 popsummary parameter names
GRIDS = {
    "mass_1":     np.linspace(3.0, 100.0, 200),
    "mass_ratio": np.linspace(0.02, 1.0, 100),
    "chi_eff":    np.linspace(-1.0, 1.0, 200),
    "a_1":        np.linspace(0.0, 1.0, 100),
    "cos_tilt_1": np.linspace(-1.0, 1.0, 100),
}
BW = {"mass_1": 0.06, "mass_ratio": 0.05, "chi_eff": 0.05, "a_1": 0.06, "cos_tilt_1": 0.06}


def _draw_population(th, rng):
    """Forward-sample one posterior draw -> per-event (m1,m2,q,chi_eff,a1,cos_t1,w)."""
    comb, nA = pl.make_combined_pl(3000, 2500)
    s, lw = comb(**th); s = np.asarray(s); w = np.exp(np.asarray(lw))
    m0, m1c, c0, c1 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    isA = np.zeros(len(s), bool); isA[:nA] = True
    prim = m0 >= m1c
    m1 = np.where(prim, m0, m1c); m2 = np.where(prim, m1c, m0)
    # channel A isotropic tilts; channel B aligned (cos t = 1)
    ct = rng.uniform(-1, 1, (len(s), 2))
    cos0 = np.where(isA, ct[:, 0], 1.0); cos1 = np.where(isA, ct[:, 1], 1.0)
    a1_prim = np.abs(np.where(prim, c0, c1)); cost_prim = np.where(prim, cos0, cos1)
    chieff = (m0 * np.abs(c0) * cos0 + m1c * np.abs(c1) * cos1) / (m0 + m1c)
    q = np.clip(m2 / np.maximum(m1, 1e-3), 1e-3, 1.0)
    ok = np.isfinite(m1) & np.isfinite(m2) & (w > 0)
    return dict(m1=m1[ok], m2=m2[ok], mass_1=m1[ok], mass_ratio=q[ok], chi_eff=chieff[ok],
                a_1=a1_prim[ok], cos_tilt_1=cost_prim[ok], w=w[ok])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--f-dep", type=float, default=0.90)
    p.add_argument("--rundir", default=None, help="posterior dir (default figures_nal_f0<fd>)")
    p.add_argument("--n-draw", type=int, default=40)
    p.add_argument("--out", default=os.path.join(ROOT, "figures_nal_f090",
                                                 "coagulation_f090_popsummary_result.h5"))
    args = p.parse_args()
    M.F_DEP = args.f_dep
    rundir = args.rundir or os.path.join(ROOT, f"figures_nal_f{int(round(args.f_dep*100)):03d}")
    post = np.load(os.path.join(rundir, "two_channel_pl_post.npz"))
    n = len(post["m_max_A"]); idx = np.linspace(0, n - 1, min(args.n_draw, n)).astype(int)
    inj = V._load_injections()
    base = jax.random.PRNGKey(0); rng = np.random.default_rng(3)

    hyper = np.zeros((len(idx), len(KS)))
    rates = {g: np.zeros((len(idx), GRIDS[g].size)) for g in GRIDS}
    R0s = []
    for row, i in enumerate(idx):
        M.PROPOSAL_KEY = jax.random.fold_in(base, row)
        M.SAMPLER_KEY = jax.random.fold_in(base, 100000 + row)
        th = {k: float(post[k][i]) for k in KS}
        hyper[row] = [th[k] for k in KS]
        # STALENESS REPAIR (2026-08-04): make_combined_pl's `combined()` gained a
        # REQUIRED keyword-only `v_esc` after this script was last run, so it had
        # been failing outright. Take it per-draw from the posterior when present,
        # else M.V_ESC (= 300 km/s) -- the committed default these runs were
        # fitted under (they do not pass --v-esc, whose default IS M.V_ESC).
        # NOT 1000: that would silently change the physics, not repair the API.
        th["v_esc"] = float(np.asarray(post["v_esc"])[i]) if "v_esc" in post.files else float(M.V_ESC)
        d = _draw_population(th, rng)
        R0 = V.rate_R0(V.mass_density_from_samples(d["m1"], d["m2"], d["w"]), inj=inj)
        R0s.append(R0)
        for g in GRIDS:
            x = np.clip(d[g], GRIDS[g][0], GRIDS[g][-1])
            pk = np.clip(gaussian_kde(x, weights=d["w"], bw_method=BW[g])(GRIDS[g]), 0, None)
            Z = np.trapezoid(pk, GRIDS[g])
            rates[g][row] = R0 * pk / Z if Z > 0 else 0.0     # dR/dx = R0 * normalized marginal
        print(f"  draw {row+1}/{len(idx)}  R0={R0:.3g} Gpc^-3 yr^-1", flush=True)
    print(f"median self-calibrated R0 = {np.median(R0s):.4g} "
          f"[{np.percentile(R0s,5):.3g}, {np.percentile(R0s,95):.3g}] Gpc^-3 yr^-1")

    sdt = h5py.string_dtype()
    with h5py.File(args.out, "w") as f:
        f.attrs["hyperparameters"] = np.array(KS, dtype=sdt)
        f.attrs["hyperparameter_latex_labels"] = np.array([LATEX[k] for k in KS], dtype=sdt)
        f.attrs["model_names"] = np.array(["coagulation_two_channel_powerlaw"], dtype=sdt)
        f.attrs["log_bayes_factor"] = np.nan
        f.attrs["log_evidence"] = np.nan
        f.attrs["f_dep"] = float(args.f_dep)
        f.attrs["rate_units"] = "Gpc^-3 yr^-1 per unit parameter (z=0, kappa=%.2f)" % V.KAPPA_LVK
        pst = f.create_group("posterior")
        pst.create_dataset("hyperparameter_samples", data=hyper, dtype="f8")
        rog = pst.create_group("rates_on_grids")
        for g in GRIDS:
            grp = rog.create_group(g)
            grp.attrs["parameters"] = np.array([g], dtype=sdt)
            grp.attrs["normalised"] = np.bool_(False)     # these are real rate densities
            grp.create_dataset("positions", data=GRIDS[g][None, :], dtype="f8")
            grp.create_dataset("rates", data=rates[g], dtype="f8")
        f.create_group("prior")
    print(f"wrote {args.out}")

    # round-trip: read it back the way gw()/hierarchical_AD_engine would
    with h5py.File(args.out, "r") as h:
        x = np.asarray(h["posterior/rates_on_grids/mass_1/positions"]).squeeze()
        r = np.asarray(h["posterior/rates_on_grids/mass_1/rates"])
        med = np.nanpercentile(r, 50, 0)
        print(f"[verify] mass_1 grid {x.shape}, rates {r.shape}, "
              f"int median dR/dm1 = {np.trapezoid(med, x):.4g} Gpc^-3 yr^-1 (= R0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
