"""Numerical test of the analytic coagulation-cascade predictions:
  * BH number density  n(m) = dN/dm  ~  m^{-(3+a)/2}   (kernel-DEPENDENT)
  * merger rate         dR/dm1       ~  m^{-2}          (kernel-INDEPENDENT)
  * dR/dln m slope distinguishes constant mass flux (-1) from rate equilibrium (0).

Uses a clean deep cascade: a LOW natal cut-off (wide inertial range) + a bright
cascade, run for two kernels K=(m1+m2)^a q^b with a in {0,2} (b=2). Reads the
standing population (m_active/w_active) and the mergers (w_mergers) directly from
the particle-MC engine.

  python cascade_slope_test.py
"""
import importlib.util, os, numpy as np, jax.numpy as jnp

_HERE = os.path.dirname(__file__)
PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
M = pl.M
M.M_MAX = 70.0


def run_cascade(a, b, m_max, alpha, spin, fdep, n_seed):
    M.F_DEP = fdep
    m_prop, chi_prop, log_q = M.make_proposal(n_seed)
    def proposal(_k):
        return m_prop, chi_prop, log_q
    def cfg(warr):
        return M.MCConfig(
            m_min=jnp.asarray(M.M_MIN), m_max=jnp.asarray(70.0),
            natal_spin_loc=jnp.asarray(M.CFG_SPIN_LOC0), natal_spin_scale=jnp.asarray(M.CFG_SPIN_SCALE0),
            kernel_a=jnp.asarray(a), kernel_b=jnp.asarray(b), v_esc=jnp.asarray(M.V_ESC),
            n_seed=int(n_seed), depletion_fraction=float(fdep),
            mu_natal=jnp.asarray(M.CFG_MU0), sigma_natal=jnp.asarray(M.CFG_SIGMA0),
            proposal=proposal, weight_array=warr)
    log_p_cfg = M._natal_log_pdf_jax(m_prop, chi_prop, cfg(None))
    log_p_target = pl._log_p_mass_pl_cutoff(m_prop, alpha, m_max) + M.log_p_spin(chi_prop, spin)
    warr = jnp.exp(log_p_target - log_p_cfg)
    return M.particle_mc_jax_soft(cfg(warr), M.SAMPLER_KEY)


def loglog_slope(m, w, lo, hi, nbin=18):
    edges = np.logspace(np.log10(lo), np.log10(hi), nbin + 1); cen = np.sqrt(edges[:-1] * edges[1:])
    h, _ = np.histogram(m, edges, weights=w); y = h / np.diff(edges)   # dN/dm
    ok = (y > 0) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan, np.nan
    X = np.log(cen[ok]); Y = np.log(y[ok]); p = np.polyfit(X, Y, 1)
    r2 = 1 - np.sum((Y - np.polyval(p, X)) ** 2) / np.sum((Y - Y.mean()) ** 2)
    return p[0], r2


def main():
    m_max, alpha, spin, fdep, n_seed = 8.0, 3.0, 0.05, 1.2, 10000
    RANGES = [(15.0, 45.0), (15.0, 80.0), (25.0, 90.0)]
    print(f"clean cascade: natal cut-off {m_max}, alpha {alpha}, f_dep {fdep}, n_seed {n_seed}\n")
    for a in [0.0, 1.0, 2.0]:
        out = run_cascade(a, 2.0, m_max, alpha, spin, fdep, n_seed)
        ma = np.asarray(out["m_active"]); wa = np.asarray(out["w_active"])
        mg = np.asarray(out["mergers"]); wm = np.asarray(out["w_mergers"]) * np.asarray(out["step_active"])
        m1 = np.maximum(mg[:, 0], mg[:, 1])
        okA = np.isfinite(ma) & (wa > 0); okM = np.isfinite(m1) & (wm > 0)
        print(f"a={a:.0f}  [pred n(m)={-(3+a)/2:+.2f}, dR/dm1=-2.00]   max m: active={ma[okA].max():.0f} merger={m1[okM].max():.0f}")
        for lo, hi in RANGES:
            sn, r2n = loglog_slope(ma[okA], wa[okA], lo, hi)
            sR, r2R = loglog_slope(m1[okM], wm[okM], lo, hi)
            print(f"    {lo:.0f}-{hi:.0f}: n(m)={sn:+.2f} (R2 {r2n:.2f}) | dR/dm1={sR:+.2f} (R2 {r2R:.2f}) | dR/dlnm={sR+1:+.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
