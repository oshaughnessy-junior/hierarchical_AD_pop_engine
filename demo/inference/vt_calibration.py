"""Absolute-rate calibration from the O3+O4a+O4b sensitivity-injection set.

Turns a population MASS distribution p(m1,m2) into a real merger rate
R0 [Gpc^-3 yr^-1] with NO free normalization, using the LVK/Essick injection-VT
recipe (vt_data/vt_doc.md, arXiv:2508.10638):

    VT = T_obs/N_gen * sum_found weights * exp( ln p(m1,m2) + bar(z) + ln p_s(spins)
                                                 + kappa*ln(1+z) - lnpdraw )     [Gpc^3 yr]
    R0 = n_obs / VT                                                              [Gpc^-3 yr^-1]

where bar(z) = ln[(1+z)^-1 dVc/dz] is the comoving source-frame 4-volume element,
p_s is a NORMALIZED population spin density (isotropic + uniform magnitude; the
detection prob depends on the spin SHAPE at <1%, so only its normalization -- the
factor the reduced mass_pdet_grid dropped, a 7.6x error -- matters), and
kappa*ln(1+z) imposes a fixed redshift evolution R(z)=R0 (1+z)^kappa near the LVK
median (our model does not fit kappa). p_s(s)=1/(4 pi |s|^2) per spin.

Self-test: with GWTC-5.0's own mass+redshift population this reproduces the
observed catalog count (mu ~ n_obs) to ~20%, validating every factor.

  python vt_calibration.py            # runs the GWTC-5 self-test
"""
import os, glob, numpy as np, h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
INJ = "/home/richard.oshaughnessy/coag_container/vt_data/inj_real_o3_o4a_o4b_cartesian.hdf"
_CACHE = "/home/richard.oshaughnessy/coag_container/vt_data/_vt_inj_arrays.npz"
_LNP = ("lnpdraw_mass1_source_mass2_source_redshift_"
        "spin1x_spin1y_spin1z_spin2x_spin2y_spin2z")
GW5 = os.path.expanduser("~/coag_container/vt_data/gwtc5_pop/popsummary_files")

KAPPA_LVK = 2.7        # fixed redshift-evolution slope near the LVK/GWTC median
N_OBS = 238            # NAL catalog size (O3+O4a+O4b BBH, FAR<1/yr)
SEC_PER_YR = 365.25 * 86400.0


def _load_injections():
    """Cache the FAR<1/yr found-injection arrays + precomputed volume/spin terms."""
    if os.path.exists(_CACHE):
        return dict(np.load(_CACHE))
    from astropy.cosmology import Planck15
    import astropy.units as u
    with h5py.File(INJ, "r") as o:
        ev = o["events"]; nm = ev.dtype.names
        N_gen = float(o.attrs["total_generated"])
        T = float(o.attrs["total_analysis_time"]) / SEC_PER_YR
        searches = [s.decode() if isinstance(s, bytes) else s for s in o.attrs["searches"]]
        m1 = np.asarray(ev["mass1_source"]); m2 = np.asarray(ev["mass2_source"])
        z = np.asarray(ev["redshift"]); lnpd = np.asarray(ev[_LNP]); w = np.asarray(ev["weights"])
        s1 = np.stack([np.asarray(ev[f"spin1{a}"]) for a in "xyz"], 1)
        s2 = np.stack([np.asarray(ev[f"spin2{a}"]) for a in "xyz"], 1)
        far = np.min(np.stack([np.asarray(ev[f"{s}_far"]) for s in searches
                               if f"{s}_far" in nm], 0), 0)
    sel = far < 1.0
    m1, m2, z, lnpd, w = m1[sel], m2[sel], z[sel], lnpd[sel], w[sel]
    s1, s2 = s1[sel], s2[sel]
    dVcdz = 4 * np.pi * Planck15.differential_comoving_volume(z).to(u.Gpc**3 / u.sr).value
    ln_vol = np.log(dVcdz / (1 + z))                       # bar(z), comoving 4-volume element
    # floor |a| at 0.05 (see build_mass_pdet.py): tames the 1/|a|^2 spin-density
    # variance from near-zero-spin draws; the a<0.05 volume is negligible.
    a1 = np.clip(np.linalg.norm(s1, axis=1), 0.05, 0.999)
    a2 = np.clip(np.linalg.norm(s2, axis=1), 0.05, 0.999)
    # isotropic-orientation + uniform-magnitude Cartesian spin density 1/(4 pi a^2), per spin
    ln_ps = -(np.log(4 * np.pi) + 2 * np.log(a1)) - (np.log(4 * np.pi) + 2 * np.log(a2))
    d = dict(m1=m1, m2=m2, z=z, lnpd=lnpd, w=w, ln_vol=ln_vol, ln_ps=ln_ps,
             N_gen=np.float64(N_gen), T=np.float64(T))
    np.savez(_CACHE, **d)
    return d


def compute_VT(p_mass_fn, kappa=KAPPA_LVK, inj=None):
    """VT [Gpc^3 yr] for a population whose (m1,m2) density is p_mass_fn (normalized
    so int p dm1 dm2 = 1), assuming R(z) ~ (1+z)^kappa and iso-uniform spins."""
    d = inj if inj is not None else _load_injections()
    ln_pmass = np.log(np.clip(p_mass_fn(d["m1"], d["m2"]), 1e-300, None))
    ln_integrand = ln_pmass + d["ln_vol"] + d["ln_ps"] + kappa * np.log(1 + d["z"]) - d["lnpd"]
    return float(d["T"] * np.sum(d["w"] * np.exp(ln_integrand)) / d["N_gen"])


def rate_R0(p_mass_fn, n_obs=N_OBS, kappa=KAPPA_LVK, inj=None):
    """Self-calibrated z=0 rate R0 [Gpc^-3 yr^-1] = n_obs / VT."""
    return n_obs / compute_VT(p_mass_fn, kappa=kappa, inj=inj)


def mass_density_from_samples(m1, m2, w, ngrid=90):
    """Build a normalized p(m1,m2) [int=1] from weighted population samples, via a
    smoothed 2D log-mass histogram (symmetrized in m1<->m2). Returns a callable."""
    from scipy.ndimage import gaussian_filter
    ok = np.isfinite(m1) & np.isfinite(m2) & (w > 0) & (m1 > 0) & (m2 > 0)
    m1, m2, w = m1[ok], m2[ok], w[ok]
    big, small = np.maximum(m1, m2), np.minimum(m1, m2)
    edges = np.linspace(np.log10(2.0), np.log10(200.0), ngrid + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:]); dl = edges[1] - edges[0]
    H, _, _ = np.histogram2d(np.log10(big), np.log10(small), bins=[edges, edges], weights=w)
    H = H + H.T - np.diag(np.diag(H))
    H = gaussian_filter(H, 1.0)
    M1c = (10.0 ** ctr)[:, None]; M2c = (10.0 ** ctr)[None, :]
    area = M1c * M2c * (np.log(10.0) * dl) ** 2           # linear (m1,m2) bin area
    dens = H / area
    dens /= np.sum(dens * area)                            # normalize int p dm1 dm2 = 1

    def p_mass(a, b):
        A, B = np.maximum(a, b), np.minimum(a, b)
        i = np.clip(np.searchsorted(edges, np.log10(np.clip(A, 2, 200))) - 1, 0, ngrid - 1)
        j = np.clip(np.searchsorted(edges, np.log10(np.clip(B, 2, 200))) - 1, 0, ngrid - 1)
        return dens[i, j]
    return p_mass


def _gw5_mass_density():
    """GWTC-5.0 p(m1,m2)=p(m1) p(q)/m1 from its rate grids (validation reference)."""
    path = glob.glob(f"{GW5}/**/*default_mmax*popsummary_result.h5", recursive=True)[0]
    with h5py.File(path, "r") as h:
        x1 = np.asarray(h["posterior/rates_on_grids/mass_1/positions"]).squeeze()
        r1 = np.nanpercentile(np.asarray(h["posterior/rates_on_grids/mass_1/rates"]), 50, 0)
        xq = np.asarray(h["posterior/rates_on_grids/mass_ratio/positions"]).squeeze()
        rq = np.nanpercentile(np.asarray(h["posterior/rates_on_grids/mass_ratio/rates"]), 50, 0)
    n = lambda x, r: np.clip(r, 0, None) / np.trapezoid(np.clip(r, 0, None), x)
    r1n, rqn = n(x1, r1), n(xq, rq)

    def p_mass(a, b):
        m1, m2 = np.maximum(a, b), np.minimum(a, b)
        q = np.clip(m2 / m1, 1e-3, 1.0)
        return np.where(m1 > 0, np.interp(m1, x1, r1n, 0, 0) * np.interp(q, xq, rqn, 0, 0) / m1, 0.0)
    return p_mass


def _self_test():
    inj = _load_injections()
    p = _gw5_mass_density()
    VT = compute_VT(p, inj=inj)
    mu = 18.3 * VT
    R0 = rate_R0(p, inj=inj)
    print(f"[vt_calibration self-test]  kappa={KAPPA_LVK}, T={float(inj['T']):.4f} yr, "
          f"N_gen={float(inj['N_gen']):.3e}, N_found={inj['m1'].size}")
    print(f"  GWTC-5 mass pop: VT={VT:.4g} Gpc^3 yr")
    print(f"  predicted mu at GWTC-5 rate (18.3) = {mu:.1f}   (observed catalog ~{N_OBS})")
    print(f"  back-calibrated R0(GWTC-5 mass, {N_OBS} ev) = {R0:.4g} Gpc^-3 yr^-1  (cf 18.3)")
    ok = 0.7 < mu / N_OBS < 1.4
    print(f"  SELF-TEST {'PASS' if ok else 'FAIL'} (mu/N_obs = {mu/N_OBS:.2f}, want 0.7-1.4)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
