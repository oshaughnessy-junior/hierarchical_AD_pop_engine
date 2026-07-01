"""Build NAL (Normal-Approximate-Likelihood) per-event terms from REAL reviewed
PE posterior samples (GWTC5-HLV production, IMRPhenomXPHM, APPROVED), for the
S-named O4a/O4b events.

Difference from the OLD catalog (planning_o4_mixed/extract_*.py):
  OLD: per-event (Mc,eta,chi_eff) moments from the source-frame POSTERIOR, with
       only the marginal mass-redshift prior deconvolved. chi_eff kept its
       POSTERIOR width -> artificially narrow at high mass (the suspected bug).
  NEW (this script): also divide out the induced chi_eff PE prior. The standard
       PE spin prior (uniform |a| + isotropic tilts) induces a chi_eff prior
       peaked at 0 (std ~0.28). Dividing it out converts the chi_eff POSTERIOR
       into the chi_eff LIKELIHOOD, which is what population inference needs and
       which is BROADER than the posterior.

Method (diagonal-compatible, drop-in for GenieMcEtaChieffLikelihood):
  * Mass/eta: same convention as OLD -- weight w_mass = 1/[(1+z)^2 * dVc/dz/(1+z)]
    (uniform detector masses x comoving-vol-per-source-time, Planck15). This
    deconvolves the marginal source-frame mass-redshift prior. Spins not in it.
  * chi_eff: build the induced chi_eff PE prior from the HDF's own prior draws
    (priors/samples/chi_eff, 5000 draws, isotropic verified) via a Gaussian KDE;
    weight w_chi = 1/p_prior(chi_eff). This is the crux fix.
  * Total weight w = w_mass * w_chi. Fit NAL by weighted mean+cov in (Mc,eta,chi_eff).
  * Store diagonal sigs (drop-in) AND the full 3x3 cov (covs) for a later upgrade.
  * Report per-event ESS.

Reads ONLY (never writes) under /home/pe.o4.
Writes a NEW file: planning_o4_mixed/full_o3_o4a_o4b_catalog_NAL.npz
"""
import os, json, numpy as np, h5py
from astropy.cosmology import Planck15
import astropy.units as u
from scipy.stats import gaussian_kde

LED = "/home/pe.o4/GWTC5-HLV/project/.asimov/ledger.yml"
RES = "/home/pe.o4/GWTC5-HLV/project/results"
OLD_CAT = "/home/richard.oshaughnessy/unixhome/Projects/LIGO-ScienceMode/communications/20260516-Me-CoagulationEngine/planning_o4_mixed/full_o3_o4a_o4b_catalog.npz"
OUT = "/home/richard.oshaughnessy/unixhome/Projects/LIGO-ScienceMode/communications/20260516-Me-CoagulationEngine/planning_o4_mixed/full_o3_o4a_o4b_catalog_NAL.npz"


def dVcdz_1pz(z):
    return 4 * np.pi * Planck15.differential_comoving_volume(z).to(
        u.Gpc**3 / u.sr).value / (1 + z)


def approved_xphm(prods):
    """Production name of the APPROVED IMRPhenomXPHM analysis (prefer SpinTaylor),
    skipping REJECTED/DEPRECATED. None if no approved XPHM."""
    cands = []
    for p in prods:
        for pname, pval in p.items():
            if not isinstance(pval, dict):
                continue
            if ((pval.get("waveform") or {}).get("approximant")) != "IMRPhenomXPHM":
                continue
            statuses = [r.get("status") for r in (pval.get("review") or [])
                        if isinstance(r, dict)]
            if "REJECTED" in statuses or "DEPRECATED" in statuses:
                continue
            if "APPROVED" in statuses:
                cands.append(pname)
    if not cands:
        return None
    st = [c for c in cands if "SpinTaylor" in c]
    return (st or cands)[0]


def wmoments(x, w):
    """Weighted mean (3,) and covariance (3,3). x:(n,3) w:(n,)"""
    w = w / w.sum()
    mu = np.sum(w[:, None] * x, 0)
    d = x - mu
    cov = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
    return mu, cov


def event_nal(name, grp, h5):
    with h5py.File(h5, "r") as h:
        g = h[grp]
        appx = np.asarray(g["approximant"]).ravel()
        appx = appx[0].decode() if isinstance(appx[0], bytes) else str(appx[0])
        if appx != "IMRPhenomXPHM":
            return None, f"approximant mismatch in HDF ({appx})"
        ps = g["posterior_samples"]
        nm = ps.dtype.names
        col = lambda *c: next((np.asarray(ps[x]) for x in c if x in nm), None)
        m1 = col("mass_1_source"); m2 = col("mass_2_source")
        z = col("redshift"); chie = col("chi_eff")
        if m1 is None or z is None or chie is None:
            return None, "missing fields"
        # induced chi_eff PE prior from the run's own prior draws
        prg = g["priors"]["samples"]
        chie_pr = np.asarray(prg["chi_eff"]) if "chi_eff" in prg else None
    if chie_pr is None:
        return None, "no chi_eff prior draws"
    ok = np.isfinite(m1) & np.isfinite(m2) & np.isfinite(z) & np.isfinite(chie) & (m1 > 1) & (m2 > 1)
    m1, m2, z, chie = m1[ok], m2[ok], z[ok], chie[ok]
    if len(m1) < 200:
        return None, f"too few samples ({len(m1)})"
    Mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    eta = (m1 * m2) / (m1 + m2) ** 2

    # mass-redshift marginal source-frame prior deconvolution (OLD convention)
    w_mass = 1.0 / ((1 + z) ** 2 * dVcdz_1pz(z))

    # chi_eff prior deconvolution via KDE of the prior draws
    chie_pr = chie_pr[np.isfinite(chie_pr)]
    kde = gaussian_kde(chie_pr, bw_method=0.15)
    p_chi = np.maximum(kde(chie), 1e-6)
    w_chi = 1.0 / p_chi

    w = w_mass * w_chi
    w = w / w.sum()
    ess = 1.0 / np.sum(w ** 2)

    x = np.stack([Mc, eta, chie], 1)
    mu, cov = wmoments(x, w)
    sig = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    # also OLD-style (mass-only) for the comparison table
    mu_old, cov_old = wmoments(x, w_mass / w_mass.sum())
    sig_old = np.sqrt(np.clip(np.diag(cov_old), 1e-12, None))
    return dict(name=name, center=mu, sig=sig, cov=cov, ess=float(ess),
                n=len(m1), mc=float(mu[0]),
                sig_old=sig_old, chi_post_sig=float(chie.std()),
                chi_prior_sig=float(chie_pr.std())), None


def main():
    import yaml
    with open(LED) as f:
        led = yaml.safe_load(f)

    rows, dropped = [], []
    for e in led["events"]:
        nm = e["name"]
        sel = approved_xphm(e.get("productions") or [])
        if not sel:
            dropped.append((nm, "no approved IMRPhenomXPHM"))
            continue
        h5 = os.path.join(RES, nm, sel, "posterior_samples.h5")
        if not os.path.exists(h5):
            dropped.append((nm, f"h5 missing for {sel}"))
            continue
        r, err = event_nal(nm, sel, h5)
        if r is None:
            dropped.append((nm, err))
            continue
        r["sel"] = sel
        rows.append(r)
        print(f"  {nm} [{sel}]: Mc={r['mc']:.1f} chi_eff sig OLD={r['sig_old'][2]:.3f} "
              f"NAL={r['sig'][2]:.3f} (post={r['chi_post_sig']:.3f}, prior={r['chi_prior_sig']:.3f}) "
              f"ESS={r['ess']:.0f}", flush=True)

    names = np.array([r["name"] for r in rows])
    # label S24 -> O4a, S25 -> O4b (S-named production events)
    labels = np.array(["O4a" if r["name"].startswith("S24") else
                       ("O4b" if r["name"].startswith("S25") else "O4") for r in rows])
    centers = np.array([r["center"] for r in rows])
    sigs = np.array([r["sig"] for r in rows])
    covs = np.array([r["cov"] for r in rows])
    ess = np.array([r["ess"] for r in rows])
    sels = np.array([r["sel"] for r in rows])

    np.savez(OUT, names=names, labels=labels, centers=centers, sigs=sigs,
             covs=covs, ess=ess, analyses=sels)
    print(f"\nWROTE {OUT}  ({len(rows)} S-named O4 events)")
    print(f"  O4a(S24)={int((labels=='O4a').sum())}  O4b(S25)={int((labels=='O4b').sum())}")
    print(f"DROPPED ({len(dropped)}):")
    for d in dropped:
        print("   ", d)

    # save diagnostics for the comparison table
    diag = {r["name"]: dict(mc=r["mc"], sig_old=r["sig_old"].tolist(),
                            sig_nal=r["sig"].tolist(), ess=r["ess"],
                            chi_post=r["chi_post_sig"], chi_prior=r["chi_prior_sig"])
            for r in rows}
    json.dump(diag, open(os.path.join(os.path.dirname(OUT), "_nal_diag.json"), "w"))
    print("wrote _nal_diag.json")


if __name__ == "__main__":
    raise SystemExit(main())
