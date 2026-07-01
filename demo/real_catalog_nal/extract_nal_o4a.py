"""Rebuild NAL per-event likelihoods for the 84 GW-named O4a (2023) events from the
PRODUCTION GWTC4 project posteriors (/home/pe.o4/GWTC4, READ-ONLY).

Same recipe as extract_nal_catalog.py but pointed at the GWTC4 project:
  ledger /home/pe.o4/GWTC4/.asimov/ledger.yml -> approved IMRPhenomXPHM production
  -> results/<S-EVENT>/<analysis>/posterior_samples.h5 (avoid archive/)
  -> mass-z marginal source-frame deconvolution + chi_eff PE-prior division (KDE of
     priors/samples/chi_eff). Stored under the catalog's GW-name (via GPS GW<->S map).

Writes planning_o4_mixed/_nal_o4a_only.npz with both NAL and old-style (mass-only,
SAME samples) widths so the comparison is apples-to-apples.
"""
import os, json, numpy as np, h5py
from astropy.cosmology import Planck15
import astropy.units as u
from scipy.stats import gaussian_kde

LED = "/home/pe.o4/GWTC4/.asimov/ledger.yml"
RES = "/home/pe.o4/GWTC4/results"
GWOSC = "/home/richard.oshaughnessy/coag_container/vt_data/gwosc_gwtc4.json"
KEEP = "/home/richard.oshaughnessy/coag_container/vt_data/o4a_keep_list.json"
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "_nal_o4a_only.npz")


def dVcdz_1pz(z):
    return 4 * np.pi * Planck15.differential_comoving_volume(z).to(
        u.Gpc**3 / u.sr).value / (1 + z)


def approved_xphm(prods):
    cands = []
    for p in prods:
        for pname, pval in p.items():
            if not isinstance(pval, dict):
                continue
            if ((pval.get("waveform") or {}).get("approximant")) != "IMRPhenomXPHM":
                continue
            statuses = [r.get("status") if isinstance(r, dict) else r
                        for r in (pval.get("review") or [])]
            if "REJECTED" in statuses or "DEPRECATED" in statuses:
                continue
            if "APPROVED" in statuses:
                cands.append(pname)
    if not cands:
        return None
    st = [c for c in cands if "SpinTaylor" in c]
    return (st or cands)[0]


def wmoments(x, w):
    w = w / w.sum()
    mu = np.sum(w[:, None] * x, 0)
    d = x - mu
    cov = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
    return mu, cov


def event_nal(grp, h5):
    with h5py.File(h5, "r") as h:
        g = h[grp]
        appx = np.asarray(g["approximant"]).ravel()[0]
        appx = appx.decode() if isinstance(appx, bytes) else str(appx)
        if appx != "IMRPhenomXPHM":
            return None, f"approximant mismatch ({appx})"
        ps = g["posterior_samples"]; nm = ps.dtype.names
        col = lambda *c: next((np.asarray(ps[x]) for x in c if x in nm), None)
        m1 = col("mass_1_source"); m2 = col("mass_2_source")
        z = col("redshift"); chie = col("chi_eff")
        prg = g["priors"]["samples"] if "priors" in g and "samples" in g["priors"] else None
        chie_pr = np.asarray(prg["chi_eff"]) if (prg is not None and "chi_eff" in prg) else None
    if m1 is None or z is None or chie is None:
        return None, "missing fields"
    if chie_pr is None:
        return None, "no chi_eff prior draws"
    ok = np.isfinite(m1) & np.isfinite(m2) & np.isfinite(z) & np.isfinite(chie) & (m1 > 1) & (m2 > 1)
    m1, m2, z, chie = m1[ok], m2[ok], z[ok], chie[ok]
    if len(m1) < 200:
        return None, f"few samples ({len(m1)})"
    Mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    eta = (m1 * m2) / (m1 + m2) ** 2
    w_mass = 1.0 / ((1 + z) ** 2 * dVcdz_1pz(z))
    chie_pr = chie_pr[np.isfinite(chie_pr)]
    kde = gaussian_kde(chie_pr, bw_method=0.15)
    w = w_mass / np.maximum(kde(chie), 1e-6)
    w = w / w.sum()
    ess = 1.0 / np.sum(w ** 2)
    x = np.stack([Mc, eta, chie], 1)
    mu, cov = wmoments(x, w)
    sig = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    mu_old, cov_old = wmoments(x, w_mass / w_mass.sum())
    sig_old = np.sqrt(np.clip(np.diag(cov_old), 1e-12, None))
    return dict(center=mu, sig=sig, cov=cov, ess=float(ess), n=len(m1),
                center_old=mu_old, sig_old=sig_old,
                chi_prior_sig=float(chie_pr.std())), None


def build_gw_to_s():
    import yaml
    led = yaml.safe_load(open(LED))
    s_gps = {e["name"]: float(e["event time"]) for e in led["events"]
             if e.get("event time") is not None}
    gw = json.load(open(GWOSC))["events"]
    gw_gps = {}
    for k, v in gw.items():
        nm = v.get("commonName") or k.split("-v")[0]
        if v.get("GPS") is not None:
            gw_gps[nm] = float(v["GPS"])
    keep = json.load(open(KEEP))
    mapping, unmapped = {}, []
    for ev in keep:
        if ev not in gw_gps:
            unmapped.append((ev, "no GPS")); continue
        best, bd = None, 1e9
        for s, sg in s_gps.items():
            d = abs(sg - gw_gps[ev])
            if d < bd:
                bd, best = d, s
        if bd < 2.0:
            mapping[ev] = best
        else:
            unmapped.append((ev, f"no S within 2s (nearest {best} {bd:.1f}s)"))
    return mapping, unmapped, {e["name"]: e for e in led["events"]}


def main():
    mapping, unmapped, ledmap = build_gw_to_s()
    print(f"GW->S mapped {len(mapping)}/84; unmapped {len(unmapped)}: {unmapped}")
    rows, dropped = [], list(unmapped)
    for gwname, sname in mapping.items():
        sel = approved_xphm(ledmap[sname].get("productions") or [])
        if not sel:
            dropped.append((gwname, f"{sname}: no approved XPHM")); continue
        h5 = os.path.join(RES, sname, sel, "posterior_samples.h5")
        if not os.path.exists(h5):
            dropped.append((gwname, f"{sname}/{sel}: h5 missing")); continue
        r, err = event_nal(sel, h5)
        if r is None:
            dropped.append((gwname, f"{sname}: {err}")); continue
        r["name"] = gwname; r["sname"] = sname; r["sel"] = sel
        rows.append(r)
        print(f"  {gwname} ({sname}) [{sel}]: Mc={r['center'][0]:.1f} "
              f"chi_eff OLD={r['sig_old'][2]:.3f} NAL={r['sig'][2]:.3f} "
              f"(prior {r['chi_prior_sig']:.3f}) ESS={r['ess']:.0f}", flush=True)

    names = np.array([r["name"] for r in rows])
    centers = np.array([r["center"] for r in rows])
    sigs = np.array([r["sig"] for r in rows])
    covs = np.array([r["cov"] for r in rows])
    ess = np.array([r["ess"] for r in rows])
    centers_old = np.array([r["center_old"] for r in rows])
    sigs_old = np.array([r["sig_old"] for r in rows])
    snames = np.array([r["sname"] for r in rows])
    np.savez(OUT, names=names, centers=centers, sigs=sigs, covs=covs, ess=ess,
             centers_old=centers_old, sigs_old=sigs_old, snames=snames)
    print(f"\nWROTE {OUT}  ({len(rows)} O4a events rebuilt)")
    print(f"DROPPED ({len(dropped)}): {dropped}")
    # high-mass spotlight
    print("\n=== O4a high-mass (Mc>35) OLD vs NAL chi_eff sig ===")
    for r in sorted(rows, key=lambda r: -r["center"][0]):
        if r["center"][0] > 35:
            print(f"  {r['name']:18s} Mc={r['center'][0]:6.1f}  OLD={r['sig_old'][2]:.3f} "
                  f"NAL={r['sig'][2]:.3f} ratio={r['sig'][2]/r['sig_old'][2]:.2f} "
                  f"chi_eff_mean={r['center'][2]:+.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
