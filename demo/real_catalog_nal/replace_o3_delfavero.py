"""Replace the 50 O3 old_summary rows with Delfavero PUBLISHED NAL fits.

Source (READ-ONLY clone): .../scratchpad/nal-data/{GWTC-1,GWTC-2,GWTC-2p1,GWTC-3}.nal.hdf5
  file -> group[EVENT, GW-named] -> group[LABEL] -> dset mean(3,)+cov(3,3), attr coords,kl.
  USE aligned3d_source:IMRPhenomXPHM:select (coords = chirp_mass_source, symmetric_mass_ratio,
  chi_eff = engine order, source-frame, prior-divided). Fallback if absent:
  Mixed:select -> SEOBNRv4PHM:select -> PublicationSamples:select -> any aligned3d_source:*:select.

Matching catalog O3 events (S-named via O3 config trigger-time GPS; GW-named by name/UTC->GPS)
to Delfavero GW-names via GPS (GW UTC name parsed to GPS with astropy). Records label used + kl.

Writes _o3_delfavero.npz (the matched O3 rows in engine coords).  READ-ONLY on the clone.
"""
import os, re, json, glob, numpy as np, h5py
from astropy.time import Time

HERE = os.path.dirname(__file__)
SCR = "/tmp/claude-40428/-home-richard-oshaughnessy-unixhome-Projects-LIGO-ScienceMode-communications-20260516-Me-CoagulationEngine/ec64a3a6-cbd1-4e03-bc7f-056a52192d52/scratchpad"
NALDIR = os.path.join(SCR, "nal-data")
CAT = os.path.join(HERE, "full_o3_o4a_o4b_catalog_NAL.npz")
OUT = os.path.join(HERE, "_o3_delfavero.npz")

FALLBACK = ["IMRPhenomXPHM", "Mixed", "SEOBNRv4PHM", "PublicationSamples"]
ENGINE_COORDS = ["chirp_mass_source", "symmetric_mass_ratio", "chi_eff"]

# Explicit overrides for events whose config GPS is preliminary/offset or whose catalog
# name is truncated vs Delfavero's full name. All verified by matching chirp mass.
OVERRIDE = {
    "S190412a": "GW190412",
    "GW190917": "GW190917_114630",
    "GW190926": "GW190926_050336",
}


def gw_name_to_gps(name):
    """GW190519_153544 -> GPS. Names without _HHMMSS (e.g. GW190521) -> date-only (noon)."""
    m = re.match(r"GW(\d{2})(\d{2})(\d{2})(?:_(\d{2})(\d{2})(\d{2}))?", name)
    if not m:
        return None
    yy, mo, dd, hh, mi, ss = m.groups()
    yr = 2000 + int(yy)
    if hh is None:
        iso = f"{yr:04d}-{mo}-{dd}T12:00:00"
    else:
        iso = f"{yr:04d}-{mo}-{dd}T{hh}:{mi}:{ss}"
    try:
        return float(Time(iso, format="isot", scale="utc").gps)
    except Exception:
        return None


def load_nal_index():
    """Map GW-name -> (file, gps). Prefer 2p1/3 over 2 over 1 on duplicate names."""
    order = ["GWTC-3", "GWTC-2p1", "GWTC-2", "GWTC-1"]
    idx = {}
    for tag in order:
        f = os.path.join(NALDIR, f"{tag}.nal.hdf5")
        if not os.path.exists(f):
            continue
        with h5py.File(f, "r") as h:
            for ev in h:
                if ev not in idx:
                    idx[ev] = (f, gw_name_to_gps(ev))
    return idx


def pick_label(grp):
    """Return (label, waveform_tag, kl) by fallback preference."""
    labs = [l for l in grp if l.startswith("aligned3d_source:") and l.endswith(":select")]
    bytag = {l.split(":")[1]: l for l in labs}
    for wf in FALLBACK:
        if wf in bytag:
            l = bytag[wf]
            return l, wf, float(np.ravel(grp[l].attrs.get("kl", [np.nan]))[0])
    if labs:
        l = labs[0]
        return l, l.split(":")[1], float(np.ravel(grp[l].attrs.get("kl", [np.nan]))[0])
    return None, None, None


def fetch(nalfile, gwname):
    with h5py.File(nalfile, "r") as h:
        if gwname not in h:
            return None
        grp = h[gwname]
        lab, wf, kl = pick_label(grp)
        if lab is None:
            return None
        g = grp[lab]
        coords = [c.decode() if isinstance(c, bytes) else c for c in g.attrs["coords"]]
        if coords != ENGINE_COORDS:
            return None, f"coords mismatch {coords}"
        mean = np.asarray(g["mean"], float)
        cov = np.asarray(g["cov"], float)
        return dict(center=mean, cov=cov, sig=np.sqrt(np.clip(np.diag(cov), 1e-12, None)),
                    label=lab, waveform=wf, kl=kl)


def main():
    c = np.load(CAT, allow_pickle=True)
    o3 = [str(c["names"][i]) for i in range(len(c["names"]))
          if c["method"][i] == "old_summary" and c["labels"][i] == "O3"]
    gpsinfo = json.load(open(os.path.join(SCR, "o3_gps.json")))
    sgps = gpsinfo["sgps"]

    nal = load_nal_index()  # GW-name -> (file, gps)
    # build gps -> gwname for matching
    nal_gps = [(gw, info[1], info[0]) for gw, info in nal.items() if info[1] is not None]

    rows, unmatched = [], []
    for ev in o3:
        # target GPS
        if ev.startswith("S"):
            tgps = sgps.get(ev)
        else:
            tgps = gw_name_to_gps(ev)
        # 0) explicit override (verified by chirp mass)
        gwmatch = None
        if ev in OVERRIDE and OVERRIDE[ev] in nal:
            gwmatch = OVERRIDE[ev]
        # 1) direct name hit (GW-named catalog events)
        if gwmatch is None and ev in nal:
            gwmatch = ev
        elif tgps is not None:
            best, bd = None, 1e9
            for gw, g, f in nal_gps:
                d = abs(g - tgps)
                if d < bd:
                    bd, best = d, gw
            if bd < 5.0:   # GPS within 5s
                gwmatch = best
            else:
                # date-only fallback (S-name vs GW date-only entries) within 1 day & unique
                cands = [gw for gw, g, f in nal_gps if abs(g - tgps) < 86400]
                if len(cands) == 1:
                    gwmatch = cands[0]
        if gwmatch is None:
            unmatched.append((ev, f"no Delfavero match (tgps={tgps})"))
            continue
        r = fetch(nal[gwmatch][0], gwmatch)
        if r is None or (isinstance(r, tuple)):
            unmatched.append((ev, f"{gwmatch}: {r[1] if isinstance(r,tuple) else 'no usable label'}"))
            continue
        r["name"] = ev; r["gwmatch"] = gwmatch
        rows.append(r)

    names = np.array([r["name"] for r in rows])
    centers = np.array([r["center"] for r in rows])
    sigs = np.array([r["sig"] for r in rows])
    covs = np.array([r["cov"] for r in rows])
    waveforms = np.array([r["waveform"] for r in rows])
    kls = np.array([r["kl"] for r in rows])
    gwmatches = np.array([r["gwmatch"] for r in rows])
    np.savez(OUT, names=names, centers=centers, sigs=sigs, covs=covs,
             waveforms=waveforms, kls=kls, gwmatches=gwmatches)

    from collections import Counter
    print(f"matched {len(rows)}/{len(o3)} O3 events")
    print("  waveform used:", dict(Counter(waveforms.tolist())))
    nonxphm = [(str(r["name"]), str(r["gwmatch"]), r["waveform"]) for r in rows if r["waveform"] != "IMRPhenomXPHM"]
    print(f"  fell back to non-XPHM ({len(nonxphm)}): {nonxphm}")
    print(f"  UNMATCHED ({len(unmatched)}): {unmatched}")
    print(f"  kl: median {np.nanmedian(kls):.4f} max {np.nanmax(kls):.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
