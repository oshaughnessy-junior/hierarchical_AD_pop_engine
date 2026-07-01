"""Merge the NAL-rebuilt S-named events into a full drop-in catalog.

Start from the OLD 238-event catalog (full_o3_o4a_o4b_catalog.npz) and REPLACE
the rows whose event is one of the S-named production events we rebuilt with the
NAL (prior-deconvolved chi_eff) values. O3 (GW-named) and O4a-Zenodo (GW-named)
rows are kept VERBATIM from the OLD catalog (they still use the old summary
method -- flagged in the report; chi_eff there is NOT prior-deconvolved).

Preserves the OLD catalog's label for each event so this is a true drop-in.
Adds a `covs (N,3,3)` array (NAL events get the real weighted cov; non-rebuilt
events get diag(sigs^2)) and a `method (N,)` string array ('NAL' / 'old_summary').

Output: planning_o4_mixed/full_o3_o4a_o4b_catalog_NAL.npz  (OVERWRITES the
S-only file written by extract_nal_catalog.py with the full merged 238-ev catalog).
"""
import os, numpy as np

HERE = os.path.dirname(__file__)
OLD = os.path.join(HERE, "full_o3_o4a_o4b_catalog.npz")
NAL_S = os.path.join(HERE, "_nal_s_only.npz")   # renamed S-only intermediate
OUT = os.path.join(HERE, "full_o3_o4a_o4b_catalog_NAL.npz")


def main():
    old = np.load(OLD, allow_pickle=True)
    names = list(old["names"]); labels = list(old["labels"])
    centers = [np.asarray(c) for c in old["centers"]]
    sigs = [np.asarray(s) for s in old["sigs"]]
    ess = list(old["ess"])
    covs = [np.diag(np.asarray(s) ** 2) for s in old["sigs"]]
    method = ["old_summary"] * len(names)

    nal = np.load(NAL_S, allow_pickle=True)
    nal_idx = {str(n): i for i, n in enumerate(nal["names"])}

    replaced, added, unmatched = 0, 0, []
    pos = {n: i for i, n in enumerate(names)}
    for nm, i in nal_idx.items():
        if nm in pos:
            j = pos[nm]
            centers[j] = nal["centers"][i]
            sigs[j] = nal["sigs"][i]
            covs[j] = nal["covs"][i]
            ess[j] = float(nal["ess"][i])
            method[j] = "NAL"
            replaced += 1
        else:
            names.append(nm); labels.append(str(nal["labels"][i]))
            centers.append(nal["centers"][i]); sigs.append(nal["sigs"][i])
            covs.append(nal["covs"][i]); ess.append(float(nal["ess"][i]))
            method.append("NAL"); added += 1

    # events in OLD catalog that are S-named but were NOT rebuilt (dropped)
    s_old = [n for n in old["names"] if str(n).startswith("S")]
    s_nal = set(nal_idx)
    not_rebuilt = [n for n in s_old if str(n) not in s_nal]

    names = np.array(names); labels = np.array(labels)
    centers = np.array(centers); sigs = np.array(sigs)
    covs = np.array(covs); ess = np.array(ess); method = np.array(method)

    np.savez(OUT, names=names, labels=labels, centers=centers, sigs=sigs,
             covs=covs, ess=ess, method=method)
    print(f"merged: {len(names)} events; replaced {replaced} with NAL, added {added}")
    print(f"  method counts: NAL={int((method=='NAL').sum())} "
          f"old_summary={int((method=='old_summary').sum())}")
    print(f"  S-named in OLD not rebuilt ({len(not_rebuilt)}): {not_rebuilt}")
    Mc = centers[:, 0]
    print(f"  Mc range [{Mc.min():.1f},{Mc.max():.1f}]; Mc>30: {(Mc>30).sum()}  Mc>40: {(Mc>40).sum()}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
