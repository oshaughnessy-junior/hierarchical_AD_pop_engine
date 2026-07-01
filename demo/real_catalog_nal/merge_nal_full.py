"""Merge BOTH NAL intermediates (S-named GWTC5-HLV + GW-named GWTC4 O4a) into the
full drop-in catalog, replacing the corresponding OLD-catalog rows with NAL values.

Start from the OLD 238-event catalog; replace S-named rows with _nal_s_only.npz and
GW-named O4a rows with _nal_o4a_only.npz. O3 rows stay old_summary (flagged). Adds
covs(N,3,3) and method(N,).  Output: full_o3_o4a_o4b_catalog_NAL.npz (OLD untouched).
"""
import os, numpy as np

HERE = os.path.dirname(__file__)
OLD = os.path.join(HERE, "full_o3_o4a_o4b_catalog.npz")
NAL_S = os.path.join(HERE, "_nal_s_only.npz")
NAL_O4A = os.path.join(HERE, "_nal_o4a_only.npz")
OUT = os.path.join(HERE, "full_o3_o4a_o4b_catalog_NAL.npz")


def main():
    old = np.load(OLD, allow_pickle=True)
    names = [str(n) for n in old["names"]]
    labels = list(old["labels"])
    centers = [np.asarray(c) for c in old["centers"]]
    sigs = [np.asarray(s) for s in old["sigs"]]
    ess = list(old["ess"])
    covs = [np.diag(np.asarray(s) ** 2) for s in old["sigs"]]
    method = ["old_summary"] * len(names)
    pos = {n: i for i, n in enumerate(names)}

    def apply(nal, tag):
        rep, add = 0, 0
        for k, nm in enumerate(nal["names"]):
            nm = str(nm)
            if nm in pos:
                j = pos[nm]
                centers[j] = nal["centers"][k]; sigs[j] = nal["sigs"][k]
                covs[j] = nal["covs"][k]; ess[j] = float(nal["ess"][k])
                method[j] = "NAL"; rep += 1
            else:
                names.append(nm); labels.append("O4a" if nm.startswith("GW23") or nm.startswith("GW24")
                                                else ("O4a" if nm.startswith("S24") else "O4b"))
                centers.append(nal["centers"][k]); sigs.append(nal["sigs"][k])
                covs.append(nal["covs"][k]); ess.append(float(nal["ess"][k]))
                method.append("NAL"); pos[nm] = len(names) - 1; add += 1
        print(f"  {tag}: replaced {rep}, added {add}")

    apply(np.load(NAL_S, allow_pickle=True), "S-named (GWTC5-HLV)")
    apply(np.load(NAL_O4A, allow_pickle=True), "GW-named O4a (GWTC4)")

    names = np.array(names); labels = np.array(labels)
    centers = np.array(centers); sigs = np.array(sigs)
    covs = np.array(covs); ess = np.array(ess); method = np.array(method)
    np.savez(OUT, names=names, labels=labels, centers=centers, sigs=sigs,
             covs=covs, ess=ess, method=method)

    nal_m = method == "NAL"
    Mc = centers[:, 0]
    print(f"\nTOTAL {len(names)} events; NAL={int(nal_m.sum())} old_summary={int((~nal_m).sum())}")
    # which old_summary remain
    leftover = [(names[i], labels[i]) for i in np.where(~nal_m)[0]]
    from collections import Counter
    print("  old_summary by label:", dict(Counter(l for _, l in leftover)))
    hi = Mc > 40
    print(f"  Mc>40: {hi.sum()} events; of those NAL={int((nal_m & hi).sum())} old_summary={int(((~nal_m)&hi).sum())}")
    hi35 = Mc > 35
    print(f"  Mc>35: {hi35.sum()} events; of those NAL={int((nal_m & hi35).sum())} old_summary={int(((~nal_m)&hi35).sum())}")
    os_hi = [(names[i], round(float(Mc[i]), 1)) for i in np.where((~nal_m) & hi35)[0]]
    print(f"  old_summary remaining at Mc>35: {os_hi}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
