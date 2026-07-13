"""Posterior corner for the free-m_max two-channel NAL fit (headline f_dep=0.9,
flat-natal alpha_A=0). Reproducibility figure for the real-data results section.

alpha_A is a fixed constant in the headline (flat-natal) fits, so it has zero
posterior variance; such pinned parameters are dropped automatically (corner
cannot plot a delta column) and their fixed value is noted in the title.
"""
import argparse, numpy as np, corner
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
LAB = [r"$\alpha_A$", r"$m_{\max}$", r"$\sigma_\chi$", r"$\mu_B$", r"$\sigma_B$",
       r"$\mu_{\chi B}$", r"$\sigma_{\chi B}$", r"$\log R_A$", r"$\log R_B$"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--post", default="/home/richard.oshaughnessy/coag_container/figures_nal_f090_fa0w/two_channel_pl_post.npz")
    p.add_argument("--out", default="/home/richard.oshaughnessy/unixhome/Projects/LIGO-ScienceMode/communications/20260516-Me-CoagulationEngine/gwk-hierarchical-initial/paper/figures/real_o3o4_corner_nal.png")
    p.add_argument("--fdep", type=float, default=0.9, help="cascade brightness for the title (must match --post)")
    args = p.parse_args()
    z = np.load(args.post)
    # Drop pinned (zero-variance) parameters -- e.g. the fixed flat-natal alpha_A --
    # which corner cannot render as a delta column; note them in the title.
    keep = [k for k in KS if np.std(z[k]) > 1e-9]
    fixed = [(k, float(np.mean(z[k]))) for k in KS if np.std(z[k]) <= 1e-9]
    lab = [LAB[KS.index(k)] for k in keep]
    data = np.column_stack([z[k] for k in keep])
    fig = corner.corner(data, labels=lab, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_fmt=".2f", title_kwargs={"fontsize": 8}, label_kwargs={"fontsize": 10},
                        smooth=1.0, plot_datapoints=True, data_kwargs={"alpha": 0.3})
    fixed_note = (r", " + ", ".join(LAB[KS.index(k)] + r"$=%.2g$" % v for k, v in fixed)) if fixed else ""
    fig.suptitle((r"Free-$m_{\max}$ two-channel NAL fit ($F_{\rm dep}=%.1f$" % args.fdep) + fixed_note + r", 238 events)", fontsize=13, y=1.0)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
