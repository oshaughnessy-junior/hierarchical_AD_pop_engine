"""Posterior corner for the free-m_max two-channel NAL fit (preferred f_dep=0.6).
Reproducibility figure for the real-data results section.
"""
import argparse, numpy as np, corner
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
LAB = [r"$\alpha_A$", r"$m_{\max}$", r"$\sigma_\chi$", r"$\mu_B$", r"$\sigma_B$",
       r"$\mu_{\chi B}$", r"$\sigma_{\chi B}$", r"$\log R_A$", r"$\log R_B$"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--post", default="/home/richard.oshaughnessy/coag_container/figures_nal_f060/two_channel_pl_post.npz")
    p.add_argument("--out", default="/home/richard.oshaughnessy/unixhome/Projects/LIGO-ScienceMode/communications/20260516-Me-CoagulationEngine/gwk-hierarchical-initial/paper/figures/real_o3o4_corner_nal.png")
    args = p.parse_args()
    z = np.load(args.post)
    data = np.column_stack([z[k] for k in KS])
    fig = corner.corner(data, labels=LAB, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_fmt=".2f", title_kwargs={"fontsize": 8}, label_kwargs={"fontsize": 10},
                        smooth=1.0, plot_datapoints=True, data_kwargs={"alpha": 0.3})
    fig.suptitle(r"Free-$m_{\max}$ two-channel NAL fit ($F_{\rm dep}=0.6$, 238 events)", fontsize=13, y=1.0)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
