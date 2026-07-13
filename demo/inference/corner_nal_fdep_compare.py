"""Overlaid posterior corner comparing the free-m_max two-channel NAL fit across
the profiled cascade-brightness (F_dep) grid, for the stability appendix. Pure
replot from the SAVED posteriors (figures_nal_f0XX/two_channel_pl_post.npz) --
no MC / no GPU.

Each F_dep posterior is drawn as contour-only (50/90% credible) in a distinct
colour; a shared axis range is computed across all runs so the overlays align.
Shows at a glance which hyperparameters move with cascade brightness (m_max and
sigma_chi trade off against F_dep) and which are invariant (the low-mass Gaussian
mu_B/sig_B/mu_chiB/sig_chiB and the two rates).

  python corner_nal_fdep_compare.py
"""
import argparse, os, numpy as np, corner
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# alpha_A is a fixed constant (flat natal, alpha_A=0) in the headline fits, so it
# has zero posterior variance and cannot be contoured -- omit it from the overlay.
KS = ["m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
LAB = [r"$m_{\max}$", r"$\sigma_\chi$", r"$\mu_B$", r"$\sigma_B$",
       r"$\mu_{\chi_{\rm eff}B}$", r"$\sigma_{\chi_{\rm eff}B}$", r"$\log R_A$", r"$\log R_B$"]
# (F_dep, run-dir) over the profiled grid; missing dirs are skipped
RUNS = [(0.30, "figures_nal_f030_fa0w"), (0.60, "figures_nal_f060_fa0w"), (0.70, "figures_nal_f070_fa0w"),
        (0.80, "figures_nal_f080_fa0w"), (0.90, "figures_nal_f090_fa0w"), (1.00, "figures_nal_f100_fa0w"),
        (1.20, "figures_nal_f120_fa0w")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/richard.oshaughnessy/coag_container")
    p.add_argument("--out", default="/home/richard.oshaughnessy/unixhome/Projects/LIGO-ScienceMode/communications/20260516-Me-CoagulationEngine/gwk-hierarchical-initial/paper/figures/real_o3o4_corner_fdep_compare.png")
    args = p.parse_args()

    runs = [(f, d) for f, d in RUNS if os.path.exists(os.path.join(args.root, d, "two_channel_pl_post.npz"))]
    datasets = []
    for fdep, d in runs:
        z = np.load(os.path.join(args.root, d, "two_channel_pl_post.npz"))
        datasets.append((fdep, np.column_stack([z[k] for k in KS])))
    print(f"overlaying {len(datasets)} F_dep posteriors: {[f for f, _ in datasets]}")

    # shared axis range across ALL runs (1st/99th percentile per parameter, padded)
    alld = np.vstack([d for _, d in datasets])
    lo = np.percentile(alld, 1, axis=0); hi = np.percentile(alld, 99, axis=0)
    pad = 0.15 * (hi - lo); rng = list(zip(lo - pad, hi + pad))

    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(datasets)))
    fig = None
    for (fdep, data), col in zip(datasets, cmap):
        fig = corner.corner(
            data, labels=LAB, fig=fig, color=col, range=rng,
            plot_datapoints=False, plot_density=False, no_fill_contours=True,
            levels=(0.5, 0.9), smooth=1.0, hist_kwargs={"density": True},
            contour_kwargs={"linewidths": 1.1}, label_kwargs={"fontsize": 11})

    handles = [Line2D([0], [0], color=c, lw=2, label=rf"$F_{{\rm dep}}={f:.1f}$")
               for (f, _), c in zip(datasets, cmap)]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=12,
               bbox_to_anchor=(0.98, 0.98), title=r"cascade brightness")
    fig.suptitle(r"Free-$m_{\max}$ two-channel NAL fit vs cascade brightness "
                 r"$F_{\rm dep}$ (238 events, 50/90\% contours)", fontsize=14, y=1.0)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
