"""Overlay the coagulation-model intrinsic distributions on the GWTC-5.0 inferred
population, for the real-data paper figure.

Panels (2x3): (1) dR/dm1 mass function; (2) p(a) component spin magnitude;
(3) p(cos_tilt) orientation -- the direct test of our ISOTROPIC assumption;
(4) p(chi_eff); (5) sigma(chi_eff|m1) spin-vs-mass; (6) p(chi_p) (GWTC-5 only --
our aligned-spin model has no in-plane component).
GWTC-5 mass/a/cos_tilt from the default popsummary; chi_eff/chi_p from the
skew-normal chi_eff popsummary. rates_on_grids/<param>/{positions, rates} ->
5/50/95 band; shapes normalised to unit integral (rate scale differs).

  python gwtc5_overlay.py
"""
import argparse, glob, os, numpy as np, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def gw_band(path, param):
    with h5py.File(path, "r") as h:
        g = h[f"posterior/rates_on_grids/{param}"]
        x = np.asarray(g["positions"]).squeeze()
        r = np.asarray(g["rates"])
        return x, *np.nanpercentile(r, [5, 50, 95], axis=0)


def norm(x, y):
    a = np.trapezoid(y, x); return y / a if a > 0 else y


def overlay_gw(ax, path, param, xr=None):
    x, lo, md, hi = gw_band(path, param)
    if xr: sel = (x >= xr[0]) & (x <= xr[1]); x, lo, md, hi = x[sel], lo[sel], md[sel], hi[sel]
    ax.fill_between(x, norm(x, lo), norm(x, hi), color="grey", alpha=0.35, label="GWTC-5 90%")
    ax.plot(x, norm(x, md), color="k", lw=2)


def overlay_model(ax, x, band):
    ax.fill_between(x, norm(x, band[0]), norm(x, band[2]), color="C0", alpha=0.3, label="coag model 90%")
    ax.plot(x, norm(x, band[1]), color="C0", lw=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/richard.oshaughnessy/coag_container/figures_fdep030/model_intrinsic.npz")
    p.add_argument("--out", default="/home/richard.oshaughnessy/coag_container/figures_fdep030/gwtc5_overlay.png")
    args = p.parse_args()
    base = os.path.expanduser("~/coag_container/vt_data/gwtc5_pop/popsummary_files")
    gw_def = glob.glob(f"{base}/**/*default_mmax*popsummary_result.h5", recursive=True)[0]
    gw_chi = glob.glob(f"{base}/*ChiEff*/*with2Drates.h5")[0]
    m = np.load(args.model)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    overlay_gw(ax[0, 0], gw_def, "mass_1", xr=(3, 100)); overlay_model(ax[0, 0], m["m1_grid"], m["pm1"])
    ax[0, 0].set_xlim(3, 90); ax[0, 0].set_xlabel(r"$m_1\ [M_\odot]$"); ax[0, 0].set_ylabel("norm. density")
    ax[0, 0].set_title("Primary mass"); ax[0, 0].legend(fontsize=8)

    overlay_gw(ax[0, 1], gw_def, "a_1"); overlay_model(ax[0, 1], m["a_grid"], m["pa"])
    ax[0, 1].set_xlabel(r"$a$ (spin magnitude)"); ax[0, 1].set_title("Spin magnitude"); ax[0, 1].legend(fontsize=8)

    overlay_gw(ax[0, 2], gw_def, "cos_tilt_1")
    ax[0, 2].axhline(0.5, color="C0", lw=2, ls="--", label="coag model (isotropic)")
    ax[0, 2].set_xlabel(r"$\cos\theta$ (spin tilt)"); ax[0, 2].set_title("Spin orientation (isotropy test)"); ax[0, 2].legend(fontsize=8)

    overlay_gw(ax[1, 0], gw_chi, "chi_eff"); overlay_model(ax[1, 0], m["ce_grid"], m["pce"])
    ax[1, 0].set_xlabel(r"$\chi_{\rm eff}$"); ax[1, 0].set_title("Effective spin"); ax[1, 0].legend(fontsize=8)

    m1c, sb = m["m1c"], m["sig_ce_m"]
    ax[1, 1].fill_between(m1c, sb[0], sb[2], color="C0", alpha=0.3, label="coag model 90%")
    ax[1, 1].plot(m1c, sb[1], color="C0", lw=2, marker="o", ms=4)
    ax[1, 1].set_xlabel(r"$m_1\ [M_\odot]$"); ax[1, 1].set_ylabel(r"$\sigma(\chi_{\rm eff}\,|\,m_1)$")
    ax[1, 1].set_title("Spin-vs-mass (hierarchical signature)"); ax[1, 1].legend(fontsize=8)

    overlay_gw(ax[1, 2], gw_chi, "chi_p")
    ax[1, 2].set_xlabel(r"$\chi_p$"); ax[1, 2].set_title(r"Precessing spin (model: none)"); ax[1, 2].legend(fontsize=8)

    fig.suptitle("Coagulation model (F_DEP=0.30, isotropic spins) vs GWTC-5.0 inferred population", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
