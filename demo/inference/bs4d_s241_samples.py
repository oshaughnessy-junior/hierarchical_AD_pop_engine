"""Sample-based S241011k term at fixed population draws (run on cardassia only; the reference stays there).
Reference: 3D likelihood reference (source Mc, eta, chi_eff; inverse-prior weights; z integrated under dVc/dz/(1+z)).
L_samp(x) = sum_s w_s K_h(x - x_s) / sum_s w_s (Gaussian kernel, Scott bandwidth from the Kish ESS, coordinates standardised by the
weighted sample sd). lnI_samp = logsumexp_j (log w_j + log L_samp(x_j)) over the population draws saved by bs4d_s241_fitterm.py.
Records the draw ESS, the effective number of reference samples under the dominant draws, the distance of the dominant draws to the
nearest sample (in kernel bandwidths), and a bootstrap sd (100 weighted resamples of the reference rows).
Usage: bs4d_s241_samples.py REF_NPZ OUT_CSV FITTERM_NPZ [...]"""
import csv, json, os, sys
import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree

REF, OUT = sys.argv[1], sys.argv[2]; FT = sys.argv[3:]
r = np.load(REF); xs, ws = np.asarray(r["x"], float), np.asarray(r["w"], float); ws = ws / ws.sum()
cols = ("mc_source", "eta", "chi_eff")
mu = (ws[:, None] * xs).sum(0); sd = np.sqrt((ws[:, None] * (xs - mu) ** 2).sum(0)); neff = 1.0 / np.sum(ws ** 2)
h = neff ** (-1.0 / 7.0)                                     # Scott, d = 3
us = (xs - mu) / sd; lognorm = -1.5 * np.log(2 * np.pi) - 3 * np.log(h) - np.sum(np.log(sd))
q = lambda p: [float(np.interp(p, np.cumsum(ws[np.argsort(xs[:, i])]), xs[np.argsort(xs[:, i]), i])) for i in range(3)]
print("reference rows", len(xs), "Kish ESS %.0f, bandwidth %.3f sd; q01 %s q50 %s q99 %s" % (neff, h, np.round(q(.01), 3), np.round(q(.5), 3), np.round(q(.99), 3)), flush=True)
lo, hi = us.min(0) - 8 * h, us.max(0) + 8 * h


def logL(xd, wsel=None, idx=None, chunk=512):
    """log L_samp at draws xd (n,3), vectorised in chunks; draws outside the padded box get -inf.
    Also returns the effective number of samples under the kernel and the nearest-sample distance (bandwidths)."""
    ud = (xd - mu) / sd; inside = np.where(np.all((ud > lo) & (ud < hi), 1))[0]
    out = np.full(len(xd), -np.inf); nef = np.zeros(len(xd)); dmin = np.full(len(xd), np.inf)
    w_ = ws if wsel is None else wsel; u_ = us if idx is None else us[idx]; lw_ = np.log(np.maximum(w_, 1e-300)); uu = np.sum(u_ ** 2, 1)
    for c0 in range(0, len(inside), chunk):
        sel = inside[c0:c0 + chunk]; ud_ = ud[sel]
        d2 = (np.sum(ud_ ** 2, 1)[:, None] + uu[None, :] - 2 * ud_ @ u_.T) / h ** 2; d2 = np.maximum(d2, 0)
        lk = lw_[None, :] - 0.5 * d2; m = lk.max(1); p = np.exp(lk - m[:, None])
        out[sel] = m + np.log(p.sum(1)) + lognorm; nef[sel] = p.sum(1) ** 2 / np.sum(p ** 2, 1); dmin[sel] = np.sqrt(d2.min(1))
    return out, nef, dmin


rows = []
rng = np.random.default_rng(20261126)
for f in FT:
    z = np.load(f, allow_pickle=True); tag = str(z["tag"])
    for k in sorted({k.rsplit("|", 1)[0] for k in z.files if k.endswith("|draws")}):
        D = np.asarray(z[f"{k}|draws"], float); xd, lw = D[:, :3], D[:, 3]
        ll, nef, dmin = logL(xd); c = lw + ll; tot = logsumexp(c)
        if not np.isfinite(tot):
            dd, _ = cKDTree(us / h).query((xd - mu) / sd / h); j = int(np.argmin(dd)); pw = np.exp(lw - logsumexp(lw))
            wq = lambda p_, i: float(np.interp(p_, np.cumsum(pw[np.argsort(xd[:, i])]), xd[np.argsort(xd[:, i]), i]))
            rows.append(dict(tag=tag, config=k, lnI_samp=float("-inf"), ess_draws=0, note="no draw inside the reference support",
                             draws_min_dist_h=float(dd[j]), closest_draw_chi=float(xd[j, 2]), closest_draw_Mc=float(xd[j, 0]), closest_draw_eta=float(xd[j, 1]),
                             draws_chi_q50=wq(.5, 2), draws_chi_q99=wq(.99, 2), draws_frac_chi_ge_0p45=float(pw[xd[:, 2] >= 0.45].sum()),
                             lnI3fit=float(z[f"{k}|lnI3fit"]), ess3fit=float(z[f"{k}|ess3fit"]), lnI4=float(z[f"{k}|lnI4"]), ess4=float(z[f"{k}|ess4"]),
                             lnI3nal=float(z[f"{k}|lnI3nal"]), ess3nal=float(z[f"{k}|ess3nal"]), kappa=float(z[f"{k}|kappa"])))
            print(tag, k, "no support: min dist %.1f h, closest draw chi %.3f Mc %.2f eta %.3f; draws chi q50 %.3f q99 %.3f, weight chi>=0.45 %.2e; fit4D %.2f (ess %.1f) NAL %.2f" % (
                dd[j], xd[j, 2], xd[j, 0], xd[j, 1], rows[-1]["draws_chi_q50"], rows[-1]["draws_chi_q99"], rows[-1]["draws_frac_chi_ge_0p45"], rows[-1]["lnI4"], rows[-1]["ess4"], rows[-1]["lnI3nal"]), flush=True)
            continue
        ess = float(np.exp(2 * tot - logsumexp(2 * c[np.isfinite(c)])))
        order = np.argsort(-c); top = order[:5]; share = np.exp(c[top] - tot)
        rel = np.where(c > tot - 25)[0]                         # draws that matter, for the bootstrap
        bs = []
        for _ in range(100):
            idx = rng.choice(len(xs), size=len(xs), p=ws); wb = np.full(len(xs), 1.0 / len(xs))
            lb, _, _ = logL(xd[rel], wsel=wb, idx=idx); bs.append(logsumexp(lw[rel] + lb))
        rows.append(dict(tag=tag, config=k, lnI_samp=float(tot), ess_draws=ess, boot_sd=float(np.std(bs, ddof=1)), n_relevant_draws=int(len(rel)),
                         top_share=";".join(f"{s:.2f}" for s in share), top_chi=";".join(f"{xd[i, 2]:.3f}" for i in top), top_Mc=";".join(f"{xd[i, 0]:.2f}" for i in top),
                         top_samples_under_kernel=";".join(f"{nef[i]:.1f}" for i in top), top_dist_to_nearest_sample_h=";".join(f"{dmin[i]:.2f}" for i in top),
                         lnI3fit=float(z[f"{k}|lnI3fit"]), ess3fit=float(z[f"{k}|ess3fit"]), lnI4=float(z[f"{k}|lnI4"]), ess4=float(z[f"{k}|ess4"]),
                         lnI3nal=float(z[f"{k}|lnI3nal"]), ess3nal=float(z[f"{k}|ess3nal"]), kappa=float(z[f"{k}|kappa"])))
        print(tag, k, "samp %.2f (ess %.1f, boot sd %.2f) fit3D %.2f fit4D %.2f NAL %.2f | top chi %s, samples under kernel %s, dist %s" % (
            tot, ess, rows[-1]["boot_sd"], rows[-1]["lnI3fit"], rows[-1]["lnI4"], rows[-1]["lnI3nal"], rows[-1]["top_chi"], rows[-1]["top_samples_under_kernel"], rows[-1]["top_dist_to_nearest_sample_h"]), flush=True)
keys = list(dict.fromkeys(k for r in sorted(rows, key=lambda r: -len(r)) for k in r))
with open(OUT, "w", newline="") as fo:
    w = csv.DictWriter(fo, keys, restval=""); w.writeheader(); [w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}) for r in rows]
ref = dict(reference=REF, rows=int(len(xs)), kish_ess=float(neff), bandwidth_sd_units=float(h), weighted_q01=q(.01), weighted_q50=q(.5), weighted_q99=q(.99), coords=cols)
json.dump(ref, open(OUT.replace(".csv", "_reference_summary.json"), "w"), indent=1)
