"""Redshift population, z-resolved selection and per-event integrals for native4d fits.

Population: p(z | kappa) = dVc/dz (1+z)^(kappa-1) / N(kappa) on (0, ZMAX_POP], independent
of mass and spin, in the fits' cosmology FlatLambdaCDM(67.9, 0.3065, Tcmb0=0).
The rate parameter R is then the merger rate per unit detector-frame time over z < 2.5,
and mu = R * T_obs * E_pop[ xi_kappa(m1, m2) ] with

    xi_kappa(m1, m2) = int pdet(m1, m2, z) p(z | kappa) dz,

pdet trilinear in (m1, m2, z) on the refinedz grid (linear in mass between the log-spaced
nodes, as in RebootAuto's combined_pdet_grid_loader.pdet_eval). Integrating the
piecewise-linear z interpolant exactly gives xi_kappa = sum_k c_k(kappa) pdet[:, :, k],
c_k = int hat_k p dz, then bilinear in (m1, m2). Masses are ordered (m1 >= m2) and clamped
to [2, 120]; the grid is zero for m2 > m1.

Per-event 4D integral, for population draws (Mc_j, eta_j, chi_j) with log weights lw_j:

    I_i = sum_j w_j int dz p(z|kappa) L_i(Mc_j, eta_j, chi_j, z)
        = sum_j w_j int dlog(Mc_det) (Mc_det / Mc_j) p(z_jk) L_i(..., z_jk),  z = Mc_det / Mc_j - 1.

The inner integral uses fixed Gauss-Legendre nodes in log Mc_det over the event's
detector-chirp-mass range (native4d_event_tables.py): L_i is narrow in Mc_det, so a
z grid would need thousands of nodes for low-mass events, while this one is fixed per event.
No z sampling, so no added Monte Carlo noise.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import native4d_likelihood as N4

jax.config.update("jax_enable_x64", True)
ZMAX_POP = 2.5


def install_rift_stub():
    """gwkokab.errors imports RIFT.lalsimutils for MsunInSec only; RIFT is absent on cardassia.
    Register a minimal stand-in so the engine imports. No-op when RIFT is installed."""
    import sys, types
    try:
        import RIFT.lalsimutils  # noqa: F401
        return False
    except ImportError:
        rift = types.ModuleType("RIFT"); lsu = types.ModuleType("RIFT.lalsimutils")
        lsu.MsunInSec = 4.925490947641267e-06          # G Msun / c^3 [s] (lal.MTSUN_SI)
        rift.lalsimutils = lsu; sys.modules["RIFT"] = rift; sys.modules["RIFT.lalsimutils"] = lsu
        return True
_MASKED = -1e10            # finite stand-in for log 0 inside logsumexp (keeps grads finite)

_xg, _wg = np.polynomial.legendre.leggauss(256)
_ZN = jnp.asarray(ZMAX_POP * (_xg + 1) / 2); _WN = jnp.asarray(ZMAX_POP * _wg / 2)


def log_norm_pz(kappa):
    return logsumexp(N4.log_dvc_dz_sr(_ZN) + (kappa - 1) * jnp.log1p(_ZN) + jnp.log(_WN))


def log_pz(z, kappa, log_norm=None, dc=None):
    """log p(z|kappa), normalized on (0, ZMAX_POP]; _MASKED outside. dc: comoving_distance(z)."""
    ln = log_norm_pz(kappa) if log_norm is None else log_norm
    ok = (z > 0) & (z <= ZMAX_POP)
    zs = jnp.where(ok, z, 0.5)
    if dc is None:
        ldv = N4.log_dvc_dz_sr(zs)
    else:
        ldv = math.log(N4.DH) + 2 * jnp.log(jnp.where(ok, dc, 1.0)) - jnp.log(N4.E(zs))
    return jnp.where(ok, ldv + (kappa - 1) * jnp.log1p(zs) - ln, _MASKED)


# ---- selection ---------------------------------------------------------------
class ZSelection:
    """xi_kappa(m1, m2) from pdet(m1, m2, z) on the refinedz grid."""

    def __init__(self, grid_npz, floor=0.0):
        g = np.load(grid_npz, allow_pickle=True)
        self.lm = np.log(np.asarray(g["m1_grid"])); lm2 = np.log(np.asarray(g["m2_grid"]))
        assert np.allclose(self.lm, lm2)
        self.zg = np.asarray(g["z_grid"]); self.P = jnp.asarray(np.asarray(g["pdet"]))
        self.floor = float(floor)
        self.mmin, self.mmax = float(np.exp(self.lm[0])), float(np.exp(self.lm[-1]))
        a, b = self.zg[:-1], self.zg[1:]; x, w = np.polynomial.legendre.leggauss(16)
        zz = (a + b)[:, None] / 2 + (b - a)[:, None] / 2 * x; ww = (b - a)[:, None] / 2 * w
        t = (zz - a[:, None]) / (b - a)[:, None]
        self._zz, self._ww, self._t = jnp.asarray(zz), jnp.asarray(ww), jnp.asarray(t)
        self.T_obs_years = None
        try:
            import json
            self.T_obs_years = float(json.loads(str(g["metadata_json"]))["T_obs_years"])
        except Exception:
            pass

    def zweights(self, kappa):
        """c_k(kappa) = int hat_k(z) p(z|kappa) dz over the grid's z range; sum c_k <= 1."""
        p = jnp.exp(log_pz(self._zz, kappa)) * self._ww          # (nint, 16)
        left = jnp.sum(p * (1 - self._t), 1); right = jnp.sum(p * self._t, 1)
        return jnp.concatenate([left, jnp.zeros(1)]) + jnp.concatenate([jnp.zeros(1), right])

    def xi_table(self, kappa):
        return jnp.tensordot(self.P, self.zweights(kappa), axes=([2], [0]))

    def xi(self, samples, kappa, table=None):
        """samples (n, >=2) source-frame (m1, m2, ...) -> xi_kappa(m1, m2)."""
        X = self.xi_table(kappa) if table is None else table
        a, b = samples[:, 0], samples[:, 1]
        m1 = jnp.clip(jnp.maximum(a, b), self.mmin, self.mmax); m2 = jnp.clip(jnp.minimum(a, b), self.mmin, self.mmax)
        n = X.shape[0]; mg = jnp.exp(jnp.asarray(self.lm))       # log-spaced above 3, finer below
        i1 = jnp.clip(jnp.searchsorted(mg, m1, side="right") - 1, 0, n - 2)
        i2 = jnp.clip(jnp.searchsorted(mg, m2, side="right") - 1, 0, n - 2)
        t1 = jnp.clip((m1 - mg[i1]) / (mg[i1 + 1] - mg[i1]), 0.0, 1.0)
        t2 = jnp.clip((m2 - mg[i2]) / (mg[i2 + 1] - mg[i2]), 0.0, 1.0)
        v = ((1 - t1) * (1 - t2) * X[i1, i2] + t1 * (1 - t2) * X[i1 + 1, i2]
             + (1 - t1) * t2 * X[i1, i2 + 1] + t1 * t2 * X[i1 + 1, i2 + 1])
        return jnp.maximum(v, self.floor)

    def clamp_fraction(self, samples, log_w):
        """Fraction of population weight with m1 > grid max (clamped)."""
        w = jnp.exp(log_w - logsumexp(log_w))
        m1 = jnp.maximum(samples[:, 0], samples[:, 1]); m2 = jnp.minimum(samples[:, 0], samples[:, 1])
        return dict(m1_above=float(jnp.sum(w * (m1 > self.mmax))), m2_below=float(jnp.sum(w * (m2 < self.mmin))))


# ---- per-event integrals -----------------------------------------------------
def mc_eta_chieff(samples):
    m1, m2, c1, c2 = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3]
    mt = m1 + m2; eta = m1 * m2 / mt ** 2
    return jnp.stack([mt * eta ** 0.6, eta, (m1 * c1 + m2 * c2) / mt], 1)


def gl_nodes(lo, hi, n):
    """Per-event GL nodes/log-weights in log Mc_det: arrays (n_ev, n)."""
    x, w = np.polynomial.legendre.leggauss(n)
    lo = np.asarray(lo)[:, None]; hi = np.asarray(hi)[:, None]
    t = (lo + hi) / 2 + (hi - lo) / 2 * x; lw = np.log((hi - lo) / 2 * w)
    return jnp.asarray(t), jnp.asarray(lw)


def event_inner(P, t, lwt, x, kappa, log_norm):
    """log int dz p(z) L(x_j, z) for one event; x (n,3) = (Mc, eta, chi). Returns (n,)."""
    mc = x[:, 0:1]
    z = jnp.exp(t)[None, :] / mc - 1.0                                      # (n, K)
    dc = N4.comoving_distance(jnp.clip(z, 0.0, 5.9))
    pts = jnp.concatenate([jnp.broadcast_to(x[:, None, :], z.shape + (3,)), z[..., None]], -1)
    ll = N4.log_likelihood(pts, P, dc)
    ll = jnp.where(jnp.isfinite(ll), ll, _MASKED)
    a = lwt[None, :] + t[None, :] - jnp.log(mc) + log_pz(z, kappa, log_norm, dc) + ll
    return logsumexp(a, 1)


def event_log_integrals(stacked, t, lwt, x, log_w, kappa, nA=None, checkpoint=True):
    """Per event: log I_i, log sum (w_j inner_ij)^2 and (if nA) log I_i over channel A.

    Sequential over events (lax.map) with remat, so reverse-mode memory stays per event.
    """
    log_norm = log_norm_pz(kappa)

    def one(args):
        P, ti, li = args
        a = log_w + event_inner(P, ti, li, x, kappa, log_norm)
        out = [logsumexp(a), logsumexp(2 * a)]
        if nA is not None:
            out.append(logsumexp(a[:nA]))
        return jnp.stack(out)

    f = jax.checkpoint(one) if checkpoint else one
    return jax.lax.map(f, (stacked, t, lwt))


class WindowedIntegrals:
    """event_log_integrals restricted, per event, to population draws with source chirp mass in
    [mcs_lo, mcs_hi] (native4d_event_tables.py). Draws are sorted by Mc at each call and each event
    reads a static-length slice; slice lengths are set from a reference population (1.3x count + 64)
    and grouped into buckets. `overflow` > 0 in the diagnostics means a slice was too short.
    """

    def __init__(self, stacked, t, lwt, mcs_lo, mcs_hi, mc_ref, margin=1.3, pad=64):
        self.P, self.t, self.lwt = stacked, t, lwt
        self.lo, self.hi = jnp.asarray(mcs_lo), jnp.asarray(mcs_hi)
        mref = np.sort(np.asarray(mc_ref)); n = len(mref)
        cnt = np.searchsorted(mref, np.asarray(mcs_hi), "right") - np.searchsorted(mref, np.asarray(mcs_lo))
        ladder = np.unique(np.minimum(n, np.round(2 ** np.arange(6, np.log2(n) + 1.01, 0.5)).astype(int)).tolist() + [n])
        need = np.minimum(np.ceil(margin * cnt + pad), n)
        W = ladder[np.searchsorted(ladder, need)]
        self.buckets = [(np.where(W == w)[0], int(w)) for w in np.unique(W)]
        self.counts_ref, self.n_ref = cnt, n
        self.cost_fraction = float(sum(len(i) * w for i, w in self.buckets) / (n * len(W)))

    def __call__(self, x, log_w, kappa, nA=None, checkpoint=True):
        ln = log_norm_pz(kappa); n = x.shape[0]
        order = jnp.argsort(x[:, 0]); xs = x[order]; lws = log_w[order]
        isA = order < (nA if nA is not None else 0)
        start = jnp.searchsorted(xs[:, 0], self.lo); count = jnp.searchsorted(xs[:, 0], self.hi, side="right") - start
        out = jnp.zeros((len(self.lo), 3)); over = jnp.zeros(len(self.lo), jnp.int32)
        for ev, W in self.buckets:
            def one(args, W=W):
                P, ti, li, st, cn = args
                idx = jnp.clip(st + jnp.arange(W), 0, n - 1); ok = jnp.arange(W) < cn
                a = jnp.where(ok, lws[idx] + event_inner(P, ti, li, xs[idx], kappa, ln), _MASKED)
                return jnp.stack([logsumexp(a), logsumexp(2 * a), logsumexp(jnp.where(isA[idx], a, _MASKED))])
            f = jax.checkpoint(one) if checkpoint else one
            sub = {k: v[ev] for k, v in self.P.items()}
            r = jax.lax.map(f, (sub, self.t[ev], self.lwt[ev], start[ev], count[ev]))
            out = out.at[ev].set(r); over = over.at[ev].set(jnp.maximum(count[ev] - W, 0))
        return out, over


def nal3d_log_integrals(centers, covs, x, log_w, nA=None):
    """Old 3D NAL (GenieMcEtaChieffLikelihood with cov): log I, log sum a^2, log I_A per event."""
    covs = 0.5 * (covs + np.swapaxes(covs, 1, 2)) + 1e-12 * np.eye(3)
    prec = jnp.asarray(np.linalg.inv(covs)); logdet = jnp.asarray(np.linalg.slogdet(covs)[1])
    centers = jnp.asarray(centers)

    def one(args):
        c, pr, ld = args
        d = x - c
        ll = -0.5 * jnp.einsum("ni,ij,nj->n", d, pr, d) - 0.5 * ld - 1.5 * math.log(2 * math.pi)
        a = log_w + ll
        out = [logsumexp(a), logsumexp(2 * a)]
        if nA is not None:
            out.append(logsumexp(a[:nA]))
        return jnp.stack(out)
    return jax.lax.map(one, (centers, prec, logdet))


def load_event_tables(path, tiers=("primary",)):
    z = np.load(path, allow_pickle=True)
    keep = np.isin(z["tier"], list(tiers))
    P = {k[2:]: jnp.asarray(z[k][keep]) for k in z.files if k.startswith("P_")}
    meta = {k: z[k][keep] for k in ["catalog_index", "name", "tier", "collection", "analysis_id",
                                     "lmcd_lo", "lmcd_hi", "validation", "mcs_lo", "mcs_hi"] if k in z.files}
    return P, meta


class HybridPoissonLikelihood:
    """Inhomogeneous-Poisson population log-likelihood with native4d events and 3D NAL fallbacks.

        ln L = sum_i ln I_i + N ln T_obs - mu,   mu = T_obs sum_j w_j xi_kappa(m1_j, m2_j),

    the same assembly as gwkokab's sampled_poisson_likelihood_fn. Native4d events integrate z
    against p(z|kappa); fallback events keep their 3D NAL, whose z was integrated under the PE
    prior, so they do not respond to kappa.
    """

    def __init__(self, catalog, tables, pdet_z_grid, tiers=("primary",), nodes=48, T_obs=None, mc_ref=None):
        names = [str(n) for n in catalog["names"]]
        self.P, meta = load_event_tables(tables, tiers)
        idx = np.asarray(meta["catalog_index"]).astype(int)
        bad = [(i, str(n)) for i, n in zip(idx, meta["name"]) if names[i] != str(n)]
        if bad:
            raise ValueError("event table does not match this catalog: %r" % bad[:3])
        self.t, self.lwt = gl_nodes(meta["lmcd_lo"], meta["lmcd_hi"], nodes)
        rest = np.setdiff1d(np.arange(len(names)), idx)
        self.centers3 = np.asarray(catalog["centers"])[rest]; self.covs3 = np.asarray(catalog["covs"])[rest]
        self.n4, self.n3, self.n = len(idx), len(rest), len(names)
        self.names4 = [names[i] for i in idx]; self.names3 = [names[i] for i in rest]
        self.sel = ZSelection(pdet_z_grid)
        self.T_obs = float(T_obs if T_obs is not None else self.sel.T_obs_years)
        self.window = None
        if mc_ref is not None:
            self.window = WindowedIntegrals(self.P, self.t, self.lwt, meta["mcs_lo"], meta["mcs_hi"], mc_ref)

    def __call__(self, samples, log_w, kappa, checkpoint=True):
        x = mc_eta_chieff(samples)
        over = jnp.zeros(1, jnp.int32)
        if self.window is None:
            e4 = event_log_integrals(self.P, self.t, self.lwt, x, log_w, kappa, checkpoint=checkpoint)
        else:
            e4, over = self.window(x, log_w, kappa, checkpoint=checkpoint)
        e3 = nal3d_log_integrals(self.centers3, self.covs3, x, log_w)
        log_mu = math.log(self.T_obs) + logsumexp(log_w + jnp.log(self.sel.xi(samples, kappa)))
        lnL = jnp.sum(e4[:, 0]) + jnp.sum(e3[:, 0]) + self.n * math.log(self.T_obs) - jnp.exp(log_mu)
        return lnL, dict(log_I4=e4[:, 0], ess4=jnp.exp(2 * e4[:, 0] - e4[:, 1]),
                         log_I3=e3[:, 0], ess3=jnp.exp(2 * e3[:, 0] - e3[:, 1]), mu=jnp.exp(log_mu),
                         window_overflow=jnp.max(over))
