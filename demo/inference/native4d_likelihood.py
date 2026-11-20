"""JAX port of the native_spin_base_redshift scalar 4D event likelihood.

Reference: fits/src/native_spin_base_redshift.py (schema
aligned-redshift-native-spin-base-posterior-v1) with the distance factor of
fits/src/native_posterior_redshift.py and the support test of
fits/src/aligned_redshift.py, at fits @e58a873.

    log L(x) = log sum_j p_j g_j(delta, chi) N_j(Mc_det, log z | u, w)
               - log(Mc_det eta^-1.2 / 2) - log z - log[p_DL(dL) dDL/dz]

for x = [Mc_source, eta, chi_eff, z], -inf outside the analysis support or beyond the
spin-cap edge.  The event constant is arbitrary (as in the reference).

Cosmology: FlatLambdaCDM(H0=67.9, Om0=0.3065, Tcmb0=0), the fits' source-time cosmology.
The comoving distance is a cubic-Hermite table (exact derivative c/H(z)) on z in [0, 6],
step 1e-3; its relative error is below 1e-12.

Distance priors present in the collections: 'distance_squared' (log p = 2 log dL) and the
common source-frame cosmology ('UniformSourceFrame' with Planck15_LAL, or
'common_source_time'), for which log p_DL + log dDL/dz = log dVc/dz - log(1+z).  The
reference evaluates the latter at z' = Pchip-inverse(dL(z)); here z is used directly.

Packing: `pack_model` turns a fit dict into a fixed-shape pytree (components padded to
KMAX with log-weight -1e30); `stack_models` stacks events on a leading axis, so
`jax.vmap(log_likelihood, (0, None))` evaluates every event at shared points.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

SCHEMA = "aligned-redshift-native-spin-base-posterior-v1"
H0, OM0 = 67.9, 0.3065
C_KMS = 299792.458
DH = C_KMS / H0                         # Hubble distance [Mpc]
KMAX = 8
SUPPORT_KEYS = ("mc", "mc_det", "m1_det", "m2_det", "mtotal_det", "q", "z")
DIST_KINDS = {"distance_squared": 0, "UniformSourceFrame": 1, "common_source_time": 1}
_NEG = -1e30

# ---- cosmology ---------------------------------------------------------------
_ZT_MAX, _ZT_N = 6.0, 6001


def _E_np(z):
    return np.sqrt(OM0 * (1 + z) ** 3 + 1 - OM0)


def _dc_table():
    z = np.linspace(0.0, _ZT_MAX, _ZT_N)
    x, w = np.polynomial.legendre.leggauss(24)
    a, b = z[:-1], z[1:]
    zz = (a + b)[:, None] / 2 + (b - a)[:, None] / 2 * x
    inc = (b - a) / 2 * np.sum(w / _E_np(zz), 1)
    return z, DH * np.concatenate([[0.0], np.cumsum(inc)])


_ZT, _DCT = _dc_table()
_ZT_H = float(_ZT[1] - _ZT[0])
_ZT_J, _DCT_J, _DDT_J = jnp.asarray(_ZT), jnp.asarray(_DCT), jnp.asarray(DH / _E_np(_ZT))


def E(z):
    return jnp.sqrt(OM0 * (1 + z) ** 3 + 1 - OM0)


def comoving_distance(z):
    """Line-of-sight comoving distance [Mpc]; cubic Hermite on the table (z <= 6)."""
    z = jnp.asarray(z)
    i = jnp.clip(jnp.floor(z / _ZT_H).astype(jnp.int32), 0, _ZT_N - 2)
    t = (z - _ZT_J[i]) / _ZT_H
    h00 = (1 + 2 * t) * (1 - t) ** 2; h10 = t * (1 - t) ** 2
    h01 = t * t * (3 - 2 * t); h11 = t * t * (t - 1)
    return (h00 * _DCT_J[i] + h10 * _ZT_H * _DDT_J[i]
            + h01 * _DCT_J[i + 1] + h11 * _ZT_H * _DDT_J[i + 1])


def luminosity_distance(z):
    return (1 + z) * comoving_distance(z)


def log_dvc_dz_sr(z):
    """log of astropy differential_comoving_volume [Mpc^3 / sr] (flat)."""
    return math.log(DH) + 2 * jnp.log(comoving_distance(z)) - jnp.log(E(z))


def log_dlprime(z):
    return jnp.log(comoving_distance(z) + (1 + z) * DH / E(z))


def distance_logfactor(z, kind):
    """log p_DL(dL(z)) + log dDL/dz, up to the reference's event constant."""
    sq = 2 * jnp.log(luminosity_distance(z)) + log_dlprime(z)
    sf = log_dvc_dz_sr(z) - jnp.log1p(z)
    return jnp.where(kind == 0, sq, sf)


# ---- packing -----------------------------------------------------------------
def _is_common_cosmology(analytic):
    """Mirror native_redshift_prior.distance_cosmology's test for the common cosmology."""
    import re
    m = re.search(r"cosmology=['\"]([A-Za-z0-9_]+)['\"]", analytic)
    if m:
        return m.group(1) == "Planck15_LAL"
    if "LambdaCDM(" in analytic:
        vals = []
        for k in ["H0", "Om0", "Ode0", "Tcmb0"]:
            v = re.search(r"\b" + k + r"\s*=\s*([-+0-9.eE]+)", analytic)
            vals.append(float(v.group(1)) if v else None)
        return vals == [67.9, .3065, .6935, 0.]
    return False


def pack_model(model, kmax=None):
    if model.get("schema") != SCHEMA:
        raise ValueError("not a native-spin-base model: %r" % model.get("schema"))
    if model.get("rho_embedded_in_scalar_likelihood", False) is not False:
        raise ValueError("scalar likelihood embeds a population redshift law")
    comps = model["components"]; K = len(comps)
    kmax = max(K, KMAX) if kmax is None else kmax
    if K > kmax:
        raise ValueError("more components than kmax")
    kind = model["prior_contract"]["distance_prior"]["kind"]
    if kind not in DIST_KINDS:
        raise ValueError("unsupported distance prior kind " + kind)
    if kind == "UniformSourceFrame" and not _is_common_cosmology(model["prior_contract"]["distance_prior"]["analytic"]):
        raise ValueError("UniformSourceFrame outside the common cosmology")
    logw = np.full(kmax, _NEG); theta = np.zeros((kmax, 5)); logZ = np.zeros(kmax)
    coef = np.zeros((kmax, 3, 2)); prec = np.tile(np.eye(2), (kmax, 1, 1)); lnorm = np.zeros(kmax)
    for j, (c, p) in enumerate(zip(comps, model["component_weights"])):
        cov = np.asarray(c["cov"], float)
        logw[j] = np.log(p); theta[j] = c["theta"]; logZ[j] = c["logZ"]; coef[j] = c["coef"]
        prec[j] = np.linalg.inv(cov); lnorm[j] = -np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(cov)[1]
    b = model.get("analysis_support", {}); amax = float(b.get("amax", 0.99))
    lo = np.array([float(b[k + "_min"]) if b.get(k + "_min") is not None else -np.inf for k in SUPPORT_KEYS])
    hi = np.array([float(b[k + "_max"]) if b.get(k + "_max") is not None else np.inf for k in SUPPORT_KEYS])
    return dict(logw=logw, theta=theta, logZ=logZ, coef=coef, prec=prec, lnorm=lnorm,
                center=np.asarray(model["center"], float), scale=np.asarray(model["scale"], float),
                caps=np.asarray(model["spin_caps"], float),
                amax=np.float64(amax), a1max=np.float64(b.get("a1max", amax)), a2max=np.float64(b.get("a2max", amax)),
                lo=lo, hi=hi, kind=np.int32(DIST_KINDS[kind]))


_PAD = dict(logw=_NEG, theta=0.0, logZ=0.0, coef=0.0, lnorm=0.0)


def _pad(p, kmax):
    k = len(p["logw"])
    if k == kmax:
        return p
    q = dict(p)
    for name, val in _PAD.items():
        extra = np.full((kmax - k,) + p[name].shape[1:], val)
        q[name] = np.concatenate([p[name], extra])
    q["prec"] = np.concatenate([p["prec"], np.tile(np.eye(2), (kmax - k, 1, 1))])
    return q


def stack_models(packed):
    """Stack packed events on a leading axis, padding components to a common count."""
    kmax = max(len(p["logw"]) for p in packed)
    packed = [_pad(p, kmax) for p in packed]
    return {k: jnp.asarray(np.stack([p[k] for p in packed])) for k in packed[0]}


def as_jax(p):
    return {k: jnp.asarray(v) for k, v in p.items()}


# ---- likelihood --------------------------------------------------------------
def support_mask(x, P):
    """aligned_redshift.support plus the spin-cap edge test of log_likelihood_x."""
    mc, eta, chi, z = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    good = jnp.isfinite(x).all(-1) & (mc > 0) & (eta > 0) & (eta <= 0.25) & (jnp.abs(chi) <= P["amax"]) & (z > 0)
    se = jnp.where(good, eta, 0.2); sm = jnp.where(good, mc, 20.0); sz = jnp.where(good, z, 0.1)
    delta = jnp.sqrt(jnp.maximum(1 - 4 * se, 0.0)); tot = sm * se ** (-0.6)
    m1 = tot * (1 + delta) / 2; m2 = tot * (1 - delta) / 2; q = m2 / m1
    good &= jnp.abs(chi) <= (P["a1max"] + q * P["a2max"]) / (1 + q)
    vals = jnp.stack([sm, sm * (1 + sz), m1 * (1 + sz), m2 * (1 + sz), tot * (1 + sz), q, sz], -1)
    good &= jnp.all((vals >= P["lo"]) & (vals <= P["hi"]), -1)
    good &= jnp.abs(chi) <= (P["caps"][0] + q * P["caps"][1]) / (1 + q)
    return good


def component_terms(mc, eta, chi, z, P):
    """log p_j + log g_j + log N_j, shape (..., KMAX)."""
    delta = jnp.sqrt(jnp.maximum(1 - 4 * eta, 0.0))
    u = (delta - P["center"][0]) / P["scale"][0]; w = (chi - P["center"][1]) / P["scale"][1]
    T = jnp.stack([u, w, u * u, u * w, w * w], -1)                        # (...,5)
    Y = jnp.stack([mc * (1 + z), jnp.log(z)], -1)                         # (...,2)
    X = jnp.stack([jnp.ones_like(u), u, w], -1)                           # (...,3)
    mean = jnp.einsum("...a,kab->...kb", X, P["coef"])                    # (...,K,2)
    r = Y[..., None, :] - mean
    quad = jnp.einsum("...ki,kij,...kj->...k", r, P["prec"], r)
    return P["logw"] + T @ P["theta"].T - P["logZ"] + P["lnorm"] - 0.5 * quad


def log_likelihood_unmasked(x, P):
    mc, eta, chi, z = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    lf = logsumexp(component_terms(mc, eta, chi, z, P), -1)
    logJ = jnp.log(mc * (1 + z)) - 1.2 * jnp.log(eta) - math.log(2.0)
    return lf - logJ - jnp.log(z) - distance_logfactor(z, P["kind"])


def log_likelihood(x, P):
    """x (..., 4) = [Mc_source, eta, chi_eff, z]; -inf outside support. Safe under grad."""
    good = support_mask(x, P)
    safe = jnp.where(good[..., None], x, jnp.array([20.0, 0.2, 0.0, 0.1]))
    return jnp.where(good, log_likelihood_unmasked(safe, P), -jnp.inf)


# vmapped over a stacked event axis, shared points
log_likelihood_events = jax.vmap(log_likelihood, in_axes=(None, 0))
