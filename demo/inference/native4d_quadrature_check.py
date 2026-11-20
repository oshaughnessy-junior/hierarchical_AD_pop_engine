"""Check the fixed log-Mc_det Gauss-Legendre z integral against brute force.

At the headline-mean population, for selected events:
  GL48 (production), GL96 on a doubled range, a dense trapezoid in log z (20000 nodes
  over the event's z support, capped at 2.5), and Monte Carlo z ~ p(z|kappa)
  (4000 draws per population sample). Brute force runs on the samples carrying 99.99%
  of the GL48 integral (at most 600) plus 200 random others (to catch mass GL48 misses).
Output: runs/A_rescore/quadrature_check.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path.home() / '20261120-hier-reassess'
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import native4d_rescore as R  # noqa: E402  (sets sys.path, x64)
import jax, jax.numpy as jnp  # noqa: E402
from jax.scipy.special import logsumexp  # noqa: E402
import native4d_population as NP  # noqa: E402
import native4d_likelihood as N4  # noqa: E402

EVENTS = ['S241011k', 'S241113p', 'S241225v', 'S240413p', 'S241109bn', 'S241109p', 'S190521b', 'GW231123_135430']


def main(n_seed=3000, n_b=3000, kappa=2.7):
    post = np.load(ROOT / 'inputs/headline_combined_vesc1000_post.npz', allow_pickle=True)
    P, meta = NP.load_event_tables(ROOT / 'runs/A_tables/native4d_events.npz', ('primary', 'alt'))
    names = list(meta['name'])
    pl, M, comb, nA = R.setup_engine(post, 11, float(post['f_dep']), n_seed, n_b)
    base = {k: float(np.asarray(post[k]).mean()) for k in R.COLS}; base['v_esc'] = 1000.0
    s, lw = comb(**base); x = NP.mc_eta_chieff(jnp.asarray(s)); lw = jnp.asarray(lw)
    kap = jnp.asarray(kappa); ln = NP.log_norm_pz(kap)
    rng = np.random.default_rng(5); out = []
    for ev in EVENTS:
        if ev not in names:
            continue
        i = names.index(ev); Pi = {k: v[i] for k, v in P.items()}
        lo, hi = float(meta['lmcd_lo'][i]), float(meta['lmcd_hi'][i])
        t0 = time.time()
        res = {}
        for tag, n, a_, b_ in [('gl48', 48, lo, hi), ('gl96_wide', 96, lo - (hi - lo) / 2, hi + (hi - lo) / 2)]:
            t, l = NP.gl_nodes([a_], [b_], n)
            res[tag] = np.asarray(jax.jit(NP.event_inner)(Pi, t[0], l[0], x, kap, ln))
        a = np.asarray(lw) + res['gl48']; lI = float(logsumexp(a))
        o = np.argsort(-a); c = np.cumsum(np.exp(a[o] - lI)); top = o[:min(int(np.searchsorted(c, 0.9999)) + 1, 600)]
        covered = float(c[len(top) - 1])
        others = rng.choice(np.setdiff1d(np.arange(len(a)), top), min(200, len(a) - len(top)), replace=False)
        sub = np.concatenate([top, others])
        zlo = float(Pi['lo'][6]); zhi = min(float(Pi['hi'][6]), 2.5)
        zg = jnp.exp(jnp.linspace(np.log(zlo), np.log(zhi), 20000)); dlz = float(np.log(zhi / zlo) / 19999)
        tw = jnp.full(20000, dlz).at[0].mul(.5).at[-1].mul(.5)

        def dense(xj):
            pts = jnp.concatenate([jnp.broadcast_to(xj, (20000, 3)), zg[:, None]], 1)
            ll = N4.log_likelihood(pts, Pi); ll = jnp.where(jnp.isfinite(ll), ll, -1e10)
            return logsumexp(ll + NP.log_pz(zg, kap, ln) + jnp.log(zg) + jnp.log(tw))
        bf = np.asarray(jax.jit(lambda xs: jax.lax.map(dense, xs))(x[sub]))
        # Monte Carlo z ~ p(z|kappa): inverse CDF on a fine grid
        zc = np.linspace(1e-6, 2.5, 200001); pc = np.exp(np.asarray(NP.log_pz(jnp.asarray(zc), kap, ln)))
        cdf = np.cumsum(pc); cdf /= cdf[-1]
        zs = jnp.asarray(np.interp(rng.random(4000), cdf, zc))

        def mc(xj):
            pts = jnp.concatenate([jnp.broadcast_to(xj, (4000, 3)), zs[:, None]], 1)
            ll = N4.log_likelihood(pts, Pi); ll = jnp.where(jnp.isfinite(ll), ll, -1e10)
            return logsumexp(ll) - np.log(4000), logsumexp(2 * ll) - np.log(4000)
        mcv, mc2 = [np.asarray(v) for v in jax.jit(lambda xs: jax.lax.map(mc, xs))(x[sub])]
        lws = np.asarray(lw)[sub]
        tot = lambda v: float(logsumexp(lws + v))
        top_n = len(top)
        r = dict(event=ev, tier=str(meta['tier'][i]), mcdet_range=[float(np.exp(lo)), float(np.exp(hi))], z_support=[zlo, zhi],
                 n_samples=int(len(a)), n_top=int(top_n), gl48_fraction_of_I_in_top=covered,
                 lnI_gl48_all=lI, lnI_gl96wide_all=float(logsumexp(np.asarray(lw) + res['gl96_wide'])),
                 lnI_sub_gl48=tot(res['gl48'][sub]), lnI_sub_dense=tot(bf), lnI_sub_mc=tot(mcv),
                 max_abs_inner_diff_top_gl48_vs_dense=float(np.max(np.abs(res['gl48'][top] - bf[:top_n]))),
                 others_max_contribution_dense_rel=float(np.max(np.exp(lws[top_n:] + bf[top_n:] - lI))) if len(others) else 0.0,
                 mc_rel_sd_estimate=float(np.sqrt(np.sum(np.exp(2 * lws + mc2 - np.log(4000))) / np.exp(2 * tot(mcv)))),
                 seconds=time.time() - t0)
        r['dlnI_gl48_minus_dense'] = r['lnI_sub_gl48'] - r['lnI_sub_dense']
        r['dlnI_gl48_minus_mc'] = r['lnI_sub_gl48'] - r['lnI_sub_mc']
        r['dlnI_gl48_minus_gl96wide'] = r['lnI_gl48_all'] - r['lnI_gl96wide_all']
        print(json.dumps(r), flush=True); out.append(r)
    (ROOT / 'runs/A_rescore').mkdir(parents=True, exist_ok=True)
    (ROOT / 'runs/A_rescore/quadrature_check.json').write_text(json.dumps(dict(n_seed=n_seed, n_b=n_b, kappa=kappa, events=out), indent=1))


if __name__ == '__main__':
    main()
