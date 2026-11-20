"""Windowed vs full native4d event integrals, and timing of value/gradient.

At the headline-mean population and two posterior draws: max |ln I_window - ln I_full|
over the 189 native4d events, slice overflow, and wall time of forward and
value_and_grad (w.r.t. the population log-weights and draw positions) for both.
Output: runs/A_rescore/window_check.json
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path.home() / '20261120-hier-reassess'
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import native4d_rescore as R  # noqa: E402
import jax, jax.numpy as jnp  # noqa: E402
import native4d_population as NP  # noqa: E402


def main(n_seed=int(sys.argv[1]) if len(sys.argv) > 1 else 3000, n_b=int(sys.argv[2]) if len(sys.argv) > 2 else 3000):
    post = np.load(ROOT / 'inputs/headline_combined_vesc1000_post.npz', allow_pickle=True)
    P, meta = NP.load_event_tables(ROOT / 'runs/A_tables/native4d_events.npz', ('primary',))
    t, l = NP.gl_nodes(meta['lmcd_lo'], meta['lmcd_hi'], 48)
    pl, M, comb, nA = R.setup_engine(post, 11, float(post['f_dep']), n_seed, n_b)
    pops = []
    for i in [None, 0, 399]:
        th = {k: float(np.asarray(post[k]).mean() if i is None else post[k][i]) for k in R.COLS}; th['v_esc'] = 1000.0
        s, lw = comb(**th); pops.append((jnp.asarray(s), jnp.asarray(lw)))
    x0 = NP.mc_eta_chieff(pops[0][0])
    W = NP.WindowedIntegrals(P, t, l, meta['mcs_lo'], meta['mcs_hi'], np.asarray(x0[:, 0]))
    kap = jnp.asarray(2.7)
    full = jax.jit(lambda x, lw: NP.event_log_integrals(P, t, l, x, lw, kap, nA=nA, checkpoint=False))
    win = jax.jit(lambda x, lw: W(x, lw, kap, nA=nA, checkpoint=False))
    gfull = jax.jit(jax.value_and_grad(lambda x, lw: jnp.sum(NP.event_log_integrals(P, t, l, x, lw, kap, checkpoint=True)[:, 0]), (0, 1)))
    gwin = jax.jit(jax.value_and_grad(lambda x, lw: jnp.sum(W(x, lw, kap, checkpoint=True)[0][:, 0]), (0, 1)))
    res = dict(n_seed=n_seed, n_b=n_b, n_samples=int(x0.shape[0]), buckets=[(len(e), w) for e, w in W.buckets],
               cost_fraction=W.cost_fraction, pops=[])
    for k, (s, lw) in enumerate(pops):
        x = NP.mc_eta_chieff(s)
        t0 = time.time(); a = np.asarray(full(x, lw)); tf = time.time() - t0
        t0 = time.time(); b, over = win(x, lw); b = np.asarray(b); tw = time.time() - t0
        d = np.abs(a[:, 0] - b[:, 0])
        res['pops'].append(dict(pop=['mean', 'post_000', 'post_399'][k], max_abs_dlnI=float(d.max()),
                                worst_event=str(meta['name'][int(d.argmax())]), max_abs_dlnI_A=float(np.abs(a[:, 2] - b[:, 2]).max()),
                                overflow=int(np.max(np.asarray(over))), t_full_s=tf, t_window_s=tw))
        print(res['pops'][-1], flush=True)
    s, lw = pops[0]; x = NP.mc_eta_chieff(s)
    for tag, g in [('window', gwin), ('full', gfull)]:
        t0 = time.time(); v, gr = g(x, lw); jax.block_until_ready(gr); t1 = time.time() - t0
        t0 = time.time(); v, gr = g(x, lw); jax.block_until_ready(gr); t2 = time.time() - t0
        res[f'grad_{tag}'] = dict(first_s=t1, repeat_s=t2, value=float(v))
        print(tag, res[f'grad_{tag}'], flush=True)
    (ROOT / 'runs/A_rescore/window_check_%d_%d.json' % (n_seed, n_b)).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
