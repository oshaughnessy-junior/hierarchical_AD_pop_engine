"""Frozen-population rescoring: old 3D NAL vs native4d (kappa fixed) event integrals.

Populations: the headline combined posterior mean and NDRAWS evenly spaced posterior
draws (as ~/20261022-nal-impact/coag_assess_v4.py), same sampler key and engine knobs.
Per population and event: log I (old 3D, new 4D), per-event ESS, channel-A share of I.
Selection: old mu = sum_j w_j xi_old(m1, m2) (kappa=2.7 z-averaged grid, floor 1e-4, T=1),
as in the headline fit; new mu = sum_j w_j xi_kappa(m1, m2) from the refinedz pdet grid
(trilinear, no floor). Both read the sampler rates as expected counts over the run
(R_new * T_obs == R_old), so the two mu are directly comparable.

Arbitrary event constants cancel only in differences at fixed likelihood, so model
comparisons use each population minus the shared mean population.

Resumable: per-population caches in runs/A_rescore/cache/.
"""
import argparse, csv, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path.home() / '20261120-hier-reassess'
ENG = ROOT / 'engine'
sys.path[:0] = [str(ENG / 'src'), str(ENG), str(ENG / 'demo/inference'), str(ROOT / 'gwkokab-sampled/src')]
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
jax.config.update('jax_enable_x64', True)
from jax.scipy.special import logsumexp  # noqa: E402
import native4d_population as NP  # noqa: E402
NP.install_rift_stub()

PRIORITY = ['S241011k', 'S241113p', 'GW231123_135430', 'S241225v', 'S240413p']
COLS = ['alpha_A', 'm_max_A', 'spin_chi', 'mu_B', 'sig_B', 'mu_chiB', 'sig_chiB', 'log_R_A', 'log_R_B', 'f_peakB', 'alpha_B']


def setup_engine(post, key, f_dep, n_seed, n_b):
    spec = importlib.util.spec_from_file_location('pl', ENG / 'demo/inference/two_channel_pl_demo.py')
    pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl); M = pl.M
    import red_team.backup_particle_mc as bpm
    M.SAMPLER_KEY = jax.random.PRNGKey(key)
    bpm.EXCHANGE_MODE = 0; bpm.KERNEL_GETA_ON = True
    for target, source in [('KERNEL_GETA_FORM', 'geta_form'), ('KERNEL_GETA_ETA0', 'geta_eta0'), ('KERNEL_GETA_P', 'geta_p'),
                           ('KERNEL_GETA_W', 'geta_w'), ('KERNEL_GETA_FLOOR', 'geta_floor')]:
        setattr(bpm, target, str(post[source]) if source == 'geta_form' else float(post[source]))
    pl.CHANNEL_B_MIX = True; pl.BMIX_MLO = float(post['bmix_mlo']); pl.BMIX_MHI = float(post['bmix_mhi'])
    M.F_DEP = f_dep; M.V_ESC = float(np.asarray(post['v_esc']).flat[0])
    comb, nA = pl.make_combined_pl(n_seed, n_b)
    return pl, M, jax.jit(comb), nA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-seed', type=int, default=8000); ap.add_argument('--n-b', type=int, default=6000)
    ap.add_argument('--draws', type=int, default=16); ap.add_argument('--key', type=int, default=11)
    ap.add_argument('--kappa', type=float, default=2.7); ap.add_argument('--nodes', type=int, default=48)
    ap.add_argument('--post', default=str(ROOT / 'inputs/headline_combined_vesc1000_post.npz'))
    ap.add_argument('--out', default=str(ROOT / 'runs/A_rescore'))
    ap.add_argument('--only-mean', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); cache = out / 'cache'; cache.mkdir(parents=True, exist_ok=True)
    post = np.load(a.post, allow_pickle=True)
    cat = np.load(ROOT / 'inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz', allow_pickle=True)
    names = [str(n) for n in cat['names']]; nev = len(names)
    xw = list(csv.DictReader(open(ROOT / 'runs/A_crosswalk/crosswalk.csv')))
    P1, m1 = NP.load_event_tables(ROOT / 'runs/A_tables/native4d_events.npz', ('primary',))
    P2, m2 = NP.load_event_tables(ROOT / 'runs/A_tables/native4d_events.npz', ('alt',))
    t1, l1 = NP.gl_nodes(m1['lmcd_lo'], m1['lmcd_hi'], a.nodes); t2, l2 = NP.gl_nodes(m2['lmcd_lo'], m2['lmcd_hi'], a.nodes)
    sel = NP.ZSelection(ROOT / 'inputs/combined_pdet_grid_mmin2_refinedz.npz')
    from mass_pdet_interp import make_real_pdet_fn
    pdet_old = make_real_pdet_fn(str(ROOT / 'inputs/mass_pdet_grid_kappa2.7.npz'))
    pl, M, comb, nA = setup_engine(post, a.key, float(post['f_dep']), a.n_seed, a.n_b)
    print(f'engine ready: nA={nA} n_b={a.n_b}; native4d primary {len(m1["name"])}, alt {len(m2["name"])}', flush=True)

    base = {k: float(np.asarray(post[k]).mean()) for k in COLS}; base['v_esc'] = float(np.asarray(post['v_esc']).flat[0])
    tasks = [('mean', base)]
    if not a.only_mean:
        for i in np.linspace(0, len(post['m_max_A']) - 1, a.draws).astype(int):
            tasks.append(('post_%03d' % i, dict({k: float(post[k][i]) for k in COLS}, v_esc=base['v_esc'])))

    kappa = jnp.asarray(a.kappa); xi_tab = sel.xi_table(kappa)
    f_new = jax.jit(lambda x, lw: NP.event_log_integrals(P1, t1, l1, x, lw, kappa, nA=nA, checkpoint=False))
    f_alt = jax.jit(lambda x, lw: NP.event_log_integrals(P2, t2, l2, x, lw, kappa, nA=nA, checkpoint=False))
    f_old = jax.jit(lambda x, lw: NP.nal3d_log_integrals(cat['centers'], cat['covs'], x, lw, nA=nA))
    rows = []
    for label, th in tasks:
        c = cache / f'{label}.npz'
        if c.exists():
            z = np.load(c); rows.append({k: z[k] for k in z.files}); print(label, 'cached', flush=True); continue
        t0 = time.time()
        s, lw = comb(**th); s = jnp.asarray(s); lw = jnp.asarray(lw)
        x = NP.mc_eta_chieff(s); tpop = time.time() - t0
        old = np.asarray(f_old(x, lw)); new = np.asarray(f_new(x, lw)); alt = np.asarray(f_alt(x, lw))
        mu_old = float(jnp.sum(jnp.exp(lw) * pdet_old(s)))
        mu_new = float(jnp.sum(jnp.exp(lw) * sel.xi(s, kappa, xi_tab)))
        cl = sel.clamp_fraction(s, lw)
        r = dict(label=label, params=json.dumps(th), old=old, new=new, alt=alt, mu_old=mu_old, mu_new=mu_new,
                 log_R=float(logsumexp(lw)), clamp_m1_above=cl['m1_above'], clamp_m2_below=cl['m2_below'],
                 seconds=time.time() - t0, seconds_pop=tpop)
        np.savez(c, **r); rows.append(r)
        print(f'{label}: pop {tpop:.1f}s total {r["seconds"]:.1f}s mu_old={mu_old:.2f} mu_new={mu_new:.2f} '
              f'clamp(m1>120)={cl["m1_above"]:.2e}', flush=True)
    finalize(a, out, rows, names, nev, xw, m1, m2, nA)


def ess(lI, lsq):
    return np.exp(2 * lI - lsq)


def finalize(a, out, rows, names, nev, xw, m1, m2, nA):
    D = len(rows); idx1 = np.asarray(m1['catalog_index']); idx2 = np.asarray(m2['catalog_index'])
    old = np.stack([np.asarray(r['old']) for r in rows])                   # (D, nev, 3)
    new = np.full_like(old, np.nan); new[:, idx1] = np.stack([np.asarray(r['new']) for r in rows])
    ext = new.copy(); ext[:, idx2] = np.stack([np.asarray(r['alt']) for r in rows])
    has4 = np.zeros(nev, bool); has4[idx1] = True; hasx = has4.copy(); hasx[idx2] = True
    hyb = np.where(has4[None, :, None], new, old); hybx = np.where(hasx[None, :, None], ext, old)
    mu_old = np.array([float(r['mu_old']) for r in rows]); mu_new = np.array([float(r['mu_new']) for r in rows])
    lI_old, lI_hyb, lI_ext = old[..., 0], hyb[..., 0], hybx[..., 0]
    e_old, e_hyb, e_ext = ess(old[..., 0], old[..., 1]), ess(hyb[..., 0], hyb[..., 1]), ess(hybx[..., 0], hybx[..., 1])
    fA_old = np.exp(old[..., 2] - old[..., 0]); fA_hyb = np.exp(hyb[..., 2] - hyb[..., 0]); fA_ext = np.exp(hybx[..., 2] - hybx[..., 0])
    d_old = lI_old - lI_old[0]; d_hyb = lI_hyb - lI_hyb[0]; d_ext = lI_ext - lI_ext[0]
    change = d_hyb - d_old; change_x = d_ext - d_old
    infl = np.max(np.abs(change), 0); infl_x = np.max(np.abs(change_x), 0)
    N = nev
    per_pop = []
    for j, r in enumerate(rows):
        per_pop.append(dict(label=str(r['label']), params=json.loads(str(r['params'])), mu_old=mu_old[j], mu_new=mu_new[j],
                            log_R_total=float(r['log_R']), clamp_m1_above_120=float(r['clamp_m1_above']),
                            dlnL_events_old=float(d_old[j].sum()), dlnL_events_hybrid=float(d_hyb[j].sum()),
                            dlnL_events_extended=float(d_ext[j].sum()),
                            dmu_old=float(mu_old[j] - mu_old[0]), dmu_new=float(mu_new[j] - mu_new[0]),
                            dlnL_total_old=float(d_old[j].sum() - (mu_old[j] - mu_old[0])),
                            dlnL_total_hybrid=float(d_hyb[j].sum() - (mu_new[j] - mu_new[0])),
                            dlnL_total_extended=float(d_ext[j].sum() - (mu_new[j] - mu_new[0])),
                            ess_min_old=float(e_old[j].min()), ess_min_hybrid=float(e_hyb[j].min()),
                            ess_min_native4d_only=float(e_hyb[j][has4].min()),
                            n_ess_below_10_old=int((e_old[j] < 10).sum()), n_ess_below_10_hybrid=int((e_hyb[j] < 10).sum()),
                            seconds=float(r['seconds'])))
    xwmap = {r['name']: r for r in xw}
    ev = []
    for i, n in enumerate(names):
        r = xwmap[n]
        ev.append(dict(name=n, canonical=r['canonical'], likelihood='native4d' if has4[i] else ('nal3d_fallback(alt:%s)' % r.get('alt_analysis_id') if hasx[i] else 'nal3d_fallback'),
                       validation=r['validation'], lnI_old_mean=lI_old[0, i], lnI_new_mean=lI_hyb[0, i] if has4[i] else np.nan,
                       lnI_ext_mean=lI_ext[0, i] if hasx[i] else np.nan,
                       ess_old_mean=e_old[0, i], ess_new_mean=e_hyb[0, i] if has4[i] else np.nan,
                       ess_ext_mean=e_ext[0, i] if hasx[i] else np.nan,
                       ess_old_min_draws=e_old[:, i].min(), ess_new_min_draws=e_hyb[:, i].min() if has4[i] else np.nan,
                       fracA_old_mean=fA_old[0, i], fracA_new_mean=fA_hyb[0, i] if has4[i] else np.nan,
                       fracA_ext_mean=fA_ext[0, i] if hasx[i] else np.nan,
                       sd_draws_dlnI_old=d_old[1:, i].std() if D > 1 else np.nan,
                       sd_draws_dlnI_new=d_hyb[1:, i].std() if (D > 1 and has4[i]) else np.nan,
                       max_abs_change_vs_mean=infl[i] if has4[i] else 0.0,
                       max_abs_change_vs_mean_ext=infl_x[i] if hasx[i] else 0.0,
                       priority=n in PRIORITY))
    order = np.argsort(-np.array([e['max_abs_change_vs_mean_ext'] for e in ev]))
    for k, i in enumerate(order):
        ev[i]['rank_change'] = k + 1
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'events.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(ev[0])); w.writeheader()
        for e in ev:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, (float, np.floating)) else v) for k, v in e.items()})
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    summ = dict(
        scope='Frozen-population rescoring, not a refit. Event constants are arbitrary per likelihood; '
              'cross-population comparisons difference against the shared mean population.',
        n_events=nev, n_native4d=int(has4.sum()), n_alt_filled=int((hasx & ~has4).sum()), n_nal3d_fallback=int((~has4).sum()),
        n_seed=a.n_seed, n_b=a.n_b, nA=int(nA), key=a.key, kappa=a.kappa, gl_nodes=a.nodes, n_populations=D,
        posterior=a.post, posterior_sha256=sha(a.post),
        selection=dict(old='mass_pdet_grid_kappa2.7 xi(m1,m2), floor 1e-4, T_OBS=1',
                       new='combined_pdet_grid_mmin2_refinedz trilinear (log m1, log m2, z), z-integrated with p(z|kappa), no floor; '
                           'R_new*T_obs read as R_old',
                       mu_old_mean=mu_old[0], mu_new_mean=mu_new[0], n_obs=N,
                       profiled_selection_dlnL_new_minus_old=float(-N * np.log(mu_new[0] / mu_old[0])),
                       note='At fixed rate, selection term changes by -(mu_new-mu_old). With the overall rate profiled out '
                            'the change is -N ln(mu_new/mu_old).'),
        ess_mean_population=dict(old_min=float(e_old[0].min()), old_min_event=names[int(e_old[0].argmin())],
                                 new_hybrid_min=float(e_hyb[0].min()), new_hybrid_min_event=names[int(e_hyb[0].argmin())],
                                 native4d_only_min=float(e_hyb[0][has4].min()),
                                 native4d_only_min_event=names[int(np.where(has4)[0][e_hyb[0][has4].argmin()])],
                                 old_median=float(np.median(e_old[0])), new_hybrid_median=float(np.median(e_hyb[0])),
                                 n_below_10_old=int((e_old[0] < 10).sum()), n_below_10_new=int((e_hyb[0] < 10).sum())),
        populations=per_pop,
        priority_events=[ev[names.index(n)] for n in PRIORITY],
        biggest_movers=[ev[i] for i in order[:15]])
    if D > 1:
        ix = slice(1, D)
        for scope, delta in [('hybrid', d_hyb - d_old), ('extended', d_ext - d_old)]:
            lwr = delta[ix].sum(1) - (mu_new[ix] - mu_new[0]) + (mu_old[ix] - mu_old[0])
            wr = np.exp(lwr - lwr.max()); wr /= wr.sum()
            summ.setdefault('exploratory_reweighting', {})[scope] = dict(
                ess=float(1 / (wr @ wr)), n_draws=D - 1, max_weight=float(wr.max()),
                log_weights=lwr.tolist(),
                note='Importance weights of the old-posterior draws under new/old likelihood ratio (events + selection). Only 16 draws: indicative only.')
    (out / 'summary.json').write_text(json.dumps(summ, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print(json.dumps(dict(selection=summ['selection'], ess=summ['ess_mean_population']), indent=1, default=float))


if __name__ == '__main__':
    main()
