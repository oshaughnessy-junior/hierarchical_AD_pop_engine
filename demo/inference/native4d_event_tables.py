"""Build the per-event table the population code needs (runs once, needs astropy).

For every catalog event with a 4D fit (primary collection, plus consumer-selection
alternates for fallback events, kept separate as tier 'alt'):
  * packed model arrays (native4d_likelihood.pack_model), stacked;
  * the detector-frame chirp-mass quadrature range [a, b] in log Mc_det: weighted
    quantiles 1e-5 and 1-1e-5 of the reference integration bank (16384 draws),
    widened by PAD x width on each side and clipped to the analysis support.
Output: runs/A_tables/native4d_events.npz (+ .json summary).
"""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path.home() / '20261120-hier-reassess'
FITS = ROOT / 'fits'
sys.path[:0] = [str(HERE), str(FITS / 'src')]
import native4d_likelihood as N4  # noqa: E402
from collection_registry import Collection  # noqa: E402
import native_spin_base_redshift as ref  # noqa: E402

PRIMARY = 'quasicircular/20261110-public-longem-4d-v1'
PAD = 0.3


def wquant(x, w, qs):
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]
    return np.interp(qs, c, x[o])


def main():
    rows = list(csv.DictReader(open(ROOT / 'runs/A_crosswalk/crosswalk.csv')))
    cols = {}
    out = dict(catalog_index=[], name=[], tier=[], collection=[], analysis_id=[], lmcd_lo=[], lmcd_hi=[],
               lmcd_q=[], z_q=[], validation=[])
    packed = []
    for r in rows:
        for tier, cid, aid in [('primary', PRIMARY, r['fit_id']), ('alt', r.get('alt_collection'), r.get('alt_analysis_id'))]:
            if not aid or aid == 'NONE':
                continue
            col = cols.setdefault(cid, Collection(FITS, cid))
            m = col.fit(analysis_id=aid, allow_candidate=True, acknowledge_source_issues=True)
            _, lw, x, _ = ref.integration_bank(m, n=16384, seed=1064, grid_size=192)
            w = np.exp(lw - lw.max()); lmcd = np.log(x[:, 0] * (1 + x[:, 3]))
            q = wquant(lmcd, w, [1e-5, 0.5, 1 - 1e-5]); width = q[2] - q[0]
            s = m['analysis_support']
            lo = max(q[0] - PAD * width, np.log(s['mc_det_min']) if s.get('mc_det_min') else -np.inf)
            hi = min(q[2] + PAD * width, np.log(s['mc_det_max']) if s.get('mc_det_max') else np.inf)
            out['catalog_index'].append(int(r['index'])); out['name'].append(r['name']); out['tier'].append(tier)
            out['collection'].append(cid); out['analysis_id'].append(aid)
            out['lmcd_lo'].append(lo); out['lmcd_hi'].append(hi); out['lmcd_q'].append(q)
            out['z_q'].append(wquant(x[:, 3], w, [1e-3, .5, 1 - 1e-3]))
            out['validation'].append(r['validation'] if tier == 'primary' else 'alt:' + cid.split('/')[-1])
            packed.append(N4.pack_model(m))
            print(r['name'], tier, aid, 'Mc_det [%.3f, %.3f] median %.3f; z 0.1%%-99.9%% %.3f-%.3f'
                  % (np.exp(lo), np.exp(hi), np.exp(q[1]), out['z_q'][-1][0], out['z_q'][-1][2]), flush=True)
    st = N4.stack_models(packed)
    dest = ROOT / 'runs/A_tables'; dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / 'native4d_events.npz', **{k: np.asarray(v) for k, v in out.items()},
             **{'P_' + k: np.asarray(v) for k, v in st.items()})
    (dest / 'native4d_events.json').write_text(json.dumps(dict(
        n=len(packed), n_primary=out['tier'].count('primary'), n_alt=out['tier'].count('alt'), pad=PAD,
        bank=dict(n=16384, seed=1064, grid_size=192)), indent=1))


if __name__ == '__main__':
    main()
