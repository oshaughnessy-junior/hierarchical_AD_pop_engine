"""Parity: JAX native4d log L vs the NumPy reference on every fit.

For each fit in the 4D collection (and the consumer-selection alternates used to fill
fallback events), evaluate both implementations at
  * 1024 points from the reference integration_bank (posterior bulk), and
  * 4096 points uniform in a box around the analysis support (mix of in/out of support).
Pass: identical support masks and max |dlogL| < 1e-8 over in-support points.

Run (needs astropy for the reference): venv-A/bin/python engine/demo/inference/native4d_parity_test.py
"""
import csv, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path.home() / '20261120-hier-reassess'
FITS = ROOT / 'fits'
sys.path[:0] = [str(HERE), str(FITS / 'src')]
import jax, jax.numpy as jnp  # noqa: E402
import native4d_likelihood as N4  # noqa: E402
from collection_registry import Collection  # noqa: E402
import native_spin_base_redshift as ref  # noqa: E402

TOL = 1e-8


def fits_to_test():
    out = []
    for cid in ['quasicircular/20261110-public-longem-4d-v1']:
        col = Collection(FITS, cid)
        for aid in sorted(col.entries):
            out.append((cid, aid, col.fit(analysis_id=aid, allow_candidate=True, acknowledge_source_issues=True)))
    xw = ROOT / 'runs/A_crosswalk/crosswalk.csv'
    if xw.exists():
        cols = {}
        for r in csv.DictReader(open(xw)):
            if r.get('alt_analysis_id'):
                c = cols.setdefault(r['alt_collection'], Collection(FITS, r['alt_collection']))
                out.append((r['alt_collection'], r['alt_analysis_id'],
                            c.fit(analysis_id=r['alt_analysis_id'], allow_candidate=True, acknowledge_source_issues=True)))
    return out


def main():
    rng = np.random.default_rng(20261120)
    f = jax.jit(N4.log_likelihood)
    rows = []; worst = 0.0; t0 = time.time()
    for cid, aid, m in fits_to_test():
        _, _, xb, _ = ref.integration_bank(m, n=1024, seed=int(rng.integers(1 << 30)), grid_size=128)
        s = m['analysis_support']
        lo = np.array([s['mc_min'] * .9, 0.01, -1.0, max(s['z_min'] * .8, 1e-4)])
        hi = np.array([s['mc_max'] * 1.1, 0.25, 1.0, s['z_max'] * 1.2])
        xu = lo + (hi - lo) * rng.random((4096, 4))
        x = np.vstack([xb, xu])
        a = ref.log_likelihood_x(x, m)
        b = np.asarray(f(jnp.asarray(x), N4.as_jax(N4.pack_model(m))))
        fa, fb = np.isfinite(a), np.isfinite(b)
        d = np.abs(a[fa & fb] - b[fa & fb])
        r = dict(collection=cid, analysis_id=aid, n_points=len(x), n_in_support=int(fa.sum()),
                 mask_mismatch=int((fa != fb).sum()), max_abs_dlogL=float(d.max()) if d.size else 0.0,
                 max_abs_dlogL_bank=float(np.abs(a[:1024] - b[:1024]).max()),
                 logL_range=f"{a[fa].min():.1f}..{a[fa].max():.1f}" if fa.any() else '')
        rows.append(r); worst = max(worst, r['max_abs_dlogL'])
        if r['mask_mismatch'] or r['max_abs_dlogL'] > TOL:
            print('FAIL', r, flush=True)
    out = ROOT / 'runs/A_parity'; out.mkdir(parents=True, exist_ok=True)
    with open(out / 'parity.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
    summ = dict(n_fits=len(rows), n_points_total=int(sum(r['n_points'] for r in rows)),
                n_in_support_total=int(sum(r['n_in_support'] for r in rows)),
                mask_mismatches=int(sum(r['mask_mismatch'] for r in rows)),
                max_abs_dlogL=worst, median_of_per_fit_max=float(np.median([r['max_abs_dlogL'] for r in rows])),
                tolerance=TOL, passed=bool(worst < TOL and not any(r['mask_mismatch'] for r in rows)),
                seconds=time.time() - t0)
    (out / 'parity_summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 0 if summ['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
