"""Crosswalk: 236 coag catalog names -> 4D native fits (20261110-public-longem-4d-v1).

Name resolution, in order:
  1. fits/provenance/older_event_union.json alias_rules (O3a synthetic short S names,
     short O3 GW names -> canonical GW names);
  2. fits/provenance/all_paper_events.json name -> gw_name (O4 superevents);
  3. the collection manifest's event / aliases fields.
Events without a fit keep the old 3D NAL (hybrid) and are flagged. No event is dropped.

Run: venv-A/bin/python engine/demo/inference/native4d_crosswalk.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path.home() / '20261120-hier-reassess'
FITS = ROOT / 'fits'
CID = 'quasicircular/20261110-public-longem-4d-v1'
CDIR = FITS / 'collections' / CID
CAT = ROOT / 'inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz'
OUT = ROOT / 'runs/A_crosswalk'


def main():
    sys.path.insert(0, str(FITS / 'src'))
    from collection_registry import Collection
    col = Collection(FITS, CID)
    cat = np.load(CAT, allow_pickle=True)
    names = [str(n) for n in cat['names']]; labels = [str(x) for x in cat['labels']]
    methods = [str(x) for x in cat['method']]
    alias_rules = json.load(open(FITS / 'provenance/older_event_union.json'))['alias_rules']
    paper = {e['name']: e for e in json.load(open(FITS / 'provenance/all_paper_events.json'))['events']}
    summ = {m['event']: m for m in json.load(open(CDIR / 'summary.json'))['metrics']}
    sel = json.load(open(FITS / 'provenance/consumer_selections/20261110-popsynth-current-public-fits.json'))
    selmap = {}
    for r in sel['records']:
        for k in {r.get('event'), r.get('canonical')}:
            if k:
                selmap[k] = r
    lookup = {}
    for row in col.entries.values():
        for k in [row['event']] + list(row.get('aliases', [])):
            lookup.setdefault(k, set()).add(row['analysis_id'])
    rows = []; used = {}
    for i, n in enumerate(names):
        if n in alias_rules:
            gw, how = alias_rules[n], 'older_event_union.alias_rules'
        elif n in paper and paper[n]['gw_name'] != n:
            gw, how = paper[n]['gw_name'], 'all_paper_events.gw_name'
        else:
            gw, how = n, 'identity'
        cands = set()
        for k in {n, gw}:
            cands |= lookup.get(k, set())
        if len(cands) > 1:
            raise SystemExit(f'ambiguous fit for {n}: {cands}')
        r = dict(index=i, name=n, run=labels[i], old_method=methods[i], canonical=gw, name_source=how)
        if cands:
            aid = cands.pop(); row = col.entries[aid]
            if aid in used:
                raise SystemExit(f'fit {aid} matched twice: {used[aid]} and {n}')
            used[aid] = n
            fit = col.fit(analysis_id=aid, allow_candidate=True, acknowledge_source_issues=True)
            sup = fit['analysis_support']; dc = row['acceptance']['declared_checks']
            all3 = bool(dc['published_split_passes'] and all(x['passes'] for x in dc['replication']))
            m = summ.get(row['event'], {})
            failed = dc.get('failed_checks', [])
            r.update(fit_id=aid, fit_event=row['event'], likelihood='native4d',
                     validation='pass_all_three_declared' if all3 else 'fails_declared:' + (';'.join(failed) if failed else 'replication'),
                     published_split_pass=dc['published_split_passes'],
                     replication_pass=';'.join(f"{x['split_seed']}:{x['passes']}" for x in dc['replication']),
                     native_shape_three_split_pass=m.get('native_shape_three_split_pass'),
                     acceptance_status=row['acceptance']['status'],
                     z_min=sup.get('z_min'), z_max=sup.get('z_max'),
                     mc_det_min=sup.get('mc_det_min'), mc_det_max=sup.get('mc_det_max'),
                     distance_prior=fit['prior_contract']['distance_prior']['kind'],
                     n_components=len(fit['components']))
        else:
            s = selmap.get(gw) or selmap.get(n)
            r.update(fit_id='NONE', fit_event='', likelihood='nal3d_fallback', validation='no_4d_fit_in_collection',
                     published_split_pass='', replication_pass='', native_shape_three_split_pass='',
                     acceptance_status='', z_min='', z_max='', mc_det_min='', mc_det_max='', distance_prior='', n_components='')
            r['note'] = ('consumer selection uses ' + s['collection_id'] + ' (' + s['status'] + ')') if s else 'not in consumer selection'
            if s and s.get('status') == 'selected' and s['collection_id'] != CID:
                r['alt_collection'] = s['collection_id']; r['alt_analysis_id'] = s['analysis_id']
        rows.append(r)
    unused = sorted(set(col.entries) - set(used))
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / 'crosswalk.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow(r) for r in rows]
    n4 = sum(r['fit_id'] != 'NONE' for r in rows)
    npass = sum(r['validation'] == 'pass_all_three_declared' for r in rows)
    summary = dict(collection=CID, catalog=str(CAT), n_catalog=len(rows), n_native4d=n4, n_fallback_nal3d=len(rows) - n4,
                   n_native4d_pass_all_three=npass, n_native4d_fail_declared=n4 - npass,
                   fits_in_collection_not_in_catalog=unused,
                   n_fallback_with_consumer_selection_alt_fit=sum(bool(r.get('alt_analysis_id')) for r in rows),
                   fallback_events=[r['name'] + '|' + r['canonical'] + '|' + r.get('note', '') for r in rows if r['fit_id'] == 'NONE'],
                   by_run={lab: dict(native4d=sum(r['fit_id'] != 'NONE' for r in rows if r['run'] == lab),
                                     fallback=sum(r['fit_id'] == 'NONE' for r in rows if r['run'] == lab)) for lab in sorted(set(labels))})
    (OUT / 'crosswalk_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'fallback_events'}, indent=1))
    for x in summary['fallback_events']:
        print('  fallback', x)


if __name__ == '__main__':
    main()
