"""Summarise the declared validation gates of GW241011_233834's 4D fit (and the precessing alternatives) from the fits repo.
Usage (in the fits repo): val_summary.py OUT_CSV"""
import csv, glob, json, sys
rows = []
base = "collections/quasicircular/20261110-public-longem-4d-v1/validation/GW241011_233834__public_longem_4d"
for p in sorted(glob.glob(base + "__s*.json")):
    d = json.load(open(p)); th = d["thresholds"]; m = d["metrics"]; fl = d.get("PE_train_vs_heldout_floors", {})
    for gate in ("max_marginal_JS_bits", "max_axis_KS"):
        stat = "JS_bits" if "JS" in gate else "KS"
        worst = max(((tgt, dim, v[stat]) for tgt, t in m.items() for dim, v in t.get("marginals", {}).items()), key=lambda x: x[2])
        fv = fl.get(worst[0], {}).get("marginals", {}).get(worst[1], {}).get(stat)
        rows.append(dict(split=p.split("__")[-1][:-5], gate=gate, value=d[gate], threshold=th[gate], passes=d[gate] <= th[gate], worst_target=worst[0], worst_dim=worst[1], PE_split_floor=fv))
    for gate in ("max_projection_JS_bits", "max_conditional_q_given_z_JS_bits", "min_reference_information_ess", "min_proposal_ess"):
        lo = gate.startswith("min"); v = d[gate]; t = th[gate]
        rows.append(dict(split=p.split("__")[-1][:-5], gate=gate, value=v, threshold=t, passes=(v >= t) if lo else (v <= t)))
    for gate, tk in (("population_relative_fit_minus_reference_nats", "max_population_response_error_nats"), ("population_half_response_difference_nats", "max_half_population_response_difference_nats")):
        v = d[gate]; vv = max(abs(x) for x in v) if isinstance(v, list) else (max(abs(x) for x in v.values()) if isinstance(v, dict) else abs(v))
        rows.append(dict(split=p.split("__")[-1][:-5], gate=gate + " (max abs)", value=vv, threshold=th[tk], passes=vv <= th[tk]))
for coll in ("precessing/20261102-o4b-public-recovery-v1", "precessing/20261102-o4b-cosmo-prior-corrected-v2"):
    d = json.load(open(f"collections/{coll}/validation/GW241011_233834.json"))
    rows.append(dict(split=coll, gate="status / passes_declared_checks", value=str(d.get("status")) + " / " + str(d.get("passes_declared_checks", d.get("upstream_passes_declared_checks"))), threshold="", passes=d.get("passes_declared_checks")))
w = csv.DictWriter(open(sys.argv[1], "w", newline=""), ["split", "gate", "value", "threshold", "passes", "worst_target", "worst_dim", "PE_split_floor"], restval="")
w.writeheader(); [w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()}) for r in rows]
for r in rows: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
