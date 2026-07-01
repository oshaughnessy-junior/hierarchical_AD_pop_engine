# Real-data NAL catalog pipeline

Builds the per-event **Normal-Approximate Likelihood (NAL)** catalog used for the
real-data (O3+O4a+O4b) coagulation analysis: a multivariate normal fit to each
single-event *likelihood* in source-frame `(Mc, eta, chi_eff)` coordinates, with
the full 3x3 covariance. The likelihood is the posterior reweighted by the
inverse PE prior — crucially including the chi_eff prior induced by the isotropic
component-spin prior, whose omission artificially narrows the high-mass chi_eff.

Reference: Delfavero, O'Shaughnessy, Wysocki & Yelikar, *Normal Approximate
Likelihoods to Gravitational Wave Events*, arXiv:2107.13082
(fits + data: https://gitlab.com/xevra/nal-data).

## Sources (all READ-ONLY)
- **O4 (S-named events)**: reviewed production PE. Analysis chosen via each
  project's `.asimov/ledger.yml` — take the entry with `approximant:
  IMRPhenomXPHM` **and** a `review: status: APPROVED`; samples at
  `results/<S-EVENT>/<analysis>/posterior_samples.h5` (avoid `archive/`).
  - GWTC5-HLV project: `/home/pe.o4/GWTC5-HLV/project`
  - O4a (GW-named 2023 events): `/home/pe.o4/GWTC4` (GW->S map by GPS time)
- **O3 (GW-named events)**: published Delfavero NAL fits, label
  `aligned3d_source:<wf>:select` (prefer IMRPhenomXPHM), from
  https://gitlab.com/xevra/nal-data (GWTC-1/2/2p1/3.nal.hdf5).

## Order
1. `extract_nal_catalog.py`  — O4 production events (GWTC5-HLV) -> per-event NAL.
2. `extract_nal_o4a.py`      — O4a events from the GWTC4 project.
3. `replace_o3_delfavero.py` — O3 rows from the Delfavero published NAL fits.
4. `merge_nal_catalog.py` / `merge_nal_full.py` — assemble the 238-event
   drop-in `full_o3_o4a_o4b_catalog_NAL.npz` (fields: names, labels,
   centers(N,3), sigs(N,3), covs(N,3,3), ess, method, kl).
5. `test_marginal_deconv.py` — validation of the prior-deconvolution.

The resulting catalog is consumed by `GenieMcEtaChieffLikelihood(center, cov=...)`
(full-covariance multivariate normal) in the population fits.

NOTE: paths to `/home/pe.o4` and the local scratch catalog are hard-coded for the
LDG/CIT environment these were developed on; adjust for other sites.
