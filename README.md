# BS_nuts: exact-cascade NUTS for the B-series members (2026-11-27)

These runs replace the posteriors from `two_channel_pl_demo.py --mode nuts`, which used the
committed broad proposal. Each run pins the cascade proposal at its member's MAP
`final_proposal`, so the proposal equals the target there.

## Code

- Engine branch `reassess-nuts-20261127`, commit `ce359e3`, off `0fde60f`. It is pushed to the
  engine origin.
- Driver: `demo/inference/hier_nuts.py`. It reuses `hier_map_laplace.build()`.
- MAP inputs: `demo/inference/bs_nuts_maps/<TAG>/{result,reeval_n8000}.json`, copied from the
  scan directories listed in `bs_nuts_maps/SOURCE.txt`.
- cardassia worktree: `~/20261120-hier-reassess/engine-nuts`, environment `runs/BS_nuts/env.sh`.
- SPORC worktree: `~/hier-reassess-20261121/engine-nuts`, fetched from git. Environment
  `runs/BS_nuts/env.sh` there. The SPORC catalog and pdet grid match cardassia's sha256.

## Likelihood, priors and rates

- lnL is `build()["parts"]`: the full Poisson lnL with 3D NAL events, the kappa 2.7 pdet grid and
  the etafix catalog. The fit's L-BFGS-B polish, final lnL and Laplace step use the same function.
- The rates `log_R_*` are sampled. They are not profiled. Every parameter has a uniform prior on
  the bounds stored in the MAP JSON.
- The cascade sampler key is 11 for every chain, with n_seed 8000 and n_b 6000. NUTS seeds are
  0 to 3.
- Each chain starts at the MAP theta, moved at most 1e-4 of the prior width inside an edge.
  Most MAPs sit on the m_max_A floor.

## Runs

| TAG | model | f_dep | m_max_A prior | MAP source |
|---|---|---|---|---|
| NULL_floor40 | NULL | 0.4 | U[40,70] | BS_scan/NULL_f0.4 |
| BB_floor40 | BB | 0.4 | U[40,70] | BS_scan/BB_f0.4 |
| BC_floor40 | BC | 0.4 | U[40,70] | BS_scan/BC_f0.4 |
| BD_floor40 | BD | 0.4 | U[40,70] | BS_scan/BD_f0.4 |
| BMIX_floor40 | BMIX | 0.5 | U[40,70] | BS_scan/BMIX_f0.5 |
| NULLG_floor40 | NULLG (null + g(eta), exch 0) | 0.3 | U[40,70] | BS_scan/NULLG_f0.3 |
| REFRC_floor40 | REFRC (combined) | 0.4 | U[40,70] | BS_scan/REFRC_f0.4 |
| REFRC_floor20 | REFRC (combined) | 0.5 | U[20,70] | BS_scan_mmax20/REFRC_f0.5_s25 |

Every run has 4 chains of 500 warmup and 1000 samples. The settings are dense mass,
max_tree_depth 6, target_accept 0.8 and chunk 25. All 32 chains run on SPORC tier3
(`-A grav-wave-rift`, 4 CPUs, 8 GB, 4-day limit). Job IDs are in `jobs.txt` on SPORC
(22181487 to 22181519). The bias checks are in `bias_jobs.txt` (22181524 to 22181531), each
with `afterany` on its four chains. At submission (2026-11-27 21:00) all 32 were pending on
Priority, with an estimated start of 2026-11-29 00:01.

Chain command (from `chain.sbatch` on SPORC):

```
$PY $E/demo/inference/hier_nuts.py --mode nuts --map-json $MAPS/$TAG/result.json \
  --outdir runs/BS_nuts/$TAG/chain$CHAIN --chain-seed $CHAIN $INP \
  --key 11 --n-seed 8000 --n-b 6000 --num-warmup 500 --num-samples 1000 --chunk 25 \
  --max-tree-depth 6 --target-accept 0.8 --dense-mass 1 --progress 1
```

Health check, on either host:

```
source runs/BS_nuts/env.sh
$PY $E/demo/inference/hier_nuts.py --mode status --ckpt-dirs "runs/BS_nuts/*/chain*"
```

A chain that hits the time limit after warmup resumes from `nuts_ckpt.pkl` when resubmitted
with the same `sbatch` line.

## Why SPORC CPUs

Measured value_and_grad at n_seed 8000 and n_b 6000 for NULL at the MAP:

| host | resources | s per gradient |
|---|---|---|
| cardassia | 3 threads, load 12 to 21 | 14.8 |
| SPORC skl-a-54 (tier3) | 4 CPUs, Gold 6150 | 6.9 |
| SPORC clx-a-02 (onboard) | A100 shard | 11.2 |

The cascade scan runs sequentially and gets no GPU speedup. On cardassia one chain would take
about 5 days. MaxRSS is 2.1 GB.

Leapfrog steps per iteration come from proxy chains at n_seed 1000 on cardassia (200 warmup,
100 samples, `smoke/*_n1000_w200s100`). Both had 0 divergences and 0 non-finite values:

| proxy | warmup steps/iter | sample steps/iter | accept | step size |
|---|---|---|---|---|
| NULL_floor40 | about 26 | 11.2 | 0.84 | 0.41 |
| REFRC_floor20 | about 26 | 15.9 | 0.87 | 0.33 |

Projected cost on a SPORC tier3 slot at 6.9 s per gradient:

| item | estimate |
|---|---|
| seconds per sample | 75 to 110 |
| warmup, 500 iterations | about 25 h |
| 1000 samples | 21 to 31 h |
| one chain | 2.0 to 2.4 days |

Early warmup iterations hit the 63-step cap. Late in warmup, steps fall to 7 to 31.

Two n_seed 8000 probes check this projection:

- cardassia `smoke/NULL_n8000_w20s20` (20 warmup, 20 samples, 3 threads).
- SPORC debug job 22181532, `probe/debug_REFRC_floor20` (150 warmup, 50 samples).

At n_seed 1000 the REFRC floor-20 proxy moved to mu_B 14.5 and sig_B 0.28, far from the MAP
(mu_B 9.3). The low-n_seed estimator has spurious high-lnL regions. That proxy measures cost
only, not the posterior.

## Init-lnL parity

At the MAP theta, key 11, n_seed 8000 and n_b 6000, the pinned-proposal lnL equals the key-11
entry of the MAP's `reeval_n8000.json`:

| member | lnL here | reeval_n8000 key 11 | difference |
|---|---|---|---|
| NULL (cardassia) | 2048.785671 | 2048.785671 | 0 |
| NULL (SPORC CPU) | 2048.785671 | 2048.785671 | 0 |
| NULL (SPORC A100) | 2048.785671 | 2048.785671 | 4.5e-13 |
| REFRC floor 20 (cardassia) | 2071.209863 | 2071.209863 | 0 |

Each chain also writes `lnL_map`, `lnL_init` and `ref_lnL` to its `meta.json`.

## Known limits

- `run_nuts_with_checkpoints` checkpoints after warmup and after each chunk. A failure during
  warmup restarts that chain's warmup.
- The driver wraps that helper's MCMC class so each chunk also stores `diverging`,
  `num_steps`, `accept_prob` and `potential_energy`, under keys prefixed with `_`.

## Bias check

`--mode bias` draws 20 posterior samples from all chains. At each draw it evaluates lnL with
the pinned proposal and with the proposal recentred at the draw, which is exact there. It
writes `<TAG>/bias_check.json`. A constant offset does not change the posterior. The spread of
`diff` across draws and its trend with spin_chi and m_max_A do. Recentring changes the cascade
realization at a fixed key, so per-key scatter (1.6 to 6 nats sd at n_seed 8000) bounds what
the check can resolve.

A pilot used the 10-sample smoke chain at n_seed 1000. Its 3 draws had spin_chi near 0.05,
against 0.096 in the proposal. They gave pinned minus exact of -6.3, -7.7 and -7.0 nats.

## Returning results to cardassia

SPORC cannot reach cardassia, and cardassia has no key for SPORC. The laptop is not used as a
relay. `sync_results.sh` on SPORC pushes the meta, checkpoint, sample and bias files to the
engine-origin branch `bs-nuts-results-20261127`. cardassia then fetches it. The script has not
been run.
