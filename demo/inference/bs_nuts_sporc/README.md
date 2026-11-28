# SPORC scripts for the path-weighted BS_nuts relaunch (2026-11-28)

Copies of the scripts in SPORC `~/hier-reassess-20261121/runs/BS_nuts`. `chain.sbatch`,
`bias.sbatch` and `probe/parity_path.sbatch` are installed there. `submit_run.sh` is not.
The submission was refused by the agent's permission check and no job was submitted.
See `runs/BS_nuts/README.md` on cardassia for the run list, proposals and caps.

Submission, from `runs/BS_nuts` on SPORC:

    ./submit_run.sh NULL_floor40 exact 45
    ./submit_run.sh BB_floor40 exact 45
    ./submit_run.sh BMIX_floor40 exact 45
    ./submit_run.sh REFRC_floor40 defensive none
    ./submit_run.sh NULLG_floor40 defensive none
    sbatch -M sporc -J bsnuts-sync-daily --begin=now+24hours --export=ALL,MODE=daily -o sync_%j.out sync.sbatch
