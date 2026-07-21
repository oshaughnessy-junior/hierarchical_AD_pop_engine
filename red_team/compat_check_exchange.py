"""Backward-compat gate for the opt-in EXCHANGE / SEGREGATION additions.

Runs particle_mc_jax_soft at a fixed cfg + key with all new flags at their
defaults and prints SHA256 hashes of every output array.  Run once on the
pre-edit engine (--save ref.npz), then after the edit (--check ref.npz):
identical hashes = the committed-fit code path is bit-for-bit unchanged.

  python red_team/compat_check_exchange.py --save  /tmp/ref.npz
  python red_team/compat_check_exchange.py --check /tmp/ref.npz
"""
import argparse, hashlib, sys
import numpy as np, jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from red_team.backup_particle_mc import MCConfig, particle_mc_jax_soft


def run():
    cfg = MCConfig(
        mu_natal=8.0, sigma_natal=2.0, m_min=5.0, m_max=70.0,
        natal_spin_loc=0.0, natal_spin_scale=0.10,
        kernel_a=0.0, kernel_b=2.0, v_esc=300.0,
        n_seed=2000, depletion_fraction=0.5,
    )
    out = particle_mc_jax_soft(cfg, jax.random.PRNGKey(20260714))
    keys = ["m_active", "chi_active", "w_active", "gen_active",
            "occupancy", "mergers", "v_kicks", "w_mergers", "step_active"]
    return {k: np.asarray(out[k]) for k in keys}


def hashes(arrs):
    return {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
            for k, v in arrs.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", default=None)
    p.add_argument("--check", default=None)
    args = p.parse_args()
    arrs = run()
    h = hashes(arrs)
    for k, v in h.items():
        print(f"  {k:12s} {v[:16]}")
    if args.save:
        np.savez(args.save, **arrs)
        print(f"saved reference -> {args.save}")
    if args.check:
        ref = np.load(args.check)
        bad = [k for k in arrs
               if hashes({k: ref[k]})[k] != h[k]]
        if bad:
            print(f"MISMATCH in: {bad}")
            return 1
        print("PASS: all outputs bit-identical to reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
