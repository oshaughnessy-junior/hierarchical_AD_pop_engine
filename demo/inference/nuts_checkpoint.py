"""Checkpoint-safe NUTS for long cluster runs that may be evicted/rebooted.

Strategy (numpyro 0.21 API):
  * run warmup ONCE, then checkpoint the adapted sampler state (mass matrix +
    step size + position + RNG) -- the expensive, fragile part;
  * sample in CHUNKS of K, atomically checkpointing (last_state + ALL samples
    accumulated so far) after each chunk;
  * on (re)start, resume from the newest checkpoint and continue.

Pair with condor `when_to_transfer_output = ON_EXIT_OR_EVICT` so the checkpoint
dir is transferred back on eviction and re-fed as input on restart. A resumed
run reproduces an uninterrupted run bit-for-bit (RNG threads through last_state).

  from nuts_checkpoint import run_nuts_with_checkpoints
  post = run_nuts_with_checkpoints(model, (T_obs,), num_warmup=300,
             num_samples=600, key=key, ckpt_dir="ckpt", chunk=50,
             nuts_kwargs=dict(init_strategy=..., max_tree_depth=7))
"""
import os, pickle
import numpy as np
import jax
from numpyro.infer import MCMC, NUTS


def _atomic_pickle(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)          # atomic: a kill mid-write can't corrupt ckpt


def run_nuts_with_checkpoints(model, run_args, *, num_warmup, num_samples, key,
                              ckpt_dir, chunk=50, nuts_kwargs=None, progress=True):
    nuts_kwargs = nuts_kwargs or {}
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, "nuts_ckpt.pkl")

    state, merged, n_done, phase = None, None, 0, "warmup"
    if os.path.exists(ckpt):
        d = pickle.load(open(ckpt, "rb"))
        phase, state, n_done, merged = d["phase"], d["state"], d["n_done"], d["samples"]
        print(f"[ckpt] RESUME phase={phase} n_done={n_done}/{num_samples}", flush=True)

    # ---- warmup once -> checkpoint adapted state ----
    if phase == "warmup":
        m = MCMC(NUTS(model, **nuts_kwargs), num_warmup=num_warmup,
                 num_samples=chunk, progress_bar=progress)
        m.warmup(key, *run_args)
        state = jax.device_get(m.last_state)
        _atomic_pickle({"phase": "sampling", "state": state, "n_done": 0,
                        "samples": None}, ckpt)
        print(f"[ckpt] warmup done + checkpointed "
              f"(step_size={float(m.last_state.adapt_state.step_size):.3g})", flush=True)

    # ---- sampling in chunks ----
    acc = {k: [v] for k, v in merged.items()} if merged else None
    cur = state
    while n_done < num_samples:
        this = min(chunk, num_samples - n_done)
        m = MCMC(NUTS(model, **nuts_kwargs), num_warmup=0, num_samples=this,
                 progress_bar=progress)
        m.post_warmup_state = cur
        m.run(cur.rng_key, *run_args)
        s = {k: np.asarray(v) for k, v in m.get_samples().items()}
        if acc is None:
            acc = {k: [v] for k, v in s.items()}
        else:
            for k in acc:
                acc[k].append(s[k])
        n_done += this
        cur = jax.device_get(m.last_state)
        merged = {k: np.concatenate(v) for k, v in acc.items()}
        _atomic_pickle({"phase": "sampling", "state": cur, "n_done": n_done,
                        "samples": merged}, ckpt)
        print(f"[ckpt] sampled {n_done}/{num_samples}", flush=True)

    return {k: np.concatenate(v) for k, v in acc.items()}
