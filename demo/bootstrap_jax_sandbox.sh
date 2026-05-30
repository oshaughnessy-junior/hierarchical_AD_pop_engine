#!/usr/bin/env bash
# Recreate a CPU JAX + pytest dev/debug environment in an ephemeral
# sandbox (e.g. the Cowork VM) so the JAX path can be run locally — no
# loop out to a workstation just to execute pytest / SVI smoke tests.
#
# Verified 2026-05-29 in the Cowork sandbox (aarch64, CPU-only, Py3.10):
#   jax 0.6.2 installed from PyPI; tests/test_mc_particle_mc_jax_soft.py
#   → 8 passed in ~5s.  The install is EPHEMERAL (gone next session);
#   re-run this (~1–2 min) at the start of a JAX dev session.
#
# Usage (from the repo root, hierarchical_AD_pop_engine/):
#   source demo/bootstrap_jax_sandbox.sh         # installs + exports PYTHONPATH
#   python3 -m pytest tests/test_mc_particle_mc_jax_soft.py -q
#
# Caveats:
#   * CPU-only here (no GPU).  For GPU use jax[cuda] on a real host.
#   * Shell tools often cap each command at ~45s — the full M4 SVI
#     (2000 steps) can exceed it; run reduced steps or background it.
#   * `--break-system-packages` is required on the Debian-style sandbox.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

echo "[bootstrap] installing CPU JAX + inference stack (PyPI) ..."
pip install --break-system-packages -q \
    jax jaxlib equinox numpyro pytest jaxtyping scipy matplotlib

# src/ layout: the package lives under src/, plus repo root for red_team/ + demo/.
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
echo "[bootstrap] PYTHONPATH=${PYTHONPATH}"

python3 - <<'PY'
import jax, jax.numpy as jnp
print(f"[bootstrap] jax {jax.__version__}  grad-smoke d(x^3)/dx|_2 ="
      f" {float(jax.grad(lambda x: x**3)(2.0))}  (expect 12.0)")
PY

cat <<'EOF'
[bootstrap] ready.  Common commands (run from the repo root):

  python3 -m pytest tests/test_mc_particle_mc_jax_soft.py -q          # B.2 tier
  python3 -m pytest tests/test_mc_particle_mc_jax.py \
                    tests/test_evaluation_mc_jax.py \
                    tests/test_response_mc_jax.py -q                  # B.1 + evaluator
  python3 -m pytest -m slow -k integration_stage1_recovery_mc_jax -q  # M4 (slow; may need backgrounding)

Note: pytest installs to ~/.local/bin (often off PATH) — prefer `python3 -m pytest`.
EOF
