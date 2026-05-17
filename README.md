# hierarchical_AD_pop_engine

Autodifferentiable engine for hierarchical compact-binary populations.
Year-1 prototype of the AFI (autodifferentiable functional inference)
programme described in the project proposal; builds on the coagulation
proof-of-concept of Doctor et al. (2020, ApJ 893:35) and plugs into
[`gwkokab`](https://github.com/gwkokab/gwkokab).

See [`DESIGN.md`](DESIGN.md) for the architectural overview, the math, and
the v1 → v2 → v3 roadmap.

## Architecture in one line

`ResponseFunction(θ_b | θ_env) × FormationRate(θ_env, z) → SyntheticUniverse → numpyro.Distribution`,
all autodifferentiable end-to-end in JAX.

## Install

```bash
pip install -e .[cpu,test]   # CPU JAX
pip install -e .[cuda12,test]   # CUDA-12 JAX
```

The dependency on `gwkokab` is required (the natal IMF, spin, and redshift
distributions are reused directly).

If you already have an environment with an older `numpyro` (< 0.20), upgrade
it explicitly — `gwkokab.inference` imports `numpyro._typing.DistributionT`
which was added in `numpyro` 0.20:

```bash
pip install --upgrade 'numpyro>=0.20'
```

## Usage

The simplest entry point is `coagulation_model`, a drop-in replacement for
gwkokab's `PowerlawPeak` factory:

```python
import jax.numpy as jnp
from hierarchical_ad_pop_engine.distributions import coagulation_model

mix = coagulation_model(
    alpha=jnp.asarray(3.0),
    delta_m=jnp.asarray(5.0),
    lambda_peak=jnp.asarray(0.03),
    mu_peak=jnp.asarray(33.0),
    sigma_peak=jnp.asarray(3.0),
    m_min=jnp.asarray(5.0),
    m_max=jnp.asarray(80.0),
    natal_spin_loc=jnp.asarray(0.1),
    natal_spin_scale=jnp.asarray(0.15),
    kernel_a=jnp.asarray(0.0),
    kernel_b=jnp.asarray(2.0),
    log_R_env=jnp.asarray(0.0),
    log_R_formation=jnp.asarray(0.0),
    log_mu_v_esc=jnp.asarray(jnp.log(300.0)),
    log_sigma_v_esc=jnp.asarray(0.5),
    z_max=jnp.asarray(3.0),
    kappa=jnp.asarray(2.7),
    gamma=jnp.asarray(2.9),
    z_peak=jnp.asarray(1.9),
    log_rate=jnp.asarray(2.0),
)

# mix is a gwkokab.models.utils.ScaledMixture; pass it to
# gwkokab.inference.numpyro_poisson_likelihood directly.
value = jnp.asarray([40.0, 20.0, 0.2, 0.1, 0.5])  # [m1, m2, chi1, chi2, z]
log_p = mix.log_prob(value)
```

For a single-environment hypothesis (no env distribution), use
`coagulation_model_delta_formation` instead.

## Tests

```bash
pytest -q
```

The verification suite includes:

- per-module sanity checks (natal, remnants, kick, retention, kernel),
- coagulation-solver limit tests (zero-retention kills higher generations;
  high retention grows them),
- the **headline architectural invariant** that a `DeltaFormation` recovers
  the bare response function exactly,
- finite-difference vs `jax.grad` checks for each parameter through the
  full pipeline.

## Module layout

```
src/hierarchical_ad_pop_engine/
├── grids.py                     # (m, χ, z, v_esc) grid utilities
├── interp.py                    # smooth log-density interpolation
├── natal/                       # thin wrappers around gwkokab
├── kernels.py                   # pairing kernel K(m1, m2; θ_K)
├── remnants/                    # NR-calibrated mass/spin/kick fits
├── retention.py                 # P(v_kick < v_esc | binary)
├── response/
│   ├── protocol.py              # ResponseFunction protocol
│   └── coagulation.py           # generational solver (v1)
├── formation/
│   ├── protocol.py              # FormationRate protocol
│   ├── delta.py                 # DeltaFormation (single-env hypothesis)
│   └── parametric.py            # LogNormalFormation × Madau-Dickinson
├── universe.py                  # SyntheticUniverse: R ⊗ F convolution
└── distributions.py             # numpyro.Distribution wrapper for gwkokab
```
