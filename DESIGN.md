# hierarchical_AD_pop_engine — design

Autodifferentiable engine for hierarchical compact-binary populations. Year-1 prototype of the AFI (autodifferentiable functional inference) program described in the project proposal, building on the coagulation proof-of-concept of Doctor et al. (2020, ApJ 893:35). Targets plug-in compatibility with [`gwkokab`](https://github.com/gwkokab/gwkokab).

## 1 Scope and intent

We want a differentiable map from *physics* (natal stellar population + pairing/interaction parameters + environment) to *observables* (the merger-rate density `dN/dV_c dt dθ_binary`). For v1 the physics is low-dimensional parametric — non-parametric / functional inputs come later.

End use: given LVK posterior samples for compact-binary events, fit physics parameters by maximising the gwkokab Poisson likelihood, with gradients propagated through the entire physics stack via JAX.

## 2 Architecture — three composable layers

Population modelling cleanly factors into three layers. Each must be autodifferentiable so AD propagates end-to-end through the composition.

### Layer A — `ResponseFunction`: *one* dense environment

A response function `R` takes binary parameters `θ_b = (m1, m2, χ1, χ2)` and environment parameters `θ_env = (v_esc, ...)` and returns the per-environment differential merger rate

```
R(θ_b | θ_env)  =  dN_merge / dt / dm1 dm2 dχ1 dχ2     per single environment
```

For v1, our only response function is the **generational coagulation solver**. It internally tracks `n_k(m,χ)` for generations `k = 1,…,K_max` via convolution of the natal distribution under a pairing kernel, then maps through NR-calibrated remnant fits and a retention probability (set by `v_esc`). The output is a *smooth* differentiable density.

Future response functions (e.g. isolated binary evolution, hyperbolic capture) plug in at this layer without disturbing anything downstream.

### Layer B — `FormationRate`: the *universe*

A formation rate `F` describes how often environments of a given kind exist across the universe:

```
F(θ_env, z)  =  dN_env / dV_c / dz / dθ_env       comoving number density of environments
```

Examples: AGN-disk number density vs. `M_SMBH` and `z`; nuclear star cluster mass function with formation-history weighting; a simple Madau-Dickinson-like rate modulated by an environment-parameter distribution.

For v1 we ship two: a **`DeltaFormation`** (a single environment at fixed `θ_env`, weighted by an arbitrary `R(z)`) for trivial verification and unit testing, and **`ParametricFormation`** (a parametric env-parameter distribution × Madau-Dickinson rate). Both are AD-friendly.

### Layer C — `SyntheticUniverse`: convolution

The observable rate density is the convolution

```
dN_obs / dV_c / dt / dθ_b / dz  =  ∫ R(θ_b | θ_env) · F(θ_env, z)  dθ_env
```

implemented as quadrature on an `env`-parameter grid. AD passes cleanly through both layers and the quadrature sum.

### Layer D — `numpyro.Distribution` wrapper

Wraps the synthetic-universe rate density as a single `Distribution` over the joint event vector `(m1, m2, χ1, χ2, z)` (extendable). `log_prob` returns `log(dN/dV_c dt dθ)` in the gwkokab "rate density" convention (not unit-normalized — gwkokab's redshift models follow the same convention). Wrapped in `ScaledMixture([log_rate], [self])` so the Poisson likelihood gets its rate.

---

This layering is the architectural backbone. Each layer is its own subpackage with a tiny protocol/ABC interface, so we can swap response functions, formation rates, or convolution backends without disturbing the rest.

## 3 Mathematics

### 3.1 Generational coagulation (response function)

Following Doctor et al. (2020):

**1st generation** (natal). The natal population factorises:
```
n_1(m, χ; z)  =  f_IMF(m; θ_IMF) · g_spin(χ; θ_spin) · ψ(z; θ_z)
```
with `f_IMF` a truncated power-law + Gaussian peak (Talbot-Thrane style), `g_spin` a truncated normal on `χ ∈ [0,1]` (often peaked near 0; smooth and well-behaved everywhere on the unit interval, unlike the Beta which can develop boundary singularities at small mean/variance), and `ψ(z)` a Madau-Dickinson-shaped rate profile.

**Binary pairing**. The rate density of *encounters* that produce a merging binary of given parameters within a single environment is

```
dn_pair,(j,k) / dm1 dm2 dχ1 dχ2  ∝  K(m1,m2,χ1,χ2; θ_K) · n_j(m1,χ1) · n_k(m2,χ2)
```

where `K` is the pairing/encounter kernel, parametric in v1 (e.g. `K ∝ (m1+m2)^a q^b` with optional spin-dependence). The kernel encodes mass-ratio bias and interaction cross sections.

**Remnant mapping**. Each merging binary produces a remnant whose mass, spin, and recoil kick are determined by NR-calibrated fits (Lousto, Healy, Zlochower; Hofmann-Barausse-Rezzolla for spin; Healy-Lousto for kick). All implemented as smooth, differentiable closed-form functions.

**Retention**. v1 uses *pure aligned-spin* kick physics — `v_kick(m1,m2,χ1,χ2)` is the deterministic Lousto-Zlochower aligned-spin combination of mass-asymmetry and aligned-spin pieces (no in-plane "superkick", no tilt marginalisation). Retention is therefore a Heaviside step:
```
P_retain(m1,m2,χ1,χ2; v_esc)  =  𝟙[v_kick(m1,m2,χ1,χ2) < v_esc]
```
This is the **only** place `v_esc` enters at this layer. **AD consequence**: gradients w.r.t. `v_esc` are zero almost everywhere (Heaviside has no gradient). Gradients w.r.t. every other parameter (natal IMF, spin, kernel, response normalisation) flow normally because those change the integrand of the retained-volume integral smoothly. Adding tilt-marginalised retention (a smooth analytic / pre-tabulated kick CDF over tilt distributions) is the v2 upgrade and restores AD in `v_esc`.

**Higher generations**. Retained remnants populate the next generation:
```
n_{k+1}(m,χ)  =  ∫∫ dm1 dm2 dχ1 dχ2  P_pairing(j,k') · K(...) ·
                  n_j(m1,χ1) n_k'(m2,χ2) · P_retain(...) ·
                  δ(m - M_rem(...)) δ(χ - χ_rem(...))
```
summed over progenitor-generation pairs `(j, k')` with `j+k' = k+1` (treating `k+1`-th generation in the "deepest progenitor" convention). The deltas are resolved by change-of-variables on the `(m,χ)` grid, retaining differentiability.

**Per-environment observed merger rate density**:
```
dN_R / dt / dm1 dm2 dχ1 dχ2  =  Σ_{(j,k')}  R_{j,k'} · n_j(m1,χ1) n_k'(m2,χ2) K(m1,m2,χ1,χ2)
```
with `R_{j,k'}` the per-environment-per-time encounter rate normalisation.

For v1 generations are truncated at `K_max = 4` or `5` (configurable). The hierarchy converges quickly because retention probabilities are typically `< 0.1` per merger.

### 3.2 Smoothness / differentiability

The internal representation of every `n_k` is values on a **fixed `(m, χ)` grid**, with `log` taken so positivity is preserved under interpolation. Convolutions are quadrature on the grid (vectorised, single `jnp.einsum`-style sum). Evaluation at arbitrary binary points uses `jax.scipy.ndimage.map_coordinates` (linear or cubic-spline) on the log-densities. All operations are JIT-able and AD-friendly.

The redshift axis is handled at the **formation-rate** layer, not the response function — the environment is approximated as quasi-instantaneous on cosmological timescales for v1 (justified: NSC / AGN-disk coagulation timescales `≪` Hubble time for the relevant primary masses). v2 will lift this assumption.

### 3.3 Synthetic-universe convolution

Concretely with environment parameter `θ_env = v_esc` and an environment number density `dN_env / dV_c dz dv_esc`:

```
dN_obs / dV_c dt dθ_b dz  =  ∫ R(θ_b | v_esc) · dN_env/dV_c dz dv_esc · dv_esc
```

For v1 we expose either:
- `DeltaFormation(v_esc_0, R_z=ψ_MD)`: single environment type, redshift profile from Madau-Dickinson — observed rate is `R(θ_b | v_esc_0) · ψ_MD(z)`;
- `LogNormalFormation(μ, σ, R_z)`: log-normal in `v_esc` × redshift profile;

Both are AD-differentiable in all parameters.

## 4 Module layout

```
hierarchical_AD_pop_engine/
├── pyproject.toml
├── DESIGN.md                        # this file
├── README.md
├── src/hierarchical_ad_pop_engine/
│   ├── __init__.py
│   ├── grids.py                     # (m, χ, z, v_esc) grid utilities; log-density I/O
│   ├── interp.py                    # smooth log-density interpolation on grids
│   ├── natal/
│   │   ├── mass.py                  # truncated power-law + Gaussian peak
│   │   ├── spin.py                  # truncated Gaussian / beta on χ
│   │   └── redshift.py              # Madau-Dickinson and power-law rates
│   ├── kernels.py                   # pairing kernel K(m1,m2,χ1,χ2; θ_K)
│   ├── remnants/
│   │   ├── mass.py                  # E_rad fit → M_rem
│   │   ├── spin.py                  # Hofmann-Barausse-Rezzolla
│   │   └── kick.py                  # Healy-Lousto kick magnitude
│   ├── retention.py                 # P(v_kick < v_esc | binary params)
│   ├── response/
│   │   ├── protocol.py              # ResponseFunction ABC
│   │   └── coagulation.py           # generational solver
│   ├── formation/
│   │   ├── protocol.py              # FormationRate ABC
│   │   ├── delta.py                 # DeltaFormation
│   │   └── parametric.py            # LogNormalFormation, …
│   ├── universe.py                  # SyntheticUniverse: R ⊗ F convolution
│   └── distributions.py             # numpyro.Distribution wrapper for gwkokab
└── tests/
    ├── test_grids_interp.py
    ├── test_natal.py
    ├── test_remnants.py
    ├── test_retention.py
    ├── test_kernels.py
    ├── test_coagulation.py          # zero-coag limit, v_esc=∞ / 0 limits
    ├── test_formation.py
    ├── test_universe.py             # delta-formation recovers response
    ├── test_gradients.py            # finite-difference vs jax.grad
    └── test_gwkokab_integration.py
```

Every module exposes pure functions taking explicit parameters (no globals), so they're trivially JIT-able and gradable. Where state is helpful — pre-computed grids, equinox `Module` instances — we use `equinox.Module` to match gwkokab's pytree conventions.

## 5 Conventions

- **Units**: solar masses for `m`, dimensionless for `χ`, km/s for kicks and `v_esc`, `Gpc^-3 yr^-1` for rates, redshift dimensionless. Match gwkokab where ambiguous (their cosmology returns `dV_c / dz` in `Gpc^3`).
- **Parameter names**: align with `gwkokab.parameters.Parameters` so distributions interoperate. We use `mass_1_source`, `mass_2_source`, `a_1`, `a_2`, `redshift` for the event-space coordinates. The coagulation engine works internally with `(m1, m2, χ_1, χ_2)` ordered so `m1 ≥ m2`; the wrapper symmetrises.
- **Rate convention**: `log_prob` returns `log(dN / dV_c dt dθ)` (rate density), not a unit-normalised PDF — matching gwkokab's `_RedshiftModel.log_prob`. The Poisson normalisation `μ(Λ) = E[VT(θ)]` is computed by the gwkokab likelihood.
- **Grids**: `m` log-spaced ∈ `[2 M⊙, 200 M⊙]`, default `N_m = 96`; `χ` linear ∈ `[0, 1]`, default `N_χ = 32`; `z` linear ∈ `[0, 4]`, default `N_z = 64`; `v_esc` log-spaced ∈ `[1, 1000] km/s`, default `N_ve = 24`. All sizes configurable; bigger grids tighten quadrature at the cost of compile-time memory.

## 6 gwkokab integration

The engine's public face on the gwkokab side is

```python
from hierarchical_ad_pop_engine.distributions import CoagulationPopulation

def coagulation_model_factory(**params, validate_args=None) -> ScaledMixture:
    response  = CoagulationResponse.from_params(**params)
    formation = build_formation(**params)
    universe  = SyntheticUniverse(response=response, formation=formation, grids=...)
    dist      = CoagulationPopulation(universe, validate_args=validate_args)
    return ScaledMixture(
        log_scales=jnp.asarray([params["log_rate"]]),
        component_distributions=[dist],
        support=dist.support,
        validate_args=validate_args,
    )
```

This is a drop-in replacement for `gwkokab.models.hybrids.PowerlawPeak`, usable directly with `gwkokab.inference.numpyro_poisson_likelihood`. The factory pattern matches gwkokab's other models exactly.

## 7 v1 scope and roadmap

**v1 (this iteration):**
- JAX only.
- Generational coagulation response with NR-calibrated remnant + retention.
- Two FormationRate models (Delta, LogNormal × Madau-Dickinson).
- `numpyro.Distribution` wrapper plug-in to gwkokab.
- Unit tests including finite-difference gradient check.
- A demo script: invert noise-free synthetic data with `jax.grad` / Adam to recover injected `θ`.

**v2** (proposal Year 2):
- Continuous (ODE) coagulation backend via `diffrax`, swap-in compatible.
- Average over source environments — joint inference over `θ_env` distribution shape.
- Secular BH-parameter evolution (spindown, accretion).

**v3** (proposal Year 3):
- Non-BH populations (BH-star encounters).
- Orbital eccentricity.
- First application to the GW census.

## 8 Open questions / decisions to revisit

1. **Spin representation in v1**: aligned-component `(χ_1, χ_2)`. v1 uses *pure aligned-spin* kick physics — deterministic `v_kick(m1, m2, χ1, χ2)` and a Heaviside retention step (no tilt marginalisation). Tilt-marginalised kicks are a clean v2 swap target.
2. **Generation cap `K_max`**: 4 by default. Verified by tail-probability check (next generation contributes `< 1e-3` of total rate at fiducial parameters).
3. **Whether the `(j, k')` pair-rate `R_{j,k'}` is a free parameter or pinned by `n_j n_k'` integrals**: for v1, `R_{j,k'} = n_j(env) · n_k'(env) · σ(env)` where `σ(env)` is a single environment-level cross section — one parameter. Future work can expand this.
4. **Redshift evolution within an environment**: ignored in v1 (instantaneous coagulation); v2 lifts.
5. **Spin tilts in the kick formula**: deferred to v2. v1 is *pure aligned-spin*, Heaviside retention, no tilt smearing. v2 will add a proper tilt-marginalised kick CDF (analytic for simple tilt priors, pre-tabulated for general ones), restoring AD in `v_esc`.

## 9 Verification strategy

1. **Limits**: zero pairing kernel → only natal density survives. `v_esc → ∞` → all retained, generations grow indefinitely (test with finite `K_max`). `v_esc → 0` → only 1+1 mergers survive.
2. **Delta-formation collapses universe to bare response.**
3. **Finite-difference vs `jax.grad`** for the expected-rate integral with respect to every exposed parameter.
4. **JIT compilation** of the full pipeline (no concrete-shape escapes).
5. **Round-trip recovery**: generate synthetic posterior samples from fiducial `θ_0`, run gradient-based inference, recover `θ_0` to within posterior width.
