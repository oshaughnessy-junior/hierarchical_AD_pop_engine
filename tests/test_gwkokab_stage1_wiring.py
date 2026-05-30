r"""Stage-1 wiring tests — priors / variables / variables_index + likelihood.

Validates :mod:`hierarchical_ad_pop_engine.inference.stage1_wiring`, the glue
that makes the gwkokab bridge a *runnable* Stage-1 fit.

Two tiers (same philosophy as ``test_gwkokab_bridge.py``):

* **In-sandbox tier (W1–W6)** — needs only ``jax`` + ``numpyro``.  Pins the
  pure-numpyro pieces: the 6 priors and their bounds, the **sorted-order
  ``variables_index`` contract**, the prior-key validation, and — the
  load-bearing ones — an inline replica of the likelihood's
  ``sorted_variables → variables_samples → mapped_params → dist_fn`` routing
  (W5 proves the index map sends each sample to the right keyword; W6 runs the
  full analytical-Poisson contract against a duck-typed response and
  differentiates through it).  These are exactly the bugs the real gwkokab
  likelihood is silent about.

* **Full-stack tier (W7–W9)** — builds the REAL gwkokab ``JointDistribution``
  (W7), checks its marginal order agrees with ``variables_index`` (W8), and
  runs our priors/variables/variables_index through the REAL
  ``numpyro_analytical_poisson_likelihood`` against a duck-typed dist_fn,
  differentiating the resulting log-density (W9).  ``import gwkokab`` pulls
  RIFT/lalsuite, so these SKIP in the lightweight sandbox and run on a
  workstation / cardassia.  An import/env failure is a legitimate skip; any
  OTHER error (API drift, a wiring bug) FAILS loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("numpyro")
import jax.numpy as jnp  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
for p in (str(_REPO), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (  # noqa: E402
    CoagulationPopulation,
    coagulation_poisson_mean_estimator,
)
from hierarchical_ad_pop_engine.inference.stage1_wiring import (  # noqa: E402
    STAGE1_PARAM_NAMES,
    build_coagulation_likelihood,
    build_stage1_priors,
    build_stage1_variables,
    stage1_prior_factories,
)

_M_MIN, _M_MAX, _V_ESC = 4.0, 80.0, 1000.0

# The Stage-1 prior bounds (PLAN_gwkokab_fit.md), keyed by name → (low, high).
_EXPECTED_BOUNDS = {
    "mu_natal": (5.0, 50.0),
    "sigma_natal": (0.5, 10.0),
    "natal_spin_loc": (0.0, 0.3),
    "natal_spin_scale": (0.02, 0.3),
    "kernel_b": (-2.0, 6.0),
    "log_rate": (-2.0, 6.0),
}
# Alphabetical sort of the 6 names — the gwkokab sample/index order.
_SORTED = sorted(STAGE1_PARAM_NAMES)  # kernel_b, log_rate, mu_natal, natal_spin_loc, natal_spin_scale, sigma_natal
_EXPECTED_INDEX = {name: i for i, name in enumerate(_SORTED)}


# ---------------------------------------------------------------------------
# Duck-typed response + dist_fn (AD-traceable; NOT physics — physics is the
# bridge G6/G7 + M4's job; here we drive the wiring's routing + AD).
# ---------------------------------------------------------------------------
def _fake_dist_fn(**const):
    r"""Return a 6-kwarg dist_fn(**mapped_params) -> CoagulationPopulation whose
    log-intensity / total_rate depend on the sampled params so gradients flow
    and so a mis-routed index shows up as a wrong number."""
    env = {"v_esc": jnp.asarray(_V_ESC)}

    def dist_fn(*, mu_natal, sigma_natal, natal_spin_loc,
                natal_spin_scale, kernel_b, log_rate, **_extra):
        class _FakeResp:
            class mc_jax_response:  # noqa: N801
                @staticmethod
                def total_rate(*, n_quad=64):
                    return jnp.exp(log_rate)

            def log_rate_density(self, m1, m2, chi1, chi2, _env):
                # depends distinctly on several params so a swap is visible
                return (log_rate
                        - 0.5 * ((m1 - mu_natal) ** 2 + (m2 - mu_natal) ** 2)
                        / (sigma_natal ** 2)
                        - kernel_b * 0.01 * (m1 - m2) ** 2
                        - (chi1 - natal_spin_loc) ** 2 / (2 * natal_spin_scale ** 2)
                        - (chi2 - natal_spin_loc) ** 2 / (2 * natal_spin_scale ** 2))

        resp = _FakeResp()
        return CoagulationPopulation(
            resp, env, lambda **kw: resp.mc_jax_response.total_rate(**kw),
            m_min=_M_MIN, m_max=_M_MAX,
        )

    return dist_fn


def _events(n=12):
    rng = np.random.default_rng(0)
    m1 = rng.uniform(8.0, 30.0, n)
    m2 = np.minimum(m1, rng.uniform(6.0, 25.0, n))
    chi1 = rng.uniform(0.0, 0.9, n)
    chi2 = rng.uniform(0.0, 0.9, n)
    return jnp.asarray(np.stack([m1, m2, chi1, chi2], axis=-1))


# ===========================================================================
# In-sandbox tier — W1–W6.
# ===========================================================================
def test_variables_are_the_six_priors():
    variables, _ = build_stage1_variables()
    assert set(variables) == set(STAGE1_PARAM_NAMES)
    for name, d in variables.items():
        assert isinstance(d, dist.Distribution), name
    print(f"\n[W1] variables = the 6 Stage-1 priors {tuple(sorted(variables))}")


def test_variables_index_is_sorted_contract():
    r"""variables_index[name] == position of name in sorted(variables) — the
    exact gwkokab convention (guru.classify_model_parameters)."""
    variables, variables_index = build_stage1_variables()
    assert variables_index == _EXPECTED_INDEX
    # And it really is the alphabetical enumerate, independent of insertion order.
    assert variables_index == {k: i for i, k in enumerate(sorted(variables))}
    assert sorted(variables_index.values()) == list(range(len(STAGE1_PARAM_NAMES)))
    print(f"\n[W2] variables_index = sorted-order contract {variables_index}")


def test_prior_bounds_match_plan():
    variables, _ = build_stage1_variables()
    for name, (lo, hi) in _EXPECTED_BOUNDS.items():
        d = variables[name]
        np.testing.assert_allclose(float(d.support.lower_bound), lo, rtol=0, atol=1e-9)
        np.testing.assert_allclose(float(d.support.upper_bound), hi, rtol=0, atol=1e-9)
        # type: Uniform vs LogUniform per PLAN
        is_log = name in ("sigma_natal", "natal_spin_scale")
        assert isinstance(d, dist.LogUniform) == is_log, name
    print("\n[W3] all 6 prior bounds + Uniform/LogUniform types match the PLAN")


def test_prior_factories_validate_keys():
    bad = dict(stage1_prior_factories())
    bad.pop("kernel_b")
    with pytest.raises(ValueError, match="kernel_b"):
        build_stage1_variables(bad)
    bad2 = dict(stage1_prior_factories())
    bad2["bogus"] = lambda: dist.Uniform(0.0, 1.0)
    with pytest.raises(ValueError, match="bogus"):
        build_stage1_variables(bad2)
    print("\n[W4] build_stage1_variables rejects missing / extra prior keys")


def test_index_routes_each_sample_to_its_name():
    r"""Inline replica of the likelihood's routing with the SAMPLES replaced by
    sentinels: variables_samples[i] := sorted_names[i].  A correct
    variables_index sends sorted_names[variables_index[name]] back to `name`
    for every name — i.e. mapped_params is the identity name→name.  Catches any
    off-by-index in the sorted-order contract."""
    variables, variables_index = build_stage1_variables()
    sorted_names = [name for name, _ in sorted(variables.items(), key=lambda kv: kv[0])]
    variables_samples = list(sorted_names)                     # sample i ≡ its name
    mapped_params = {name: variables_samples[i] for name, i in variables_index.items()}
    assert mapped_params == {name: name for name in STAGE1_PARAM_NAMES}
    print(f"\n[W5] index routing is the identity name→name; no slot is swapped")


def test_inline_analytical_poisson_contract_with_routing():
    r"""Full wiring path with a duck-typed response: route DISTINCT numeric
    sentinel values through sorted→index→mapped_params→dist_fn, run the inline
    analytical-Poisson contract (support.check + vmap log_prob + poisson_mean),
    and differentiate the -logL wrt a routed param.  Proves the assembled
    wiring evaluates and is AD-traceable end-to-end without gwkokab."""
    variables, variables_index = build_stage1_variables()
    sorted_names = [n for n, _ in sorted(variables.items(), key=lambda kv: kv[0])]
    dist_fn = _fake_dist_fn()

    x = _events(12)[:, None, :]            # (n_events, n_samples=1, 4)
    ln_offsets = jnp.zeros((12, 1))
    n_events = x.shape[0]

    # distinct, physically-plausible sentinel values per param (in sorted order)
    base = {
        "mu_natal": 9.0, "sigma_natal": 1.3, "natal_spin_loc": 0.04,
        "natal_spin_scale": 0.12, "kernel_b": 1.7, "log_rate": 0.5,
    }

    def neg_log_lik(mu_val):
        sample_vals = dict(base)
        sample_vals["mu_natal"] = mu_val
        variables_samples = [sample_vals[name] for name in sorted_names]
        mapped_params = {name: variables_samples[i] for name, i in variables_index.items()}
        model = dist_fn(**mapped_params)
        mask = model.support.check(x)
        lp = jax.vmap(model.log_prob)(x)
        total = jnp.sum(jax.nn.logsumexp(lp + ln_offsets, axis=1, where=mask))
        mu, _ = coagulation_poisson_mean_estimator(model, T_obs=1.0, n_quad=8)
        return -(total + n_events * jnp.log(1.0) - mu)

    val = float(neg_log_lik(jnp.asarray(9.0)))
    g = float(jax.grad(neg_log_lik)(jnp.asarray(9.0)))
    print(f"\n[W6] assembled-wiring -logL = {val:.3f}, d/d mu_natal = {g:.3f}")
    assert np.isfinite(val)
    assert np.isfinite(g) and g != 0.0


# ===========================================================================
# Full-stack tier — W7–W9.  Skip iff gwkokab (→ RIFT/lalsuite) not importable.
# ===========================================================================
_ENV_IMPORT_ERRORS = (ImportError, ModuleNotFoundError, OSError)


def _real_priors_or_skip():
    variables, variables_index = build_stage1_variables()
    try:
        priors = build_stage1_priors(variables)
    except _ENV_IMPORT_ERRORS as exc:
        pytest.skip(
            f"gwkokab JointDistribution not importable here (environment, "
            f"NOT a wiring bug): {type(exc).__name__}: {exc}"
        )
    return variables, variables_index, priors


def test_real_jointdistribution_shape_and_sample():
    variables, variables_index, priors = _real_priors_or_skip()
    # 6 marginals, event_shape (6,)
    assert len(priors.marginal_distributions) == len(STAGE1_PARAM_NAMES)
    s = priors.sample(jax.random.PRNGKey(0))
    assert s.shape == (len(STAGE1_PARAM_NAMES),)
    assert np.all(np.isfinite(np.asarray(priors.log_prob(s))))
    print(f"\n[W7] real JointDistribution: 6 marginals, sample shape {tuple(s.shape)}")


def test_real_marginal_order_matches_variables_index():
    r"""The JointDistribution marginal at slot variables_index[name] must be the
    prior for `name` — i.e. the joint's order and the index map agree (both are
    sorted(variables)).  Checks against the REAL gwkokab object."""
    variables, variables_index, priors = _real_priors_or_skip()
    for name, i in variables_index.items():
        marg = priors.marginal_distributions[i]
        lo, hi = _EXPECTED_BOUNDS[name]
        np.testing.assert_allclose(float(marg.support.lower_bound), lo, atol=1e-9)
        np.testing.assert_allclose(float(marg.support.upper_bound), hi, atol=1e-9)
    print("\n[W8] real JointDistribution marginal order agrees with variables_index")


def test_real_numpyro_analytical_likelihood_runs_and_differentiates():
    r"""Drive our (priors, variables, variables_index) through the REAL
    gwkokab numpyro_analytical_poisson_likelihood against a duck-typed dist_fn
    (bypasses the RIFT-pulling engine), evaluate the log-density inside a
    numpyro trace, and differentiate it wrt a substituted param value."""
    variables, variables_index, priors = _real_priors_or_skip()
    try:
        from numpyro.infer.util import log_density
        log_lik_fn = build_coagulation_likelihood(
            dist_fn=_fake_dist_fn(),
            priors=priors,
            variables=variables,
            variables_index=variables_index,
            poisson_mean_estimator=coagulation_poisson_mean_estimator,
        )
    except _ENV_IMPORT_ERRORS as exc:
        pytest.skip(
            f"gwkokab.inference not importable here (environment): "
            f"{type(exc).__name__}: {exc}"
        )

    x = _events(12)[:, None, :]
    ln_offsets = jnp.zeros((12, 1))
    pmean_kwargs = {"T_obs": 1.0, "n_quad": 8}
    fixed = {  # in-support substituted values for every sampled site
        "mu_natal": 9.0, "sigma_natal": 1.3, "natal_spin_loc": 0.04,
        "natal_spin_scale": 0.12, "kernel_b": 1.7, "log_rate": 0.5,
    }

    # log_density runs the numpyro model under substitute+trace and returns the
    # total log-joint (Σ prior log_probs of the sampled sites + the
    # analytical-Poisson ``log_likelihood`` factor) — a scalar, AD-traceable.
    # Differentiating wrt mu_natal flows through the likelihood factor (the
    # Uniform prior term is flat in mu, contributing 0), so a nonzero grad
    # proves the real gwkokab likelihood differentiates through our wiring.
    def logdensity(mu_val):
        params = dict(fixed, mu_natal=mu_val)
        lp, _ = log_density(log_lik_fn, (x, ln_offsets, pmean_kwargs), {}, params)
        return lp

    val = float(logdensity(jnp.asarray(9.0)))
    g = float(jax.grad(logdensity)(jnp.asarray(9.0)))
    print(f"\n[W9] REAL gwkokab analytical-Poisson log-density = {val:.3f}, "
          f"d/d mu_natal = {g:.3f}")
    assert np.isfinite(val)
    assert np.isfinite(g) and g != 0.0
