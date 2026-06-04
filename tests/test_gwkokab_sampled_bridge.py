r"""S-gates — sampled-mode (likelihood-mode) gwkokab wiring (PLAN item 5).

Engine side of ``demo/PLAN_gwkokab_sampled_likelihood.md``: the particle MC's
draws + weights exposed as the fork's ``sampler_fn``, per-event likelihood
loading, and the one-call ``build_sampled_likelihood`` wiring.

Tiers (same pattern as the bridge's G-gates / wiring W-gates):

* **S1–S3** — sampler contract: needs jax + ``red_team.backup_particle_mc``
  only.
* **S4–S8** — fork-facing: additionally need the gwkokab fork branch
  ``feature/sampled-poisson-likelihood`` importable (in the sandbox, under
  the RIFT import stub); ``importorskip``-gated otherwise.

Gradient gates assert **finite + nonzero**, never single-key |AD−FD|
(``feedback_single_key_fd_not_ad_reference``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

pytest.importorskip("red_team.backup_particle_mc")

from hierarchical_ad_pop_engine.inference.sampled_wiring import (  # noqa: E402
    SAMPLER_FREE_PARAMS,
    gaussian_event_log_likelihoods_from_posteriors,
    load_rift_event_log_likelihoods,
    make_coagulation_sampler_fn,
)


# Stage-1-like single-pulse constants (toy conventions)
THETA = dict(
    mu_natal=8.0,
    sigma_natal=1.0,
    natal_spin_loc=0.0,
    natal_spin_scale=0.05,
    kernel_b=2.0,
    log_rate=math.log(60.0),
)
CONST = dict(v_esc=300.0, m_min=4.0, m_max=80.0, n_seed=400,
             depletion_fraction=0.10)


@pytest.fixture(scope="module")
def sampler_fn():
    return make_coagulation_sampler_fn(gate="soft", **CONST)


@pytest.fixture(scope="module")
def draws(sampler_fn):
    return sampler_fn(**THETA)


# ───────────────────────── S1 — sampler contract ───────────────────────── #
def test_S1_sampler_contract(draws):
    samples, log_w = draws
    assert samples.ndim == 2 and samples.shape[1] == 4
    assert log_w.shape == (samples.shape[0],)
    assert bool(jnp.isfinite(samples).all())
    assert bool(jnp.isfinite(log_w).all())
    # heavier-first merger convention
    assert bool((samples[:, 0] >= samples[:, 1] - 1e-9).all())
    # Σ w_j = e^{log_rate}, exactly (normalisation is algebraic)
    lse = jax.scipy.special.logsumexp(log_w)
    assert abs(float(lse) - THETA["log_rate"]) < 1e-9


def test_S2_log_rate_pure_scale(sampler_fn, draws):
    """log_rate shifts every log-weight by δ and leaves the draws alone."""
    samples0, log_w0 = draws
    theta = dict(THETA, log_rate=THETA["log_rate"] + 0.7)
    samples1, log_w1 = sampler_fn(**theta)
    assert bool(jnp.array_equal(samples0, samples1))
    np.testing.assert_allclose(
        np.asarray(log_w1 - log_w0), 0.7, rtol=0, atol=1e-9
    )


def test_S3_grad_through_sampler_finite_nonzero(sampler_fn):
    """AD through draws AND weights to mu_natal / kernel_b: finite + nonzero."""

    def evidence_like(mu_natal, kernel_b):
        theta = dict(THETA, mu_natal=mu_natal, kernel_b=kernel_b)
        samples, log_w = sampler_fn(**theta)
        # Gaussian "event" centred near the 1g remnant peak — gradient must
        # flow through both the sample coordinates and the weights.
        log_L = (
            -0.5 * ((samples[:, 0] - 16.0) / 2.0) ** 2
            - 0.5 * ((samples[:, 1] - 8.0) / 2.0) ** 2
        )
        return jax.scipy.special.logsumexp(log_w + log_L)

    g_mu, g_b = jax.grad(evidence_like, argnums=(0, 1))(
        THETA["mu_natal"], THETA["kernel_b"]
    )
    for name, g in (("mu_natal", g_mu), ("kernel_b", g_b)):
        assert bool(jnp.isfinite(g)), f"grad wrt {name} not finite"
        assert float(jnp.abs(g)) > 0.0, f"grad wrt {name} is exactly zero"


# ──────────────────── S4–S5 — event-likelihood loading ─────────────────── #
def test_S4_gaussian_fits_from_posteriors():
    pytest.importorskip("gwkokab.inference.event_likelihoods")
    rng = np.random.default_rng(3)
    mean_true = np.array([12.0, 9.0, 0.1, -0.05])
    sig_true = np.array([1.0, 0.8, 0.15, 0.12])
    clouds = mean_true + sig_true * rng.standard_normal((3, 4000, 4))

    Ls = gaussian_event_log_likelihoods_from_posteriors(jnp.asarray(clouds))
    assert len(Ls) == 3
    # evaluator peaks at the fitted mean; fitted mean ~ true mean
    for L in Ls:
        np.testing.assert_allclose(
            np.asarray(L.mean), mean_true, atol=0.1
        )
        at_mean = float(L(L.mean[None, :])[0])
        away = float(L((L.mean + 3 * sig_true)[None, :])[0])
        assert at_mean > away


def test_S5_rift_loader_interface():
    pytest.importorskip("gwkokab.inference.event_likelihoods")

    def fake_interp(x):
        return -0.5 * jnp.sum(x**2, axis=-1)

    # callable passthrough
    (L,) = load_rift_event_log_likelihoods([fake_interp])
    x = jnp.ones((5, 4))
    np.testing.assert_allclose(np.asarray(L(x)), -2.0)

    # builder dispatch for opaque handles
    built = {}

    def builder(path):
        built["path"] = path
        return fake_interp

    (L2,) = load_rift_event_log_likelihoods(["event0.composite"], builder=builder)
    assert built["path"] == "event0.composite"
    np.testing.assert_allclose(np.asarray(L2(x)), -2.0)

    # non-callable without builder is a loud TypeError
    with pytest.raises(TypeError, match="builder"):
        load_rift_event_log_likelihoods(["event0.composite"])

    # linear-likelihood interpolants get logged
    (L3,) = load_rift_event_log_likelihoods(
        [lambda x: jnp.exp(fake_interp(x))], log_input=False
    )
    np.testing.assert_allclose(np.asarray(L3(x)), -2.0, atol=1e-12)


# ───────────────── S6–S8 — full wiring (fork required) ─────────────────── #
def _mock_events(sampler_fn, n_obs=6, seed=11):
    """Weighted-resample truth mergers + Gaussian scatter (toy convention)."""
    samples, log_w = sampler_fn(**THETA)
    x = np.asarray(samples)
    p = np.exp(np.asarray(log_w - jax.scipy.special.logsumexp(log_w)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n_obs, replace=True, p=p)
    truths = x[idx]
    sig = np.stack(
        [0.10 * truths[:, 0], 0.10 * truths[:, 1],
         np.full(n_obs, 0.2), np.full(n_obs, 0.2)], axis=1,
    )
    observed = truths + sig * rng.standard_normal(truths.shape)
    observed[:, 2:] = np.clip(observed[:, 2:], -0.998, 0.998)
    clouds = observed[:, None, :] + sig[:, None, :] * rng.standard_normal(
        (n_obs, 400, 4)
    )
    return gaussian_event_log_likelihoods_from_posteriors(jnp.asarray(clouds))


def test_S6_build_sampled_likelihood_wiring(sampler_fn):
    pytest.importorskip("gwkokab.models.utils")
    from hierarchical_ad_pop_engine.inference.sampled_wiring import (
        build_sampled_likelihood,
    )

    Ls = _mock_events(sampler_fn)
    built = build_sampled_likelihood(
        event_log_likelihood_fns=Ls, gate="soft", **CONST
    )
    assert set(built) == {
        "sampler_fn", "variables", "variables_index", "priors",
        "log_likelihood_fn",
    }
    # the load-bearing sorted-order contract (same as W3 in the stage-1 tests)
    assert sorted(built["variables"]) == sorted(SAMPLER_FREE_PARAMS)
    for i, name in enumerate(sorted(built["variables"])):
        assert built["variables_index"][name] == i


def test_S7_numpyro_log_density_finite_and_differentiable(sampler_fn):
    pytest.importorskip("gwkokab.models.utils")
    numpyro = pytest.importorskip("numpyro")
    from numpyro.infer.util import log_density
    from hierarchical_ad_pop_engine.inference.sampled_wiring import (
        build_sampled_likelihood,
    )

    Ls = _mock_events(sampler_fn)
    built = build_sampled_likelihood(
        event_log_likelihood_fns=Ls, gate="soft", **CONST
    )
    model = built["log_likelihood_fn"]

    def potential(mu):
        params = dict(THETA, mu_natal=mu)
        lp, _ = log_density(model, (jnp.asarray(1.0),), {}, params)
        return lp

    val = potential(jnp.asarray(THETA["mu_natal"]))
    grad = jax.grad(potential)(jnp.asarray(THETA["mu_natal"]))
    assert bool(jnp.isfinite(val))
    assert bool(jnp.isfinite(grad)) and float(jnp.abs(grad)) > 0.0


def test_S8_ess_diagnostic_flags_tail_event(sampler_fn):
    pytest.importorskip("gwkokab.inference.poissonlikelihood_utils")
    from gwkokab.inference.event_likelihoods import GaussianEventLikelihood
    from gwkokab.inference.poissonlikelihood_utils import (
        low_ess_events,
        sampled_poisson_likelihood_fn,
    )

    samples, log_w = sampler_fn(**THETA)
    n = samples.shape[0]
    central = GaussianEventLikelihood(
        jnp.asarray([16.0, 8.0, 0.0, 0.0]), jnp.asarray([4.0, 4.0, 0.04, 0.04])
    )
    tail = GaussianEventLikelihood(  # far outside the single-pulse population
        jnp.asarray([70.0, 65.0, 0.9, 0.9]),
        jnp.asarray([1.0, 1.0, 0.01, 0.01]),
    )
    ll, diag = sampled_poisson_likelihood_fn(
        (central, tail), samples, log_w, None, jnp.asarray(1.0), None
    )
    assert bool(jnp.isfinite(ll))
    ess = np.asarray(diag["ess_per_event"])
    assert ess[0] > ess[1]
    flagged = low_ess_events(diag["ess_per_event"], n, frac=0.01)
    assert 1 in flagged
