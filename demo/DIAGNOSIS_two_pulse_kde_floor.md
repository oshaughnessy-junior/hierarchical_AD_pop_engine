# Two-pulse recovery floor = the J3 KDE bandwidth (NOT the data, NOT the two-pulse impl)

**Symptom (workstation, 2026-05-30):** two-pulse fit recovers the pulse
*widths* completely wrong; **more observations don't help**; label swap fails.

**Root cause:** the `solver="mc_jax"` likelihood reads the population density
through the **J3 fixed-bandwidth Gaussian KDE** (`evaluation/mc_jax.py`,
`MCEvaluatorJax`), whose bandwidth is Silverman's rule on the *whole* surviving
pool: `h_log_m = σ̂_logm · n_eff^(−1/6)`.  For a **bimodal** natal IMF, `σ̂_logm`
is dominated by the **inter-pulse separation** (`ln(30/10)=1.10`), not the
within-pulse widths — so the bandwidth is inflated and **oversmooths both
pulses**.  This is the bimodal-acute form of the already-tracked
`[[project-mc-jax-total-rate-bias]]` / `[[feedback-gaussian-kde-over-smooths-cascade-peaks]]`.

### Why more observations can't help (the worrying part, explained)

There are two independent sample sizes:

* `n_seed` — cascade particles → builds the KDE pool → **sets the bandwidth**.
* `n_obs` — observed events → sets the **likelihood's statistical power**.

The bandwidth depends on `n_seed`/`n_eff`, **never on `n_obs`**.  So the
model's *representable* pulse width is fixed before a single event is seen.
Adding observations only tightens the posterior **around the KDE-smeared
width** — it can never recover a width narrower than the bandwidth.  That is
exactly "increasing observations didn't make it better."

And increasing `n_seed` barely helps: `n_eff^(−1/6)` is a brutally weak shrink
(10× particles → h × 0.68), and `σ̂_logm` stays bimodal-inflated.

### Measured (sandbox, IS-weighted 1G natal pool; truth σ=1 @10, σ=2 @30)

| estimator | pulse-1 FWHM (M⊙) | pulse-2 FWHM (M⊙) |
|---|---|---|
| **truth** | 2.2 | 4.5 |
| **J3 Silverman KDE** (h_logm≈0.14–0.21) | 3.9 (1.8×) | **10.5 (2.3×)** |
| narrow / adaptive KDE (h_logm=0.04) | 2.5 | 5.1 |

Bandwidth vs within-pulse log-width: `h/σ_pulse1 ≈ 1.4–2.1×`, `h/σ_pulse2 ≈
2.1–3.1×` — i.e. each pulse is smeared to 2–3× its true width, independent of
`n_obs`.  Figure: `demo/figures/two_pulse_kde_bandwidth_diagnosis.png`.

**Label swap** is a *consequence*: with pulse 2 smeared to ~10 M⊙ FWHM the two
blobs overlap heavily, the `(μ₁,μ₂)` likelihood is shallow, and only the
disjoint-window prior separates them — no sharp basin for a swapped start to
fall into.  Width recovery returns the label structure.

**The two-pulse stack itself is sound** — natals, IS weights, cascade, and AD
all validated (`tests/test_two_pulse_natal.py` 7/7; λ AD rel 1.6e-8). The
bottleneck is purely the density *estimator* the likelihood reads through.

---

## Fix options (AD-compatible — SVI/NUTS need a differentiable density)

1. **Hybrid analytic-1G + KDE-(2G+) density (recommended).**  The dominant
   first-generation population *is* the parametric natal model — evaluate it
   with the exact `log_natal_mass_two_gaussian × natal_spin` (widths/locations/λ
   exactly recoverable, fully differentiable), and KDE **only** the `gen≥2`
   remnant pool (which has no closed form).  `N(m,χ) = n₁^analytic + n_{≥2}^KDE`;
   the cascade already tracks `gen_active`, so the pool splits cleanly.  Most
   physical: KDE-ing the very component you're inferring throws away its
   structure.  Engine change in `MCResponseJax.log_rate_density` composition.

2. **AD-smooth adaptive (k-NN balloon) bandwidth — the deferred J3.5.**  Replace
   global Silverman with a local bandwidth (the k-NN balloon already resolves
   these peaks in the numpy `MCEvaluator`, 7.6% rel-L1).  More general but the
   AD-clean + numerically-stable balloon bandwidth is the harder build.
   Honours `[[feedback-no-hardcoded-physical-scales]]` (bandwidth data-discovered)
   and `[[feedback-mise-kernel-not-the-lever]]` (bandwidth, not kernel, is the lever).

3. (Not recommended) hardcode / model-anchor the bandwidth — circular and
   violates `[[feedback-no-hardcoded-physical-scales]]`.

**Recommendation:** option 1 for the methods-paper two-pulse run (exact natal
recovery, minimal new machinery), with option 2 as the general-purpose
follow-up that also fixes the 2G+ χ-cascade peaks and the `total_rate` metric
bias.
