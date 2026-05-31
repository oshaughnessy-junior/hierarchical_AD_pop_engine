r""":class:`MCEvaluatorJax` — JAX-pure point-query evaluator for MC samples.

Session **J3** of ``demo/PLAN_jax_mc.md`` — the JAX-traceable companion
to :class:`MCEvaluator` (the numpy / scipy.cKDTree k-NN balloon
estimator in ``mc.py``).  Built so that the eventual J4 SVI integration
test under ``solver="mc_jax"`` has a likelihood surface that can be
differentiated end-to-end through the J2 :func:`particle_mc_jax`
cascade.

Design choices (locked by ``demo/PLAN_jax_mc.md`` § J3):

* **Fixed-bandwidth anisotropic Gaussian KDE** in ``(log m, χ)``.  The
  bandwidth is set by Silverman's rule of thumb on the
  *weighted-sample effective sample size* per axis:

  .. math::

      h_d \;=\; \hat\sigma_{w,d} \cdot n_{\rm eff}^{-1/(d_{\rm eff}+4)},
      \qquad
      \hat\sigma_{w,d}^2 \;=\; \frac{\sum_i w_i\,(x_{i,d} - \bar x_{w,d})^2}{W},
      \quad n_{\rm eff} \;=\; \tfrac{(\sum w)^2}{\sum w^2}.

  Both ``log m`` and ``χ`` get their own bandwidth (``d_eff = 2``).
  Anisotropic so the natural-scale differences (log m has σ ≈ 0.1, χ
  has σ ≈ 0.3) don't smear features asymmetrically.
* **Differentiable in the bandwidth AND the samples**.  The bandwidth
  depends on per-sample weights via ``σ̂_w``; weights flow back through
  the J2 cascade to ``θ_pop``.  Sample positions also depend on
  ``θ_pop`` via the J1 reparametrisation.  AD-end-to-end.
* **Reflection at χ ∈ {0, 1}** — matches the numpy MCEvaluator's
  boundary handling.  ``log m`` is not reflected (natal distribution
  decays smoothly across the support).  Reflected copies inherit
  parent weights.  Density on χ ∈ (0, 1) is the sum of in-domain
  Gaussian kernels plus the two reflected sets; W_total stays the
  *original-sample* weight sum because reflections widen the kernel
  domain, not the probability mass.
* **Same engine-units convention as numpy MCEvaluator**:
  ``∫∫ N(m, χ) dm dχ ≈ W_total = Σ w_i``.  4-D rate density formula
  bit-for-bit identical: ``½ R_env K(m₁,m₂) N(m₁,χ₁) N(m₂,χ₂)``.
* **AD path**: full AD through ``density`` and ``rate_density_4d``.
  ``marginal_*`` and ``total_rate`` are not required AD-traceable
  per the Protocol; J3 implements ``total_rate`` AD-traceable
  anyway (cheap quadrature over the bandwidth-shaped grid) for
  future SVI gradient flow.  ``marginal_1d`` / ``marginal_2d``
  follow-up.

User objection logged: see
``[[feedback-gaussian-kde-over-smooths-cascade-peaks]]``.  Silverman-
on-n_eff is the simplest first JAX-pure path but is expected to
oversmooth narrow cascade χ peaks; the numpy k-NN balloon MCEvaluator
remains the production-quality density estimator for sharp-feature
work (Session D parity battery).

eqx.Module so the evaluator slots into the JAX pytree alongside
:class:`ImplicitEvaluator` and :class:`BasisEvaluator`.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .protocol import RateDensityEvaluator  # noqa: F401  (typing only)


__all__ = ["MCEvaluatorJax"]


_TWO_PI = 2.0 * jnp.pi


def _log_pairing_kernel_jax(m1: Array, m2: Array, a: float, b: float) -> Array:
    """JAX mirror of ``mc._log_pairing_kernel_numpy``."""
    m_big = jnp.maximum(m1, m2)
    m_sml = jnp.minimum(m1, m2)
    q = m_sml / m_big
    return a * jnp.log(m1 + m2) + b * jnp.log(q)


def _weighted_mean_var(
    x: Array, w: Array, W: Array
) -> Tuple[Array, Array]:
    """Weighted mean and (population) variance.

    ``W = Σ w`` is passed in to avoid recomputation; ``x`` and ``w``
    are 1-D arrays of equal length.  Variance is computed about the
    *weighted* mean.
    """
    mean = jnp.sum(w * x) / W
    var = jnp.sum(w * (x - mean) ** 2) / W
    return mean, var


def _silverman_bandwidth_1d(sigma_w: Array, n_eff: Array) -> Array:
    """Silverman's rule of thumb adapted for weighted samples.

    ``h = σ̂_w · n_eff^(-1/(d_eff + 4))`` with ``d_eff = 2`` for the
    2-D ``(log m, χ)`` problem.  Returns a scalar.
    """
    return sigma_w * n_eff ** (-1.0 / 6.0)


def _pilot_lambda(
    log_m: Array, chi: Array, w_kde: Array, W_kde: Array,
    h_log_m: Array, h_chi: Array, *, alpha: float = 0.5,
) -> Array:
    r"""Per-sample local-bandwidth multipliers λ_i (Abramson sample-point KDE).

    Classic variable-bandwidth recipe (Abramson 1982 / Silverman §5.3):

    1. evaluate a FIXED-bandwidth pilot density ``f̃`` at each sample point;
    2. ``λ_i = (f̃_i / G)^{-α}`` with ``G`` the (weighted) geometric mean of
       ``f̃`` and ``α = 1/2`` — so the local bandwidth is ``h·λ_i``: SMALLER
       where the cascade is dense (sharpens the remnant peaks the global
       Silverman bandwidth over-smooths), LARGER in sparse tails.

    The estimator stays exactly mass-preserving (each kernel still integrates
    to ``w_i/W``), so the engine-units convention / H2 is untouched.  Pilot is
    interior-only (no χ-reflection) — it only needs to rank density, not nail
    the boundary — keeping it a single ``S×S`` pass computed once at
    construction.  AD-clean: every op is smooth in the sample positions /
    weights (which carry θ tangents); no neighbour selection.

    gen==1 samples have ``w_kde = 0`` so they neither contribute to the pilot
    nor need a meaningful λ; theirs is pinned to 1 (they are zero-weighted in
    the KDE sum regardless).
    """
    # Pilot: fixed-bw KDE evaluated AT each sample (S×S, weighted by w_kde).
    d_lm = log_m[:, None] - log_m[None, :]
    d_ch = chi[:, None] - chi[None, :]
    u2 = (d_lm / h_log_m) ** 2 + (d_ch / h_chi) ** 2
    norm = 1.0 / (_TWO_PI * h_log_m * h_chi)
    K = norm * jnp.exp(-0.5 * u2)                       # (S, S)
    f_pilot = jnp.sum(w_kde[None, :] * K, axis=1) / W_kde
    f_pilot = jnp.maximum(f_pilot, 1e-300)
    # Weighted geometric mean G over the gen>=2 pool (w_kde>0).
    log_G = jnp.sum(w_kde * jnp.log(f_pilot)) / W_kde
    G = jnp.exp(log_G)
    lam = (f_pilot / G) ** (-alpha)
    # Pin λ=1 for zero-weight (gen==1) samples — they drop out of the KDE sum
    # anyway, this just avoids a meaningless (0/G)^-α blow-up tangent.
    return jnp.where(w_kde > 0.0, lam, 1.0)


class MCEvaluatorJax(eqx.Module):
    r"""JAX-pure point-query evaluator over particle-MC samples.

    Implements :class:`RateDensityEvaluator` via a fixed-bandwidth
    anisotropic Gaussian KDE in ``(log m, χ)`` with Silverman's rule
    on weighted samples for the bandwidth.

    Parameters
    ----------
    m_active, chi_active : Array, shape (n_active,)
        Surviving single-BH pool from a J2 :func:`particle_mc_jax` run.
    mergers : Array, shape (n_mergers, 8)
        Accepted-merger array from the MC run.  Used for derived-axis
        marginals (deferred follow-up; matches the numpy
        MCEvaluator's step-3 scope).
    log_R_env : Array | float
        ``log(per-env-per-time merger amplitude)``.
    kernel_params : mapping
        ``{'a': float-like, 'b': float-like}`` pairing-kernel
        parameters.  Stored as a flat ``(a, b)`` pair internally so
        the eqx pytree is well-defined.
    weights : Array | None, shape (n_active,)
        Per-sample importance weights from the J2 cascade (or M3a/M3b
        natal IS).  ``None`` ⇒ uniform.  Strictly positive.
    reflect_chi : bool, default True
        Reflect samples across ``χ ∈ {0, 1}`` for boundary handling.
        ``False`` for inspection.

    Notes
    -----
    No scipy dependency — `n_active × Q` Gaussian evaluations in one
    fused jnp expression.  At n_active = 10⁴ and Q = 10³ this is ~40 MB
    of intermediate; under JIT it streams.

    The eqx.Module fields are split into:

    * static (no AD): ``reflect_chi``.
    * dynamic (AD): everything else — sample arrays, weights, log_R_env,
      kernel a / b.  This lets AD flow through ``θ_pop`` cleanly.
    """

    # ---- dynamic fields (AD-traceable) ------------------------------------
    m_active: Array
    chi_active: Array
    mergers: Array
    log_R_env: Array
    kernel_a: Array
    kernel_b: Array
    weights: Array            # uniform-ones if no user weights
    # Cached at construction so density() doesn't recompute every call.
    h_log_m: Array            # Silverman bandwidth in log m
    h_chi: Array              # Silverman bandwidth in χ
    W_total: Array            # Σ w_i
    n_eff: Array              # Kish effective sample size

    # ---- static fields (no AD) --------------------------------------------
    reflect_chi: bool = eqx.field(static=True)
    n_active: int = eqx.field(static=True)

    # --- hybrid analytic-1G extension (None / off unless hybrid_1g) -------- #
    # When ``hybrid_1g`` the gen==1 (natal) population is evaluated with the
    # EXACT parametric natal model (widths / locations / λ recoverable, fully
    # differentiable) and only the gen>=2 cascade remnants are KDE'd.  Fixes
    # the bimodal-Silverman width-recovery floor — see
    # ``demo/PLAN_hybrid_analytic_1g_density.md`` +
    # ``demo/DIAGNOSIS_two_pulse_kde_floor.md``.  All None when off, so the
    # default KDE path is bit-for-bit unchanged.
    w_kde: Array = None            # per-particle KDE weights (gen>=2 only)
    W_kde: Array = None            # Σ w_kde
    W_1g: Array = None             # Σ weight·(gen==1) — analytic-1G amplitude
    h_kde_log_m: Array = None      # Silverman bandwidth (gen>=2 pool) in log m
    h_kde_chi: Array = None        # Silverman bandwidth (gen>=2 pool) in χ
    # Per-sample local bandwidth multipliers λ_i for the gen>=2 KDE
    # (sample-point / Abramson variable-bandwidth estimator).  None unless
    # ``kde_adaptive``.  h_i = (h_kde_log_m·λ_i, h_kde_chi·λ_i): narrow in dense
    # cascade-peak regions, wide in sparse tails.  Mass-preserving (each kernel
    # still integrates to w_i/W), so H2 holds for free.  See _pilot_lambda.
    lam_kde: Array = None
    natal_mass_params: dict = None
    natal_spin_params: dict = None
    hybrid_1g: bool = eqx.field(static=True, default=False)
    natal_mass_kind: str = eqx.field(static=True, default=None)
    kde_adaptive: bool = eqx.field(static=True, default=False)

    # --- memory-bounded query evaluation ---------------------------------- #
    # ``density`` materialises a dense ``(Q_query × S_sample)`` Gaussian-kernel
    # matrix.  At production resolution (S = 3·n_seed reflected samples, Q =
    # n_quad² grid points) that is hundreds of MB per call and is retained
    # across the reverse-mode AD tape — too large for small dev machines.
    # When ``query_chunk`` is set, queries are evaluated in blocks of that
    # many points via a rematerialised ``lax.scan`` so peak memory (forward
    # AND backward) is bounded by ``query_chunk × S`` regardless of Q.  The
    # per-point kernel reduction is unchanged, so results are bit-for-bit
    # identical to the unchunked path (``query_chunk=None``, the default).
    query_chunk: int = eqx.field(static=True, default=None)

    @classmethod
    def from_run(
        cls,
        m_active: Array,
        chi_active: Array,
        mergers: Array,
        *,
        log_R_env: Array,
        kernel_params: Mapping[str, Array],
        weights: Array | None = None,
        reflect_chi: bool = True,
        gen_active: Array | None = None,
        hybrid_1g: bool = False,
        natal_mass_kind: str | None = None,
        natal_mass_params: dict | None = None,
        natal_spin_params: dict | None = None,
        query_chunk: int | None = None,
        kde_adaptive: bool = False,
    ) -> "MCEvaluatorJax":
        r"""Build from raw J2 output arrays, computing the cached bandwidths.

        Use this rather than the bare ``__init__`` — bandwidth caching
        is awkward to do inside an eqx.Module constructor (we'd need
        to recompute on every replace), so the named-constructor
        pattern matches :class:`ImplicitEvaluator.from_response` /
        :class:`BasisEvaluator.from_result`.
        """
        m = jnp.asarray(m_active).ravel()
        chi = jnp.asarray(chi_active).ravel()
        if m.shape != chi.shape:
            raise ValueError(
                f"m_active and chi_active must share shape; got "
                f"{m.shape} vs {chi.shape}"
            )
        if int(m.shape[0]) < 2:
            raise ValueError(
                "MCEvaluatorJax needs at least 2 active particles; got "
                f"n_active = {int(m.shape[0])}"
            )
        n_active = int(m.shape[0])

        # Weights: default to uniform-ones so the AD path is well-defined.
        if weights is None:
            w = jnp.ones(n_active, dtype=m.dtype)
        else:
            w = jnp.asarray(weights).ravel()
            if w.shape != m.shape:
                raise ValueError(
                    f"weights must have shape (n_active,) = {m.shape}; "
                    f"got {w.shape}"
                )

        # Bandwidth from weighted-sample Silverman.  σ̂_w uses the
        # weighted variance about the weighted mean; n_eff is Kish 1965.
        W_total = jnp.sum(w)
        sum_w_sq = jnp.sum(w * w)
        # Guard W=0 / sum_w_sq=0 via eps — won't trip in practice since
        # particle_mc_jax outputs strictly-positive weights.
        n_eff = (W_total * W_total) / jnp.maximum(sum_w_sq, 1e-30)

        log_m = jnp.log(jnp.maximum(m, 1e-30))
        _, var_log_m = _weighted_mean_var(log_m, w, jnp.maximum(W_total, 1e-30))
        _, var_chi = _weighted_mean_var(chi, w, jnp.maximum(W_total, 1e-30))
        sigma_log_m = jnp.sqrt(jnp.maximum(var_log_m, 1e-30))
        sigma_chi = jnp.sqrt(jnp.maximum(var_chi, 1e-30))
        h_log_m = _silverman_bandwidth_1d(sigma_log_m, n_eff)
        h_chi = _silverman_bandwidth_1d(sigma_chi, n_eff)

        # --- hybrid analytic-1G extension (off unless hybrid_1g) ----------- #
        # Split the pool by generation: gen==1 weight feeds the analytic
        # natal model (W_1g); gen>=2 weight feeds a SEPARATE KDE with its own
        # Silverman bandwidth (un-bimodal-inflated).  ∫∫N = W_1g + W_kde =
        # W_total, so the engine-units convention is preserved downstream.
        w_kde = W_kde = W_1g = h_kde_log_m = h_kde_chi = lam_kde = None
        if hybrid_1g:
            if gen_active is None:
                raise ValueError("hybrid_1g=True requires gen_active")
            g = jnp.asarray(gen_active).ravel()
            w_kde = w * (g >= 2).astype(w.dtype)
            W_1g = jnp.sum(w * (g == 1).astype(w.dtype))
            W_kde = jnp.sum(w_kde)
            sum_w_kde_sq = jnp.sum(w_kde * w_kde)
            n_eff_kde = (W_kde * W_kde) / jnp.maximum(sum_w_kde_sq, 1e-30)
            W_kde_safe = jnp.maximum(W_kde, 1e-30)
            _, var_lm_kde = _weighted_mean_var(log_m, w_kde, W_kde_safe)
            _, var_chi_kde = _weighted_mean_var(chi, w_kde, W_kde_safe)
            h_kde_log_m = _silverman_bandwidth_1d(
                jnp.sqrt(jnp.maximum(var_lm_kde, 1e-30)), n_eff_kde
            )
            h_kde_chi = _silverman_bandwidth_1d(
                jnp.sqrt(jnp.maximum(var_chi_kde, 1e-30)), n_eff_kde
            )
            if kde_adaptive:
                lam_kde = _pilot_lambda(
                    log_m, chi, w_kde, W_kde_safe, h_kde_log_m, h_kde_chi
                )

        return cls(
            m_active=m,
            chi_active=chi,
            mergers=jnp.asarray(mergers),
            log_R_env=jnp.asarray(log_R_env),
            kernel_a=jnp.asarray(kernel_params["a"]),
            kernel_b=jnp.asarray(kernel_params["b"]),
            weights=w,
            h_log_m=h_log_m,
            h_chi=h_chi,
            W_total=W_total,
            n_eff=n_eff,
            reflect_chi=bool(reflect_chi),
            n_active=n_active,
            w_kde=w_kde,
            W_kde=W_kde,
            W_1g=W_1g,
            h_kde_log_m=h_kde_log_m,
            h_kde_chi=h_kde_chi,
            lam_kde=lam_kde,
            natal_mass_params=natal_mass_params,
            natal_spin_params=natal_spin_params,
            hybrid_1g=hybrid_1g,
            natal_mass_kind=natal_mass_kind,
            kde_adaptive=bool(kde_adaptive),
            query_chunk=(None if query_chunk is None else int(query_chunk)),
        )

    # ---- 1. Single-BH density ---------------------------------------------
    def density(self, m: Array, chi: Array) -> Array:
        r"""``N(m, χ)`` via anisotropic Gaussian KDE in ``(log m, χ)``.

        Engine-units: ``∫∫ N dm dχ ≈ W_total`` (= n_active for uniform
        weights, bit-for-bit matching the numpy MCEvaluator
        convention).

        Out-of-domain queries (``χ ∉ [0, 1]`` or ``m ≤ 0``) get 0.

        AD-traceable in ``self.m_active`` / ``self.chi_active`` /
        ``self.weights`` (which carry θ_pop derivatives through the
        J2 cascade) and in the bandwidths (which depend on the same).
        """
        m_arr = jnp.asarray(m)
        chi_arr = jnp.asarray(chi)
        shape_out = jnp.broadcast_shapes(m_arr.shape, chi_arr.shape)
        m_b = jnp.broadcast_to(m_arr, shape_out).ravel()
        chi_b = jnp.broadcast_to(chi_arr, shape_out).ravel()

        if self.query_chunk is None:
            N = self._density_flat(m_b, chi_b)
        else:
            N = self._density_chunked(m_b, chi_b)
        return N.reshape(shape_out)

    def _density_flat(self, m_b: Array, chi_b: Array) -> Array:
        r"""``N(m, χ)`` on flat 1-D query arrays — the un-chunked core.

        Materialises the full ``(Q × S)`` Gaussian-kernel matrix in one
        shot.  Split out of :meth:`density` so the memory-bounded chunked
        path can call it once per query block.
        """
        log_m_q = jnp.log(jnp.maximum(m_b, 1e-30))

        # f_hat(log m, χ) — empirical density in (log m, χ) space,
        # ∫f_hat = 1 under uniform-ones weights when ignoring reflection.
        # With reflection, f_hat covers χ ∈ [-1, 2] and integrates to 1
        # there; restricting to χ ∈ [0, 1] gives ≈ 1/3 mass which
        # exactly compensates the 3x reflection augmentation factor.
        # (The W_total denominator uses the ORIGINAL weight sum, not the
        # augmented one — matches numpy MCEvaluator's convention.)
        if self.hybrid_1g:
            # gen>=2 KDE (own bandwidth) + EXACT analytic gen==1 component.
            f_kde = self._f_hat_weighted(
                log_m_q, chi_b, self.w_kde, self.W_kde,
                self.h_kde_log_m, self.h_kde_chi,
                lam=(self.lam_kde if self.kde_adaptive else None),
            )
            N = (
                self.W_kde * f_kde / jnp.maximum(m_b, 1e-30)
                + self._analytic_1g_density(m_b, chi_b)
            )
        else:
            f_hat = self._f_hat_log_m_chi(log_m_q, chi_b)
            # Change of variables to (m, χ): divide by m.  Multiply by
            # W_total so the result has engine-units (∫N dm dχ = W_total).
            N = self.W_total * f_hat / jnp.maximum(m_b, 1e-30)

        valid = (chi_b >= 0.0) & (chi_b <= 1.0) & (m_b > 0.0)
        return jnp.where(valid, N, 0.0)

    def _density_chunked(self, m_b: Array, chi_b: Array) -> Array:
        r"""Memory-bounded :meth:`_density_flat` over ``query_chunk`` blocks.

        Pads the flat query arrays up to a multiple of ``query_chunk``,
        reshapes to ``(n_chunks, query_chunk)`` and evaluates one block per
        ``lax.scan`` step.  The per-step density call is wrapped in
        ``jax.checkpoint`` so reverse-mode AD **recomputes** each block's
        ``(query_chunk × S)`` kernel matrix instead of stashing all
        ``n_chunks`` of them on the tape — this is what bounds the
        backward-pass memory (a plain scan would stack the residuals and
        restore the full ``Q × S`` footprint).

        Padded entries use ``m = 0`` (out-of-support → density 0) and are
        sliced off, so the result is bit-for-bit equal to
        :meth:`_density_flat` on the original queries.
        """
        chunk = int(self.query_chunk)
        Q = int(m_b.shape[0])
        n_chunks = -(-Q // chunk)  # ceil
        pad = n_chunks * chunk - Q
        m_pad = jnp.concatenate([m_b, jnp.zeros(pad, m_b.dtype)])
        chi_pad = jnp.concatenate([chi_b, jnp.zeros(pad, chi_b.dtype)])
        m_r = m_pad.reshape(n_chunks, chunk)
        chi_r = chi_pad.reshape(n_chunks, chunk)

        # Partition self so the scan/checkpoint only ever HASH the static
        # fields; the array leaves pass through as tracked tangents.  Wrapping
        # a bound method of an eqx.Module directly (``jax.checkpoint(
        # self._density_flat)``) makes plain JAX try to hash the whole module,
        # whose concrete array leaves are unhashable ("unhashable type:
        # ArrayImpl").  Combining inside the checkpointed block avoids that and
        # keeps the rematerialisation (the actual memory bound) intact.
        dynamic, static = eqx.partition(self, eqx.is_array)

        @jax.checkpoint
        def flat_block(dyn, mm, cc):
            ev = eqx.combine(dyn, static)
            return ev._density_flat(mm, cc)

        def body(_carry, xs):
            mm, cc = xs
            return _carry, flat_block(dynamic, mm, cc)

        _, N_r = jax.lax.scan(body, None, (m_r, chi_r))
        return N_r.reshape(-1)[:Q]

    def _f_hat_log_m_chi(self, log_m_q: Array, chi_q: Array) -> Array:
        r"""Empirical density in (log m, χ) at query points.

        Sums Gaussian kernels over all samples (+ reflected copies if
        ``reflect_chi``).  Bandwidths are the cached ``h_log_m`` and
        ``h_chi``.  Returns an array shaped like ``log_m_q``.
        """
        log_m_s = jnp.log(jnp.maximum(self.m_active, 1e-30))
        chi_s = self.chi_active
        w_s = self.weights

        if self.reflect_chi:
            # Reflect across χ = 0 and χ = 1.  Each reflected copy
            # inherits its parent's weight.
            chi_s_all = jnp.concatenate([chi_s, -chi_s, 2.0 - chi_s])
            log_m_s_all = jnp.concatenate([log_m_s, log_m_s, log_m_s])
            w_s_all = jnp.concatenate([w_s, w_s, w_s])
        else:
            chi_s_all = chi_s
            log_m_s_all = log_m_s
            w_s_all = w_s

        # Pairwise distances in (log m, χ).  Shape: (Q, S).  S = n_active
        # or 3 × n_active under reflection.
        d_log_m = log_m_q[:, None] - log_m_s_all[None, :]
        d_chi = chi_q[:, None] - chi_s_all[None, :]
        u2 = (d_log_m / self.h_log_m) ** 2 + (d_chi / self.h_chi) ** 2
        # 2-D anisotropic Gaussian kernel:
        #   K(u, v) = (1 / (2π h_log_m h_chi)) · exp(-0.5 (u² + v²))
        # Σ w_i K weighted KDE; divided by W_total (the
        # original-sample weight sum, NOT the augmented one — matches
        # numpy MCEvaluator's reflection convention bit-for-bit).
        # Under reflect_chi the augmented w_s_all sums to 3·W_total
        # but the kernel-sum integrates to 1 over χ ∈ [-1, 2] (3-wide
        # domain), so the (sample-sum) / (3·W_total) division gives the
        # correct in-domain density.
        log_norm = -jnp.log(_TWO_PI * self.h_log_m * self.h_chi)
        log_kernel = log_norm - 0.5 * u2
        kernel = jnp.exp(log_kernel)
        # Reflection: 3·W_total in the denominator under reflection so
        # the in-domain density integrates to 1.
        denom = jnp.where(
            jnp.array(self.reflect_chi),
            3.0 * self.W_total,
            self.W_total,
        )
        f_hat = jnp.sum(w_s_all[None, :] * kernel, axis=1) / denom
        # Under reflect_chi the augmented samples-summed kernel covers
        # χ ∈ [-1, 2]; restricting to [0,1] is the in-domain density.
        # But we also need the augmented mass: ∫_{-1}^{2} f̂ dχ = 1 by
        # construction, and ∫_{0}^{1} = ⅓.  Multiply by 3 so the
        # original sample distribution's mass is recovered in [0,1].
        f_hat = jnp.where(
            jnp.array(self.reflect_chi),
            3.0 * f_hat,
            f_hat,
        )
        return f_hat

    # ---- 1b. hybrid analytic-1G helpers -----------------------------------
    def _f_hat_weighted(self, log_m_q, chi_q, weights, W, h_log_m, h_chi,
                        lam=None):
        r"""Reflection KDE with explicit weights / W / bandwidths.

        Parameterised mirror of :meth:`_f_hat_log_m_chi` so the hybrid 2G+
        component can use the gen>=2 weight subset and its own Silverman
        bandwidth.  Returns ``f_hat`` normalised to integrate to 1 over
        (log m, χ∈[0,1]) under ``W`` (density() then multiplies by ``W``).

        ``lam`` (per-sample, length S) engages the Abramson variable-bandwidth
        estimator: sample ``i`` uses ``(h_log_m·λ_i, h_chi·λ_i)``, with the
        per-sample 2-D Gaussian normalisation ``1/(2π h_log_m h_chi λ_i²)`` so
        the estimator stays mass-preserving.  ``None`` → the fixed-bandwidth
        path (bit-for-bit unchanged).
        """
        log_m_s = jnp.log(jnp.maximum(self.m_active, 1e-30))
        chi_s = self.chi_active
        w_s = weights

        if self.reflect_chi:
            chi_s_all = jnp.concatenate([chi_s, -chi_s, 2.0 - chi_s])
            log_m_s_all = jnp.concatenate([log_m_s, log_m_s, log_m_s])
            w_s_all = jnp.concatenate([w_s, w_s, w_s])
            lam_all = (None if lam is None
                       else jnp.concatenate([lam, lam, lam]))
        else:
            chi_s_all = chi_s
            log_m_s_all = log_m_s
            w_s_all = w_s
            lam_all = lam

        d_log_m = log_m_q[:, None] - log_m_s_all[None, :]
        d_chi = chi_q[:, None] - chi_s_all[None, :]
        if lam_all is None:
            u2 = (d_log_m / h_log_m) ** 2 + (d_chi / h_chi) ** 2
            log_norm = -jnp.log(_TWO_PI * h_log_m * h_chi)
            kernel = jnp.exp(log_norm - 0.5 * u2)
        else:
            hl = h_log_m * lam_all[None, :]
            hc = h_chi * lam_all[None, :]
            u2 = (d_log_m / hl) ** 2 + (d_chi / hc) ** 2
            # per-sample normalisation 1/(2π hl hc) keeps each kernel unit-mass.
            kernel = jnp.exp(-0.5 * u2) / (_TWO_PI * hl * hc)
        denom = jnp.where(jnp.array(self.reflect_chi), 3.0 * W, W)
        f_hat = jnp.sum(w_s_all[None, :] * kernel, axis=1) / jnp.maximum(
            denom, 1e-30
        )
        f_hat = jnp.where(jnp.array(self.reflect_chi), 3.0 * f_hat, f_hat)
        return f_hat

    def _analytic_1g_density(self, m, chi):
        r"""Exact parametric 1G density ``W_1g · f_mass(m) · f_spin(χ)``.

        Per unit ``m`` per unit ``χ``.  Out-of-support points -> 0.  ``m`` /
        ``chi`` are the already-ravelled query arrays from :meth:`density`.

        **AD-safe out-of-support handling** (the load-bearing subtlety).  The
        natal logpdfs (``truncnorm.logpdf``) return ``-inf`` outside their
        support, so the FORWARD value ``exp(-inf) = 0`` is correct — but the
        ``-inf``'s *gradient* is ``nan``, which poisons reverse-mode the moment
        a query lands out of support (e.g. a high-mass cascade remnant, or a
        measurement-scattered event, above ``m_max``).  That nan-in-grad is
        exactly what made the two-pulse MAP skip every step.  Fix per
        ``[[feedback-where-guards-forward-not-backward]]``: evaluate the
        logpdfs on inputs CLAMPED to the interior (so the differentiated branch
        only ever sees finite values), then ``jnp.where`` the result to 0
        outside.  Mirrors the same pattern ``natal.mass.log_natal_mass`` uses
        internally (``m_safe = where(inside, m, interior)``).
        """
        from ..natal.mass import (
            log_natal_mass_gaussian,
            log_natal_mass_two_gaussian,
        )
        from ..natal.spin import build_natal_spin, log_natal_spin

        params = self.natal_mass_params
        m_lo = jnp.asarray(params["m_min"])
        m_hi = jnp.asarray(params["m_max"])
        # INCLUSIVE bounds: the truncnorm natal pdfs are finite (and the spin
        # truncnorm with loc=0 PEAKS) AT the support boundary, and the mock
        # clips scattered events to exactly χ∈{0,1} — so masking the boundary
        # to 0 throws away real mass right where the data sits (it under-
        # integrated H2 by ~6% and drove natal_spin_scale to its rail).  We
        # mask only STRICTLY-out-of-support points (where logpdf=-inf → nan
        # grad); the boundary itself is in-support with finite gradient.
        inside_m = (m >= m_lo) & (m <= m_hi)
        # Interior surrogate for out-of-support queries (geometric mid-point,
        # same choice as natal.mass.log_natal_mass); never differentiated for
        # its value because we mask it out below — only needs to be finite.
        m_safe = jnp.where(inside_m, m, jnp.sqrt(m_lo * m_hi))

        inside_chi = (chi >= 0.0) & (chi <= 1.0)
        chi_safe = jnp.where(inside_chi, chi, 0.5)

        if self.natal_mass_kind == "two_gaussian":
            log_fm = log_natal_mass_two_gaussian(m_safe, **params)
        elif self.natal_mass_kind == "gaussian":
            log_fm = log_natal_mass_gaussian(m_safe, **params)
        else:
            raise ValueError(
                "hybrid_1g unsupported natal_mass_kind "
                f"{self.natal_mass_kind!r}"
            )
        spin_dist = build_natal_spin(**self.natal_spin_params)
        log_fchi = log_natal_spin(spin_dist, chi_safe)

        val = self.W_1g * jnp.exp(log_fm + log_fchi)
        return jnp.where(inside_m & inside_chi, val, 0.0)

    # ---- 2. 4-D rate density ----------------------------------------------
    def rate_density_4d(
        self,
        m1: Array,
        m2: Array,
        chi1: Array,
        chi2: Array,
    ) -> Array:
        r"""``½ R_env K(m₁,m₂) N(m₁,χ₁) N(m₂,χ₂)`` at point queries.

        Same algebraic form as :meth:`ImplicitEvaluator.rate_density_4d`
        / :meth:`BasisEvaluator.rate_density_4d` / numpy
        :meth:`MCEvaluator.rate_density_4d`; cross-method comparison
        is apples-to-apples.

        AD-traceable end-to-end (the J3 win over numpy).
        """
        m1_a = jnp.asarray(m1)
        m2_a = jnp.asarray(m2)
        chi1_a = jnp.asarray(chi1)
        chi2_a = jnp.asarray(chi2)
        log_K = _log_pairing_kernel_jax(
            m1_a, m2_a, self.kernel_a, self.kernel_b,
        )
        N1 = self.density(m1_a, chi1_a)
        N2 = self.density(m2_a, chi2_a)
        R_env = jnp.exp(self.log_R_env)
        return 0.5 * R_env * jnp.exp(log_K) * N1 * N2

    # ---- 3. log_rate_density (the SVI workhorse) --------------------------
    def log_rate_density(
        self,
        m1: Array,
        m2: Array,
        chi1: Array,
        chi2: Array,
    ) -> Array:
        r"""``log(½ R_env K · N · N)`` at point queries.

        Computed in log space to avoid underflow when densities are
        tiny.  AD-traceable; this is the surface gwkokab
        PoissonLikelihood eventually consumes under
        ``solver="mc_jax"``.
        """
        rate = self.rate_density_4d(m1, m2, chi1, chi2)
        # eps protects against log(0) at out-of-domain queries.  ``where``-guard
        # so the floored branch carries ZERO gradient (a bare
        # ``log(maximum(rate, eps))`` still routes 1/rate back through a
        # tiny-but-positive rate → overflow → 0·inf = nan in any downstream
        # logsumexp; see MCResponseJax._safe_log_floor /
        # ``[[feedback-where-guards-forward-not-backward]]``).
        floor = 1e-300
        rate_safe = jnp.maximum(rate, floor)
        return jnp.where(rate > floor, jnp.log(rate_safe), jnp.log(floor))

    # ---- 4. total_rate ----------------------------------------------------
    def total_rate(self, *, n_quad: int = 64) -> Array:
        r"""``∫∫∫∫ ½ R_env K N N dm₁ dm₂ dχ₁ dχ₂`` via grid quadrature.

        Internal grid spans ``m ∈ [min(m_active), max(m_active)]``
        (geomspaced) and ``χ ∈ [0, 1]`` (linear).  Same convention as
        the numpy MCEvaluator's :meth:`total_rate`.

        AD-traceable (the implementation is pure jnp).
        """
        m_min = jnp.min(self.m_active)
        m_max = jnp.max(self.m_active)
        log_m_grid = jnp.linspace(jnp.log(m_min), jnp.log(m_max), n_quad)
        m_grid = jnp.exp(log_m_grid)
        chi_grid = jnp.linspace(0.0, 1.0, n_quad)

        # Density on the (m × χ) grid.
        M_grid, C_grid = jnp.meshgrid(m_grid, chi_grid, indexing="ij")
        N_grid = self.density(M_grid, C_grid)  # (n_quad, n_quad)

        # Trapezoid quadrature weights.
        # m: integral ∫ f dm on geomspaced grid via log-trapezoid.
        dlog = jnp.diff(log_m_grid)
        w_log_m = jnp.concatenate(
            [dlog[:1], 0.5 * (dlog[:-1] + dlog[1:]), dlog[-1:]]
        )
        w_m = w_log_m * m_grid

        # χ: linear trapezoid.
        d_chi = jnp.diff(chi_grid)
        w_chi = jnp.concatenate(
            [d_chi[:1], 0.5 * (d_chi[:-1] + d_chi[1:]), d_chi[-1:]]
        )

        # N marginalised over χ → 1-D function of m.
        N_int_chi = jnp.sum(N_grid * w_chi[None, :], axis=1)  # (n_quad,)

        # log K(m1, m2) on the grid.
        log_K_mm = _log_pairing_kernel_jax(
            m_grid[:, None], m_grid[None, :], self.kernel_a, self.kernel_b,
        )
        K_mm = jnp.exp(log_K_mm)
        R_env = jnp.exp(self.log_R_env)

        return 0.5 * R_env * jnp.einsum(
            "i,k,ik,i,k->", N_int_chi, N_int_chi, K_mm, w_m, w_m,
        )

    # ---- 5. marginal_1d / marginal_2d (follow-up; raise for now) ----------
    def marginal_1d(self, axis: str, x: Array) -> Array:
        raise NotImplementedError(
            "MCEvaluatorJax.marginal_1d is deferred to a follow-up J3.5 "
            "commit; the J3 minimum surface is density + rate_density_4d "
            "+ log_rate_density + total_rate (the SVI/PoissonLikelihood "
            "workhorses).  For now, use the numpy MCEvaluator if you need "
            "1-D marginals (Session D parity battery)."
        )

    def marginal_2d(
        self,
        axes: Tuple[str, str],
        x_a: Array,
        x_b: Array,
    ) -> Array:
        raise NotImplementedError(
            "MCEvaluatorJax.marginal_2d is deferred to a follow-up J3.5 "
            "commit; the J3 minimum surface is density + rate_density_4d "
            "+ log_rate_density + total_rate."
        )
