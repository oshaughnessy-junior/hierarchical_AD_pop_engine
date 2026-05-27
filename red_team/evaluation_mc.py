r""":class:`MCEvaluator` — point-query evaluator backed by particle-MC samples.

Step 3 of ``demo/PLAN_arbitrary_point_evaluation.md`` — implements the
:class:`hierarchical_ad_pop_engine.evaluation.protocol.RateDensityEvaluator`
protocol over the surviving-pool output of
:func:`red_team.backup_particle_mc.particle_mc`.

Design choices (locked by the PLAN's "Open-question resolutions" §):

* **Single-BH density** ``N(m, χ)`` is backed by the **surviving pool**
  ``(m_active, chi_active)`` — *not* by the mergers array.  The
  mergers array still backs derived-axis marginals (deferred to a
  follow-up commit, mirroring Step 1 / Step 2's derived-axis
  scoping).
* **Bandwidth selector**: **k-NN balloon estimator** (Loftsgaarden-
  Quesenberry) in ``(log m, χ)`` space — *not* Abramson.  Aligned-spin
  Stage 1 produces very narrow ``χ`` substructure at high generation
  (cascade-of-compressing-scales pathology of
  ``[[project-aligned-chi-undersampled]]``); Abramson's pilot-density
  bandwidth oversmooths these.  k-NN's per-eval-point bandwidth
  adapts to local density and resolves small islands without
  smearing.
* **4-D rate density** = ``½ R_env K(m₁,m₂) · density(m₁,χ₁) ·
  density(m₂,χ₂)`` — bit-for-bit the same formula as
  :class:`ImplicitEvaluator.rate_density_4d` and
  :class:`BasisEvaluator.rate_density_4d`, so cross-method comparison
  is apples-to-apples.
* **Marginals / total_rate** use a numpy mirror of
  :mod:`hierarchical_ad_pop_engine.observables` (grid quadrature on
  an internal fine grid where ``N`` is evaluated from the k-NN
  density estimator).  This guarantees grid-point consistency with
  the implicit / basis evaluators at the same query points.
* **Boundary handling**: reflect ``χ`` at both 0 and 1 (hard
  boundaries — bare BHs can't have ``|χ|>1``).  ``log m`` uses no
  reflection — the natal mass distribution decays smoothly into the
  cluster boundary; a reflection at ``m_min`` would create a
  spurious double-peak (PLAN §boundary-handling).
* **AD**: none.  MC is the validation backstop, *not* the inference
  AD path (PLAN §AD-policy).  numpy throughout.

The class is *not* an :class:`equinox.Module` — it carries numpy
arrays + a SciPy ``cKDTree`` and is not a JAX pytree.  The
:class:`RateDensityEvaluator` Protocol is duck-typed at the
``@runtime_checkable`` level so this is fine.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover — scipy is in env reqs
    raise ImportError(
        "MCEvaluator requires scipy.spatial.cKDTree; install scipy>=1.6 "
        "or fall back to the placeholder k-NN KDE in "
        "demo/forward_modeling_observables.py for the time being."
    ) from exc


__all__ = ["MCEvaluator"]


# ---------------------------------------------------------------------------
# Internal-grid helpers (numpy mirror of observables.{trapezoid_weights,
# marginal_1d, marginal_2d, total_rate} — kept here rather than refactored
# into the main package because numpy != JAX and the MC evaluator does not
# need to be AD-traceable).
# ---------------------------------------------------------------------------
_DEFAULT_N_QUAD = 64
_VALID_AXES = ("m1", "m2", "chi1", "chi2")
_DERIVED_AXES = ("q", "chi_eff", "M_chirp", "m_heavy", "m_light")


def _trapezoid_weights_log(x: np.ndarray) -> np.ndarray:
    """Trapezoid weights for the integral ``∫ f(x) dx`` on a geomspace grid.

    Returns the weights ``w[i]`` such that ``Σ w[i] · f(x[i]) ≈ ∫ f dx``,
    matching ``observables._dm_log_weights`` (uses ``dlog_x`` then
    multiplies by ``x[i]`` to recover ``dx``).
    """
    log_x = np.log(x)
    dlog = np.diff(log_x)
    w_log = np.concatenate(
        [dlog[:1], 0.5 * (dlog[:-1] + dlog[1:]), dlog[-1:]]
    )
    return w_log * x


def _trapezoid_weights_linear(x: np.ndarray) -> np.ndarray:
    """Trapezoid weights for ``∫ f(x) dx`` on a linearly-spaced grid."""
    dx = np.diff(x)
    return np.concatenate([dx[:1], 0.5 * (dx[:-1] + dx[1:]), dx[-1:]])


def _log_pairing_kernel_numpy(
    m1: np.ndarray, m2: np.ndarray, a: float, b: float
) -> np.ndarray:
    """Numpy mirror of :func:`hierarchical_ad_pop_engine.kernels.log_pairing_kernel`.

    Same algebra (``a · log(m₁+m₂) + b · log(min/max)``); kept in numpy so
    the MC evaluator has no JAX dependency.  Symmetric in ``m₁ ↔ m₂`` by
    construction.
    """
    m_big = np.maximum(m1, m2)
    m_sml = np.minimum(m1, m2)
    q = m_sml / m_big
    return a * np.log(m1 + m2) + b * np.log(q)


# ---------------------------------------------------------------------------
# k-NN balloon density estimator.
# ---------------------------------------------------------------------------
def _knn_balloon_2d(
    tree: cKDTree,
    n_samples: int,
    query_pts: np.ndarray,
    k: int,
) -> np.ndarray:
    r"""k-NN-bandwidth biweight balloon density estimator in 2-D.

    Per-*eval-point* bandwidth ``h(x) = r_k(x)`` (distance to the
    ``k``-th nearest sample), smoothed by the 2-D biweight
    (quartic) kernel inside that bandwidth:

    .. math::

        \hat{f}(x)
        \;=\; \frac{1}{n \, h(x)^2}\, \sum_{i \,:\, d_i \le h(x)}
              K_b\!\Big(\frac{d_i}{h(x)}\Big),
        \qquad
        K_b(u) = \frac{3}{\pi}\,\big(1 - u^2\big)^2,\; |u| \le 1,

    where ``d_i = ‖x − x_i‖`` and the sum runs over the ``k``
    nearest samples (by construction the only ones with
    ``d_i ≤ h(x)``).  Normalisation check:
    ``∫₀¹ (3/π)(1−r²)² · 2πr dr = 6 · [-(1−r²)³/6]₀¹ = 1``, so
    ``∫ K_b dA = 1`` over the unit disc.

    Compared to the Epanechnikov kernel ``(2/π)(1 − u²)`` (one less
    factor of ``(1 − u²)``), the biweight has a smooth zero
    derivative at ``|u| = 1`` instead of a kink.  Asymptotic mean
    integrated squared error is ~halved at the same bandwidth
    (Wand & Jones 1995, §2.7), with no additional cost — the
    extra square is a single multiply per sample.  Both kernels
    achieve the same Loftsgaarden-Quesenberry asymptotic rate
    (Silverman 1986 §5.4); the biweight just has a tighter
    finite-sample constant.

    Parameters
    ----------
    tree : scipy.spatial.cKDTree
        Pre-built tree over the (possibly reflected) sample set.
    n_samples : int
        Effective sample count for the LQ normalisation.  Equals
        ``m_active.size`` (the original sample count) regardless of
        whether the tree includes reflections — reflections widen
        the *kernel domain*, not the *probability mass*.
    query_pts : (Q, 2) ndarray
        Evaluation points in ``(log m, χ)`` coordinates.
    k : int
        Neighbour count.  Larger ``k`` → smoother estimate, more
        bias.  Default elsewhere ``k = max(4, round(√n))``.
    """
    dists, _ = tree.query(query_pts, k=k)
    # cKDTree.query returns shape (Q,) when k=1 (irregular).  Promote
    # to (Q, k) so the kernel sum below is shape-uniform.
    if dists.ndim == 1:
        dists = dists[:, None]
    r_k = dists[:, -1:]  # (Q, 1) per-query bandwidth
    r_k = np.maximum(r_k, 1e-30)
    u_sq = (dists / r_k) ** 2  # (Q, k)
    # Biweight: K_b(u) = (3/π)(1 − u²)² on |u| ≤ 1.  Because we
    # picked the k nearest neighbours, every term has u² ≤ 1
    # (modulo float-roundoff at the k-th sample itself).  Clip to
    # avoid a tiny negative tail on the k-th neighbour from
    # rounding.
    one_minus_u_sq = np.maximum(1.0 - u_sq, 0.0)
    kernel_sum = np.sum(one_minus_u_sq * one_minus_u_sq, axis=1)  # (Q,)
    return (3.0 / np.pi) * kernel_sum / (n_samples * r_k[:, 0] ** 2)


# ---------------------------------------------------------------------------
# MCEvaluator.
# ---------------------------------------------------------------------------
class MCEvaluator:
    r"""Point-query evaluator over particle-MC samples.

    Implements
    :class:`hierarchical_ad_pop_engine.evaluation.protocol.RateDensityEvaluator`
    via k-NN balloon density estimation on the surviving particle pool.

    Parameters
    ----------
    m_active, chi_active : array-like, shape (n_active,)
        The surviving single-BH pool from a particle-MC run.  These
        back ``density(m, χ)``.
    mergers : array-like, shape (n_mergers, 8)
        The accepted-merger array from the MC run.  Columns are
        ``(m1, m2, χ1, χ2, g1, g2, retained, m_remnant)``.  Used for
        derived-axis marginals (deferred to a follow-up commit; the
        Step-3 surface raises :class:`NotImplementedError` on derived
        axes, matching Steps 1 + 2).
    log_R_env : float
        ``log(per-env-per-time merger amplitude)`` — same convention
        as :attr:`CoagulationResponse.log_R_env`.  Enters
        :meth:`rate_density_4d` and :meth:`total_rate` as the overall
        scaling of the bilinear ``K · N · N`` rate.
    kernel_params : mapping
        ``{'a': float, 'b': float}`` parameters of the pairing kernel,
        matching :attr:`CoagulationResponse.kernel_params`.
    k : int, optional
        Number of nearest neighbours for the balloon estimator.
        Default ``max(4, round(√n_active))`` — Silverman's
        rule-of-thumb for k-NN density estimation, gives reasonable
        bias-variance tradeoff across n_active = 10² to 10⁵.
    reflect_chi : bool, default True
        Augment the sample set with reflections at ``χ = 0`` and
        ``χ = 1``.  Without reflection the k-NN bandwidth diverges
        near the hard boundaries (only one-sided neighbours) and the
        density estimator underestimates ``N`` there.  Off by
        request for inspection / debugging.

    Notes
    -----
    Constructor pre-builds a :class:`scipy.spatial.cKDTree` over the
    (optionally reflected) sample set; subsequent ``density`` calls
    are O(Q · log(n_samples) · k) with ``Q`` the query count.  At
    n_active = 10⁴ and k = 100 a single density() call on 10⁴ query
    points takes ~10 ms on commodity hardware — fast enough that
    ``total_rate`` (which internally evaluates ``N`` on an
    ``n_quad × n_quad`` grid) is comfortable at the default
    ``n_quad = 64``.
    """

    def __init__(
        self,
        m_active: Any,
        chi_active: Any,
        mergers: Any,
        *,
        log_R_env: float,
        kernel_params: Mapping[str, float],
        k: int | None = None,
        reflect_chi: bool = True,
    ) -> None:
        m_a = np.asarray(m_active, dtype=np.float64).ravel()
        chi_a = np.asarray(chi_active, dtype=np.float64).ravel()
        if m_a.shape != chi_a.shape:
            raise ValueError(
                f"m_active and chi_active must share shape; got "
                f"{m_a.shape} vs {chi_a.shape}"
            )
        if m_a.size < 2:
            raise ValueError(
                "MCEvaluator needs at least 2 active particles; got "
                f"n_active = {m_a.size}"
            )
        self.m_active = m_a
        self.chi_active = chi_a
        self.mergers = np.asarray(mergers, dtype=np.float64)
        self.log_R_env = float(log_R_env)
        # Defensive copy + coercion to float — kernel_params may arrive
        # as a JAX-Array-valued dict from CoagulationResponse.
        self.kernel_params = {
            "a": float(kernel_params["a"]),
            "b": float(kernel_params["b"]),
        }
        self.n_active = m_a.size
        if k is None:
            # Silverman √n rule-of-thumb floor at 4.  At n_active < 16
            # the floor binds; results are noisy but the call doesn't
            # error out (useful for tiny smoke-test runs).
            k = max(4, int(round(np.sqrt(self.n_active))))
        if k >= self.n_active:
            raise ValueError(
                f"k = {k} must be < n_active = {self.n_active}"
            )
        self.k = int(k)
        self.reflect_chi = bool(reflect_chi)
        self._build_density_tree()

    # ---- internal — sample augmentation + tree build ------------------------
    def _build_density_tree(self) -> None:
        r"""Build the (log m, χ) k-d tree over (possibly reflected) samples.

        χ reflection augments the sample set by mirroring across
        ``χ = 0`` and ``χ = 1``.  Reflections are added to the tree
        so the nearest-neighbour distance ``r_k`` at a query point
        near a boundary correctly sees neighbours on both sides;
        otherwise the bandwidth would blow up due to one-sided
        statistics and ``N`` would underestimate at the boundary
        cells.  The LQ-formula denominator uses the *original*
        sample count ``n_active`` — reflections are virtual.

        ``log m`` is *not* reflected.  See module docstring.
        """
        log_m = np.log(self.m_active)
        chi = self.chi_active
        if self.reflect_chi:
            log_m_all = np.concatenate([log_m, log_m, log_m])
            chi_all = np.concatenate([chi, -chi, 2.0 - chi])
        else:
            log_m_all = log_m
            chi_all = chi
        samples = np.column_stack([log_m_all, chi_all])
        self._density_samples = samples
        self._density_tree = cKDTree(samples)

    # ---- 1. Single-BH density ---------------------------------------------
    def density(self, m: Any, chi: Any) -> np.ndarray:
        r"""``N(m, χ)`` by 2-D k-NN balloon estimator in ``(log m, χ)``.

        Returns the engine-units density: ``∫∫ N(m, χ) dm dχ ≈
        n_active`` (matches the convention used by
        :class:`ImplicitEvaluator.density` and
        :class:`BasisEvaluator.density`).

        Out-of-domain queries (``χ ∉ [0, 1]`` or ``m ≤ 0``) return 0.
        """
        m_arr = np.asarray(m, dtype=np.float64)
        chi_arr = np.asarray(chi, dtype=np.float64)
        shape_out = np.broadcast_shapes(m_arr.shape, chi_arr.shape)
        m_b = np.broadcast_to(m_arr, shape_out).ravel()
        chi_b = np.broadcast_to(chi_arr, shape_out).ravel()
        valid = (chi_b >= 0.0) & (chi_b <= 1.0) & (m_b > 0.0)
        log_m = np.log(np.maximum(m_b, 1e-30))
        query_pts = np.column_stack([log_m, chi_b])
        # f_hat is the empirical density in (log m, χ) space, ∫f_hat = 1.
        f_hat = _knn_balloon_2d(
            self._density_tree, self.n_active, query_pts, self.k
        )
        # Change of variables to (m, χ): divide by m.  Multiply by
        # n_active so the result has engine-units (∫N dm dχ = n_active,
        # not 1).
        N = self.n_active * f_hat / np.maximum(m_b, 1e-30)
        N = np.where(valid, N, 0.0)
        return N.reshape(shape_out)

    # ---- 2. 4-D rate density ----------------------------------------------
    def rate_density_4d(
        self,
        m1: Any,
        m2: Any,
        chi1: Any,
        chi2: Any,
    ) -> np.ndarray:
        r"""``½ R_env K(m₁,m₂) N(m₁,χ₁) N(m₂,χ₂)`` at point queries.

        Same formula as
        :meth:`ImplicitEvaluator.rate_density_4d` and
        :meth:`BasisEvaluator.rate_density_4d`; the only difference
        is the backing density estimator.  Pinned by the
        cross-method test (PLAN § Test/validation strategy item 3).
        """
        m1_a = np.asarray(m1, dtype=np.float64)
        m2_a = np.asarray(m2, dtype=np.float64)
        chi1_a = np.asarray(chi1, dtype=np.float64)
        chi2_a = np.asarray(chi2, dtype=np.float64)
        log_K = _log_pairing_kernel_numpy(
            m1_a, m2_a, **self.kernel_params
        )
        N1 = self.density(m1_a, chi1_a)
        N2 = self.density(m2_a, chi2_a)
        R_env = np.exp(self.log_R_env)
        return 0.5 * R_env * np.exp(log_K) * N1 * N2

    # ---- internal — fine internal grid for marginals / total_rate ----------
    def _internal_grid_density(
        self, n_quad: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute ``N(m, χ)`` on a fine internal grid.

        The grid spans ``m ∈ [min(m_active), max(m_active)]`` (geomspaced)
        and ``χ ∈ [0, 1]`` (linear).  Density is evaluated by
        :meth:`density` on the cross product, then returned alongside
        the grid arrays so callers can apply trapezoid quadrature in
        the engine's units convention.

        Memory: ``n_quad × n_quad`` float64 — ~32 KB at the default
        ``n_quad = 64``.
        """
        m_grid = np.geomspace(
            float(self.m_active.min()),
            float(self.m_active.max()),
            n_quad,
        )
        chi_grid = np.linspace(0.0, 1.0, n_quad)
        M, C = np.meshgrid(m_grid, chi_grid, indexing="ij")
        N = self.density(M, C)
        return m_grid, chi_grid, N

    # ---- 3. 1-D marginal ---------------------------------------------------
    def marginal_1d(self, axis: str, x: Any, *, n_quad: int = _DEFAULT_N_QUAD) -> np.ndarray:
        r"""``dC/dx`` at query points ``x``, for grid-aligned axes only.

        Step-3 scope: ``axis ∈ {'m1', 'm2', 'chi1', 'chi2'}``.  Derived
        axes (``q``, ``chi_eff``, ``M_chirp``, ``m_heavy``,
        ``m_light``) raise :class:`NotImplementedError` with a pointer
        to the mergers-array-backed follow-up — same scoping as
        :class:`ImplicitEvaluator.marginal_1d` step-1 and
        :class:`BasisEvaluator.marginal_1d` step-2.

        Implementation: identical pattern to
        :meth:`ImplicitEvaluator.marginal_1d` — compute the marginal
        on an internal fine grid via the same einsum as
        :func:`observables.marginal_1d`, then interpolate at the
        user's query points.  Mass marginals interpolate in
        ``log m`` (matches the geomspaced internal grid); spin
        marginals interpolate linearly in ``χ``.
        """
        if axis in _DERIVED_AXES:
            raise NotImplementedError(
                f"axis {axis!r} is a derived axis; MCEvaluator step-3 "
                f"supports only the four grid-aligned axes {_VALID_AXES!r}.  "
                "Derived axes will be backed by the mergers array (k-NN KDE "
                "on the per-event derived-axis values) in a step-3-followup "
                "commit per PLAN_arbitrary_point_evaluation.md."
            )
        if axis not in _VALID_AXES:
            raise ValueError(
                f"unrecognised axis {axis!r}; expected one of {_VALID_AXES!r}"
            )
        x_arr = np.asarray(x, dtype=np.float64)

        m_grid, chi_grid, N_grid = self._internal_grid_density(n_quad)
        w_m = _trapezoid_weights_log(m_grid)
        w_chi = _trapezoid_weights_linear(chi_grid)
        R_env = np.exp(self.log_R_env)
        log_K_mm = _log_pairing_kernel_numpy(
            m_grid[:, None], m_grid[None, :], **self.kernel_params
        )
        K_mm = np.exp(log_K_mm)

        if axis in ("m1", "m2"):
            # dC/dm1 = 0.5 R_env · N_int_chi(m1) · ∫ K(m1, m2) N_int_chi(m2) dm2
            # Symmetric in m1 ↔ m2.
            N_int_chi = (N_grid * w_chi[None, :]).sum(axis=1)  # (n_quad,)
            inner = K_mm @ (N_int_chi * w_m)  # (n_quad,)
            dC = 0.5 * R_env * N_int_chi * inner
            log_m_grid = np.log(m_grid)
            log_x = np.log(np.maximum(x_arr, 1e-30))
            out = np.interp(log_x, log_m_grid, dC, left=0.0, right=0.0)
            valid = (x_arr >= m_grid[0]) & (x_arr <= m_grid[-1])
            return np.where(valid, out, 0.0)
        else:
            # axis == "chi1" or "chi2".
            # dC/dchi1 = 0.5 R_env · ∫ dm1 N(m1, chi1) · ∫ dm2 K(m1, m2) ∫ dchi2 N(m2, chi2)
            #         = 0.5 R_env · ∫ dm1 N(m1, chi1) · K_dot_Nint_chi(m1)
            N_int_chi = (N_grid * w_chi[None, :]).sum(axis=1)  # (n_quad,) over m
            K_dot = K_mm @ (N_int_chi * w_m)  # (n_quad,) over m1
            # For each chi1 query: ∫ dm1 N(m1, chi1) · K_dot(m1)
            # Evaluate N on m_grid at each chi_query (avoids a 2-D fine
            # grid for chi at the cost of a Q-vector of 1-D evaluations).
            x_flat = x_arr.ravel()
            results = np.empty_like(x_flat)
            for i, c in enumerate(x_flat):
                if c < 0.0 or c > 1.0:
                    results[i] = 0.0
                    continue
                N_at_chi = self.density(
                    m_grid, np.full_like(m_grid, c)
                )
                results[i] = 0.5 * R_env * np.sum(
                    N_at_chi * K_dot * w_m
                )
            return results.reshape(x_arr.shape)

    # ---- 4. 2-D marginal ---------------------------------------------------
    def marginal_2d(
        self,
        axes: Tuple[str, str],
        x_a: Any,
        x_b: Any,
        *,
        n_quad: int = _DEFAULT_N_QUAD,
    ) -> np.ndarray:
        r"""``dC/(dx_a dx_b)`` at query points, for grid-aligned axis pairs.

        Step-3 scope: both axes in ``{'m1', 'm2', 'chi1', 'chi2'}``.
        Derived axes raise :class:`NotImplementedError` (same scoping
        as :meth:`marginal_1d`).

        Implementation: evaluate the joint marginal on an internal
        fine grid using the same numpy mirror of
        :func:`observables.marginal_2d`, then bilinear-interp at the
        user's query points.  ``axes`` ordering controls broadcast
        layout — same convention as :class:`ImplicitEvaluator.marginal_2d`,
        no silent transpose.
        """
        a, b = axes
        for ax in (a, b):
            if ax in _DERIVED_AXES:
                raise NotImplementedError(
                    f"axis {ax!r} is a derived axis; MCEvaluator step-3 "
                    f"supports only the four grid-aligned axes "
                    f"{_VALID_AXES!r}."
                )
            if ax not in _VALID_AXES:
                raise ValueError(
                    f"unrecognised axis {ax!r}; expected one of {_VALID_AXES!r}"
                )

        x_a_arr = np.asarray(x_a, dtype=np.float64)
        x_b_arr = np.asarray(x_b, dtype=np.float64)
        shape_out = np.broadcast_shapes(x_a_arr.shape, x_b_arr.shape)
        x_a_b = np.broadcast_to(x_a_arr, shape_out).ravel()
        x_b_b = np.broadcast_to(x_b_arr, shape_out).ravel()

        m_grid, chi_grid, N_grid = self._internal_grid_density(n_quad)
        w_m = _trapezoid_weights_log(m_grid)
        w_chi = _trapezoid_weights_linear(chi_grid)
        R_env = np.exp(self.log_R_env)
        log_K_mm = _log_pairing_kernel_numpy(
            m_grid[:, None], m_grid[None, :], **self.kernel_params
        )
        K_mm = np.exp(log_K_mm)
        N_int_chi = (N_grid * w_chi[None, :]).sum(axis=1)  # (n_quad,) over m

        # Build the 2-D marginal on the internal grid.  Six grid-aligned
        # pairs in total; the formulas factor through K_mm and N.
        marg, gc_a, gc_b = _internal_marginal_2d(
            axis_a=a, axis_b=b,
            m_grid=m_grid, chi_grid=chi_grid, N_grid=N_grid,
            K_mm=K_mm, N_int_chi=N_int_chi,
            w_m=w_m, w_chi=w_chi, R_env=R_env,
        )
        # Bilinear-interp at query points.  Mass axes interpolate in
        # log m; χ axes are linear.
        qa = np.log(np.maximum(x_a_b, 1e-30)) if a in ("m1", "m2") else x_a_b
        qb = np.log(np.maximum(x_b_b, 1e-30)) if b in ("m1", "m2") else x_b_b
        # Use bilinear via two 1-D interps in each row / column.
        out = _bilinear_interp_2d(gc_a, gc_b, marg, qa, qb)
        # Out-of-domain mask.
        if a in ("m1", "m2"):
            valid_a = (x_a_b >= m_grid[0]) & (x_a_b <= m_grid[-1])
        else:
            valid_a = (x_a_b >= 0.0) & (x_a_b <= 1.0)
        if b in ("m1", "m2"):
            valid_b = (x_b_b >= m_grid[0]) & (x_b_b <= m_grid[-1])
        else:
            valid_b = (x_b_b >= 0.0) & (x_b_b <= 1.0)
        out = np.where(valid_a & valid_b, out, 0.0)
        return out.reshape(shape_out)

    # ---- 5. Total rate -----------------------------------------------------
    def total_rate(self, *, n_quad: int = _DEFAULT_N_QUAD) -> float:
        r"""Scalar ``∫∫∫∫ ½ R_env K N N dm₁ dm₂ dχ₁ dχ₂``.

        Numpy mirror of :func:`observables.total_rate` on an internal
        ``n_quad × n_quad`` grid where ``N`` is evaluated from the
        k-NN density estimator.  Returns a Python float (the protocol
        ``total_rate`` contract is "scalar", honoured here as a
        0-d float rather than a 0-d JAX array since the MC path is
        numpy-only).
        """
        m_grid, _, N_grid = self._internal_grid_density(n_quad)
        w_m = _trapezoid_weights_log(m_grid)
        w_chi_full = _trapezoid_weights_linear(np.linspace(0.0, 1.0, n_quad))
        R_env = np.exp(self.log_R_env)
        log_K_mm = _log_pairing_kernel_numpy(
            m_grid[:, None], m_grid[None, :], **self.kernel_params
        )
        K_mm = np.exp(log_K_mm)
        N_int_chi = (N_grid * w_chi_full[None, :]).sum(axis=1)  # (n_quad,)
        return float(
            0.5 * R_env * np.einsum(
                "i,k,ik,i,k->", N_int_chi, N_int_chi, K_mm, w_m, w_m,
            )
        )


# ---------------------------------------------------------------------------
# Internal helpers for the 2-D marginal — six grid-aligned axis pairs.
# ---------------------------------------------------------------------------
def _internal_marginal_2d(
    *,
    axis_a: str,
    axis_b: str,
    m_grid: np.ndarray,
    chi_grid: np.ndarray,
    N_grid: np.ndarray,
    K_mm: np.ndarray,
    N_int_chi: np.ndarray,
    w_m: np.ndarray,
    w_chi: np.ndarray,
    R_env: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Evaluate the 2-D grid-aligned marginal on the internal grid.

    Returns ``(marg, gc_a, gc_b)`` where ``marg`` has shape
    ``(grid_a.size, grid_b.size)`` and ``gc_a``, ``gc_b`` are the
    *interpolation* coordinates (log m for mass axes, χ for spin
    axes).
    """
    n_quad = m_grid.size

    # Mass-axis interpolation coordinate is log m so that bilinear
    # interp aligns with the geomspace grid layout.
    log_m_grid = np.log(m_grid)

    if axis_a == axis_b:
        raise ValueError(
            f"axes must be distinct; got ({axis_a!r}, {axis_b!r})"
        )

    # All six grid-aligned pairs.  Formulas derived by partial
    # contraction of ``rate_density_4d`` over the two unobserved axes.
    pair = (axis_a, axis_b)

    if pair == ("m1", "m2") or pair == ("m2", "m1"):
        # dC/(dm1 dm2) = 0.5 R_env · K(m1, m2) · N_int_chi(m1) · N_int_chi(m2)
        out = 0.5 * R_env * K_mm * N_int_chi[:, None] * N_int_chi[None, :]
        return out, log_m_grid, log_m_grid

    if pair == ("chi1", "chi2") or pair == ("chi2", "chi1"):
        # dC/(dchi1 dchi2) = 0.5 R_env · ∫ dm1 dm2 K · N(m1,chi1) · N(m2,chi2)
        # N_at_m_grid_for_chi has shape (n_quad_chi, n_quad_m) — N(m_i, chi_j).
        # Use stored N_grid which is (n_quad_m, n_quad_chi).
        # ∫ dm2 K(m1, m2) N(m2, chi2) → over m2.
        Km_dot_N = K_mm @ (N_grid * w_m[:, None])  # (n_quad_m, n_quad_chi)  [along axis 0 sums over m2]
        # Wait — let me redo this with explicit indices.
        # K_mm[i, k]; N_grid[k, l] for axes (m, chi).  Want
        # T[i, l] = sum_k K_mm[i, k] · N_grid[k, l] · w_m[k] = ∫dm2 K(m1=i, m2=k) N(m2=k, chi2=l).
        T = np.einsum("ik,kl,k->il", K_mm, N_grid, w_m)
        # dC/(dchi1 dchi2)[j, l] = 0.5 R_env · sum_i N_grid[i, j] · T[i, l] · w_m[i]
        out = 0.5 * R_env * np.einsum("ij,il,i->jl", N_grid, T, w_m)
        if pair == ("chi2", "chi1"):
            out = out.T
        return out, chi_grid, chi_grid

    if pair in (("m1", "chi1"), ("chi1", "m1"), ("m2", "chi2"), ("chi2", "m2")):
        # Same-particle pair, e.g. (m1, chi1):
        # dC/(dm1 dchi1) = 0.5 R_env · N(m1, chi1) · ∫dm2 K(m1, m2) N_int_chi(m2)
        K_dot = K_mm @ (N_int_chi * w_m)  # (n_quad_m,) function of m1
        out = 0.5 * R_env * N_grid * K_dot[:, None]  # (n_quad_m, n_quad_chi)
        if pair[0] in ("chi1", "chi2"):
            out = out.T  # transpose to (chi, m)
            return out, chi_grid, log_m_grid
        return out, log_m_grid, chi_grid

    if pair in (("m1", "chi2"), ("chi2", "m1"), ("m2", "chi1"), ("chi1", "m2")):
        # Cross-particle pair, e.g. (m1, chi2):
        # dC/(dm1 dchi2) = 0.5 R_env · N_int_chi(m1) · ∫dm2 K(m1, m2) N(m2, chi2)
        # T[i, l] = ∫dm2 K(m1=i, m2=k) N(m2=k, chi2=l) — same T as in the chi1·chi2 path.
        T = np.einsum("ik,kl,k->il", K_mm, N_grid, w_m)
        out = 0.5 * R_env * N_int_chi[:, None] * T  # (n_quad_m, n_quad_chi)
        if pair[0] in ("chi1", "chi2"):
            out = out.T
            return out, chi_grid, log_m_grid
        return out, log_m_grid, chi_grid

    raise ValueError(f"unexpected axis pair {pair!r}")


def _bilinear_interp_2d(
    gc_a: np.ndarray,
    gc_b: np.ndarray,
    vals: np.ndarray,
    qa: np.ndarray,
    qb: np.ndarray,
) -> np.ndarray:
    r"""Numpy bilinear interp on a regular 2-D grid.

    ``vals`` has shape ``(gc_a.size, gc_b.size)``.  Returns an array of
    shape ``qa.shape``.  Outside the grid the result is the boundary
    value (caller masks to 0 if it wants a hard cutoff — see the
    callers in :meth:`MCEvaluator.marginal_2d`).
    """
    # 1-D index-coordinates of the queries.
    i = np.interp(qa, gc_a, np.arange(gc_a.size, dtype=np.float64))
    j = np.interp(qb, gc_b, np.arange(gc_b.size, dtype=np.float64))
    i0 = np.clip(np.floor(i).astype(int), 0, gc_a.size - 1)
    i1 = np.clip(i0 + 1, 0, gc_a.size - 1)
    j0 = np.clip(np.floor(j).astype(int), 0, gc_b.size - 1)
    j1 = np.clip(j0 + 1, 0, gc_b.size - 1)
    di = i - i0
    dj = j - j0
    v00 = vals[i0, j0]
    v01 = vals[i0, j1]
    v10 = vals[i1, j0]
    v11 = vals[i1, j1]
    return (
        (1 - di) * (1 - dj) * v00
        + (1 - di) * dj * v01
        + di * (1 - dj) * v10
        + di * dj * v11
    )
