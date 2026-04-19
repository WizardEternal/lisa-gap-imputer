"""Classical imputation baselines for masked LISA strain segments.

Four imputers are provided, increasing in sophistication:

1. :func:`impute_zero` — fills masked positions with 0.  Establishes a
   lower-bound floor on reconstruction error; any serious method must beat it.

2. :func:`impute_linear` — piecewise linear interpolation between adjacent
   observed samples via :func:`numpy.interp`.  Fast and causal-free, but
   cannot follow the colored PSD of the noise and introduces abrupt slope
   discontinuities at gap edges that leak power into adjacent Fourier bins.

3. :func:`impute_cubic_spline` — cubic spline through all observed samples
   via :class:`scipy.interpolate.CubicSpline` with not-a-knot boundary
   conditions.  Smoother than linear, but still a deterministic polynomial fit
   that bears no structural relation to the true noise covariance.

4. :func:`impute_gp` — Gaussian process regression conditioned on a subsample
   of the observed data.  The GP kernel (Matérn-3/2 or RBF) is chosen to
   mimic the smoothness of the colored strain; in expectation the predictive
   mean matches what a Wiener filter would return for a stationary process with
   the given kernel covariance.  The main practical limitation is cubic scaling
   in the number of training points, mitigated here by subsampling.

All four share the same call signature so they can be stored in the
:data:`BASELINES` registry and exercised interchangeably by the evaluation
harness.

**Numerical conventions**

- The ``strain`` array at observed (non-masked) positions is **never
  modified**.  Every imputer starts with ``out = strain.copy()`` and writes
  only into positions where ``mask`` is ``True``.  This guarantees that
  ``out[~mask] == strain[~mask]`` exactly, with no floating-point drift.
- Sample indices are the natural domain axis for all interpolants.  For the
  GP, indices are normalised to ``[0, 1]`` before fitting so that
  ``length_scale`` is dimensionless and on the order of the gap width
  relative to the segment length.

References
----------
Rasmussen, C. E., & Williams, C. K. I. (2006).
    *Gaussian Processes for Machine Learning*.  MIT Press.
Virtanen, P. et al. (2020).
    SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python.
    *Nature Methods*, 17, 261–272.  https://doi.org/10.1038/s41592-019-0686-2
Pedregosa, F. et al. (2011).
    Scikit-learn: Machine Learning in Python.
    *JMLR*, 12, 2825–2830.

Author
------
Karan Akbari
"""

from __future__ import annotations

import functools
import warnings
from typing import Callable

import numpy as np
import numpy.typing as npt

__all__ = [
    "impute_zero",
    "impute_linear",
    "impute_cubic_spline",
    "impute_gp",
    "BASELINES",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_inputs(strain: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_]) -> None:
    """Validate common preconditions shared by every imputer.

    Parameters
    ----------
    strain : npt.NDArray[np.float64]
        Raw strain array, possibly with garbage at masked positions.
    mask : npt.NDArray[np.bool_]
        Boolean mask; ``True`` denotes a missing (gap) sample.

    Raises
    ------
    ValueError
        If shapes differ, arrays are not 1-D, or ``mask`` is not boolean dtype.
    """
    if strain.ndim != 1:
        raise ValueError(
            f"strain must be 1-D; got shape {strain.shape}."
        )
    if mask.ndim != 1:
        raise ValueError(
            f"mask must be 1-D; got shape {mask.shape}."
        )
    if strain.shape != mask.shape:
        raise ValueError(
            f"strain and mask must have the same shape; "
            f"got {strain.shape} and {mask.shape}."
        )
    if mask.dtype != np.bool_:
        raise ValueError(
            f"mask must have dtype bool; got {mask.dtype}."
        )


def _no_gaps_copy(strain: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a copy of ``strain`` — used when no gaps are present."""
    return strain.copy()


# ---------------------------------------------------------------------------
# 1. Zero imputer
# ---------------------------------------------------------------------------


def impute_zero(
    strain: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Fill masked positions with 0.0.

    This is the dumb floor — any physically motivated method should beat it on
    MSE for noise that has zero mean, but it is the only imputer guaranteed to
    introduce zero spectral energy at gap positions.  In practice it introduces
    sharp step discontinuities at gap edges that produce Gibbs-like ringing in
    the frequency domain.

    Parameters
    ----------
    strain : npt.NDArray[np.float64]
        Observed strain, shape ``(N,)``.  Values at masked positions are
        ignored.
    mask : npt.NDArray[np.bool_]
        Boolean gap mask, shape ``(N,)``.  ``True`` = missing.

    Returns
    -------
    npt.NDArray[np.float64]
        Shape ``(N,)``.  Observed samples are preserved exactly; masked
        positions are set to ``0.0``.

    Notes
    -----
    Even though 0.0 is an unbiased estimator for zero-mean Gaussian noise in
    expectation, the variance of the estimator error is :math:`\\sigma^2` —
    equal to the noise variance — so the MSE is as large as the signal power
    itself.  For a narrow gap this is sometimes *worse* than simple linear
    interpolation because the neighbours on either side are correlated with the
    gap contents via the noise covariance.
    """
    _check_inputs(strain, mask)
    n_missing = int(mask.sum())
    if n_missing == 0:
        return _no_gaps_copy(strain)
    out: npt.NDArray[np.float64] = strain.copy()
    out[mask] = 0.0
    return out


# ---------------------------------------------------------------------------
# 2. Linear imputer
# ---------------------------------------------------------------------------


def impute_linear(
    strain: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Linear interpolation across gaps using observed neighbours.

    Gaps are filled by piecewise-linear interpolation between the nearest
    observed samples on either side via :func:`numpy.interp`.  Edge gaps
    (runs of missing samples that touch index 0 or index ``N-1``) are filled
    with the first or last observed value respectively (constant extrapolation).

    Parameters
    ----------
    strain : npt.NDArray[np.float64]
        Observed strain, shape ``(N,)``.  Masked positions may hold any value.
    mask : npt.NDArray[np.bool_]
        Boolean gap mask, shape ``(N,)``.  ``True`` = missing.

    Returns
    -------
    npt.NDArray[np.float64]
        Shape ``(N,)``.  Observed samples are preserved exactly.

    Notes
    -----
    :func:`numpy.interp` performs piecewise-linear interpolation on sorted
    breakpoints.  The ``left`` and ``right`` keyword arguments control the
    constant extrapolation beyond the first and last observed sample; here
    they are set to the first and last observed values rather than the
    :func:`numpy.interp` default of ``nan``.

    Linear interpolation preserves DC but introduces a kink (slope
    discontinuity) at every gap boundary.  The spectral consequence is an
    :math:`O(1/f^2)` power floor injected into the gap region — acceptable
    for short gaps but damaging for long ones, where the interpolant can drift
    far from the true signal.

    **Fewer than 2 observed samples** — cannot form any interpolant; returns
    zeros with a :class:`UserWarning`.

    **All samples missing** — same fallback.
    """
    _check_inputs(strain, mask)
    n_missing = int(mask.sum())
    N = len(strain)

    if n_missing == 0:
        return _no_gaps_copy(strain)

    obs_idx: npt.NDArray[np.intp] = np.where(~mask)[0]

    if obs_idx.size == 0:
        warnings.warn(
            "impute_linear: all samples are masked — returning zeros.",
            UserWarning,
            stacklevel=2,
        )
        return np.zeros(N, dtype=np.float64)

    if obs_idx.size < 2:
        warnings.warn(
            "impute_linear: fewer than 2 observed samples — cannot interpolate; "
            "filling masked positions with the single observed value.",
            UserWarning,
            stacklevel=2,
        )
        out: npt.NDArray[np.float64] = strain.copy()
        out[mask] = strain[obs_idx[0]]
        return out

    out = strain.copy()
    out[mask] = np.interp(
        np.where(mask)[0],
        obs_idx,
        strain[obs_idx],
        left=strain[obs_idx[0]],
        right=strain[obs_idx[-1]],
    )
    return out


# ---------------------------------------------------------------------------
# 3. Cubic-spline imputer
# ---------------------------------------------------------------------------


def impute_cubic_spline(
    strain: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Cubic spline through the observed samples, gap positions interpolated.

    Uses :class:`scipy.interpolate.CubicSpline` with ``bc_type='not-a-knot'``
    (the default), which imposes that the third derivative is continuous across
    the second and second-to-last knots.  The spline is queried with
    ``extrapolate=True`` so that edge gaps (runs at the start or end of the
    array) are filled by the natural polynomial extrapolation of the outermost
    cubic piece.

    Parameters
    ----------
    strain : npt.NDArray[np.float64]
        Observed strain, shape ``(N,)``.  Masked positions may hold any value.
    mask : npt.NDArray[np.bool_]
        Boolean gap mask, shape ``(N,)``.  ``True`` = missing.

    Returns
    -------
    npt.NDArray[np.float64]
        Shape ``(N,)``.  Observed samples are preserved exactly.

    Notes
    -----
    **Minimum observed samples** — :class:`scipy.interpolate.CubicSpline`
    requires at least 4 breakpoints (it needs to solve for 4 coefficients per
    segment under the not-a-knot conditions).  When fewer than 4 observed
    samples are available the function silently falls back to
    :func:`impute_linear`.  When fewer than 2 observed samples exist (the
    :func:`impute_linear` threshold), the function returns zeros with a
    :class:`UserWarning`.

    **Extrapolation behaviour** — for edge gaps the cubic extrapolant can
    diverge rapidly if the spline's outermost segment has high curvature.
    This is an inherent limitation of polynomial extrapolation; no constraint
    is placed on the extrapolant here.  If divergence is a problem in practice
    one could switch ``extrapolate=False`` and fall back to constant
    extrapolation for out-of-range queries, but this costs the C¹-continuity
    at the first and last observed sample.

    **Spectral artifacts** — cubic splines have C² continuity everywhere
    (including at gap boundaries), which suppresses the :math:`O(1/f^2)`
    edge ringing seen in linear interpolation.  For narrow gaps the spline
    residual is typically dominated by the mismatch between the polynomial
    family and the true covariance structure of colored noise.
    """
    from scipy.interpolate import CubicSpline  # local import to keep top-level clean

    _check_inputs(strain, mask)
    n_missing = int(mask.sum())
    N = len(strain)

    if n_missing == 0:
        return _no_gaps_copy(strain)

    obs_idx: npt.NDArray[np.intp] = np.where(~mask)[0]

    if obs_idx.size == 0:
        warnings.warn(
            "impute_cubic_spline: all samples are masked — returning zeros.",
            UserWarning,
            stacklevel=2,
        )
        return np.zeros(N, dtype=np.float64)

    if obs_idx.size < 4:
        # CubicSpline cannot be constructed; fall back to linear.
        return impute_linear(strain, mask)

    cs = CubicSpline(obs_idx, strain[obs_idx], bc_type="not-a-knot", extrapolate=True)

    out: npt.NDArray[np.float64] = strain.copy()
    gap_idx: npt.NDArray[np.intp] = np.where(mask)[0]
    out[gap_idx] = cs(gap_idx)
    return out


# ---------------------------------------------------------------------------
# 4. Gaussian-process imputer
# ---------------------------------------------------------------------------


def impute_gp(
    strain: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    *,
    max_observed: int = 400,
    kernel: str = "matern32",
    length_scale: float = 50.0,
    noise_level: float = 1.0,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Gaussian process regression imputer.

    Fits a :class:`sklearn.gaussian_process.GaussianProcessRegressor` to a
    (possibly subsampled) set of observed strain values and predicts the
    posterior mean at the masked positions.

    Parameters
    ----------
    strain : npt.NDArray[np.float64]
        Observed strain, shape ``(N,)``.  Masked positions may hold any value.
    mask : npt.NDArray[np.bool_]
        Boolean gap mask, shape ``(N,)``.  ``True`` = missing.
    max_observed : int, optional
        Maximum number of observed samples used to fit the GP.  If more than
        ``max_observed`` observed samples are available, ``max_observed`` of
        them are selected by uniform index-space subsampling (stratified, not
        random), with optional jitter when ``rng`` is supplied.  Default 400.
        Increasing this improves fidelity at the cost of :math:`O(n^3)` GP
        training time.
    kernel : {"matern32", "rbf"}, optional
        Base kernel family.  ``"matern32"`` uses
        :class:`sklearn.gaussian_process.kernels.Matern` with ``nu=1.5``
        (once-differentiable sample paths, appropriate for LISA noise which
        has a steeply colored PSD);  ``"rbf"`` uses an infinitely-smooth RBF
        (squared exponential), which is smoother but can over-smooth sharp
        gap edges.  Both are wrapped in a :class:`ConstantKernel` amplitude
        and a :class:`WhiteKernel` observation-noise term.
    length_scale : float, optional
        Initial length scale in normalised index units (i.e. in units of ``N``
        samples).  The GP optimiser may adjust this during ``fit``.  Default
        50.0, which corresponds to 50 sample-widths in the normalised
        :math:`[0, 1]` feature space — suitable for gaps on the order of
        10–20 % of the segment.
    noise_level : float, optional
        Initial amplitude of the :class:`WhiteKernel` observation noise term.
        Scaled by the same normalisation as the feature axis.  Default 1.0.
    rng : np.random.Generator or None, optional
        NumPy random generator.  If supplied, used to add a small integer
        jitter to the uniformly-spaced subsample indices so that the training
        set is not rigidly grid-aligned (grid alignment can cause numerical
        ill-conditioning in the kernel matrix).  If ``None``, the uniform
        grid is used as-is.

    Returns
    -------
    npt.NDArray[np.float64]
        Shape ``(N,)``.  Observed samples are preserved exactly; masked
        positions are filled with the GP posterior mean.

    Notes
    -----
    **Complexity** — GP fitting is :math:`O(m^3)` in the number of training
    points ``m``.  For a segment of length ``N = 4096`` with 20 % missing,
    the observed set has ~3277 points.  Fitting on all of them is prohibitive;
    ``max_observed = 400`` keeps wall-clock cost under a second on a modern
    CPU while still capturing the covariance structure across the segment.

    **Feature normalisation** — sample indices :math:`i \\in \\{0, \\ldots, N-1\\}`
    are normalised to :math:`x_i = i / N \\in [0, 1)` before being passed to
    the GP.  This decouples the kernel ``length_scale`` from the absolute
    segment length and makes the default ``length_scale = 50.0`` interpretable
    as "the GP expects correlations over roughly 50 samples regardless of
    segment length."  Note that post-normalisation the ``length_scale`` is
    expressed in units of :math:`1/N`, so the physical correlation length
    in samples is ``length_scale * N``.

    **Subsampling strategy** — when the observed set exceeds ``max_observed``,
    indices are selected by::

        sel = np.linspace(0, len(obs_idx) - 1, max_observed).astype(int)
        train_idx = obs_idx[sel]

    This provides uniform coverage of the observed index axis.  If ``rng`` is
    provided, a small random shift (up to ±1 index in the obs array) is added
    before clipping to valid range, breaking any residual grid periodicity.

    **Fallback** — if the sklearn GP raises any exception (e.g. convergence
    failure, numerical ill-conditioning of the kernel matrix, or negative
    diagonal after Cholesky), :func:`impute_cubic_spline` is called instead
    and a :class:`UserWarning` is emitted.

    **Prediction** — only the posterior mean is used to fill the gap.
    Sampling from the posterior predictive distribution would give a
    statistically consistent realisation of the noise, which is more
    principled for downstream matched-filter tests, but it introduces
    stochasticity that complicates deterministic benchmarking.  A posterior
    sample mode could be added via the ``return_std=True`` path of
    :meth:`~sklearn.gaussian_process.GaussianProcessRegressor.predict` in a
    future extension.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor  # lazy import
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        Matern,
        RBF,
        WhiteKernel,
    )

    _check_inputs(strain, mask)
    n_missing = int(mask.sum())
    N = len(strain)

    if n_missing == 0:
        return _no_gaps_copy(strain)

    obs_idx: npt.NDArray[np.intp] = np.where(~mask)[0]

    if obs_idx.size == 0:
        warnings.warn(
            "impute_gp: all samples are masked — returning zeros.",
            UserWarning,
            stacklevel=2,
        )
        return np.zeros(N, dtype=np.float64)

    if obs_idx.size < 2:
        warnings.warn(
            "impute_gp: fewer than 2 observed samples — falling back to "
            "impute_cubic_spline (which will in turn fall back to zeros).",
            UserWarning,
            stacklevel=2,
        )
        return impute_cubic_spline(strain, mask)

    # ------------------------------------------------------------------ #
    # Build training set — subsample if needed.
    # ------------------------------------------------------------------ #
    if obs_idx.size > max_observed:
        sel: npt.NDArray[np.intp] = np.linspace(
            0, obs_idx.size - 1, max_observed
        ).astype(int)
        if rng is not None:
            # Jitter by ±1 position in the obs-index array, then clip.
            jitter = rng.integers(-1, 2, size=max_observed)  # {-1, 0, +1}
            sel = np.clip(sel + jitter, 0, obs_idx.size - 1)
            # Remove duplicates that may arise after jitter and clip, keeping order.
            sel = np.unique(sel)
        train_idx: npt.NDArray[np.intp] = obs_idx[sel]
    else:
        train_idx = obs_idx

    # ------------------------------------------------------------------ #
    # Normalise features to [0, 1).
    # ------------------------------------------------------------------ #
    X_train: npt.NDArray[np.float64] = (
        train_idx.astype(np.float64).reshape(-1, 1) / N
    )
    y_train: npt.NDArray[np.float64] = strain[train_idx]

    gap_idx: npt.NDArray[np.intp] = np.where(mask)[0]
    X_pred: npt.NDArray[np.float64] = (
        gap_idx.astype(np.float64).reshape(-1, 1) / N
    )

    # ------------------------------------------------------------------ #
    # Build kernel.
    # ------------------------------------------------------------------ #
    amplitude = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
    white = WhiteKernel(
        noise_level=noise_level,
        noise_level_bounds=(1e-5, 1e2),
    )
    kernel_lower = kernel.lower()
    if kernel_lower == "matern32":
        base = Matern(
            length_scale=length_scale,
            length_scale_bounds=(1e-2, 1e4),
            nu=1.5,
        )
    elif kernel_lower == "rbf":
        base = RBF(
            length_scale=length_scale,
            length_scale_bounds=(1e-2, 1e4),
        )
    else:
        raise ValueError(
            f"Unknown kernel '{kernel}'; choose 'matern32' or 'rbf'."
        )
    full_kernel = amplitude * base + white

    gpr = GaussianProcessRegressor(
        kernel=full_kernel,
        normalize_y=True,  # subtract empirical mean; helps optimiser convergence
        n_restarts_optimizer=2,
        random_state=0,  # reproducible hyper-parameter optimisation
    )

    # ------------------------------------------------------------------ #
    # Fit + predict, with fallback on any sklearn error.
    # ------------------------------------------------------------------ #
    try:
        gpr.fit(X_train, y_train)
        y_pred: npt.NDArray[np.float64] = gpr.predict(X_pred)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"impute_gp: sklearn GP raised {type(exc).__name__}: {exc}. "
            "Falling back to impute_cubic_spline.",
            UserWarning,
            stacklevel=2,
        )
        return impute_cubic_spline(strain, mask)

    out: npt.NDArray[np.float64] = strain.copy()
    out[gap_idx] = y_pred
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINES: dict[str, Callable[[npt.NDArray[np.float64], npt.NDArray[np.bool_]], npt.NDArray[np.float64]]] = {
    "zero": impute_zero,
    "linear": impute_linear,
    "cubic_spline": impute_cubic_spline,
    "gp_matern32": functools.partial(impute_gp, kernel="matern32"),
}
"""Canonical registry of baseline imputers.

Keys map to callable imputers with signature
``(strain, mask) -> imputed_strain``.  Use this dict to loop over baselines
in the evaluation harness::

    from lisa_gap_imputer.baselines import BASELINES

    for name, fn in BASELINES.items():
        result = fn(strain, mask)

Available keys
--------------
``"zero"``
    :func:`impute_zero` — zero-fill.
``"linear"``
    :func:`impute_linear` — piecewise-linear interpolation.
``"cubic_spline"``
    :func:`impute_cubic_spline` — not-a-knot cubic spline.
``"gp_matern32"``
    :func:`impute_gp` with ``kernel="matern32"`` and default hyperparameters.
"""
