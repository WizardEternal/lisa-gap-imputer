from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.random import default_rng

from lisa_gap_imputer.baselines import BASELINES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strain_and_mask(
    n: int = 256,
    seed: int = 0,
    gap_start: int = 64,
    gap_end: int = 192,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a random float64 strain and a simple contiguous gap mask."""
    rng = default_rng(seed)
    strain = rng.standard_normal(n).astype(np.float64)
    mask = np.zeros(n, dtype=np.bool_)
    mask[gap_start:gap_end] = True
    return strain, mask


# ---------------------------------------------------------------------------
# Parametrised over all baselines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(BASELINES.keys()))
def test_baseline_output_shape_and_dtype(name: str) -> None:
    """Every baseline must return an array of the same shape and float64 dtype."""
    strain, mask = _make_strain_and_mask()
    fn = BASELINES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fn(strain, mask)
    assert out.shape == strain.shape, (
        f"Baseline '{name}': output shape {out.shape} != input shape {strain.shape}"
    )
    assert out.dtype == np.float64, (
        f"Baseline '{name}': output dtype {out.dtype} != float64"
    )


@pytest.mark.parametrize("name", list(BASELINES.keys()))
def test_baseline_preserves_observed(name: str) -> None:
    """Every baseline must preserve observed (non-masked) values exactly.

    For the GP imputer we allow atol=1e-10 to absorb any normalisation
    rounding.  For the others we require exact bit-equality.
    """
    strain, mask = _make_strain_and_mask()
    fn = BASELINES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fn(strain, mask)
    observed = ~mask
    if name == "gp_matern32":
        np.testing.assert_allclose(
            out[observed],
            strain[observed],
            atol=1e-10,
            err_msg=f"GP baseline changed observed values beyond atol=1e-10",
        )
    else:
        np.testing.assert_array_equal(
            out[observed],
            strain[observed],
            err_msg=f"Baseline '{name}' changed observed (non-masked) values",
        )


# ---------------------------------------------------------------------------
# Zero imputer specific
# ---------------------------------------------------------------------------


def test_zero_fills_masked_with_zero() -> None:
    """Zero imputer must write exactly 0.0 at every masked position."""
    strain, mask = _make_strain_and_mask()
    out = BASELINES["zero"](strain, mask)
    assert np.all(out[mask] == 0.0), "Zero imputer wrote non-zero values at masked positions"


# ---------------------------------------------------------------------------
# Linear imputer specific
# ---------------------------------------------------------------------------


def test_linear_handles_edge_gaps() -> None:
    """Linear imputer must not produce NaN when gaps touch array boundaries."""
    n = 256
    strain = np.random.default_rng(5).standard_normal(n).astype(np.float64)

    # Gap at the very start.
    mask_start = np.zeros(n, dtype=np.bool_)
    mask_start[:30] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_start = BASELINES["linear"](strain, mask_start)
    assert np.all(np.isfinite(out_start)), "Linear imputer produced NaN/inf for start gap"

    # Gap at the very end.
    mask_end = np.zeros(n, dtype=np.bool_)
    mask_end[-30:] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_end = BASELINES["linear"](strain, mask_end)
    assert np.all(np.isfinite(out_end)), "Linear imputer produced NaN/inf for end gap"


# ---------------------------------------------------------------------------
# Cubic spline specific
# ---------------------------------------------------------------------------


def test_cubic_spline_smooth_on_smooth_input() -> None:
    """Cubic spline should reconstruct a sinusoid across a short gap to within 1e-3."""
    n = 256
    t = np.arange(n, dtype=np.float64)
    # Period 100 samples, gap of 8 samples in the middle (well under one period).
    strain = np.sin(2.0 * np.pi * t / 100.0)

    mask = np.zeros(n, dtype=np.bool_)
    mask[124:132] = True  # 8-sample gap, ~1/12 of a period

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = BASELINES["cubic_spline"](strain, mask)

    gap_truth = strain[mask]
    gap_pred = out[mask]
    max_err = np.max(np.abs(gap_pred - gap_truth))
    assert max_err < 1e-3, (
        f"Cubic spline max error on smooth sine = {max_err:.2e}, expected < 1e-3"
    )


# ---------------------------------------------------------------------------
# All baselines: fully observed input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(BASELINES.keys()))
def test_all_baselines_handle_fully_observed(name: str) -> None:
    """With mask all False (no gaps) every baseline must return a copy of input."""
    n = 128
    strain = default_rng(9).standard_normal(n).astype(np.float64)
    mask = np.zeros(n, dtype=np.bool_)
    fn = BASELINES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fn(strain, mask)
    np.testing.assert_array_equal(
        out, strain,
        err_msg=f"Baseline '{name}' changed values when mask is all-False",
    )


# ---------------------------------------------------------------------------
# All baselines: fully masked input (no observed data)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(BASELINES.keys()))
def test_all_baselines_handle_no_observed_falls_back(name: str) -> None:
    """With mask all True (no observed data) baselines must not crash.

    - zero: must return all zeros.
    - linear, cubic_spline, gp_matern32: documented fallback is zeros with a
      UserWarning (see baselines.py).  We accept either all-zeros output or a
      ValueError but never an unhandled exception of any other kind.
    """
    n = 64
    strain = default_rng(11).standard_normal(n).astype(np.float64)
    mask = np.ones(n, dtype=np.bool_)
    fn = BASELINES[name]

    if name == "zero":
        out = fn(strain, mask)
        assert np.all(out == 0.0), "Zero imputer did not return zeros for all-masked input"
        return

    # For other baselines: allow fallback-to-zeros (with warning) or ValueError.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fn(strain, mask)
        # If it didn't raise, it must not have crashed with an unhandled error.
        # The documented fallback is zeros.
        assert np.all(np.isfinite(out)), (
            f"Baseline '{name}' returned non-finite values for all-masked input"
        )
    except ValueError:
        # A documented ValueError is also an acceptable contract.
        pass
