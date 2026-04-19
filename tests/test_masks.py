from __future__ import annotations

import numpy as np
import pytest
from numpy.random import default_rng

from lisa_gap_imputer.masks import (
    MaskRealization,
    apply_mask,
    sample_mask,
)


# ---------------------------------------------------------------------------
# sample_mask — shape and type
# ---------------------------------------------------------------------------


def test_mask_is_bool_and_right_shape() -> None:
    """sample_mask must return a MaskRealization whose .mask is bool, shape (n,)."""
    real = sample_mask(1024, default_rng(0))
    assert isinstance(real, MaskRealization)
    assert real.mask.dtype == bool
    assert real.mask.shape == (1024,)


def test_gaps_in_bounds() -> None:
    """Every GapSpec must satisfy 0 <= start < end <= n_samples."""
    n = 1024
    real = sample_mask(n, default_rng(5))
    for gap in real.gaps:
        assert 0 <= gap.start, f"gap.start={gap.start} is negative"
        assert gap.start < gap.end, f"empty gap: start={gap.start}, end={gap.end}"
        assert gap.end <= n, f"gap.end={gap.end} exceeds n_samples={n}"


def test_gaps_nonoverlapping_in_mask() -> None:
    """Scan the boolean mask for contiguous True-runs; they must be non-overlapping
    by construction (True-runs are the merged connected components of all gaps).
    Verify each run is enclosed within one or more recorded GapSpec intervals."""
    n = 2048
    real = sample_mask(n, default_rng(13))
    mask = real.mask

    # Find contiguous True-runs in the mask.
    runs: list[tuple[int, int]] = []
    in_gap = False
    run_start = 0
    for i, val in enumerate(mask):
        if val and not in_gap:
            run_start = i
            in_gap = True
        elif not val and in_gap:
            runs.append((run_start, i))
            in_gap = False
    if in_gap:
        runs.append((run_start, n))

    # Each run must be covered by at least one recorded GapSpec.
    for run_s, run_e in runs:
        covered = any(
            gap.start <= run_s and gap.end >= run_e for gap in real.gaps
        )
        assert covered, (
            f"Mask run [{run_s}, {run_e}) is not covered by any recorded GapSpec"
        )

    # Runs must be sorted and non-overlapping.
    for (s1, e1), (s2, e2) in zip(runs, runs[1:]):
        assert e1 <= s2, f"Overlapping True-runs in mask: [{s1},{e1}) and [{s2},{e2})"


def test_include_periodic_false_excludes_periodic_kind() -> None:
    """With include_periodic=False all recorded gaps must be of kind 'stochastic'."""
    real = sample_mask(4096, default_rng(3), include_periodic=False)
    for gap in real.gaps:
        assert gap.kind == "stochastic", (
            f"Found gap of kind '{gap.kind}' when include_periodic=False"
        )


def test_stochastic_rate_scales_with_length() -> None:
    """Number of stochastic gaps should grow roughly linearly with n_samples.

    Checks that the mean count ratio (n=4096 / n=512) lies within 2× of the
    theoretical ratio 8 (= 4096/512).  100 realisations per length.
    """
    n_trials = 100
    n_small, n_large = 512, 4096
    theoretical_ratio = n_large / n_small  # 8.0

    counts_small: list[int] = []
    counts_large: list[int] = []
    rng_s = default_rng(7)
    rng_l = default_rng(8)

    for _ in range(n_trials):
        real_s = sample_mask(n_small, rng_s, include_periodic=False)
        counts_small.append(sum(1 for g in real_s.gaps if g.kind == "stochastic"))

    for _ in range(n_trials):
        real_l = sample_mask(n_large, rng_l, include_periodic=False)
        counts_large.append(sum(1 for g in real_l.gaps if g.kind == "stochastic"))

    mean_small = np.mean(counts_small)
    mean_large = np.mean(counts_large)

    # Avoid divide-by-zero if small count is unexpectedly 0.
    if mean_small < 0.5:
        pytest.skip("Mean stochastic count for n=512 was near zero; skipping ratio check.")

    ratio = mean_large / mean_small
    assert ratio > theoretical_ratio / 2.0, (
        f"Stochastic count ratio {ratio:.2f} is less than half of theoretical {theoretical_ratio}"
    )
    assert ratio < theoretical_ratio * 2.0, (
        f"Stochastic count ratio {ratio:.2f} is more than double theoretical {theoretical_ratio}"
    )


def test_mask_reproducible() -> None:
    """Same seed must produce an identical mask array."""
    real_a = sample_mask(1024, default_rng(0))
    real_b = sample_mask(1024, default_rng(0))
    np.testing.assert_array_equal(real_a.mask, real_b.mask)


# ---------------------------------------------------------------------------
# apply_mask
# ---------------------------------------------------------------------------


def test_apply_mask_preserves_observed() -> None:
    """apply_mask must leave non-masked positions exactly unchanged."""
    rng_m = default_rng(2)
    n = 512
    strain = rng_m.standard_normal(n)
    real = sample_mask(n, rng_m)
    result = apply_mask(strain, real.mask)

    unmasked = ~real.mask
    np.testing.assert_array_equal(
        result[unmasked],
        strain[unmasked],
        err_msg="apply_mask changed values at observed (unmasked) positions",
    )
