from __future__ import annotations

import numpy as np
import pytest
from numpy.random import default_rng

from lisa_gap_imputer.noise import NoiseSegment, generate_colored_noise, lisa_psd_sn


# ---------------------------------------------------------------------------
# lisa_psd_sn
# ---------------------------------------------------------------------------


def test_psd_at_dc_is_inf() -> None:
    """PSD at f=0 (DC) must be +inf by convention."""
    result = lisa_psd_sn(np.array([0.0]))
    assert result[0] == np.inf


def test_psd_positive_on_valid_range() -> None:
    """PSD must be finite and strictly positive over 1e-4 to 1e-1 Hz."""
    freqs = np.logspace(-4, -1, 50)
    sn = lisa_psd_sn(freqs)
    assert np.all(np.isfinite(sn)), "PSD has non-finite values in valid range"
    assert np.all(sn > 0.0), "PSD has non-positive values in valid range"


# ---------------------------------------------------------------------------
# generate_colored_noise — basic properties
# ---------------------------------------------------------------------------


def test_colored_noise_shape() -> None:
    """Returned NoiseSegment must have strain of shape (n_samples,) and correct fs_hz."""
    seg = generate_colored_noise(n_samples=256, fs_hz=0.1, rng=default_rng(0))
    assert isinstance(seg, NoiseSegment)
    assert seg.strain.shape == (256,)
    assert seg.fs_hz == pytest.approx(0.1)


def test_colored_noise_is_real() -> None:
    """Strain must be real-valued float64 — no complex leakage through irfft."""
    seg = generate_colored_noise(n_samples=256, fs_hz=0.1, rng=default_rng(1))
    assert seg.strain.dtype == np.float64
    assert not np.iscomplexobj(seg.strain)


def test_colored_noise_reproducible() -> None:
    """Same seed must produce bit-identical strain arrays."""
    seg_a = generate_colored_noise(n_samples=512, fs_hz=0.1, rng=default_rng(42))
    seg_b = generate_colored_noise(n_samples=512, fs_hz=0.1, rng=default_rng(42))
    np.testing.assert_array_equal(seg_a.strain, seg_b.strain)


def test_colored_noise_different_seeds_differ() -> None:
    """Different seeds must produce different strain arrays."""
    seg_a = generate_colored_noise(n_samples=512, fs_hz=0.1, rng=default_rng(0))
    seg_b = generate_colored_noise(n_samples=512, fs_hz=0.1, rng=default_rng(1))
    assert not np.array_equal(seg_a.strain, seg_b.strain)


# ---------------------------------------------------------------------------
# generate_colored_noise — PSD consistency (reduced to keep < 60 s total)
# ---------------------------------------------------------------------------


def test_colored_noise_psd_matches_theory() -> None:
    """Average periodogram over realisations should match the analytic PSD within 3x.

    Uses n=1024 and 16 realisations (fast) instead of n=4096 and 32 to stay
    within the 60-second total-suite budget.  Factor-of-3 tolerance absorbs
    finite-sample variance and rectangular-window bias.
    """
    n_samples = 1024
    fs_hz = 0.1
    n_real = 16
    rng_psd = default_rng(99)

    # Accumulate one-sided periodograms.
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs_hz)
    psd_sum = np.zeros(freqs.size, dtype=np.float64)
    for _ in range(n_real):
        seg = generate_colored_noise(n_samples, fs_hz, rng_psd)
        spectrum = np.fft.rfft(seg.strain)
        # One-sided periodogram estimate (density scaling matching scipy.signal.welch)
        periodogram = (np.abs(spectrum) ** 2) / (fs_hz * n_samples)
        # Interior bins are counted once; double them for one-sided convention.
        periodogram[1:-1] *= 2.0
        psd_sum += periodogram

    avg_psd = psd_sum / n_real
    theory_psd = lisa_psd_sn(freqs)

    # Restrict comparison to [1e-3, 1e-2] Hz where the analytic fit is well-
    # calibrated and the segment is long enough for decent frequency resolution.
    band = (freqs >= 1e-3) & (freqs <= 1e-2)
    assert np.any(band), "No frequency bins fall in [1e-3, 1e-2] Hz"

    ratio = avg_psd[band] / theory_psd[band]
    assert np.all(ratio < 3.0), (
        f"Average periodogram exceeds 3x theory PSD at some frequencies in [1e-3, 1e-2] Hz. "
        f"Max ratio: {ratio.max():.2f}"
    )
    assert np.all(ratio > 1.0 / 3.0), (
        f"Average periodogram is less than 1/3 of theory PSD at some frequencies. "
        f"Min ratio: {ratio.min():.2f}"
    )


# ---------------------------------------------------------------------------
# generate_colored_noise — input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 0, "fs_hz": 0.1},
        {"n_samples": 256, "fs_hz": -1.0},
        {"n_samples": 256, "fs_hz": 0.1, "f_min_hz": -0.01},
        {"n_samples": 256, "fs_hz": 0.1, "f_min_hz": 0.05},  # f_min_hz == nyquist
    ],
    ids=["n_samples=0", "fs_hz=-1", "f_min_hz=-0.01", "f_min_hz=nyquist"],
)
def test_colored_noise_rejects_bad_args(kwargs: dict) -> None:
    """generate_colored_noise must raise ValueError for invalid arguments."""
    with pytest.raises(ValueError):
        generate_colored_noise(rng=default_rng(0), **kwargs)
