"""Colored Gaussian noise generation from the Robson+19 LISA sensitivity curve.

This module draws time-domain strain noise whose one-sided power spectral density
matches the analytic LISA noise model of Robson, Cornish & Liu (2019), CQG 36,
105011. The PSD is obtained by reusing
``smbhb_inspiral.sensitivity.lisa_sensitivity_hc`` (which returns the
characteristic strain :math:`h_c(f) = \\sqrt{f\\,S_n(f)}`) and inverting to
:math:`S_n(f) = h_c^2(f) / f`. The time series is then synthesised via the
standard inverse-FFT recipe for stationary Gaussian noise.

**Scope caveats**

- The analytic PSD *does not* include the unresolved Galactic-binary confusion
  foreground, which fills in the sensitivity trough around 3 mHz during mission
  phases with the full Galaxy in band (Robson+19 Eq. 14). For a gap-imputation
  study this omission is conservative — it makes the noise slightly whiter than
  real LISA noise, which means imputation is somewhat harder, not easier.
- The noise is stationary and Gaussian. Non-stationary drifts, glitches, and
  non-Gaussian tails are handled by :mod:`lisa_gap_imputer.masks` (as gaps and
  glitch corruption) rather than being baked into the noise draw itself.

References
----------
Robson, T., Cornish, N. J., & Liu, C. (2019).
    The construction and use of LISA sensitivity curves.
    *Classical and Quantum Gravity*, 36, 105011.
    https://doi.org/10.1088/1361-6382/ab1101
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from smbhb_inspiral.sensitivity import lisa_sensitivity_hc

__all__ = [
    "NoiseSegment",
    "lisa_psd_sn",
    "generate_colored_noise",
]


# Minimum frequency at which the Robson+19 analytic fit is treated as physical.
# Below this the OMS noise's 1/f^2 term and the acceleration noise's 1/f tail
# both blow up; the fit is not calibrated there. We apply a hard high-pass by
# zeroing Fourier bins with f < _F_MIN_HZ when synthesising noise.
_F_MIN_HZ: float = 1.0e-5

# Maximum frequency above which the analytic fit loses physical meaning (the
# transfer-function approximation breaks down). The Nyquist frequency of a
# sensibly sampled LISA segment (fs <= 1 Hz) lies comfortably below this limit.
_F_MAX_HZ: float = 1.0  # Hz


@dataclass(frozen=True)
class NoiseSegment:
    """A single realization of colored Gaussian noise on a uniform time grid.

    Attributes
    ----------
    strain : npt.NDArray[np.float64]
        Strain time series, shape ``(n_samples,)``, dimensionless.
    fs_hz : float
        Sampling frequency in Hz.
    f_min_hz : float
        High-pass frequency actually applied when synthesising the segment.
    """

    strain: npt.NDArray[np.float64]
    fs_hz: float
    f_min_hz: float

    @property
    def n_samples(self) -> int:
        return int(self.strain.size)

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs_hz


def lisa_psd_sn(f_hz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """One-sided LISA strain noise PSD :math:`S_n(f)` in units of Hz :sup:`-1`.

    Obtained from the sky-averaged characteristic strain returned by
    :func:`smbhb_inspiral.sensitivity.lisa_sensitivity_hc` via
    :math:`S_n(f) = h_c^2(f) / f`. The value at ``f = 0`` is undefined and is
    returned as ``np.inf`` so that callers doing inverse-variance weighting
    naturally zero the DC bin.

    Parameters
    ----------
    f_hz : npt.NDArray[np.float64]
        Frequencies in Hz. May include ``f = 0`` (DC) which is mapped to
        ``+inf``.

    Returns
    -------
    npt.NDArray[np.float64]
        :math:`S_n(f)` in Hz :sup:`-1`. Same shape as *f_hz*.
    """
    f: npt.NDArray[np.float64] = np.asarray(f_hz, dtype=np.float64)
    s_n: npt.NDArray[np.float64] = np.full_like(f, fill_value=np.inf)

    positive: npt.NDArray[np.bool_] = f > 0.0
    if np.any(positive):
        h_c = lisa_sensitivity_hc(f[positive])
        s_n[positive] = h_c**2 / f[positive]
    return s_n


def generate_colored_noise(
    n_samples: int,
    fs_hz: float,
    rng: np.random.Generator,
    f_min_hz: float | None = None,
) -> NoiseSegment:
    """Synthesize one realization of stationary LISA-colored Gaussian noise.

    Uses the inverse-FFT recipe: draw i.i.d. complex Gaussian amplitudes in the
    positive-frequency bins, scale by :math:`\\sqrt{S_n(f) f_s N / 4}` so the
    resulting one-sided periodogram has expectation :math:`S_n(f)`, enforce
    Hermitian symmetry, then inverse-FFT to the time domain. Bins below
    ``f_min_hz`` (the high-pass cutoff) are zeroed — this is both a physical
    honesty about the analytic fit's validity range and a numerical necessity
    because the PSD diverges as ``f → 0``.

    Parameters
    ----------
    n_samples : int
        Length of the output time series. Any positive integer is accepted; the
        underlying ``np.fft.rfft`` / ``np.fft.irfft`` handle both even and odd
        lengths.
    fs_hz : float
        Sampling frequency in Hz. Must be positive. For physical LISA-like
        cadences use something in the range ``0.1`` to ``1.0`` Hz; the
        analytic PSD is only calibrated up to ~1 Hz.
    rng : np.random.Generator
        NumPy random generator. The caller is responsible for seed management —
        segments in the train, val, and test splits must be drawn from
        generators with different master seeds (see ``dataset.py``).
    f_min_hz : float or None, optional
        High-pass cutoff in Hz. Fourier bins below this frequency are zeroed
        before the inverse FFT. Defaults to ``1e-5`` Hz, below which the
        Robson+19 fit is physically unreliable.

    Returns
    -------
    NoiseSegment
        The strain time series together with the cadence and the applied
        high-pass frequency.

    Raises
    ------
    ValueError
        If ``n_samples`` or ``fs_hz`` are non-positive, or if ``f_min_hz`` lies
        outside ``[0, fs_hz / 2)``.

    Notes
    -----
    The normalization convention used here is the one that makes

    .. math::

        \\langle |\\tilde{x}(f_k)|^2 \\rangle = \\frac{f_s N}{2} S_n(f_k),

    where :math:`\\tilde{x}(f_k)` is the complex DFT coefficient. This is the
    same convention as ``scipy.signal.welch`` with ``scaling='density'`` and
    one-sided output, so the Welch-estimated PSD of a generated segment will
    recover :math:`S_n(f)` up to finite-sample variance.

    Memory: the routine allocates a single ``rfft`` spectrum of length
    ``n_samples // 2 + 1`` in addition to the output array.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive; got {n_samples}.")
    if fs_hz <= 0.0:
        raise ValueError(f"fs_hz must be positive; got {fs_hz}.")

    f_min: float = _F_MIN_HZ if f_min_hz is None else float(f_min_hz)
    nyquist: float = 0.5 * fs_hz
    if not (0.0 <= f_min < nyquist):
        raise ValueError(
            f"f_min_hz must lie in [0, fs/2={nyquist:.6g}); got {f_min}."
        )

    # Frequencies of the rfft bins [0, fs/2] inclusive.
    freqs: npt.NDArray[np.float64] = np.fft.rfftfreq(n_samples, d=1.0 / fs_hz)

    # PSD at every bin; DC (f=0) is +inf by construction.
    s_n: npt.NDArray[np.float64] = lisa_psd_sn(freqs)

    # Per-bin amplitude standard deviation. Expectation of |X_k|^2 over the
    # complex Gaussian draw below is fs * N * S_n / 2, matching the one-sided
    # PSD normalization used by scipy.signal.welch density scaling.
    # Scale: draw each of Re(X_k), Im(X_k) ~ N(0, sigma^2) with
    #   sigma^2 = fs * N * S_n(f_k) / 4  →  E[|X_k|^2] = 2 sigma^2 = fs N S_n / 2.
    scale: npt.NDArray[np.float64] = np.sqrt(
        0.25 * fs_hz * n_samples * s_n
    )

    # Complex white noise, shape matching rfft bins.
    real_part: npt.NDArray[np.float64] = rng.standard_normal(freqs.size)
    imag_part: npt.NDArray[np.float64] = rng.standard_normal(freqs.size)
    spectrum: npt.NDArray[np.complex128] = scale * (
        real_part + 1j * imag_part
    )

    # DC bin must be real.
    spectrum[0] = 0.0 + 0.0j

    # For even-length signals the Nyquist bin (last) must also be real, so
    # drop its imaginary part.
    if n_samples % 2 == 0:
        spectrum[-1] = spectrum[-1].real + 0.0j

    # Apply the high-pass cutoff.
    spectrum[freqs < f_min] = 0.0 + 0.0j

    # Also zero any bins beyond the Robson+19 physical validity upper edge. In
    # practice fs <= 1 Hz means this is a no-op, but it keeps the routine
    # defensible if a caller uses a higher cadence.
    spectrum[freqs > _F_MAX_HZ] = 0.0 + 0.0j

    strain: npt.NDArray[np.float64] = np.fft.irfft(spectrum, n=n_samples)

    return NoiseSegment(strain=strain, fs_hz=float(fs_hz), f_min_hz=f_min)
