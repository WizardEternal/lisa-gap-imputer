"""Periodic and stochastic gap-pattern sampling for LISA-like strain segments.

LISA's strain time series will be interrupted by two qualitatively distinct families
of data gaps:

1. **Structured / periodic gaps** — monthly antenna repointing for Earth downlink,
   scheduled maintenance windows, and similar planned interruptions. Within any
   short training segment we model this as *one* rectangular gap whose duration is
   a fixed fraction of the segment length and whose centre is drawn uniformly from
   the interior of the segment. The physical duty-cycle literature (e.g. Amaro-Seoane
   et al. 2017) quotes ~75–82 % uptime; a 5–10 % gap per segment is a conservative
   lower bound on the fractional loss.

2. **Stochastic / Poisson gaps** — micrometeorite impacts on the test masses, laser
   frequency lock losses, and thruster noise transients. These are modelled here as a
   Poisson-distributed *number* of gaps whose *durations* are drawn from a log-normal
   distribution scaled to the segment length so that the typical individual gap is
   0.1–2 % of the segment. The Poisson assumption (exponentially distributed inter-
   arrival times) is a standard first-order approximation for rare-event processes
   with no memory.

**Key design choices:**

- Overlap between gaps is *allowed* and handled by logical-OR of the individual
  rectangular masks. The ``gaps`` tuple in :class:`MaskRealization` records the
  pre-merge specs because downstream evaluation wants to stratify metrics by gap type
  and individual duration, not by merged connected components.
- Gap positions are constrained so every gap lies *entirely* inside the segment.
  This is enforced by construction and asserted explicitly inside :func:`sample_mask`.
- The log-normal duration draw uses a mean that scales linearly with ``n_samples``
  (the ``+ ln(n_samples)`` shift on the log-mean), so the *fractional* duration
  distribution is the same regardless of segment length. The default parameters
  ``mu_log = -5.5``, ``sigma_log = 0.7`` target a median fractional duration of
  ``exp(-5.5) ≈ 0.004`` (0.4 % of segment length) with a 5th–95th percentile range
  of roughly 0.1–2 %, consistent with micrometeorite impact statistics described in
  the LISA mission documentation.

References
----------
Amaro-Seoane, P. et al. (2017). *Laser Interferometer Space Antenna*.
    arXiv:1702.00786.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "GapSpec",
    "MaskRealization",
    "sample_mask",
    "apply_mask",
    "per_gap_durations",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GapSpec:
    """The specification of a single rectangular gap in the time series.

    Parameters follow the Python slice convention: ``start`` is inclusive,
    ``end`` is exclusive. The masked region covers sample indices
    ``[start, end)``, i.e. ``end - start`` samples in total.

    Attributes
    ----------
    start : int
        Index of the first masked sample (inclusive). Must satisfy
        ``0 <= start < end``.
    end : int
        One past the index of the last masked sample (exclusive). Must satisfy
        ``start < end``.
    kind : {"periodic", "stochastic"}
        Physical origin of the gap. ``"periodic"`` denotes a structured
        antenna-repointing or maintenance gap; ``"stochastic"`` denotes a
        transient micrometeorite or instrument-glitch gap.
    """

    start: int
    end: int
    kind: Literal["periodic", "stochastic"]

    @property
    def duration_samples(self) -> int:
        """Number of samples masked by this gap (``end - start``)."""
        return self.end - self.start


@dataclass(frozen=True)
class MaskRealization:
    """A concrete gap-mask realisation for one strain segment.

    Attributes
    ----------
    mask : npt.NDArray[np.bool_]
        Boolean array of shape ``(n_samples,)``. ``True`` at positions that are
        masked (missing / corrupted); ``False`` at clean positions.
    gaps : tuple[GapSpec, ...]
        The individual gap specs that were sampled to compose the mask. Gaps
        are listed *before* any merging — overlapping specs are kept separately
        so downstream evaluation can stratify metrics by gap type and individual
        duration. The final ``mask`` is the logical-OR of all gap intervals.

    Notes
    -----
    ``len(gaps) == 0`` and an all-False ``mask`` arise when
    :func:`sample_mask` is called with ``n_samples = 0`` or when the Poisson
    draw returns zero stochastic gaps *and* ``include_periodic = False``.
    """

    mask: npt.NDArray[np.bool_]
    gaps: tuple[GapSpec, ...]


# ---------------------------------------------------------------------------
# Core sampler
# ---------------------------------------------------------------------------


def sample_mask(
    n_samples: int,
    rng: np.random.Generator,
    *,
    periodic_fraction_range: tuple[float, float] = (0.05, 0.10),
    stochastic_rate_per_1k_samples: float = 4.0,
    stochastic_duration_log_mean: float = -5.5,
    stochastic_duration_log_std: float = 0.7,
    include_periodic: bool = True,
) -> MaskRealization:
    """Sample a composite gap mask for one strain segment.

    Produces one rectangular "antenna repointing" gap (periodic regime) and a
    Poisson-distributed number of short "micrometeorite / instrument-glitch"
    gaps (stochastic regime). The final boolean mask is the logical-OR of all
    sampled gap intervals.

    Parameters
    ----------
    n_samples : int
        Segment length in samples. If ``0``, an empty mask is returned
        immediately.
    rng : np.random.Generator
        Caller-supplied NumPy random generator. Pass generators with distinct
        seeds for the train, validation, and test splits; do *not* share one
        generator across splits, as that would make the test-set masks
        dependent on how many train/val segments were drawn first.
    periodic_fraction_range : tuple[float, float], optional
        ``(f_min, f_max)`` — the duration of the periodic gap is drawn
        uniformly in ``[f_min * n_samples, f_max * n_samples]`` samples.
        Defaults to ``(0.05, 0.10)``, corresponding to a 5–10 % data-loss
        fraction per segment from the antenna-repointing event.
    stochastic_rate_per_1k_samples : float, optional
        Expected number of stochastic gaps per 1000 samples. The Poisson
        parameter is ``rate * (n_samples / 1000)``. Defaults to ``4.0``
        (four gaps per 4096-sample segment on average).
    stochastic_duration_log_mean : float, optional
        Log-mean ``mu`` of the log-normal duration distribution *before* the
        length-scaling shift. The actual log-mean used is
        ``mu + ln(n_samples)`` so that durations scale proportionally with
        segment length and the fractional-duration distribution is independent
        of ``n_samples``. Defaults to ``-5.5``.
    stochastic_duration_log_std : float, optional
        Log-standard-deviation ``sigma`` of the log-normal duration
        distribution. Defaults to ``0.7``.
    include_periodic : bool, optional
        If ``False``, the periodic gap is skipped entirely (useful for
        ablations or when only stochastic gaps are desired). Defaults to
        ``True``.

    Returns
    -------
    MaskRealization
        The boolean mask and the list of constituent gap specs.

    Raises
    ------
    ValueError
        If ``periodic_fraction_range`` fractions are outside ``(0, 1)`` or in
        the wrong order; or if rate / std parameters are negative.

    Notes
    -----
    **Periodic gap construction.**
    The gap duration ``T_p`` is drawn uniformly in
    ``[f_min * n_samples, f_max * n_samples]`` (rounded to the nearest
    integer). The gap centre ``c`` is then drawn uniformly in
    ``[T_p // 2,  n_samples - T_p // 2 - (T_p % 2)]`` so that the half-open
    interval ``[c - T_p//2, c - T_p//2 + T_p)`` lies strictly inside
    ``[0, n_samples)``. The ceiling/floor arithmetic here is the most common
    source of off-by-one errors; the explicit assertions at the end of the
    function serve as regression guards.

    **Stochastic gap log-normal scaling.**
    Given default ``mu_log = -5.5`` and ``sigma_log = 0.7``, the fractional
    duration distribution has:

    - Median fractional duration: ``exp(-5.5) ≈ 0.0041`` (0.41 %)
    - 5th percentile: ``exp(-5.5 - 1.645 * 0.7) ≈ 0.0014`` (0.14 %)
    - 95th percentile: ``exp(-5.5 + 1.645 * 0.7) ≈ 0.012`` (1.2 %)

    For ``n_samples = 4096`` the median absolute duration is ~17 samples, the
    5th percentile is ~6 samples, and the 95th percentile is ~49 samples —
    broadly consistent with micrometeorite impact windows discussed in the LISA
    mission documentation.

    The duration is additionally clipped to ``[1, n_samples // 20]`` to avoid
    stochastic gaps that are either sub-sample or competitive in length with
    the periodic gap.

    **Overlap handling.**
    Gaps are allowed to overlap. The mask is the logical-OR of all individual
    rectangular windows. The ``gaps`` tuple records each gap's original spec
    *before* merging so that evaluation code can stratify reconstruction
    metrics by gap type without needing to re-attribute pixels of merged
    connected components.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if n_samples == 0:
        return MaskRealization(
            mask=np.zeros(0, dtype=np.bool_),
            gaps=(),
        )
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative; got {n_samples}.")
    f_lo, f_hi = periodic_fraction_range
    if not (0.0 < f_lo <= f_hi < 1.0):
        raise ValueError(
            f"periodic_fraction_range must satisfy 0 < f_lo <= f_hi < 1; "
            f"got ({f_lo}, {f_hi})."
        )
    if stochastic_rate_per_1k_samples < 0.0:
        raise ValueError(
            f"stochastic_rate_per_1k_samples must be non-negative; "
            f"got {stochastic_rate_per_1k_samples}."
        )
    if stochastic_duration_log_std < 0.0:
        raise ValueError(
            f"stochastic_duration_log_std must be non-negative; "
            f"got {stochastic_duration_log_std}."
        )

    # ------------------------------------------------------------------
    # Accumulate gap specs
    # ------------------------------------------------------------------
    gap_list: list[GapSpec] = []

    # --- Periodic gap ---------------------------------------------------
    if include_periodic and n_samples >= 2:
        # Duration in samples, drawn uniformly in the fraction range.
        dur_lo: int = max(1, round(f_lo * n_samples))
        dur_hi: int = max(dur_lo, round(f_hi * n_samples))
        # Clip to strictly less than n_samples so there is always at least
        # one clean sample left — a fully-masked segment is not useful.
        dur_hi = min(dur_hi, n_samples - 1)
        p_dur: int = int(rng.integers(dur_lo, dur_hi + 1))

        # Draw the gap centre uniformly so the gap fits entirely inside
        # [0, n_samples). The half-open interval [start, end) with
        #   start = centre - p_dur // 2
        #   end   = start + p_dur
        # requires:
        #   start >= 0  →  centre >= p_dur // 2
        #   end   <= n_samples  →  centre <= n_samples - p_dur + p_dur // 2
        #                                   = n_samples - (p_dur - p_dur // 2)
        #                                   = n_samples - math.ceil(p_dur / 2)
        c_lo: int = p_dur // 2
        c_hi: int = n_samples - math.ceil(p_dur / 2)
        # c_lo <= c_hi is guaranteed when p_dur <= n_samples, which holds
        # because dur_hi <= n_samples - 1 < n_samples.
        centre: int = int(rng.integers(c_lo, c_hi + 1))
        p_start: int = centre - p_dur // 2
        p_end: int = p_start + p_dur
        gap_list.append(GapSpec(start=p_start, end=p_end, kind="periodic"))

    # --- Stochastic gaps -----------------------------------------------
    lam: float = stochastic_rate_per_1k_samples * (n_samples / 1000.0)
    n_stochastic: int = int(rng.poisson(lam))

    if n_stochastic > 0:
        # Log-normal duration draw with length-scaled log-mean.
        # mu_eff = mu_log + ln(n_samples) so that
        #   exp(mu_eff) = n_samples * exp(mu_log)
        # i.e. the median duration is a fixed fraction of n_samples.
        mu_eff: float = stochastic_duration_log_mean + math.log(n_samples)
        log_dur_draws: npt.NDArray[np.float64] = rng.normal(
            loc=mu_eff,
            scale=stochastic_duration_log_std,
            size=n_stochastic,
        )
        max_stoch_dur: int = max(1, n_samples // 20)

        for log_dur in log_dur_draws:
            s_dur: int = int(math.ceil(math.exp(log_dur)))
            # Clip to [1, n_samples // 20].
            s_dur = max(1, min(s_dur, max_stoch_dur))

            # Draw start position so gap fits entirely inside [0, n_samples).
            # start in [0, n_samples - s_dur] → end = start + s_dur <= n_samples.
            s_start: int = int(rng.integers(0, n_samples - s_dur + 1))
            s_end: int = s_start + s_dur
            gap_list.append(GapSpec(start=s_start, end=s_end, kind="stochastic"))

    # ------------------------------------------------------------------
    # Build the boolean mask (logical-OR of all gap intervals)
    # ------------------------------------------------------------------
    mask: npt.NDArray[np.bool_] = np.zeros(n_samples, dtype=np.bool_)
    for gap in gap_list:
        mask[gap.start : gap.end] = True

    # ------------------------------------------------------------------
    # Correctness assertions (not just docstring promises)
    # ------------------------------------------------------------------
    for gap in gap_list:
        assert gap.start >= 0, (
            f"Gap start {gap.start} is negative — off-by-one in centre arithmetic."
        )
        assert gap.end <= n_samples, (
            f"Gap end {gap.end} exceeds n_samples={n_samples} — "
            "gap extends past segment boundary."
        )
        assert gap.start < gap.end, (
            f"Empty gap: start={gap.start} >= end={gap.end}."
        )

    return MaskRealization(mask=mask, gaps=tuple(gap_list))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def apply_mask(
    strain: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    fill_value: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Return a copy of *strain* with masked positions replaced by *fill_value*.

    The input arrays are not mutated. This is a thin convenience wrapper around
    ``np.where`` rather than a sophisticated imputer — it produces the zero-filled
    (or constant-filled) input that the transformer model and the baselines operate
    on.

    Parameters
    ----------
    strain : npt.NDArray[np.float64]
        Clean (or partially clean) strain time series, shape ``(n_samples,)``.
    mask : npt.NDArray[np.bool_]
        Boolean mask of the same shape. ``True`` at positions to be filled.
    fill_value : float, optional
        Scalar fill value written into masked positions. Defaults to ``0.0``
        (zero-fill). The zero-fill baseline is the dumbest imputer and serves
        as the floor for all comparisons.

    Returns
    -------
    npt.NDArray[np.float64]
        A new array equal to *strain* at unmasked positions and *fill_value*
        at masked positions.

    Raises
    ------
    ValueError
        If *strain* and *mask* have different shapes.
    """
    strain = np.asarray(strain, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.bool_)
    if strain.shape != mask.shape:
        raise ValueError(
            f"strain and mask must have the same shape; "
            f"got strain.shape={strain.shape}, mask.shape={mask.shape}."
        )
    masked: npt.NDArray[np.float64] = np.where(mask, fill_value, strain)
    return masked


def per_gap_durations(realization: MaskRealization) -> dict[str, npt.NDArray[np.int_]]:
    """Extract per-gap durations grouped by gap kind.

    Useful for constructing the "reconstruction error vs gap duration"
    diagnostic plots, which need to stratify model metrics by whether a given
    gap was periodic (large, structured) or stochastic (short, random).

    Parameters
    ----------
    realization : MaskRealization
        A mask realisation as returned by :func:`sample_mask`.

    Returns
    -------
    dict[str, npt.NDArray[np.int_]]
        A dictionary with keys ``"periodic"`` and ``"stochastic"``, each
        mapping to a 1-D integer array of individual gap durations in samples.
        Either array may be empty (zero length) if no gaps of that kind were
        sampled.

    Notes
    -----
    Durations are taken from :attr:`GapSpec.duration_samples` (i.e.
    ``end - start``) and therefore reflect the *original* sampled extent of
    each gap before logical-OR merging. Overlapping gaps of the same kind will
    each contribute their own duration entry, which is what you want for
    building the evaluation scatter plots — you are characterising the
    *difficulty* of each gap event, not the post-merge connected component.
    """
    periodic_durs: list[int] = []
    stochastic_durs: list[int] = []
    for gap in realization.gaps:
        if gap.kind == "periodic":
            periodic_durs.append(gap.duration_samples)
        else:
            stochastic_durs.append(gap.duration_samples)
    return {
        "periodic": np.array(periodic_durs, dtype=np.intp),
        "stochastic": np.array(stochastic_durs, dtype=np.intp),
    }


# ---------------------------------------------------------------------------
# Sanity check (not a test suite — just a quick smoke-screen)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    N = 4096
    N_TRIALS = 10_000
    rng_check = np.random.default_rng(seed=0)

    periodic_durs_all: list[int] = []
    stochastic_durs_all: list[int] = []
    mask_fracs: list[float] = []

    for _ in range(N_TRIALS):
        real = sample_mask(N, rng_check)
        durs = per_gap_durations(real)
        periodic_durs_all.extend(durs["periodic"].tolist())
        stochastic_durs_all.extend(durs["stochastic"].tolist())
        mask_fracs.append(real.mask.mean())

    p_arr = np.array(periodic_durs_all)
    s_arr = np.array(stochastic_durs_all)
    frac_arr = np.array(mask_fracs)

    print(f"Sanity check: {N_TRIALS:,} realisations at n_samples={N}", flush=True)
    print(
        f"  Periodic gap duration  — "
        f"mean={p_arr.mean():.1f}, "
        f"median={np.median(p_arr):.1f}, "
        f"[p5, p95]=[{np.percentile(p_arr, 5):.1f}, {np.percentile(p_arr, 95):.1f}] samples"
    )
    print(
        f"    As fraction of N     — "
        f"mean={p_arr.mean()/N:.3f}, "
        f"[p5, p95]=[{np.percentile(p_arr, 5)/N:.3f}, {np.percentile(p_arr, 95)/N:.3f}]"
    )
    print(
        f"  Stochastic gap duration — "
        f"mean={s_arr.mean():.1f}, "
        f"median={np.median(s_arr):.1f}, "
        f"[p5, p95]=[{np.percentile(s_arr, 5):.1f}, {np.percentile(s_arr, 95):.1f}] samples"
    )
    print(
        f"    As fraction of N     — "
        f"mean={s_arr.mean()/N:.4f}, "
        f"[p5, p95]=[{np.percentile(s_arr, 5)/N:.4f}, {np.percentile(s_arr, 95)/N:.4f}]"
    )
    print(
        f"  Stochastic gaps per realization — "
        f"mean={(len(s_arr)/N_TRIALS):.2f} "
        f"(expected {4.0 * N / 1000:.2f})"
    )
    print(
        f"  Mask fraction          — "
        f"mean={frac_arr.mean():.4f}, "
        f"[p5, p95]=[{np.percentile(frac_arr, 5):.4f}, {np.percentile(frac_arr, 95):.4f}]"
    )

    # Quick check that durations are in the intended ballpark
    target_frac_med = math.exp(-5.5)  # ~0.0041
    actual_frac_med = float(np.median(s_arr)) / N
    tol = 0.5  # 50 % relative tolerance — we are checking order-of-magnitude
    if abs(actual_frac_med - target_frac_med) / target_frac_med > tol:
        print(
            f"WARNING: stochastic median fractional duration {actual_frac_med:.4f} "
            f"deviates by >{tol*100:.0f}% from analytic target {target_frac_med:.4f}.",
            file=sys.stderr,
        )
    else:
        print(
            f"  Duration check PASSED: median fractional duration "
            f"{actual_frac_med:.4f} within {tol*100:.0f}% of target {target_frac_med:.4f}."
        )
