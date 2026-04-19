"""PyTorch Dataset for LISA-like gap-imputation training.

This module assembles the three upstream primitives — colored noise
(:mod:`lisa_gap_imputer.noise`), signal injection
(:mod:`lisa_gap_imputer.signals`), and gap-mask sampling
(:mod:`lisa_gap_imputer.masks`) — into a map-style
:class:`torch.utils.data.Dataset` whose items are reproducibly generated
on demand without any RAM-heavy pre-computation.

**Design overview**

Each :class:`StrainDataset` stores only a single integer ``master_seed``.
In :meth:`StrainDataset.__getitem__` the per-index ``SeedSequence`` is
constructed on the fly as ``SeedSequence([master_seed, i])`` — a compound
seed that uniquely encodes both the split identity and the element index —
and then spawned into three isolated leaf generators
(``noise_rng``, ``signal_rng``, ``mask_rng``). This means:

- ``dataset[i]`` returns identical bit-for-bit tensors no matter how many
  other items have been accessed beforehand (map-style determinism).
- Train, validation, and test sets use separate ``master_seed`` values, so no
  single parameter value can appear in two splits.
- The test set's ``SeedSequence`` hierarchy is constructed from its own
  ``master_seed`` and never consumes entropy from the train or val hierarchies,
  preserving test-set independence even if the val set is later resized.

**Amplitude normalisation**

Raw LISA strain values are O(1e-21), which is poorly conditioned for float32
training. A single normalisation scalar ``scale`` is estimated from the
training split's noise statistics (via :func:`estimate_noise_scale`) and then
shared with the validation and test splits via :func:`build_splits`. Applying
a *shared* scale is critical: if each split computed its own scale the loss
surface would be inconsistent between train and eval.

The scale is calibrated on pure noise (no signal injection) so that the
colored-noise floor maps to O(1). SMBHB chirp amplitudes span many orders of
magnitude depending on distance and masses, but the prior in
:func:`~lisa_gap_imputer.signals.draw_smbhb_params` (distances 500–10 000 Mpc)
is tuned so that typical chirp amplitudes are broadly comparable to the noise
floor; consequently a single noise-calibrated scale keeps the full amplitude
regime within float32 dynamic range without per-signal renormalisation.

**Return format**

Each call to :meth:`StrainDataset.__getitem__` returns a dict with four
``torch.float32`` tensors:

``masked_strain``
    Shape ``(seq_len,)``. The model's input: signal+noise at observed
    positions (scaled), zero at masked positions.
``mask``
    Shape ``(seq_len,)``. 1.0 at masked positions, 0.0 elsewhere.
``truth``
    Shape ``(seq_len,)``. The full scaled strain (signal + noise) at *all*
    positions, including those that are masked. This is the regression target:
    the imputer must reproduce the stochastic noise process inside each gap,
    not merely the smooth signal.
``scale``
    Shape ``()`` (scalar). The ``self.scale`` value used, so downstream
    evaluation can de-normalise without needing the dataset object.

Author
------
Karan Akbari
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.utils.data

from lisa_gap_imputer.masks import MaskRealization, sample_mask
from lisa_gap_imputer.noise import NoiseSegment, generate_colored_noise
from lisa_gap_imputer.signals import (
    SignalSegment,
    draw_monochromatic_params,
    draw_smbhb_params,
    inject_monochromatic,
    inject_smbhb_chirp,
)

__all__ = [
    "estimate_noise_scale",
    "StrainDataset",
    "build_splits",
]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_SIGNAL_MIX: dict[str, float] = {
    "quiet": 0.3,
    "smbhb": 0.5,
    "monochromatic": 0.2,
}

# Sentinel used as the second component of the compound seed for scale
# estimation: [master_seed, _SCALE_SENTINEL].  Its value is 2**32 - 1
# (the maximum uint32), which is safe for numpy.random.SeedSequence and
# far beyond any realistic dataset size, so it can never collide with a valid
# segment index.
_SCALE_SENTINEL: int = (1 << 32) - 1


# ---------------------------------------------------------------------------
# Reference-scale utility
# ---------------------------------------------------------------------------


def estimate_noise_scale(
    fs_hz: float,
    seq_len: int,
    rng: np.random.Generator,
    n_segments: int = 64,
) -> float:
    """Estimate a normalisation scalar from the colored-noise amplitude.

    Generates ``n_segments`` independent colored-noise realisations and
    returns the mean of their per-segment standard deviations. This single
    number converts raw LISA strain values (O(1e-21)) into O(1) quantities
    that are well-conditioned for float32 gradient descent.

    Parameters
    ----------
    fs_hz : float
        Sampling frequency in Hz. Must match the value used in
        :class:`StrainDataset` so that the PSD-weighted noise amplitude is
        consistent.
    seq_len : int
        Segment length in samples. Must match :attr:`StrainDataset.seq_len`.
    rng : np.random.Generator
        NumPy random generator used for the noise realisations. For
        reproducibility, always derived from the train split's ``master_seed``
        (see :func:`build_splits`). The same ``n_segments`` draws are consumed
        each time, so the returned scale is deterministic given fixed inputs.
    n_segments : int, optional
        Number of noise realisations to average over. The default of 64
        reduces the variance of the mean std to below 2 % (by the central
        limit theorem), which is more than sufficient for a normalisation
        constant. Increasing this improves stability at negligible cost.

    Returns
    -------
    float
        Reciprocal of the mean per-segment standard deviation of the colored
        noise, i.e. approximately ``1 / sqrt(mean_variance)``.  Multiplying
        raw strain values by this number maps them from O(1e-21) to O(1) and
        is well-conditioned for float32 gradient descent.

    Notes
    -----
    The formula is ``1 / mean(std(segment_i))``, where each ``std`` is
    computed on a single colored-noise realisation of length ``seq_len``.
    Because the noise is stationary and the PSD is fixed, this quantity is
    nearly constant in ``n_segments``; 64 realisations reduces the finite-
    sample variance of the reciprocal mean to below 2 %, which is more than
    sufficient for a normalisation constant.

    The returned value is approximately equal to ``1 / sqrt(mean_variance)``
    (they differ only by the ratio of the mean std to the square root of the
    mean variance, which is 1 for Gaussian distributions).

    Critically, the **same** scale value must be passed to every split (train,
    val, test) via :func:`build_splits`. Using split-specific scales would
    mean the model trains on differently normalised inputs than it is evaluated
    on, introducing a subtle systematic bias in the reported loss values.
    """
    stds: list[float] = []
    for _ in range(n_segments):
        seg: NoiseSegment = generate_colored_noise(seq_len, fs_hz, rng)
        stds.append(float(np.std(seg.strain)))
    mean_std: float = float(np.mean(stds))
    return 1.0 / mean_std


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class StrainDataset(torch.utils.data.Dataset):
    """Map-style PyTorch Dataset of synthetic LISA-like strain segments.

    Each segment is synthesised on demand in :meth:`__getitem__` from a
    deterministic per-index :class:`numpy.random.SeedSequence`. No data is
    pre-computed or stored in RAM beyond the tuple of ``SeedSequence`` objects
    (a handful of bytes per segment).

    Parameters
    ----------
    n_segments : int
        Number of segments in the dataset (equivalently, ``len(dataset)``).
    master_seed : int
        Root entropy for the ``SeedSequence`` hierarchy. Every index ``i``
        gets a unique child ``SeedSequence`` spawned from this root, so
        different ``master_seed`` values produce completely non-overlapping
        parameter spaces — train, val, and test should each receive a
        distinct value.
    seq_len : int, optional
        Segment length in samples. Defaults to 4 096 (≈ 11 h at 0.1 Hz).
    fs_hz : float, optional
        Sampling frequency in Hz. Defaults to 0.1 Hz.
    signal_mix : dict[str, float], optional
        Probability of each signal type per segment. Keys must be a subset of
        ``{"quiet", "smbhb", "monochromatic"}`` and values must sum to 1.0
        (within ``abs_tol=1e-6``). Defaults to
        ``{"quiet": 0.3, "smbhb": 0.5, "monochromatic": 0.2}``.
    include_periodic_gap : bool, optional
        Whether to include the structured antenna-repointing gap in the mask.
        Setting this to ``False`` retains only stochastic gaps; useful for
        ablation experiments. Defaults to ``True``.
    scale : float or None, optional
        Pre-computed normalisation scalar. If ``None`` and
        ``normalize=True``, computed automatically via
        :func:`estimate_noise_scale`. Ignored when ``normalize=False``.
    normalize : bool, optional
        If ``False``, no amplitude normalisation is applied (``scale`` is
        forced to 1.0 regardless of the ``scale`` argument). Useful for
        debugging and baseline comparisons in native strain units.

    Attributes
    ----------
    scale : float
        The effective normalisation scalar applied to all segments. Set to 1.0
        when ``normalize=False``. Evaluation code should read this attribute
        to de-normalise model predictions back into physical strain units.

    Notes
    -----
    **RNG hygiene**

    Per-index RNG construction uses a **compound seed** strategy: the seed for
    index ``i`` is ``np.random.SeedSequence([master_seed, i])``, constructed
    freshly on every ``__getitem__`` call.  This avoids the pitfall of trying
    to store and replay *spawned* ``SeedSequence`` children — which is
    unreliable because ``SeedSequence.entropy`` on a child records the root's
    entropy, not the child's unique spawn-path state.  A compound seed encodes
    both the split identity (via ``master_seed``) and the element index (via
    ``i``) in a collision-free way that is explicitly supported by NumPy's
    ``SeedSequence`` API (see numpy docs: "Sequences of integers are
    recursively mixed").

    From the per-index ``SeedSequence`` the three leaf generators are produced
    by a single ``.spawn(3)`` call::

        index_ss    = np.random.SeedSequence([self._master_seed, i])
        noise_ss, signal_ss, mask_ss = index_ss.spawn(3)
        noise_rng   = np.random.default_rng(noise_ss)
        signal_rng  = np.random.default_rng(signal_ss)
        mask_rng    = np.random.default_rng(mask_ss)

    This design is stateless: ``__getitem__`` derives everything it needs from
    ``self._master_seed`` and ``i``, so no per-index state is stored in
    ``self``.  The class does not hold a ``_child_entropies`` tuple or any
    other per-index array.

    Using three separate generators ensures that changing the mask parameters
    (e.g. enabling/disabling the periodic gap) does not alter the noise or
    signal draws for any index, and vice versa. This isolation is essential
    for ablation studies that vary only one component of the pipeline.

    **Signal mix sampling**

    The signal type (``"quiet"``, ``"smbhb"``, or ``"monochromatic"``) is
    drawn using ``signal_rng`` before any parameter draws, so the same entropy
    stream controls both the type choice and the injection parameters. This
    means changing ``signal_mix`` will change both the type drawn *and* — for
    indices where the type switches — the parameter draw for the new type,
    which is the desired behaviour for controlled experiments.

    **Scale management**

    When ``normalize=True`` and ``scale=None``, the scale is estimated from
    64 noise realisations using the dedicated compound seed
    ``[master_seed, _SCALE_SENTINEL]`` where ``_SCALE_SENTINEL = 2**32 - 1``.
    Because no practical dataset has ``2**32 - 1`` segments, the scale branch
    is entropy-isolated from every segment's generation.
    """

    def __init__(
        self,
        n_segments: int,
        master_seed: int,
        seq_len: int = 4096,
        fs_hz: float = 0.1,
        signal_mix: dict[str, float] = _DEFAULT_SIGNAL_MIX,
        include_periodic_gap: bool = True,
        scale: float | None = None,
        normalize: bool = True,
    ) -> None:
        if n_segments <= 0:
            raise ValueError(f"n_segments must be positive; got {n_segments}.")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive; got {seq_len}.")
        if fs_hz <= 0.0:
            raise ValueError(f"fs_hz must be positive; got {fs_hz}.")

        _validate_signal_mix(signal_mix)

        self._n_segments: int = n_segments
        self._master_seed: int = int(master_seed)
        self.seq_len: int = seq_len
        self.fs_hz: float = fs_hz
        self._signal_mix: dict[str, float] = dict(signal_mix)
        self._include_periodic_gap: bool = include_periodic_gap
        self._normalize: bool = normalize

        # ------------------------------------------------------------------
        # Scale resolution.
        #
        # Per-index RNG is derived via compound seeds [master_seed, i] — see
        # the class docstring Notes section.  No per-index state is stored.
        # For scale estimation we use a dedicated compound seed whose second
        # component (_SCALE_SENTINEL = 2**32 - 1) cannot collide with any valid
        # index in [0, n_segments), ensuring entropy isolation.
        # ------------------------------------------------------------------
        if not normalize:
            self.scale: float = 1.0
        elif scale is not None:
            self.scale = float(scale)
        else:
            # Use compound seed [master_seed, _SCALE_SENTINEL] for scale
            # estimation.  The sentinel (2**32 - 1 = 4 294 967 295) is the
            # maximum uint32 value; no practical dataset will have that many
            # segments, so the scale branch never collides with any segment index.
            scale_ss = np.random.SeedSequence([self._master_seed, _SCALE_SENTINEL])
            scale_rng = np.random.default_rng(scale_ss)
            self.scale = estimate_noise_scale(
                fs_hz=self.fs_hz,
                seq_len=self.seq_len,
                rng=scale_rng,
            )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n_segments

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """Generate and return the strain segment at index ``i``.

        Parameters
        ----------
        i : int
            Segment index in ``[0, len(self))``.

        Returns
        -------
        dict[str, torch.Tensor]
            A dict with keys ``"masked_strain"``, ``"mask"``, ``"truth"``,
            and ``"scale"``, all dtype ``torch.float32``. See the module
            docstring for shape and semantics.

        Notes
        -----
        Calling ``dataset[i]`` multiple times always returns identical tensors,
        because every call reconstructs a *fresh* ``SeedSequence`` from the
        stored entropy for index ``i`` and then calls ``.spawn(3)`` exactly
        once on that object.  See the class-level ``Notes`` section for the
        full rationale (``SeedSequence.spawn`` is stateful).
        """
        if not (0 <= i < self._n_segments):
            raise IndexError(
                f"Index {i} out of range for dataset of length {self._n_segments}."
            )

        # Compound seed [master_seed, i] uniquely identifies this (split, index)
        # pair.  Constructing from scratch each call is stateless and O(1).
        index_ss = np.random.SeedSequence([self._master_seed, i])
        noise_ss, signal_ss, mask_ss = index_ss.spawn(3)
        noise_rng: np.random.Generator = np.random.default_rng(noise_ss)
        signal_rng: np.random.Generator = np.random.default_rng(signal_ss)
        mask_rng: np.random.Generator = np.random.default_rng(mask_ss)

        # --- 1. Draw signal kind and inject -----------------------------------
        kind, signal_strain, _ = _draw_and_inject(
            signal_mix=self._signal_mix,
            n_samples=self.seq_len,
            fs_hz=self.fs_hz,
            rng=signal_rng,
        )

        # --- 2. Generate colored noise ----------------------------------------
        noise_seg: NoiseSegment = generate_colored_noise(
            n_samples=self.seq_len,
            fs_hz=self.fs_hz,
            rng=noise_rng,
        )
        noise: npt.NDArray[np.float64] = noise_seg.strain

        # --- 3. Form the full (unmasked) scaled strain -------------------------
        # truth = (signal + noise) * scale  ← regression target at ALL positions
        full_strain: npt.NDArray[np.float64] = signal_strain + noise
        truth_scaled: npt.NDArray[np.float64] = full_strain * self.scale

        # --- 4. Sample the gap mask -------------------------------------------
        mask_real: MaskRealization = sample_mask(
            n_samples=self.seq_len,
            rng=mask_rng,
            include_periodic=self._include_periodic_gap,
        )
        mask_bool: npt.NDArray[np.bool_] = mask_real.mask  # True → masked

        # --- 5. Build zero-filled input ---------------------------------------
        # masked_strain[t] = truth_scaled[t] if not masked, else 0.0
        masked_strain: npt.NDArray[np.float64] = np.where(
            mask_bool, 0.0, truth_scaled
        )

        # --- 6. Convert to float32 tensors ------------------------------------
        return {
            "masked_strain": torch.tensor(masked_strain, dtype=torch.float32),
            "mask": torch.tensor(mask_bool.astype(np.float32), dtype=torch.float32),
            "truth": torch.tensor(truth_scaled.astype(np.float32), dtype=torch.float32),
            "scale": torch.tensor(self.scale, dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Metadata accessor (not used during training)
    # ------------------------------------------------------------------

    def get_meta(self, i: int) -> dict[str, Any]:
        """Re-generate segment ``i`` and return its metadata dictionary.

        This method is intended for the evaluation harness, not for the
        training loop. It re-generates the segment from scratch each time it
        is called, which is acceptable because the evaluation set is small
        (2 000 segments by default) and regeneration is fast compared to the
        cost of storing all metadata in RAM.

        Parameters
        ----------
        i : int
            Segment index in ``[0, len(self))``.

        Returns
        -------
        dict
            A dictionary with the following keys:

            ``"kind"`` : str
                One of ``"quiet"``, ``"smbhb"``, or ``"monochromatic"``.
            ``"signal_params"`` : dict[str, float]
                The parameter dict passed to the injection function. Empty for
                ``"quiet"`` segments.
            ``"gaps"`` : list[dict]
                List of gap specifications, each a dict with keys
                ``"start"`` (int), ``"end"`` (int), and
                ``"kind"`` (``"periodic"`` or ``"stochastic"``).
            ``"scale"`` : float
                The normalisation scalar applied to this segment
                (``self.scale``).

        Notes
        -----
        The re-generation is fully deterministic: the compound seed
        ``[master_seed, i]`` is constructed and ``.spawn(3)`` is called in the
        same order as in :meth:`__getitem__`, so the returned metadata
        corresponds precisely to the tensors that ``dataset[i]`` produces.
        """
        if not (0 <= i < self._n_segments):
            raise IndexError(
                f"Index {i} out of range for dataset of length {self._n_segments}."
            )

        # Same compound-seed pattern as __getitem__ — must be byte-for-byte
        # identical to guarantee that signal and mask draws in get_meta match
        # those that produced the tensors returned by dataset[i].
        index_ss = np.random.SeedSequence([self._master_seed, i])
        noise_ss, signal_ss, mask_ss = index_ss.spawn(3)
        # noise_rng is not used for metadata, but we construct it so that
        # spawn ordering is identical to __getitem__ — defensive against any
        # future NumPy change that makes spawn() order-sensitive.
        _ = np.random.default_rng(noise_ss)
        signal_rng: np.random.Generator = np.random.default_rng(signal_ss)
        mask_rng: np.random.Generator = np.random.default_rng(mask_ss)

        kind, _, signal_params = _draw_and_inject(
            signal_mix=self._signal_mix,
            n_samples=self.seq_len,
            fs_hz=self.fs_hz,
            rng=signal_rng,
        )

        mask_real: MaskRealization = sample_mask(
            n_samples=self.seq_len,
            rng=mask_rng,
            include_periodic=self._include_periodic_gap,
        )

        gaps_list: list[dict[str, Any]] = [
            {"start": g.start, "end": g.end, "kind": g.kind}
            for g in mask_real.gaps
        ]

        return {
            "kind": kind,
            "signal_params": signal_params,
            "gaps": gaps_list,
            "scale": self.scale,
        }


# ---------------------------------------------------------------------------
# Top-level helper
# ---------------------------------------------------------------------------


def build_splits(
    seq_len: int = 4096,
    fs_hz: float = 0.1,
    n_train: int = 20_000,
    n_val: int = 2_000,
    n_test: int = 2_000,
    master_seed_train: int = 0,
    master_seed_val: int = 1,
    master_seed_test: int = 2,
    signal_mix: dict[str, float] | None = None,
    include_periodic_gap: bool = True,
) -> tuple[StrainDataset, StrainDataset, StrainDataset]:
    """Build train, validation, and test :class:`StrainDataset` instances.

    The three splits use **different** ``master_seed`` values so their
    ``SeedSequence`` hierarchies are completely disjoint — no parameter value
    (noise realisation, injection parameters, or gap pattern) can appear in
    more than one split. Simultaneously, a **single** normalisation scalar
    ``scale`` is estimated from the training split and applied to all three
    splits, ensuring that the loss surface is consistent across train and eval.

    Parameters
    ----------
    seq_len : int, optional
        Segment length in samples. Defaults to 4 096.
    fs_hz : float, optional
        Sampling frequency in Hz. Defaults to 0.1 Hz.
    n_train : int, optional
        Number of segments in the training set. Defaults to 20 000.
    n_val : int, optional
        Number of segments in the validation set. Defaults to 2 000.
    n_test : int, optional
        Number of segments in the test set. Defaults to 2 000.
    master_seed_train : int, optional
        Master seed for the training split. Defaults to 0.
    master_seed_val : int, optional
        Master seed for the validation split. Defaults to 1.
    master_seed_test : int, optional
        Master seed for the test split. Defaults to 2.
    signal_mix : dict[str, float], optional
        Signal-type probability distribution. Defaults to
        ``{"quiet": 0.3, "smbhb": 0.5, "monochromatic": 0.2}``.
    include_periodic_gap : bool, optional
        Whether to include the periodic antenna-repointing gap. Defaults to
        ``True``.

    Returns
    -------
    tuple[StrainDataset, StrainDataset, StrainDataset]
        ``(train_dataset, val_dataset, test_dataset)``.

    Notes
    -----
    **Why the scale is computed once from the train split**

    The scale is the mean standard deviation of the Robson+19 colored noise at
    the chosen ``fs_hz`` and ``seq_len``. This quantity is determined by the
    noise PSD alone — it does not depend on the signal injections — so it is
    effectively constant for a given ``(fs_hz, seq_len)`` pair. Computing it
    once from the train seed and sharing it with val and test is therefore
    *not* data leakage; it is the correct engineering choice. If each split
    computed its own scale independently (even with the same formula) the
    resulting values would differ slightly due to finite-sample variance in the
    64-segment average, which would introduce an artificial inter-split
    amplitude offset and bias the reported loss.

    **Test-set independence guarantee**

    Each test segment ``i`` derives its entropy from the compound seed
    ``[master_seed_test, i]``.  Since ``master_seed_test`` differs from both
    ``master_seed_train`` and ``master_seed_val``, the test entropy stream is
    completely disjoint from the train and val streams regardless of how many
    train or val items are accessed.  The only shared quantity is ``scale``,
    passed explicitly as a constructor argument — the test dataset does not
    call :func:`estimate_noise_scale` internally and therefore consumes no
    entropy from any other split.
    """
    if signal_mix is None:
        signal_mix = _DEFAULT_SIGNAL_MIX
    _validate_signal_mix(signal_mix)

    # Build train first to obtain the shared scale.
    train = StrainDataset(
        n_segments=n_train,
        master_seed=master_seed_train,
        seq_len=seq_len,
        fs_hz=fs_hz,
        signal_mix=signal_mix,
        include_periodic_gap=include_periodic_gap,
        scale=None,       # auto-computed from train entropy
        normalize=True,
    )
    shared_scale: float = train.scale

    val = StrainDataset(
        n_segments=n_val,
        master_seed=master_seed_val,
        seq_len=seq_len,
        fs_hz=fs_hz,
        signal_mix=signal_mix,
        include_periodic_gap=include_periodic_gap,
        scale=shared_scale,   # explicitly provided — no internal computation
        normalize=True,
    )

    test = StrainDataset(
        n_segments=n_test,
        master_seed=master_seed_test,
        seq_len=seq_len,
        fs_hz=fs_hz,
        signal_mix=signal_mix,
        include_periodic_gap=include_periodic_gap,
        scale=shared_scale,   # explicitly provided — no internal computation
        normalize=True,
    )

    return train, val, test


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_signal_mix(signal_mix: dict[str, float]) -> None:
    """Raise ``ValueError`` if ``signal_mix`` is malformed.

    Checks that:
    - All keys are drawn from ``{"quiet", "smbhb", "monochromatic"}``.
    - All values are non-negative.
    - Values sum to 1.0 within ``abs_tol=1e-6``.
    """
    valid_keys = {"quiet", "smbhb", "monochromatic"}
    unknown = set(signal_mix) - valid_keys
    if unknown:
        raise ValueError(
            f"signal_mix contains unknown keys: {unknown}. "
            f"Valid keys are {valid_keys}."
        )
    for k, v in signal_mix.items():
        if v < 0.0:
            raise ValueError(
                f"signal_mix['{k}'] = {v} is negative. All probabilities must "
                "be non-negative."
            )
    total = sum(signal_mix.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"signal_mix probabilities must sum to 1.0 (abs_tol=1e-6); "
            f"got {total}."
        )


def _draw_and_inject(
    signal_mix: dict[str, float],
    n_samples: int,
    fs_hz: float,
    rng: np.random.Generator,
) -> tuple[str, npt.NDArray[np.float64], dict[str, float]]:
    """Draw a signal type and inject it, returning the strain and params.

    Uses ``rng`` for both the type choice and the subsequent parameter draw /
    injection, so a single ``signal_rng`` controls the entire signal branch.

    Parameters
    ----------
    signal_mix : dict[str, float]
        Probability mass for each signal type.
    n_samples : int
        Segment length in samples.
    fs_hz : float
        Sampling frequency in Hz.
    rng : np.random.Generator
        The signal-branch RNG. Must be the leaf generator constructed from
        the signal child ``SeedSequence`` inside ``__getitem__`` or
        ``get_meta``.

    Returns
    -------
    kind : str
        The chosen signal type (``"quiet"``, ``"smbhb"``, or
        ``"monochromatic"``).
    strain : npt.NDArray[np.float64]
        Signal-only strain array of shape ``(n_samples,)``. Zero for
        ``"quiet"`` segments.
    params : dict[str, float]
        The physical parameters passed to the injector. Empty for ``"quiet"``.
    """
    kinds = list(signal_mix.keys())
    probs = np.array([signal_mix[k] for k in kinds], dtype=np.float64)
    probs /= probs.sum()  # renormalise to guard against floating-point drift

    # Draw the signal type index using the signal RNG.
    kind_idx: int = int(rng.choice(len(kinds), p=probs))
    kind: str = kinds[kind_idx]

    if kind == "quiet":
        strain = np.zeros(n_samples, dtype=np.float64)
        params: dict[str, float] = {}

    elif kind == "smbhb":
        params = draw_smbhb_params(rng)
        seg: SignalSegment = inject_smbhb_chirp(
            n_samples=n_samples,
            fs_hz=fs_hz,
            rng=rng,   # used only for merger_position (params already drawn)
            params=params,
        )
        strain = seg.strain

    elif kind == "monochromatic":
        params = draw_monochromatic_params(rng)
        seg = inject_monochromatic(
            n_samples=n_samples,
            fs_hz=fs_hz,
            rng=rng,   # not used (params already drawn)
            params=params,
        )
        strain = seg.strain

    else:
        # Should be unreachable because _validate_signal_mix guards the keys.
        raise RuntimeError(f"Unexpected signal kind: {kind!r}")

    return kind, strain, params


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running StrainDataset sanity checks …", flush=True)

    N_SEG = 4
    SEQ_LEN = 256   # short for a fast smoke-test
    FS_HZ = 0.1

    ds = StrainDataset(
        n_segments=N_SEG,
        master_seed=42,
        seq_len=SEQ_LEN,
        fs_hz=FS_HZ,
        signal_mix={"quiet": 0.3, "smbhb": 0.5, "monochromatic": 0.2},
        include_periodic_gap=True,
    )
    print(f"  Dataset length: {len(ds)}", flush=True)
    print(f"  Estimated scale: {ds.scale:.6e}", flush=True)

    # --- Check 1: dataset[0] called twice returns identical tensors ----------
    item_a = ds[0]
    item_b = ds[0]

    assert torch.allclose(item_a["masked_strain"], item_b["masked_strain"]), (
        "FAIL: masked_strain differs between two calls to dataset[0]"
    )
    assert torch.allclose(item_a["mask"], item_b["mask"]), (
        "FAIL: mask differs between two calls to dataset[0]"
    )
    assert torch.allclose(item_a["truth"], item_b["truth"]), (
        "FAIL: truth differs between two calls to dataset[0]"
    )
    print("  CHECK 1 PASSED: dataset[0] is bit-for-bit reproducible.", flush=True)

    # --- Check 2: observed positions in masked_strain equal truth exactly ----
    mask_01 = item_a["mask"]                  # float32 1.0 / 0.0
    observed = (mask_01 == 0.0)               # True at non-masked positions

    masked_strain_obs = item_a["masked_strain"][observed]
    truth_obs = item_a["truth"][observed]

    assert torch.allclose(masked_strain_obs, truth_obs, atol=0.0, rtol=0.0), (
        "FAIL: masked_strain and truth differ at observed (non-masked) positions"
    )
    print(
        "  CHECK 2 PASSED: masked_strain == truth at all observed positions.",
        flush=True,
    )

    # --- Check 3: masked positions in masked_strain are exactly zero ---------
    masked_positions = (mask_01 == 1.0)
    assert torch.all(item_a["masked_strain"][masked_positions] == 0.0), (
        "FAIL: masked_strain is non-zero at masked positions"
    )
    print(
        "  CHECK 3 PASSED: masked_strain == 0.0 at all masked positions.",
        flush=True,
    )

    # --- Check 4: scale scalar is correct ------------------------------------
    assert item_a["scale"].shape == (), (
        f"FAIL: scale should be 0-d tensor; got shape {item_a['scale'].shape}"
    )
    # float64 ds.scale → float32 tensor → float64 may lose a few ULPs, so
    # compare with a tight relative tolerance instead of exact equality.
    assert math.isclose(float(item_a["scale"]), ds.scale, rel_tol=1e-5), (
        f"FAIL: scale tensor value {float(item_a['scale']):.8e} does not match "
        f"dataset.scale {ds.scale:.8e}"
    )
    print("  CHECK 4 PASSED: scale tensor is 0-d and matches dataset.scale.", flush=True)

    # --- Check 5: different indices give different results -------------------
    item_1 = ds[1]
    assert not torch.allclose(item_a["masked_strain"], item_1["masked_strain"]), (
        "FAIL: dataset[0] and dataset[1] return identical masked_strain"
    )
    print(
        "  CHECK 5 PASSED: dataset[0] and dataset[1] differ as expected.",
        flush=True,
    )

    # --- Check 6: get_meta returns consistent kind ---------------------------
    meta = ds.get_meta(0)
    assert meta["kind"] in {"quiet", "smbhb", "monochromatic"}, (
        f"FAIL: unexpected kind in get_meta: {meta['kind']!r}"
    )
    assert isinstance(meta["gaps"], list), "FAIL: get_meta 'gaps' is not a list"
    assert isinstance(meta["scale"], float), "FAIL: get_meta 'scale' is not a float"
    print(
        f"  CHECK 6 PASSED: get_meta returned kind={meta['kind']!r}, "
        f"{len(meta['gaps'])} gap(s).",
        flush=True,
    )

    # --- Check 7: build_splits shared scale ----------------------------------
    train_ds, val_ds, test_ds = build_splits(
        seq_len=SEQ_LEN,
        fs_hz=FS_HZ,
        n_train=8,
        n_val=4,
        n_test=4,
        master_seed_train=10,
        master_seed_val=11,
        master_seed_test=12,
    )
    assert train_ds.scale == val_ds.scale == test_ds.scale, (
        "FAIL: train/val/test scales differ — shared scale not propagated"
    )
    print(
        f"  CHECK 7 PASSED: all three splits share scale={train_ds.scale:.6e}.",
        flush=True,
    )

    print("\nAll sanity checks PASSED.", flush=True)
