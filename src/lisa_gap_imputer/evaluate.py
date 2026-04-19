"""Evaluation harness for the LISA gap-imputation transformer vs. classical baselines.

This module loads a trained :class:`~lisa_gap_imputer.model.GapImputer` checkpoint
and a freshly generated test split, then runs every registered method — the
transformer model and all four classical baselines — across every test segment. The
results are serialised as a single :mod:`pickle` file whose schema is described
below.  The companion :mod:`lisa_gap_imputer.plotting` module reads that file
directly.

Output schema
-------------
The pickled object is a plain :class:`dict` with the following top-level keys:

``"methods"``
    :class:`list` of :class:`str` — the method names in evaluation order
    (``["model", "zero", "linear", "cubic_spline", "gp_matern32"]`` by default).

``"n_segments"``
    :class:`int` — number of test segments evaluated.

``"seq_len"``
    :class:`int` — segment length in samples.

``"fs_hz"``
    :class:`float` — sampling frequency in Hz.

``"scale"``
    :class:`float` — the normalisation scalar used to map raw strain values to the
    training domain.  All per-segment MSE / MAE values in this file are in *scaled*
    units; divide by ``scale`` to recover physical units.

``"per_segment"``
    :class:`dict` mapping *method name* to a :class:`dict` of 1-D arrays (one
    entry per test segment):

    - ``"mse_masked"``  — MSE on gap positions only, in scaled units.
    - ``"mae_masked"``  — MAE on gap positions only, in scaled units.
    - ``"mse_observed"`` — MSE on observed positions (should be ~0 for baselines
      that preserve the observed data exactly; small but non-zero for the model).
    - ``"kind"``           — signal type string: ``"quiet"``, ``"smbhb"``, or
      ``"monochromatic"``.
    - ``"n_gaps"``         — number of gaps in the mask.
    - ``"total_gap_samples"`` — total number of masked samples.
    - ``"longest_gap_samples"`` — length of the longest single gap.
    - ``"has_periodic_gap"`` — whether any gap has ``kind == "periodic"``.

``"per_gap"``
    :class:`dict` mapping *method name* to arrays flattened across all gaps in all
    segments:

    - ``"segment_idx"``    — index of the parent segment.
    - ``"gap_kind"``       — ``"periodic"`` or ``"stochastic"``.
    - ``"duration_samples"`` — gap length in samples.
    - ``"mse"``            — MSE on this gap's samples, in scaled units.
    - ``"mae"``            — MAE on this gap's samples, in scaled units.

``"psd_residual"``
    :class:`dict` mapping *method name* to:

    - ``"freqs_hz"`` — frequency axis of the Welch estimate, shape ``(K,)``.
    - ``"mean_power"`` — Welch PSD of (truth - recon), averaged over test
      segments, in *physical* (unscaled) strain units squared per Hz.
      Computed as ``mean_power = mean( welch(residual_physical) )`` where
      ``residual_physical = (truth - recon) / scale``.

``"snr"``
    :class:`dict` mapping *method name* to arrays containing only SMBHB segments:

    - ``"segment_idx"`` — original test-set index.
    - ``"snr_truth"``   — matched-filter SNR of the truth waveform against itself.
    - ``"snr_recon"``   — matched-filter SNR of the truth template against the
      reconstructed strain.
    - ``"snr_masked"``  — matched-filter SNR of the truth template against the
      zero-filled masked input (floor reference).

Matched-filter convention
-------------------------
The matched-filter inner product used here is the standard frequency-domain
one-sided form::

    <a, b> = 4 Re integral_0^{f_Nyquist} a_tilde(f) b_tilde*(f) / S_n(f) df

where ``a_tilde``, ``b_tilde`` are the one-sided FFTs obtained via
:func:`numpy.fft.rfft` and ``S_n(f)`` is the LISA one-sided noise PSD from
:func:`~lisa_gap_imputer.noise.lisa_psd_sn`. DC and any bins where
``S_n(f) = inf`` are given weight zero. All time series are divided by
``scale`` before the FFT so the inner product is in physical strain units.

The SNR of data ``d`` with template ``s`` is ``<d, s> / sqrt(<s, s>)``.

Why truth-as-template rather than a regenerated clean signal
------------------------------------------------------------
The SMBHB merger position is drawn stochastically from ``rng`` *inside*
:func:`~lisa_gap_imputer.signals.inject_smbhb_chirp` when ``merger_position``
is ``None``. The dataset's :meth:`~lisa_gap_imputer.dataset.StrainDataset.get_meta`
method returns the physical injection parameters (``signal_params``) but not
the merger position that was drawn. Reproducing the exact waveform without the
merger position would require re-running the signal RNG chain, which is fragile
and couples this module tightly to the internal dataset RNG ordering.

Instead, we use the full *truth* waveform — signal plus noise — as the template.
For high-SNR chirps the noise contribution to the template is small compared to
the signal, so the resulting SNR is approximately equal to the classical signal-
only matched-filter SNR, modulo a noise bias term of order ``sqrt(2*bandwidth)``.
For the purposes of comparing reconstruction methods to each other and to the
zero-filled floor, this approximation is entirely sufficient: the same template
is used for all three SNRs (``snr_truth``, ``snr_recon``, ``snr_masked``), so
the systematic bias cancels in the ratio.

Author
------
Karan Akbari
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import pickle
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from scipy.signal import welch
from tqdm import tqdm

from lisa_gap_imputer.baselines import BASELINES
from lisa_gap_imputer.dataset import StrainDataset, build_splits
from lisa_gap_imputer.model import GapImputer
from lisa_gap_imputer.noise import lisa_psd_sn

__all__ = ["evaluate", "main"]

logger = logging.getLogger(__name__)

# Default ordered method list used when methods=None.
_ALL_METHODS: list[str] = ["model", "zero", "linear", "cubic_spline", "gp_matern32"]


# ---------------------------------------------------------------------------
# Matched-filter inner product
# ---------------------------------------------------------------------------


def _mf_inner(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    fs_hz: float,
) -> float:
    """Compute 4 Re integral a~(f) b~*(f) / S_n(f) df (one-sided, physical units).

    Both ``a`` and ``b`` must already be in physical strain units (i.e. divided by
    the dataset scale before calling this function).

    Parameters
    ----------
    a, b : npt.NDArray[np.float64]
        Time-domain strain arrays of the same length ``L``.
    fs_hz : float
        Sampling frequency in Hz.

    Returns
    -------
    float
        The scalar matched-filter inner product.
    """
    n = len(a)
    df = fs_hz / n  # frequency resolution

    a_tilde = np.fft.rfft(a)
    b_tilde = np.fft.rfft(b)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)

    sn = lisa_psd_sn(freqs)  # shape (K,); +inf at DC

    # Weight: 1/S_n(f); zero where S_n is infinite or zero (no physical power).
    weight = np.where(np.isfinite(sn) & (sn > 0.0), 1.0 / sn, 0.0)

    integrand = weight * (a_tilde * np.conj(b_tilde)).real
    return float(4.0 * df * integrand.sum())


def _snr(
    data: npt.NDArray[np.float64],
    template: npt.NDArray[np.float64],
    fs_hz: float,
) -> float:
    """Matched-filter SNR of *data* against *template*.

    SNR = <data, template> / sqrt(<template, template>).

    A template norm of 0 returns 0 rather than NaN.
    """
    norm_sq = _mf_inner(template, template, fs_hz)
    if norm_sq <= 0.0:
        return 0.0
    return _mf_inner(data, template, fs_hz) / float(np.sqrt(norm_sq))


# ---------------------------------------------------------------------------
# Model inference helper
# ---------------------------------------------------------------------------


def _run_model_batched(
    model: GapImputer,
    test_ds: StrainDataset,
    n_segments: int,
    batch_size: int,
    device: torch.device,
) -> list[npt.NDArray[np.float32]]:
    """Run the model over all test segments and return per-segment reconstructions.

    Returns
    -------
    list of np.ndarray
        Length ``n_segments``; each element is shape ``(L,)`` float32, in scaled
        units, predicting the full strain at every position.
    """
    use_amp = device.type == "cuda"
    results: list[npt.NDArray[np.float32]] = [None] * n_segments  # type: ignore[list-item]

    # Accumulate batches.
    batch_indices: list[int] = []
    batch_strain: list[torch.Tensor] = []
    batch_mask: list[torch.Tensor] = []

    def _flush(indices: list[int], strains: list[torch.Tensor], masks: list[torch.Tensor]) -> None:
        if not indices:
            return
        s = torch.stack(strains).to(device)   # (B, L)
        m = torch.stack(masks).to(device)     # (B, L)
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    out = model(s, m)
            else:
                out = model(s, m)
        out_np = out.cpu().float().numpy()  # (B, L)
        for local_i, seg_idx in enumerate(indices):
            results[seg_idx] = out_np[local_i]

    model.eval()
    with tqdm(total=n_segments, desc="model", unit="seg") as pbar:
        for i in range(n_segments):
            item = test_ds[i]
            batch_indices.append(i)
            batch_strain.append(item["masked_strain"])
            batch_mask.append(item["mask"])
            if len(batch_indices) == batch_size:
                _flush(batch_indices, batch_strain, batch_mask)
                batch_indices = []
                batch_strain = []
                batch_mask = []
            pbar.update(1)

    _flush(batch_indices, batch_strain, batch_mask)
    return results


# ---------------------------------------------------------------------------
# Welch helper
# ---------------------------------------------------------------------------


def _welch_psd(
    residual_physical: npt.NDArray[np.float64],
    fs_hz: float,
    seq_len: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute Welch PSD of a physical-unit residual array."""
    nperseg = min(1024, seq_len)
    freqs, power = welch(
        residual_physical,
        fs=fs_hz,
        nperseg=nperseg,
        scaling="density",
        return_onesided=True,
    )
    return freqs.astype(np.float64), power.astype(np.float64)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def evaluate(
    checkpoint_path: str | pathlib.Path,
    out_path: str | pathlib.Path,
    *,
    n_test: int = 2_000,
    batch_size: int = 64,
    num_workers: int = 2,
    methods: list[str] | None = None,
    device: str | None = None,
    seq_len: int = 4096,
    fs_hz: float = 0.1,
    master_seed_train: int = 0,
    master_seed_val: int = 1,
    master_seed_test: int = 2,
    signal_mix: dict[str, float] | None = None,
    include_periodic_gap: bool = True,
) -> pathlib.Path:
    """Load a checkpoint, evaluate the model and baselines on the test split.

    The normalisation ``scale`` is taken from the checkpoint rather than
    re-estimated, so the test data is in exactly the same scaled units the
    model was trained on.  Specifically, the test :class:`~lisa_gap_imputer.dataset.StrainDataset`
    is constructed with ``scale=checkpoint_scale`` so no entropy is consumed
    from the training split's noise-estimation branch.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        Path to a PyTorch checkpoint saved by the training script.  Must
        contain ``"model_state_dict"``, ``"config"`` (dict of ``GapImputer``
        constructor kwargs), and ``"scale"`` (float).
    out_path : str or pathlib.Path
        Destination ``.pkl`` file.  Parent directory is created if necessary.
    n_test : int, optional
        Number of test segments to evaluate.  Defaults to 2 000.
    batch_size : int, optional
        Batch size for model inference.  Baselines run per-segment regardless.
    num_workers : int, optional
        Passed to :class:`~torch.utils.data.DataLoader` (currently unused because
        baselines iterate per-segment; reserved for future use).
    methods : list of str or None, optional
        Subset of ``["model", "zero", "linear", "cubic_spline", "gp_matern32"]``
        to evaluate.  ``None`` runs all of them.
    device : str or None, optional
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.).
        ``None`` auto-selects CUDA if available, otherwise CPU.
    seq_len : int, optional
        Segment length in samples.  Must match the checkpoint's training config.
    fs_hz : float, optional
        Sampling frequency in Hz.  Must match the checkpoint's training config.
    master_seed_train, master_seed_val, master_seed_test : int, optional
        Master seeds for each split.  ``master_seed_train`` is used to cross-
        check the checkpoint scale.  ``master_seed_test`` determines the test set.
    signal_mix : dict or None, optional
        Signal-type probability distribution passed to :class:`~lisa_gap_imputer.dataset.StrainDataset`.
        ``None`` uses the default ``{"quiet": 0.3, "smbhb": 0.5, "monochromatic": 0.2}``.
    include_periodic_gap : bool, optional
        Whether to include the periodic antenna-repointing gap in the mask.

    Returns
    -------
    pathlib.Path
        Absolute path of the written ``.pkl`` file.

    Notes
    -----
    Scale consistency
        The checkpoint must contain a ``"scale"`` key.  The training-split dataset
        is rebuilt to verify that its self-computed scale agrees with the checkpoint
        scale within 0.1 % (``abs(delta) / checkpoint_scale < 1e-3``).  If not, a
        warning is logged and the checkpoint scale is used — this ensures that all
        methods are evaluated on the same amplitude normalisation.

    Baseline memory
        The GP baseline (``"gp_matern32"``) calls scikit-learn's
        :class:`~sklearn.gaussian_process.GaussianProcessRegressor` per segment.
        For ``n_test = 2 000`` and ``seq_len = 4 096`` this takes roughly 5–10
        minutes on a modern CPU — acceptable for an offline evaluation pass.
    """
    checkpoint_path = pathlib.Path(checkpoint_path)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = list(_ALL_METHODS)
    # Validate method names.
    known = {"model"} | set(BASELINES.keys())
    unknown_methods = set(methods) - known
    if unknown_methods:
        raise ValueError(
            f"Unknown methods: {unknown_methods}. "
            f"Known methods: {known}"
        )

    _default_mix: dict[str, float] = {"quiet": 0.3, "smbhb": 0.5, "monochromatic": 0.2}
    if signal_mix is None:
        signal_mix = _default_mix

    # ------------------------------------------------------------------
    # Device selection.
    # ------------------------------------------------------------------
    if device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)
    logger.info("Using device: %s", _device)

    # ------------------------------------------------------------------
    # Load checkpoint.
    # ------------------------------------------------------------------
    logger.info("Loading checkpoint from %s", checkpoint_path)
    ckpt: dict[str, Any] = torch.load(
        checkpoint_path, map_location=_device, weights_only=False
    )

    if "scale" not in ckpt:
        raise KeyError(
            f"Checkpoint at '{checkpoint_path}' does not contain a 'scale' key. "
            "Re-run training with the current training script that saves 'scale'."
        )
    checkpoint_scale: float = float(ckpt["scale"])
    logger.info("Checkpoint scale: %.6e", checkpoint_scale)

    if "config" not in ckpt:
        raise KeyError(
            f"Checkpoint at '{checkpoint_path}' does not contain a 'config' key."
        )
    model_config: dict[str, Any] = ckpt["config"]

    # ------------------------------------------------------------------
    # Build the model (if requested).
    # ------------------------------------------------------------------
    model: GapImputer | None = None
    if "model" in methods:
        model = GapImputer(**model_config)
        model.load_state_dict(ckpt["state_dict"])
        model.to(_device)
        model.eval()
        logger.info(
            "Model loaded: %d trainable parameters", model.count_parameters()
        )

    # ------------------------------------------------------------------
    # Cross-check scale from training split (informational only).
    # ------------------------------------------------------------------
    logger.info(
        "Rebuilding training split to cross-check scale "
        "(master_seed_train=%d) ...",
        master_seed_train,
    )
    train_ds_check = StrainDataset(
        n_segments=1,  # minimal; we only need the scale attribute
        master_seed=master_seed_train,
        seq_len=seq_len,
        fs_hz=fs_hz,
        signal_mix=signal_mix,
        include_periodic_gap=include_periodic_gap,
        scale=None,    # auto-compute from train entropy
        normalize=True,
    )
    train_scale = train_ds_check.scale
    rel_diff = abs(train_scale - checkpoint_scale) / checkpoint_scale
    if rel_diff >= 1e-3:
        warnings.warn(
            f"Training-split scale ({train_scale:.6e}) differs from checkpoint "
            f"scale ({checkpoint_scale:.6e}) by {rel_diff*100:.4f} % "
            f"(threshold 0.1 %). Using checkpoint scale for evaluation.",
            UserWarning,
            stacklevel=2,
        )
    else:
        logger.info(
            "Scale cross-check passed: train_scale=%.6e, ckpt_scale=%.6e "
            "(rel_diff=%.4f %%)",
            train_scale,
            checkpoint_scale,
            rel_diff * 100,
        )

    # ------------------------------------------------------------------
    # Build test dataset using the checkpoint scale.
    # ------------------------------------------------------------------
    logger.info(
        "Building test dataset: n_test=%d, master_seed_test=%d, scale=%.6e",
        n_test, master_seed_test, checkpoint_scale,
    )
    test_ds = StrainDataset(
        n_segments=n_test,
        master_seed=master_seed_test,
        seq_len=seq_len,
        fs_hz=fs_hz,
        signal_mix=signal_mix,
        include_periodic_gap=include_periodic_gap,
        scale=checkpoint_scale,   # bypass internal scale estimation
        normalize=True,
    )

    # ------------------------------------------------------------------
    # Run model inference (batched).
    # ------------------------------------------------------------------
    model_recons: list[npt.NDArray[np.float32]] | None = None
    if model is not None:
        logger.info("Running model inference on %d segments ...", n_test)
        model_recons = _run_model_batched(
            model, test_ds, n_test, batch_size, _device
        )

    # ------------------------------------------------------------------
    # Initialise accumulators.
    # ------------------------------------------------------------------
    # Per-segment accumulators: dict[method] -> list of scalars.
    per_seg_mse_masked: dict[str, list[float]] = {m: [] for m in methods}
    per_seg_mae_masked: dict[str, list[float]] = {m: [] for m in methods}
    per_seg_mse_observed: dict[str, list[float]] = {m: [] for m in methods}
    per_seg_kind: list[str] = []
    per_seg_n_gaps: list[int] = []
    per_seg_total_gap: list[int] = []
    per_seg_longest_gap: list[int] = []
    per_seg_has_periodic: list[bool] = []

    # Per-gap accumulators: dict[method] -> list of scalars.
    per_gap_seg_idx: dict[str, list[int]] = {m: [] for m in methods}
    per_gap_kind: dict[str, list[str]] = {m: [] for m in methods}
    per_gap_dur: dict[str, list[int]] = {m: [] for m in methods}
    per_gap_mse: dict[str, list[float]] = {m: [] for m in methods}
    per_gap_mae: dict[str, list[float]] = {m: [] for m in methods}

    # PSD residual accumulators: accumulated power spectra (physical units).
    psd_accum: dict[str, npt.NDArray[np.float64] | None] = {m: None for m in methods}
    psd_freqs: npt.NDArray[np.float64] | None = None
    psd_count: int = 0

    # SNR accumulators: only for smbhb segments.
    snr_seg_idx: dict[str, list[int]] = {m: [] for m in methods}
    snr_truth: dict[str, list[float]] = {m: [] for m in methods}
    snr_recon: dict[str, list[float]] = {m: [] for m in methods}
    snr_masked: dict[str, list[float]] = {m: [] for m in methods}

    # ------------------------------------------------------------------
    # Baseline method ordering (model already done above).
    # ------------------------------------------------------------------
    baseline_names = [m for m in methods if m != "model"]

    # ------------------------------------------------------------------
    # Main evaluation loop.
    # ------------------------------------------------------------------
    logger.info("Starting per-segment evaluation loop ...")

    for i in tqdm(range(n_test), desc="eval", unit="seg"):
        item = test_ds[i]
        meta = test_ds.get_meta(i)

        # Ground-truth tensors (scaled float32).
        masked_strain_t: torch.Tensor = item["masked_strain"]   # (L,)
        mask_t: torch.Tensor = item["mask"]                     # (L,) float32 0/1
        truth_t: torch.Tensor = item["truth"]                   # (L,)

        # Convert to numpy float64 for all metric computations.
        masked_strain_np = masked_strain_t.double().numpy()     # (L,) f64
        mask_f32 = mask_t.numpy()                               # (L,) f32, 0/1
        mask_bool: npt.NDArray[np.bool_] = mask_f32.astype(bool)
        truth_np = truth_t.double().numpy()                     # (L,) f64

        obs_bool = ~mask_bool  # True at observed positions

        # ---- Metadata ---------------------------------------------------
        kind: str = meta["kind"]
        gaps: list[dict[str, Any]] = meta["gaps"]

        per_seg_kind.append(kind)
        per_seg_n_gaps.append(len(gaps))
        total_gap = sum(g["end"] - g["start"] for g in gaps)
        per_seg_total_gap.append(total_gap)
        longest_gap = max((g["end"] - g["start"] for g in gaps), default=0)
        per_seg_longest_gap.append(longest_gap)
        has_periodic = any(g["kind"] == "periodic" for g in gaps)
        per_seg_has_periodic.append(has_periodic)

        # ---- Build per-method reconstructions ---------------------------
        # recon_np[method] -> (L,) f64 scaled reconstruction.
        recon_np: dict[str, npt.NDArray[np.float64]] = {}

        if "model" in methods and model_recons is not None:
            # Paste observed positions from truth so the model is compared
            # on equal footing with the baselines, which all preserve
            # ``strain[~mask]`` exactly by construction. In practice you
            # would only use the model's output at masked positions anyway.
            model_raw = model_recons[i].astype(np.float64)
            model_preserved = truth_np.copy()
            model_preserved[mask_bool] = model_raw[mask_bool]
            recon_np["model"] = model_preserved

        for bname in baseline_names:
            fn = BASELINES[bname]
            # Baselines expect a float64 strain with masked positions as
            # zero (the masked_strain already has zeros at gap positions)
            # and a bool mask.
            recon_np[bname] = fn(masked_strain_np, mask_bool)

        # ---- Per-segment metrics ----------------------------------------
        for m in methods:
            recon = recon_np[m]  # (L,) f64

            # MSE / MAE on masked positions.
            if mask_bool.any():
                diff_masked = recon[mask_bool] - truth_np[mask_bool]
                mse_m = float(np.mean(diff_masked ** 2))
                mae_m = float(np.mean(np.abs(diff_masked)))
            else:
                mse_m = 0.0
                mae_m = 0.0

            # MSE on observed positions.
            if obs_bool.any():
                diff_obs = recon[obs_bool] - truth_np[obs_bool]
                mse_o = float(np.mean(diff_obs ** 2))
            else:
                mse_o = 0.0

            per_seg_mse_masked[m].append(mse_m)
            per_seg_mae_masked[m].append(mae_m)
            per_seg_mse_observed[m].append(mse_o)

        # ---- Per-gap metrics --------------------------------------------
        for m in methods:
            recon = recon_np[m]
            for g in gaps:
                start: int = g["start"]
                end: int = g["end"]
                gkind: str = g["kind"]

                gap_recon = recon[start:end]
                gap_truth = truth_np[start:end]
                diff_gap = gap_recon - gap_truth
                mse_g = float(np.mean(diff_gap ** 2)) if diff_gap.size > 0 else 0.0
                mae_g = float(np.mean(np.abs(diff_gap))) if diff_gap.size > 0 else 0.0

                per_gap_seg_idx[m].append(i)
                per_gap_kind[m].append(gkind)
                per_gap_dur[m].append(end - start)
                per_gap_mse[m].append(mse_g)
                per_gap_mae[m].append(mae_g)

        # ---- Residual PSD -----------------------------------------------
        # Convert scaled residual to physical units: divide by scale.
        scale_val = checkpoint_scale
        for m in methods:
            recon = recon_np[m]
            residual_physical = (truth_np - recon) / scale_val
            freqs_w, power_w = _welch_psd(residual_physical, fs_hz, seq_len)

            if psd_freqs is None:
                psd_freqs = freqs_w

            if psd_accum[m] is None:
                psd_accum[m] = power_w.copy()
            else:
                psd_accum[m] = psd_accum[m] + power_w

        psd_count += 1

        # ---- Matched-filter SNR (smbhb only) ----------------------------
        if kind == "smbhb":
            # Template = truth waveform in physical units.
            template_phys = truth_np / scale_val

            for m in methods:
                recon = recon_np[m]
                recon_phys = recon / scale_val
                masked_phys = masked_strain_np / scale_val

                s_truth = _snr(template_phys, template_phys, fs_hz)
                s_recon = _snr(recon_phys, template_phys, fs_hz)
                s_masked = _snr(masked_phys, template_phys, fs_hz)

                snr_seg_idx[m].append(i)
                snr_truth[m].append(s_truth)
                snr_recon[m].append(s_recon)
                snr_masked[m].append(s_masked)

    # ------------------------------------------------------------------
    # Assemble final results dict.
    # ------------------------------------------------------------------
    logger.info("Assembling results dictionary ...")

    # PSD mean.
    psd_mean: dict[str, npt.NDArray[np.float64]] = {}
    for m in methods:
        if psd_accum[m] is not None and psd_count > 0:
            psd_mean[m] = psd_accum[m] / psd_count
        else:
            psd_mean[m] = np.zeros_like(psd_freqs) if psd_freqs is not None else np.array([], dtype=np.float64)

    results: dict[str, Any] = {
        "methods": list(methods),
        "n_segments": n_test,
        "seq_len": seq_len,
        "fs_hz": fs_hz,
        "scale": checkpoint_scale,
        "per_segment": {
            m: {
                "mse_masked": np.array(per_seg_mse_masked[m], dtype=np.float64),
                "mae_masked": np.array(per_seg_mae_masked[m], dtype=np.float64),
                "mse_observed": np.array(per_seg_mse_observed[m], dtype=np.float64),
                "kind": np.array(per_seg_kind, dtype=object),
                "n_gaps": np.array(per_seg_n_gaps, dtype=np.int64),
                "total_gap_samples": np.array(per_seg_total_gap, dtype=np.int64),
                "longest_gap_samples": np.array(per_seg_longest_gap, dtype=np.int64),
                "has_periodic_gap": np.array(per_seg_has_periodic, dtype=bool),
            }
            for m in methods
        },
        "per_gap": {
            m: {
                "segment_idx": np.array(per_gap_seg_idx[m], dtype=np.int64),
                "gap_kind": np.array(per_gap_kind[m], dtype=object),
                "duration_samples": np.array(per_gap_dur[m], dtype=np.int64),
                "mse": np.array(per_gap_mse[m], dtype=np.float64),
                "mae": np.array(per_gap_mae[m], dtype=np.float64),
            }
            for m in methods
        },
        "psd_residual": {
            m: {
                "freqs_hz": psd_freqs if psd_freqs is not None else np.array([], dtype=np.float64),
                "mean_power": psd_mean[m],
            }
            for m in methods
        },
        "snr": {
            m: {
                "segment_idx": np.array(snr_seg_idx[m], dtype=np.int64),
                "snr_truth": np.array(snr_truth[m], dtype=np.float64),
                "snr_recon": np.array(snr_recon[m], dtype=np.float64),
                "snr_masked": np.array(snr_masked[m], dtype=np.float64),
            }
            for m in methods
        },
    }

    # ------------------------------------------------------------------
    # Write pickle.
    # ------------------------------------------------------------------
    logger.info("Writing results to %s", out_path)
    with out_path.open("wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Evaluation complete. Output: %s", out_path)
    return out_path.resolve()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for the evaluation harness.

    Example
    -------
    ::

        python -m lisa_gap_imputer.evaluate \\
            --checkpoint runs/exp01/best.pt \\
            --out results/eval.pkl \\
            --n-test 2000 \\
            --batch-size 64 \\
            --device cuda
    """
    parser = argparse.ArgumentParser(
        prog="lisa_gap_imputer.evaluate",
        description="Evaluate the GapImputer model and classical baselines on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Path to the trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        required=True,
        help="Output path for the results pickle (.pkl).",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=2_000,
        help="Number of test segments to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers (reserved; baselines run single-threaded).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device string (e.g. 'cpu', 'cuda', 'cuda:0'). Auto if omitted.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help="Segment length in samples.",
    )
    parser.add_argument(
        "--fs-hz",
        type=float,
        default=0.1,
        help="Sampling frequency in Hz.",
    )
    parser.add_argument(
        "--seed-train",
        type=int,
        default=0,
        dest="master_seed_train",
        help="Master seed for the training split (used for scale cross-check).",
    )
    parser.add_argument(
        "--seed-val",
        type=int,
        default=1,
        dest="master_seed_val",
        help="Master seed for the validation split (unused in evaluation; reserved).",
    )
    parser.add_argument(
        "--seed-test",
        type=int,
        default=2,
        dest="master_seed_test",
        help="Master seed for the test split.",
    )
    parser.add_argument(
        "--no-periodic-gap",
        action="store_false",
        dest="include_periodic_gap",
        help="Disable the periodic antenna-repointing gap (keep only stochastic gaps).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Subset of methods to evaluate. "
            "Choices: model zero linear cubic_spline gp_matern32. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    out = evaluate(
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        n_test=args.n_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        methods=args.methods,
        device=args.device,
        seq_len=args.seq_len,
        fs_hz=args.fs_hz,
        master_seed_train=args.master_seed_train,
        master_seed_val=args.master_seed_val,
        master_seed_test=args.master_seed_test,
        include_periodic_gap=args.include_periodic_gap,
    )
    print(f"Results written to: {out}")


if __name__ == "__main__":
    main()
