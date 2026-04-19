"""Money plots for the LISA gap-imputation exploratory transformer.

Five outputs are produced by ``make_all_plots``:

1. ``recon_mse_vs_gap_duration`` — per-gap MSE vs gap duration (log-log),
   binned median + IQR, split by gap kind (periodic / stochastic).
2. ``snr_recovery`` — scatter of SNR recovery fraction vs truth SNR, one panel
   per method, with the masked (do-nothing) floor overlaid.
3. ``residual_psd`` — log-log power spectral density of the imputation residual
   per method, compared against the theoretical LISA S_n(f).
4. ``mse_boxplot`` — log-scale boxplot of per-segment masked MSE across methods
   with jittered individual points.
5. ``summary_table.csv`` — CSV summary of median MSE, IQR, and SMBHB SNR
   recovery per method.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import warnings
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from lisa_gap_imputer.noise import lisa_psd_sn

matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["figure.figsize"] = (7, 4.5)

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------

_METHOD_COLOR: dict[str, str] = {
    "model": "C0",
    "zero": "C7",
    "linear": "C1",
    "cubic_spline": "C2",
    "gp_matern32": "C3",
}

_METHOD_LABEL: dict[str, str] = {
    "model": "Model",
    "zero": "Zero fill",
    "linear": "Linear",
    "cubic_spline": "Cubic spline",
    "gp_matern32": "GP (Matern-3/2)",
}

_LW: dict[str, float] = {m: (2.0 if m == "model" else 1.2) for m in _METHOD_COLOR}
_ALPHA_MARKER: dict[str, float] = {
    m: (0.9 if m == "model" else 0.6) for m in _METHOD_COLOR
}

_N_BINS: int = 15  # log-spaced bins for gap-duration axis


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _method_color(method: str) -> str:
    return _METHOD_COLOR.get(method, "C4")


def _method_label(method: str) -> str:
    return _METHOD_LABEL.get(method, method)


def _method_lw(method: str) -> float:
    return _LW.get(method, 1.2)


def _method_alpha(method: str) -> float:
    return _ALPHA_MARKER.get(method, 0.6)


def _save(fig: matplotlib.figure.Figure, out_path: pathlib.Path) -> None:
    """Save figure as both PNG and PDF, then close it."""
    png_path = out_path.with_suffix(".png")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _log_bin_stat(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    n_bins: int = _N_BINS,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Bin ``y`` into ``n_bins`` log-spaced bins of ``x``.

    Returns (bin_centers, median, q25, q75) — all NaN for empty bins.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    positive = (x > 0) & (y > 0)
    x = x[positive]
    y = y[positive]

    if x.size == 0:
        empty: npt.NDArray[np.float64] = np.full(n_bins, np.nan)
        return empty, empty, empty, empty

    log_edges = np.linspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    bin_centers = 10.0 ** (0.5 * (log_edges[:-1] + log_edges[1:]))
    log_x = np.log10(x)

    medians = np.full(n_bins, np.nan)
    q25s = np.full(n_bins, np.nan)
    q75s = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (log_x >= log_edges[i]) & (log_x < log_edges[i + 1])
        if mask.sum() == 0:
            continue
        vals = y[mask]
        medians[i] = float(np.median(vals))
        q25s[i] = float(np.percentile(vals, 25))
        q75s[i] = float(np.percentile(vals, 75))

    return bin_centers, medians, q25s, q75s


# ---------------------------------------------------------------------------
# Figure 1 — MSE vs gap duration
# ---------------------------------------------------------------------------


def plot_recon_mse_vs_gap_duration(
    results: dict[str, Any], out_path: pathlib.Path
) -> pathlib.Path:
    """Per-gap MSE vs gap duration, split by gap kind.

    Two side-by-side panels (periodic / stochastic), log-log axes.
    Median + IQR band per method per bin.
    """
    gap_kinds: list[str] = ["periodic", "stochastic"]
    methods: list[str] = results.get("methods", list(results["per_gap"].keys()))
    per_gap: dict[str, Any] = results["per_gap"]

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4.5), sharey=True, constrained_layout=True
    )

    for ax, kind in zip(axes, gap_kinds):
        any_drawn = False
        for method in methods:
            data = per_gap.get(method)
            if data is None:
                continue
            kind_mask: npt.NDArray[np.bool_] = (
                np.asarray(data["gap_kind"]) == kind
            )
            if kind_mask.sum() == 0:
                continue

            dur = np.asarray(data["duration_samples"], dtype=np.float64)[kind_mask]
            mse = np.asarray(data["mse"], dtype=np.float64)[kind_mask]

            centers, medians, q25, q75 = _log_bin_stat(dur, mse)

            valid = ~np.isnan(medians)
            if valid.sum() == 0:
                continue

            color = _method_color(method)
            lw = _method_lw(method)
            label = _method_label(method)

            ax.plot(
                centers[valid],
                medians[valid],
                color=color,
                linewidth=lw,
                label=label,
                marker="o",
                markersize=3,
            )
            ax.fill_between(
                centers[valid],
                q25[valid],
                q75[valid],
                color=color,
                alpha=0.15,
            )
            any_drawn = True

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Gap duration (samples)")
        ax.set_title(f"{kind.capitalize()} gaps")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

        if not any_drawn:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="grey",
            )

    axes[0].set_ylabel("Per-gap MSE")
    axes[0].legend(loc="upper left", fontsize=8)

    fig.suptitle("Reconstruction MSE vs gap duration", y=1.01)

    _save(fig, out_path)
    return out_path.with_suffix(".png")


# ---------------------------------------------------------------------------
# Figure 2 — SNR recovery
# ---------------------------------------------------------------------------


def plot_snr_recovery(
    results: dict[str, Any], out_path: pathlib.Path
) -> pathlib.Path | None:
    """SNR recovery fraction vs truth SNR.

    Scatter plot with y = snr_recon / snr_truth (solid) and
    y = snr_masked / snr_truth (faint dashed, "do nothing" floor).
    One panel per method.  Skips with a warning if snr dict is empty.
    """
    snr_data: dict[str, Any] = results.get("snr", {})
    if not snr_data:
        warnings.warn(
            "snr dict is empty — no SMBHB segments found; skipping snr_recovery.png",
            stacklevel=2,
        )
        return None

    # Drop methods whose SNR arrays are all-empty
    active_methods: list[str] = [
        m
        for m in results.get("methods", list(snr_data.keys()))
        if m in snr_data and np.asarray(snr_data[m]["snr_truth"]).size > 0
    ]

    if not active_methods:
        warnings.warn(
            "All SNR arrays are empty — skipping snr_recovery.png",
            stacklevel=2,
        )
        return None

    n_methods = len(active_methods)
    fig, axes = plt.subplots(
        1,
        n_methods,
        figsize=(3.0 * n_methods, 4.5),
        sharey=True,
        constrained_layout=True,
    )
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, active_methods):
        d = snr_data[method]
        snr_truth = np.asarray(d["snr_truth"], dtype=np.float64)
        snr_recon = np.asarray(d["snr_recon"], dtype=np.float64)
        snr_masked = np.asarray(d["snr_masked"], dtype=np.float64)

        pos = snr_truth > 0
        truth = snr_truth[pos]
        recovery = snr_recon[pos] / truth
        floor = snr_masked[pos] / truth

        color = _method_color(method)
        alpha = _method_alpha(method)

        ax.scatter(
            truth,
            recovery,
            s=8,
            color=color,
            alpha=alpha,
            label="Reconstructed",
            rasterized=True,
        )
        ax.scatter(
            truth,
            floor,
            s=5,
            color="grey",
            alpha=0.3,
            marker="x",
            label="Masked (no fill)",
            rasterized=True,
        )
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="y = 1")

        ax.set_xlabel("SNR (truth)")
        ax.set_title(_method_label(method), fontsize=9)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

        if method == active_methods[0]:
            ax.set_ylabel("SNR recovery fraction")
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("SNR recovery", y=1.01)

    _save(fig, out_path)
    return out_path.with_suffix(".png")


# ---------------------------------------------------------------------------
# Figure 3 — Residual PSD
# ---------------------------------------------------------------------------


def plot_residual_psd(
    results: dict[str, Any], out_path: pathlib.Path
) -> pathlib.Path:
    """Log-log residual PSD per method vs theoretical LISA S_n(f)."""
    psd_data: dict[str, Any] = results.get("psd_residual", {})
    methods: list[str] = results.get("methods", list(psd_data.keys()))

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    # Collect all positive frequencies to build the S_n reference grid.
    ref_freqs: npt.NDArray[np.float64] | None = None

    for method in methods:
        d = psd_data.get(method)
        if d is None:
            continue
        freqs = np.asarray(d["freqs_hz"], dtype=np.float64)
        power = np.asarray(d["mean_power"], dtype=np.float64)

        pos = freqs > 0
        if pos.sum() == 0:
            continue

        if ref_freqs is None:
            ref_freqs = freqs[pos]

        color = _method_color(method)
        lw = _method_lw(method)
        ax.plot(
            freqs[pos],
            power[pos],
            color=color,
            linewidth=lw,
            label=_method_label(method),
            alpha=0.85,
        )

    # Theoretical S_n(f) reference.
    if ref_freqs is not None and ref_freqs.size > 0:
        sn = lisa_psd_sn(ref_freqs)
        finite = np.isfinite(sn) & (sn > 0)
        if finite.sum() > 0:
            ax.plot(
                ref_freqs[finite],
                sn[finite],
                color="black",
                linewidth=1.5,
                linestyle="--",
                label=r"LISA $S_n(f)$",
                zorder=5,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Power spectral density (Hz$^{-1}$)")
    ax.set_title("Residual PSD vs LISA noise floor")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    _save(fig, out_path)
    return out_path.with_suffix(".png")


# ---------------------------------------------------------------------------
# Figure 4 — MSE boxplot
# ---------------------------------------------------------------------------


def plot_mse_boxplot(
    results: dict[str, Any], out_path: pathlib.Path
) -> pathlib.Path:
    """Boxplot of per-segment masked MSE across methods (log y-axis).

    Individual points (up to 500 per method) are overlaid with jitter.
    """
    per_segment: dict[str, Any] = results.get("per_segment", {})
    methods: list[str] = results.get("methods", list(per_segment.keys()))

    data_list: list[npt.NDArray[np.float64]] = []
    valid_methods: list[str] = []

    for method in methods:
        d = per_segment.get(method)
        if d is None:
            continue
        vals = np.asarray(d["mse_masked"], dtype=np.float64)
        vals = vals[(vals > 0) & np.isfinite(vals)]
        if vals.size == 0:
            continue
        data_list.append(vals)
        valid_methods.append(method)

    if not valid_methods:
        warnings.warn("No valid MSE data found; skipping mse_boxplot.png", stacklevel=2)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _save(fig, out_path)
        return out_path.with_suffix(".png")

    n_methods = len(valid_methods)
    positions = np.arange(1, n_methods + 1)

    fig, ax = plt.subplots(figsize=(max(5, 1.5 * n_methods), 4.5), constrained_layout=True)

    rng = np.random.default_rng(42)

    for pos, method, vals in zip(positions, valid_methods, data_list):
        color = _method_color(method)

        bp = ax.boxplot(
            vals,
            positions=[pos],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="white", linewidth=1.5),
            boxprops=dict(facecolor=color, alpha=0.5, linewidth=0.8),
            whiskerprops=dict(color=color, linewidth=0.8),
            capprops=dict(color=color, linewidth=0.8),
        )
        _ = bp  # suppress unused warning

        # Jittered individual points — at most 500 per method.
        n_pts = min(500, vals.size)
        idx = rng.choice(vals.size, size=n_pts, replace=False)
        jitter = rng.uniform(-0.18, 0.18, size=n_pts)
        ax.scatter(
            pos + jitter,
            vals[idx],
            s=3,
            color=color,
            alpha=0.1,
            zorder=2,
            rasterized=True,
        )

    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [_method_label(m) for m in valid_methods], rotation=20, ha="right"
    )
    ax.set_ylabel("MSE (masked region)")
    ax.set_title("Per-segment masked MSE by method")
    ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    _save(fig, out_path)
    return out_path.with_suffix(".png")


# ---------------------------------------------------------------------------
# Table — summary CSV
# ---------------------------------------------------------------------------


def write_summary_table(
    results: dict[str, Any], out_path: pathlib.Path
) -> pathlib.Path:
    """Write a CSV summary with one row per method.

    Columns: method, median_mse_masked, iqr_mse_masked,
    median_snr_recovery (smbhb only), n_smbhb, n_segments.
    """
    per_segment: dict[str, Any] = results.get("per_segment", {})
    snr_data: dict[str, Any] = results.get("snr", {})
    methods: list[str] = results.get("methods", list(per_segment.keys()))

    rows: list[str] = [
        "method,median_mse_masked,iqr_mse_masked,"
        "median_snr_recovery_smbhb,n_smbhb,n_segments"
    ]

    for method in methods:
        d = per_segment.get(method)
        if d is None:
            continue

        mse_all = np.asarray(d["mse_masked"], dtype=np.float64)
        kind_arr = np.asarray(d["kind"])
        n_segments = int(mse_all.size)

        valid_mse = mse_all[np.isfinite(mse_all) & (mse_all > 0)]
        median_mse = float(np.median(valid_mse)) if valid_mse.size > 0 else float("nan")
        iqr_mse = float(
            np.percentile(valid_mse, 75) - np.percentile(valid_mse, 25)
        ) if valid_mse.size > 0 else float("nan")

        # SMBHB SNR recovery.
        smbhb_mask = kind_arr == "smbhb"
        n_smbhb = int(smbhb_mask.sum())

        snr_recovery = float("nan")
        if n_smbhb > 0 and method in snr_data:
            sd = snr_data[method]
            seg_idx = np.asarray(sd["segment_idx"], dtype=np.int64)
            snr_truth = np.asarray(sd["snr_truth"], dtype=np.float64)
            snr_recon = np.asarray(sd["snr_recon"], dtype=np.float64)

            # Find per-segment indices that correspond to smbhb.
            smbhb_seg_indices = np.where(smbhb_mask)[0]
            sel = np.isin(seg_idx, smbhb_seg_indices)

            if sel.sum() > 0 and np.any(snr_truth[sel] > 0):
                pos = sel & (snr_truth > 0)
                recoveries = snr_recon[pos] / snr_truth[pos]
                snr_recovery = float(np.median(recoveries))

        rows.append(
            f"{method},{median_mse:.6e},{iqr_mse:.6e},"
            f"{snr_recovery:.4f},{n_smbhb},{n_segments}"
        )

    csv_path = out_path.with_suffix(".csv")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return csv_path


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def make_all_plots(
    results_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
) -> list[pathlib.Path]:
    """Produce all money plots and the summary CSV.

    Parameters
    ----------
    results_path:
        Path to the pickle file produced by ``evaluate.py``.
    out_dir:
        Directory where figures and the CSV are written.  Created if absent.

    Returns
    -------
    list[pathlib.Path]
        Paths to every file written (PNG, PDF, CSV).
    """
    results_path = pathlib.Path(results_path)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with results_path.open("rb") as fh:
        results: dict[str, Any] = pickle.load(fh)

    written: list[pathlib.Path] = []

    # ------------------------------------------------------------------
    # Figure 1
    # ------------------------------------------------------------------
    base1 = out_dir / "recon_mse_vs_gap_duration"
    png1 = plot_recon_mse_vs_gap_duration(results, base1)
    written.extend([png1, base1.with_suffix(".pdf")])

    # ------------------------------------------------------------------
    # Figure 2
    # ------------------------------------------------------------------
    base2 = out_dir / "snr_recovery"
    result2 = plot_snr_recovery(results, base2)
    if result2 is not None:
        written.extend([result2, base2.with_suffix(".pdf")])

    # ------------------------------------------------------------------
    # Figure 3
    # ------------------------------------------------------------------
    base3 = out_dir / "residual_psd"
    png3 = plot_residual_psd(results, base3)
    written.extend([png3, base3.with_suffix(".pdf")])

    # ------------------------------------------------------------------
    # Figure 4
    # ------------------------------------------------------------------
    base4 = out_dir / "mse_boxplot"
    png4 = plot_mse_boxplot(results, base4)
    written.extend([png4, base4.with_suffix(".pdf")])

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    csv_path = write_summary_table(results, out_dir / "summary_table")
    written.append(csv_path)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate money plots for a LISA gap-imputer evaluation pickle."
    )
    parser.add_argument(
        "--results",
        required=True,
        metavar="PATH",
        help="Path to the results pickle produced by evaluate.py.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        metavar="DIR",
        help="Directory in which figures and CSV are saved.",
    )
    args = parser.parse_args(argv)

    written = make_all_plots(args.results, args.out_dir)
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
