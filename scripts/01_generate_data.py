"""01_generate_data.py — preview and cache dataset samples.

Sanity-checks the data pipeline without training. Writes a small HDF5 file
containing N preview segments from each of train/val/test splits, plus a
preview plot PNG showing 4 random segments with gaps overlaid in red.

Usage
-----
    python scripts/01_generate_data.py --out-dir outputs/data_preview
    python -m lisa_gap_imputer.scripts.01_generate_data --out-dir outputs/data_preview

Author: Karan Akbari
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_signal_mix(raw: str) -> dict[str, float]:
    """Parse 'key=val,key=val' into a dict of floats."""
    result: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        result[k.strip()] = float(v.strip())
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="01_generate_data",
        description="Preview and cache LISA dataset samples to HDF5 + PNG.",
    )
    p.add_argument("--out-dir", required=True, help="Directory for output files.")
    p.add_argument(
        "--n-preview",
        type=int,
        default=16,
        help="Number of segments to preview from each split (default: 16).",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help="Sequence length in samples (default: 4096).",
    )
    p.add_argument(
        "--fs-hz",
        type=float,
        default=0.1,
        help="Sampling frequency in Hz (default: 0.1).",
    )
    p.add_argument(
        "--master-seed-train",
        type=int,
        default=0,
        help="Master RNG seed for train split (default: 0).",
    )
    p.add_argument(
        "--master-seed-val",
        type=int,
        default=1,
        help="Master RNG seed for val split (default: 1).",
    )
    p.add_argument(
        "--master-seed-test",
        type=int,
        default=2,
        help="Master RNG seed for test split (default: 2).",
    )
    p.add_argument(
        "--no-periodic-gap",
        dest="include_periodic_gap",
        action="store_false",
        help="Disable periodic gap injection.",
    )
    p.add_argument(
        "--signal-mix",
        type=str,
        default=None,
        help=(
            "Signal mixture as comma-separated key=value pairs, "
            "e.g. 'gbwd=0.7,noise=0.3'. Defaults to dataset default."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Deferred imports so argparse --help works without heavy deps installed.
    import numpy as np
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lisa_gap_imputer.dataset import build_splits

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signal_mix = _parse_signal_mix(args.signal_mix) if args.signal_mix else None
    n = args.n_preview

    print(
        f"[01_generate_data] Building splits with n_preview={n} per split, "
        f"seq_len={args.seq_len}, fs_hz={args.fs_hz} ..."
    )

    train_ds, val_ds, test_ds = build_splits(
        seq_len=args.seq_len,
        fs_hz=args.fs_hz,
        n_train=n,
        n_val=n,
        n_test=n,
        master_seed_train=args.master_seed_train,
        master_seed_val=args.master_seed_val,
        master_seed_test=args.master_seed_test,
        signal_mix=signal_mix,
        include_periodic_gap=args.include_periodic_gap,
    )

    h5_path = out_dir / "preview_data.h5"
    print(f"[01_generate_data] Writing HDF5 → {h5_path}")

    with h5py.File(h5_path, "w") as f:
        for split_name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
            grp = f.create_group(split_name)
            truths, masked_strains, masks, kinds = [], [], [], []
            n_gaps_list, total_gap_list, longest_gap_list = [], [], []

            for i in range(n):
                item = ds[i]
                truths.append(item["truth"].numpy())
                masked_strains.append(item["masked_strain"].numpy())
                masks.append(item["mask"].numpy())

                meta = ds.get_meta(i)
                kinds.append(str(meta["kind"]).encode())
                gaps = meta["gaps"]
                durations = [g["end"] - g["start"] for g in gaps]
                n_gaps_list.append(len(gaps))
                total_gap_list.append(int(sum(durations)))
                longest_gap_list.append(int(max(durations)) if durations else 0)

            grp.create_dataset("truth", data=np.stack(truths))
            grp.create_dataset("masked_strain", data=np.stack(masked_strains))
            grp.create_dataset("mask", data=np.stack(masks))
            grp.create_dataset("kind", data=np.array(kinds, dtype=h5py.special_dtype(vlen=str)))
            grp.create_dataset("n_gaps", data=np.array(n_gaps_list, dtype=np.int64))
            grp.create_dataset("total_gap_samples", data=np.array(total_gap_list, dtype=np.int64))
            grp.create_dataset("longest_gap_samples", data=np.array(longest_gap_list, dtype=np.int64))

    print(f"[01_generate_data] HDF5 written ({3 * n} segments total).")

    # ------------------------------------------------------------------ plot --
    png_path = out_dir / "preview_segments.png"
    print(f"[01_generate_data] Plotting preview → {png_path}")

    rng = np.random.default_rng(42)
    plot_indices = rng.choice(n, size=min(4, n), replace=False).tolist()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    fig.suptitle("LISA gap-imputer — data preview (train split)", fontsize=12)

    time_axis = np.arange(args.seq_len) / args.fs_hz  # seconds

    for ax, idx in zip(axes, plot_indices):
        item = train_ds[idx]
        truth = item["truth"].numpy()
        mask = item["mask"].numpy()
        meta = train_ds.get_meta(idx)

        ax.plot(time_axis, truth, color="steelblue", lw=0.8, label="truth", alpha=0.9)

        # mask == 1.0 at MASKED positions; shade those in red.
        in_gap = mask > 0.5
        if in_gap.any():
            transitions = np.diff(in_gap.astype(int), prepend=0, append=0)
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]
            for gs, ge in zip(starts, ends):
                ax.axvspan(
                    time_axis[gs],
                    time_axis[min(ge, len(time_axis) - 1)],
                    color="red",
                    alpha=0.25,
                    label="_gap",
                )

        kind = meta.get("kind", "?")
        ax.set_ylabel("h(t)", fontsize=8)
        ax.set_title(f"segment {idx} — kind={kind}", fontsize=9)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Time [s]", fontsize=9)

    # Single legend entry for gap shading.
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color="steelblue", lw=1.5, label="truth"),
        Patch(facecolor="red", alpha=0.3, label="gap"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"[01_generate_data] Done. Outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
