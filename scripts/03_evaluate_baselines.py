"""03_evaluate_baselines.py — evaluate baseline imputation methods only.

Runs lisa_gap_imputer.evaluate.evaluate with methods restricted to
["zero", "linear", "cubic_spline", "gp_matern32"] (excludes the trained
model). A checkpoint path is still required because evaluate.py uses
checkpoint metadata (e.g. normalisation scale) regardless of method set.

Usage
-----
    python scripts/03_evaluate_baselines.py \\
        --checkpoint outputs/run1/best.pt \\
        --out outputs/run1/baseline_results.pkl

    python -m lisa_gap_imputer.scripts.03_evaluate_baselines \\
        --checkpoint outputs/run1/best.pt \\
        --out outputs/run1/baseline_results.pkl

Author: Karan Akbari
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BASELINE_METHODS = ["zero", "linear", "cubic_spline", "gp_matern32"]


def _parse_signal_mix(raw: str) -> dict[str, float]:
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
        prog="03_evaluate_baselines",
        description=(
            "Evaluate baseline imputation methods (zero, linear, cubic_spline, "
            "gp_matern32) and write results to a pickle file."
        ),
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint (.pt). Used for dataset metadata.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output path for results pickle (.pkl).",
    )
    p.add_argument(
        "--n-test",
        type=int,
        default=2000,
        help="Number of test segments to evaluate (default: 2000).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model inference (default: 64).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count (default: 2).",
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
        help="Master RNG seed used during training (default: 0).",
    )
    p.add_argument(
        "--master-seed-val",
        type=int,
        default=1,
        help="Master RNG seed used for validation (default: 1).",
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
        help="Disable periodic gap injection (must match training setting).",
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
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (e.g. 'cuda', 'cpu'). Auto-detected if omitted.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from lisa_gap_imputer.evaluate import evaluate

    checkpoint = Path(args.checkpoint)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    signal_mix = _parse_signal_mix(args.signal_mix) if args.signal_mix else None

    print(
        f"[03_evaluate_baselines] Running methods: {_BASELINE_METHODS}\n"
        f"  checkpoint : {checkpoint}\n"
        f"  output     : {out_path}\n"
        f"  n_test     : {args.n_test}"
    )

    result_path = evaluate(
        checkpoint_path=checkpoint,
        out_path=out_path,
        n_test=args.n_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        methods=_BASELINE_METHODS,
        device=args.device,
        seq_len=args.seq_len,
        fs_hz=args.fs_hz,
        master_seed_train=args.master_seed_train,
        master_seed_val=args.master_seed_val,
        master_seed_test=args.master_seed_test,
        signal_mix=signal_mix,
        include_periodic_gap=args.include_periodic_gap,
    )

    print(f"[03_evaluate_baselines] Done. Results written to {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
