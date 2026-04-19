"""02_train.py — thin wrapper around lisa_gap_imputer.train.

Exists for discoverability: users look in scripts/ for entry points.
All argument parsing is delegated to lisa_gap_imputer.train.main.

Usage
-----
    python scripts/02_train.py --out-dir outputs/run1 --epochs 50
    python -m lisa_gap_imputer.scripts.02_train --out-dir outputs/run1 --epochs 50

Author: Karan Akbari
"""
from __future__ import annotations

import sys

from lisa_gap_imputer.train import main as _train_main


def main(argv: list[str] | None = None) -> int:
    print("[02_train] Delegating to lisa_gap_imputer.train ...")
    return _train_main(argv if argv is not None else sys.argv[1:]) or 0


if __name__ == "__main__":
    sys.exit(main())
