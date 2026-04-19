from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from lisa_gap_imputer.dataset import StrainDataset, build_splits


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SEQ_LEN = 256   # short to keep tests fast
_FS_HZ = 0.1
_SIGNAL_MIX = {"quiet": 0.3, "smbhb": 0.5, "monochromatic": 0.2}


def _make_dataset(n: int = 4, seed: int = 42) -> StrainDataset:
    return StrainDataset(
        n_segments=n,
        master_seed=seed,
        seq_len=_SEQ_LEN,
        fs_hz=_FS_HZ,
        signal_mix=_SIGNAL_MIX,
        include_periodic_gap=True,
    )


# ---------------------------------------------------------------------------
# Length and item keys
# ---------------------------------------------------------------------------


def test_dataset_length_and_keys() -> None:
    """Dataset must have correct length and each item must have the four expected keys."""
    n = 6
    ds = _make_dataset(n=n)
    assert len(ds) == n

    item = ds[0]
    expected_keys = {"masked_strain", "mask", "truth", "scale"}
    assert set(item.keys()) == expected_keys, (
        f"Item keys {set(item.keys())} differ from expected {expected_keys}"
    )

    # All four must be float32.
    for k in expected_keys:
        assert item[k].dtype == torch.float32, f"Key '{k}' has dtype {item[k].dtype}"

    # Shape checks: (seq_len,) for the arrays, () for scale.
    for k in ("masked_strain", "mask", "truth"):
        assert item[k].shape == (_SEQ_LEN,), (
            f"Key '{k}' has shape {item[k].shape}, expected ({_SEQ_LEN},)"
        )
    assert item["scale"].shape == (), (
        f"'scale' should be 0-d; got shape {item['scale'].shape}"
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_dataset_reproducible() -> None:
    """Calling ds[0] twice must return bit-identical tensors."""
    ds = _make_dataset()
    a = ds[0]
    b = ds[0]
    for k in ("masked_strain", "mask", "truth", "scale"):
        assert torch.equal(a[k], b[k]), f"Key '{k}' differs between two calls to ds[0]"


def test_dataset_index_independence() -> None:
    """ds[0] must not change after accessing ds[1], ds[2], etc. (map-style)."""
    ds = _make_dataset(n=4)
    item0_before = ds[0]
    # Access other indices.
    _ = ds[1]
    _ = ds[2]
    _ = ds[3]
    item0_after = ds[0]
    for k in ("masked_strain", "mask", "truth"):
        assert torch.equal(item0_before[k], item0_after[k]), (
            f"Key '{k}' in ds[0] changed after accessing other indices"
        )


# ---------------------------------------------------------------------------
# Mask / strain semantics
# ---------------------------------------------------------------------------


def test_masked_strain_zeros_at_mask() -> None:
    """masked_strain must be exactly 0 wherever mask == 1."""
    ds = _make_dataset()
    item = ds[0]
    masked_positions = item["mask"] == 1.0
    assert torch.all(item["masked_strain"][masked_positions] == 0.0), (
        "masked_strain is non-zero at masked positions"
    )


def test_masked_strain_equals_truth_at_observed() -> None:
    """masked_strain must equal truth at all observed (mask==0) positions."""
    ds = _make_dataset()
    item = ds[0]
    observed = item["mask"] == 0.0
    assert torch.all(item["masked_strain"][observed] == item["truth"][observed]), (
        "masked_strain differs from truth at observed (non-masked) positions"
    )


# ---------------------------------------------------------------------------
# build_splits
# ---------------------------------------------------------------------------


def test_build_splits_shared_scale() -> None:
    """All three splits returned by build_splits must share the same scale."""
    train, val, test = build_splits(
        seq_len=_SEQ_LEN,
        fs_hz=_FS_HZ,
        n_train=4,
        n_val=2,
        n_test=2,
        master_seed_train=10,
        master_seed_val=11,
        master_seed_test=12,
        signal_mix=_SIGNAL_MIX,
    )
    assert train.scale == val.scale, "train.scale != val.scale"
    assert train.scale == test.scale, "train.scale != test.scale"


def test_build_splits_distinct_entropy() -> None:
    """First segment of train, val, test must produce different masked_strain tensors."""
    train, val, test = build_splits(
        seq_len=_SEQ_LEN,
        fs_hz=_FS_HZ,
        n_train=4,
        n_val=2,
        n_test=2,
        master_seed_train=10,
        master_seed_val=11,
        master_seed_test=12,
        signal_mix=_SIGNAL_MIX,
    )
    t0 = train[0]["masked_strain"]
    v0 = val[0]["masked_strain"]
    e0 = test[0]["masked_strain"]
    assert not torch.equal(t0, v0), "train[0] and val[0] are identical — seeds not distinct"
    assert not torch.equal(t0, e0), "train[0] and test[0] are identical — seeds not distinct"
    assert not torch.equal(v0, e0), "val[0] and test[0] are identical — seeds not distinct"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_signal_mix_sum_raises() -> None:
    """signal_mix summing to 0.9 (not 1.0) must raise ValueError."""
    with pytest.raises(ValueError, match="sum"):
        StrainDataset(
            n_segments=2,
            master_seed=0,
            seq_len=_SEQ_LEN,
            fs_hz=_FS_HZ,
            signal_mix={"quiet": 0.3, "smbhb": 0.4, "monochromatic": 0.2},  # sums to 0.9
        )


def test_invalid_signal_mix_unknown_key_raises() -> None:
    """signal_mix with an unknown key must raise ValueError."""
    with pytest.raises(ValueError):
        StrainDataset(
            n_segments=2,
            master_seed=0,
            seq_len=_SEQ_LEN,
            fs_hz=_FS_HZ,
            signal_mix={"quiet": 0.5, "unknown_source": 0.5},
        )
