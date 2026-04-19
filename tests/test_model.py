from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from lisa_gap_imputer.model import GapImputer, combined_loss, masked_mse_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    batch: int = 2,
    seq_len: int = 4096,
    masked_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (strain, mask) float32 tensors of shape (batch, seq_len)."""
    torch.manual_seed(seed)
    strain = torch.randn(batch, seq_len)
    mask = torch.zeros(batch, seq_len)
    n_masked = int(seq_len * masked_fraction)
    for b in range(batch):
        start = (seq_len - n_masked) // 2
        mask[b, start : start + n_masked] = 1.0
    # Zero-fill masked positions in strain.
    strain = strain * (1.0 - mask)
    return strain, mask


# ---------------------------------------------------------------------------
# Forward shape tests
# ---------------------------------------------------------------------------


def test_forward_shapes() -> None:
    """GapImputer with default config must map (2, 4096) inputs to (2, 4096) output."""
    model = GapImputer()
    model.eval()
    strain, mask = _make_batch(batch=2, seq_len=4096)
    with torch.no_grad():
        out = model(strain, mask)
    assert out.shape == (2, 4096), f"Expected (2, 4096), got {tuple(out.shape)}"
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"


def test_small_config_fits() -> None:
    """A small GapImputer must forward correctly on (2, 128) inputs and produce finite output."""
    model = GapImputer(seq_len=128, patch_size=8, d_model=32, nhead=2, num_layers=2)
    model.eval()
    strain, mask = _make_batch(batch=2, seq_len=128)
    with torch.no_grad():
        out = model(strain, mask)
    assert out.shape == (2, 128), f"Expected (2, 128), got {tuple(out.shape)}"
    assert torch.all(torch.isfinite(out)), "Small config forward produced non-finite output"


# ---------------------------------------------------------------------------
# masked_mse_loss
# ---------------------------------------------------------------------------


def test_masked_mse_loss_zero_on_perfect_pred() -> None:
    """masked_mse_loss(target, target, mask) must be < 1e-8 (i.e. effectively zero)."""
    _, mask = _make_batch()
    target = torch.randn(2, 4096)
    loss = masked_mse_loss(target, target, mask)
    assert float(loss) < 1e-8, f"Perfect prediction gave non-zero loss: {float(loss)}"


def test_masked_mse_loss_only_counts_masked_positions() -> None:
    """Loss must be ~0 when pred matches target at mask==1 positions, regardless of
    how much pred differs from target at mask==0 positions."""
    _, mask = _make_batch(seq_len=128)
    target = torch.randn(2, 128)
    # Perfect prediction at masked positions; garbage at observed positions.
    pred = target.clone()
    pred[mask == 0] = pred[mask == 0] + 100.0  # large error at observed positions
    loss = masked_mse_loss(pred, target, mask)
    assert float(loss) < 1e-8, (
        f"masked_mse_loss counted observed-position errors; loss={float(loss):.4e}"
    )


# ---------------------------------------------------------------------------
# combined_loss
# ---------------------------------------------------------------------------


def test_combined_loss_is_weighted_sum() -> None:
    """combined_loss must equal masked_weight * masked_mse + observed_weight * observed_mse."""
    torch.manual_seed(0)
    batch, seq_len = 2, 128
    pred = torch.randn(batch, seq_len)
    target = torch.randn(batch, seq_len)
    mask = torch.zeros(batch, seq_len)
    mask[:, 40:80] = 1.0  # simple central gap

    masked_w = 1.0
    observed_w = 0.1

    loss_combined = combined_loss(pred, target, mask, masked_weight=masked_w, observed_weight=observed_w)

    # Manually compute each term.
    observed = 1.0 - mask
    masked_mse_manual = (((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0))
    observed_mse_manual = (((pred - target) ** 2 * observed).sum() / observed.sum().clamp_min(1.0))
    expected = masked_w * masked_mse_manual + observed_w * observed_mse_manual

    assert torch.isclose(loss_combined, expected, rtol=1e-5), (
        f"combined_loss {float(loss_combined):.6e} != expected {float(expected):.6e}"
    )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow() -> None:
    """After a forward + backward pass every trainable parameter must have a non-None grad."""
    model = GapImputer(seq_len=128, patch_size=8, d_model=32, nhead=2, num_layers=2)
    model.train()
    strain, mask = _make_batch(batch=2, seq_len=128)
    target = torch.randn(2, 128)

    out = model(strain, mask)
    loss = masked_mse_loss(out, target, mask)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter '{name}' has no gradient after backward"
            assert torch.any(param.grad != 0.0), (
                f"Parameter '{name}' has an all-zero gradient — may indicate a dead path"
            )


# ---------------------------------------------------------------------------
# Reproducibility with manual seed
# ---------------------------------------------------------------------------


def test_model_reproducible_with_seed() -> None:
    """Two GapImputer instances initialised with the same manual seed must give identical output."""
    def _build_and_run() -> torch.Tensor:
        torch.manual_seed(0)
        model = GapImputer(seq_len=128, patch_size=8, d_model=32, nhead=2, num_layers=2)
        model.eval()
        strain, mask = _make_batch(batch=2, seq_len=128, seed=0)
        with torch.no_grad():
            return model(strain, mask)

    out_a = _build_and_run()
    out_b = _build_and_run()
    assert torch.equal(out_a, out_b), (
        "Two identically seeded GapImputer instances gave different output"
    )
