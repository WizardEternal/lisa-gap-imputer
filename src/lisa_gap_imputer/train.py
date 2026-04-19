"""Training loop for the LISA gap-imputation transformer.

Design notes
------------
**Amplitude normalisation**
Raw LISA strain amplitudes are O(1e-21), which is poorly conditioned for
float32 gradient descent. The dataset layer (``build_splits``) computes a
single ``scale`` from 64 noise realisations of the training split and applies
it uniformly to all splits. This maps the colored-noise floor to O(1) and
keeps SMBHB chirp amplitudes within float32 dynamic range without any
per-segment renormalisation.

**Combined loss rationale**
``combined_loss`` drives the model with two terms: a primary MSE on the gap
(masked) positions and a lightweight MSE on the observed positions. The
observed term (default weight 0.1) prevents the model from distorting clean
data and provides a dense gradient signal early in training when gap coverage
is sparse. Validation early-stopping tracks the masked component alone, which
is the true measure of imputation quality.

**AMP + OneCycleLR**
Mixed-precision (float16 on CUDA) roughly halves activation memory, allowing a
batch size of 32 with seq_len=4096 to fit comfortably on a 6 GB GPU. The
GradScaler handles the loss-scaling book-keeping transparently. OneCycleLR with
a short warm-up (10 % of steps) and cosine annealing eliminates the need to
hand-tune a separate warm-up schedule and consistently reaches lower validation
loss than a fixed learning rate within the same epoch budget.

Author
------
Karan Akbari
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from lisa_gap_imputer.dataset import build_splits
from lisa_gap_imputer.model import GapImputer, combined_loss, masked_mse_loss

__all__ = ["train", "main"]

# ---------------------------------------------------------------------------
# Module-level default constants (mirrors function defaults for reference)
# ---------------------------------------------------------------------------

DEFAULT_SEQ_LEN: int = 4096
DEFAULT_FS_HZ: float = 0.1
DEFAULT_N_TRAIN: int = 20_000
DEFAULT_N_VAL: int = 2_000
DEFAULT_EPOCHS: int = 40
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LR: float = 3e-4
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_MASKED_WEIGHT: float = 1.0
DEFAULT_OBSERVED_WEIGHT: float = 0.1
DEFAULT_PATIENCE: int = 7
DEFAULT_NUM_WORKERS: int = 2
DEFAULT_PATCH_SIZE: int = 8
DEFAULT_D_MODEL: int = 128
DEFAULT_NHEAD: int = 4
DEFAULT_NUM_LAYERS: int = 5
DEFAULT_DROPOUT: float = 0.1
DEFAULT_SEED: int = 0
DEFAULT_MASTER_SEED_TRAIN: int = 0
DEFAULT_MASTER_SEED_VAL: int = 1
DEFAULT_INCLUDE_PERIODIC_GAP: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _set_seeds(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _resolve_device(device: str | None) -> torch.device:
    """Return the target ``torch.device``, auto-detecting CUDA if *device* is None."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loaders(
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Construct train and validation ``DataLoader`` objects."""
    persistent = num_workers > 0
    common_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": persistent,
    }
    train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        train_ds, shuffle=True, **common_kwargs
    )
    val_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        val_ds, shuffle=False, **common_kwargs
    )
    return train_loader, val_loader


def _train_epoch(
    model: GapImputer,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    masked_weight: float,
    observed_weight: float,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Run one training epoch.

    Returns
    -------
    tuple[float, float]
        ``(mean_combined_loss, mean_masked_loss)`` averaged over all batches.
    """
    model.train()
    total_combined = 0.0
    total_masked = 0.0
    n_batches = len(loader)

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [train]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in pbar:
        masked_strain: torch.Tensor = batch["masked_strain"].to(device, non_blocking=True)
        mask: torch.Tensor = batch["mask"].to(device, non_blocking=True)
        truth: torch.Tensor = batch["truth"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred: torch.Tensor = model(masked_strain, mask)
                loss: torch.Tensor = combined_loss(
                    pred, truth, mask,
                    masked_weight=masked_weight,
                    observed_weight=observed_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Only advance LR if optimizer.step() actually ran (scaler skips on overflow).
            if scaler.get_scale() >= scale_before:
                scheduler.step()
        else:
            pred = model(masked_strain, mask)
            loss = combined_loss(
                pred, truth, mask,
                masked_weight=masked_weight,
                observed_weight=observed_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            m_loss: torch.Tensor = masked_mse_loss(pred, truth, mask)

        combined_val = loss.item()
        masked_val = m_loss.item()
        total_combined += combined_val
        total_masked += masked_val

        pbar.set_postfix(
            combined=f"{combined_val:.4f}",
            masked=f"{masked_val:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    pbar.close()
    return total_combined / n_batches, total_masked / n_batches


@torch.no_grad()
def _val_epoch(
    model: GapImputer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    masked_weight: float,
    observed_weight: float,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Run one validation pass.

    Returns
    -------
    tuple[float, float]
        ``(mean_combined_loss, mean_masked_loss)`` averaged over all batches.
    """
    model.eval()
    total_combined = 0.0
    total_masked = 0.0
    n_batches = len(loader)

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [val]  ",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in pbar:
        masked_strain: torch.Tensor = batch["masked_strain"].to(device, non_blocking=True)
        mask: torch.Tensor = batch["mask"].to(device, non_blocking=True)
        truth: torch.Tensor = batch["truth"].to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred: torch.Tensor = model(masked_strain, mask)
        else:
            pred = model(masked_strain, mask)

        c_loss: torch.Tensor = combined_loss(
            pred, truth, mask,
            masked_weight=masked_weight,
            observed_weight=observed_weight,
        )
        m_loss: torch.Tensor = masked_mse_loss(pred, truth, mask)

        total_combined += c_loss.item()
        total_masked += m_loss.item()

        pbar.set_postfix(
            combined=f"{c_loss.item():.4f}",
            masked=f"{m_loss.item():.4f}",
        )

    pbar.close()
    return total_combined / n_batches, total_masked / n_batches


def _build_checkpoint(
    model: GapImputer,
    epoch: int,
    val_masked_loss: float,
    train_config: dict[str, Any],
    model_config: dict[str, Any],
    scale: float,
) -> dict[str, Any]:
    """Assemble a checkpoint dictionary."""
    return {
        "state_dict": model.state_dict(),
        "config": model_config,
        "epoch": epoch,
        "val_masked_loss": val_masked_loss,
        "train_config": train_config,
        "scale": scale,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train(
    out_dir: str | pathlib.Path,
    *,
    seq_len: int = DEFAULT_SEQ_LEN,
    fs_hz: float = DEFAULT_FS_HZ,
    n_train: int = DEFAULT_N_TRAIN,
    n_val: int = DEFAULT_N_VAL,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    masked_weight: float = DEFAULT_MASKED_WEIGHT,
    observed_weight: float = DEFAULT_OBSERVED_WEIGHT,
    patience: int = DEFAULT_PATIENCE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    patch_size: int = DEFAULT_PATCH_SIZE,
    d_model: int = DEFAULT_D_MODEL,
    nhead: int = DEFAULT_NHEAD,
    num_layers: int = DEFAULT_NUM_LAYERS,
    dropout: float = DEFAULT_DROPOUT,
    seed: int = DEFAULT_SEED,
    master_seed_train: int = DEFAULT_MASTER_SEED_TRAIN,
    master_seed_val: int = DEFAULT_MASTER_SEED_VAL,
    include_periodic_gap: bool = DEFAULT_INCLUDE_PERIODIC_GAP,
    signal_mix: dict[str, float] | None = None,
    device: str | None = None,
) -> pathlib.Path:
    """Train ``GapImputer`` and return the path to the best checkpoint.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        Directory where ``best.pt``, ``last.pt``, and ``history.json`` are
        written. Created if it does not exist.
    seq_len : int, optional
        Segment length in samples. Default 4096.
    fs_hz : float, optional
        Sampling frequency in Hz. Default 0.1.
    n_train : int, optional
        Number of training segments. Default 20 000.
    n_val : int, optional
        Number of validation segments. Default 2 000.
    epochs : int, optional
        Maximum number of training epochs. Default 40.
    batch_size : int, optional
        Mini-batch size for both loaders. Default 32.
    lr : float, optional
        Peak learning rate for OneCycleLR (also AdamW base lr). Default 3e-4.
    weight_decay : float, optional
        AdamW weight decay. Default 1e-4.
    masked_weight : float, optional
        Weight on the gap-position MSE term in ``combined_loss``. Default 1.0.
    observed_weight : float, optional
        Weight on the observed-position MSE term in ``combined_loss``. Default
        0.1.
    patience : int, optional
        Early-stopping patience in epochs (on ``val_masked_loss``). Default 7.
    num_workers : int, optional
        DataLoader worker processes. Default 2.
    patch_size : int, optional
        Conv-stem patch size (must divide ``seq_len``). Default 8.
    d_model : int, optional
        Transformer token dimension. Default 128.
    nhead : int, optional
        Number of attention heads. Default 4.
    num_layers : int, optional
        Number of transformer encoder layers. Default 5.
    dropout : float, optional
        Dropout probability inside the transformer. Default 0.1.
    seed : int, optional
        Global PyTorch/NumPy seed for reproducibility. Default 0.
    master_seed_train : int, optional
        Dataset master seed for the training split. Default 0.
    master_seed_val : int, optional
        Dataset master seed for the validation split. Default 1.
    include_periodic_gap : bool, optional
        Include the structured antenna-repointing gap in masks. Default True.
    signal_mix : dict[str, float] or None, optional
        Signal-type probability distribution. ``None`` uses the dataset
        default ``{"quiet": 0.3, "smbhb": 0.5, "monochromatic": 0.2}``.
    device : str or None, optional
        Target device string (e.g. ``"cuda"``, ``"cpu"``). ``None`` auto-
        detects CUDA.

    Returns
    -------
    pathlib.Path
        Absolute path to ``best.pt``.
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    out_path = pathlib.Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    _set_seeds(seed)

    dev = _resolve_device(device)
    use_amp: bool = dev.type == "cuda"

    torch.backends.cudnn.benchmark = True

    # Default signal mix (keep as None for build_splits to use its own default)
    _signal_mix: dict[str, float] = (
        signal_mix
        if signal_mix is not None
        else {"quiet": 0.3, "smbhb": 0.5, "monochromatic": 0.2}
    )

    # ------------------------------------------------------------------
    # Datasets and loaders
    # ------------------------------------------------------------------
    print(f"Building datasets  (n_train={n_train:,}, n_val={n_val:,}) …", flush=True)
    t0 = time.perf_counter()
    train_ds, val_ds, _ = build_splits(
        seq_len=seq_len,
        fs_hz=fs_hz,
        n_train=n_train,
        n_val=n_val,
        n_test=1,             # test split unused here; minimise build cost
        master_seed_train=master_seed_train,
        master_seed_val=master_seed_val,
        master_seed_test=999,
        signal_mix=_signal_mix,
        include_periodic_gap=include_periodic_gap,
    )
    scale: float = train_ds.scale
    print(
        f"  Done in {time.perf_counter() - t0:.1f}s  |  scale={scale:.4e}",
        flush=True,
    )

    train_loader, val_loader = _make_loaders(
        train_ds, val_ds, batch_size=batch_size, num_workers=num_workers
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_config: dict[str, Any] = {
        "seq_len": seq_len,
        "patch_size": patch_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dropout": dropout,
    }
    model = GapImputer(**model_config).to(dev)
    print(
        f"Model on {dev}  |  parameters: {model.count_parameters():,}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Optimiser, scheduler, scaler
    # ------------------------------------------------------------------
    optimizer: torch.optim.AdamW = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps: int = len(train_loader) * epochs
    scheduler: torch.optim.lr_scheduler.OneCycleLR = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    scaler: torch.amp.GradScaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ------------------------------------------------------------------
    # Training config dict (saved into checkpoints)
    # ------------------------------------------------------------------
    train_config: dict[str, Any] = {
        "seq_len": seq_len,
        "fs_hz": fs_hz,
        "n_train": n_train,
        "n_val": n_val,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "masked_weight": masked_weight,
        "observed_weight": observed_weight,
        "patience": patience,
        "num_workers": num_workers,
        "seed": seed,
        "master_seed_train": master_seed_train,
        "master_seed_val": master_seed_val,
        "include_periodic_gap": include_periodic_gap,
        "signal_mix": _signal_mix,
        "device": str(dev),
        "use_amp": use_amp,
    }

    best_pt = out_path / "best.pt"
    last_pt = out_path / "last.pt"
    history_json = out_path / "history.json"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_masked: float = float("inf")
    epochs_no_improve: int = 0
    history: list[dict[str, Any]] = []

    print(
        f"\nStarting training for up to {epochs} epochs  "
        f"(early-stop patience={patience} on val_masked_loss)\n",
        flush=True,
    )
    header = (
        f"{'Ep':>4}  {'train_loss':>10}  {'train_mask':>10}  "
        f"{'val_loss':>10}  {'val_mask':>10}  {'lr':>9}  {'time':>6}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()

        train_loss, train_masked = _train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=dev,
            masked_weight=masked_weight,
            observed_weight=observed_weight,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=epochs,
        )

        val_loss, val_masked = _val_epoch(
            model=model,
            loader=val_loader,
            device=dev,
            masked_weight=masked_weight,
            observed_weight=observed_weight,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=epochs,
        )

        current_lr: float = scheduler.get_last_lr()[0]
        elapsed: float = time.perf_counter() - epoch_t0

        # Per-epoch one-liner
        print(
            f"{epoch:>4}  {train_loss:>10.5f}  {train_masked:>10.5f}  "
            f"{val_loss:>10.5f}  {val_masked:>10.5f}  "
            f"{current_lr:>9.2e}  {elapsed:>5.1f}s",
            flush=True,
        )

        # History record
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_masked_loss": train_masked,
            "val_loss": val_loss,
            "val_masked_loss": val_masked,
            "lr": current_lr,
            "elapsed_s": elapsed,
        }
        history.append(epoch_record)

        # Save last checkpoint unconditionally
        torch.save(
            _build_checkpoint(
                model=model,
                epoch=epoch,
                val_masked_loss=val_masked,
                train_config=train_config,
                model_config=model_config,
                scale=scale,
            ),
            last_pt,
        )

        # Save best checkpoint and update early-stopping counter
        if val_masked < best_val_masked:
            best_val_masked = val_masked
            epochs_no_improve = 0
            torch.save(
                _build_checkpoint(
                    model=model,
                    epoch=epoch,
                    val_masked_loss=val_masked,
                    train_config=train_config,
                    model_config=model_config,
                    scale=scale,
                ),
                best_pt,
            )
            print(f"       -> new best val_masked_loss={best_val_masked:.5f}  (saved)", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"\nEarly stopping: val_masked_loss has not improved for "
                    f"{patience} consecutive epochs.",
                    flush=True,
                )
                break

        # Persist history after every epoch so a crash doesn't lose it
        with open(history_json, "w") as fh:
            json.dump(history, fh, indent=2)

    # Final history flush (redundant after loop but harmless)
    with open(history_json, "w") as fh:
        json.dump(history, fh, indent=2)

    print(
        f"\nTraining complete.  Best val_masked_loss={best_val_masked:.5f}\n"
        f"  best.pt  -> {best_pt}\n"
        f"  last.pt  -> {last_pt}\n"
        f"  history  -> {history_json}",
        flush=True,
    )

    return best_pt


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Command-line interface for ``train``.

    Example
    -------
    .. code-block:: bash

        python -m lisa_gap_imputer.train --out-dir runs/v1 --epochs 20
    """
    parser = argparse.ArgumentParser(
        description="Train the LISA gap-imputation transformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--out-dir", required=True, help="Output directory for checkpoints and history.")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Segment length in samples.")
    parser.add_argument("--fs-hz", type=float, default=DEFAULT_FS_HZ, help="Sampling frequency in Hz.")
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN, help="Number of training segments.")
    parser.add_argument("--n-val", type=int, default=DEFAULT_N_VAL, help="Number of validation segments.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum number of epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Peak learning rate.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay.")
    parser.add_argument("--masked-weight", type=float, default=DEFAULT_MASKED_WEIGHT, help="Gap-position loss weight.")
    parser.add_argument("--observed-weight", type=float, default=DEFAULT_OBSERVED_WEIGHT, help="Observed-position loss weight.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early-stopping patience (epochs).")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader worker processes.")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE, help="Conv-stem patch size.")
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL, help="Transformer token dimension.")
    parser.add_argument("--nhead", type=int, default=DEFAULT_NHEAD, help="Number of attention heads.")
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS, help="Number of encoder layers.")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout probability.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global RNG seed.")
    parser.add_argument("--master-seed-train", type=int, default=DEFAULT_MASTER_SEED_TRAIN, help="Dataset master seed (train).")
    parser.add_argument("--master-seed-val", type=int, default=DEFAULT_MASTER_SEED_VAL, help="Dataset master seed (val).")
    parser.add_argument(
        "--no-periodic-gap",
        dest="include_periodic_gap",
        action="store_false",
        default=DEFAULT_INCLUDE_PERIODIC_GAP,
        help="Disable the structured antenna-repointing gap.",
    )
    parser.add_argument(
        "--signal-mix",
        type=str,
        default=None,
        help=(
            'JSON string for signal-type probabilities, e.g. '
            '\'{"quiet":0.3,"smbhb":0.5,"monochromatic":0.2}\'. '
            "Defaults to the dataset module default."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help='Device string (e.g. "cuda", "cpu"). Auto-detected if omitted.')

    args = parser.parse_args(argv)

    signal_mix: dict[str, float] | None = None
    if args.signal_mix is not None:
        signal_mix = json.loads(args.signal_mix)

    train(
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        fs_hz=args.fs_hz,
        n_train=args.n_train,
        n_val=args.n_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        masked_weight=args.masked_weight,
        observed_weight=args.observed_weight,
        patience=args.patience,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        master_seed_train=args.master_seed_train,
        master_seed_val=args.master_seed_val,
        include_periodic_gap=args.include_periodic_gap,
        signal_mix=signal_mix,
        device=args.device,
    )


if __name__ == "__main__":
    main()
