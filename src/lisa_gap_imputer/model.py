"""Transformer-encoder gap imputer for LISA-like strain time series.

This module implements ``GapImputer``, a small patch-based transformer encoder that
takes a masked 1-D strain segment and reconstructs the missing samples. The design
follows the "compact convolutional transformer" philosophy (Hassani et al. 2021): a
strided 1-D convolutional stem first maps overlapping raw samples into a shorter
sequence of patch tokens; a standard pre-layer-norm transformer encoder then attends
over all tokens from both sides of every gap; finally a linear head unpacks each token
back into contiguous time samples, recovering the original resolution.

Two module-level loss functions are also provided:

- ``masked_mse_loss`` — MSE computed only on the gap positions (the primary training
  signal).
- ``combined_loss`` — a weighted combination of the masked-position MSE and a lighter
  reconstruction term on the observed positions, encouraging the model to pass through
  clean data cleanly without over-weighting the (typically smaller) gap fraction.

**Architecture at a glance**

.. code-block:: none

    Input  (B, 2, L)          ← channel-0: zero-filled strain; channel-1: binary mask
        │
    Conv1d stem               ← kernel_size = stride = patch_size  → (B, d_model, L//P)
        │
    Transpose                 → (B, L//P, d_model)
        │
    Sinusoidal PE             → added in-place; preserves shape
        │
    TransformerEncoder        ← pre-LN (norm_first=True), GELU, batch_first=True
        │  num_layers × (MultiheadSA + FFN), d_model, nhead
        │
    Linear head               → (B, L//P, patch_size)
        │
    Reshape                   → (B, L)   ← contiguous; innermost dim = patch samples
    Output (B, L)

References
----------
Hassani, A., Walton, S., Shah, N., Abuduweili, A., Li, J., & Shi, H. (2021).
    Escaping the Big Data Paradigm with Compact Transformers. arXiv:2104.05704.
Vaswani, A., Shazeer, N., Parmar, N., et al. (2017).
    Attention Is All You Need. arXiv:1706.03762.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = [
    "SinusoidalPositionalEncoding",
    "GapImputer",
    "masked_mse_loss",
    "combined_loss",
]

# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding as in Vaswani et al. (2017).

    A ``(max_len, d_model)`` table is pre-computed once in ``__init__`` and
    stored as a non-parameter buffer. During ``forward`` the first ``T`` rows are
    sliced and broadcast-added to the input sequence, so no gradient flows
    through the encoding itself.

    Parameters
    ----------
    d_model : int
        Token embedding dimension. Must be even (the encoding pairs sine and
        cosine for each half of the dimension).
    max_len : int
        Maximum sequence length the table is pre-built for. Any ``forward``
        call with ``T <= max_len`` is supported; ``T > max_len`` raises an
        ``IndexError``.
    dropout : float, optional
        Dropout probability applied *after* adding the positional encoding.
        Matches the convention in the original paper. Default ``0.0`` (no
        dropout; the training dropout lives inside ``TransformerEncoderLayer``).

    Notes
    -----
    **Why sinusoidal rather than learned?**
    A learned table requires ``max_len × d_model`` additional parameters and,
    more importantly, cannot generalise to sequence lengths longer than those
    seen during training. The gap-imputation dataset is synthetic, so segment
    lengths could vary across experiments; sinusoidal encoding handles arbitrary
    ``T ≤ max_len`` with zero extra parameters and no re-training. The
    empirical gap between learned and sinusoidal encodings is small for the
    length scales used here (≤ 512 tokens after the conv stem).
    """

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build table: position indices along rows, dimension indices along columns.
        position: torch.Tensor = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        # Denominators: 10000^(2i/d_model) for i = 0, 1, ..., d_model//2 - 1.
        div_term: torch.Tensor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe: torch.Tensor = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)           # even dims  → sine
        pe[:, 1::2] = torch.cos(position * div_term)           # odd  dims  → cosine

        # Register as a buffer so it moves to the correct device with `.to()` /
        # `.cuda()` but is excluded from `model.parameters()`.
        self.register_buffer("pe", pe)  # shape (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal encoding to a batch of token sequences.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, d_model)``. Modified in-place via broadcasting.

        Returns
        -------
        torch.Tensor
            Same shape as *x*, with positional encoding added (and optional
            dropout applied).
        """
        # pe is (max_len, d_model); slice to (T, d_model), then broadcast over B.
        x = x + self.pe[: x.size(1)]  # type: ignore[index]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class GapImputer(nn.Module):
    """Patch-based transformer encoder for 1-D gap imputation.

    Takes a zero-filled masked strain segment together with a binary mask and
    returns a full-resolution reconstructed strain. The architecture uses a
    strided convolutional stem to tokenise the raw sequence into non-overlapping
    patches, a sinusoidal positional encoding, a stack of pre-layer-norm
    transformer encoder blocks, and a linear head that unpacks each token back
    into contiguous time samples.

    Parameters
    ----------
    seq_len : int, optional
        Expected input length ``L``. Used only for validation in ``forward``;
        the model itself places no hard constraint on ``L`` beyond divisibility
        by ``patch_size``. Default ``4096``.
    patch_size : int, optional
        Convolutional kernel size *and* stride. Must divide ``L`` exactly.
        After the stem the sequence length becomes ``L // patch_size``.
        Default ``8``.
    d_model : int, optional
        Internal token embedding dimension (number of output channels of the
        conv stem, and width of the transformer layers). Default ``128``.
    nhead : int, optional
        Number of attention heads. Must divide ``d_model``. Default ``4``.
    num_layers : int, optional
        Number of ``TransformerEncoderLayer`` blocks. Default ``5``.
    dropout : float, optional
        Dropout probability passed to each ``TransformerEncoderLayer``. Does
        **not** apply to the conv stem or the positional encoding. Default
        ``0.1``.
    max_len : int, optional
        Maximum sequence length in *tokens* (i.e. after the conv stem,
        ``max_len = max_input_length // patch_size``). The sinusoidal positional
        encoding table is pre-built to this size. Default ``8192``.

    Notes
    -----
    **Conv stem (tokenisation)**
    Using a single strided ``Conv1d`` with ``kernel_size == stride == patch_size``
    is equivalent to a non-overlapping patch projection identical in spirit to
    the patch embedding used in Vision Transformers (Dosovitskiy et al. 2021).
    It is preferable to a learnable embedding table because it can handle
    arbitrary input lengths and leverages local inductive bias — nearby raw
    samples within a patch are mixed before attention ever runs, which is
    especially helpful when the gap width is smaller than the patch size.

    **Pre-layer-norm (norm_first=True)**
    The "pre-LN" variant normalises inside the residual branch before the
    sub-layer computation rather than after (the original "post-LN" of Vaswani
    et al. 2017). Pre-LN is empirically more stable during early training
    (Xiong et al. 2020, arXiv:2002.04745) and eliminates the need for a
    learning-rate warm-up, which matters here because the dataset is small and
    training should be forgiving of hyperparameter choices.

    **Linear head with reshape**
    The output head is a single ``Linear(d_model, patch_size)`` applied per
    token, yielding shape ``(B, L//P, P)``. A call to ``.reshape(B, L)`` then
    packs the patches contiguously: for token ``k`` the ``P`` output values
    become samples ``k*P : (k+1)*P`` in the output sequence. This ordering is
    guaranteed by PyTorch's C-contiguous memory layout — the last dimension is
    innermost, so reshaping flattens the ``(L//P, P)`` patch grid in row-major
    order (first over tokens, then within a token's patch), which is exactly the
    desired left-to-right time ordering.

    **No batch normalisation**
    Batch norm requires a minimum batch size to give well-estimated statistics
    and interacts poorly with variable-length masking patterns (each segment
    within a batch has a different fraction of masked samples). Layer norm,
    which is applied inside each ``TransformerEncoderLayer``, normalises over
    the ``d_model`` dimension per token per sample and is unaffected by batch
    composition.
    """

    def __init__(
        self,
        seq_len: int = 4096,
        patch_size: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 5,
        dropout: float = 0.1,
        max_len: int = 8192,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.patch_size = patch_size
        self.d_model = d_model

        # ------------------------------------------------------------------
        # Conv stem: maps (B, 2, L) → (B, d_model, L // patch_size).
        # Two input channels: channel-0 = zero-filled strain, channel-1 = mask.
        # kernel_size == stride → non-overlapping patch projection.
        # ------------------------------------------------------------------
        self.conv_stem = nn.Conv1d(
            in_channels=2,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        nn.init.xavier_uniform_(self.conv_stem.weight)
        nn.init.zeros_(self.conv_stem.bias)  # type: ignore[arg-type]

        # ------------------------------------------------------------------
        # Sinusoidal positional encoding: no parameters.
        # ------------------------------------------------------------------
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

        # ------------------------------------------------------------------
        # Transformer encoder stack (pre-LN, GELU, batch_first).
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            norm_first=True,       # pre-LN for training stability
            batch_first=True,      # input/output shape (B, T, d_model)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # ------------------------------------------------------------------
        # Output head: (B, T, d_model) → (B, T, patch_size), then reshape to
        # (B, L). Xavier init; zero bias.
        # ------------------------------------------------------------------
        self.head = nn.Linear(d_model, patch_size)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, strain: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reconstruct a full-resolution strain segment.

        Parameters
        ----------
        strain : torch.Tensor
            Shape ``(B, L)``, float32. The caller must zero-fill positions
            covered by the gap mask before passing to this method (i.e. already
            ``strain * (1 - mask)``).
        mask : torch.Tensor
            Shape ``(B, L)``, float32. Values in ``{0.0, 1.0}``:
            ``1.0`` at masked (gap) positions, ``0.0`` at observed positions.

        Returns
        -------
        torch.Tensor
            Shape ``(B, L)``, float32. Full-resolution strain reconstruction.
            The model outputs a value at *every* position; the loss function
            (:func:`masked_mse_loss` or :func:`combined_loss`) selects which
            positions are penalised.

        Raises
        ------
        AssertionError
            If ``L`` is not divisible by ``patch_size``.
        """
        B, L = strain.shape

        assert L % self.patch_size == 0, (
            f"Input length L={L} must be divisible by patch_size={self.patch_size}. "
            f"Received a sequence whose length is not an integer multiple of the patch "
            f"size; either pad the input to the next multiple or choose a patch_size "
            f"that divides L."
        )

        # Stack channels: (B, 2, L).
        x: torch.Tensor = torch.stack([strain, mask], dim=1)  # (B, 2, L)

        # Conv stem → (B, d_model, L // patch_size).
        x = self.conv_stem(x)  # (B, d_model, T)  where T = L // patch_size

        # Transformer expects (B, T, d_model).
        x = x.transpose(1, 2)  # (B, T, d_model)

        # Add sinusoidal positional encoding (no parameters, no gradient).
        x = self.pos_enc(x)    # (B, T, d_model)

        # Transformer encoder: T tokens attend to each other.
        x = self.transformer(x)  # (B, T, d_model)

        # Linear head: project each token to patch_size scalars.
        x = self.head(x)       # (B, T, patch_size)

        # Reshape to full resolution.
        # x is C-contiguous with shape (B, T, P); reshape to (B, T*P) = (B, L).
        # PyTorch's C-contiguous row-major ordering guarantees that token k's P
        # values occupy output positions k*P : (k+1)*P, which is the correct
        # left-to-right temporal ordering. No copy is needed for a contiguous tensor.
        out: torch.Tensor = x.reshape(B, L)  # (B, L)

        return out

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters.

        Returns
        -------
        int
            Sum of ``p.numel()`` over all parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error evaluated only on the masked (gap) positions.

    Parameters
    ----------
    pred : torch.Tensor
        Model output, shape ``(B, L)``, float32.
    target : torch.Tensor
        Ground-truth clean strain, shape ``(B, L)``, float32.
    mask : torch.Tensor
        Binary gap mask, shape ``(B, L)``, float32. ``1.0`` at gap positions,
        ``0.0`` at observed positions.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor. Returns ``0.0`` (as a tensor) if ``mask`` is
        all-zero (no gap positions), which avoids NaN from a zero denominator.

    Notes
    -----
    The denominator is ``mask.sum()`` (the total number of masked samples across
    the batch), clamped to 1.0 from below so an empty mask does not cause a
    division by zero. This is the primary loss used for the gap-imputation task
    and is the only term that directly supervises the model on missing data.
    """
    loss: torch.Tensor = (
        ((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
    )
    return loss


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    masked_weight: float = 1.0,
    observed_weight: float = 0.1,
) -> torch.Tensor:
    """Weighted combination of gap-MSE and observed-position reconstruction loss.

    The primary term penalises reconstruction error in the gap (masked) positions.
    The secondary term penalises any distortion the model introduces in the
    observed (unmasked) positions — the model should pass clean data through
    cleanly. The default weighting (1.0 : 0.1) keeps the imputation task dominant
    while discouraging the model from hallucinating or smoothing clean samples.

    Parameters
    ----------
    pred : torch.Tensor
        Model output, shape ``(B, L)``, float32.
    target : torch.Tensor
        Ground-truth clean strain, shape ``(B, L)``, float32.
    mask : torch.Tensor
        Binary gap mask, shape ``(B, L)``, float32. ``1.0`` at gap positions,
        ``0.0`` at observed positions.
    masked_weight : float, optional
        Scalar multiplier for the gap-position MSE term. Default ``1.0``.
    observed_weight : float, optional
        Scalar multiplier for the observed-position MSE term. Default ``0.1``.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor.

    Notes
    -----
    Both terms are normalised by their respective sample counts (clamped to 1)
    so the weighting ratio is interpretable independently of the mask fraction.
    If the mask covers half the segment, neither term dominates the other purely
    due to sample count imbalance — only the explicit weights do.
    """
    observed: torch.Tensor = 1.0 - mask

    masked_loss: torch.Tensor = (
        ((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
    )
    observed_loss: torch.Tensor = (
        ((pred - target) ** 2 * observed).sum() / observed.sum().clamp_min(1.0)
    )
    return masked_weight * masked_loss + observed_weight * observed_loss


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _BATCH = 2
    _SEQ_LEN = 4096

    model = GapImputer()
    model.eval()

    _strain = torch.randn(_BATCH, _SEQ_LEN)
    _mask = torch.zeros(_BATCH, _SEQ_LEN)
    # Mask a synthetic central gap of 512 samples.
    _mask[:, 1792:2304] = 1.0
    _strain = _strain * (1.0 - _mask)  # zero-fill masked positions

    with torch.no_grad():
        _out = model(_strain, _mask)

    print(f"Output shape : {tuple(_out.shape)}")          # expect (2, 4096)
    print(f"Parameters   : {model.count_parameters():,}")  # expect ~995 k with defaults
