"""lisa-gap-imputer: transformer-based reconstruction of masked LISA strain segments.

This package is an exploratory exercise in applying a small transformer encoder to the
gap-imputation problem for LISA-like time series, benchmarked against interpolation and
Gaussian-process baselines. It is not a production LISA data-analysis tool; for that,
see the LISA Data Challenge software (LDC), ``lisabeta``, ``balrog``, and related
packages maintained by the LISA consortium.

Modules
-------
noise
    Colored Gaussian noise generation from the Robson+19 LISA sensitivity curve.
signals
    SMBHB chirp and monochromatic galactic-binary injections.
masks
    Periodic (antenna repointing) and Poisson-stochastic (micrometeorite) gap
    pattern sampling.
dataset
    ``torch.utils.data.Dataset`` yielding (masked_strain, mask, truth) triples.
baselines
    Zero-fill, linear interpolation, cubic spline, and Gaussian-process imputers.
model
    Conv-stem + transformer-encoder + upsampling-head imputer.
train
    AdamW + OneCycleLR + AMP training loop with early stopping.
evaluate
    Reconstruction metrics and matched-filter SNR recovery.
plotting
    Figures for reconstruction error vs gap duration and SNR recovery fraction.
"""

from __future__ import annotations

__version__ = "0.1.0"
