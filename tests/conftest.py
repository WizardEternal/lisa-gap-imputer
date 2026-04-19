from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded NumPy default_rng for deterministic test runs."""
    return np.random.default_rng(0)


@pytest.fixture
def short_seq_len() -> int:
    """Short segment length so tests run fast."""
    return 256
