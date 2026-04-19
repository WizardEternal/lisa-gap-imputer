from __future__ import annotations

import numpy as np
import pytest
from numpy.random import default_rng

from lisa_gap_imputer.signals import (
    SignalSegment,
    draw_monochromatic_params,
    draw_smbhb_params,
    inject_monochromatic,
    inject_smbhb_chirp,
)


# ---------------------------------------------------------------------------
# inject_smbhb_chirp
# ---------------------------------------------------------------------------


def test_smbhb_chirp_shape() -> None:
    """inject_smbhb_chirp must return a SignalSegment with the right shape and dtype."""
    seg = inject_smbhb_chirp(n_samples=512, fs_hz=0.1, rng=default_rng(0))
    assert isinstance(seg, SignalSegment)
    assert seg.strain.shape == (512,)
    assert seg.strain.dtype == np.float64
    assert np.all(np.isfinite(seg.strain)), "SMBHB chirp strain has NaNs or infs"


def test_smbhb_chirp_reproducible() -> None:
    """Same seed + explicit params + merger_position must give bit-identical strain."""
    rng_a = default_rng(7)
    params = draw_smbhb_params(rng_a)
    seg_a = inject_smbhb_chirp(
        n_samples=256, fs_hz=0.1, rng=default_rng(7), params=params, merger_position=0.5
    )
    seg_b = inject_smbhb_chirp(
        n_samples=256, fs_hz=0.1, rng=default_rng(7), params=params, merger_position=0.5
    )
    np.testing.assert_array_equal(seg_a.strain, seg_b.strain)


# ---------------------------------------------------------------------------
# inject_monochromatic
# ---------------------------------------------------------------------------


def test_monochromatic_shape_and_finite() -> None:
    """inject_monochromatic must return shape (n_samples,), float64, all finite."""
    seg = inject_monochromatic(n_samples=512, fs_hz=0.1, rng=default_rng(3))
    assert isinstance(seg, SignalSegment)
    assert seg.strain.shape == (512,)
    assert seg.strain.dtype == np.float64
    assert np.all(np.isfinite(seg.strain)), "Monochromatic strain has NaNs or infs"


# ---------------------------------------------------------------------------
# draw_smbhb_params
# ---------------------------------------------------------------------------


def test_draw_smbhb_params_in_range() -> None:
    """Every parameter returned by draw_smbhb_params must lie in documented ranges.

    Ranges match the defaults of draw_smbhb_params:
        m_total_msun in [1e5, 1e7]
        q in [0.1, 1.0]
        d_l_mpc in [500, 10000]
        iota_rad in [0, pi]
    """
    rng_p = default_rng(0)
    for _ in range(100):
        p = draw_smbhb_params(rng_p)
        assert 1e5 <= p["m_total_msun"] <= 1e7, f"m_total_msun out of range: {p['m_total_msun']}"
        assert 0.1 <= p["q"] <= 1.0, f"q out of range: {p['q']}"
        assert 500.0 <= p["d_l_mpc"] <= 1e4, f"d_l_mpc out of range: {p['d_l_mpc']}"
        assert 0.0 <= p["iota_rad"] <= np.pi, f"iota_rad out of range: {p['iota_rad']}"
        # Sanity: derived component masses must be positive and sum to m_total
        assert p["m1_msun"] > 0 and p["m2_msun"] > 0
        assert abs(p["m1_msun"] + p["m2_msun"] - p["m_total_msun"]) < 1.0


# ---------------------------------------------------------------------------
# params override
# ---------------------------------------------------------------------------


def test_smbhb_params_override() -> None:
    """Passing two different param dicts must produce different strains."""
    rng_base = default_rng(0)
    params_a = draw_smbhb_params(default_rng(10))
    params_b = draw_smbhb_params(default_rng(20))

    seg_a = inject_smbhb_chirp(
        n_samples=256, fs_hz=0.1, rng=rng_base, params=params_a, merger_position=0.5
    )
    seg_b = inject_smbhb_chirp(
        n_samples=256, fs_hz=0.1, rng=rng_base, params=params_b, merger_position=0.5
    )

    # Different physical parameters must produce different waveforms.
    assert not np.array_equal(seg_a.strain, seg_b.strain), (
        "Different param dicts produced identical strain — override not respected"
    )

    # Meta should record the correct params.
    if hasattr(seg_a, "meta") and seg_a.meta is not None:
        assert seg_a.meta.params == params_a
        assert seg_b.meta.params == params_b
