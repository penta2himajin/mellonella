"""Smoke tests for ``mellonella_bench.runners.run_tse_dnsmos``.

The runner consumes external ONNX models (TSE + DNSMOS) and a prepared
audio fixture; we do not exercise those code paths in CI. These tests
cover only the pure-numpy helpers (mixing, SI-SDR, dither / state
plumbing math) so that refactors to the runner do not silently break
the metric definitions or scenario layout.
"""

from __future__ import annotations

import numpy as np
import pytest

from mellonella_bench.runners import run_tse_dnsmos as r


def test_si_sdr_perfect_reconstruction_is_high():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(48_000).astype(np.float32)
    # 100 dB ceiling guards against future inf returns from the formula.
    assert r.si_sdr(s, s) > 100.0


def test_si_sdr_orthogonal_noise_is_low():
    rng = np.random.default_rng(1)
    s = rng.standard_normal(48_000).astype(np.float32)
    n = rng.standard_normal(48_000).astype(np.float32)
    assert r.si_sdr(s, n) < 0.0


def test_mix_at_db_sets_relative_rms():
    rng = np.random.default_rng(2)
    s1 = rng.standard_normal(10_000).astype(np.float32)
    s2 = rng.standard_normal(10_000).astype(np.float32)
    scaled = r._mix_at_db(s1, s2, 6.0)
    rms1 = float(np.sqrt(np.mean(s1**2)))
    rms_scaled = float(np.sqrt(np.mean(scaled.astype(np.float64) ** 2)))
    # s2 is now 6 dB below s1.
    assert rms_scaled == pytest.approx(rms1 / 2.0, rel=0.05)


def test_align_pads_short_and_truncates_long():
    short = np.ones(100, dtype=np.float32)
    long = np.ones(300, dtype=np.float32)
    assert r._align(short, 200).shape == (200,)
    assert r._align(long, 200).shape == (200,)


def test_build_scenarios_lays_out_four_named_mixtures():
    rng = np.random.default_rng(3)
    t = rng.standard_normal(r.CHUNK * 10).astype(np.float32)
    ifr = rng.standard_normal(r.CHUNK * 10).astype(np.float32)
    ns = rng.standard_normal(r.CHUNK * 10).astype(np.float32)
    scenarios = r._build_scenarios(t, ifr, ns)
    names = [sc.name for sc in scenarios]
    assert names == ["A_clean", "B_t_noise_10dB", "C_t_inter_0dB", "D_t_inter_n_5dB"]
    for sc in scenarios:
        # Every scenario must be a multiple of the TSE chunk so the
        # streaming inference loop drains cleanly.
        assert len(sc.mixture) % r.CHUNK == 0
        assert len(sc.reference) == len(sc.mixture)


def test_clip_for_overload_keeps_under_unity():
    x = np.array([0.1, -0.2, 1.5, 0.3, -2.0], dtype=np.float32)
    clipped = r._clip_for_overload(x)
    assert float(np.max(np.abs(clipped))) <= 0.95


def test_clip_for_overload_passes_through_quiet_signal():
    x = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    out = r._clip_for_overload(x)
    assert np.allclose(out, x)


def test_cli_requires_paths():
    parser = r._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
