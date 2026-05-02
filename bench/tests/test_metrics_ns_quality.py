"""Tests for noise-suppression quality metrics."""

from __future__ import annotations

import numpy as np
import pytest

from mellonella_bench.metrics.ns_quality import si_sdr


def _sine(freq: float, sr: int, duration: float) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_si_sdr_identical_signals_is_high():
    sr = 16_000
    s = _sine(220.0, sr, 0.5)
    score = si_sdr(s, s.copy())
    assert score > 100.0


def test_si_sdr_inverted_signal_is_high():
    sr = 16_000
    s = _sine(220.0, sr, 0.5)
    score = si_sdr(s, -s)
    # SI-SDR is sign-invariant by construction.
    assert score > 100.0


def test_si_sdr_decreases_with_added_noise():
    rng = np.random.default_rng(0)
    sr = 16_000
    s = _sine(220.0, sr, 0.5)
    quiet = s + 0.01 * rng.standard_normal(s.size).astype(np.float32)
    loud = s + 0.5 * rng.standard_normal(s.size).astype(np.float32)
    assert si_sdr(s, quiet) > si_sdr(s, loud)


def test_si_sdr_scale_invariant():
    sr = 16_000
    s = _sine(220.0, sr, 0.5)
    estimate = 4.2 * s
    assert si_sdr(s, estimate) > 100.0


def test_si_sdr_shape_mismatch():
    s = _sine(220.0, 16_000, 0.5)
    with pytest.raises(ValueError):
        si_sdr(s, s[:-1])


def test_si_sdr_empty_signal_raises():
    with pytest.raises(ValueError):
        si_sdr(np.array([]), np.array([]))
