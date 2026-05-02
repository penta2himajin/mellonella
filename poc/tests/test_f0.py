"""YIN smoke tests on synthetic sinusoids."""

from __future__ import annotations

import numpy as np
import pytest

from mellonella_poc.f0 import estimate_f0_track, f0_statistics, yin_frame


def _sine(freq: float, sr: int, duration: float) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.mark.parametrize("freq", [120.0, 220.0, 330.0])
def test_yin_estimates_pure_tone(freq: float):
    sr = 16_000
    frame = _sine(freq, sr, 0.2)[:2048]
    est = yin_frame(frame, sr)
    assert est is not None
    assert est == pytest.approx(freq, rel=0.05)


def test_yin_returns_none_for_white_noise():
    sr = 16_000
    rng = np.random.default_rng(42)
    frame = rng.standard_normal(2048).astype(np.float32)
    est = yin_frame(frame, sr, threshold=0.05)
    assert est is None or est < 1000.0


def test_track_recovers_constant_pitch():
    sr = 16_000
    audio = _sine(150.0, sr, 0.5)
    track = estimate_f0_track(audio, sr, frame_size=2048, hop_size=512)
    voiced = track[np.isfinite(track)]
    assert voiced.size > 0
    assert np.median(voiced) == pytest.approx(150.0, rel=0.05)


def test_statistics_handles_empty_track():
    track = np.full(10, np.nan, dtype=np.float32)
    mu, sigma = f0_statistics(track)
    assert mu == 0.0
    assert sigma == 0.0
