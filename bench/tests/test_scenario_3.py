"""Tests for the alternating-speech scenario (Scenario 3)."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import soundfile as sf

from mellonella_bench.scenarios.base import StubPipelineProvider
from mellonella_bench.scenarios.scenario_3 import (
    DEFAULT_SEGMENTS,
    Scenario3Item,
    assemble_audio,
    evaluate_one,
    latencies_per_transition,
    run,
    voicing_mask_at_frame_rate,
)


def _sine(freq: float, sr: int, duration: float) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_assemble_audio_lengths():
    sr = 16_000
    target = _sine(220.0, sr, 5.0)
    other = _sine(330.0, sr, 5.0)
    segments = (("target", 1.0), ("silence", 0.5), ("other", 1.0))
    audio = assemble_audio(segments, target, other, sr)
    expected = int(1.0 * sr) + int(0.5 * sr) + int(1.0 * sr)
    assert audio.size == expected
    # silence should be zero
    silence = audio[sr : sr + sr // 2]
    assert np.allclose(silence, 0.0)


def test_assemble_audio_tiles_short_source():
    sr = 16_000
    target = _sine(220.0, sr, 0.5)  # only 0.5s available
    other = _sine(330.0, sr, 0.5)
    segments = (("target", 2.0),)
    audio = assemble_audio(segments, target, other, sr)
    assert audio.size == 2 * sr


def test_assemble_audio_rejects_unknown_label():
    sr = 16_000
    with pytest.raises(ValueError):
        assemble_audio(
            (("noise", 1.0),),  # type: ignore[arg-type]
            _sine(220.0, sr, 1.0),
            _sine(330.0, sr, 1.0),
            sr,
        )


def test_voicing_mask_alignment():
    sr = 16_000
    samples_per_frame = 512  # 32 ms
    segments = (("target", 1.0), ("silence", 0.5), ("other", 1.0))
    target_mask, label_codes = voicing_mask_at_frame_rate(segments, sr, samples_per_frame)
    # 1 s = 31 full frames (15872 samples), 0.5 s = 15 frames, 1 s = 31 frames
    assert target_mask.size == 31 + 15 + 31
    assert label_codes.size == target_mask.size
    assert target_mask[:31].all()
    assert not target_mask[31:46].any()
    assert not target_mask[46:].any()
    np.testing.assert_array_equal(label_codes[:31], 1)
    np.testing.assert_array_equal(label_codes[31:46], 0)
    np.testing.assert_array_equal(label_codes[46:], 2)


def test_latencies_perfect_prediction_gives_zero():
    target_mask = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
    onset, offset = latencies_per_transition(target_mask, target_mask, frame_ms=20.0)
    assert onset == [0.0, 0.0]
    assert offset == [0.0, 0.0]


def test_latencies_delayed_prediction():
    target_mask = np.array([0, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    pred = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=bool)
    # target run is [1..5). pred goes True at idx 3 = 2 frames late.
    # target ends at idx 5; pred goes False at idx 7 = 2 frames late.
    onset, offset = latencies_per_transition(target_mask, pred, frame_ms=10.0)
    assert onset == [20.0]
    assert offset == [20.0]


def test_latencies_skip_when_prediction_never_catches_up():
    target_mask = np.array([0, 1, 1, 0], dtype=bool)
    pred = np.zeros_like(target_mask)
    onset, offset = latencies_per_transition(target_mask, pred, frame_ms=10.0)
    assert onset == []
    # offset latency is "first False after target ends" — pred is already
    # False at index 3, so offset = 0.
    assert offset == [0.0]


def test_latencies_shape_check():
    with pytest.raises(ValueError):
        latencies_per_transition(np.array([1]), np.array([1, 0]), frame_ms=10.0)


def test_evaluate_one_with_stub(tmp_path):
    sr = 16_000
    target_path = tmp_path / "target.wav"
    other_path = tmp_path / "other.wav"
    sf.write(str(target_path), _sine(220.0, sr, 6.0), sr)
    sf.write(str(other_path), _sine(330.0, sr, 6.0), sr)

    item = Scenario3Item(
        sample_id="utt_alt_001",
        target_path=target_path,
        other_path=other_path,
        segments=(("target", 1.0), ("other", 1.0), ("target", 1.0)),
    )
    entry = evaluate_one(item, StubPipelineProvider(), sample_rate=sr)
    # Stub gate is always True. So TPR=1, FPR=1, frame_accuracy = ratio of target frames.
    assert entry.gate_tpr == pytest.approx(1.0)
    assert entry.gate_fpr == pytest.approx(1.0)
    assert entry.gate_tnr == pytest.approx(0.0)
    # 2 target segments + 1 other → roughly 2/3 frames are target
    assert 0.5 < (entry.frame_accuracy or 0) < 0.8
    # Onset latencies should be 0 (stub passes immediately); two target runs
    assert entry.onset_latency_ms == pytest.approx(0.0)


def test_run_emits_csv(tmp_path):
    sr = 16_000
    target_path = tmp_path / "target.wav"
    other_path = tmp_path / "other.wav"
    sf.write(str(target_path), _sine(220.0, sr, 6.0), sr)
    sf.write(str(other_path), _sine(330.0, sr, 6.0), sr)

    items = [
        Scenario3Item(
            sample_id="utt_a",
            target_path=target_path,
            other_path=other_path,
            segments=(("target", 1.0), ("other", 1.0)),
        )
    ]
    csv_path = tmp_path / "scenario_3.csv"
    result = run(items, StubPipelineProvider(), sample_rate=sr, output_csv=csv_path)
    assert result.scenario == "scenario_3"
    assert result.n_samples == 1
    assert "frame_accuracy_mean" in result.metrics

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["scenario"] == "scenario_3"
    assert row["sample_id"] == "utt_a"
    assert row["frame_accuracy"] != ""
    assert row["snr_db"] == ""  # not applicable for scenario 3


def test_default_segments_alternation():
    """Sanity-check the docs-aligned default pattern."""
    labels = [label for label, _ in DEFAULT_SEGMENTS]
    assert labels.count("target") >= 2
    assert labels.count("other") >= 2
    assert "silence" in labels


def test_run_rejects_sample_rate_mismatch(tmp_path):
    sr = 16_000
    target_path = tmp_path / "target.wav"
    other_path = tmp_path / "other.wav"
    sf.write(str(target_path), _sine(220.0, sr, 1.0), sr)
    sf.write(str(other_path), _sine(330.0, 8_000, 1.0), 8_000)
    item = Scenario3Item(
        sample_id="bad",
        target_path=target_path,
        other_path=other_path,
        segments=(("target", 0.5), ("other", 0.5)),
    )
    with pytest.raises(ValueError):
        run([item], StubPipelineProvider(), sample_rate=sr)
