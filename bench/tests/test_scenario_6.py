"""Tests for the drift-verification scenario (Scenario 6)."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import soundfile as sf

from mellonella_bench.scenarios.base import StubPipelineProvider
from mellonella_bench.scenarios.scenario_6 import (
    Scenario6Item,
    assemble_track,
    evaluate_one,
    run,
)


def _sine(freq: float, sr: int, duration: float) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_assemble_track_concatenates_full_variants(tmp_path):
    sr = 16_000
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sf.write(str(a), _sine(220.0, sr, 1.0), sr)
    sf.write(str(b), _sine(220.0, sr, 0.5), sr)
    item = Scenario6Item(
        sample_id="t",
        enrollment_path=a,
        variant_paths=(a, b),
    )
    audio = assemble_track(item, sr)
    assert audio.size == int(1.0 * sr) + int(0.5 * sr)


def test_assemble_track_truncates_per_variant(tmp_path):
    sr = 16_000
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sf.write(str(a), _sine(220.0, sr, 2.0), sr)
    sf.write(str(b), _sine(220.0, sr, 2.0), sr)
    item = Scenario6Item(
        sample_id="t",
        enrollment_path=a,
        variant_paths=(a, b),
        variant_durations_sec=(0.5, 1.0),
    )
    audio = assemble_track(item, sr)
    assert audio.size == int(0.5 * sr) + int(1.0 * sr)


def test_assemble_track_partial_durations_default_to_full(tmp_path):
    sr = 16_000
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sf.write(str(a), _sine(220.0, sr, 1.0), sr)
    sf.write(str(b), _sine(220.0, sr, 1.0), sr)
    item = Scenario6Item(
        sample_id="t",
        enrollment_path=a,
        variant_paths=(a, b),
        variant_durations_sec=(0.25,),  # only first variant truncated
    )
    audio = assemble_track(item, sr)
    assert audio.size == int(0.25 * sr) + sr


def test_assemble_track_rejects_empty_variants(tmp_path):
    item = Scenario6Item(
        sample_id="t",
        enrollment_path=tmp_path / "a.wav",
        variant_paths=(),
    )
    with pytest.raises(ValueError):
        assemble_track(item, 16_000)


def test_assemble_track_rejects_sample_rate_mismatch(tmp_path):
    sr = 16_000
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sf.write(str(a), _sine(220.0, sr, 1.0), sr)
    sf.write(str(b), _sine(220.0, 8_000, 1.0), 8_000)
    item = Scenario6Item(
        sample_id="t",
        enrollment_path=a,
        variant_paths=(a, b),
    )
    with pytest.raises(ValueError):
        assemble_track(item, sr)


def test_evaluate_one_with_stub(tmp_path):
    sr = 16_000
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    sf.write(str(a), _sine(220.0, sr, 1.0), sr)
    sf.write(str(b), _sine(220.0, sr, 1.0), sr)
    item = Scenario6Item(
        sample_id="t",
        enrollment_path=a,
        variant_paths=(a, b),
    )
    entry = evaluate_one(item, StubPipelineProvider(), sample_rate=sr)
    # Stub gate is always on, so frame_accuracy = 1
    assert entry.frame_accuracy == pytest.approx(1.0)
    assert entry.gate_tpr == pytest.approx(1.0)
    # Stub leaves auto-learn at zeros / None
    assert entry.auto_learn_admissions == 0
    assert entry.auto_learn_resets == 0
    assert entry.anchor_distance_final is None


def test_run_emits_csv(tmp_path):
    sr = 16_000
    a = tmp_path / "a.wav"
    sf.write(str(a), _sine(220.0, sr, 1.0), sr)
    item = Scenario6Item(
        sample_id="utt_drift_001",
        enrollment_path=a,
        variant_paths=(a, a),
    )
    csv_path = tmp_path / "scenario_6.csv"
    result = run([item], StubPipelineProvider(), sample_rate=sr, output_csv=csv_path)
    assert result.scenario == "scenario_6"
    assert result.n_samples == 1
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["scenario"] == "scenario_6"
    assert row["sample_id"] == "utt_drift_001"
    assert row["frame_accuracy"] != ""
    assert row["snr_db"] == ""  # not applicable
    assert row["auto_learn_admissions"] == "0"
    assert row["auto_learn_resets"] == "0"


def test_run_with_default_provider_uses_stub(tmp_path):
    sr = 16_000
    a = tmp_path / "a.wav"
    sf.write(str(a), _sine(220.0, sr, 1.0), sr)
    item = Scenario6Item(
        sample_id="t",
        enrollment_path=a,
        variant_paths=(a,),
    )
    result = run([item], provider=None, sample_rate=sr)
    assert result.n_samples == 1
    assert result.metrics["frame_accuracy_mean"] == pytest.approx(1.0)
