"""Tests for the solo-other-speaker scenario (Scenario 2)."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import soundfile as sf

from mellonella_bench.scenarios.base import StubPipelineProvider
from mellonella_bench.scenarios.scenario_2 import (
    Scenario2Item,
    _rms_db,
    evaluate_one,
    run,
)


def _sine(freq: float, sr: int, duration: float) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_rms_db_unit_amplitude():
    sr = 16_000
    audio = _sine(220.0, sr, 1.0)
    # sine of amplitude 1.0 has RMS 1/sqrt(2) ≈ -3 dBFS
    assert _rms_db(audio) == pytest.approx(-3.01, abs=0.05)


def test_rms_db_silence_is_negative_inf():
    audio = np.zeros(1024, dtype=np.float32)
    assert _rms_db(audio) == float("-inf")


def test_rms_db_empty_is_negative_inf():
    assert _rms_db(np.empty(0, dtype=np.float32)) == float("-inf")


def test_evaluate_one_with_stub_yields_full_fpr(tmp_path):
    """Stub gate is always-on, so for Scenario 2 every voiced frame leaks → FPR=1."""
    sr = 16_000
    other_path = tmp_path / "other.wav"
    noise_path = tmp_path / "noise.wav"
    sf.write(str(other_path), _sine(220.0, sr, 6.0), sr)
    rng = np.random.default_rng(0)
    sf.write(str(noise_path), 0.01 * rng.standard_normal(sr * 6).astype(np.float32), sr)
    voiced_mask = np.ones(int(6.0 * sr) // 512, dtype=bool)

    item = Scenario2Item(
        sample_id="utt_other_001",
        other_speaker_path=other_path,
        noise_path=noise_path,
        voiced_mask=voiced_mask,
    )
    rows = evaluate_one(item, StubPipelineProvider(), sample_rate=sr, snrs_db=(10.0,))
    assert len(rows) == 1
    row = rows[0]
    assert row.gate_fpr == pytest.approx(1.0)
    assert row.gate_tnr == pytest.approx(0.0)
    # Stub passes the mixture through unchanged → output_rms_db is finite
    assert row.output_rms_db is not None
    assert np.isfinite(row.output_rms_db)


def test_evaluate_one_handles_zero_voiced_frames(tmp_path):
    sr = 16_000
    other_path = tmp_path / "other.wav"
    noise_path = tmp_path / "noise.wav"
    sf.write(str(other_path), _sine(220.0, sr, 1.0), sr)
    sf.write(str(noise_path), _sine(50.0, sr, 1.0), sr)
    voiced_mask = np.zeros(int(1.0 * sr) // 512, dtype=bool)

    item = Scenario2Item(
        sample_id="utt_silent_voicing",
        other_speaker_path=other_path,
        noise_path=noise_path,
        voiced_mask=voiced_mask,
    )
    rows = evaluate_one(item, StubPipelineProvider(), sample_rate=sr, snrs_db=(10.0,))
    assert rows[0].gate_fpr == 0.0
    assert rows[0].gate_tnr == 0.0


def test_run_emits_csv(tmp_path):
    sr = 16_000
    other_path = tmp_path / "other.wav"
    noise_path = tmp_path / "noise.wav"
    sf.write(str(other_path), _sine(220.0, sr, 4.0), sr)
    rng = np.random.default_rng(0)
    sf.write(str(noise_path), 0.01 * rng.standard_normal(sr * 4).astype(np.float32), sr)
    voiced_mask = np.ones(int(4.0 * sr) // 512, dtype=bool)

    item = Scenario2Item(
        sample_id="utt_a",
        other_speaker_path=other_path,
        noise_path=noise_path,
        voiced_mask=voiced_mask,
    )
    csv_path = tmp_path / "scenario_2.csv"
    result = run(
        [item], StubPipelineProvider(), sample_rate=sr, output_csv=csv_path, snrs_db=(0.0, 10.0)
    )
    assert result.scenario == "scenario_2"
    assert result.n_samples == 1
    assert "gate_tnr_mean" in result.metrics
    assert "output_rms_db_mean" in result.metrics

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    assert rows[0]["scenario"] == "scenario_2"
    assert rows[0]["sample_id"] == "utt_a"
    assert float(rows[0]["snr_db"]) == 0.0
    assert rows[0]["output_rms_db"] != ""


def test_run_rejects_sample_rate_mismatch(tmp_path):
    sr = 16_000
    other_path = tmp_path / "other.wav"
    noise_path = tmp_path / "noise.wav"
    sf.write(str(other_path), _sine(220.0, sr, 1.0), sr)
    sf.write(str(noise_path), _sine(50.0, 8_000, 1.0), 8_000)
    voiced_mask = np.ones(int(1.0 * sr) // 512, dtype=bool)
    item = Scenario2Item(
        sample_id="bad",
        other_speaker_path=other_path,
        noise_path=noise_path,
        voiced_mask=voiced_mask,
    )
    with pytest.raises(ValueError):
        run([item], StubPipelineProvider(), sample_rate=sr, snrs_db=(0.0,))


def test_run_with_default_provider_uses_stub(tmp_path):
    sr = 16_000
    other_path = tmp_path / "other.wav"
    noise_path = tmp_path / "noise.wav"
    sf.write(str(other_path), _sine(220.0, sr, 2.0), sr)
    rng = np.random.default_rng(0)
    sf.write(str(noise_path), 0.01 * rng.standard_normal(sr * 2).astype(np.float32), sr)
    voiced_mask = np.ones(int(2.0 * sr) // 512, dtype=bool)
    item = Scenario2Item(
        sample_id="utt_default",
        other_speaker_path=other_path,
        noise_path=noise_path,
        voiced_mask=voiced_mask,
    )
    result = run([item], provider=None, sample_rate=sr, snrs_db=(0.0,))
    assert result.n_samples == 1
    # stub leaves output as-is, so RMS should be finite
    assert result.metrics["output_rms_db_mean"] > -float("inf")
