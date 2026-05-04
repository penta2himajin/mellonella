"""Tests for the multilingual-robustness scenario (Scenario 5)."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import soundfile as sf

from mellonella_bench.scenarios.base import StubPipelineProvider
from mellonella_bench.scenarios.scenario_5 import (
    DEFAULT_SNRS_DB,
    Scenario5Item,
    _aggregate_by_language,
    evaluate_one,
    run,
)


def _sine(freq: float, sr: int, duration: float) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _make_item(tmp_path, *, sample_id: str, language: str, sr: int = 16_000) -> Scenario5Item:
    target_path = tmp_path / f"{sample_id}_target.wav"
    other_path = tmp_path / f"{sample_id}_other.wav"
    noise_path = tmp_path / f"{sample_id}_noise.wav"
    duration = 4.0
    sf.write(str(target_path), _sine(220.0, sr, duration), sr)
    sf.write(str(other_path), _sine(330.0, sr, duration), sr)
    rng = np.random.default_rng(0)
    sf.write(
        str(noise_path),
        0.01 * rng.standard_normal(int(sr * duration)).astype(np.float32),
        sr,
    )
    n_frames = int(sr * duration) // 512
    return Scenario5Item(
        sample_id=sample_id,
        language=language,
        target_path=target_path,
        other_path=other_path,
        noise_path=noise_path,
        target_voiced_mask=np.ones(n_frames, dtype=bool),
        other_voiced_mask=np.ones(n_frames, dtype=bool),
        target_speaker=f"{language}_spk_t",
        other_speaker=f"{language}_spk_o",
    )


def test_evaluate_one_emits_two_rows_per_snr(tmp_path):
    sr = 16_000
    item = _make_item(tmp_path, sample_id="utt_en_001", language="en", sr=sr)
    rows = evaluate_one(item, StubPipelineProvider(), sample_rate=sr, snrs_db=(0.0, 10.0))
    assert len(rows) == 4
    modes = [r.notes for r in rows]
    assert modes == ["mode=target", "mode=other", "mode=target", "mode=other"]
    # Stub pipeline has gate-on everywhere → TPR=1, FPR=1
    target_rows = [r for r in rows if r.notes == "mode=target"]
    other_rows = [r for r in rows if r.notes == "mode=other"]
    assert all(r.gate_tpr == pytest.approx(1.0) for r in target_rows)
    assert all(r.gate_fpr == pytest.approx(1.0) for r in other_rows)


def test_evaluate_one_populates_language(tmp_path):
    sr = 16_000
    item = _make_item(tmp_path, sample_id="utt_ja_001", language="ja", sr=sr)
    rows = evaluate_one(item, StubPipelineProvider(), sample_rate=sr, snrs_db=(5.0,))
    assert all(r.language == "ja" for r in rows)
    assert all(r.target_speaker == "ja_spk_t" for r in rows)


def test_evaluate_one_handles_zero_voiced_other(tmp_path):
    sr = 16_000
    item = _make_item(tmp_path, sample_id="utt_de_001", language="de", sr=sr)
    item.other_voiced_mask = np.zeros_like(item.other_voiced_mask)
    rows = evaluate_one(item, StubPipelineProvider(), sample_rate=sr, snrs_db=(10.0,))
    other_row = next(r for r in rows if r.notes == "mode=other")
    assert other_row.gate_fpr == 0.0
    assert other_row.gate_tnr == 0.0


def test_evaluate_one_rejects_sample_rate_mismatch(tmp_path):
    sr = 16_000
    item = _make_item(tmp_path, sample_id="utt_x", language="en", sr=sr)
    bad_path = tmp_path / "bad_other.wav"
    sf.write(str(bad_path), _sine(330.0, 8_000, 1.0), 8_000)
    item.other_path = bad_path
    with pytest.raises(ValueError):
        evaluate_one(item, StubPipelineProvider(), sample_rate=sr, snrs_db=(0.0,))


def test_run_emits_per_language_aggregates(tmp_path):
    sr = 16_000
    items = [
        _make_item(tmp_path, sample_id="en_001", language="en", sr=sr),
        _make_item(tmp_path, sample_id="ja_001", language="ja", sr=sr),
        _make_item(tmp_path, sample_id="de_001", language="de", sr=sr),
    ]
    csv_path = tmp_path / "scenario_5.csv"
    result = run(items, output_csv=csv_path, sample_rate=sr, snrs_db=(0.0, 10.0))
    assert result.scenario == "scenario_5"
    assert result.n_samples == 3
    # Per-language keys present
    for lang in ("en", "ja", "de"):
        assert f"gate_tpr_mean__{lang}" in result.metrics
        assert f"gate_fpr_mean__{lang}" in result.metrics
    # Cross-language stddev present
    assert "gate_tpr_std_across_languages" in result.metrics
    assert "gate_fpr_std_across_languages" in result.metrics
    # Stub passes everything → all per-lang means are 1.0 → stddev is 0
    assert result.metrics["gate_tpr_std_across_languages"] == pytest.approx(0.0)


def test_run_writes_csv_with_language_column(tmp_path):
    sr = 16_000
    item = _make_item(tmp_path, sample_id="en_csv", language="en", sr=sr)
    csv_path = tmp_path / "scenario_5.csv"
    run([item], output_csv=csv_path, sample_rate=sr, snrs_db=(5.0,))
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    assert rows[0]["scenario"] == "scenario_5"
    assert rows[0]["language"] == "en"
    assert rows[0]["target_speaker"] == "en_spk_t"


def test_run_with_default_provider_uses_stub(tmp_path):
    sr = 16_000
    item = _make_item(tmp_path, sample_id="en_def", language="en", sr=sr)
    result = run([item], provider=None, sample_rate=sr, snrs_db=(0.0,))
    assert result.n_samples == 1
    assert result.metrics["gate_tpr_mean__en"] == pytest.approx(1.0)


def test_aggregate_by_language_handles_empty():
    assert _aggregate_by_language([]) == {}


def test_default_snrs_cover_realistic_range():
    assert min(DEFAULT_SNRS_DB) >= -5.0
    assert max(DEFAULT_SNRS_DB) <= 30.0
