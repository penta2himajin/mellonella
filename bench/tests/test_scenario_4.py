"""Tests for the simultaneous-speech scenario (Scenario 4)."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import soundfile as sf

from mellonella_bench.scenarios.base import StubPipelineProvider
from mellonella_bench.scenarios.scenario_4 import (
    DEFAULT_RATIOS_DB,
    Scenario4Item,
    evaluate_one,
    mix_at_ratio,
    run,
)


def _sine(freq: float, sr: int, duration: float) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_mix_at_ratio_target_only_returns_target():
    sr = 16_000
    target = _sine(220.0, sr, 1.0)
    other = _sine(330.0, sr, 1.0)
    out = mix_at_ratio(target, other, float("inf"))
    np.testing.assert_array_equal(out, target)


def test_mix_at_ratio_other_only_returns_other():
    sr = 16_000
    target = _sine(220.0, sr, 1.0)
    other = _sine(330.0, sr, 1.0)
    out = mix_at_ratio(target, other, float("-inf"))
    np.testing.assert_array_equal(out, other)


@pytest.mark.parametrize("ratio_db", [-9.0, -3.0, 0.0, 3.0, 9.0])
def test_mix_at_ratio_matches_requested_db(ratio_db: float):
    sr = 16_000
    rng = np.random.default_rng(0)
    # Use orthogonal-ish noise to avoid cross-correlation skewing the
    # target/residual decomposition.
    target = rng.standard_normal(sr).astype(np.float32)
    other = rng.standard_normal(sr).astype(np.float32)
    out = mix_at_ratio(target, other, ratio_db)
    residual = out - target
    target_pow = float(np.mean(target.astype(np.float64) ** 2))
    residual_pow = float(np.mean(residual.astype(np.float64) ** 2))
    measured_db = 10.0 * np.log10(target_pow / residual_pow)
    assert measured_db == pytest.approx(ratio_db, abs=0.5)


def test_mix_at_ratio_truncates_long_other():
    sr = 16_000
    target = _sine(220.0, sr, 1.0)
    other = _sine(330.0, sr, 2.0)  # twice as long
    out = mix_at_ratio(target, other, 0.0)
    assert out.size == target.size


def test_mix_at_ratio_pads_short_other():
    sr = 16_000
    target = _sine(220.0, sr, 1.0)
    other = _sine(330.0, sr, 0.5)  # half as long
    out = mix_at_ratio(target, other, 0.0)
    assert out.size == target.size


def test_mix_at_ratio_rejects_zero_energy():
    sr = 16_000
    target = np.zeros(sr, dtype=np.float32)
    other = _sine(330.0, sr, 1.0)
    with pytest.raises(ValueError):
        mix_at_ratio(target, other, 0.0)


def test_evaluate_one_with_stub(tmp_path):
    """Stub gate is always-on, so TPR=1 across every ratio."""
    sr = 16_000
    target_path = tmp_path / "target.wav"
    other_path = tmp_path / "other.wav"
    sf.write(str(target_path), _sine(220.0, sr, 4.0), sr)
    sf.write(str(other_path), _sine(330.0, sr, 4.0), sr)
    voiced_mask = np.ones(int(4.0 * sr) // 512, dtype=bool)

    item = Scenario4Item(
        sample_id="utt_alt_001",
        target_path=target_path,
        other_path=other_path,
        voiced_mask=voiced_mask,
        target_to_other_ratios_db=(0.0, float("-inf")),
    )
    rows = evaluate_one(item, StubPipelineProvider(), sample_rate=sr, output_sr=sr)
    assert len(rows) == 2
    for row in rows:
        assert row.gate_tpr == pytest.approx(1.0)
        # Stub passes the mixture; SI-SDR is well-defined and finite.
        assert row.si_sdr is not None
        assert np.isfinite(row.si_sdr)
    # +0 dB mix → SI-SDR of stubbed (target+other) vs target ≈ 0 dB
    assert rows[0].si_sdr == pytest.approx(0.0, abs=1.0)
    # other_only mix → SI-SDR much worse (stub forwards the other speaker as-is)
    assert rows[1].si_sdr < rows[0].si_sdr


def test_evaluate_one_records_inf_ratio_with_no_snr(tmp_path):
    sr = 16_000
    target_path = tmp_path / "target.wav"
    other_path = tmp_path / "other.wav"
    sf.write(str(target_path), _sine(220.0, sr, 1.0), sr)
    sf.write(str(other_path), _sine(330.0, sr, 1.0), sr)
    voiced_mask = np.ones(int(1.0 * sr) // 512, dtype=bool)

    item = Scenario4Item(
        sample_id="utt_a",
        target_path=target_path,
        other_path=other_path,
        voiced_mask=voiced_mask,
        target_to_other_ratios_db=(float("inf"), 0.0),
    )
    rows = evaluate_one(item, StubPipelineProvider(), sample_rate=sr, output_sr=sr)
    assert rows[0].snr_db is None
    assert rows[0].notes == "target_only"
    assert rows[1].snr_db == pytest.approx(0.0)
    assert rows[1].notes == ""


def test_run_emits_csv(tmp_path):
    sr = 16_000
    target_path = tmp_path / "target.wav"
    other_path = tmp_path / "other.wav"
    sf.write(str(target_path), _sine(220.0, sr, 2.0), sr)
    sf.write(str(other_path), _sine(330.0, sr, 2.0), sr)
    voiced_mask = np.ones(int(2.0 * sr) // 512, dtype=bool)

    item = Scenario4Item(
        sample_id="utt_a",
        target_path=target_path,
        other_path=other_path,
        voiced_mask=voiced_mask,
        target_to_other_ratios_db=(0.0, -3.0, 3.0),
    )
    csv_path = tmp_path / "scenario_4.csv"
    result = run([item], StubPipelineProvider(), sample_rate=sr, output_csv=csv_path, output_sr=sr)
    assert result.scenario == "scenario_4"
    assert result.n_samples == 1
    assert "gate_tpr_mean" in result.metrics
    assert "si_sdr_mean" in result.metrics

    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 3
    assert rows[0]["scenario"] == "scenario_4"
    assert rows[0]["sample_id"] == "utt_a"
    # Three finite ratios → snr_db column populated
    for row in rows:
        assert row["snr_db"] != ""
        assert row["si_sdr"] != ""
        assert row["gate_tpr"] != ""


def test_run_rejects_sample_rate_mismatch(tmp_path):
    sr = 16_000
    target_path = tmp_path / "target.wav"
    other_path = tmp_path / "other.wav"
    sf.write(str(target_path), _sine(220.0, sr, 1.0), sr)
    sf.write(str(other_path), _sine(330.0, 8_000, 1.0), 8_000)
    voiced_mask = np.ones(int(1.0 * sr) // 512, dtype=bool)
    item = Scenario4Item(
        sample_id="bad",
        target_path=target_path,
        other_path=other_path,
        voiced_mask=voiced_mask,
        target_to_other_ratios_db=(0.0,),
    )
    with pytest.raises(ValueError):
        run([item], StubPipelineProvider(), sample_rate=sr)


def test_default_ratios_cover_target_and_other_only_endpoints():
    assert float("inf") in DEFAULT_RATIOS_DB
    assert float("-inf") in DEFAULT_RATIOS_DB
    finite = [r for r in DEFAULT_RATIOS_DB if np.isfinite(r)]
    # docs/benchmarks.md scenario_4: target:other ∈ {0:1, 1:3, 1:1, 3:1, 1:0}
    # 1:3 ≈ -9.5 dB, 3:1 ≈ +9.5 dB; we use ±9 / ±3 / 0 as approximate cover
    assert max(finite) > 0.0
    assert min(finite) < 0.0
    assert 0.0 in finite
